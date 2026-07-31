"""Tests for the free real-time market data engine (services.market_data)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from services.market_data import (
    CandleProvider,
    MarketDataService,
    NseArchiveCandleProvider,
    ParquetCandleProvider,
    StoreCandleProvider,
    YahooCandleProvider,
    _resample,
    candles_to_payload,
    compute_indicators,
    get_market_data_service,
)


@pytest.fixture
def ohlcv():
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(100000, 1000000, n),
    }, index=dates)
    df.index.name = "date"
    return df


class TestComputeIndicators:
    def test_adds_all_columns(self, ohlcv):
        out = compute_indicators(ohlcv)
        for col in ["rsi", "macd", "macd_signal", "macd_hist", "atr",
                    "bb_middle", "bb_upper", "bb_lower", "bb_width", "volume_z"]:
            assert col in out.columns

    def test_rsi_bounded(self, ohlcv):
        rsi = compute_indicators(ohlcv)["rsi"].dropna()
        assert ((rsi >= 0) & (rsi <= 100)).all()

    def test_macd_hist_is_diff(self, ohlcv):
        out = compute_indicators(ohlcv)
        assert np.allclose(out["macd_hist"], out["macd"] - out["macd_signal"], equal_nan=True)

    def test_bollinger_ordering(self, ohlcv):
        out = compute_indicators(ohlcv).dropna(subset=["bb_upper"])
        assert (out["bb_upper"] >= out["bb_middle"]).all()
        assert (out["bb_middle"] >= out["bb_lower"]).all()

    def test_atr_nonnegative(self, ohlcv):
        atr = compute_indicators(ohlcv)["atr"].dropna()
        assert (atr >= 0).all()

    def test_volume_z_finite(self, ohlcv):
        vz = compute_indicators(ohlcv)["volume_z"].dropna()
        assert np.isfinite(vz).all()
        assert abs(vz.mean()) < 0.05


class TestResample:
    def test_resamples_minutes_to_5m(self):
        idx = pd.date_range("2024-01-01 09:15", periods=10, freq="1min")
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }, index=idx)
        out = _resample(df, "5min")
        assert len(out) == 2
        assert out.iloc[0]["open"] == 100
        assert out.iloc[0]["close"] == 104.5
        assert out.iloc[0]["high"] == 105
        assert out.iloc[0]["low"] == 99
        assert out.iloc[0]["volume"] == 5


class FakeProvider(CandleProvider):
    def __init__(self, name, intervals, df=None, error=None):
        self.name = name
        self.supports = set(intervals)
        self.df = df
        self.error = error
        self.calls = 0

    def fetch(self, symbol, interval, limit):
        self.calls += 1
        if self.error:
            raise self.error
        return self.df


class TestMarketDataService:
    def test_falls_back_to_secondary_source(self, ohlcv):
        primary = FakeProvider("primary", ["1d"], error=RuntimeError("down"))
        secondary = FakeProvider("secondary", ["1d"], df=ohlcv)
        service = MarketDataService(providers=[primary, secondary], cache_ttl=0)
        payload = service.get_candles("X.NS", interval="1d", limit=50, indicators=False)
        assert payload["source"] == "secondary"
        assert payload["rows"] == 50
        assert primary.calls == 1

    def test_primary_source_preferred(self, ohlcv):
        primary = FakeProvider("primary", ["1d"], df=ohlcv)
        secondary = FakeProvider("secondary", ["1d"], df=ohlcv)
        service = MarketDataService(providers=[primary, secondary], cache_ttl=0)
        payload = service.get_candles("X.NS", interval="1d")
        assert payload["source"] == "primary"
        assert secondary.calls == 0

    def test_all_sources_fail_raises(self):
        service = MarketDataService(
            providers=[FakeProvider("a", ["1d"], error=RuntimeError("x")),
                       FakeProvider("b", ["1d"], error=RuntimeError("y"))],
            cache_ttl=0,
        )
        with pytest.raises(RuntimeError, match="All candle sources failed"):
            service.get_candles("X.NS", interval="1d")

    def test_unsupported_interval_rejected(self, ohlcv):
        service = MarketDataService(providers=[FakeProvider("p", ["1d"], df=ohlcv)])
        with pytest.raises(ValueError, match="interval must be one of"):
            service.get_candles("X.NS", interval="7d")

    def test_indicators_stitched_when_requested(self, ohlcv):
        service = MarketDataService(providers=[FakeProvider("p", ["1d"], df=ohlcv)], cache_ttl=0)
        payload = service.get_candles("X.NS", interval="1d", limit=100)
        bar = payload["bars"][-1]
        assert "rsi" in bar and "atr" in bar and "volume_z" in bar
        assert payload["indicators"] is True

    def test_providers_filtered_by_interval(self, ohlcv):
        daily = FakeProvider("daily", ["1d"], df=ohlcv)
        intraday = FakeProvider("intraday", ["1m"], df=ohlcv, error=RuntimeError("no"))
        service = MarketDataService(providers=[daily, intraday], cache_ttl=0)
        service.get_candles("X.NS", interval="1d")
        assert intraday.calls == 0

    def test_daily_returns(self, ohlcv):
        service = MarketDataService(providers=[FakeProvider("p", ["1d"], df=ohlcv)], cache_ttl=0)
        result = service.get_daily_returns("X.NS", days=50)
        assert result["source"] == "p"
        assert len(result["returns"]) == 49
        last = result["returns"][-1]
        assert "timestamp" in last and "return_pct" in last and "close" in last

    def test_cache_hits_skip_fetch(self, ohlcv):
        provider = FakeProvider("p", ["1d"], df=ohlcv)
        service = MarketDataService(providers=[provider], cache_ttl=3600)
        service.get_candles("X.NS", interval="1d")
        service.get_candles("X.NS", interval="1d")
        assert provider.calls == 1

    def test_force_refresh_bypasses_cache(self, ohlcv):
        provider = FakeProvider("p", ["1d"], df=ohlcv)
        service = MarketDataService(providers=[provider], cache_ttl=3600)
        service.get_candles("X.NS", interval="1d")
        service.get_candles("X.NS", interval="1d", force=True)
        assert provider.calls == 2

    def test_health_reports_sources(self, ohlcv):
        service = MarketDataService(
            providers=[FakeProvider("b", ["1d"], error=RuntimeError("x")),
                       FakeProvider("a", ["1d", "1m"], df=ohlcv)],
            cache_ttl=0,
        )
        service.get_candles("X.NS", interval="1d")
        health = service.sources_health()
        by_name = {p["name"]: p for p in health["providers"]}
        assert by_name["a"]["last_status"] is True
        assert by_name["b"]["last_status"] is False
        assert by_name["a"]["intervals"] == ["1d", "1m"]


class TestPayload:
    def test_candles_to_payload_shape(self, ohlcv):
        payload = candles_to_payload("X.NS", "1d", ohlcv, "test", indicators=False, limit=10)
        assert payload["symbol"] == "X.NS"
        assert payload["interval"] == "1d"
        assert payload["source"] == "test"
        assert payload["rows"] == 10
        bar = payload["bars"][0]
        assert set(bar) == {"timestamp", "open", "high", "low", "close", "volume"}
        assert isinstance(bar["volume"], int)

    def test_candles_to_payload_indicators(self, ohlcv):
        payload = candles_to_payload("X.NS", "1d", compute_indicators(ohlcv), "test", indicators=True, limit=10)
        assert "rsi" in payload["bars"][-1]

    def test_drops_na_close(self, ohlcv):
        ohlcv.loc[ohlcv.index[0], "close"] = np.nan
        payload = candles_to_payload("X.NS", "1d", ohlcv, "test", indicators=False, limit=500)
        assert payload["rows"] == len(ohlcv) - 1


class TestSingleton:
    def test_singleton(self):
        assert get_market_data_service() is get_market_data_service()

    def test_provider_classes_exist(self):
        assert YahooCandleProvider.supports == {"1m", "5m", "15m", "1d"}
        assert StoreCandleProvider.supports == {"1m", "5m", "15m", "1d"}
        assert NseArchiveCandleProvider.supports == {"1d"}
        assert ParquetCandleProvider.supports == {"15m", "1d"}
