"""Tests for the live data engine and fetcher abstraction."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.data.live import (
    DepthLevel,
    IDataFetcher,
    LiveDataEngine,
    LiveQuote,
    MarketDepth,
    _extract_gf_price,
    _parse_nse_depth,
    _parse_nse_quote,
)


class TestIDataFetcherAbstraction:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            IDataFetcher()

    def test_default_depth_returns_none(self):
        class DummyFetcher(IDataFetcher):
            name = "dummy"

            def get_quote(self, symbol):
                return LiveQuote(symbol=symbol, price=100.0)

        fetcher = DummyFetcher()
        assert fetcher.get_depth("X.NS") is None
        assert fetcher.get_pcr() is None
        assert fetcher.get_fii_dii() is None
        assert fetcher.is_available() is True


NSE_QUOTE_FIXTURE = {
    "priceInfo": {
        "lastPrice": 1300.5,
        "previousClose": 1290.0,
        "change": 10.5,
        "pChange": 0.81,
    },
    "ohlc": {"open": 1295.0, "high": 1310.0, "low": 1288.0},
    "totalTradedVolume": 1234567,
    "marketDeptOrderBook": {
        "bid": [
            {"price": 1300.0, "quantity": 1200, "numberOfOrders": 3},
            {"price": 1299.5, "quantity": 800, "numberOfOrders": 2},
        ],
        "ask": [
            {"price": 1300.5, "quantity": 1500, "numberOfOrders": 4},
            {"price": 1301.0, "quantity": 900, "numberOfOrders": 1},
        ],
    },
}


class TestNSEParsing:
    def test_parse_nse_quote(self):
        quote = _parse_nse_quote("RELIANCE.NS", NSE_QUOTE_FIXTURE)
        assert quote.price == pytest.approx(1300.5)
        assert quote.prev_close == pytest.approx(1290.0)
        assert quote.change_pct == pytest.approx(0.81)
        assert quote.volume == 1234567
        assert quote.source == "nse"

    def test_parse_nse_depth_top5(self):
        depth = _parse_nse_depth("RELIANCE.NS", NSE_QUOTE_FIXTURE)
        assert len(depth.bids) == 2
        assert len(depth.asks) == 2
        assert depth.bids[0].price == pytest.approx(1300.0)
        assert depth.bids[0].quantity == 1200
        assert depth.asks[0].price == pytest.approx(1300.5)

    def test_parse_nse_depth_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_nse_depth("X.NS", {"marketDeptOrderBook": {"bid": [], "ask": []}})

    def test_to_dict_payloads(self):
        quote = _parse_nse_quote("RELIANCE.NS", NSE_QUOTE_FIXTURE)
        payload = quote.to_dict()
        assert payload["price"] == pytest.approx(1300.5)
        assert "timestamp" in payload and "source" in payload
        depth = _parse_nse_depth("RELIANCE.NS", NSE_QUOTE_FIXTURE)
        dpayload = depth.to_dict()
        assert dpayload["bids"][0]["price"] == pytest.approx(1300.0)


class TestGoogleExtraction:
    def test_extract_price_from_embedded_json(self):
        html = (
            '<div>["RELIANCE:NSE",[1,1234.55,1,"1234.55",0,0,null,1]]</div>'
        )
        price = _extract_gf_price(html, "RELIANCE", "NSE")
        assert price == pytest.approx(1234.55)

    def test_extract_price_missing(self):
        assert _extract_gf_price("<html>nothing</html>", "RELIANCE", "NSE") is None


class _FailingFetcher(IDataFetcher):
    name = "failing"

    def __init__(self):
        self.calls = 0

    def get_quote(self, symbol):
        self.calls += 1
        raise RuntimeError("boom")


class _GoodFetcher(IDataFetcher):
    name = "good"

    def __init__(self):
        self.calls = 0

    def get_quote(self, symbol):
        self.calls += 1
        return LiveQuote(symbol=symbol, price=42.0, source="good")


class _DepthFetcher(IDataFetcher):
    name = "depthy"

    def get_quote(self, symbol):
        return LiveQuote(symbol=symbol, price=1.0)

    def get_depth(self, symbol):
        return MarketDepth(
            symbol=symbol,
            bids=[DepthLevel(price=9.0, quantity=100, orders=1)],
            asks=[DepthLevel(price=10.0, quantity=200, orders=2)],
            source="depthy",
        )


class TestLiveEngineFallback:
    def test_quote_falls_back_to_working_fetcher(self):
        engine = LiveDataEngine(fetchers=[_FailingFetcher(), _GoodFetcher()])
        quote = engine.get_quote("RELIANCE.NS")
        assert quote.price == pytest.approx(42.0)
        assert quote.source == "good"

    def test_quote_all_fail_raises(self):
        engine = LiveDataEngine(fetchers=[_FailingFetcher()])
        with pytest.raises(RuntimeError):
            engine.get_quote("X.NS")

    def test_quote_uses_stale_cache_when_all_fail(self):
        engine = LiveDataEngine(fetchers=[_GoodFetcher(), _FailingFetcher()])
        first = engine.get_quote("X.NS")
        assert first.price == pytest.approx(42.0)
        engine.fetchers = [_FailingFetcher()]
        stale = engine.get_quote("X.NS", force=True)  # bypass fresh-cache TTL
        assert stale.is_stale is True
        assert stale.price == pytest.approx(42.0)

    def test_quote_cache_ttl(self):
        fetcher = _GoodFetcher()
        engine = LiveDataEngine(fetchers=[fetcher])
        engine.get_quote("X.NS")
        calls_after_cache = fetcher.calls
        engine.get_quote("X.NS")  # served from cache — no new calls
        assert fetcher.calls == calls_after_cache
        engine.get_quote("X.NS", force=True)  # force bypasses cache
        assert fetcher.calls == calls_after_cache + 1

    def test_depth_uses_first_provider_with_depth(self):
        engine = LiveDataEngine(fetchers=[_GoodFetcher(), _DepthFetcher()])
        depth = engine.get_depth("X.NS")
        assert depth.source == "depthy"
        assert depth.bids[0].price == pytest.approx(9.0)
        assert len(depth.asks) == 1

    def test_depth_all_fail_raises(self):
        engine = LiveDataEngine(fetchers=[_GoodFetcher()])
        with pytest.raises(RuntimeError):
            engine.get_depth("X.NS")


class TestMinuteAggregation:
    def test_accumulate_and_flush_writes_minute_bar(self, tmp_path):
        from src.data.store import MarketDataStore

        store = MarketDataStore(str(tmp_path / "live.db"))
        engine = LiveDataEngine(fetchers=[_GoodFetcher()], store=store)
        # Quote volume is session-cumulative: only the delta between
        # observations belongs to the bar (#24).
        engine._accumulate_minute("X.NS", LiveQuote(symbol="X.NS", price=100.0, volume=100))
        engine._accumulate_minute("X.NS", LiveQuote(symbol="X.NS", price=102.0, volume=150))
        engine._accumulate_minute("X.NS", LiveQuote(symbol="X.NS", price=101.0, volume=200))
        engine.flush_all_minutes()
        hist = store.get_history("X.NS", interval="minute")
        assert len(hist) == 1
        assert hist["open"].iloc[0] == pytest.approx(100.0)
        assert hist["high"].iloc[0] == pytest.approx(102.0)
        assert hist["low"].iloc[0] == pytest.approx(100.0)
        assert hist["close"].iloc[0] == pytest.approx(101.0)
        assert hist["volume"].iloc[0] == 100  # deltas only: 0 + 50 + 50
        store.close()

    def test_cumulative_volume_not_double_counted(self, tmp_path):
        """Regression: repeated polls must not multiply cumulative volume."""
        from src.data.store import MarketDataStore

        store = MarketDataStore(str(tmp_path / "live3.db"))
        engine = LiveDataEngine(fetchers=[_GoodFetcher()], store=store)
        # A feed that keeps reporting the same cumulative volume (no new
        # trades) must not inflate the bar's volume.
        engine._accumulate_minute("X.NS", LiveQuote(symbol="X.NS", price=100.0, volume=500))
        engine._accumulate_minute("X.NS", LiveQuote(symbol="X.NS", price=100.5, volume=500))
        engine._accumulate_minute("X.NS", LiveQuote(symbol="X.NS", price=100.2, volume=500))
        engine.flush_all_minutes()
        hist = store.get_history("X.NS", interval="minute")
        assert hist["volume"].iloc[0] == 0
        store.close()

    def test_new_minute_flushes_previous(self, tmp_path):
        from src.data.store import MarketDataStore

        store = MarketDataStore(str(tmp_path / "live2.db"))
        engine = LiveDataEngine(fetchers=[_GoodFetcher()], store=store)
        engine._accumulate_minute("X.NS", LiveQuote(symbol="X.NS", price=100.0))
        # simulate a new minute by shifting the accumulated key
        engine._minute_key["X.NS"] = engine._minute_key["X.NS"].replace(
            minute=(engine._minute_key["X.NS"].minute - 1) % 60
        )
        engine._accumulate_minute("X.NS", LiveQuote(symbol="X.NS", price=101.0))
        assert len(store.query("SELECT * FROM minute_bars")) == 1
        engine.flush_all_minutes()
        assert len(store.query("SELECT * FROM minute_bars")) == 2
        store.close()


class TestEngineHealth:
    def test_health_reports_fetcher_status(self):
        engine = LiveDataEngine(fetchers=[_FailingFetcher()])
        with pytest.raises(RuntimeError):
            engine.get_quote("X.NS")
        health = engine.health()
        assert health["fetchers"]["failing"] is False
        assert health["cached_quotes"] == 0

    def test_health_reports_ok_after_success(self):
        engine = LiveDataEngine(fetchers=[_FailingFetcher(), _GoodFetcher()])
        engine.get_quote("X.NS")
        health = engine.health()
        assert health["fetchers"]["failing"] is False
        assert health["fetchers"]["good"] is True
        assert health["cached_quotes"] == 1
