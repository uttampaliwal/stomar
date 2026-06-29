"""Tests for volatility forecasting module."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_ohlcv(n=500):
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    close_vals = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high_vals = close_vals + abs(np.random.randn(n) * 0.5)
    low_vals = close_vals - abs(np.random.randn(n) * 0.5)
    open_vals = close_vals + np.random.randn(n) * 0.2
    volume = np.random.randint(100000, 1000000, n)

    df = pd.DataFrame({
        "close": close_vals, "high": high_vals, "low": low_vals,
        "open": open_vals, "volume": volume,
    }, index=dates)
    df["returns"] = df["close"].pct_change().fillna(0)
    return df


class TestHistoricalVolatility:
    def test_output_length(self):
        from src.volatility import historical_volatility
        returns = _make_ohlcv()["returns"]
        vol = historical_volatility(returns, window=20)
        assert len(vol) == len(returns)

    def test_annualized(self):
        from src.volatility import historical_volatility
        returns = _make_ohlcv()["returns"]
        vol = historical_volatility(returns, window=20, annualize=True)
        valid = vol.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 0

    def test_not_annualized(self):
        from src.volatility import historical_volatility
        returns = _make_ohlcv()["returns"]
        vol = historical_volatility(returns, window=20, annualize=False)
        valid = vol.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 0


class TestEWMAMVolatility:
    def test_output_length(self):
        from src.volatility import ewma_volatility
        returns = _make_ohlcv()["returns"]
        vol = ewma_volatility(returns, span=20)
        assert len(vol) == len(returns)

    def test_positive(self):
        from src.volatility import ewma_volatility
        returns = _make_ohlcv()["returns"]
        vol = ewma_volatility(returns, span=20)
        assert vol.dropna().iloc[-1] > 0


class TestParkinsonVolatility:
    def test_output_length(self):
        from src.volatility import parkinson_volatility
        df = _make_ohlcv()
        vol = parkinson_volatility(df["high"], df["low"], window=20)
        assert len(vol) == len(df)

    def test_positive(self):
        from src.volatility import parkinson_volatility
        df = _make_ohlcv()
        vol = parkinson_volatility(df["high"], df["low"], window=20)
        assert vol.dropna().iloc[-1] > 0


class TestGarmanKlassVolatility:
    def test_output_length(self):
        from src.volatility import garman_klass_volatility
        df = _make_ohlcv()
        vol = garman_klass_volatility(df["open"], df["high"], df["low"], df["close"], window=20)
        assert len(vol) == len(df)

    def test_positive(self):
        from src.volatility import garman_klass_volatility
        df = _make_ohlcv()
        vol = garman_klass_volatility(df["open"], df["high"], df["low"], df["close"], window=20)
        assert vol.dropna().iloc[-1] > 0


class TestYangZhangVolatility:
    def test_output_length(self):
        from src.volatility import yang_zhang_volatility
        df = _make_ohlcv()
        vol = yang_zhang_volatility(df["open"], df["high"], df["low"], df["close"], window=20)
        assert len(vol) == len(df)

    def test_positive(self):
        from src.volatility import yang_zhang_volatility
        df = _make_ohlcv()
        vol = yang_zhang_volatility(df["open"], df["high"], df["low"], df["close"], window=20)
        assert vol.dropna().iloc[-1] > 0


class TestForecastVolatility:
    def test_returns_all_keys(self):
        from src.volatility import forecast_volatility
        returns = _make_ohlcv()["returns"]
        result = forecast_volatility(returns, method="ewma", horizon=5)
        expected = ["current_vol", "forecast_vols", "long_term_vol", "vol_std", "horizon", "method"]
        for k in expected:
            assert k in result

    def test_forecast_length(self):
        from src.volatility import forecast_volatility
        returns = _make_ohlcv()["returns"]
        result = forecast_volatility(returns, horizon=5)
        assert len(result["forecast_vols"]) == 5

    def test_historical_method(self):
        from src.volatility import forecast_volatility
        returns = _make_ohlcv()["returns"]
        result = forecast_volatility(returns, method="historical", horizon=3)
        assert result["method"] == "historical"


class TestDetectVolatilityRegime:
    def test_returns_all_keys(self):
        from src.volatility import detect_volatility_regime
        returns = _make_ohlcv()["returns"]
        result = detect_volatility_regime(returns)
        expected = ["current_regime", "current_vol", "vol_percentile",
                    "regime_color", "regime_percentages"]
        for k in expected:
            assert k in result

    def test_regime_is_valid(self):
        from src.volatility import detect_volatility_regime
        returns = _make_ohlcv()["returns"]
        result = detect_volatility_regime(returns)
        assert result["current_regime"] in ["Low", "Medium", "High"]

    def test_regime_percentages_sum_to_one(self):
        from src.volatility import detect_volatility_regime
        returns = _make_ohlcv()["returns"]
        result = detect_volatility_regime(returns)
        total = sum(result["regime_percentages"].values())
        assert abs(total - 1.0) < 0.1


class TestBollingerBands:
    def test_returns_all_keys(self):
        from src.volatility import compute_bollinger_bands
        close = _make_ohlcv()["close"]
        result = compute_bollinger_bands(close, window=20)
        expected = ["upper", "middle", "lower", "bandwidth", "percent_b"]
        for k in expected:
            assert k in result

    def test_upper_above_lower(self):
        from src.volatility import compute_bollinger_bands
        close = _make_ohlcv()["close"]
        result = compute_bollinger_bands(close, window=20)
        valid = result["upper"].dropna() > result["lower"].dropna()
        assert valid.all()


class TestVolatilityCone:
    def test_returns_all_windows(self):
        from src.volatility import volatility_cone
        df = _make_ohlcv()
        result = volatility_cone(df["high"], df["low"], df["close"], windows=[10, 20])
        assert 10 in result
        assert 20 in result

    def test_min_less_than_max(self):
        from src.volatility import volatility_cone
        df = _make_ohlcv()
        result = volatility_cone(df["high"], df["low"], df["close"], windows=[20])
        assert result[20]["min"] <= result[20]["max"] or np.isnan(result[20]["min"])


class TestComputeATR:
    def test_output_length(self):
        from src.volatility import compute_atr
        df = _make_ohlcv()
        atr = compute_atr(df["high"], df["low"], df["close"], window=14)
        assert len(atr) == len(df)

    def test_positive(self):
        from src.volatility import compute_atr
        df = _make_ohlcv()
        atr = compute_atr(df["high"], df["low"], df["close"], window=14)
        assert atr.dropna().iloc[-1] > 0


class TestPositionSizeForVol:
    def test_default_size(self):
        from src.volatility import position_size_for_vol
        result = position_size_for_vol()
        assert result["position_size_pct"] > 0
        assert result["vol_scalar"] == 1.0

    def test_high_vol_reduces(self):
        from src.volatility import position_size_for_vol
        result = position_size_for_vol(current_vol=0.30, target_vol=0.15)
        assert result["vol_scalar"] < 1.0

    def test_low_vol_increases(self):
        from src.volatility import position_size_for_vol
        result = position_size_for_vol(current_vol=0.10, target_vol=0.15)
        assert result["vol_scalar"] > 1.0


class TestFullVolatilityAnalysis:
    def test_returns_all_keys(self):
        from src.volatility import full_volatility_analysis
        df = _make_ohlcv()
        result = full_volatility_analysis(df)
        expected = ["current", "regime", "forecast", "bollinger",
                    "cone", "position_sizing", "estimators"]
        for k in expected:
            assert k in result

    def test_current_has_all_estimators(self):
        from src.volatility import full_volatility_analysis
        df = _make_ohlcv()
        result = full_volatility_analysis(df)
        estimators = ["hist_vol", "ewma_vol", "parkinson_vol",
                      "garman_klass_vol", "yang_zhang_vol", "atr"]
        for k in estimators:
            assert k in result["current"]
            assert result["current"][k] >= 0
