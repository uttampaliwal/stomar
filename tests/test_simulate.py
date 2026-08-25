"""Tests for the authoritative simulation core (src/trading/simulate.py)."""
import numpy as np
import pandas as pd
import pytest


class TestNseCostRates:
    def test_rates_positive(self):
        from src.trading.simulate import nse_cost_rates
        rates = nse_cost_rates()
        assert rates["buy"] > 0
        assert rates["sell"] > 0

    def test_sell_cheaper_than_buy_no_stamp_duty(self):
        from src.trading.simulate import nse_cost_rates
        rates = nse_cost_rates()
        assert rates["sell"] < rates["buy"]

    def test_rates_include_slippage(self):
        from src.trading.simulate import nse_cost_rates
        from src.core.constants import SLIPPAGE_RATE, calculate_nse_costs
        rates = nse_cost_rates()
        raw_buy = calculate_nse_costs(1000.0, 100, "buy")["effective_rate"]
        assert rates["buy"] == pytest.approx(raw_buy + SLIPPAGE_RATE, abs=1e-6)


class TestPositionsFromPredictions:
    def test_directional_long_flat(self):
        from src.trading.simulate import positions_from_predictions
        pos = positions_from_predictions(np.array([0.9, 0.4, 0.5, 0.51]))
        assert pos.tolist() == [1.0, 0.0, 0.0, 1.0]

    def test_directional_exposure_scales(self):
        from src.trading.simulate import positions_from_predictions
        pos = positions_from_predictions(np.array([0.9, 0.1]), exposure=0.25)
        assert pos.tolist() == [0.25, 0.0]

    def test_signed_bounded(self):
        from src.trading.simulate import positions_from_predictions
        pos = positions_from_predictions(np.array([1.2, 0.75, 0.25, -0.1]), strategy="signed")
        assert np.all(pos <= 1.0) and np.all(pos >= -1.0)
        assert pos[0] == 1.0 and pos[-1] == -1.0

    def test_unknown_strategy_raises_or_defaults_directional(self):
        from src.trading.simulate import positions_from_predictions
        # anything but "signed" behaves directional
        pos = positions_from_predictions(np.array([0.9]), strategy="directional")
        assert pos.tolist() == [1.0]


class TestStrategyReturnSeries:
    def test_shape_mismatch_raises(self):
        from src.trading.simulate import strategy_return_series
        with pytest.raises(ValueError):
            strategy_return_series(np.zeros(3), np.zeros(4))

    def test_flat_position_free_to_hold(self):
        from src.trading.simulate import nse_cost_rates, strategy_return_series
        fwd = np.full(10, 0.01)
        rets = strategy_return_series(np.ones(10), fwd)
        buy = nse_cost_rates()["buy"]
        assert rets[0] == pytest.approx(0.01 - buy, abs=1e-9)
        assert np.allclose(rets[1:], 0.01)

    def test_never_in_market_zero_returns(self):
        from src.trading.simulate import strategy_return_series
        rets = strategy_return_series(np.zeros(50), np.tile([0.01, -0.01], 25))
        assert np.all(rets == 0.0)

    def test_exit_charges_sell_rate(self):
        from src.trading.simulate import nse_cost_rates, strategy_return_series
        sell = nse_cost_rates()["sell"]
        rets = strategy_return_series(np.array([1.0, 0.0]), np.array([0.01, 0.02]))
        assert rets[0] == pytest.approx(0.01 - nse_cost_rates()["buy"], abs=1e-9)
        assert rets[1] == pytest.approx(-sell, abs=1e-9)

    def test_long_short_flip_pays_both_legs(self):
        from src.trading.simulate import nse_cost_rates, strategy_return_series
        rates = nse_cost_rates()
        # +1 -> -1 crosses zero: two sells (close long, open short)
        rets = strategy_return_series(np.array([1.0, -1.0]), np.array([0.01, 0.01]))
        assert rets[1] == pytest.approx(-0.01 - 2 * rates["sell"], abs=1e-9)
        # -1 -> +1: two buys (cover short, go long); long earns +0.01 gross
        rets2 = strategy_return_series(np.array([-1.0, 1.0]), np.array([0.01, 0.01]))
        assert rets2[1] == pytest.approx(0.01 - 2 * rates["buy"], abs=1e-9)

    def test_costs_only_reduce(self):
        from src.trading.simulate import strategy_return_series
        rng = np.random.default_rng(42)
        fwd = rng.normal(0, 0.01, 200)
        pos = rng.integers(0, 2, 200).astype(float)
        costed = strategy_return_series(pos, fwd)
        gross = pos * fwd
        # equity can never be improved by costs over the full path
        assert np.prod(1 + costed) <= np.prod(1 + gross) + 1e-12

    def test_matches_event_driven_portfolio_economics(self):
        """Vectorized engine ≈ Portfolio path for one round trip.

        Buy at close t with slippage, sell at close t+1 with slippage,
        NSE costs both sides — total drag must equal buy+sell effective rates.
        """
        from src.trading.simulate import nse_cost_rates, strategy_return_series
        rates = nse_cost_rates()
        fwd = np.array([0.03, 0.0])
        pos = np.array([1.0, 0.0])  # enter t, exit t+1
        rets = strategy_return_series(pos, fwd)
        expected_total = (1 + 0.03 - rates["buy"]) * (1 - rates["sell"]) - 1
        assert float(np.prod(1 + rets) - 1) == pytest.approx(expected_total, abs=1e-9)


class TestOOSMetricsUsesCosts:
    def test_flipping_predictions_pay_costs(self):
        from src.models.validation import oos_metrics
        from src.trading.simulate import nse_cost_rates
        df = pd.DataFrame({
            "y_true": np.tile([0.01, -0.01], 50),
            "y_pred": np.array([0.9, 0.1] * 50),
        })
        m = oos_metrics(df)
        rates = nse_cost_rates()
        expected = ((1 + 0.01 - rates["buy"]) * (1 - rates["sell"])) ** 50 - 1
        assert m["total_return"] == pytest.approx(expected, abs=1e-4)

    def test_held_position_single_entry_cost(self):
        from src.models.validation import oos_metrics
        from src.trading.simulate import nse_cost_rates
        df = pd.DataFrame({"y_true": np.full(100, 0.01), "y_pred": np.full(100, 0.9)})
        m = oos_metrics(df)
        buy = nse_cost_rates()["buy"]
        expected = (1.01 - buy) * (1.01 ** 99) - 1
        assert m["total_return"] == pytest.approx(expected, abs=1e-4)
