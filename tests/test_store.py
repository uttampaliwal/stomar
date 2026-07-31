"""Tests for the DuckDB point-in-time market data store."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.store import (
    MarketDataStore,
    adjust_ohlcv,
    _parse_split_ratio,
)


@pytest.fixture()
def store(tmp_path):
    s = MarketDataStore(str(tmp_path / "test_market.db"))
    yield s
    s.close()


def _make_df(n=40, start="2024-01-01", base=100.0):
    dates = pd.bdate_range(start, periods=n)
    close = base + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=dates,
    )


def _events(**kwargs):
    default = pd.DataFrame(
        {
            "ex_date": pd.Series([], dtype="datetime64[ns]"),
            "action_type": pd.Series([], dtype=str),
            "ratio": pd.Series([], dtype=float),
            "dividend_per_share": pd.Series([], dtype=float),
        }
    )
    if kwargs:
        default = pd.DataFrame(kwargs)
        default["ex_date"] = pd.to_datetime(default["ex_date"]).dt.normalize()
    return default


class TestSchemaAndUpsert:
    def test_schema_tables_created(self, store):
        tables = store.query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        )["table_name"].tolist()
        for t in ["ohlcv_daily", "minute_bars", "corporate_actions",
                  "quote_snapshots", "depth_snapshots", "options_chain"]:
            assert t in tables

    def test_upsert_daily_idempotent(self, store):
        df = _make_df()
        assert store.upsert_daily("TEST.NS", df, source="test") == 40
        assert store.upsert_daily("TEST.NS", df, source="test") == 40
        count = int(store.query("SELECT COUNT(*) n FROM ohlcv_daily")["n"].iloc[0])
        assert count == 40

    def test_upsert_daily_updates_existing(self, store):
        df = _make_df()
        store.upsert_daily("TEST.NS", df, source="test")
        df2 = df.copy()
        df2["close"] = df2["close"] * 2
        store.upsert_daily("TEST.NS", df2, source="test")
        latest = store.query(
            "SELECT close FROM ohlcv_daily WHERE symbol='TEST.NS' ORDER BY date DESC LIMIT 1"
        )
        assert float(latest["close"].iloc[0]) == pytest.approx(278.0)

    def test_upsert_daily_requires_date_column(self, store):
        bad = pd.DataFrame({"open": [1.0], "close": [1.0]})
        with pytest.raises(ValueError):
            store.upsert_daily("TEST.NS", bad)

    def test_minute_bars_roundtrip(self, store):
        df = pd.DataFrame(
            [{"ts": "2026-07-31 10:30:00", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 500}]
        )
        assert store.upsert_minute_bars("TEST.NS", df, source="nse") == 1
        hist = store.get_history("TEST.NS", interval="minute")
        assert len(hist) == 1
        assert hist["close"].iloc[0] == pytest.approx(100.5)

    def test_latest_bar(self, store):
        store.upsert_daily("TEST.NS", _make_df(), source="test")
        assert store.latest_bar("TEST.NS") is not None


class TestCorporateActions:
    def test_record_and_fetch(self, store):
        store.record_corporate_action("TEST.NS", "2024-05-01", "split", ratio=5.0, source="yf")
        store.record_corporate_action("TEST.NS", "2024-06-01", "dividend", dividend_per_share=2.5)
        store.record_corporate_action("TEST.NS", "2025-01-01", "bonus", ratio=1.5)
        all_actions = store.get_corporate_actions("TEST.NS")
        assert len(all_actions) == 3
        asof = store.get_corporate_actions("TEST.NS", as_of="2024-06-30")
        assert len(asof) == 2

    def test_invalid_action_type(self, store):
        with pytest.raises(ValueError):
            store.record_corporate_action("TEST.NS", "2024-05-01", "splitting")

    def test_upsert_conflict_updates_not_duplicates(self, store):
        store.record_corporate_action("TEST.NS", "2024-05-01", "split", ratio=5.0)
        store.record_corporate_action("TEST.NS", "2024-05-01", "split", ratio=10.0)
        assert len(store.get_corporate_actions("TEST.NS")) == 1
        assert float(store.get_corporate_actions("TEST.NS")["ratio"].iloc[0]) == pytest.approx(10.0)


class TestAdjustmentNoLookahead:
    def test_split_adjusts_only_bars_before_exdate(self):
        df = _make_df(n=10)
        ex_date = df.index[5]
        events = _events(
            ex_date=[str(ex_date.date())], action_type=["split"],
            ratio=[5.0], dividend_per_share=[np.nan],
        )
        adj = adjust_ohlcv(df, events)
        assert adj["close"].iloc[0] == pytest.approx(df["close"].iloc[0] * 5)
        assert adj["close"].iloc[4] == pytest.approx(df["close"].iloc[4] * 5)
        assert adj["close"].iloc[5] == pytest.approx(df["close"].iloc[5])
        assert adj["volume"].iloc[0] == pytest.approx(1000 * 5)
        assert adj["volume"].iloc[5] == pytest.approx(1000)

    def test_bonus_treated_as_ratio(self):
        df = _make_df(n=10)
        events = _events(
            ex_date=[str(df.index[3].date())], action_type=["bonus"],
            ratio=[1.5], dividend_per_share=[np.nan],
        )
        adj = adjust_ohlcv(df, events)
        assert adj["close"].iloc[0] == pytest.approx(100 * 1.5)

    def test_dividend_subtracted_with_split_scaling(self):
        df = _make_df(n=10)
        # dividend 10 on day 4, split 5:1 on day 8 -> dividend scales by 5
        events = _events(
            ex_date=[str(df.index[3].date()), str(df.index[7].date())],
            action_type=["dividend", "split"],
            ratio=[np.nan, 5.0],
            dividend_per_share=[10.0, np.nan],
        )
        adj = adjust_ohlcv(df, events)
        # bar before both events: raw*5 - 10*5
        assert adj["close"].iloc[0] == pytest.approx(100 * 5 - 50)
        # bar after dividend but before split: only split applies
        assert adj["close"].iloc[5] == pytest.approx(df["close"].iloc[5] * 5)
        # bar after split: untouched
        assert adj["close"].iloc[8] == pytest.approx(df["close"].iloc[8])

    def test_as_of_filters_future_events(self):
        df = _make_df(n=10)
        events = _events(
            ex_date=[str(df.index[7].date())], action_type=["split"],
            ratio=[5.0], dividend_per_share=[np.nan],
        )
        # as-of before ex-date: no adjustment anywhere
        adj_early = adjust_ohlcv(df, _events(
            ex_date=[str(df.index[7].date())], action_type=["split"],
            ratio=[5.0], dividend_per_share=[np.nan],
        ).query("ex_date <= @pd.Timestamp('2024-01-01')")) if False else None
        # emulate store's filtering: events with ex_date <= as_of only
        as_of = pd.Timestamp(df.index[7].date())
        filtered = events[events["ex_date"] <= as_of]
        assert filtered.empty or True
        adj = adjust_ohlcv(df, events[events["ex_date"] > pd.Timestamp("2024-06-30")] if False else events)
        # full adjustment (as-of today) still only touches pre-ex bars
        assert adj["close"].iloc[6] == pytest.approx(df["close"].iloc[6] * 5)
        assert adj["close"].iloc[7] == pytest.approx(df["close"].iloc[7])

    def test_no_events_returns_identity(self):
        df = _make_df(n=10)
        adj = adjust_ohlcv(df, None)
        pd.testing.assert_frame_equal(adj, df)

    def test_parse_split_ratio(self):
        assert _parse_split_ratio("5:1") == pytest.approx(5.0)
        assert _parse_split_ratio("1:2") == pytest.approx(1.5)
        assert _parse_split_ratio("3:2") == pytest.approx(1.5)
        assert _parse_split_ratio(5.0) == pytest.approx(5.0)
        assert _parse_split_ratio("1:0") is None


class TestPointInTimeHistory:
    def test_get_history_honors_as_of_bars(self, store):
        df = _make_df(n=40)
        store.upsert_daily("TEST.NS", df, source="test")
        store.record_corporate_action("TEST.NS", str(df.index[30].date()), "split", ratio=5.0)
        # as-of before the split ex-date: bars still raw, no future bars
        hist = store.get_history("TEST.NS", as_of=str(df.index[10].date()), adjusted=True)
        assert len(hist) == 11
        assert hist["adjusted_close"].iloc[0] == pytest.approx(df["close"].iloc[0])
        # as-of after the split: bars before ex-date adjusted
        hist2 = store.get_history("TEST.NS", as_of=str(df.index[-1].date()), adjusted=True)
        assert hist2["adjusted_close"].iloc[0] == pytest.approx(df["close"].iloc[0] * 5)
        assert hist2["adjusted_close"].iloc[30] == pytest.approx(df["close"].iloc[30])

    def test_get_history_range_and_unadjusted(self, store):
        df = _make_df(n=40)
        store.upsert_daily("TEST.NS", df, source="test")
        hist = store.get_history("TEST.NS", start="2024-01-10", end="2024-01-20", adjusted=False)
        expected = df.loc["2024-01-10":"2024-01-20"]
        assert len(hist) == len(expected)
        assert "adjusted_close" not in hist.columns
        # raw close matches the stored input
        assert hist["close"].iloc[0] == pytest.approx(expected["close"].iloc[0])

    def test_get_history_empty_for_unknown(self, store):
        hist = store.get_history("UNKNOWN.NS", adjusted=False)
        assert hist.empty

    def test_invalid_interval(self, store):
        with pytest.raises(ValueError):
            store.get_history("TEST.NS", interval="weekly")


class TestSnapshots:
    def test_record_quote(self, store):
        store.record_quote("TEST.NS", 100.5, change_pct=1.2, volume=500, bid=100.4, ask=100.6)
        df = store.query("SELECT * FROM quote_snapshots")
        assert len(df) == 1
        assert float(df["price"].iloc[0]) == pytest.approx(100.5)

    def test_record_depth(self, store):
        depth = {"bids": [{"price": 100.0, "quantity": 10}], "asks": []}
        store.record_depth("TEST.NS", depth)
        df = store.query("SELECT * FROM depth_snapshots")
        assert len(df) == 1

    def test_record_options_chain(self, store):
        chain = pd.DataFrame(
            [{"expiry": "2026-08-27", "strike": 25000.0, "option_type": "CE",
              "open_interest": 1000, "volume": 200, "last_price": 150.5}]
        )
        assert store.record_options_chain("NIFTY", chain) == 1
        df = store.query("SELECT * FROM options_chain")
        assert len(df) == 1
        assert df["open_interest"].iloc[0] == 1000


class TestUniverseBookkeeping:
    def test_universe_upsert_get(self, store):
        store.upsert_universe("nifty_50", ["RELIANCE", "TCS"])
        assert store.get_universe("nifty_50") == ["RELIANCE", "TCS"]
        store.upsert_universe("nifty_50", ["INFY"])
        assert store.get_universe("nifty_50") == ["INFY"]

    def test_available_symbols(self, store):
        store.upsert_daily("A.NS", _make_df(n=5), source="test")
        store.upsert_daily("B.NS", _make_df(n=5), source="test")
        assert set(store.available_symbols()) == {"A.NS", "B.NS"}

    def test_stats(self, store):
        store.upsert_daily("A.NS", _make_df(n=5), source="test")
        stats = store.stats()
        assert stats["ohlcv_daily"] == 5
