"""Tests for stock universe definitions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.data.universes import (
    NIFTY_50,
    NIFTY_MIDCAP_100,
    NIFTY_NEXT_50,
    UNIVERSES,
    custom_universe,
    get_universe,
    nse_symbols,
    yf_ticker,
    yf_tickers,
)


class TestUniverseSizes:
    def test_nifty_50_has_50(self):
        assert len(NIFTY_50) == 50

    def test_next_50_has_50(self):
        assert len(NIFTY_NEXT_50) == 50

    def test_midcap_100_has_100(self):
        assert len(NIFTY_MIDCAP_100) == 100


class TestUniverseIntegrity:
    @pytest.mark.parametrize("name", ["nifty_50", "nifty_next_50", "nifty_midcap_100"])
    def test_no_duplicates(self, name):
        symbols = get_universe(name)
        assert len(symbols) == len(set(symbols))

    @pytest.mark.parametrize("name", ["nifty_50", "nifty_next_50", "nifty_midcap_100"])
    def test_uppercase_no_suffix(self, name):
        for symbol in get_universe(name):
            assert symbol.isupper()
            assert not symbol.endswith(".NS")

    def test_get_universe_aliases(self):
        assert get_universe("nifty50") == get_universe("nifty_50")
        assert get_universe("NIFTY") == get_universe("nifty_50")
        assert get_universe("midcap") == get_universe("nifty_midcap_100")

    def test_unknown_universe_raises(self):
        with pytest.raises(KeyError):
            get_universe("not-a-universe")

    def test_all_universes_deduplicated(self):
        from src.data.universes import all_universes

        symbols = all_universes()
        assert len(symbols) == len(set(symbols))
        assert "RELIANCE" in symbols


class TestSymbolConversion:
    def test_yf_ticker_appends_ns(self):
        assert yf_ticker("RELIANCE") == "RELIANCE.NS"

    def test_yf_ticker_idempotent(self):
        assert yf_ticker("TCS.NS") == "TCS.NS"

    def test_yf_ticker_keeps_index(self):
        assert yf_ticker("^NSEI") == "^NSEI"

    def test_yf_tickers_converts_list(self):
        assert yf_tickers(["RELIANCE", "TCS"]) == ["RELIANCE.NS", "TCS.NS"]

    def test_nse_symbols_strips_suffix(self):
        assert nse_symbols(["RELIANCE.NS", "TCS"]) == ["RELIANCE", "TCS"]

    def test_get_universe_as_yf(self):
        symbols = get_universe("nifty_50", as_yf=True)
        assert len(symbols) == 50
        assert all(s.endswith(".NS") for s in symbols)


class TestCustomUniverse:
    def test_custom_universe_normalizes(self):
        result = custom_universe(["reliance.ns", "TCS", "INFY.NS"])
        assert result == ["RELIANCE", "TCS", "INFY"]

    def test_custom_universe_as_yf(self):
        result = custom_universe(["RELIANCE", "TCS.NS"], as_yf=True)
        assert result == ["RELIANCE.NS", "TCS.NS"]

    def test_universes_registry_keys(self):
        assert set(UNIVERSES) == {"nifty_50", "nifty_next_50", "nifty_midcap_100"}
