"""Tests for src/flow.py — FII/DII flow analysis."""

import numpy as np
import pandas as pd
import pytest

from src.flow import get_flow_sentiment, _compute_pcr


class TestGetFlowSentiment:
    def test_bullish_fii_inflow(self):
        result = get_flow_sentiment(fii_net=5000, dii_net=1000)
        assert isinstance(result, str)

    def test_bearish_fii_outflow(self):
        result = get_flow_sentiment(fii_net=-5000, dii_net=1000)
        assert isinstance(result, str)

    def test_mixed_fii_out_dii_in(self):
        result = get_flow_sentiment(fii_net=-5000, dii_net=5000)
        assert isinstance(result, str)

    def test_neutral_flow(self):
        result = get_flow_sentiment(fii_net=0, dii_net=0)
        assert isinstance(result, str)

    def test_both_negative(self):
        result = get_flow_sentiment(fii_net=-5000, dii_net=-5000)
        assert isinstance(result, str)

    def test_returns_string(self):
        result = get_flow_sentiment(fii_net=1000, dii_net=-500)
        assert isinstance(result, str)

    def test_strong_bullish(self):
        result = get_flow_sentiment(fii_net=10000, dii_net=5000)
        assert isinstance(result, str)

    def test_strong_bearish(self):
        result = get_flow_sentiment(fii_net=-10000, dii_net=-5000)
        assert isinstance(result, str)


class TestComputePcr:
    def test_basic_pcr_from_option_chain(self):
        data = {
            "records": [
                {"strikePrice": 100, "CE": {"openInterest": 500, "totalTradedVolume": 200}, "PE": {"openInterest": 800, "totalTradedVolume": 300}},
                {"strikePrice": 110, "CE": {"openInterest": 300, "totalTradedVolume": 100}, "PE": {"openInterest": 400, "totalTradedVolume": 150}},
            ]
        }
        result = _compute_pcr(data)
        assert isinstance(result, dict)
        assert "pcr_oi" in result
        assert "pcr_volume" in result
        assert result["pcr_oi"] > 0

    def test_empty_records(self):
        data = {"records": []}
        result = _compute_pcr(data)
        assert result["pcr_oi"] == 0

    def test_no_records_key(self):
        data = {}
        result = _compute_pcr(data)
        assert result["pcr_oi"] == 0

    def test_zero_oi(self):
        data = {
            "records": [
                {"strikePrice": 100, "CE": {"openInterest": 0, "totalTradedVolume": 0}, "PE": {"openInterest": 0, "totalTradedVolume": 0}},
            ]
        }
        result = _compute_pcr(data)
        assert result["pcr_oi"] == 0.0
        assert result["pcr_volume"] == 0.0

    def test_has_max_pain(self):
        data = {
            "records": [
                {"strikePrice": 100, "CE": {"openInterest": 500, "totalTradedVolume": 200}, "PE": {"openInterest": 800, "totalTradedVolume": 300}},
                {"strikePrice": 110, "CE": {"openInterest": 300, "totalTradedVolume": 100}, "PE": {"openInterest": 400, "totalTradedVolume": 150}},
            ]
        }
        result = _compute_pcr(data)
        assert "max_pain" in result
        assert result["max_pain"] >= 0

    def test_put_heavy_pcr_gt_1(self):
        data = {
            "records": [
                {"strikePrice": 100, "CE": {"openInterest": 100, "totalTradedVolume": 50}, "PE": {"openInterest": 500, "totalTradedVolume": 300}},
            ]
        }
        result = _compute_pcr(data)
        assert result["pcr_oi"] > 1.0

    def test_call_heavy_pcr_lt_1(self):
        data = {
            "records": [
                {"strikePrice": 100, "CE": {"openInterest": 500, "totalTradedVolume": 300}, "PE": {"openInterest": 100, "totalTradedVolume": 50}},
            ]
        }
        result = _compute_pcr(data)
        assert result["pcr_oi"] < 1.0
