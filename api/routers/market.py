"""Market status and pulse endpoint."""

from fastapi import APIRouter
from src.data_fetcher import NSE_STOCKS, get_market_status
from src.flow import fetch_fii_dii, fetch_options_pcr, get_flow_sentiment
from src.multitimeframe import fetch_mtf_data, get_combined_signal

router = APIRouter()


@router.get("/status")
def market_status():
    return {"status": get_market_status()}


@router.get("/stocks")
def list_stocks():
    return {"stocks": NSE_STOCKS}


@router.get("/pulse")
def market_pulse():
    try:
        result = {}

        fii_dii = fetch_fii_dii()
        if fii_dii is not None and not fii_dii.empty:
            latest = fii_dii.iloc[-1]
            fii_net = float(latest.get("FII", 0))
            dii_net = float(latest.get("DII", 0))
            result["fii_dii"] = {
                "fii_net": round(fii_net, 2),
                "dii_net": round(dii_net, 2),
                "flow_sentiment": get_flow_sentiment(fii_net, dii_net),
            }

        pcr = fetch_options_pcr()
        if pcr:
            result["options_pcr"] = pcr

        mtf = fetch_mtf_data("RELIANCE.NS")
        if mtf:
            combined = get_combined_signal(mtf)
            result["multi_timeframe"] = combined
            result["timeframes"] = {}
            for tf_name, tf_data in mtf.items():
                if isinstance(tf_data, dict):
                    result["timeframes"][tf_name] = tf_data

        return result
    except Exception as e:
        return {"error": str(e)}
