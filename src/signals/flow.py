import requests
import pandas as pd
import json
import os
import time
import logging
import threading

from src.core.constants import DATA_DIR
from src.data.resilience import nse_breaker

logger = logging.getLogger(__name__)

FLOW_CACHE_DIR = DATA_DIR
os.makedirs(FLOW_CACHE_DIR, exist_ok=True)
_flow_lock = threading.Lock()
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/html",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/fii-dii",
}


def _get_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        nse_breaker.call(session.get, "https://www.nseindia.com", timeout=10)
        time.sleep(0.3)
    except Exception:
        pass
    return session


def fetch_fii_dii() -> pd.DataFrame:
    cache_file = os.path.join(FLOW_CACHE_DIR, "fii_dii.parquet")
    with _flow_lock:
        if os.path.exists(cache_file):
            cached = pd.read_parquet(cache_file)
            if len(cached) > 0:
                last_date = pd.to_datetime(cached.iloc[0]["date"])
                if pd.Timestamp.now().normalize() - last_date <= pd.Timedelta(days=1):
                    return cached

    try:
        session = _get_session()
        resp = nse_breaker.call(session.get, "https://www.nseindia.com/api/fiidiiTradeReact", timeout=15)
        resp.raise_for_status()
        data = resp.json()

        fii_row = None
        dii_row = None
        if isinstance(data, list):
            for row in data:
                cat = row.get("category", "")
                if "FII" in cat or "FPI" in cat:
                    fii_row = row
                elif "DII" in cat:
                    dii_row = row

        if not fii_row:
            logger.warning("FII row not found in NSE response, using cached data")
            with _flow_lock:
                if os.path.exists(cache_file):
                    return pd.read_parquet(cache_file)
            return pd.DataFrame()

        date = pd.Timestamp(fii_row.get("date", "")).tz_localize(None).floor("D")
        fii_buy = _parse_val(fii_row.get("buyValue", 0))
        fii_sell = _parse_val(fii_row.get("sellValue", 0))
        fii_net = _parse_val(fii_row.get("netValue", 0))

        dii_buy = _parse_val(dii_row.get("buyValue", 0)) if dii_row is not None else 0
        dii_sell = _parse_val(dii_row.get("sellValue", 0)) if dii_row is not None else 0
        dii_net = _parse_val(dii_row.get("netValue", 0)) if dii_row is not None else 0

        new_row = pd.DataFrame([{
            "date": date, "fii_buy": fii_buy, "fii_sell": fii_sell,
            "fii_net": fii_net, "dii_buy": dii_buy, "dii_sell": dii_sell,
            "dii_net": dii_net,
        }])

        with _flow_lock:
            if os.path.exists(cache_file):
                old = pd.read_parquet(cache_file)
                combined = pd.concat([new_row, old], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="first").head(60)
            else:
                combined = new_row
            combined.to_parquet(cache_file)
        return combined
    except Exception:
        with _flow_lock:
            if os.path.exists(cache_file):
                return pd.read_parquet(cache_file)
        return pd.DataFrame()


def fetch_options_pcr() -> dict:
    cache_file = os.path.join(FLOW_CACHE_DIR, "options_pcr.json")
    with _flow_lock:
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < 3600:
                try:
                    with open(cache_file) as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to read PCR cache: %s", e)

    for symbol in ["NIFTY", "BANKNIFTY"]:
        try:
            session = _get_session()
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            resp = nse_breaker.call(session.get, url, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            pcr = _compute_pcr(data)
            pcr["symbol"] = symbol
            import tempfile
            fd, tmp_path = tempfile.mkstemp(dir=FLOW_CACHE_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(pcr, f)
                with _flow_lock:
                    os.replace(tmp_path, cache_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return pcr
        except Exception as e:
            logger.debug("PCR fetch failed for %s: %s", symbol, e)
            continue

    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"pcr_oi": 0, "pcr_volume": 0, "max_pain": 0, "call_oi": 0, "put_oi": 0, "symbol": "NIFTY"}


def _compute_pcr(data: dict) -> dict:
    records = data.get("records", [])
    if not records:
        return {"pcr_oi": 0, "pcr_volume": 0, "max_pain": 0, "call_oi": 0, "put_oi": 0}

    call_oi = 0
    put_oi = 0
    call_vol = 0
    put_vol = 0
    oi_map = {}

    for rec in records:
        strike = rec.get("strikePrice", 0)
        ce = rec.get("CE", {})
        pe = rec.get("PE", {})
        co = ce.get("openInterest", 0)
        po = pe.get("openInterest", 0)
        cv = ce.get("totalTradedVolume", 0)
        pv = pe.get("totalTradedVolume", 0)
        call_oi += co
        put_oi += po
        call_vol += cv
        put_vol += pv
        if strike not in oi_map:
            oi_map[strike] = {"ce": 0, "pe": 0}
        oi_map[strike]["ce"] += co
        oi_map[strike]["pe"] += po

    pcr_oi = round(put_oi / max(call_oi, 1), 3)
    pcr_vol = round(put_vol / max(call_vol, 1), 3)

    max_pain = 0
    min_pain = float("inf")
    strikes = sorted(oi_map.keys())
    for s in strikes:
        pain = sum(
            max(0, s - k) * v["pe"] + max(0, k - s) * v["ce"]
            for k, v in oi_map.items()
        )
        if pain < min_pain:
            min_pain = pain
            max_pain = s

    return {"pcr_oi": pcr_oi, "pcr_volume": pcr_vol, "max_pain": max_pain, "call_oi": call_oi, "put_oi": put_oi}


def _parse_val(val):
    if isinstance(val, str):
        val = val.replace(",", "").replace(" ", "")
        if val in ("", "-", "--"):
            return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0
    return float(val or 0)


def get_flow_sentiment(fii_net: float, dii_net: float) -> str:
    if fii_net > 1000 and dii_net > 0:
        return "Strong Bullish"
    elif fii_net > 0 and dii_net > 0:
        return "Bullish"
    elif fii_net > 0 and dii_net < 0:
        return "Mixed"
    elif fii_net < -1000 and dii_net > 0:
        return "Divergent"
    elif fii_net < 0:
        return "Bearish"
    return "Neutral"
