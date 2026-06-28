from __future__ import annotations

from functools import lru_cache

import yfinance as yf


def get_yahoo_currency(ticker: str) -> str:
    try:
        tobj = yf.Ticker(ticker)
    except Exception:
        return ""
    try:
        info = getattr(tobj, "info", None)
        if isinstance(info, dict):
            ccy = (info.get("currency", "") or "").strip()
            if ccy:
                return ccy
    except Exception:
        pass
    try:
        fi = getattr(tobj, "fast_info", None)
        if fi is not None:
            ccy = (getattr(fi, "currency", "") or "").strip()
            if ccy:
                return ccy
    except Exception:
        pass
    return ""


@lru_cache(maxsize=512)
def get_yahoo_currency_cached(ticker: str) -> str:
    return get_yahoo_currency(ticker)
