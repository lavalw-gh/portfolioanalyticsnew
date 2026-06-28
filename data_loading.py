from __future__ import annotations

from datetime import datetime
from io import BytesIO
import math
from pathlib import Path

import pandas as pd
import yfinance as yf

from .constants import CASH_SYMBOL
from .yahoo_metadata import get_yahoo_currency_cached


def _extract_latest_close(data: pd.DataFrame, ticker: str) -> float | None:
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"]
        else:
            return None
        series = close[ticker] if isinstance(close, pd.DataFrame) and ticker in close.columns else close.iloc[:, 0]
    else:
        if "Close" in data.columns:
            series = data["Close"]
        elif "Adj Close" in data.columns:
            series = data["Adj Close"]
        else:
            return None
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    price = float(series.iloc[-1])
    return price if math.isfinite(price) and price > 0 else None


def _download_comparison_close(ticker: str, csv_date) -> float | None:
    start = pd.Timestamp(csv_date) - pd.Timedelta(days=10)
    end = min(pd.Timestamp(csv_date) + pd.Timedelta(days=10), pd.Timestamp(datetime.now()))
    if end <= start:
        end = pd.Timestamp(datetime.now())
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        return _extract_latest_close(data, ticker)
    except Exception:
        return None


def _csv_price_should_divide_by_100(csv_price: float, yahoo_price_major_unit: float | None) -> bool:
    if yahoo_price_major_unit is None or not math.isfinite(yahoo_price_major_unit) or yahoo_price_major_unit <= 0:
        return False
    if not math.isfinite(csv_price) or csv_price <= 0:
        return False
    raw_ratio = csv_price / yahoo_price_major_unit
    divided_ratio = (csv_price / 100.0) / yahoo_price_major_unit
    raw_err = abs(math.log(raw_ratio))
    divided_err = abs(math.log(divided_ratio))
    return 20.0 <= raw_ratio <= 250.0 and divided_err <= raw_err * 0.35 and 0.50 <= divided_ratio <= 2.00


def normalize_and_rebase_to_pounds(df_tx: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    tickers = df_tx["ticker"].unique()
    conversion_log: list[str] = []
    conversion_factors: dict[str, float] = {}
    yahoo_names: dict[str, str] = {}
    for ticker in tickers:
        if ticker == CASH_SYMBOL:
            yahoo_names[ticker] = ticker
            conversion_factors[ticker] = 1.0
            conversion_log.append(
                f"✅ {ticker}: treated as GBP cash with constant £1 valuation")
            continue
        try:
            ticker_info = yf.Ticker(ticker)
            long_name = None
            try:
                info_dict = ticker_info.info
                long_name = info_dict.get(
                    "longName") or info_dict.get("shortName")
            except Exception:
                long_name = None
            yahoo_names[ticker] = str(long_name) if long_name else ticker
            currency = get_yahoo_currency_cached(ticker)
            ticker_txs = df_tx[df_tx["ticker"] == ticker]
            csv_price_original = float(ticker_txs["price"].iloc[-1])
            csv_date = ticker_txs["date"].iloc[-1]
            yahoo_close = _download_comparison_close(ticker, csv_date)
            if currency in ("GBp", "GBX"):
                conversion_log.append(
                    f"✅ {ticker}: Yahoo in pence ({currency}) → Yahoo prices ÷100 to £")
                conversion_factors[ticker] = 100.0
                yahoo_price_major = yahoo_close / 100.0 if yahoo_close else None
                if _csv_price_should_divide_by_100(csv_price_original, yahoo_price_major):
                    conversion_log.append(
                        f" CSV {csv_price_original:.0f} converted to £{csv_price_original / 100:.2f}")
                    mask = df_tx["ticker"] == ticker
                    df_tx.loc[mask, "price"] = df_tx.loc[mask, "price"] / 100.0
                else:
                    if yahoo_price_major is None:
                        conversion_log.append(
                            f" CSV price left unchanged at £{csv_price_original:.2f}; comparison price unavailable")
                    else:
                        conversion_log.append(
                            f" CSV already appears in pounds at £{csv_price_original:.2f}")
            elif currency in ("GBP", "USD", "EUR"):
                conversion_log.append(
                    f"✅ {ticker}: Yahoo in major currency units ({currency})")
                conversion_factors[ticker] = 1.0
                if _csv_price_should_divide_by_100(csv_price_original, yahoo_close):
                    conversion_log.append(
                        f" CSV {csv_price_original:.0f} converted to £{csv_price_original / 100:.2f}")
                    mask = df_tx["ticker"] == ticker
                    df_tx.loc[mask, "price"] = df_tx.loc[mask, "price"] / 100.0
                else:
                    conversion_log.append(
                        f" CSV already appears in pounds at £{csv_price_original:.2f}")
            else:
                conversion_log.append(
                    f"⚠️ {ticker}: unknown Yahoo currency ({currency}), using ratio fallback")
                if yahoo_close is not None:
                    yahoo_price = yahoo_close
                    ratio = yahoo_price / csv_price_original if csv_price_original else 1.0
                    if ratio > 50:
                        conversion_log.append(
                            f" Ratio {ratio:.1f}x suggests Yahoo in pence")
                        conversion_factors[ticker] = 100.0
                    else:
                        conversion_log.append(
                            f" Ratio {ratio:.2f}x suggests prices already in major units")
                        conversion_factors[ticker] = 1.0
                else:
                    conversion_log.append(
                        " Could not download comparison price, assuming major currency units")
                    conversion_factors[ticker] = 1.0
        except Exception as e:
            conversion_log.append(f"❌ {ticker}: normalization error - {e}")
            conversion_factors[ticker] = 1.0
        if ticker not in yahoo_names:
            yahoo_names[ticker] = ticker
    return df_tx, conversion_factors, yahoo_names, conversion_log


def _prepare_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "ticker", "shares", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"].astype(
        str).str.strip(), format="mixed", dayfirst=True, errors="raise")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["shares"] = pd.to_numeric(df["shares"], errors="raise")
    df["price"] = pd.to_numeric(df["price"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    if "total_cost" in df.columns:
        df["total_cost"] = pd.to_numeric(df["total_cost"], errors="coerce")
        derived_total_cost_mask = df["total_cost"].isna()
    else:
        df["total_cost"] = pd.NA
        derived_total_cost_mask = pd.Series(True, index=df.index)

    df, conversion_factors, yahoo_names, notes = normalize_and_rebase_to_pounds(
        df)
    if derived_total_cost_mask.any():
        df.loc[derived_total_cost_mask, "total_cost"] = (
            df.loc[derived_total_cost_mask, "shares"]
            * df.loc[derived_total_cost_mask, "price"]
        )
    df["total_cost"] = pd.to_numeric(df["total_cost"], errors="coerce")
    return df, conversion_factors, yahoo_names, notes


def load_transactions(file_path: str | Path) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    return _prepare_transactions(pd.read_csv(file_path))


def load_transactions_from_bytes(data: bytes) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    return _prepare_transactions(pd.read_csv(BytesIO(data)))
