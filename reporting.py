from __future__ import annotations

from datetime import date
import warnings

import numpy as np
import pandas as pd


def format_shares(x):
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return f"{x:,.4f}".rstrip("0").rstrip(".")


def build_holdings_overview(required, pdata, yahoo_names_all: dict[str, str], start_effective: date, end_effective: date) -> tuple[pd.DataFrame, float]:
    holdings_asof = pdata["holdings_eod"][
        pdata["holdings_eod"].index <= pd.Timestamp(end_effective)
    ]
    values_asof = pdata["values_eod"][
        pdata["values_eod"].index <= pd.Timestamp(end_effective)
    ]
    latest_hold = holdings_asof.iloc[-1]
    latest_values = values_asof.iloc[-1]
    total_val = float(latest_values.sum())

    end_hold_mask = latest_hold != 0
    tickers_idx = latest_hold[end_hold_mask].index
    names_series = pd.Index(tickers_idx.map(lambda t: yahoo_names_all.get(t, t)))
    latest_hold_filtered = latest_hold[end_hold_mask]
    latest_values_filtered = latest_values[end_hold_mask]
    weights_now = latest_values_filtered / total_val if total_val > 0 else latest_values_filtered * 0

    asset_prices = pdata["asset_prices"]
    ytd_start = date(end_effective.year, 1, 1)
    ytd_prices = asset_prices[(asset_prices.index >= pd.Timestamp(ytd_start)) & (asset_prices.index <= pd.Timestamp(end_effective))]
    period_prices = asset_prices[(asset_prices.index >= pd.Timestamp(start_effective)) & (asset_prices.index <= pd.Timestamp(end_effective))]

    ytd_ret = (ytd_prices.iloc[-1] / ytd_prices.iloc[0] - 1.0) if len(ytd_prices) >= 2 else pd.Series(np.nan, index=asset_prices.columns)
    period_ret = (period_prices.iloc[-1] / period_prices.iloc[0] - 1.0) if len(period_prices) >= 2 else pd.Series(np.nan, index=asset_prices.columns)

    tx = required["tx"]
    cost_by_ticker = tx.groupby("ticker")["total_cost"].sum().reindex(tickers_idx).fillna(0.0)

    holdings_df = pd.DataFrame({
        "Ticker": tickers_idx,
        "Name": names_series,
        "Shares": latest_hold_filtered.reindex(tickers_idx).values,
        "Cost (£)": cost_by_ticker.values,
        "Value (£)": latest_values_filtered.reindex(tickers_idx).values,
        "Weight": weights_now.reindex(tickers_idx).values,
        "YTD Return": ytd_ret.reindex(tickers_idx).values,
        "Period Return": period_ret.reindex(tickers_idx).values,
    })

    holdings_df["Shares"] = holdings_df["Shares"].map(format_shares)
    holdings_df["Cost (£)"] = holdings_df["Cost (£)"].map(lambda x: f"£{x:,.2f}")
    holdings_df["Value (£)"] = holdings_df["Value (£)"].map(lambda x: f"£{x:,.2f}")
    holdings_df["Weight"] = holdings_df["Weight"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    holdings_df["YTD Return"] = holdings_df["YTD Return"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
    holdings_df["Period Return"] = holdings_df["Period Return"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")

    holdings_df = holdings_df.sort_values(
        "Value (£)", ascending=False,
        key=lambda s: s.str.replace("£", "", regex=False).str.replace(",", "", regex=False).astype(float)
    )
    holdings_df = holdings_df.reset_index(drop=True)

    # --- TOTAL ROW ---
    try:
        total_cost = sum(
            float(str(v).replace("£", "").replace(",", ""))
            for v in holdings_df["Cost (£)"] if str(v).strip() not in ("", "nan")
        )
        total_value = sum(
            float(str(v).replace("£", "").replace(",", ""))
            for v in holdings_df["Value (£)"] if str(v).strip() not in ("", "nan")
        )
        total_row = {col: "" for col in holdings_df.columns}
        total_row["Ticker"] = "TOTAL"
        total_row["Name"] = ""
        total_row["Cost (£)"] = f"£{total_cost:,.2f}"
        total_row["Value (£)"] = f"£{total_value:,.2f}"
        total_row["Weight"] = "100.00%"
        holdings_df = pd.concat(
            [holdings_df, pd.DataFrame([total_row])], ignore_index=True
        )
    except Exception as exc:
        warnings.warn(f"Could not add holdings total row: {exc}", RuntimeWarning)

    return holdings_df, total_val


def build_metrics_table(portfolio_name: str, metrics: dict, benchmarkmetrics: dict | None = None) -> pd.DataFrame:
    metric_order = [
        ("Total Return", "total_return", True),
        ("Volatility (annual)", "vol_annual", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Downside Dev", "downside_dev", True),
        ("Max Drawdown", "max_drawdown", True),
    ]

    def fmt(value, is_pct: bool) -> str:
        if pd.isna(value):
            return ""
        return f"{value:.2%}" if is_pct else f"{value:.2f}"

    rows = []
    for label, key, is_pct in metric_order:
        rows.append({
            "Metric": label,
            "Portfolio Value": fmt(metrics.get(key, np.nan), is_pct),
            "Benchmark Value": fmt(benchmarkmetrics.get(key, np.nan), is_pct) if benchmarkmetrics is not None else "",
        })

    return pd.DataFrame(rows, columns=["Metric", "Portfolio Value", "Benchmark Value"])


def build_price_quality_table(price_df: pd.DataFrame) -> pd.DataFrame:
    attrs = getattr(price_df, "attrs", {}) or {}
    rows: list[dict] = []

    for item in attrs.get("yahoo_issues", []) or []:
        rows.append({
            "Category": "Yahoo download",
            "Symbol": item.get("symbol", ""),
            "Date / Range": "",
            "Issue": item.get("problem", ""),
            "Action": "flagged",
            "Old Price": "",
            "New Price": "",
            "Method": "",
            "Confidence": "",
        })

    for item in attrs.get("missing_ranges", []) or []:
        rows.append({
            "Category": "Missing prices",
            "Symbol": item.get("symbol", ""),
            "Date / Range": f"{item.get('start', '')} to {item.get('end', '')}",
            "Issue": f"{item.get('type', '')} ({item.get('missing_days', '')} days)",
            "Action": "flagged",
            "Old Price": "",
            "New Price": "",
            "Method": "",
            "Confidence": "",
        })

    for item in attrs.get("forward_fill_report", []) or []:
        action = "carried forward"
        if item.get("exceeded_limit"):
            action = "left blank after carry limit"
        rows.append({
            "Category": "Forward fill",
            "Symbol": item.get("symbol", ""),
            "Date / Range": f"{item.get('start', '')} to {item.get('end', '')}",
            "Issue": f"{item.get('carried_days', '')} missing price days",
            "Action": action,
            "Old Price": "",
            "New Price": item.get("fill_value", ""),
            "Method": f"max {item.get('max_allowed_days', '')} days",
            "Confidence": "",
        })

    correction_groups = [
        ("Unit regime", "ltm_unit_report"),
        ("GBP unit mix", "gbp_unit_mix_report"),
        ("Single-day unit", "singleton_unit_report"),
        ("Residual spike", "residual_corrections"),
        ("Short-term residual", "stm_residual_report"),
        ("Robust local", "robust_local_report"),
    ]
    for category, attr_name in correction_groups:
        for item in attrs.get(attr_name, []) or []:
            symbol = item.get("Ticker", item.get("symbol", ""))
            dt = item.get("Date", item.get("date", ""))
            issue = item.get("Issue", item.get("issue", category))
            action = item.get("Action", "corrected")
            if action in ("no_change", "checked", "none"):
                continue
            rows.append({
                "Category": category,
                "Symbol": symbol,
                "Date / Range": str(dt)[:10],
                "Issue": issue,
                "Action": action,
                "Old Price": item.get("Old Price", item.get("old_price", "")),
                "New Price": item.get("New Price", item.get("new_price", "")),
                "Method": item.get("Method", item.get("method", "")),
                "Confidence": item.get("Confidence", ""),
            })

    columns = [
        "Category",
        "Symbol",
        "Date / Range",
        "Issue",
        "Action",
        "Old Price",
        "New Price",
        "Method",
        "Confidence",
    ]
    if not rows:
        return pd.DataFrame([{
            "Category": "Price data",
            "Symbol": "",
            "Date / Range": "",
            "Issue": "No price data issues detected",
            "Action": "",
            "Old Price": "",
            "New Price": "",
            "Method": "",
            "Confidence": "",
        }], columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_calculations_export_df(results: dict) -> pd.DataFrame:
    frames = []
    benchmark_series = results.get("bench_cum_window", pd.Series(dtype=float))
    benchmark_daily = results.get("bench_daily_window", pd.Series(dtype=float))
    start_effective = results.get("start_effective")
    end_effective = results.get("end_effective")
    for pname, pdata in results.get("portfolio_data", {}).items():
        dates = pdata["portfolio_value_window"].index
        base = pd.DataFrame(index=dates)
        base["Portfolio"] = pname
        base["Date"] = base.index.date
        base["Total Value (£)"] = pdata["portfolio_value_window"].reindex(
            dates)
        base["Portfolio Daily Return"] = pdata["twr_daily"].reindex(dates)
        base["Portfolio Cum Return"] = pdata["twr_cum"].reindex(dates)
        base["Benchmark Daily Return"] = benchmark_daily.reindex(dates)
        base["Benchmark Cum Return"] = benchmark_series.reindex(dates)
        holdings_window = pdata["holdings_eod"][(pdata["holdings_eod"].index.date >= start_effective) & (
            pdata["holdings_eod"].index.date <= end_effective)]
        values_window = pdata["values_eod"][(pdata["values_eod"].index.date >= start_effective) & (
            pdata["values_eod"].index.date <= end_effective)]
        for ticker in holdings_window.columns:
            base[f"Holdings {ticker}"] = holdings_window[ticker].reindex(dates)
            base[f"Value {ticker}"] = values_window[ticker].reindex(dates)
        frames.append(base.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

