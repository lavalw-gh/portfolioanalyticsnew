from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from .analytics import build_holdings_for_portfolios, compute_period_twr, compute_risk_metrics, compute_twr_for_portfolios
from .models import LoadedPortfolio
from .reporting import build_calculations_export_df, build_holdings_overview, build_metrics_table, build_price_quality_table
from .yahoo_prices import get_price_history


def run_analysis_job(loaded_portfolios: dict[str, LoadedPortfolio], selected_names: list[str], benchmark: str, start_requested: date, end_requested: date, preset: str, rf_annual: float, hampel_threshold: float, progress_callback: Callable[[str, int], None] | None = None) -> dict:
    def emit(message: str, pct: int):
        if progress_callback is not None:
            progress_callback(message, int(max(0, min(100, pct))))

    emit("Preparing portfolios...", 0)

    all_tickers: set[str] = set()
    merged_conversion_factors: dict[str, float] = {}
    yahoo_names_all: dict[str, str] = {}
    normalization_notes: dict[str, list[str]] = {}
    min_start_date: date | None = None
    for pname in selected_names:
        loaded = loaded_portfolios[pname]
        all_tickers.update(loaded.tx["ticker"].unique())
        merged_conversion_factors.update(loaded.conversion_factors)
        yahoo_names_all.update(loaded.yahoo_names)
        normalization_notes[pname] = loaded.normalization_notes
        pmin = loaded.tx["date"].min().date()
        min_start_date = pmin if min_start_date is None else min(min_start_date, pmin)

    start_all = min_start_date or start_requested
    emit("Loading price history...", 3)
    price_df = get_price_history(
        list(all_tickers),
        benchmark,
        start_all,
        end_requested + timedelta(days=1),
        merged_conversion_factors,
        hampel_threshold=hampel_threshold,
        progress_callback=progress_callback,
    )

    if benchmark not in price_df.columns:
        raise RuntimeError(f"Benchmark {benchmark} not found in downloaded data.")

    bench_prices = price_df[benchmark]
    asset_prices = price_df.drop(columns=[benchmark], errors="ignore")
    benchmark_first = bench_prices.dropna().index.min().date() if not bench_prices.dropna().empty else start_requested
    benchmark_last = bench_prices.dropna().index.max().date() if not bench_prices.dropna().empty else end_requested
    portfolio_first_dates = [loaded_portfolios[p].tx["date"].min().date() for p in selected_names]
    common_start = max([benchmark_first] + portfolio_first_dates)
    common_end = min(benchmark_last, end_requested)

    if preset == "Max":
        start_effective = common_start
        end_effective = common_end
    else:
        start_effective = max(start_requested, common_start)
        end_effective = min(end_requested, common_end)
    if end_effective < start_effective:
        raise RuntimeError("No overlapping date range is available for the selected portfolios and benchmark.")

    emit("Computing benchmark...", 68)
    bench_window = bench_prices[(bench_prices.index.date >= start_effective) & (bench_prices.index.date <= end_effective)]
    bench_daily = bench_window.pct_change().fillna(0.0)
    bench_cum_window = (1.0 + bench_daily).cumprod() - 1.0
    bench_rf_daily = pd.Series(rf_annual / 252.0, index=bench_daily.index)
    benchmark_metrics = compute_risk_metrics(bench_daily, bench_rf_daily)

    emit("Building holdings...", 74)
    portfolio_tx_map = {pname: loaded_portfolios[pname].tx.copy() for pname in selected_names}
    holdings_bod_map, holdings_eod_map = build_holdings_for_portfolios(portfolio_tx_map, asset_prices)

    emit("Computing returns...", 80)
    twr_daily_map, twr_cum_map = compute_twr_for_portfolios(asset_prices, holdings_bod_map)

    portfolio_data: dict[str, dict] = {}
    for pname in selected_names:
        tx = loaded_portfolios[pname].tx.copy()
        holdings_bod = holdings_bod_map[pname]
        holdings_eod = holdings_eod_map[pname]
        twr_daily_all = twr_daily_map[pname]
        twr_cum_all = twr_cum_map[pname]

        mask = (asset_prices.index >= pd.Timestamp(start_effective)) & (asset_prices.index <= pd.Timestamp(end_effective))
        asset_prices_window = asset_prices.loc[mask].copy()
        holdings_bod_window = holdings_bod.loc[mask].copy()
        holdings_eod_window = holdings_eod.loc[mask].copy()
        twr_daily_window = compute_period_twr(twr_daily_all, start_effective, end_effective)
        twr_cum_window = (1.0 + twr_daily_window).cumprod() - 1.0 if not twr_daily_window.empty else pd.Series(dtype=float)
        values_eod_window = holdings_eod_window * asset_prices_window
        portfolio_value_window = values_eod_window.sum(axis=1)
        rf_daily_window = pd.Series(rf_annual / 252.0, index=twr_daily_window.index)
        metrics = compute_risk_metrics(twr_daily_window, rf_daily_window)

        pdata = {
            "tx": tx,
            "asset_prices": asset_prices,
            "asset_prices_window": asset_prices_window,
            "holdings_bod": holdings_bod,
            "holdings_eod": holdings_eod,
            "holdings_bod_window": holdings_bod_window,
            "holdings_eod_window": holdings_eod_window,
            "values_eod": holdings_eod * asset_prices,
            "values_eod_window": values_eod_window,
            "portfolio_value_window": portfolio_value_window,
            "twr_daily": twr_daily_window,
            "twr_cum": twr_cum_window,
            "metrics": metrics,
        }

        holdings_df, total_val = build_holdings_overview({"tx": tx}, pdata, yahoo_names_all, start_effective, end_effective)
        metrics_df = build_metrics_table(pname, metrics, benchmark_metrics)
        pdata["holdings_df"] = holdings_df
        pdata["metrics_df"] = metrics_df
        pdata["total_value"] = total_val
        portfolio_data[pname] = pdata

    emit("Finalizing...", 92)
    results = {
        "selected_names": selected_names,
        "benchmark": benchmark,
        "start_requested": start_requested,
        "end_requested": end_requested,
        "start_effective": start_effective,
        "end_effective": end_effective,
        "rf_annual": rf_annual,
        "bench_daily_window": bench_daily,
        "bench_cum_window": bench_cum_window,
        "benchmark_metrics": benchmark_metrics,
        "portfolio_data": portfolio_data,
        "yahoo_names_all": yahoo_names_all,
        "normalization_notes": normalization_notes,
        "price_df": price_df,
        "price_quality_df": build_price_quality_table(price_df),
        "calculations_export_df": build_calculations_export_df({
            "portfolio_data": portfolio_data,
            "bench_daily_window": bench_daily,
            "bench_cum_window": bench_cum_window,
            "start_effective": start_effective,
            "end_effective": end_effective,
        }),
    }
    emit("Done.", 100)
    return results

