from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def build_holdings(df_tx: pd.DataFrame, price_df: pd.DataFrame, asof: str = "bod") -> pd.DataFrame:
    days = price_df.index
    tx_pivot = df_tx.pivot_table(
        index="date", columns="ticker", values="shares", aggfunc="sum")
    union_index = tx_pivot.index.union(days).sort_values()
    tx_pivot_union = tx_pivot.reindex(union_index).fillna(0.0)
    if asof == "bod":
        holdings_union = tx_pivot_union.cumsum().shift(1, fill_value=0.0)
    elif asof == "eod":
        holdings_union = tx_pivot_union.cumsum()
    else:
        raise ValueError("asof must be 'bod' or 'eod'")
    holdings = holdings_union.reindex(days).ffill().fillna(0.0)
    holdings = holdings.reindex(columns=price_df.columns, fill_value=0.0)
    return holdings


def compute_drawdown(cum_returns: pd.Series):
    wealth = 1.0 + cum_returns
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    max_dd = dd.min() if not dd.empty and not dd.isna().all() else np.nan
    return max_dd, dd


def compute_twr_from_prices(asset_prices: pd.DataFrame, holdings_bod: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    values_bod = holdings_bod * asset_prices
    port_value_bod = values_bod.sum(axis=1)
    asset_rets = asset_prices.pct_change()
    weights_lag = values_bod.shift(1).div(port_value_bod.shift(1), axis=0)
    twr_daily = (weights_lag * asset_rets).sum(axis=1)
    valid_mask = port_value_bod.shift(1) > 0
    twr_daily = twr_daily.where(valid_mask, 0.0).fillna(0.0)
    twr_cum = (1.0 + twr_daily).cumprod() - 1.0
    return twr_daily, twr_cum


def build_holdings_for_portfolios(portfolio_tx_map: dict[str, pd.DataFrame], price_df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if not portfolio_tx_map:
        return {}, {}

    portfolios = list(portfolio_tx_map.keys())
    price_days = pd.Index(price_df.index, name="date")
    all_price_cols = list(price_df.columns)

    tx_frames = []
    for pname in portfolios:
        tx = portfolio_tx_map[pname]
        if tx.empty:
            continue
        tmp = tx[["date", "ticker", "shares"]].copy()
        tmp["portfolio"] = pname
        tx_frames.append(tmp[["portfolio", "date", "ticker", "shares"]])

    if tx_frames:
        tx_all = pd.concat(tx_frames, ignore_index=True)
        tx_pivot = tx_all.pivot_table(index=["portfolio", "date"], columns="ticker", values="shares", aggfunc="sum").sort_index()
        union_dates = pd.Index(tx_all["date"].drop_duplicates()).union(price_days).sort_values()
    else:
        tx_pivot = pd.DataFrame()
        union_dates = price_days

    full_index = pd.MultiIndex.from_product([portfolios, union_dates], names=["portfolio", "date"])
    tx_full = tx_pivot.reindex(full_index, fill_value=0.0)
    tx_full = tx_full.reindex(columns=all_price_cols, fill_value=0.0)

    holdings_eod_full = tx_full.groupby(level=0).cumsum()
    holdings_bod_full = holdings_eod_full.groupby(level=0).shift(1, fill_value=0.0)

    price_index = pd.MultiIndex.from_product([portfolios, price_days], names=["portfolio", "date"])
    holdings_eod_full = holdings_eod_full.reindex(price_index).groupby(level=0).ffill().fillna(0.0)
    holdings_bod_full = holdings_bod_full.reindex(price_index).groupby(level=0).ffill().fillna(0.0)

    holdings_bod = {p: holdings_bod_full.xs(p, level="portfolio").copy() for p in portfolios}
    holdings_eod = {p: holdings_eod_full.xs(p, level="portfolio").copy() for p in portfolios}
    return holdings_bod, holdings_eod


def compute_twr_for_portfolios(asset_prices: pd.DataFrame, holdings_bod_map: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    if not holdings_bod_map:
        return {}, {}

    portfolios = list(holdings_bod_map.keys())
    stacked_holdings = pd.concat(holdings_bod_map, names=["portfolio", "date"]).sort_index()
    stacked_prices = pd.concat({p: asset_prices for p in portfolios}, names=["portfolio", "date"]).sort_index()

    values_bod = stacked_holdings * stacked_prices
    port_value_bod = values_bod.sum(axis=1)
    asset_rets = stacked_prices.groupby(level=0).pct_change()
    lag_values = values_bod.groupby(level=0).shift(1)
    lag_port_values = port_value_bod.groupby(level=0).shift(1)
    weights_lag = lag_values.div(lag_port_values, axis=0)
    twr_daily_all = (weights_lag * asset_rets).sum(axis=1)
    valid_mask = lag_port_values > 0
    twr_daily_all = twr_daily_all.where(valid_mask, 0.0).fillna(0.0)
    twr_cum_all = (1.0 + twr_daily_all).groupby(level=0).cumprod() - 1.0

    twr_daily_map = {p: twr_daily_all.xs(p, level="portfolio").copy() for p in portfolios}
    twr_cum_map = {p: twr_cum_all.xs(p, level="portfolio").copy() for p in portfolios}
    return twr_daily_map, twr_cum_map


def compute_period_twr(twr_daily_all: pd.Series, start_effective: date, end_effective: date) -> pd.Series:
    twr_period = twr_daily_all[(twr_daily_all.index >= pd.Timestamp(start_effective)) & (twr_daily_all.index <= pd.Timestamp(end_effective))].copy()
    if not twr_period.empty:
        twr_period.iloc[0] = 0.0
    return twr_period


def compute_risk_metrics(twr_daily: pd.Series, rf_daily):
    if twr_daily.empty:
        return {
            "total_return": np.nan,
            "vol_annual": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "downside_dev": np.nan,
            "max_drawdown": np.nan,
        }
    if isinstance(rf_daily, pd.Series):
        rf_aligned = rf_daily.reindex(twr_daily.index).ffill().fillna(0.0)
    else:
        rf_aligned = pd.Series(float(rf_daily), index=twr_daily.index)
    twr_cum = (1.0 + twr_daily).cumprod() - 1.0
    total_return = twr_cum.iloc[-1] if not twr_cum.empty else np.nan
    vol_daily = twr_daily.std(ddof=1)
    vol_annual = vol_daily * np.sqrt(252) if np.isfinite(vol_daily) else np.nan
    excess = twr_daily - rf_aligned
    mean_excess_daily = excess.mean()
    sharpe_daily = mean_excess_daily / vol_daily if vol_daily and np.isfinite(vol_daily) and vol_daily > 0 else np.nan
    sharpe = sharpe_daily * np.sqrt(252) if np.isfinite(sharpe_daily) else np.nan
    downside = twr_daily[twr_daily < 0]
    downside_dev_daily = downside.std(ddof=1) if len(downside) > 0 else np.nan
    downside_dev_annual = downside_dev_daily * np.sqrt(252) if np.isfinite(downside_dev_daily) else np.nan
    mean_return_annual = twr_daily.mean() * 252
    rf_mean_annual = rf_aligned.mean() * 252
    sortino = (mean_return_annual - rf_mean_annual) / downside_dev_annual if downside_dev_annual and np.isfinite(downside_dev_annual) and downside_dev_annual > 0 else np.nan
    max_dd, _ = compute_drawdown(twr_cum)
    return {
        "total_return": total_return,
        "vol_annual": vol_annual,
        "sharpe": sharpe,
        "sortino": sortino,
        "downside_dev": downside_dev_annual,
        "max_drawdown": max_dd,
    }

