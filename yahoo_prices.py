from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf

from .constants import CASH_SYMBOL
from .price_cleaning import (
    apply_residual_move_filter,
    fix_gbp_unit_mix_extremes,
    fix_ltm_unit_regime_errors,
    fix_robust_local_price_outliers,
    fix_short_term_residual_outliers,
    fix_single_day_unit_mismatches,
)
from .yahoo_metadata import get_yahoo_currency_cached


_GLOBAL_SYMBOL_PRICE_CACHE: dict[str, pd.Series] = {}
_GLOBAL_SYMBOL_RANGE_CACHE: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
_GLOBAL_CLEAN_PRICE_HISTORY_CACHE: dict[tuple, pd.DataFrame] = {}

MAX_FORWARD_FILL_DAYS = 5


def clear_price_caches() -> None:
    _GLOBAL_SYMBOL_PRICE_CACHE.clear()
    _GLOBAL_SYMBOL_RANGE_CACHE.clear()
    _GLOBAL_CLEAN_PRICE_HISTORY_CACHE.clear()


def _clone_price_df_with_attrs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.attrs = copy.deepcopy(getattr(df, "attrs", {}))
    return out


def _merge_price_series(old: pd.Series | None, new: pd.Series | None, name: str) -> pd.Series:
    if old is None or old.empty:
        return pd.Series(dtype=float, name=name) if new is None else new.sort_index().rename(name)
    if new is None or new.empty:
        return old.sort_index().rename(name)
    merged = pd.concat([old.rename(name), new.rename(name)]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.rename(name)


def _has_usable_prices(series: pd.Series | None) -> bool:
    return series is not None and not series.empty and not series.dropna().empty


def _extract_close_from_yf_download(symbol: str, data: pd.DataFrame) -> tuple[pd.Series, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if data.empty:
        issues.append({"symbol": symbol, "problem": "No data returned"})
        return pd.Series(dtype=float, name=symbol), issues
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"]
        else:
            issues.append({"symbol": symbol, "problem": "Neither 'Close' nor 'Adj Close' found"})
            return pd.Series(dtype=float, name=symbol), issues
    else:
        if "Close" in data.columns:
            close = data["Close"]
        elif "Adj Close" in data.columns:
            close = data["Adj Close"]
        else:
            issues.append({"symbol": symbol, "problem": "Neither 'Close' nor 'Adj Close' found"})
            return pd.Series(dtype=float, name=symbol), issues
    if isinstance(close, pd.DataFrame):
        series = close[symbol] if symbol in close.columns else close.iloc[:, 0]
    else:
        series = close
    series = pd.to_numeric(series, errors="coerce").rename(symbol)
    if series.isna().all():
        issues.append({"symbol": symbol, "problem": "All prices are NaN (Yahoo returned no usable data)"})
    return series, issues


def _download_single_symbol_prices(symbol: str, start_date, end_date, auto_adjust=True, progress=False) -> tuple[str, pd.Series, list[dict[str, str]]]:
    try:
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=auto_adjust, progress=progress, threads=False)
        series, issues = _extract_close_from_yf_download(symbol, data)
        return symbol, series, issues
    except Exception as e:
        return symbol, pd.Series(dtype=float, name=symbol), [{"symbol": symbol, "problem": f"Download error: {e}"}]


def fetch_yahoo_prices(symbols, start_date, end_date, auto_adjust=True, progress=False, progress_callback: Callable[[str, int], None] | None = None):
    requested = list(dict.fromkeys(symbols if isinstance(symbols, list) else [symbols]))
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if not requested:
        return pd.DataFrame(), []

    issues: list[dict[str, str]] = []
    series_map: dict[str, pd.Series] = {}
    to_download: list[str] = []

    for symbol in requested:
        cached = _GLOBAL_SYMBOL_PRICE_CACHE.get(symbol)
        cached_range = _GLOBAL_SYMBOL_RANGE_CACHE.get(symbol)
        if cached is not None and cached_range is not None and cached_range[0] <= start_ts and cached_range[1] >= end_ts:
            series_map[symbol] = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)].copy()
        else:
            to_download.append(symbol)

    if to_download:
        max_workers = min(8, max(1, len(to_download)))
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_single_symbol_prices, symbol, start_ts, end_ts, auto_adjust, progress): symbol
                for symbol in to_download
            }
            for future in as_completed(futures):
                symbol = futures[future]
                _, series, symbol_issues = future.result()
                issues.extend(symbol_issues)

                if _has_usable_prices(series):
                    existing = _GLOBAL_SYMBOL_PRICE_CACHE.get(symbol)
                    merged = _merge_price_series(existing, series, symbol)
                    _GLOBAL_SYMBOL_PRICE_CACHE[symbol] = merged

                    prev_range = _GLOBAL_SYMBOL_RANGE_CACHE.get(symbol)
                    if prev_range is None:
                        _GLOBAL_SYMBOL_RANGE_CACHE[symbol] = (start_ts, end_ts)
                    else:
                        _GLOBAL_SYMBOL_RANGE_CACHE[symbol] = (min(prev_range[0], start_ts), max(prev_range[1], end_ts))

                    series_map[symbol] = merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()
                else:
                    issues.append({"symbol": symbol, "problem": "Download not cached because it contained no usable prices"})
                    series_map[symbol] = series
                completed += 1
                if progress_callback is not None:
                    pct = 10 + int(25 * completed / max(1, len(to_download)))
                    progress_callback(f"Downloading prices... ({completed}/{len(to_download)})", pct)

    frames = []
    for symbol in requested:
        series = series_map.get(symbol, pd.Series(dtype=float, name=symbol))
        frames.append(series.rename(symbol))

    close = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    close = close.reindex(columns=requested)

    if close.empty:
        issues.append({"symbol": ",".join(requested), "problem": "No data returned for any symbol"})
        return close, issues

    for s in requested:
        if s not in close.columns:
            close[s] = np.nan
            issues.append({"symbol": s, "problem": "Symbol missing from Yahoo download"})
        elif close[s].isna().all():
            issues.append({"symbol": s, "problem": "All prices are NaN (Yahoo returned no usable data)"})
    return close, issues


def build_missing_price_report(prices: pd.DataFrame, symbols: list[str]) -> list[dict[str, str | int]]:
    report: list[dict[str, str | int]] = []
    if prices.empty:
        return report

    first_index = prices.index.min()
    last_index = prices.index.max()
    for symbol in symbols:
        if symbol not in prices.columns:
            continue
        series = prices[symbol]
        if series.isna().all():
            report.append({
                'symbol': symbol,
                'type': 'nodata',
                'start': first_index.strftime('%d/%m/%Y'),
                'end': last_index.strftime('%d/%m/%Y'),
                'missing_days': int(len(series)),
            })
            continue

        missing = series.isna()
        if not missing.any():
            continue

        block_start = None
        prev_dt = None
        for dt, is_missing in missing.items():
            if bool(is_missing) and block_start is None:
                block_start = dt
            if not bool(is_missing) and block_start is not None:
                block_end = prev_dt
                if block_end is not None:
                    block_mask = (series.index >= block_start) & (series.index <= block_end)
                    gap_type = 'leadingnan' if block_start == first_index else 'internalnan'
                    report.append({
                        'symbol': symbol,
                        'type': gap_type,
                        'start': pd.Timestamp(block_start).strftime('%d/%m/%Y'),
                        'end': pd.Timestamp(block_end).strftime('%d/%m/%Y'),
                        'missing_days': int(block_mask.sum()),
                    })
                block_start = None
            prev_dt = dt

        if block_start is not None and prev_dt is not None:
            block_mask = (series.index >= block_start) & (series.index <= prev_dt)
            gap_type = 'leadingnan' if block_start == first_index else 'trailingnan'
            report.append({
                'symbol': symbol,
                'type': gap_type,
                'start': pd.Timestamp(block_start).strftime('%d/%m/%Y'),
                'end': pd.Timestamp(prev_dt).strftime('%d/%m/%Y'),
                'missing_days': int(block_mask.sum()),
            })

    return report


def forward_fill_prices_with_audit(
    prices: pd.DataFrame,
    symbols: list[str],
    max_fill_days: int = MAX_FORWARD_FILL_DAYS,
) -> tuple[pd.DataFrame, list[dict[str, str | int | float]]]:
    filled = prices.copy()
    report: list[dict[str, str | int | float]] = []
    if filled.empty:
        return filled, report

    max_fill_days = max(0, int(max_fill_days))
    for symbol in symbols:
        if symbol not in filled.columns or symbol == CASH_SYMBOL:
            continue

        original = filled[symbol].copy()
        if original.isna().all():
            continue

        unlimited = original.ffill()
        bounded = original.ffill(limit=max_fill_days) if max_fill_days > 0 else original.copy()
        stale_mask = original.isna() & unlimited.notna()
        beyond_limit_mask = stale_mask & bounded.isna()

        if stale_mask.any():
            block_start = None
            prev_dt = None
            for dt, is_stale in stale_mask.items():
                if bool(is_stale) and block_start is None:
                    block_start = dt
                if not bool(is_stale) and block_start is not None:
                    block_end = prev_dt
                    block_mask = (stale_mask.index >= block_start) & (stale_mask.index <= block_end)
                    carried_days = int(stale_mask.loc[block_mask].sum())
                    exceeded = bool(beyond_limit_mask.loc[block_mask].any())
                    prior_valid = original.loc[:block_start].dropna()
                    report.append({
                        'symbol': symbol,
                        'type': 'forward_fill',
                        'start': pd.Timestamp(block_start).strftime('%d/%m/%Y'),
                        'end': pd.Timestamp(block_end).strftime('%d/%m/%Y'),
                        'carried_days': carried_days,
                        'max_allowed_days': max_fill_days,
                        'exceeded_limit': exceeded,
                        'fill_value': float(prior_valid.iloc[-1]) if not prior_valid.empty else np.nan,
                    })
                    block_start = None
                prev_dt = dt

            if block_start is not None and prev_dt is not None:
                block_mask = (stale_mask.index >= block_start) & (stale_mask.index <= prev_dt)
                carried_days = int(stale_mask.loc[block_mask].sum())
                exceeded = bool(beyond_limit_mask.loc[block_mask].any())
                prior_valid = original.loc[:block_start].dropna()
                report.append({
                    'symbol': symbol,
                    'type': 'forward_fill',
                    'start': pd.Timestamp(block_start).strftime('%d/%m/%Y'),
                    'end': pd.Timestamp(prev_dt).strftime('%d/%m/%Y'),
                    'carried_days': carried_days,
                    'max_allowed_days': max_fill_days,
                    'exceeded_limit': exceeded,
                    'fill_value': float(prior_valid.iloc[-1]) if not prior_valid.empty else np.nan,
                })

        filled[symbol] = bounded

    return filled, report


def enforce_reported_price_corrections(
    prices: pd.DataFrame,
    reports: List[tuple[str, list[dict]]],
) -> pd.DataFrame:
    fixed = prices.copy()
    if fixed.empty:
        return fixed

    for report_type, rows in reports:
        if not rows:
            continue
        for item in rows:
            if item.get('Action') == 'flag_only':
                continue
            if report_type in ('ltm', 'gbp', 'singleton', 'stm_level', 'stm_residual'):
                ticker = item.get('Ticker', '')
                dt = item.get('Date')
                new_px = item.get('New Price', np.nan)
            elif report_type == 'robust_local':
                ticker = item.get('Ticker', '')
                dt = item.get('Date')
                new_px = item.get('New Price', np.nan)
            elif report_type == 'residual':
                ticker = item.get('symbol', '')
                dt = item.get('date')
                new_px = item.get('new_price', np.nan)
            else:
                continue

            if not ticker or ticker not in fixed.columns:
                continue
            if dt is None or pd.isna(new_px):
                continue

            dt = pd.Timestamp(dt).normalize()
            if dt in fixed.index:
                fixed.at[dt, ticker] = float(new_px)

    return fixed


def fix_short_term_level_outliers(
    prices: pd.DataFrame,
    tickers: List[str],
    radius: int = 3,
    ratio_min: float = 1.20,
    local_tolerance: float = 0.08,
    min_valid_neighbors: int = 3,
) -> Tuple[pd.DataFrame, List[dict]]:
    def nearby_prices(series: pd.Series, pos: int, radius: int = 3) -> list[float]:
        vals: list[float] = []
        left_found = 0
        right_found = 0
        max_scan = radius * 4
        for step in range(1, max_scan + 1):
            if left_found < radius:
                left = pos - step
                if left >= 0:
                    v = series.iat[left]
                    if pd.notna(v) and float(v) > 0:
                        vals.append(float(v))
                        left_found += 1
            if right_found < radius:
                right = pos + step
                if right < len(series):
                    v = series.iat[right]
                    if pd.notna(v) and float(v) > 0:
                        vals.append(float(v))
                        right_found += 1
            if left_found >= radius and right_found >= radius:
                break
        return vals

    cleaned = prices.copy()
    corrections: List[dict] = []
    if cleaned.empty:
        return cleaned, corrections

    radius = max(1, int(radius))

    for symbol in tickers:
        if symbol not in cleaned.columns:
            continue

        s = cleaned[symbol].astype(float).copy()
        if s.dropna().shape[0] < (int(min_valid_neighbors) + 1):
            cleaned[symbol] = s
            continue

        for i in range(len(s)):
            cur_px = s.iat[i]
            if not pd.notna(cur_px) or float(cur_px) <= 0:
                continue

            neighbor_vals = nearby_prices(s, i, radius=radius)
            if len(neighbor_vals) < int(min_valid_neighbors):
                continue

            local_ref = float(np.median(neighbor_vals))
            if not np.isfinite(local_ref) or local_ref <= 0:
                continue

            local_spread = max(neighbor_vals) / min(neighbor_vals)
            if local_spread > (1.0 + float(local_tolerance)):
                continue

            raw_ratio = max(float(cur_px) / local_ref,
                            local_ref / float(cur_px))
            if raw_ratio < float(ratio_min):
                continue

            old_px = float(cur_px)
            new_px = local_ref
            s.iat[i] = new_px
            corrections.append({
                'Ticker': symbol,
                'Date': pd.Timestamp(s.index[i]).strftime('%Y-%m-%d'),
                'Issue': 'Short-term local level outlier corrected',
                'Old Price': old_px,
                'New Price': new_px,
                'Method': f'STMLevelMedian r={radius}',
                'Reference': local_ref,
                'Raw Ratio': float(raw_ratio),
                'Local Spread': float(local_spread),
                'Window': float(radius * 2 + 1),
            })

        cleaned[symbol] = s

    return cleaned, corrections


def get_price_history(tickers, benchmark, start_date, end_date, conversion_factors, hampel_threshold: float = 25.0, progress_callback: Callable[[str, int], None] | None = None) -> pd.DataFrame:
    all_symbols = list(dict.fromkeys(list(tickers) + [benchmark]))
    download_symbols = [s for s in all_symbols if s != CASH_SYMBOL]
    start_download = pd.Timestamp(start_date) - pd.Timedelta(days=400)
    end_ts = pd.Timestamp(end_date)

    conversion_key = tuple(sorted((s, float(conversion_factors.get(s, 1.0))) for s in all_symbols))
    cache_key = (tuple(all_symbols), start_download.normalize(), end_ts.normalize(), conversion_key, round(float(hampel_threshold), 6))
    cached_df = _GLOBAL_CLEAN_PRICE_HISTORY_CACHE.get(cache_key)
    if cached_df is not None:
        if progress_callback is not None:
            progress_callback("Reusing cached cleaned prices...", 55)
        return _clone_price_df_with_attrs(cached_df)

    if progress_callback is not None:
        progress_callback("Downloading prices...", 5)

    close, issues = fetch_yahoo_prices(download_symbols, start_download, end_ts, auto_adjust=True, progress=False, progress_callback=progress_callback)
    if close.empty:
        raise RuntimeError('Yahoo Finance returned no price data for any symbol.')

    if CASH_SYMBOL in all_symbols and CASH_SYMBOL not in close.columns:
        close[CASH_SYMBOL] = 1.0

    for s in all_symbols:
        if s not in close.columns:
            close[s] = np.nan
            issues.append({'symbol': s, 'problem': 'Column added as all-NaN because Yahoo omitted symbol'})

    close = close[all_symbols]
    raw_download_close = close.copy()

    if progress_callback is not None:
        progress_callback("Normalizing currencies...", 38)

    for ticker in tickers:
        if ticker in close.columns and ticker in conversion_factors:
            factor = float(conversion_factors.get(ticker, 1.0))
            if factor != 1.0:
                close[ticker] = close[ticker] / factor

    if benchmark in conversion_factors and benchmark in close.columns:
        factor = float(conversion_factors.get(benchmark, 1.0))
        if factor != 1.0:
            close[benchmark] = close[benchmark] / factor

    close = close.dropna(how='all').sort_index()
    post_initial_normalization = close.copy()

    missing_ranges = build_missing_price_report(close.copy(), all_symbols)

    if progress_callback is not None:
        progress_callback("Cleaning prices...", 48)

    ltm_unit_report: list[dict] = []
    unit_mix_tickers: list[str] = []
    for t in all_symbols:
        if t not in close.columns or t == CASH_SYMBOL:
            continue
        ccy = get_yahoo_currency_cached(t)
        factor = float(conversion_factors.get(t, 1.0))
        if factor == 100.0 or ccy in ('GBp', 'GBX'):
            unit_mix_tickers.append(t)

    stage_frames: dict[str, pd.DataFrame] = {
        'raw_download_close': raw_download_close.copy(),
        'post_initial_normalization': post_initial_normalization.copy(),
        'post_ltm': close.copy(),
        'post_gbp_unit_mix': close.copy(),
        'post_singleton': close.copy(),
        'post_residual': close.copy(),
        'post_stm_residual': close.copy(),
        'post_robust_local': close.copy(),
        'final_forward_filled': close.copy(),
    }

    if unit_mix_tickers:
        close, ltm_unit_report = fix_ltm_unit_regime_errors(close, unit_mix_tickers)
        close = enforce_reported_price_corrections(close, [('ltm', ltm_unit_report)])
        stage_frames['post_ltm'] = close.copy()

    unit_mix_report: list[dict] = []
    if unit_mix_tickers:
        close, unit_mix_report = fix_gbp_unit_mix_extremes(close, unit_mix_tickers)
        close = enforce_reported_price_corrections(close, [('ltm', ltm_unit_report), ('gbp', unit_mix_report)])
        stage_frames['post_gbp_unit_mix'] = close.copy()

    singleton_unit_report: list[dict] = []
    if unit_mix_tickers:
        close, singleton_unit_report = fix_single_day_unit_mismatches(close, unit_mix_tickers)
        close = enforce_reported_price_corrections(close, [('ltm', ltm_unit_report), ('gbp', unit_mix_report), ('singleton', singleton_unit_report)])
        stage_frames['post_singleton'] = close.copy()

    residual_corrections: list[dict] = []
    holdings_tickers = [t for t in tickers if t != CASH_SYMBOL and t in close.columns]
    if holdings_tickers:
        close, residual_corrections = apply_residual_move_filter(close, holdings_tickers, max_daily_move_pct=float(hampel_threshold))
        close = enforce_reported_price_corrections(close, [('ltm', ltm_unit_report), ('gbp', unit_mix_report), ('singleton', singleton_unit_report), ('residual', residual_corrections)])
        stage_frames['post_residual'] = close.copy()

    stm_residual_report: list[dict] = []
    if holdings_tickers:
        close, stm_residual_report = fix_short_term_residual_outliers(close, holdings_tickers, window=5)
        close = enforce_reported_price_corrections(close, [('ltm', ltm_unit_report), ('gbp', unit_mix_report), ('singleton', singleton_unit_report), ('residual', residual_corrections), ('stm_residual', stm_residual_report)])
        stage_frames['post_stm_residual'] = close.copy()

    robust_local_report: list[dict] = []
    if holdings_tickers:
        close, robust_local_report = fix_robust_local_price_outliers(
            close,
            holdings_tickers,
            radius=3,
            flag_ratio=1.20,
            correction_ratio=1.50,
            neighbor_tolerance=0.10,
            min_valid_neighbors=4,
        )
        close = enforce_reported_price_corrections(close, [('ltm', ltm_unit_report), ('gbp', unit_mix_report), ('singleton', singleton_unit_report), ('residual', residual_corrections), ('stm_residual', stm_residual_report), ('robust_local', robust_local_report)])
        stage_frames['post_robust_local'] = close.copy()

    close = close[all_symbols]
    close, forward_fill_report = forward_fill_prices_with_audit(close, all_symbols, max_fill_days=MAX_FORWARD_FILL_DAYS)
    stage_frames['final_used_in_calculations'] = close.copy()
    stage_frames['final_forward_filled'] = close.copy()

    close.attrs['yahoo_issues'] = issues
    close.attrs['missing_ranges'] = missing_ranges
    close.attrs['ltm_unit_report'] = ltm_unit_report
    close.attrs['gbp_unit_mix_report'] = unit_mix_report
    close.attrs['singleton_unit_report'] = singleton_unit_report
    close.attrs['residual_corrections'] = residual_corrections
    close.attrs['stm_residual_report'] = stm_residual_report
    close.attrs['robust_local_report'] = robust_local_report
    close.attrs['forward_fill_report'] = forward_fill_report
    close.attrs['max_forward_fill_days'] = MAX_FORWARD_FILL_DAYS
    close.attrs['spike_corrections'] = []
    close.attrs['price_stage_frames'] = {k: v.reindex(index=close.index, columns=all_symbols) for k, v in stage_frames.items()}

    _GLOBAL_CLEAN_PRICE_HISTORY_CACHE[cache_key] = _clone_price_df_with_attrs(close)
    if progress_callback is not None:
        progress_callback("Prices ready.", 62)
    return close

