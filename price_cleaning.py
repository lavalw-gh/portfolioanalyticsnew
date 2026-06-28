from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd

from .constants import CASH_SYMBOL
from .yahoo_metadata import get_yahoo_currency_cached


def fix_ltm_unit_regime_errors(
    prices_gbp: pd.DataFrame,
    tickers: List[str],
    lookback_days: int = 252,
    min_history: int = 40,
    ratio_min: float = 20.0,
    ratio_max: float = 250.0,
    close_tolerance: float = 0.80,
    improvement_factor: float = 0.25,
) -> Tuple[pd.DataFrame, List[dict]]:
    fixed = prices_gbp.copy()
    report: List[dict] = []

    if fixed.empty:
        return fixed, report

    min_periods = max(20, min(min_history, int(lookback_days)))

    for ticker in tickers:
        if ticker not in fixed.columns:
            continue

        series = fixed[ticker].astype(float).copy()
        valid = series[(series.notna()) & (series > 0)]
        if len(valid) < min_history:
            fixed[ticker] = series
            continue

        rolling_ref = series.rolling(window=int(
            lookback_days), min_periods=min_periods).median()
        ltm_ref = float(valid.tail(int(lookback_days)).median()
                        ) if len(valid) else np.nan
        if not np.isfinite(ltm_ref) or ltm_ref <= 0:
            fixed[ticker] = series
            continue

        for i, dt in enumerate(series.index):
            cur_px = series.iat[i]
            if not pd.notna(cur_px) or cur_px <= 0:
                continue

            ref_px = rolling_ref.iat[i]
            if not pd.notna(ref_px) or ref_px <= 0:
                ref_px = ltm_ref
            if not np.isfinite(ref_px) or ref_px <= 0:
                continue

            raw_px = float(cur_px)
            raw_err = abs(math.log(raw_px / float(ref_px)))
            raw_ratio = max(raw_px / float(ref_px), float(ref_px) / raw_px)
            if raw_ratio < ratio_min:
                continue

            candidates = {
                'raw': raw_px,
                'x100': raw_px * 100.0,
                '/100': raw_px / 100.0,
            }

            candidate_scores = {}
            for label, px in candidates.items():
                if not np.isfinite(px) or px <= 0:
                    continue
                candidate_scores[label] = abs(
                    math.log(float(px) / float(ref_px)))

            best_label = min(candidate_scores, key=candidate_scores.get)
            best_err = float(candidate_scores[best_label])
            best_px = float(candidates[best_label])

            if best_label == 'raw':
                continue

            close_enough = best_err <= math.log(1.0 + close_tolerance)
            materially_better = best_err <= raw_err * improvement_factor
            ratio_like_unit_mix = ratio_min <= raw_ratio <= ratio_max

            if close_enough and materially_better and ratio_like_unit_mix:
                old_px = raw_px
                new_px = best_px
                series.iat[i] = new_px
                report.append(
                    {
                        'Ticker': ticker,
                        'Date': pd.Timestamp(dt).strftime('%Y-%m-%d'),
                        'Issue': 'Long-term median unit-regime correction',
                        'Old Price': old_px,
                        'New Price': new_px,
                        'Method': f'LTMMedian {best_label}',
                        'Reference': float(ref_px),
                        'Raw Ratio': float(raw_ratio),
                    }
                )

        fixed[ticker] = series

    return fixed, report


def fix_single_day_unit_mismatches(
    prices_gbp: pd.DataFrame,
    tickers: List[str],
    neighbor_tolerance: float = 0.15,
    ratio_min: float = 20.0,
) -> Tuple[pd.DataFrame, List[dict]]:
    fixed = prices_gbp.copy()
    report: List[dict] = []

    if fixed.empty:
        return fixed, report

    for ticker in tickers:
        if ticker not in fixed.columns:
            continue

        s = fixed[ticker].astype(float).copy()
        if s.dropna().shape[0] < 3:
            fixed[ticker] = s
            continue

        for i in range(1, len(s) - 1):
            prev_px = s.iat[i - 1]
            cur_px = s.iat[i]
            next_px = s.iat[i + 1]
            if not (pd.notna(prev_px) and pd.notna(cur_px) and pd.notna(next_px)):
                continue

            prev_px = float(prev_px)
            cur_px = float(cur_px)
            next_px = float(next_px)
            if prev_px <= 0 or cur_px <= 0 or next_px <= 0:
                continue

            neighbor_ratio = max(prev_px / next_px, next_px / prev_px)
            if neighbor_ratio > (1.0 + float(neighbor_tolerance)):
                continue

            ref_px = float(np.sqrt(prev_px * next_px))
            raw_ratio = max(cur_px / ref_px, ref_px / cur_px)
            if raw_ratio < float(ratio_min):
                continue

            new_px = ref_px
            s.iat[i] = new_px
            report.append(
                {
                    'Ticker': ticker,
                    'Date': pd.Timestamp(s.index[i]).strftime('%Y-%m-%d'),
                    'Issue': 'Single-day sandwich correction',
                    'Action': 'corrected_to_neighbor_ref',
                    'Reason': 'Prev/next agree but current is far from both',
                    'Old Price': cur_px,
                    'New Price': new_px,
                    'Method': 'NeighborSandwich direct-ref',
                    'Reference': ref_px,
                    'Raw Ratio': raw_ratio,
                    'Prev Price': prev_px,
                    'Next Price': next_px,
                    'Neighbor Ratio': neighbor_ratio,
                }
            )

        fixed[ticker] = s

    return fixed, report


def fix_gbp_unit_mix_extremes(
    prices_gbp: pd.DataFrame,
    tickers: List[str],
    factor: float = 100.0,
    obvious_ratio: float = 20.0,
    close_tolerance: float = 0.80,
    radius: int = 3,
) -> Tuple[pd.DataFrame, List[dict]]:
    def nearby_prices(series: pd.Series, pos: int, radius: int = 3) -> list[float]:
        vals: list[float] = []
        for step in range(1, radius + 1):
            left = pos - step
            right = pos + step
            if left >= 0:
                v = series.iat[left]
                if pd.notna(v) and v > 0:
                    vals.append(float(v))
            if right < len(series):
                v = series.iat[right]
                if pd.notna(v) and v > 0:
                    vals.append(float(v))
        return vals

    def move_score(prev_px, cur_px, next_px) -> float:
        comps = []
        for px in (prev_px, next_px):
            if pd.notna(px) and px > 0 and pd.notna(cur_px) and cur_px > 0:
                comps.append(abs(math.log(float(cur_px) / float(px))))
        return float(np.mean(comps)) if comps else float('inf')

    fixed = prices_gbp.copy()
    report: List[dict] = []
    if fixed.empty:
        return fixed, report

    for ticker in tickers:
        if ticker not in fixed.columns:
            continue

        ccy = get_yahoo_currency_cached(ticker)
        if ccy not in ('GBp', 'GBX'):
            continue

        s = fixed[ticker].astype(float).copy()
        valid = s[(s.notna()) & (s > 0)]
        if len(valid) < 3:
            fixed[ticker] = s
            continue

        global_ref = float(valid.median())

        for i, dt in enumerate(s.index):
            cur_px = s.iat[i]
            if not pd.notna(cur_px) or cur_px <= 0:
                continue

            neighbors = nearby_prices(s, i, radius=radius)
            if len(neighbors) >= 2:
                ref_px = float(np.median(neighbors))
                ref_method = 'NearbyMedian'
            else:
                ref_px = global_ref
                ref_method = 'GlobalMedian'

            if not np.isfinite(ref_px) or ref_px <= 0:
                continue

            ratio_gap = max(float(cur_px) / ref_px, ref_px / float(cur_px))
            if ratio_gap < obvious_ratio:
                continue

            prev_px = s.iat[i - 1] if i > 0 else np.nan
            next_px = s.iat[i + 1] if i < len(s) - 1 else np.nan
            current_err = abs(math.log(float(cur_px) / ref_px))
            current_score = move_score(prev_px, cur_px, next_px)

            best = None
            for label, candidate_px in (('x100', float(cur_px) * factor), ('/100', float(cur_px) / factor)):
                if not np.isfinite(candidate_px) or candidate_px <= 0:
                    continue
                candidate_err = abs(math.log(candidate_px / ref_px))
                candidate_score = move_score(prev_px, candidate_px, next_px)
                if (
                    best is None
                    or candidate_err < best['candidate_err']
                    or (
                        math.isclose(candidate_err, best['candidate_err'])
                        and candidate_score < best['candidate_score']
                    )
                ):
                    best = {
                        'label': label,
                        'candidate_px': candidate_px,
                        'candidate_err': candidate_err,
                        'candidate_score': candidate_score,
                    }

            if best is None:
                continue

            close_enough = best['candidate_err'] <= math.log(
                1.0 + close_tolerance)
            materially_better = best['candidate_err'] <= current_err * 0.35
            move_improves = (
                not np.isfinite(current_score)
                or best['candidate_score'] <= current_score * 0.55
            )

            if close_enough and materially_better and move_improves:
                old_px = float(cur_px)
                new_px = float(best['candidate_px'])
                s.iat[i] = new_px
                report.append(
                    {
                        'Ticker': ticker,
                        'Date': pd.Timestamp(dt).strftime('%Y-%m-%d'),
                        'Currency': ccy,
                        'Issue': 'GBp/GBP unit-mix extreme corrected',
                        'Old Price': old_px,
                        'New Price': new_px,
                        'Factor': factor,
                        'Method': f"{ref_method} {best['label']}",
                    }
                )

        fixed[ticker] = s

    return fixed, report


def apply_residual_move_filter(
    prices: pd.DataFrame,
    tickers: List[str],
    max_daily_move_pct: float = 25.0,
) -> Tuple[pd.DataFrame, List[dict]]:
    cleaned = prices.copy()
    corrections: List[dict] = []
    if cleaned.empty:
        return cleaned, corrections

    limit = max(0.0, float(max_daily_move_pct)) / 100.0
    if limit <= 0:
        return cleaned, corrections

    baseline_limit = min(limit / 3.0, 0.10)

    def is_valid(px) -> bool:
        return pd.notna(px) and float(px) > 0

    def pct_move(a, b) -> float:
        return abs(float(b) / float(a) - 1.0)

    def signed_move(a, b) -> float:
        return float(b) / float(a) - 1.0

    for symbol in tickers:
        if symbol not in cleaned.columns:
            continue

        s = cleaned[symbol].astype(float).copy()
        if s.dropna().shape[0] < 3:
            cleaned[symbol] = s
            continue

        candidate_map: dict[int, dict] = {}
        n = len(s)

        for i in range(1, n - 1):
            prev_px = s.iat[i - 1]
            cur_px = s.iat[i]
            next_px = s.iat[i + 1]
            if not (is_valid(prev_px) and is_valid(cur_px) and is_valid(next_px)):
                continue

            move_from_prev = signed_move(prev_px, cur_px)
            move_to_next = signed_move(cur_px, next_px)
            baseline_move = pct_move(prev_px, next_px)

            if abs(move_from_prev) <= limit or abs(move_to_next) <= limit:
                continue
            if baseline_move > baseline_limit:
                continue
            if move_from_prev * move_to_next >= 0:
                continue

            ref_px = float(np.sqrt(float(prev_px) * float(next_px)))
            dev_from_ref = pct_move(ref_px, cur_px)
            if dev_from_ref <= limit:
                continue

            candidate_map[i] = {
                'idx': i,
                'symbol': symbol,
                'date': s.index[i],
                'old_price': float(cur_px),
                'new_price': ref_px,
                'score': float(dev_from_ref),
                'threshold': float(limit),
                'window': 3.0,
                'method': 'Neighbor geometric mean',
                'issue': 'Residual one-day move outlier',
            }

        if n >= 3:
            cur_px = s.iat[0]
            next_px = s.iat[1]
            next2_px = s.iat[2]
            if is_valid(cur_px) and is_valid(next_px) and is_valid(next2_px):
                if (
                    pct_move(cur_px, next_px) > limit
                    and pct_move(cur_px, next2_px) > limit
                    and pct_move(next_px, next2_px) <= baseline_limit
                ):
                    candidate_map[0] = {
                        'idx': 0,
                        'symbol': symbol,
                        'date': s.index[0],
                        'old_price': float(cur_px),
                        'new_price': float(np.median([float(next_px), float(next2_px)])),
                        'score': float(min(pct_move(cur_px, next_px), pct_move(cur_px, next2_px))),
                        'threshold': float(limit),
                        'window': 3.0,
                        'method': 'Edge median',
                        'issue': 'Residual edge-day move outlier',
                    }

            prev2_px = s.iat[n - 3]
            prev_px = s.iat[n - 2]
            cur_px = s.iat[n - 1]
            if is_valid(cur_px) and is_valid(prev_px) and is_valid(prev2_px):
                if (
                    pct_move(prev_px, cur_px) > limit
                    and pct_move(prev2_px, cur_px) > limit
                    and pct_move(prev2_px, prev_px) <= baseline_limit
                ):
                    candidate_map[n - 1] = {
                        'idx': n - 1,
                        'symbol': symbol,
                        'date': s.index[n - 1],
                        'old_price': float(cur_px),
                        'new_price': float(np.median([float(prev_px), float(prev2_px)])),
                        'score': float(min(pct_move(prev_px, cur_px), pct_move(prev2_px, cur_px))),
                        'threshold': float(limit),
                        'window': 3.0,
                        'method': 'Edge median',
                        'issue': 'Residual edge-day move outlier',
                    }

        for idx in sorted(candidate_map):
            if idx - 1 in candidate_map or idx + 1 in candidate_map:
                continue
            item = candidate_map[idx]
            s.iat[idx] = float(item['new_price'])
            corrections.append(
                {
                    'symbol': item['symbol'],
                    'date': item['date'],
                    'old_price': item['old_price'],
                    'new_price': item['new_price'],
                    'score': item['score'],
                    'threshold': item['threshold'],
                    'window': item['window'],
                    'method': item['method'],
                    'issue': item['issue'],
                }
            )

        cleaned[symbol] = s

    return cleaned, corrections


def fix_short_term_residual_outliers(
    prices: pd.DataFrame,
    tickers: List[str],
    window: int = 5,
    ratio_min: float = 8.0,
    local_tolerance: float = 0.12,
    min_valid_neighbors: int = 4,
) -> Tuple[pd.DataFrame, List[dict]]:
    cleaned = prices.copy()
    corrections: List[dict] = []
    if cleaned.empty:
        return cleaned, corrections

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2

    for symbol in tickers:
        if symbol not in cleaned.columns:
            continue

        s = cleaned[symbol].astype(float).copy()
        if s.dropna().shape[0] < window:
            cleaned[symbol] = s
            continue

        n = len(s)
        for i in range(half, n - half):
            cur_px = s.iat[i]
            if not pd.notna(cur_px) or float(cur_px) <= 0:
                continue

            win = s.iloc[i - half:i + half + 1].astype(float)
            if win.isna().any():
                continue

            neighbor_vals = [float(v) for j, v in enumerate(
                win.values) if j != half and pd.notna(v) and float(v) > 0]
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
                'Issue': 'Short-term median residual outlier corrected',
                'Old Price': old_px,
                'New Price': new_px,
                'Method': f'STMMedian w={window}',
                'Reference': local_ref,
                'Raw Ratio': float(raw_ratio),
                'Local Spread': float(local_spread),
                'Window': float(window),
            })

        cleaned[symbol] = s

    return cleaned, corrections


def fix_robust_local_price_outliers(
    prices: pd.DataFrame,
    tickers: List[str],
    radius: int = 3,
    flag_ratio: float = 1.20,
    correction_ratio: float = 1.50,
    neighbor_tolerance: float = 0.10,
    min_valid_neighbors: int = 4,
    unit_ratio_min: float = 20.0,
    unit_ratio_max: float = 250.0,
) -> Tuple[pd.DataFrame, List[dict]]:
    cleaned = prices.copy()
    report: List[dict] = []
    if cleaned.empty:
        return cleaned, report

    radius = max(1, int(radius))
    min_valid_neighbors = max(2, int(min_valid_neighbors))

    def nearby_prices(series: pd.Series, pos: int) -> tuple[list[float], int, int]:
        vals: list[float] = []
        left_found = 0
        right_found = 0
        max_scan = max(radius * 5, radius + 2)
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
        return vals, left_found, right_found

    for symbol in tickers:
        if symbol not in cleaned.columns:
            continue

        s = cleaned[symbol].astype(float).copy()
        if s.dropna().shape[0] < min_valid_neighbors + 1:
            cleaned[symbol] = s
            continue

        for i in range(len(s)):
            cur_px = s.iat[i]
            if not pd.notna(cur_px) or float(cur_px) <= 0:
                continue

            neighbor_vals, left_count, right_count = nearby_prices(s, i)
            if len(neighbor_vals) < min_valid_neighbors:
                continue

            local_ref = float(np.median(neighbor_vals))
            if not np.isfinite(local_ref) or local_ref <= 0:
                continue

            local_spread = max(neighbor_vals) / min(neighbor_vals)
            if local_spread > (1.0 + float(neighbor_tolerance)):
                continue

            old_px = float(cur_px)
            raw_ratio = max(old_px / local_ref, local_ref / old_px)
            if raw_ratio < float(flag_ratio):
                continue

            candidates = {
                'raw': old_px,
                'x100': old_px * 100.0,
                '/100': old_px / 100.0,
            }
            candidate_scores = {
                label: abs(math.log(px / local_ref))
                for label, px in candidates.items()
                if np.isfinite(px) and px > 0
            }
            best_label = min(candidate_scores, key=candidate_scores.get)
            best_px = float(candidates[best_label])
            best_ratio = max(best_px / local_ref, local_ref / best_px)

            has_two_sided_context = left_count > 0 and right_count > 0
            unit_like = unit_ratio_min <= raw_ratio <= unit_ratio_max
            unit_fix = (
                best_label != 'raw'
                and unit_like
                and best_ratio <= (1.0 + float(neighbor_tolerance))
            )
            singleton_fix = (
                has_two_sided_context
                and raw_ratio >= float(correction_ratio)
            )

            if unit_fix:
                action = 'corrected'
                new_px = best_px
                issue = 'Robust local 100x unit correction'
                confidence = 'high'
                method = f'RobustLocalMedian {best_label}'
                s.iat[i] = new_px
            elif singleton_fix:
                action = 'corrected'
                new_px = local_ref
                issue = 'Robust local singleton outlier corrected'
                confidence = 'high'
                method = 'RobustLocalMedian'
                s.iat[i] = new_px
            else:
                action = 'flag_only'
                new_px = old_px
                issue = 'Robust local price anomaly flagged'
                confidence = 'medium'
                method = 'RobustLocalMedian flag-only'

            report.append({
                'Ticker': symbol,
                'Date': pd.Timestamp(s.index[i]).strftime('%Y-%m-%d'),
                'Issue': issue,
                'Action': action,
                'Confidence': confidence,
                'Old Price': old_px,
                'New Price': float(new_px),
                'Method': method,
                'Reference': local_ref,
                'Raw Ratio': float(raw_ratio),
                'Local Spread': float(local_spread),
                'Left Neighbors': float(left_count),
                'Right Neighbors': float(right_count),
            })

        cleaned[symbol] = s

    return cleaned, report


def validate_and_clean_prices(prices: pd.DataFrame, threshold: float = 0.20) -> Tuple[pd.DataFrame, List[dict]]:
    cleaned = prices.copy()
    corrections: List[dict] = []
    if cleaned.empty:
        return cleaned, corrections
    for sym in cleaned.columns:
        s = cleaned[sym].copy()
        if s.isna().all():
            continue
        pct = s.pct_change()
        pct_next = s.pct_change(-1)
        spikes = (pct.abs() > threshold) & (pct_next.abs() > threshold)
        for dt in s[spikes].index:
            idx = s.index.get_loc(dt)
            if idx <= 0 or idx >= len(s) - 1:
                continue
            prev_px = s.iloc[idx - 1]
            spike_px = s.iloc[idx]
            if pd.isna(prev_px) or pd.isna(spike_px) or prev_px == 0:
                continue
            cleaned.loc[dt, sym] = prev_px
            corrections.append({
                "symbol": sym,
                "date": dt,
                "pct_move": (float(spike_px) / float(prev_px)) - 1.0,
                "old_price": float(spike_px),
                "new_price": float(prev_px),
            })
    return cleaned, corrections

