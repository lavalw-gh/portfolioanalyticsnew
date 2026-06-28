from __future__ import annotations

from datetime import date, timedelta


def compute_preset_dates(preset: str, today: date) -> tuple[date, date]:
    if preset == "1 month":
        return today - timedelta(days=31), today
    if preset == "3 months":
        return today - timedelta(days=92), today
    if preset == "6 months":
        return today - timedelta(days=183), today
    if preset == "1 year":
        return today - timedelta(days=365), today
    if preset == "3 years":
        return today - timedelta(days=365 * 3), today
    if preset == "5 years":
        return today - timedelta(days=365 * 5), today
    if preset == "Max":
        return today - timedelta(days=365 * 30), today
    return today - timedelta(days=92), today


def date_to_qdate(d: date):
    raise RuntimeError("Legacy desktop date conversion is no longer used.")


def qdate_to_date(qd) -> date:
    return date(qd.year(), qd.month(), qd.day())

