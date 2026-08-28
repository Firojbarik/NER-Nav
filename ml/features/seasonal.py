"""Leakage-safe seasonal / monsoon features derived from a prediction timestamp.

These features depend ONLY on the calendar date at which a prediction is made.
Because the prediction timestamp is known exactly at decision time (never in
the future), none of these features can leak the target or future information.
They let the model express that road-disruption risk in the North Eastern
region is strongly seasonal (concentrated in the Indian Southwest Monsoon).

Semantics / conventions
-----------------------
* All functions take a ``datetime.date`` (or ``pandas.Timestamp``) and return
  deterministic values; there is no I/O, so the logic is unit-testable with
  synthetic dates.
* The Southwest Monsoon window for Northeast India is Jun 1 - Sep 30 inclusive.
* ``pre_monsoon`` is Mar 1 - May 31; ``post_monsoon`` is Oct 1 - Dec 31;
  ``winter`` is Jan 1 - Feb 28/29.

The set of recommended model features (all numeric):
  * ``month``             integer 1-12  (direct calendar month)
  * ``seasonal_sin``      sin(2*pi*day_of_year/365)  cyclical position in [-1,1]
  * ``seasonal_cos``      cos(2*pi*day_of_year/365)  cyclical position in [-1,1]
  * ``monsoon_active``    1 if inside the Jun 1 - Sep 30 monsoon window else 0
  * ``pre_monsoon``       1 if inside Mar 1 - May 31 else 0
  * ``days_into_monsoon`` int days since Jun 1; negative before the monsoon,
                          0 at Jun 1, and grows through the monsoon window.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional

MONSOON_START = date(2000, 6, 1)   # year is a placeholder; only month/day used
MONSOON_END = date(2000, 9, 30)
PRE_MONSOON_START = date(2000, 3, 1)
PRE_MONSOON_END = date(2000, 5, 31)
WINTER_END = date(2000, 2, 28)


def _anchor(d: date) -> date:
    """Normalise a date to a reference year 2000 so month/day comparisons work."""
    if isinstance(d, date):
        return date(2000, d.month, d.day)
    raise TypeError(f"expected a date, got {type(d).__name__}")


def is_monsoon(d: date) -> bool:
    """True if ``d`` falls inside the Jun 1 - Sep 30 Southwest Monsoon window."""
    a = _anchor(d)
    return MONSOON_START <= a <= MONSOON_END


def season(d: date) -> str:
    """Return one of 'winter' | 'pre_monsoon' | 'monsoon' | 'post_monsoon'."""
    a = _anchor(d)
    if MONSOON_START <= a <= MONSOON_END:
        return "monsoon"
    if PRE_MONSOON_START <= a <= PRE_MONSOON_END:
        return "pre_monsoon"
    if a >= date(2000, 10, 1):
        return "post_monsoon"
    return "winter"


def month_feature(d: date) -> int:
    """Calendar month (1-12)."""
    return d.month


def day_of_year(d: date) -> int:
    """Day of the Gregorian year (1-366)."""
    return d.timetuple().tm_yday


def seasonal_sin(d: date) -> float:
    """Sin of the cyclical year position in [-1, 1]."""
    return math.sin(2.0 * math.pi * (day_of_year(d) - 1) / 365.0)


def seasonal_cos(d: date) -> float:
    """Cos of the cyclical year position in [-1, 1]."""
    return math.cos(2.0 * math.pi * (day_of_year(d) - 1) / 365.0)


def monsoon_active(d: date) -> int:
    """1 if inside the Jun 1 - Sep 30 monsoon window else 0."""
    return int(is_monsoon(d))


def pre_monsoon_active(d: date) -> int:
    """1 if inside the Mar 1 - May 31 pre-monsoon window else 0."""
    a = _anchor(d)
    return int(PRE_MONSOON_START <= a <= PRE_MONSOON_END)


def days_into_monsoon(d: date) -> int:
    """Days since Jun 1 (negative before the monsoon, grows through it)."""
    a = _anchor(d)
    return (a - MONSOON_START).days


_SEASON_ORDER = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]


def season_bucket(d: date) -> int:
    """Ordinal season index 1-4 (winter=1, pre_monsoon=2, monsoon=3, post=4)."""
    return _SEASON_ORDER.index(season(d)) + 1


def recommended_features(d: date) -> Dict[str, float]:
    """Return the recommended numeric seasonal feature vector for a date."""
    return {
        "month": float(month_feature(d)),
        "seasonal_sin": seasonal_sin(d),
        "seasonal_cos": seasonal_cos(d),
        "monsoon_active": float(monsoon_active(d)),
        "days_into_monsoon": float(days_into_monsoon(d)),
    }


# Ordered list of the recommended seasonal feature names (for dataset/training).
SEASONAL_FEATURES: List[str] = [
    "month",
    "seasonal_sin",
    "seasonal_cos",
    "monsoon_active",
    "days_into_monsoon",
]
