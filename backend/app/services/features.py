"""Feature derivation for the prediction API.

Mirrors the training-time derivation in ``scripts/predict_real_temporal.py``
so the API returns the same risk score as the CLI for the same inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features.seasonal import (
    days_into_monsoon,
    monsoon_active,
    month_feature,
    seasonal_cos,
    seasonal_sin,
)

MAX_RAINFALL_MM = 10_000.0

REQUIRED_RAINFALL = [
    "rainfall_1day", "rainfall_3day", "rainfall_7day",
    "rainfall_14day", "rainfall_30day",
]

# Derived features computed from base rainfall + terrain inputs.
DERIVED_FEATURES = [
    "terrain_known",
    "rain_intensity_1_vs_7",
    "rain_intensity_3_vs_14",
    "rain_concentration_3_in_7",
    "rain_trend_1_vs_3",
    "rain_cumul_ratio_7_vs_30",
    "slope_x_rain7",
    "slope_x_rain30",
    "elev_x_slope",
    "month",
    "seasonal_sin",
    "seasonal_cos",
    "monsoon_active",
    "days_into_monsoon",
]


def _coerce_numeric(value):
    try:
        return pd.to_numeric(value, errors="coerce")
    except (TypeError, ValueError):
        return np.nan


def derive_features(values: dict, prediction_timestamp: str | None = None) -> tuple[dict, list[str]]:
    """Validate base inputs and derive the full frozen feature set.

    Accepts base rainfall (5 windows) plus optional ``elevation_m`` and
    ``slope_degrees``.  Returns (derived_value_dict, flags).

    Raises ValueError on inputs that cannot produce a defensible prediction.
    """
    if not isinstance(values, dict):
        raise ValueError("features must be a JSON object")

    errors: list[str] = []
    prepared = dict(values)

    allowed = set(REQUIRED_RAINFALL) | {"elevation_m", "slope_degrees"} | set(DERIVED_FEATURES)
    unknown = sorted(set(prepared) - allowed)
    if unknown:
        errors.append(f"unknown input fields: {unknown}")

    for name in REQUIRED_RAINFALL:
        value = _coerce_numeric(prepared.get(name))
        if np.ndim(value) != 0 or not np.isfinite(value):
            errors.append(f"{name} is required and must be finite")
        elif value < 0:
            errors.append(f"{name} cannot be negative")
        elif value > MAX_RAINFALL_MM:
            errors.append(f"{name} exceeds the supported physical maximum of {MAX_RAINFALL_MM:g} mm")
        else:
            prepared[name] = float(value)

    for name in ("elevation_m", "slope_degrees"):
        raw = prepared.get(name)
        if raw is None:
            prepared[name] = None
            continue
        value = _coerce_numeric(raw)
        if np.ndim(value) != 0 or not np.isfinite(value):
            errors.append(f"{name} must be finite or null")
        else:
            prepared[name] = float(value)

    slope = prepared.get("slope_degrees")
    elevation = prepared.get("elevation_m")
    if slope is not None and not 0 <= slope <= 90:
        errors.append("slope_degrees must be between 0 and 90")
    if elevation is not None and not -500 <= elevation <= 9000:
        errors.append("elevation_m is outside the supported physical range")
    if errors:
        raise ValueError("; ".join(errors))

    eps = 0.01
    r1, r3, r7 = prepared["rainfall_1day"], prepared["rainfall_3day"], prepared["rainfall_7day"]
    r14, r30 = prepared["rainfall_14day"], prepared["rainfall_30day"]
    slope_value = slope or 0.0
    elevation_value = elevation or 0.0

    prepared.update({
        "rain_intensity_1_vs_7": r1 / max(r7, eps),
        "rain_intensity_3_vs_14": r3 / max(r14, eps),
        "rain_concentration_3_in_7": r3 / max(r7, eps),
        "rain_trend_1_vs_3": r1 / max(r3, eps),
        "rain_cumul_ratio_7_vs_30": r7 / max(r30, eps),
        "terrain_known": int(slope is not None and elevation is not None),
        "slope_x_rain7": slope_value * r7,
        "slope_x_rain30": slope_value * r30,
        "elev_x_slope": elevation_value * slope_value,
    })

    flags: list[str] = []
    if prepared["terrain_known"] == 0:
        flags.append("MISSING_TERRAIN")

    if prediction_timestamp:
        anchor_date = pd.Timestamp(prediction_timestamp).date()
        prepared.update({
            "month": float(month_feature(anchor_date)),
            "seasonal_sin": seasonal_sin(anchor_date),
            "seasonal_cos": seasonal_cos(anchor_date),
            "monsoon_active": float(monsoon_active(anchor_date)),
            "days_into_monsoon": float(days_into_monsoon(anchor_date)),
        })

    return prepared, flags