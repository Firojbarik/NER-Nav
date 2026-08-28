from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.app.schemas.predict import (
    FeedSegment,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    RiskFeedResponse,
    SegmentRainfall,
)
from backend.app.services import risk_model
from backend.app.services.features import derive_features

router = APIRouter(prefix="/api/v1", tags=["predict", "feed"])

FEED_FILE = "data/predictions/demo_risk_feed.json"


def _resolve_default_tag() -> str:
    tags = risk_model.demo_authorised_tags()
    if not tags:
        raise HTTPException(status_code=503, detail="no authorised demo model configured")
    return tags[0]


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    summaries = []
    for tag in risk_model.demo_authorised_tags():
        try:
            bundle = risk_model.cached_bundle(tag)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        summaries.append(ModelInfo(
            model_version=bundle["tag"],
            status=bundle["status"],
            dataset_version=bundle.get("dataset_version"),
            feature_version=bundle.get("feature_version"),
            intended_use="Hackathon demonstration and offline research only",
            caveat=bundle.get("caveat"),
            not_for=bundle.get("not_for"),
        ))
    return summaries


@router.get("/feed", response_model=RiskFeedResponse)
def risk_feed() -> RiskFeedResponse:
    try:
        with open(FEED_FILE, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="risk feed unavailable") from exc

    segments = []
    for item in payload.get("segments", []):
        rainfall = item.get("rainfall", {})
        segments.append(FeedSegment(
            osm_id=item["osm_id"],
            ref=item.get("ref"),
            state=item.get("state"),
            district=item.get("district"),
            latitude=item["latitude"],
            longitude=item["longitude"],
            elevation_m=item.get("elevation_m"),
            slope_degrees=item.get("slope_degrees"),
            rainfall=SegmentRainfall(
                one_day=rainfall.get("1day", 0) or 0,
                three_day=rainfall.get("3day", 0) or 0,
                seven_day=rainfall.get("7day", 0) or 0,
                fourteen_day=rainfall.get("14day", 0) or 0,
                thirty_day=rainfall.get("30day", 0) or 0,
            ),
            disruption_probability=item["disruption_probability"],
            risk_level=item["risk_level"],
            operating_decision=item["operating_decision"],
            confidence_flags=item.get("confidence_flags", []),
        ))
    return RiskFeedResponse(
        generated_at=payload.get("generated_at", ""),
        anchor_date=payload.get("anchor_date", ""),
        model_version=payload.get("model_version", ""),
        model_status=payload.get("model_status", ""),
        operating_threshold=payload.get("operating_threshold"),
        risk_policy=payload.get("risk_policy", {}),
        caveat=payload.get("caveat"),
        segments=segments,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        bundle = risk_model.cached_bundle(_resolve_default_tag())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    prediction_timestamp = (
        request.prediction_timestamp or datetime.now(timezone.utc).isoformat())

    values = request.features.model_dump()
    try:
        prepared, flags = derive_features(values, prediction_timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ood = []
    for name, bounds in (bundle["feature_profile"] or {}).items():
        value = prepared.get(name)
        if value is None or not isinstance(value, (int, float)) or not (value == value):
            continue
        low, high = bounds.get("p01"), bounds.get("p99")
        if low is not None and high is not None and not low <= value <= high:
            ood.append(name)
    if ood:
        flags.append("OUT_OF_DISTRIBUTION")
    if not request.prediction_timestamp:
        flags.append("WEATHER_FRESHNESS_UNKNOWN")

    try:
        result = risk_model.predict(bundle, prepared)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    level = risk_model.risk_level(result["calibrated_probability"], bundle["risk_policy"])
    operating_decision = (
        "ALERT" if result["calibrated_probability"] >= bundle["threshold"] else "NO_ALERT")

    model = ModelInfo(
        model_version=bundle["tag"],
        status=bundle["status"],
        dataset_version=bundle.get("dataset_version"),
        feature_version=bundle.get("feature_version"),
        intended_use="Hackathon demonstration and offline research only",
        caveat=bundle.get("caveat"),
        not_for=bundle.get("not_for"),
    )

    return PredictResponse(
        road_segment_id=request.road_segment_id,
        ref=request.ref,
        prediction_timestamp=prediction_timestamp,
        disruption_probability=result["disruption_probability"],
        raw_probability=result["raw_probability"],
        calibrated_probability=result["calibrated_probability"],
        risk_level=level,
        operating_decision=operating_decision,
        operating_threshold=bundle["threshold"],
        top_contributing_factors=result["top_contributing_factors"],
        data_quality={
            "confidence_flags": sorted(set(flags)),
            "low_confidence": bool(flags),
            "ood_features": sorted(ood),
        },
        model=model,
    )