from __future__ import annotations

from pydantic import BaseModel, Field


class BaseFeatures(BaseModel):
    rainfall_1day: float = Field(ge=0, description="Precipitation in the last 1 day (mm)")
    rainfall_3day: float = Field(ge=0, description="Precipitation in the last 3 days (mm)")
    rainfall_7day: float = Field(ge=0, description="Precipitation in the last 7 days (mm)")
    rainfall_14day: float = Field(ge=0, description="Precipitation in the last 14 days (mm)")
    rainfall_30day: float = Field(ge=0, description="Precipitation in the last 30 days (mm)")
    elevation_m: float | None = Field(default=None, description="Elevation in metres")
    slope_degrees: float | None = Field(default=None, description="Terrain slope in degrees")


class PredictRequest(BaseModel):
    road_segment_id: str = Field(
        description="Stable road segment identifier (OSM way id or internal)")
    ref: str | None = Field(default=None, description="National Highway reference, e.g. NH13")
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    features: BaseFeatures
    prediction_timestamp: str | None = Field(
        default=None,
        description="ISO-8601 prediction timestamp; defaults to now (UTC). Must not be in the future.")


class ModelInfo(BaseModel):
    model_version: str
    status: str
    dataset_version: str | None = None
    feature_version: str | None = None
    intended_use: str | None = None
    caveat: str | None = None
    not_for: str | None = None


class PredictResponse(BaseModel):
    road_segment_id: str
    ref: str | None = None
    prediction_timestamp: str
    prediction_horizon_days: int = 7
    disruption_probability: float
    raw_probability: float
    calibrated_probability: float
    risk_level: str
    operating_decision: str
    operating_threshold: float
    top_contributing_factors: list[dict]
    data_quality: dict
    model: ModelInfo


class SegmentRainfall(BaseModel):
    one_day: float
    three_day: float
    seven_day: float
    fourteen_day: float
    thirty_day: float


class FeedSegment(BaseModel):
    osm_id: int
    ref: str | None = None
    state: str | None = None
    district: str | None = None
    latitude: float
    longitude: float
    elevation_m: float | None = None
    slope_degrees: float | None = None
    rainfall: SegmentRainfall
    disruption_probability: float
    risk_level: str
    operating_decision: str
    confidence_flags: list[str] = []


class RiskFeedResponse(BaseModel):
    generated_at: str
    anchor_date: str
    model_version: str
    model_status: str
    operating_threshold: float
    risk_policy: dict
    caveat: str | None = None
    segments: list[FeedSegment]


class ApiRouteSummary(BaseModel):
    status: str
    message: str


class RouteRequest(BaseModel):
    origin: list[float] = Field(description="[longitude, latitude]")
    destination: list[float] = Field(description="[longitude, latitude]")
    avoid_high_risk: bool = Field(default=True)