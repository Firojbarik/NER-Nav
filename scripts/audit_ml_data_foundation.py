
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = DATA_DIR / "processed" / "ml"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_REPORT = OUTPUT_DIR / "data_foundation_audit.json"
MD_REPORT = OUTPUT_DIR / "data_foundation_audit.md"


SUPPORTED_TABLE_EXTENSIONS = {
    ".csv",
    ".parquet",
    ".json",
    ".geojson",
    ".gpkg",
}

KEYWORDS = {
    "hazard": [
        "hazard",
        "risk",
        "vulnerability",
        "susceptibility",
    ],
    "incident": [
        "incident",
        "event",
        "disruption",
        "landslide",
        "flood",
        "collapse",
        "damage",
        "road_block",
    ],
    "weather": [
        "rain",
        "rainfall",
        "weather",
        "temperature",
        "humidity",
        "wind",
        "precip",
    ],
    "terrain": [
        "slope",
        "elevation",
        "terrain",
        "dem",
        "aspect",
        "geology",
    ],
    "traffic": [
        "traffic",
        "congestion",
        "speed",
        "travel_time",
        "delay",
    ],
}


def classify_path(path: Path) -> list[str]:
    text = path.name.lower()

    categories = []

    for category, words in KEYWORDS.items():
        if any(word in text for word in words):
            categories.append(category)

    return categories


def safe_json_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


def inspect_parquet(path: Path) -> dict:
    result = {
        "format": "parquet",
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
    }

    try:
        df = pd.read_parquet(path)

        result.update(
            {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": [str(c) for c in df.columns],
                "dtypes": {
                    str(c): str(dtype)
                    for c, dtype in df.dtypes.items()
                },
                "missing_values": {
                    str(c): int(v)
                    for c, v in df.isna().sum().items()
                    if int(v) > 0
                },
            }
        )

        if "risk_class" in df.columns:
            result["risk_class_distribution"] = {
                str(k): int(v)
                for k, v in df["risk_class"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            }

        if "label_source" in df.columns:
            result["label_sources"] = [
                safe_json_value(x)
                for x in df["label_source"]
                .dropna()
                .unique()
                .tolist()
            ]

    except Exception as exc:
        result["error"] = repr(exc)

    return result


def inspect_csv(path: Path) -> dict:
    result = {
        "format": "csv",
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
    }

    try:
        df = pd.read_csv(path, nrows=1000)

        result.update(
            {
                "sample_rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": [str(c) for c in df.columns],
                "dtypes": {
                    str(c): str(dtype)
                    for c, dtype in df.dtypes.items()
                },
            }
        )

    except Exception as exc:
        result["error"] = repr(exc)

    return result


def inspect_json(path: Path) -> dict:
    result = {
        "format": path.suffix.lower().lstrip("."),
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
    }

    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        result["top_level_type"] = type(obj).__name__

        if isinstance(obj, dict):
            result["top_level_keys"] = list(obj.keys())[:100]

        elif isinstance(obj, list):
            result["list_length"] = len(obj)

            if obj and isinstance(obj[0], dict):
                result["first_record_keys"] = list(obj[0].keys())

    except Exception as exc:
        result["error"] = repr(exc)

    return result


def inspect_gpkg(path: Path) -> dict:
    result = {
        "format": "gpkg",
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
    }

    try:
        import geopandas as gpd

        layers = gpd.list_layers(path)

        result["layers"] = []

        for _, layer_name, geometry_type in layers.itertuples():
            layer_info = {
                "name": str(layer_name),
                "geometry_type": str(geometry_type),
            }

            try:
                gdf = gpd.read_file(path, layer=str(layer_name))

                layer_info.update(
                    {
                        "rows": int(len(gdf)),
                        "columns": int(len(gdf.columns)),
                        "column_names": [
                            str(c) for c in gdf.columns
                        ],
                        "crs": str(gdf.crs),
                    }
                )

                layer_info["geometry_valid"] = int(
                    gdf.geometry.is_valid.sum()
                )

                layer_info["geometry_empty"] = int(
                    gdf.geometry.is_empty.sum()
                )

            except Exception as exc:
                layer_info["error"] = repr(exc)

            result["layers"].append(layer_info)

    except Exception as exc:
        result["error"] = repr(exc)

    return result


def inspect_file(path: Path) -> dict:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return inspect_parquet(path)

    if suffix == ".csv":
        return inspect_csv(path)

    if suffix in {".json", ".geojson"}:
        return inspect_json(path)

    if suffix == ".gpkg":
        return inspect_gpkg(path)

    return {
        "format": suffix.lstrip("."),
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
    }


def build_source_registry(files: list[Path]) -> list[dict]:
    registry = []

    for path in files:
        categories = classify_path(path)

        registry.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "categories": categories,
                "potential_real_hazard_source": bool(
                    set(categories)
                    & {"hazard", "incident", "weather", "terrain"}
                ),
            }
        )

    return registry


def make_gap_analysis(file_reports: list[dict]) -> dict:
    all_text = json.dumps(file_reports).lower()

    return {
        "road_network": "road" in all_text
        or "roads" in all_text,
        "real_incident_data": any(
            word in all_text
            for word in [
                "incident",
                "landslide",
                "flood",
                "disruption",
            ]
        ),
        "rainfall_data": any(
            word in all_text
            for word in [
                "rain",
                "rainfall",
                "precipitation",
            ]
        ),
        "terrain_data": any(
            word in all_text
            for word in [
                "slope",
                "elevation",
                "terrain",
                "dem",
            ]
        ),
        "geological_vulnerability": any(
            word in all_text
            for word in [
                "geological",
                "vulnerability",
                "susceptibility",
            ]
        ),
        "traffic_data": any(
            word in all_text
            for word in [
                "traffic",
                "congestion",
                "travel_time",
            ]
        ),
        "delay_target": "delay_minutes" in all_text,
    }


def write_markdown(report: dict) -> None:
    lines = []

    lines.append("# NER-Nav Data Foundation Audit")
    lines.append("")
    lines.append(
        f"Generated: {report['generated_at']}"
    )
    lines.append("")

    lines.append("## Local data inventory")
    lines.append("")

    for item in report["files"]:
        lines.append(
            f"- `{item['path']}`"
        )

        if item.get("categories"):
            lines.append(
                f"  - categories: "
                f"{', '.join(item['categories'])}"
            )

    lines.append("")

    lines.append("## Dataset inspection")
    lines.append("")

    for item in report["inspections"]:
        lines.append(
            f"### `{item['path']}`"
        )

        if "rows" in item:
            lines.append(
                f"- Rows: {item['rows']:,}"
            )

        if "columns" in item:
            lines.append(
                f"- Columns: {item['columns']:,}"
            )

        if "column_names" in item:
            lines.append(
                "- Columns: "
                + ", ".join(item["column_names"])
            )

        if "error" in item:
            lines.append(
                f"- ERROR: `{item['error']}`"
            )

        lines.append("")

    lines.append("## Required ML data gap analysis")
    lines.append("")

    for key, value in report["gap_analysis"].items():
        status = "FOUND" if value else "MISSING"
        lines.append(
            f"- **{key}: {status}**"
        )

    lines.append("")

    lines.append("## Required real-data sources")
    lines.append("")
    lines.append(
        "- NRSC/ISRO Landslide Atlas: historical landslide "
        "inventory."
    )
    lines.append(
        "- NDEM/NRSC: historical flood, landslide and "
        "geo-hazard products."
    )
    lines.append(
        "- GSI: landslide susceptibility and historical "
        "landslide information."
    )
    lines.append(
        "- IMD/OpenWeather: rainfall/weather features."
    )
    lines.append(
        "- DEM/SRTM/Bhuvan: elevation and slope features."
    )
    lines.append("")

    lines.append("## Important modeling rule")
    lines.append("")
    lines.append(
        "The current synthetic risk_class dataset must remain "
        "a pipeline baseline. It must not be reported as "
        "real-world disruption prediction performance."
    )
    lines.append("")

    MD_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 70)
    print("NER-Nav — Data Foundation Audit")
    print("=" * 70)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}"
        )

    files = [
        p
        for p in DATA_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_TABLE_EXTENSIONS
    ]

    files.sort()

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data files:   {len(files):,}")

    registry = build_source_registry(files)

    print()
    print("Potential hazard/event sources:")
    print("-" * 70)

    for item in registry:
        if item["potential_real_hazard_source"]:
            print(
                f"  {item['path']}"
            )

    print()
    print("Inspecting datasets...")

    inspections = []

    for index, path in enumerate(files, start=1):
        print(
            f"  [{index}/{len(files)}] "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

        inspections.append(
            inspect_file(path)
        )

    gap_analysis = make_gap_analysis(inspections)

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": "NER-Nav",
        "purpose": (
            "Audit current data foundation before real "
            "hazard/disruption model construction."
        ),
        "files": registry,
        "inspections": inspections,
        "gap_analysis": gap_analysis,
        "current_model_warning": (
            "risk_class is synthetic/proxy and must not be "
            "treated as real incident ground truth."
        ),
        "external_source_registry": [
            {
                "source": "NRSC/ISRO Landslide Atlas",
                "purpose": (
                    "historical landslide inventory and "
                    "event/route-wise landslide information"
                ),
                "period": "1998-2022",
            },
            {
                "source": "NDEM/NRSC",
                "purpose": (
                    "flood, landslide, geo-hazard and "
                    "disaster information"
                ),
            },
            {
                "source": "GSI",
                "purpose": (
                    "landslide susceptibility and historical "
                    "landslide information"
                ),
            },
            {
                "source": "IMD/OpenWeather",
                "purpose": (
                    "rainfall and weather predictor variables"
                ),
            },
            {
                "source": "DEM/SRTM/Bhuvan",
                "purpose": (
                    "elevation and slope features"
                ),
            },
        ],
    }

    with JSON_REPORT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_markdown(report)

    print()
    print("=" * 70)
    print("DATA FOUNDATION AUDIT COMPLETE")
    print("=" * 70)

    print()
    print("Required data status:")

    for key, value in gap_analysis.items():
        status = "FOUND" if value else "MISSING"
        print(
            f"  {key:30s}: {status}"
        )

    print()
    print(f"JSON: {JSON_REPORT}")
    print(f"Report: {MD_REPORT}")
    print()
    print(
        "Next gate: acquire/prepare real hazard-event data "
        "and spatially/temporally map it to road segments."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
