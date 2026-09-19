"""Real temporal risk-sample design for NER-Nav disruption prediction.

PROVENANCE / REAL-DATA RULE
---------------------------
Every value in this module derives from REAL source-supported data:
  * `CONFIRMED_EVENTS` contains the currently reviewed hazard events
    (event_date from each event JSON; osm_id from its *_confirmed_road.parquet
     or from OSM Overpass query for the affected NH road segment).
  * Prediction times are REAL calendar dates anchored to each event date.
  * Labels are assigned by the temporal rule in `ml/labels/generate.py`:
    event labels a sample iff  prediction_time < event_time <= prediction_time + horizon.
No synthetic values are introduced anywhere in this module or its outputs.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Source-backed unaffected-road observations drive real negative anchors.
NEGATIVE_OBSERVATIONS_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "hazards" / "negative_observations.csv"
)

# (event_id, osm_id, nh_ref, event_date) - all real confirmed positives.
CONFIRMED_EVENTS: List[Tuple[str, int, str, str]] = [
    ("nagaland_kiruphema_nh29_2017_07_10", 1353575335, "NH29", "2017-07-10"),
    ("nagaland_viswema_2017_07_15", 751071339, "NH2", "2017-07-15"),
    ("manipur_nungdolan_nh37_2021_06_13", 242395881, "NH37", "2021-06-13"),
    ("manipur_saihum_nh102b_2022_07_08", 242645609, "NH102B", "2022-07-08"),
    ("meghalaya_sonapur_2022_09_06", 666960671, "NH6", "2022-09-06"),
    ("meghalaya_jowai_nh6_2023_07_01", 743647547, "NH6", "2023-07-01"),
    ("sikkim_gyalshing_legship_nh510_2023_08_21", 47417242, "NH510", "2023-08-21"),
    ("sikkim_ravangla_legship_nh510_2023_08_25", 47417252, "NH510", "2023-08-25"),
    ("sikkim_nh10_singtam_2023_10_04", 83700092, "NH510", "2023-10-04"),
    ("arunachal_sela_pass_nh13_2024_02_03", 1260641175, "NH13", "2024-02-03"),
    ("assam_harangajao_2024_05_28", 1053899051, "NH27", "2024-05-28"),
    ("mizoram_hunthar_2024_05_28", 1058852502, "NH6", "2024-05-28"),
    ("nagaland_viswema_bridge_nh2_2024_05_28", 44886105, "NH2", "2024-05-28"),
    ("manipur_irang_2024_05_29", 44884963, "NH37", "2024-05-29"),
    ("sikkim_shantinagar_nh10_2024_06_11", 213887652, "NH10", "2024-06-11"),
    ("meghalaya_kuliang_2024_06_18", 743133270, "NH6", "2024-06-18"),
    ("nagaland_leriechazou_nh2_2024_06_29", 666588410, "NH2", "2024-06-29"),
    ("arunachal_babuk_2024_07_04", 22832880, "NH13", "2024-07-04"),
    ("sikkim_nh10_sisney_2024_07_12", 606061034, "NH10", "2024-07-12"),
    ("arunachal_palizi_2024_07_16", 238496657, "NH13", "2024-07-16"),
    ("nagaland_dzuza_2024_08_18", 1353575336, "NH29", "2024-08-18"),
    ("nagaland_dzudza_2024_08_20", 621402573, "NH29", "2024-08-20"),
    ("nagaland_pherima_2024_09_03", 1038173685, "NH29", "2024-09-03"),
    ("nagaland_pherima_2024_09_04", 384669799, "NH29", "2024-09-04"),
    ("sikkim_rangrang_nh10_2024_09_27", 349554354, "NH10", "2024-09-27"),
    ("manipur_nungdalal_2025_03_15", 44884960, "NH37", "2025-03-15"),
    ("arunachal_bana_seppa_2025_06_01", 238561752, "NH13", "2025-06-01"),
    ("arunachal_tawang_2025_06_01", 238560159, "NH13", "2025-06-01"),
    ("sikkim_chungthang_2025_06_01", 47416074, "NH10", "2025-06-01"),
    ("nagaland_kisama_nh2_2025_06_01", 666588410, "NH2", "2025-06-01"),
    ("tripura_kadamtala_kurti_nh208a_2025_06_01", 978243002, "NH208A", "2025-06-01"),
    ("manipur_sinzawl_nh102b_2025_06_02", 242645601, "NH102B", "2025-06-02"),
    ("assam_jatinga_2025_06_24", 239060064, "NH27", "2025-06-24"),
    ("nagaland_phesama_nh2_2025_07_02", 666588410, "NH2", "2025-07-02"),
    ("manipur_ukhrul_nh102a_2025_07_13", 1237518330, "NH102A", "2025-07-13"),
    ("noney_nhb37_landslide_2025_07_16", 44884968, "NH37", "2025-07-16"),
    ("assam_dima_hasao_2025_07_16", 386397330, "NH27", "2025-07-16"),
    ("manipur_chiangpi_nh102b_2025_07_29", 242646804, "NH102B", "2025-07-29"),
    ("sikkim_bardang_nh10_2025_07_29", 606061034, "NH10", "2025-07-29"),
    ("nagaland_tsiesema_nh2_2025_08_25", 44886105, "NH2", "2025-08-25"),
    ("sikkim_bardang_nh10_2025_09_09", 879401691, "NH10", "2025-09-09"),
    ("tripura_nh208_kailashahar_2025_09_12", 138303255, "NH208A", "2025-09-12"),
    ("assam_lumding_2025_09_14", 311653434, "NH27", "2025-09-14"),
    ("nagaland_pagala_nh29_2025_09_15", 1353399486, "NH29", "2025-09-15"),
    ("sikkim_nagdhara_nh510_2025_09_17", 134421795, "NH510", "2025-09-17"),
    ("tripura_erapar_nh208_2026_06_10", 384497056, "NH208", "2026-06-10"),
    ("meghalaya_shillong_dawki_nh206_2026_06_21", 662679409, "NH206", "2026-06-21"),
    ("assam_jatinga_nh27_2026_06_21", 239060064, "NH27", "2026-06-21"),
    ("manipur_tamenglong_nh37_2026_06_24", 613798216, "NH37", "2026-06-24"),
    ("sikkim_gyalshing_legship_nh510_2026_06_24", 44848726, "NH510", "2026-06-24"),
    ("arunachal_rottung_nh13_2026_07_02", 961086598, "NH13", "2026-07-02"),
    ("sikkim_bardang_nh10_2026_07_07", 879401691, "NH10", "2026-07-07"),
    ("meghalaya_jorabat_nh6_2026_07_08", 977975338, "NH6", "2026-07-08"),
    ("manipur_vaorei_nh102a_2026_07_12", 1237518330, "NH102A", "2026-07-12"),
    ("arunachal_pakro_nh13_2026_07_13", 459331169, "NH13", "2026-07-13"),
    ("sikkim_bardang_nh10_2026_07_14", 879401691, "NH10", "2026-07-14"),
    ("nagaland_tuli_nh2_2026_07_20", 237490856, "NH2", "2026-07-20"),
    ("manipur_noney_awangkhul_nh37_2026_07_20", 44884968, "NH37", "2026-07-20"),
]

# Prediction task: will a disruption affect this road within `horizon` days?
HORIZON_DAYS = 7
# Prediction offsets (days BEFORE the event) that are within the 7-day horizon:
# event at pt+offset where offset in {1,3,7} satisfies pt < event <= pt+7  -> label 1.
POSITIVE_OFFSETS = [1, 3, 7]
# Same-road near-miss control: event at +14 days is AFTER the 7-day horizon -> label 0.
NEGATIVE_SAME_ROAD_OFFSETS = [14]
# Number of unaffected same-NH-corridor trunk roads sampled as negatives per event.
NEGATIVES_PER_REF = 4
# Prediction offset used for the corridor-negative samples.
NEGATIVE_CORRIDOR_OFFSET = 3
# Rainfall lookback (calendar days strictly before prediction_time) used for features.
LOOKBACK_DAYS = 30


def event_date(eid: str) -> date:
    for ev_id, _, _, ed in CONFIRMED_EVENTS:
        if ev_id == eid:
            return date.fromisoformat(ed)
    raise ValueError(f"unknown event: {eid}")


def event_rows() -> List[Tuple[str, int, str, date]]:
    return [(e, o, r, date.fromisoformat(ed)) for e, o, r, ed in CONFIRMED_EVENTS]


def positive_sample_times(eid: str) -> List[date]:
    E = event_date(eid)
    return [E - timedelta(days=off) for off in POSITIVE_OFFSETS]


def all_sample_times_for_events() -> Dict[str, List[date]]:
    """sample_times per event: positives, same-road negatives, corridor negatives."""
    out: Dict[str, List[date]] = {}
    for e, o, r, E in event_rows():
        times = set(positive_sample_times(e))
        for off in NEGATIVE_SAME_ROAD_OFFSETS:
            times.add(E - timedelta(days=off))
        times.add(E - timedelta(days=NEGATIVE_CORRIDOR_OFFSET))
        out[e] = sorted(times)
    return out


def negative_observation_prediction_dates() -> List[date]:
    """Sorted, unique prediction-time anchors from real source-backed negatives.

    Each negative_observations.csv row is a genuinely observed unaffected
    segment; its prediction_time must be covered by the rainfall feature
    extraction exactly like the event-derived anchors are.
    """
    if not NEGATIVE_OBSERVATIONS_FILE.exists():
        return []
    frame = pd.read_csv(NEGATIVE_OBSERVATIONS_FILE, usecols=["prediction_time"])
    parsed = pd.to_datetime(frame["prediction_time"], errors="coerce").dropna()
    return sorted({pd.Timestamp(v).date() for v in parsed})


def needed_prediction_dates() -> List[date]:
    """Sorted, unique prediction (anchor) dates across all samples.

    Includes both the event-derived anchors and the observation-backed
    negative anchors so the temporal rainfall parquet covers every sample
    time the production dataset merges against.
    """
    s = set()
    for times in all_sample_times_for_events().values():
        s.update(times)
    s.update(negative_observation_prediction_dates())
    return sorted(s)


def needed_chirps_dates() -> List[date]:
    """Sorted, unique calendar dates required for 30-day rainfall lookback."""
    need = set()
    for pt in needed_prediction_dates():
        for k in range(1, LOOKBACK_DAYS + 1):
            need.add(pt - timedelta(days=k))
    return sorted(need)


def build_labeled_samples() -> pd.DataFrame:
    """Real labeled (osm_id, prediction_time, label) sample frame.

    Positive samples: the confirmed affected road at each positive offset.
    Same-road negatives: the confirmed road at far offsets (event outside horizon).
    Corridor negatives are NOT generated here (need the road network to sample
    unaffected same-NH roads); `build_real_temporal_dataset.py` adds them.

    Output columns: event_id, osm_id, ref, prediction_time, label, sample_kind,
    horizon_days.
    """
    rows = []
    for e, osm, ref, E in event_rows():
        for off in POSITIVE_OFFSETS:
            rows.append({
                "event_id": e, "osm_id": osm, "ref": ref,
                "prediction_time": pd.Timestamp(E - timedelta(days=off)),
                "label": 1, "sample_kind": "positive", "horizon_days": HORIZON_DAYS,
            })
        for off in NEGATIVE_SAME_ROAD_OFFSETS:
            rows.append({
                "event_id": e, "osm_id": osm, "ref": ref,
                "prediction_time": pd.Timestamp(E - timedelta(days=off)),
                "label": 0, "sample_kind": "same_road_control",
                "horizon_days": HORIZON_DAYS,
            })
    return pd.DataFrame(rows)
