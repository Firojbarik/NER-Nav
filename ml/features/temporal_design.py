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
from typing import Dict, List, Tuple

import pandas as pd

# (event_id, osm_id, nh_ref, event_date) - all real confirmed positives.
CONFIRMED_EVENTS: List[Tuple[str, int, str, str]] = [
    ("nagaland_viswema_2017_07_15", 751071339, "NH2", "2017-07-15"),
    ("manipur_nungdolan_nh37_2021_06_13", 242395881, "NH37", "2021-06-13"),
    ("meghalaya_sonapur_2022_09_06", 666960671, "NH6", "2022-09-06"),
    ("sikkim_nh10_singtam_2023_10_04", 83700092, "NH510", "2023-10-04"),
    ("assam_harangajao_2024_05_28", 1053899051, "NH27", "2024-05-28"),
    ("mizoram_hunthar_2024_05_28", 1058852502, "NH6", "2024-05-28"),
    ("manipur_irang_2024_05_29", 44884963, "NH37", "2024-05-29"),
    ("meghalaya_kuliang_2024_06_18", 743133270, "NH6", "2024-06-18"),
    ("arunachal_babuk_2024_07_04", 22832880, "NH13", "2024-07-04"),
    ("sikkim_nh10_sisney_2024_07_12", 606061034, "NH10", "2024-07-12"),
    ("arunachal_palizi_2024_07_16", 238496657, "NH13", "2024-07-16"),
    ("nagaland_dzuza_2024_08_18", 1353575336, "NH29", "2024-08-18"),
    ("nagaland_dzudza_2024_08_20", 621402573, "NH29", "2024-08-20"),
    ("nagaland_pherima_2024_09_03", 1038173685, "NH29", "2024-09-03"),
    ("nagaland_pherima_2024_09_04", 384669799, "NH29", "2024-09-04"),
    ("manipur_nungdalal_2025_03_15", 44884960, "NH37", "2025-03-15"),
    ("arunachal_bana_seppa_2025_06_01", 238561752, "NH13", "2025-06-01"),
    ("arunachal_tawang_2025_06_01", 238560159, "NH13", "2025-06-01"),
    ("sikkim_chungthang_2025_06_01", 47416074, "NH10", "2025-06-01"),
    ("nagaland_kisama_nh2_2025_06_01", 666588410, "NH2", "2025-06-01"),
    ("manipur_sinzawl_nh102b_2025_06_02", 242645601, "NH102B", "2025-06-02"),
    ("assam_jatinga_2025_06_24", 239060064, "NH27", "2025-06-24"),
    ("noney_nhb37_landslide_2025_07_16", 44884968, "NH37", "2025-07-16"),
    ("assam_dima_hasao_2025_07_16", 386397330, "NH27", "2025-07-16"),
    ("tripura_nh208_kailashahar_2025_09_12", 138303255, "NH208A", "2025-09-12"),
    ("assam_lumding_2025_09_14", 311653434, "NH27", "2025-09-14"),
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


def needed_prediction_dates() -> List[date]:
    """Sorted, unique prediction (anchor) dates across all samples."""
    s = set()
    for times in all_sample_times_for_events().values():
        s.update(times)
    return sorted(s)


def needed_chirps_dates() -> List[date]:
    """Sorted, unique calendar dates required for 30-day rainfall lookback."""
    need = set()
    for times in all_sample_times_for_events().values():
        for pt in times:
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
