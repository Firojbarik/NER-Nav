from __future__ import annotations

import pickle
from pathlib import Path


GRAPH_FILE = Path("data/processed/graph/ner_road_graph.pkl")


def main() -> None:
    print("=" * 70)
    print("NER-Nav — Road Graph Attribute QA")
    print("=" * 70)

    print()
    print("Loading graph...")

    with GRAPH_FILE.open("rb") as f:
        G = pickle.load(f)

    print(f"Graph type: {type(G).__name__}")
    print(f"Nodes:      {G.number_of_nodes():,}")
    print(f"Edges:      {G.number_of_edges():,}")

    required = [
        "osm_id",
        "highway",
        "district",
        "state",
        "base_distance_km",
        "risk_multiplier",
        "blocked",
        "direction",
    ]

    missing = {key: 0 for key in required}
    bad_distance = 0

    direction_counts = {}
    oneway_counts = {}
    blocked_counts = {}
    highway_counts = {}

    edge_count = 0

    print()
    print("Checking edge attributes...")

    for _, _, _, data in G.edges(keys=True, data=True):
        edge_count += 1

        for key in required:
            if data.get(key) is None:
                missing[key] += 1

        distance = data.get("base_distance_km")

        if (
            not isinstance(distance, (int, float))
            or distance <= 0
        ):
            bad_distance += 1

        direction = data.get("direction")
        direction_counts[direction] = (
            direction_counts.get(direction, 0) + 1
        )

        oneway = str(data.get("oneway"))
        oneway_counts[oneway] = (
            oneway_counts.get(oneway, 0) + 1
        )

        blocked = str(data.get("blocked"))
        blocked_counts[blocked] = (
            blocked_counts.get(blocked, 0) + 1
        )

        highway = data.get("highway") or "UNKNOWN"
        highway_counts[highway] = (
            highway_counts.get(highway, 0) + 1
        )

    print()
    print("QA RESULTS")
    print("-" * 70)

    print(f"Edges checked:       {edge_count:,}")

    print()
    print("Missing required attributes:")

    total_missing = sum(missing.values())

    if total_missing == 0:
        print("  NONE")
    else:
        for key, count in missing.items():
            if count:
                print(f"  {key}: {count:,}")

    print()
    print(f"Bad distances:       {bad_distance:,}")

    print()
    print("Direction values:")

    for key, count in sorted(
        direction_counts.items(),
        key=lambda item: -item[1],
    ):
        print(f"  {key}: {count:,}")

    print()
    print("Oneway values:")

    for key, count in sorted(
        oneway_counts.items(),
        key=lambda item: -item[1],
    )[:20]:
        print(f"  {key}: {count:,}")

    print()
    print("Blocked values:")

    for key, count in sorted(
        blocked_counts.items(),
        key=lambda item: -item[1],
    ):
        print(f"  {key}: {count:,}")

    print()
    print(f"Highway classes:     {len(highway_counts):,}")

    print()
    print("Top highway classes:")

    for key, count in sorted(
        highway_counts.items(),
        key=lambda item: -item[1],
    )[:20]:
        print(f"  {str(key):20} {count:,}")

    print()
    print("=" * 70)

    if total_missing == 0 and bad_distance == 0:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

    print("=" * 70)


if __name__ == "__main__":
    main()