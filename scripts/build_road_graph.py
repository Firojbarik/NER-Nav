from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString


INPUT = Path("data/processed/roads/ner_roads_districts.gpkg")

OUTPUT_DIR = Path("data/processed/graph")
GRAPH_FILE = OUTPUT_DIR / "ner_road_graph.pkl"
QA_FILE = OUTPUT_DIR / "ner_road_graph_qa.json"


def node_key(x: float, y: float) -> tuple[float, float]:
    """
    Stable graph-node identifier based on coordinate precision.

    OSM-derived geometries can contain tiny floating-point differences.
    Rounding prevents two visually identical intersection points from
    becoming separate graph nodes.
    """
    return (round(float(x), 7), round(float(y), 7))


def haversine_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """Approximate great-circle distance between two coordinates."""
    r = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def safe_value(value):
    """Convert pandas/NumPy values into JSON/pickle-friendly values."""
    if value is None:
        return None

    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def main() -> None:
    print("=" * 70)
    print("NER-Nav — Road Network Graph Construction")
    print("=" * 70)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Input:  {INPUT}")
    print(f"Graph:  {GRAPH_FILE}")
    print()

    print("Reading attributed road network...")
    roads = gpd.read_file(INPUT)

    print(f"Road records: {len(roads):,}")
    print(f"CRS:          {roads.crs}")

    required_columns = {
        "osm_id",
        "highway",
        "district",
        "state",
        "geometry",
    }

    missing = required_columns - set(roads.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if roads.crs is None:
        raise ValueError("Road dataset has no CRS.")

    if str(roads.crs).upper() != "EPSG:4326":
        print("Reprojecting roads to EPSG:4326...")
        roads = roads.to_crs("EPSG:4326")

    invalid = int((~roads.geometry.is_valid).sum())
    empty = int(roads.geometry.is_empty.sum())

    print()
    print("Input geometry QA:")
    print(f"  Invalid: {invalid:,}")
    print(f"  Empty:   {empty:,}")

    if invalid or empty:
        raise ValueError(
            "Input geometry QA failed. Fix the road dataset before "
            "building the graph."
        )

    # ---------------------------------------------------------------
    # Graph
    # ---------------------------------------------------------------

    G = nx.MultiDiGraph()

    print()
    print("Building graph...")

    skipped = 0
    edge_count = 0

    for idx, row in roads.iterrows():
        geom = row.geometry

        if not isinstance(geom, LineString):
            skipped += 1
            continue

        if len(geom.coords) < 2:
            skipped += 1
            continue

        start_lon, start_lat = geom.coords[0]
        end_lon, end_lat = geom.coords[-1]

        u = node_key(start_lon, start_lat)
        v = node_key(end_lon, end_lat)

        if u == v:
            skipped += 1
            continue

        G.add_node(
            u,
            longitude=u[0],
            latitude=u[1],
        )

        G.add_node(
            v,
            longitude=v[0],
            latitude=v[1],
        )

        # Calculate distance from the actual LineString geometry
        # using geodesic approximation segment-by-segment.
        coords = list(geom.coords)

        distance_km = 0.0

        for a, b in zip(coords, coords[1:]):
            distance_km += haversine_km(
                a[0],
                a[1],
                b[0],
                b[1],
            )

        if distance_km <= 0:
            skipped += 1
            continue

        highway = safe_value(row.get("highway"))
        oneway = safe_value(row.get("oneway"))

        edge_attributes = {
            "osm_id": int(row["osm_id"]),
            "highway": highway,
            "name": safe_value(row.get("name")),
            "ref": safe_value(row.get("ref")),
            "surface": safe_value(row.get("surface")),
            "maxspeed": safe_value(row.get("maxspeed")),
            "access": safe_value(row.get("access")),
            "oneway": oneway,
            "bridge": safe_value(row.get("bridge")),
            "tunnel": safe_value(row.get("tunnel")),
            "lanes": safe_value(row.get("lanes")),
            "lit": safe_value(row.get("lit")),
            "smoothness": safe_value(row.get("smoothness")),
            "tracktype": safe_value(row.get("tracktype")),
            "service": safe_value(row.get("service")),
            "district": safe_value(row.get("district")),
            "state": safe_value(row.get("state")),
            "intersection_length_m": safe_value(
                row.get("intersection_length_m")
            ),
            "coverage_ratio": safe_value(
                row.get("coverage_ratio")
            ),

            # Routing fields
            "base_distance_km": distance_km,
            "risk_multiplier": 1.0,
            "blocked": False,
        }

        # -----------------------------------------------------------
        # Direction handling
        # -----------------------------------------------------------
        #
        # OSM oneway values:
        #   yes / 1 / true -> forward only
        #   -1             -> reverse only
        #   no / missing   -> both directions
        #
        # We retain the original OSM geometry direction.
        # -----------------------------------------------------------

        oneway_text = str(oneway).strip().lower() if oneway else ""

        if oneway_text in {"yes", "1", "true"}:

            G.add_edge(
                u,
                v,
                **edge_attributes,
                direction="forward",
            )

        elif oneway_text == "-1":

            G.add_edge(
                v,
                u,
                **edge_attributes,
                direction="reverse",
            )

        else:

            G.add_edge(
                u,
                v,
                **edge_attributes,
                direction="forward",
            )

            G.add_edge(
                v,
                u,
                **edge_attributes,
                direction="reverse",
            )

        edge_count += 1

        if edge_count % 50_000 == 0:
            print(f"  Processed roads: {edge_count:,}")

    print()
    print("Graph construction complete.")

    # ---------------------------------------------------------------
    # Graph QA
    # ---------------------------------------------------------------

    node_count = G.number_of_nodes()
    directed_edge_count = G.number_of_edges()

    weak_components = list(nx.weakly_connected_components(G))

    component_sizes = sorted(
        (len(component) for component in weak_components),
        reverse=True,
    )

    isolated_nodes = sum(
        1 for node in G.nodes
        if G.degree(node) == 0
    )

    self_loops = nx.number_of_selfloops(G)

    print()
    print("Graph QA:")
    print(f"  Nodes:                 {node_count:,}")
    print(f"  Directed edges:        {directed_edge_count:,}")
    print(f"  Weak components:       {len(weak_components):,}")
    print(f"  Largest component:     {component_sizes[0]:,}")
    print(f"  Isolated nodes:        {isolated_nodes:,}")
    print(f"  Self loops:            {self_loops:,}")
    print(f"  Skipped records:       {skipped:,}")

    # ---------------------------------------------------------------
    # Edge statistics
    # ---------------------------------------------------------------

    highway_counts = {}

    for _, _, data in G.edges(data=True):
        highway = data.get("highway") or "UNKNOWN"
        highway_counts[highway] = highway_counts.get(highway, 0) + 1

    distances = [
        data["base_distance_km"]
        for _, _, data in G.edges(data=True)
        if data.get("base_distance_km") is not None
    ]

    total_distance_km = sum(distances)

    qa = {
        "dataset": "NER-Nav road routing graph",
        "input": str(INPUT),
        "output": str(GRAPH_FILE),
        "source_road_records": int(len(roads)),
        "nodes": int(node_count),
        "directed_edges": int(directed_edge_count),
        "weakly_connected_components": int(len(weak_components)),
        "largest_component_nodes": int(component_sizes[0]),
        "isolated_nodes": int(isolated_nodes),
        "self_loops": int(self_loops),
        "skipped_records": int(skipped),
        "total_edge_distance_km": float(total_distance_km),
        "highway_edge_distribution": highway_counts,
        "status": "PASS",
    }

    # ---------------------------------------------------------------
    # Save graph
    # ---------------------------------------------------------------

    print()
    print("Writing graph...")

    with GRAPH_FILE.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    with QA_FILE.open("w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)

    print()
    print("=" * 70)
    print("ROAD GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"Nodes:              {node_count:,}")
    print(f"Directed edges:     {directed_edge_count:,}")
    print(f"Components:         {len(weak_components):,}")
    print(f"Largest component:  {component_sizes[0]:,}")
    print(f"Total edge length:  {total_distance_km:,.2f} km")
    print(f"Graph:              {GRAPH_FILE}")
    print(f"QA report:          {QA_FILE}")
    print("Status:             PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()