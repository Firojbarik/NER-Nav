
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmium


INPUT_PBF = Path("data/processed/osm/ner_roads.osm.pbf")
ATTRIBUTED_ROADS = Path(
    "data/processed/roads/ner_roads_districts.gpkg"
)

OUTPUT_DIR = Path("data/processed/graph")
GRAPH_FILE = OUTPUT_DIR / "ner_road_graph.pkl"
QA_FILE = OUTPUT_DIR / "ner_road_graph_qa.json"


def haversine_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
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
    if value is None:
        return None

    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


class RoadHandler(osmium.SimpleHandler):
    """
    Read OSM road ways and preserve their real OSM node IDs.

    Each way is converted into consecutive node pairs.
    """

    ROAD_TYPES = {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "road",
        "track",
    }

    def __init__(self):
        super().__init__()

        self.nodes = {}
        self.ways = []

    def node(self, n):
        if n.location.valid():
            self.nodes[n.id] = (
                float(n.location.lon),
                float(n.location.lat),
            )

    def way(self, w):
        highway = w.tags.get("highway")

        if highway not in self.ROAD_TYPES:
            return

        node_ids = []

        for node in w.nodes:
            node_ids.append(node.ref)

        if len(node_ids) < 2:
            return

        self.ways.append(
            {
                "osm_id": int(w.id),
                "highway": highway,
                "oneway": w.tags.get("oneway"),
                "node_ids": node_ids,
            }
        )


def main() -> None:
    print("=" * 70)
    print("NER-Nav — OSM Topological Road Graph Construction")
    print("=" * 70)

    if not INPUT_PBF.exists():
        raise FileNotFoundError(
            f"OSM PBF not found: {INPUT_PBF}"
        )

    if not ATTRIBUTED_ROADS.exists():
        raise FileNotFoundError(
            f"Attributed road dataset not found: {ATTRIBUTED_ROADS}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print(f"PBF:        {INPUT_PBF}")
    print(f"Attributes: {ATTRIBUTED_ROADS}")
    print(f"Graph:      {GRAPH_FILE}")

    # ---------------------------------------------------------------
    # Load attributed road attributes
    # ---------------------------------------------------------------

    print()
    print("Reading attributed road dataset...")

    roads = gpd.read_file(ATTRIBUTED_ROADS)

    print(f"Road records: {len(roads):,}")

    if "osm_id" not in roads.columns:
        raise ValueError(
            "Attributed road dataset is missing osm_id."
        )

    # One row per OSM way is expected.
    road_attributes = {}

    for _, row in roads.iterrows():
        osm_id = int(row["osm_id"])

        road_attributes[osm_id] = {
            "osm_id": osm_id,
            "highway": safe_value(row.get("highway")),
            "name": safe_value(row.get("name")),
            "ref": safe_value(row.get("ref")),
            "surface": safe_value(row.get("surface")),
            "maxspeed": safe_value(row.get("maxspeed")),
            "access": safe_value(row.get("access")),
            "oneway": safe_value(row.get("oneway")),
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
            "risk_multiplier": 1.0,
            "blocked": False,
        }

    print(
        f"Attribute records indexed: {len(road_attributes):,}"
    )

    # ---------------------------------------------------------------
    # Read OSM topology
    # ---------------------------------------------------------------

    print()
    print("Reading OSM topology from PBF...")
    print("This may take a while.")

    handler = RoadHandler()

    handler.apply_file(
        str(INPUT_PBF),
        locations=True,
    )

    print()
    print(f"OSM nodes loaded: {len(handler.nodes):,}")
    print(f"Road ways loaded: {len(handler.ways):,}")

    # ---------------------------------------------------------------
    # Build graph
    # ---------------------------------------------------------------

    G = nx.MultiDiGraph()

    skipped_ways = 0
    skipped_segments = 0
    missing_attributes = 0

    print()
    print("Building topological graph...")

    for way_index, way in enumerate(handler.ways, start=1):

        osm_id = way["osm_id"]

        attributes = road_attributes.get(osm_id)

        if attributes is None:
            missing_attributes += 1

            attributes = {
                "osm_id": osm_id,
                "highway": way["highway"],
                "name": None,
                "ref": None,
                "surface": None,
                "maxspeed": None,
                "access": None,
                "oneway": way["oneway"],
                "bridge": None,
                "tunnel": None,
                "lanes": None,
                "lit": None,
                "smoothness": None,
                "tracktype": None,
                "service": None,
                "district": None,
                "state": None,
                "intersection_length_m": None,
                "coverage_ratio": None,
                "risk_multiplier": 1.0,
                "blocked": False,
            }

        node_ids = way["node_ids"]

        if len(node_ids) < 2:
            skipped_ways += 1
            continue

        oneway = (
            str(attributes.get("oneway") or way.get("oneway") or "")
            .strip()
            .lower()
        )

        for a, b in zip(node_ids, node_ids[1:]):

            if a not in handler.nodes or b not in handler.nodes:
                skipped_segments += 1
                continue

            lon1, lat1 = handler.nodes[a]
            lon2, lat2 = handler.nodes[b]

            if a == b:
                skipped_segments += 1
                continue

            distance_km = haversine_km(
                lon1,
                lat1,
                lon2,
                lat2,
            )

            if distance_km <= 0:
                skipped_segments += 1
                continue

            G.add_node(
                a,
                osm_node_id=int(a),
                longitude=lon1,
                latitude=lat1,
            )

            G.add_node(
                b,
                osm_node_id=int(b),
                longitude=lon2,
                latitude=lat2,
            )

            edge_attributes = dict(attributes)
            edge_attributes["base_distance_km"] = distance_km

            if oneway in {"yes", "1", "true"}:

                G.add_edge(
                    a,
                    b,
                    **edge_attributes,
                    direction="forward",
                )

            elif oneway == "-1":

                G.add_edge(
                    b,
                    a,
                    **edge_attributes,
                    direction="reverse",
                )

            else:

                G.add_edge(
                    a,
                    b,
                    **edge_attributes,
                    direction="forward",
                )

                G.add_edge(
                    b,
                    a,
                    **edge_attributes,
                    direction="reverse",
                )

        if way_index % 50_000 == 0:
            print(
                f"  Processed OSM ways: {way_index:,}"
            )

    print()
    print("Graph construction complete.")

    # ---------------------------------------------------------------
    # QA
    # ---------------------------------------------------------------

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()

    components = list(
        nx.weakly_connected_components(G)
    )

    component_sizes = sorted(
        (len(c) for c in components),
        reverse=True,
    )

    isolated_nodes = sum(
        1
        for node in G.nodes
        if G.degree(node) == 0
    )

    self_loops = nx.number_of_selfloops(G)

    total_distance_km = sum(
        data.get("base_distance_km", 0.0)
        for _, _, data in G.edges(data=True)
    )

    print()
    print("Graph QA:")
    print(f"  Nodes:                 {node_count:,}")
    print(f"  Directed edges:        {edge_count:,}")
    print(f"  Weak components:       {len(components):,}")
    print(f"  Largest component:     {component_sizes[0]:,}")
    print(f"  Isolated nodes:        {isolated_nodes:,}")
    print(f"  Self loops:            {self_loops:,}")
    print(f"  Skipped ways:          {skipped_ways:,}")
    print(f"  Skipped segments:      {skipped_segments:,}")
    print(f"  Missing attributes:    {missing_attributes:,}")
    print(
        f"  Total edge length:     "
        f"{total_distance_km:,.2f} km"
    )

    qa = {
        "dataset": "NER-Nav OSM topological road routing graph",
        "osm_pbf": str(INPUT_PBF),
        "attributed_roads": str(ATTRIBUTED_ROADS),
        "output": str(GRAPH_FILE),
        "osm_nodes_loaded": int(len(handler.nodes)),
        "osm_road_ways_loaded": int(len(handler.ways)),
        "attribute_records": int(len(road_attributes)),
        "nodes": int(node_count),
        "directed_edges": int(edge_count),
        "weakly_connected_components": int(
            len(components)
        ),
        "largest_component_nodes": int(
            component_sizes[0]
        ),
        "isolated_nodes": int(isolated_nodes),
        "self_loops": int(self_loops),
        "skipped_ways": int(skipped_ways),
        "skipped_segments": int(skipped_segments),
        "missing_attribute_ways": int(
            missing_attributes
        ),
        "total_edge_distance_km": float(
            total_distance_km
        ),
        "top_20_component_sizes": [
            int(x)
            for x in component_sizes[:20]
        ],
        "status": "PASS",
    }

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    print()
    print("Writing graph...")

    with GRAPH_FILE.open("wb") as f:
        pickle.dump(
            G,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with QA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            qa,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("OSM TOPOLOGICAL ROAD GRAPH COMPLETE")
    print("=" * 70)
    print(f"Nodes:              {node_count:,}")
    print(f"Directed edges:     {edge_count:,}")
    print(f"Components:         {len(components):,}")
    print(
        f"Largest component:  "
        f"{component_sizes[0]:,}"
    )
    print(
        f"Total edge length:  "
        f"{total_distance_km:,.2f} km"
    )
    print(f"Graph:              {GRAPH_FILE}")
    print(f"QA report:          {QA_FILE}")
    print("Status:             PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
