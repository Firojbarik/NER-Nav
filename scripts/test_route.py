from __future__ import annotations

import math
import pickle
import time
from pathlib import Path

import networkx as nx


GRAPH_FILE = Path(
    "data/processed/graph/ner_road_graph.pkl"
)


# ---------------------------------------------------------------------
# Geographic utilities
# ---------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """
    Great-circle distance between two WGS84 coordinates.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2.0) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * math.asin(math.sqrt(a))
    )


def node_lon_lat(G: nx.MultiDiGraph, node):
    """
    Return (longitude, latitude) for an OSM node.

    The current NER graph uses integer OSM node IDs.
    Coordinates are stored as node attributes.
    """
    data = G.nodes[node]

    lon = data.get("longitude")
    lat = data.get("latitude")

    if lon is None or lat is None:
        raise ValueError(
            f"Node {node!r} is missing longitude/latitude attributes."
        )

    return float(lon), float(lat)


# ---------------------------------------------------------------------
# A* heuristic
# ---------------------------------------------------------------------

def make_heuristic(
    G: nx.MultiDiGraph,
    target,
):
    """
    Build an admissible geographic heuristic.

    The graph's effective edge cost is:

        base_distance_km * risk_multiplier

    with risk_multiplier >= 1.

    Therefore straight-line distance is a lower bound on the
    actual route cost.
    """

    target_lon, target_lat = node_lon_lat(
        G,
        target,
    )

    def heuristic(node, _target):
        lon, lat = node_lon_lat(
            G,
            node,
        )

        return haversine_km(
            lon,
            lat,
            target_lon,
            target_lat,
        )

    return heuristic


# ---------------------------------------------------------------------
# Dynamic edge weight
# ---------------------------------------------------------------------

def routing_weight(u, v, edge_data):
    """
    Return the effective routing cost for an edge.

    Supports both:
        base_distance_km
        risk_multiplier

    Project routing specification:

        New weight =
            Base distance × Risk multiplier

    Blocked edges are hidden from A* by returning None.
    """

    # MultiDiGraph passes a dictionary of parallel-edge
    # dictionaries to a weight function.
    #
    # Example:
    #
    # {
    #     0: {...},
    #     1: {...},
    # }
    #
    # Select the cheapest usable parallel edge.

    if isinstance(edge_data, dict):
        values = edge_data.values()

        # Detect MultiDiGraph's key -> attribute-dict structure.
        if values:
            first = next(iter(values))

            if isinstance(first, dict):
                best = None

                for attrs in values:
                    if attrs.get("blocked", False):
                        continue

                    base = attrs.get("base_distance_km")
                    risk = attrs.get("risk_multiplier", 1.0)

                    if base is None:
                        continue

                    try:
                        base = float(base)
                        risk = float(risk)
                    except (TypeError, ValueError):
                        continue

                    if base <= 0:
                        continue

                    if risk <= 0:
                        risk = 1.0

                    cost = base * risk

                    if best is None or cost < best:
                        best = cost

                return best

    # Fallback for ordinary DiGraph-style edge attributes.
    if edge_data.get("blocked", False):
        return None

    base = edge_data.get("base_distance_km")

    if base is None:
        return None

    try:
        base = float(base)
    except (TypeError, ValueError):
        return None

    risk = edge_data.get(
        "risk_multiplier",
        1.0,
    )

    try:
        risk = float(risk)
    except (TypeError, ValueError):
        risk = 1.0

    if base <= 0:
        return None

    if risk <= 0:
        risk = 1.0

    return base * risk


# ---------------------------------------------------------------------
# Graph QA
# ---------------------------------------------------------------------

def validate_graph(G: nx.MultiDiGraph) -> None:
    """
    Validate the minimum graph requirements before routing.
    """

    print()
    print("Graph validation...")
    print("-" * 70)

    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError(
            f"Expected MultiDiGraph, got {type(G).__name__}"
        )

    print(
        f"Graph type:       {type(G).__name__}"
    )

    print(
        f"Nodes:            {G.number_of_nodes():,}"
    )

    print(
        f"Directed edges:   {G.number_of_edges():,}"
    )

    if G.number_of_nodes() == 0:
        raise RuntimeError("Graph contains no nodes.")

    if G.number_of_edges() == 0:
        raise RuntimeError("Graph contains no edges.")

    # Check a small sample rather than scanning 13M edges again.
    sample_count = 0

    for node, attrs in G.nodes(data=True):
        if "longitude" not in attrs:
            raise RuntimeError(
                f"Node {node} missing longitude."
            )

        if "latitude" not in attrs:
            raise RuntimeError(
                f"Node {node} missing latitude."
            )

        sample_count += 1

        if sample_count >= 100:
            break

    print(
        "Node coordinate attributes: PASS"
    )

    sample_count = 0

    for u, v, key, attrs in G.edges(
        keys=True,
        data=True,
    ):
        base = attrs.get("base_distance_km")
        risk = attrs.get("risk_multiplier")
        blocked = attrs.get("blocked")

        if base is None:
            raise RuntimeError(
                f"Edge {u}->{v}/{key} missing "
                "base_distance_km."
            )

        if risk is None:
            raise RuntimeError(
                f"Edge {u}->{v}/{key} missing "
                "risk_multiplier."
            )

        if blocked is None:
            raise RuntimeError(
                f"Edge {u}->{v}/{key} missing "
                "blocked."
            )

        sample_count += 1

        if sample_count >= 100:
            break

    print(
        "Routing edge attributes:  PASS"
    )


# ---------------------------------------------------------------------
# Strongly connected component
# ---------------------------------------------------------------------

def find_largest_strong_component(G):
    """
    Find the largest strongly connected component.

    This is required for a directed route test.

    A weakly connected component only guarantees that nodes are
    connected if edge direction is ignored.

    A strongly connected component guarantees:

        source -> target

    and

        target -> source
    """

    print()
    print(
        "Finding largest strongly connected component..."
    )

    start = time.perf_counter()

    largest = max(
        nx.strongly_connected_components(G),
        key=len,
    )

    elapsed = time.perf_counter() - start

    size = len(largest)

    coverage = (
        100.0
        * size
        / G.number_of_nodes()
    )

    print(
        f"Largest strong component: "
        f"{size:,}"
    )

    print(
        f"Strong-component coverage: "
        f"{coverage:.2f}%"
    )

    print(
        f"Component search: "
        f"{elapsed:.2f} seconds"
    )

    if size < 2:
        raise RuntimeError(
            "Largest strongly connected component "
            "contains fewer than 2 nodes."
        )

    return largest


# ---------------------------------------------------------------------
# Test-node selection
# ---------------------------------------------------------------------

def choose_test_nodes(
    G: nx.MultiDiGraph,
    component,
):
    """
    Select geographically separated nodes from the same SCC.

    IMPORTANT:
    Graph nodes are integer OSM IDs.

    Coordinates must therefore be read from:

        G.nodes[node]["longitude"]
        G.nodes[node]["latitude"]
    """

    print()
    print("Selecting test nodes...")

    nodes = list(component)

    if len(nodes) < 2:
        raise RuntimeError(
            "Not enough nodes in selected component."
        )

    # We do not use:
    #
    #     n[0]
    #
    # because nodes are integer OSM IDs.

    west = min(
        nodes,
        key=lambda n: G.nodes[n]["longitude"],
    )

    east = max(
        nodes,
        key=lambda n: G.nodes[n]["longitude"],
    )

    # If the component happens to have equal longitude extremes,
    # choose north/south instead.

    if west == east:
        south = min(
            nodes,
            key=lambda n: G.nodes[n]["latitude"],
        )

        north = max(
            nodes,
            key=lambda n: G.nodes[n]["latitude"],
        )

        source = south
        target = north

    else:
        source = west
        target = east

    if source == target:
        raise RuntimeError(
            "Could not select two distinct test nodes."
        )

    source_lon, source_lat = node_lon_lat(
        G,
        source,
    )

    target_lon, target_lat = node_lon_lat(
        G,
        target,
    )

    straight_line = haversine_km(
        source_lon,
        source_lat,
        target_lon,
        target_lat,
    )

    print(
        f"Source:           {source}"
    )

    print(
        f"Source coords:    "
        f"({source_lon:.6f}, {source_lat:.6f})"
    )

    print(
        f"Target:           {target}"
    )

    print(
        f"Target coords:    "
        f"({target_lon:.6f}, {target_lat:.6f})"
    )

    print(
        f"Straight-line:    "
        f"{straight_line:.2f} km"
    )

    return source, target, straight_line


# ---------------------------------------------------------------------
# Route statistics
# ---------------------------------------------------------------------

def calculate_route_distance(
    G: nx.MultiDiGraph,
    path,
) -> float:
    """
    Calculate actual base distance along the selected path.

    For a MultiDiGraph, choose the parallel edge with the lowest
    effective routing cost.
    """

    total = 0.0

    for u, v in zip(
        path,
        path[1:],
    ):
        edge_data = G.get_edge_data(
            u,
            v,
        )

        if not edge_data:
            raise RuntimeError(
                f"Missing edge data for {u} -> {v}"
            )

        best_attrs = None
        best_cost = None

        for attrs in edge_data.values():

            if attrs.get("blocked", False):
                continue

            base = attrs.get(
                "base_distance_km"
            )

            risk = attrs.get(
                "risk_multiplier",
                1.0,
            )

            if base is None:
                continue

            try:
                base = float(base)
                risk = float(risk)
            except (TypeError, ValueError):
                continue

            if base <= 0:
                continue

            if risk <= 0:
                risk = 1.0

            cost = base * risk

            if (
                best_cost is None
                or cost < best_cost
            ):
                best_cost = cost
                best_attrs = attrs

        if best_attrs is None:
            raise RuntimeError(
                f"No usable edge for {u} -> {v}"
            )

        total += float(
            best_attrs["base_distance_km"]
        )

    return total


def calculate_route_cost(
    G: nx.MultiDiGraph,
    path,
) -> float:
    """
    Calculate effective dynamic route cost:

        base_distance_km * risk_multiplier
    """

    total = 0.0

    for u, v in zip(
        path,
        path[1:],
    ):
        edge_data = G.get_edge_data(
            u,
            v,
        )

        if not edge_data:
            raise RuntimeError(
                f"Missing edge data for {u} -> {v}"
            )

        best_cost = None

        for attrs in edge_data.values():

            if attrs.get("blocked", False):
                continue

            base = attrs.get(
                "base_distance_km"
            )

            risk = attrs.get(
                "risk_multiplier",
                1.0,
            )

            if base is None:
                continue

            try:
                base = float(base)
                risk = float(risk)
            except (TypeError, ValueError):
                continue

            if base <= 0:
                continue

            if risk <= 0:
                risk = 1.0

            cost = base * risk

            if (
                best_cost is None
                or cost < best_cost
            ):
                best_cost = cost

        if best_cost is None:
            raise RuntimeError(
                f"No usable edge for {u} -> {v}"
            )

        total += best_cost

    return total


# ---------------------------------------------------------------------
# Route validation
# ---------------------------------------------------------------------

def validate_route(
    G: nx.MultiDiGraph,
    path,
    source,
    target,
):
    """
    Validate the returned A* path.
    """

    if not path:
        raise RuntimeError(
            "A* returned an empty path."
        )

    if path[0] != source:
        raise RuntimeError(
            "Route does not start at source."
        )

    if path[-1] != target:
        raise RuntimeError(
            "Route does not end at target."
        )

    for u, v in zip(
        path,
        path[1:],
    ):
        if not G.has_edge(u, v):
            raise RuntimeError(
                f"Route contains missing directed "
                f"edge {u} -> {v}."
            )

    print()
    print(
        "Route validation: PASS"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print(
        "NER-Nav — A* Routing Engine Test"
    )
    print("=" * 70)

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph not found: {GRAPH_FILE}"
        )

    # ---------------------------------------------------------------
    # Load graph
    # ---------------------------------------------------------------

    print()
    print("Loading graph...")

    load_start = time.perf_counter()

    with GRAPH_FILE.open(
        "rb"
    ) as f:
        G = pickle.load(f)

    load_time = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Graph type:      "
        f"{type(G).__name__}"
    )

    print(
        f"Nodes:            "
        f"{G.number_of_nodes():,}"
    )

    print(
        f"Directed edges:   "
        f"{G.number_of_edges():,}"
    )

    print(
        f"Load time:        "
        f"{load_time:.2f} seconds"
    )

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    validate_graph(G)

    # ---------------------------------------------------------------
    # Find SCC
    # ---------------------------------------------------------------

    largest = find_largest_strong_component(
        G
    )

    # ---------------------------------------------------------------
    # Choose source / target
    # ---------------------------------------------------------------

    source, target, straight_line = (
        choose_test_nodes(
            G,
            largest,
        )
    )

    # ---------------------------------------------------------------
    # Build heuristic
    # ---------------------------------------------------------------

    heuristic = make_heuristic(
        G,
        target,
    )

    # ---------------------------------------------------------------
    # Sanity-check directed reachability
    #
    # Because source and target are in the same SCC, this should
    # succeed. This check makes the reason for any future failure
    # explicit before invoking A*.
    # ---------------------------------------------------------------

    print()
    print(
        "Checking directed reachability..."
    )

    reach_start = time.perf_counter()

    if not nx.has_path(
        G,
        source,
        target,
    ):
        raise RuntimeError(
            "Internal test error: source and target "
            "are in the same SCC but no directed path "
            "was found."
        )

    reach_time = (
        time.perf_counter()
        - reach_start
    )

    print(
        "Directed reachability: PASS"
    )

    print(
        f"Reachability check: "
        f"{reach_time:.2f} seconds"
    )

    # ---------------------------------------------------------------
    # A*
    # ---------------------------------------------------------------

    print()
    print("Running A*...")
    print(
        "Routing weight: "
        "base_distance_km × risk_multiplier"
    )
    print(
        "This may take a while on the full NER graph."
    )

    route_start = time.perf_counter()

    try:

        path = nx.astar_path(
            G,
            source,
            target,
            heuristic=heuristic,
            weight=routing_weight,
        )

    except nx.NetworkXNoPath as exc:

        elapsed = (
            time.perf_counter()
            - route_start
        )

        print()
        print(
            "ERROR: No route found."
        )

        print(
            f"A* runtime: "
            f"{elapsed:.2f} seconds"
        )

        raise RuntimeError(
            "A* reported no route even though "
            "source and target were selected from "
            "the same strongly connected component."
        ) from exc

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - route_start
        )

        print()
        print(
            "ERROR: A* failed."
        )

        print(
            f"A* runtime: "
            f"{elapsed:.2f} seconds"
        )

        raise

    route_time = (
        time.perf_counter()
        - route_start
    )

    # ---------------------------------------------------------------
    # Validate route
    # ---------------------------------------------------------------

    validate_route(
        G,
        path,
        source,
        target,
    )

    # ---------------------------------------------------------------
    # Route statistics
    # ---------------------------------------------------------------

    route_nodes = len(path)

    base_distance = (
        calculate_route_distance(
            G,
            path,
        )
    )

    weighted_cost = (
        calculate_route_cost(
            G,
            path,
        )
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("A* ROUTING TEST COMPLETE")
    print("=" * 70)

    print(
        f"Source:            {source}"
    )

    print(
        f"Target:            {target}"
    )

    print(
        f"Straight-line:     "
        f"{straight_line:.2f} km"
    )

    print(
        f"Route nodes:       "
        f"{route_nodes:,}"
    )

    print(
        f"Route distance:    "
        f"{base_distance:,.2f} km"
    )

    print(
        f"Weighted cost:     "
        f"{weighted_cost:,.2f}"
    )

    print(
        f"A* runtime:        "
        f"{route_time:.2f} seconds"
    )

    if straight_line > 0:
        detour_ratio = (
            base_distance
            / straight_line
        )

        print(
            f"Detour ratio:      "
            f"{detour_ratio:.3f}x"
        )

    print()
    print(
        "STATUS: PASS"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()