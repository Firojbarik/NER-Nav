from pathlib import Path
import pickle
import networkx as nx
import geopandas as gpd
from collections import Counter

GRAPH = Path("data/processed/graph/ner_road_graph.pkl")
ROADS = Path("data/processed/roads/ner_roads_districts.gpkg")

print("=" * 70)
print("NER-Nav — Road Graph Topology Diagnosis")
print("=" * 70)

# ---------------------------------------------------------------------
# Load graph
# ---------------------------------------------------------------------

print("\nLoading graph...")
with GRAPH.open("rb") as f:
    G = pickle.load(f)

print(f"Graph type:       {type(G).__name__}")
print(f"Nodes:            {G.number_of_nodes():,}")
print(f"Directed edges:   {G.number_of_edges():,}")

# ---------------------------------------------------------------------
# Component analysis
# ---------------------------------------------------------------------

print("\nAnalyzing weakly connected components...")

components = list(nx.weakly_connected_components(G))
sizes = sorted((len(c) for c in components), reverse=True)

print(f"Components:       {len(sizes):,}")
print(f"Largest:          {sizes[0]:,}")
print(f"Top 20:           {sizes[:20]}")

print("\nComponent size distribution:")
for n in [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000]:
    print(f"  Exactly {n:>4}: {sum(s == n for s in sizes):,}")

for threshold in [2, 5, 10, 50, 100, 500, 1000]:
    print(
        f"  >= {threshold:>4}: "
        f"{sum(s >= threshold for s in sizes):,}"
    )

# ---------------------------------------------------------------------
# Node degree analysis
# ---------------------------------------------------------------------

print("\nNode degree analysis...")

in_degrees = [d for _, d in G.in_degree()]
out_degrees = [d for _, d in G.out_degree()]
total_degrees = [d for _, d in G.degree()]

print(f"Min in-degree:    {min(in_degrees)}")
print(f"Max in-degree:    {max(in_degrees)}")
print(f"Min out-degree:   {min(out_degrees)}")
print(f"Max out-degree:   {max(out_degrees)}")

print(
    "Nodes with total degree 1:",
    sum(d == 1 for d in total_degrees)
)

print(
    "Nodes with total degree 2:",
    sum(d == 2 for d in total_degrees)
)

print(
    "Nodes with total degree >=3:",
    sum(d >= 3 for d in total_degrees)
)

# ---------------------------------------------------------------------
# Inspect node identifiers
# ---------------------------------------------------------------------

print("\nSample node IDs:")

for node in list(G.nodes)[:20]:
    print(" ", repr(node))

# ---------------------------------------------------------------------
# Inspect edge attributes
# ---------------------------------------------------------------------

print("\nSample edges:")

for u, v, data in list(G.edges(data=True))[:10]:
    print("\nFROM:", repr(u))
    print("TO:  ", repr(v))
    print("ATTR:", data)

# ---------------------------------------------------------------------
# Load source roads
# ---------------------------------------------------------------------

print("\nLoading source road dataset...")

roads = gpd.read_file(ROADS)

print(f"Road records:     {len(roads):,}")
print(f"CRS:              {roads.crs}")
print(f"Geometry types:")
print(roads.geometry.geom_type.value_counts().to_dict())

# ---------------------------------------------------------------------
# Coordinate precision analysis
# ---------------------------------------------------------------------

print("\nCoordinate precision analysis...")

coords = []

for geom in roads.geometry:
    if geom is None or geom.is_empty:
        continue

    coords.append(tuple(geom.coords[0]))
    coords.append(tuple(geom.coords[-1]))

print(f"Endpoint count:    {len(coords):,}")

unique_coords = set(coords)

print(f"Unique endpoints:  {len(unique_coords):,}")
print(
    f"Endpoint reuse:    "
    f"{len(coords) - len(unique_coords):,}"
)

# ---------------------------------------------------------------------
# Duplicate endpoint counts
# ---------------------------------------------------------------------

counts = Counter(coords)

shared = [c for c, n in counts.items() if n >= 2]

print(f"Shared endpoints:  {len(shared):,}")

print("\nMost frequently shared endpoints:")

for coord, count in counts.most_common(20):
    if count >= 2:
        print(f"  {coord} -> {count} roads")

# ---------------------------------------------------------------------
# Graph coordinate interpretation
# ---------------------------------------------------------------------

print("\nGraph node coordinate check...")

coordinate_nodes = 0
non_coordinate_nodes = 0

for node in G.nodes:
    if (
        isinstance(node, tuple)
        and len(node) == 2
        and all(isinstance(x, (int, float)) for x in node)
    ):
        coordinate_nodes += 1
    else:
        non_coordinate_nodes += 1

print(f"Coordinate nodes:     {coordinate_nodes:,}")
print(f"Other node IDs:       {non_coordinate_nodes:,}")

# ---------------------------------------------------------------------
# Final diagnosis hints
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if len(sizes) > 10000:
    print("WARNING: Graph is severely fragmented.")

if sizes[0] < 10000:
    print("WARNING: Largest component is unexpectedly small.")

if sum(s == 2 for s in sizes) > 100000:
    print("WARNING: Huge number of 2-node components detected.")

if len(unique_coords) == len(coords):
    print(
        "WARNING: Almost every road endpoint is unique. "
        "Road-to-road endpoint connectivity may be missing."
    )

print("\nDiagnosis complete.")