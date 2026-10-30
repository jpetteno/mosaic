"""
Pre-compute travel times between all POIs.
Uses NetworkX all-pairs Dijkstra on the road network.

Produces one output file:
  data/travel_times_cache.pkl  -- NxN numpy float32 arrays (~280 MB)

Note: Path geometry is handled lazily by PathGenerator in path_generator.py.
The old travel_times Python dict (74M entries, ~2.5 GB) is intentionally
omitted. region_representation.py uses the numpy arrays exclusively.
"""

import sys as _sys, os as _os
_map  = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_map)
_src  = _os.path.join(_root, 'src')
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _map, _root, _src, _p


import os
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from config import SPATIAL_CACHE_FILE, TRAVEL_TIMES_CACHE_FILE

if __name__ != "__main__":
    raise ImportError("pre_compute_travel_times.py must be run directly, not imported.")

print("=" * 70)
print("PRE-COMPUTING TRAVEL TIMES")
print("=" * 70)

# ---------------------------------------------------------------------------
# Load spatial data
# ---------------------------------------------------------------------------
with open(SPATIAL_CACHE_FILE, 'rb') as f:
    data = pickle.load(f)

pois     = data['pois']
graph    = data['graph']
nodes_df = data['nodes']

print(f"\nLoaded:")
print(f"  POIs:       {len(pois):,}")
print(f"  Road nodes: {graph.number_of_nodes():,}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WALKING_SPEED_KMH = 30.0
WALKING_SPEED_MS  = WALKING_SPEED_KMH * 1000 / 3600
print(f"\nWalking speed: {WALKING_SPEED_KMH} km/h")

# ---------------------------------------------------------------------------
# POI -> road node mapping
# ---------------------------------------------------------------------------
print("\nCreating POI -> road node mapping...")
poi_to_road_node: dict = {}
for _, row in nodes_df.iterrows():
    poi_to_road_node[row['poi_id']] = {
        'road_node':     row['road_id'],
        'snap_distance': row['distance_to_road_meters'],
    }
print(f"  Mapped {len(poi_to_road_node):,} POIs to road nodes")

# ---------------------------------------------------------------------------
# Filter to largest connected component
# ---------------------------------------------------------------------------
print("\nChecking network connectivity...")
largest_cc       = max(nx.connected_components(graph), key=len)
largest_cc_nodes = set(largest_cc)
subgraph         = graph.subgraph(largest_cc_nodes)

valid_pois = [
    poi_id for poi_id in poi_to_road_node
    if poi_to_road_node[poi_id]['road_node'] in largest_cc_nodes
]
print(f"  {len(valid_pois):,} POIs in largest connected component "
      f"({len(valid_pois)/len(poi_to_road_node)*100:.1f}%)")

# ---------------------------------------------------------------------------
# All-pairs shortest path LENGTHS only (no node sequences needed)
# ---------------------------------------------------------------------------
print(f"\nComputing all-pairs shortest path lengths...")
print(f"  (lengths only -- path geometry handled lazily by PathGenerator)")

all_lengths = {}
for source, lengths in tqdm(
    nx.all_pairs_dijkstra_path_length(subgraph, weight='weight'),
    total=subgraph.number_of_nodes(),
    desc="Dijkstra",
):
    all_lengths[source] = lengths

print("  Done")

# ---------------------------------------------------------------------------
# Build numpy matrices  (no Python dict -- saves ~2.5 GB)
# ---------------------------------------------------------------------------
print(f"\nBuilding travel time matrix...")
n          = len(valid_pois)
poi_to_idx = {poi_id: i for i, poi_id in enumerate(valid_pois)}

travel_times_matrix = np.full((n, n), 999.0, dtype=np.float32)
distances_matrix    = np.full((n, n), 999.0, dtype=np.float32)

for poi_1 in tqdm(valid_pois, desc="Building matrix"):
    road_node_1 = poi_to_road_node[poi_1]['road_node']
    snap_dist_1 = poi_to_road_node[poi_1]['snap_distance']
    i           = poi_to_idx[poi_1]
    travel_times_matrix[i, i] = 0.0
    distances_matrix[i, i]    = 0.0

    for poi_2 in valid_pois:
        if poi_1 == poi_2:
            continue
        road_node_2 = poi_to_road_node[poi_2]['road_node']
        snap_dist_2 = poi_to_road_node[poi_2]['snap_distance']
        j           = poi_to_idx[poi_2]
        try:
            road_dist_m               = all_lengths[road_node_1][road_node_2]
            total_dist_m              = snap_dist_1 + road_dist_m + snap_dist_2
            travel_times_matrix[i, j] = (total_dist_m / WALKING_SPEED_MS) / 60.0
            distances_matrix[i, j]    = total_dist_m / 1000.0
        except (KeyError, nx.NetworkXNoPath):
            pass  # stays 999.0

print(f"  Matrix built: {n}x{n}")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
valid_mask  = travel_times_matrix < 999.0
valid_times = travel_times_matrix[valid_mask]
print(f"\nTravel Time Statistics:")
print(f"  Mean:        {valid_times.mean():.1f} minutes")
print(f"  Median:      {np.median(valid_times):.1f} minutes")
print(f"  Min:         {valid_times.min():.1f} minutes")
print(f"  Max:         {valid_times.max():.1f} minutes")
print(f"  Unreachable: {(travel_times_matrix >= 999.0).sum():,} pairs")

# Free all_lengths before saving to reduce peak memory
del all_lengths

# ---------------------------------------------------------------------------
# Save  (~280 MB, not 2.7 GB)
# ---------------------------------------------------------------------------
print("\nSaving travel time cache...")
os.makedirs(os.path.dirname(TRAVEL_TIMES_CACHE_FILE), exist_ok=True)

travel_data = {
    'travel_times_array': travel_times_matrix,
    'distances_array':    distances_matrix,
    'poi_ids':            np.array(valid_pois),
    'walking_speed_kmh':  WALKING_SPEED_KMH,
    'is_projected':       False,
    'num_pois':           len(valid_pois),
    'poi_to_road_node':   poi_to_road_node,
}

with open(TRAVEL_TIMES_CACHE_FILE, 'wb') as f:
    pickle.dump(travel_data, f, protocol=pickle.HIGHEST_PROTOCOL)

size_mb = os.path.getsize(TRAVEL_TIMES_CACHE_FILE) / 1024 / 1024
print(f"  {TRAVEL_TIMES_CACHE_FILE}  ({size_mb:.0f} MB)")

print("\n" + "=" * 70)
print("PRE-COMPUTATION COMPLETE")
print("=" * 70)
print("\nNext steps:")
print(f"  1. Delete old {TRAVEL_TIMES_CACHE_FILE} if it was >500 MB")
print("  2. Run train.py")