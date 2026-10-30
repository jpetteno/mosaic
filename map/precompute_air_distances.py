"""
PRECOMPUTE AIR DISTANCE MATRIX
================================
Computes haversine air distances between all POIs and stores as a
compact numpy float32 matrix for fast location selection scoring.

Used by the location selector as a proxy for travel time during
candidate scoring. Actual road-network travel time is computed by
PathGenerator only once the destination has been selected.

Output:
    data/air_distance_cache.pkl

Format:
    {
        'distances_km':  np.ndarray float32 (N x N),
        'poi_ids':       np.ndarray int64   (N,),
        'poi_id_to_idx': dict {poi_id -> int},
        'n_pois':        int,
    }

Performance:
    40k POIs -> 40k x 40k matrix = 6.4GB -- too large.
    We store only DESTINATION POIs (non-home) against ALL POIs.
    Destination POIs ~6,500 -> 6,500 x 40,500 = ~1GB still too large.

    Solution: store ALL x ALL but as float16 (half precision).
    40k x 40k float16 = 3.2GB -- still too large.

    Practical solution: store only destination POIs (non-home)
    against all POIs. ~6,500 x 40,500 float32 ~ 1GB.
    Still large. Use float16: ~500MB.

    Better: only store destination POIs x destination POIs.
    6,500 x 6,500 float32 = 169MB -- very manageable.
    Home nodes are origins only -- we compute home->dest on the fly
    using the same matrix (home snaps to nearest dest POI for lookup).

    Final design:
    - Matrix covers all NON-HOME POIs (destinations + transport)
    - Home->dest lookups use vectorized haversine at query time
      (one home point vs N destinations = trivial numpy operation)
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
import math
import pickle
import numpy as np
from tqdm import tqdm

from region_data import get_spatial_env
import os as _os
from config import SPATIAL_CACHE_FILE as _DEFAULT_CACHE, AIR_DISTANCE_FILE as _DEFAULT_OUTPUT
SPATIAL_CACHE_FILE = _os.environ.get('MOBILITY_SPATIAL_CACHE', _DEFAULT_CACHE)
OUTPUT_FILE        = _os.environ.get('MOBILITY_OUTPUT', _DEFAULT_OUTPUT)

# Home nodes are origins only -- excluded from the distance matrix
HOME_CATEGORY = "home_residential"


def haversine_matrix(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Compute full N x N haversine distance matrix in km.
    Vectorized -- no Python loops.

    Args:
        lons: (N,) array of longitudes in degrees
        lats: (N,) array of latitudes in degrees

    Returns:
        (N, N) float32 array of distances in km
    """
    R = 6371.0  # Earth radius km

    lons_r = np.radians(lons).astype(np.float32)
    lats_r = np.radians(lats).astype(np.float32)

    # Broadcasting: (N,1) - (1,N)
    dlon = lons_r[:, None] - lons_r[None, :]   # (N, N)
    dlat = lats_r[:, None] - lats_r[None, :]   # (N, N)

    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats_r[:, None]) * np.cos(lats_r[None, :])
         * np.sin(dlon / 2) ** 2)

    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return (R * c).astype(np.float32)


def haversine_vector(
    origin_lon: float,
    origin_lat: float,
    dest_lons: np.ndarray,
    dest_lats: np.ndarray,
) -> np.ndarray:
    """
    Compute distances from one origin to N destinations in km.
    Used at query time for home -> destination lookups.
    """
    R = 6371.0
    lon1 = math.radians(origin_lon)
    lat1 = math.radians(origin_lat)
    lons2 = np.radians(dest_lons).astype(np.float64)
    lats2 = np.radians(dest_lats).astype(np.float64)

    dlon = lons2 - lon1
    dlat = lats2 - lat1

    a = (np.sin(dlat / 2) ** 2
         + math.cos(lat1) * np.cos(lats2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return (R * c).astype(np.float32)


def main():
    print("=" * 70)
    print("PRECOMPUTING AIR DISTANCE MATRIX")
    print("=" * 70)

    spatial_env = get_spatial_env(SPATIAL_CACHE_FILE)
    pois = spatial_env.pois

    # Detect coordinate columns
    lon_col = next((c for c in ["x", "lon", "longitude"] if c in pois.columns), None)
    lat_col = next((c for c in ["y", "lat", "latitude"]  if c in pois.columns), None)
    if lon_col is None or lat_col is None:
        raise ValueError(f"Cannot find coordinate columns. Available: {list(pois.columns)}")

    # Separate destination POIs from home nodes
    dest_mask = pois["poi_category"] != HOME_CATEGORY
    dest_pois = pois[dest_mask].copy().reset_index(drop=True)

    print(f"\nTotal POIs:        {len(pois):,}")
    print(f"Destination POIs:  {len(dest_pois):,}  (excl. home_residential)")
    print(f"Home nodes:        {(~dest_mask).sum():,}  (origins only, not in matrix)")

    lons = dest_pois[lon_col].values.astype(np.float32)
    lats = dest_pois[lat_col].values.astype(np.float32)
    poi_ids = dest_pois["poi_id"].values.astype(np.int64)

    n = len(poi_ids)
    print(f"\nMatrix size: {n:,} x {n:,} = {n*n*4/1024/1024:.0f} MB (float32)")

    # Build distance matrix in batches to manage memory
    print("\nComputing haversine matrix...")
    distances_km = haversine_matrix(lons, lats)
    np.fill_diagonal(distances_km, 0.0)

    print(f"\nDistance statistics:")
    mask = distances_km > 0
    print(f"  Mean:   {distances_km[mask].mean():.2f} km")
    print(f"  Median: {np.median(distances_km[mask]):.2f} km")
    print(f"  Max:    {distances_km.max():.2f} km")

    poi_id_to_idx = {int(pid): int(i) for i, pid in enumerate(poi_ids)}

    # Also store home node coordinates for fast query-time lookup
    home_pois = pois[~dest_mask].copy().reset_index(drop=True)
    home_ids   = home_pois["poi_id"].values.astype(np.int64)
    home_lons  = home_pois[lon_col].values.astype(np.float32)
    home_lats  = home_pois[lat_col].values.astype(np.float32)

    data = {
        "distances_km":  distances_km,   # (N_dest, N_dest) float32
        "poi_ids":        poi_ids,         # (N_dest,) int64
        "poi_id_to_idx":  poi_id_to_idx,   # dict poi_id -> matrix index
        "dest_lons":      lons,            # (N_dest,) float32
        "dest_lats":      lats,            # (N_dest,) float32
        "home_ids":       home_ids,        # (N_home,) int64
        "home_lons":      home_lons,       # (N_home,) float32
        "home_lats":      home_lats,       # (N_home,) float32
        "n_pois":         n,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"  Saved ({size_mb:.0f} MB)")

    print("\n" + "=" * 70)
    print("AIR DISTANCE PRECOMPUTATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()