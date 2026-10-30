from __future__ import annotations

"""
PATH LIBRARY -- MODE-AWARE ROUTING
===================================
On-demand shortest-time routing for CAR, BICYCLE, and FOOT modes.
Uses the NetworkX graph from spatial_cache_filtered.pkl.

Key design decisions
--------------------
- CAR:     Directed graph (one-ways enforced), weight = car_time_sec
- BICYCLE: Undirected subgraph (bicycle_final=yes only), weight = bike_time_sec
- FOOT:    Undirected subgraph (foot_final=yes only),    weight = foot_time_sec
- PT:      Not graph-based -- handled separately via pt_connectivity table.
           See get_pt_travel_time() for the formula-based estimate.

Cache format
------------
  Key:   (origin_poi_id, dest_poi_id, mode)   mode in {'car','bicycle','foot'}
  Value: PathResult(time_minutes, linestring)

  linestring: [(lon, lat, cum_dist_m), ...]  in WGS84 degrees

Both geometry and travel time are computed in a single Dijkstra call
and stored together. During training, only time_minutes is used.
During visualization, the full linestring is available at zero extra cost.

Air distance
------------
get_air_distance_km(poi_a, poi_b) -- pure haversine, no graph needed.
Used for radius-of-gyration analysis and baseline comparisons.
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p



import atexit
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from config import PATH_LIBRARY_FILE as CACHE_FILE, SPATIAL_CACHE_FILE as _DEFAULT_SPATIAL_CACHE
AUTOSAVE_N   = 1000

WALK_SPEED_KMH = 5.0
BIKE_SPEED_KMH = 15.0

# PT constants (formula-based, not graph-based)
PT_AVG_SPEED_KMH  = 20.0   # average bus/tram speed incl. stops
PT_AVG_WAIT_MIN   = 7.5    # average wait time (15 min headway, random arrival)
PT_WALK_SPEED_KMH = 5.0    # walking to/from stop

VALID_MODES = {"car", "bicycle", "foot"}

Linestring = List[Tuple[float, float, float]]   # (lon, lat, cum_dist_m)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PathResult:
    time_minutes: float
    linestring:   Linestring
    mode:         str
    distance_m:   float


# ---------------------------------------------------------------------------
# Coordinate handling  (WGS84 native -- v4 import stores EPSG:4326 directly)
# ---------------------------------------------------------------------------

def _to_wgs84(x: float, y: float, is_projected: bool = False) -> Tuple[float, float]:
    """
    Return (lon, lat) in WGS84 degrees.
    If coordinates are already WGS84 (v4 import), returns as-is.
    If projected EPSG:3857 (legacy), converts via pyproj or fallback.
    """
    if not is_projected:
        return float(x), float(y)   # already lon, lat

    try:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        return t.transform(x, y)
    except ImportError:
        R   = 6_378_137.0
        lon = math.degrees(x / R)
        lat = math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)
        return lon, lat


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R    = 6_371_000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _build_car_graph(base_graph: nx.Graph) -> nx.DiGraph:
    """
    Directed graph for car routing. ALL edges included.
    Penalty speeds in SQL enforce mode separation:
      - pedestrian/footway/cycleway: 5 km/h (creep)
      - steps: 0.001 km/h (near-impassable)
    Dijkstra routes around penalties naturally.
    One-ways enforced via oneway_final.
    """
    G = nx.DiGraph()
    for u, v, data in base_graph.edges(data=True):
        t = data.get("car_time_sec")
        if t is None or t <= 0:
            continue
        extra = {k: v2 for k, v2 in data.items() if k != "weight"}
        G.add_edge(u, v, weight=float(t), **extra)
        if data.get("oneway_final", "no") != "yes":
            G.add_edge(v, u, weight=float(t), **extra)
    return G


def _build_bicycle_graph(base_graph: nx.Graph) -> nx.Graph:
    """
    Undirected graph for bicycle routing. ALL edges included.
    Penalty speeds enforce separation:
      - motorway/steps: 0.001 km/h (near-impassable)
      - footway/pedestrian: 5 km/h (dismount)
    """
    G = nx.Graph()
    for u, v, data in base_graph.edges(data=True):
        t = data.get("bike_time_sec")
        if t is None or t <= 0:
            continue
        extra = {k: v2 for k, v2 in data.items() if k != "weight"}
        G.add_edge(u, v, weight=float(t), **extra)
    return G


def _build_foot_graph(base_graph: nx.Graph) -> nx.Graph:
    """
    Undirected graph for foot routing. ALL edges included.
    Penalty speeds enforce separation:
      - motorway/motorway_link: 0.001 km/h (near-impassable)
    Everything else walkable at 5 km/h.
    """
    G = nx.Graph()
    for u, v, data in base_graph.edges(data=True):
        t = data.get("foot_time_sec")
        if t is None or t <= 0:
            continue
        extra = {k: v2 for k, v2 in data.items() if k != "weight"}
        G.add_edge(u, v, weight=float(t), **extra)
    return G


# ---------------------------------------------------------------------------
# PathGenerator
# ---------------------------------------------------------------------------

class PathGenerator:
    """
    Mode-aware lazy path library.

    Usage
    -----
        lib = PathGenerator.load("data/spatial_cache_filtered.pkl")

        # Route query -- Dijkstra on first call, cache hit after
        result = lib.get_path(origin_poi_id, dest_poi_id, mode="car")
        print(result.time_minutes, result.linestring)

        # Air distance (no graph needed)
        km = lib.get_air_distance_km(poi_a, poi_b)

        # PT travel time estimate (formula-based)
        pt_time = lib.get_pt_travel_time(origin_poi_id, dest_poi_id)
    """

    def __init__(
        self,
        graphs:         Dict[str, nx.Graph],   # mode -> graph
        poi_coords:     Dict[int, Tuple[float, float]],
        poi_to_road:    Dict[int, dict],
        node_coords:    Dict[int, Tuple[float, float]],
        is_projected:   bool,
        transport_nodes: Optional[object],      # DataFrame or None
        pt_connectivity: Optional[object],      # DataFrame or None
        cache:          Dict[Tuple, PathResult] = None,
    ):
        self._graphs          = graphs
        self._poi_coords      = poi_coords
        self._poi_to_road     = poi_to_road
        self._node_coords     = node_coords
        self._is_projected    = is_projected
        self._transport_nodes = transport_nodes
        self._pt_connectivity = pt_connectivity
        self._cache: Dict[Tuple, Optional[PathResult]] = cache or {}
        self._new_since_save  = 0

        # Precompute stop lookup structures if PT data available
        self._stop_coords: Dict[int, Tuple[float, float]] = {}
        self._pt_conn_index: Dict[int, Dict[int, float]]  = {}
        self._build_pt_index()

        print(f"[PathGenerator] {len(self._poi_coords):,} POIs  |  "
              f"{len(self._node_coords):,} road nodes  |  "
              f"{len(self._cache):,} cached paths")
        for mode, g in self._graphs.items():
            print(f"  [{mode}] {g.number_of_nodes():,} nodes, "
                  f"{g.number_of_edges():,} edges")

        atexit.register(self.save)

    # ------------------------------------------------------------------
    # PT index
    # ------------------------------------------------------------------

    def _build_pt_index(self) -> None:
        """Build in-memory index for PT connectivity lookups."""
        if self._transport_nodes is None or self._pt_connectivity is None:
            return

        try:
            import pandas as pd
            tn = self._transport_nodes
            # Vectorised stop-coord index
            _sids  = tn["stop_id"].to_numpy(dtype=np.int64)
            _xs    = tn["x"].to_numpy(dtype=np.float64)
            _ys    = tn["y"].to_numpy(dtype=np.float64)
            self._stop_coords = {int(s): (float(x), float(y))
                                 for s, x, y in zip(_sids, _xs, _ys)}

            # PT connectivity index -- itertuples is ~5x faster than iterrows
            pc = self._pt_connectivity
            for row in pc[["origin_stop_id", "dest_stop_id", "geom_distance_m"]].itertuples(index=False):
                self._pt_conn_index.setdefault(
                    int(row.origin_stop_id), {}
                )[int(row.dest_stop_id)] = float(row.geom_distance_m)

            print(f"  [PT] {len(self._stop_coords):,} stops indexed, "
                  f"{len(self._pt_conn_index):,} origin stops with connections")
        except Exception as e:
            print(f"  [PT] Warning: could not build PT index: {e}")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        spatial_cache: str = None,
        cache_file:    str = CACHE_FILE,
    ) -> "PathGenerator":

        if spatial_cache is None:
            spatial_cache = _DEFAULT_SPATIAL_CACHE
        print(f"[PathGenerator] Loading: {spatial_cache}")
        with open(spatial_cache, "rb") as f:
            data = pickle.load(f)

        pois     = data["pois"]
        nodes_df = data["nodes"]
        graph    = data.get("graph")

        if graph is None:
            raise ValueError("No NetworkX graph in spatial cache.")

        # Detect coordinate system
        lon_col = next((c for c in ["lon","longitude","x"] if c in pois.columns), None)
        lat_col = next((c for c in ["lat","latitude","y"]  if c in pois.columns), None)
        if lon_col is None or lat_col is None:
            raise ValueError(f"Cannot find coordinate columns. Available: {list(pois.columns)}")

        # Detect if projected (EPSG:3857) or geographic (EPSG:4326)
        sample_x = float(pois[lon_col].iloc[0])
        is_projected = abs(sample_x) > 180.0
        print(f"  Coordinates: {'EPSG:3857 (projected)' if is_projected else 'WGS84 (geographic)'}")

        # POI coordinates -- vectorised, replaces iterrows()
        _poi_id_arr  = pois["poi_id"].to_numpy(dtype=np.int64)
        _poi_lon_arr = pois[lon_col].to_numpy(dtype=np.float64)
        _poi_lat_arr = pois[lat_col].to_numpy(dtype=np.float64)
        poi_coords   = {
            int(pid): (float(lon), float(lat))
            for pid, lon, lat in zip(_poi_id_arr, _poi_lon_arr, _poi_lat_arr)
        }
        print(f"  POI coords: {len(poi_coords):,}")

        # POI -> road node mapping -- vectorised, replaces iterrows()
        snap_col = next(
            (c for c in ["distance_to_road_meters", "snap_distance", "distance"]
             if c in nodes_df.columns), None
        )
        poi_to_road  = {}
        _pids   = nodes_df["poi_id"].to_numpy(dtype=np.int64)
        _rids   = nodes_df["road_id"].to_numpy(dtype=np.int64)
        _snaps  = (nodes_df[snap_col].to_numpy(dtype=np.float64)
                   if snap_col else np.zeros(len(nodes_df), dtype=np.float64))
        for _pid, _rid, _snap in zip(_pids, _rids, _snaps):
            poi_to_road[int(_pid)] = {
                "road_node":     int(_rid),
                "snap_distance": float(_snap),
            }
        print(f"  POI->road: {len(poi_to_road):,} entries")

        # Road node coordinates
        node_coords = {}
        for node_id, attr in graph.nodes(data=True):
            x = attr.get("x", attr.get("lon", None))
            y = attr.get("y", attr.get("lat", None))
            if x is not None and y is not None:
                node_coords[int(node_id)] = (float(x), float(y))

        if len(node_coords) == 0:
            # Derive approximate coords from POI snap positions
            lon_col = next((c for c in ["lon","longitude","x"] if c in pois.columns), None)
            lat_col = next((c for c in ["lat","latitude","y"]  if c in pois.columns), None)
            if lon_col and lat_col and "poi_id" in nodes_df.columns and "road_id" in nodes_df.columns:
                _pid_arr = nodes_df["poi_id"].to_numpy(dtype=np.int64)
                _rid_arr = nodes_df["road_id"].to_numpy(dtype=np.int64)
                _lon_arr = pois.set_index("poi_id")[lon_col].to_dict()
                _lat_arr = pois.set_index("poi_id")[lat_col].to_dict()
                for pid, rid in zip(_pid_arr, _rid_arr):
                    if rid not in node_coords and int(pid) in _lon_arr:
                        node_coords[int(rid)] = (float(_lon_arr[int(pid)]),
                                                  float(_lat_arr[int(pid)]))
            print(f"  Road node coords: {len(node_coords):,} (derived from POI snap positions)")
        else:
            print(f"  Road node coords: {len(node_coords):,}")

        # Build mode-specific graphs
        print("  Building mode graphs...")
        graphs = {
            "car":     _build_car_graph(graph),
            "bicycle": _build_bicycle_graph(graph),
            "foot":    _build_foot_graph(graph),
        }

        # Load existing path cache. A cache written by an older version can
        # reference module paths that no longer exist, which would otherwise
        # abort the whole run on an unpickling error -- the cache is purely a
        # routing speed-up, so fall back to rebuilding it instead.
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                cache = cached.get("cache", {})
                mb = os.path.getsize(cache_file) / 1024 / 1024
                print(f"  Loaded path cache: {len(cache):,} entries ({mb:.1f} MB)")
            except Exception as e:
                print(f"  [WARNING] Could not read path cache ({type(e).__name__}: {e});"
                      f" rebuilding it from scratch.")
                cache = {}

        return cls(
            graphs          = graphs,
            poi_coords      = poi_coords,
            poi_to_road     = poi_to_road,
            node_coords     = node_coords,
            is_projected    = is_projected,
            transport_nodes = data.get("transport_nodes"),
            pt_connectivity = data.get("pt_connectivity"),
            cache           = cache,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str = CACHE_FILE) -> None:
        if not self._cache:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"cache": self._cache}, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"[PathGenerator] Saved {len(self._cache):,} paths -> {path} ({mb:.1f} MB)")
        self._new_since_save = 0

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def get_path(
        self,
        origin_poi: int,
        dest_poi:   int,
        mode:       str = "car",
    ) -> Optional[PathResult]:
        """
        Return PathResult(time_minutes, linestring, mode, distance_m).
        Runs Dijkstra on first call, O(1) cache lookup after.
        Returns None if no path exists for this mode.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

        if origin_poi == dest_poi:
            return PathResult(0.0, [], mode, 0.0)

        key = (int(origin_poi), int(dest_poi), mode)

        if key in self._cache:
            return self._cache[key]

        result = self._compute_path(origin_poi, dest_poi, mode)
        self._cache[key] = result
        self._new_since_save += 1

        if self._new_since_save >= AUTOSAVE_N:
            self.save()

        return result

    def get_travel_time(
        self,
        origin_poi: int,
        dest_poi:   int,
        mode:       str = "car",
    ) -> float:
        """
        Return travel time in minutes. Returns 999.0 if unreachable.
        Fast path during training -- avoids storing geometry if not needed
        but still uses the same cache as get_path().
        """
        if origin_poi == dest_poi:
            return 0.0
        result = self.get_path(origin_poi, dest_poi, mode)
        return result.time_minutes if result is not None else 999.0

    def get_air_distance_km(self, poi_a: int, poi_b: int) -> float:
        """
        Haversine air distance in km between two POIs.
        No graph needed -- pure coordinate calculation.
        Used for radius-of-gyration analysis and baseline comparisons.
        """
        if poi_a == poi_b:
            return 0.0
        ca = self._poi_coords.get(int(poi_a))
        cb = self._poi_coords.get(int(poi_b))
        if ca is None or cb is None:
            return 0.0
        lon_a, lat_a = _to_wgs84(ca[0], ca[1], self._is_projected)
        lon_b, lat_b = _to_wgs84(cb[0], cb[1], self._is_projected)
        return _haversine_m(lon_a, lat_a, lon_b, lat_b) / 1000.0

    def get_pt_travel_time(
        self,
        origin_poi: int,
        dest_poi:   int,
        max_walk_to_stop_m: float = 600.0,
    ) -> Optional[float]:
        """
        Estimate PT travel time in minutes using formula:
            walk_to_stop + wait + in_vehicle + walk_from_stop

        Returns None if no PT-connected stop pair exists within walking
        distance of both origin and destination.

        Constants:
            PT_AVG_SPEED_KMH  = 20.0 km/h  (bus/tram incl. dwell time)
            PT_AVG_WAIT_MIN   = 7.5 min     (15 min headway, random arrival)
            PT_WALK_SPEED_KMH = 5.0 km/h
        """
        if not self._stop_coords or not self._pt_conn_index:
            return None

        origin_coord = self._poi_coords.get(int(origin_poi))
        dest_coord   = self._poi_coords.get(int(dest_poi))
        if origin_coord is None or dest_coord is None:
            return None

        # Convert to WGS84 if needed
        olon, olat = _to_wgs84(origin_coord[0], origin_coord[1], self._is_projected)
        dlon, dlat = _to_wgs84(dest_coord[0],   dest_coord[1],   self._is_projected)

        # Find stops within walking distance of origin
        origin_stops = []
        for sid, (slon, slat) in self._stop_coords.items():
            d = _haversine_m(olon, olat, slon, slat)
            if d <= max_walk_to_stop_m:
                origin_stops.append((sid, d))

        if not origin_stops:
            return None

        # Find stops within walking distance of destination
        dest_stops = []
        for sid, (slon, slat) in self._stop_coords.items():
            d = _haversine_m(dlon, dlat, slon, slat)
            if d <= max_walk_to_stop_m:
                dest_stops.append((sid, d))

        if not dest_stops:
            return None

        # Find best PT-connected pair
        best_time = None
        dest_stop_set = {sid: d for sid, d in dest_stops}

        for o_stop, walk_to_m in origin_stops:
            connections = self._pt_conn_index.get(o_stop, {})
            for d_stop, route_dist_m in connections.items():
                if d_stop not in dest_stop_set:
                    continue
                walk_from_m = dest_stop_set[d_stop]

                walk_to_min   = (walk_to_m   / 1000.0) / PT_WALK_SPEED_KMH * 60.0
                walk_from_min = (walk_from_m / 1000.0) / PT_WALK_SPEED_KMH * 60.0
                vehicle_min   = (route_dist_m / 1000.0) / PT_AVG_SPEED_KMH * 60.0
                total         = walk_to_min + PT_AVG_WAIT_MIN + vehicle_min + walk_from_min

                if best_time is None or total < best_time:
                    best_time = total

        return best_time

    # ------------------------------------------------------------------
    # Linestring compatibility (for sim.py / visualisation)
    # ------------------------------------------------------------------

    def get_linestring_id(
        self,
        origin_poi: int,
        dest_poi:   int,
        mode:       str = "car",
    ) -> Optional[int]:
        """
        Legacy-compatible wrapper. Returns a synthetic int ID.
        The ID encodes the cache key -- use get_linestring() to retrieve.
        """
        result = self.get_path(origin_poi, dest_poi, mode)
        if result is None or not result.linestring:
            return None
        # Stable integer key from the tuple
        return hash((int(origin_poi), int(dest_poi), mode)) & 0x7FFFFFFF

    def get_linestring(
        self,
        origin_poi: int,
        dest_poi:   int,
        mode:       str = "car",
    ) -> Optional[Linestring]:
        """Return linestring geometry for a POI pair and mode."""
        result = self.get_path(origin_poi, dest_poi, mode)
        return result.linestring if result is not None else None

    # ------------------------------------------------------------------
    # Internal path computation
    # ------------------------------------------------------------------

    def _compute_path(
        self,
        origin_poi: int,
        dest_poi:   int,
        mode:       str,
    ) -> Optional[PathResult]:
        """
        Single-graph routing using mode-specific edge weights.
        All edges exist in every graph -- penalty speeds (0.001 km/h on
        motorways for foot, 5 km/h on pedestrian for car) ensure Dijkstra
        naturally avoids inappropriate roads without needing mode fallbacks.
        """
        orig_info = self._poi_to_road.get(int(origin_poi))
        dest_info = self._poi_to_road.get(int(dest_poi))
        if orig_info is None or dest_info is None:
            return None

        origin_node = orig_info["road_node"]
        dest_node   = dest_info["road_node"]
        snap_o      = orig_info.get("snap_distance", 0.0)
        snap_d      = dest_info.get("snap_distance", 0.0)

        g = self._graphs[mode]

        if origin_node == dest_node:
            snap_speed_ms = WALK_SPEED_KMH / 3.6
            time_minutes  = (snap_o + snap_d) / snap_speed_ms / 60.0
            return PathResult(time_minutes, [], mode, snap_o + snap_d)

        try:
            cost, node_sequence = nx.bidirectional_dijkstra(
                g, origin_node, dest_node, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
            return None

        # cost is in seconds -- add snap distances at walk speed
        snap_speed_ms = WALK_SPEED_KMH / 3.6
        time_minutes  = (float(cost) + (snap_o + snap_d) / snap_speed_ms) / 60.0

        linestring = self._nodes_to_linestring(node_sequence, snap_o, snap_d)
        distance_m = linestring[-1][2] if linestring else 0.0

        return PathResult(
            time_minutes = float(time_minutes),
            linestring   = linestring,
            mode         = mode,
            distance_m   = float(distance_m),
        )

    def _nodes_to_linestring(
        self,
        node_sequence: list,
        snap_origin_m: float = 0.0,
        snap_dest_m:   float = 0.0,
    ) -> Linestring:
        result   = []
        cum_dist = snap_origin_m

        for i, node_id in enumerate(node_sequence):
            x, y     = self._node_coords.get(int(node_id), (0.0, 0.0))
            lon, lat = _to_wgs84(x, y, self._is_projected)

            if i > 0:
                px, py           = self._node_coords.get(int(node_sequence[i-1]), (x, y))
                prev_lon, prev_lat = _to_wgs84(px, py, self._is_projected)
                cum_dist        += _haversine_m(prev_lon, prev_lat, lon, lat)

            result.append((float(lon), float(lat), float(cum_dist)))

        if result and snap_dest_m > 0:
            lon, lat, d = result[-1]
            result[-1]  = (lon, lat, d + snap_dest_m)

        return result

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def prefetch(
        self,
        pairs: List[Tuple[int, int]],
        mode:  str = "car",
        show_progress: bool = True,
    ) -> None:
        """Pre-warm cache for known pairs (e.g. all home->work routes)."""
        missing = [(o, d) for o, d in pairs
                   if (int(o), int(d), mode) not in self._cache]
        if not missing:
            print(f"[prefetch] All {len(pairs):,} pairs already cached.")
            return

        print(f"[prefetch] Computing {len(missing):,} missing paths (mode={mode})...")
        t0 = time.time(); fails = 0

        for i, (o, d) in enumerate(missing, 1):
            if self.get_path(o, d, mode) is None:
                fails += 1
            if show_progress and i % 500 == 0:
                elapsed = time.time() - t0
                rate    = i / elapsed
                eta     = (len(missing) - i) / rate if rate > 0 else 0
                print(f"  {i:,}/{len(missing):,}  {rate:.0f}/s  ETA {eta:.0f}s  fails={fails}")

        self.save()
        elapsed = time.time() - t0
        print(f"[prefetch] Done: {len(missing):,} in {elapsed:.1f}s, {fails} unreachable")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def available(self) -> bool:
        return bool(self._graphs) and len(self._node_coords) > 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, random

    from config import SPATIAL_CACHE_FILE as SPATIAL_CACHE
    if not os.path.exists(SPATIAL_CACHE):
        print(f"ERROR: {SPATIAL_CACHE} not found.")
        sys.exit(1)

    print("=" * 60)
    print("PATH LIBRARY -- SMOKE TEST")
    print("=" * 60)

    lib = PathGenerator.load(SPATIAL_CACHE)
    if not lib.available:
        print("ERROR: PathGenerator not available.")
        sys.exit(1)

    poi_ids = list(lib._poi_coords.keys())
    if len(poi_ids) < 2:
        print("ERROR: Need at least 2 POIs.")
        sys.exit(1)

    poi_a, poi_b = poi_ids[0], poi_ids[100]
    print(f"\nTest pair: POI {poi_a} -> POI {poi_b}")
    print(f"Air distance: {lib.get_air_distance_km(poi_a, poi_b):.2f} km")

    for mode in ["car", "bicycle", "foot"]:
        t0 = time.time()
        r  = lib.get_path(poi_a, poi_b, mode)
        ms = (time.time() - t0) * 1000
        if r:
            print(f"  [{mode:8}] {r.time_minutes:.1f} min  "
                  f"{r.distance_m/1000:.2f} km  "
                  f"{len(r.linestring)} pts  ({ms:.1f} ms first query)")
        else:
            print(f"  [{mode:8}] No path found ({ms:.1f} ms)")

    # Cache hit test
    t0 = time.time()
    lib.get_path(poi_a, poi_b, "car")
    ms = (time.time() - t0) * 1000
    print(f"\n  [car cache hit] {ms:.3f} ms")

    # PT estimate
    pt = lib.get_pt_travel_time(poi_a, poi_b)
    if pt:
        print(f"  [PT estimate]   {pt:.1f} min")
    else:
        print(f"  [PT estimate]   No PT connection found")

    print("\n[OK] Done")