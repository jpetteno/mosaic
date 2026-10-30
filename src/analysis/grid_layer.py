from __future__ import annotations

"""
GRID LAYER
==========
A post-hoc spatial abstraction layer that projects simulation outputs
onto a fixed 500m x 500m grid anchored to the city's OSM bounding box.

The simulation itself is never touched. This module operates exclusively
on already-computed DataFrames (activities_df, travels_df) and adds
grid-space columns alongside the original coordinate columns.

Grid coordinate system
----------------------
- Input  : WGS84 (EPSG:4326)  lon/lat degrees
- Working: EPSG:32632 (UTM Zone 32N)  metres  -- accurate metric cells for central Europe
- cell_id: "{row}_{col}" string, row=0 at north, col=0 at west

City bbox (WGS84, min_lon, min_lat, max_lon, max_lat) is resolved PER CITY by
`resolve_city_bbox()`, from `data/raw/<city>/config.json` or, failing that, from
the extent of the city's own filtered spatial cache. `DEFAULT_CITY_BBOX` below
is Bonn's and is retained only for callers that ask for it explicitly -- it is
NOT a fallback, because a wrong bbox does not fail: `_utm_to_rowcol` clamps, so
every point of a different city silently collapses into one corner cell.
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p



import json
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bonn's OSM query bounding box. Kept for explicit callers only -- see the
# module docstring. Order: min_lon, min_lat, max_lon, max_lat (WGS84/EPSG:4326)
DEFAULT_CITY_BBOX: Tuple[float, float, float, float] = (
    7.0200, 50.6800, 7.1500, 50.7700
)

CELL_SIZE_M: int = 500  # metres

# Margin added when a bbox has to be derived from a POI extent, so that points
# exactly on the extent do not land in the outermost half-cell.
_BBOX_MARGIN_DEG: float = 0.005


def resolve_city_bbox(city: str = None) -> Tuple[float, float, float, float]:
    """
    Return the WGS84 bbox (min_lon, min_lat, max_lon, max_lat) of `city`.

    `city` defaults to the active city (config.MAP_FOLDER, i.e. the MOBILITY_CITY
    env var). Resolution order:
      1. data/raw/<city>/config.json -> "bbox"
      2. the extent of data/formatted/<city>/spatial_cache_filtered.pkl
    Raises if neither is available. There is deliberately no Bonn fallback: an
    unnoticed wrong bbox produces a single clamped cell, not an error.
    """
    import json as _json
    import os as _os
    import pickle as _pickle

    here = _os.path.dirname(_os.path.abspath(__file__))
    root = _os.path.dirname(_os.path.dirname(here))      # <repo>/src/analysis -> <repo>

    if city is None:
        try:
            import config as _config
            city = _config.MAP_FOLDER
        except Exception:
            city = _os.environ.get("MOBILITY_CITY")
    if not city:
        raise RuntimeError(
            "resolve_city_bbox: no city given and the active city could not be "
            "determined from config.MAP_FOLDER / MOBILITY_CITY."
        )

    cfg_path = _os.path.join(root, "data", "raw", city, "config.json")
    if _os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            bbox = _json.load(fh).get("bbox")
        if bbox and len(bbox) == 4:
            print(f"[CityGrid] bbox for '{city}' from data/raw/{city}/config.json")
            return tuple(float(v) for v in bbox)

    cache_path = _os.path.join(root, "data", "formatted", city,
                               "spatial_cache_filtered.pkl")
    if _os.path.isfile(cache_path):
        with open(cache_path, "rb") as fh:
            pois = _pickle.load(fh)["pois"]
        m = _BBOX_MARGIN_DEG
        bbox = (float(pois["x"].min()) - m, float(pois["y"].min()) - m,
                float(pois["x"].max()) + m, float(pois["y"].max()) + m)
        print(f"[CityGrid] bbox for '{city}' derived from the spatial cache extent "
              f"(+/-{m} deg margin)")
        return bbox

    raise FileNotFoundError(
        f"resolve_city_bbox: no bbox for city '{city}'. Expected either "
        f"{cfg_path} with a 'bbox' key, or {cache_path}."
    )


# ---------------------------------------------------------------------------
# Projectors (module-level singletons -- cheap to create, reused across calls)
# ---------------------------------------------------------------------------

_wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
_utm_to_wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)


def _to_utm(lon: float, lat: float) -> Tuple[float, float]:
    """WGS84 (lon, lat) -> UTM Zone 32N (easting, northing) in metres."""
    return _wgs84_to_utm.transform(lon, lat)


def _to_wgs84(easting: float, northing: float) -> Tuple[float, float]:
    """UTM Zone 32N -> WGS84 (lon, lat)."""
    return _utm_to_wgs84.transform(easting, northing)


# ---------------------------------------------------------------------------
# CityGrid
# ---------------------------------------------------------------------------

class CityGrid:
    """
    Fixed 500 m x 500 m grid over the city's OSM bounding box.

    Attributes
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    origin_e, origin_n : float
        UTM coordinates of the top-left (NW) corner of cell (0, 0).
    cell_size : int
        Cell edge length in metres (default 500, see CELL_SIZE_M).
    """

    def __init__(
        self,
        bbox_wgs84: Tuple[float, float, float, float] = None,
        cell_size_m: int = CELL_SIZE_M,
        city: str = None,
    ):
        self.cell_size = cell_size_m
        if bbox_wgs84 is None:
            bbox_wgs84 = resolve_city_bbox(city)
        self._bbox_wgs84 = bbox_wgs84
        self.city = city
        # Points outside the bbox are clamped to the boundary (see
        # _utm_to_rowcol), which is how a wrong-city bbox used to pass silently.
        # Count them so callers can check coverage instead of trusting it.
        self.n_clamped = 0
        self.n_assigned = 0

        min_lon, min_lat, max_lon, max_lat = bbox_wgs84

        # Project all four corners to UTM
        e_min, n_min = _to_utm(min_lon, min_lat)
        e_max, n_max = _to_utm(max_lon, max_lat)

        # Snap outward to cell_size boundaries so no POI falls outside
        self.origin_e = math.floor(e_min / cell_size_m) * cell_size_m  # west edge
        self.origin_n = math.ceil (n_max / cell_size_m) * cell_size_m  # north edge (row 0)

        end_e = math.ceil (e_max / cell_size_m) * cell_size_m
        end_n = math.floor(n_min / cell_size_m) * cell_size_m

        self.n_cols = int(round((end_e - self.origin_e) / cell_size_m))
        self.n_rows = int(round((self.origin_n - end_n) / cell_size_m))

        print(
            f"[CityGrid] bbox projected to UTM Zone 32N\n"
            f"  origin (NW corner): E={self.origin_e:.0f} m, N={self.origin_n:.0f} m\n"
            f"  grid size        : {self.n_rows} rows x {self.n_cols} cols "
            f"= {self.n_rows * self.n_cols:,} cells"
        )

    # ------------------------------------------------------------------
    # Core coordinate <-> cell conversions
    # ------------------------------------------------------------------

    def _utm_to_rowcol(self, easting: float, northing: float) -> Tuple[int, int]:
        """UTM metres -> (row, col) integers.  Clamps to grid boundary."""
        col = int((easting  - self.origin_e) / self.cell_size)
        row = int((self.origin_n - northing) / self.cell_size)
        clamped_row = max(0, min(row, self.n_rows - 1))
        clamped_col = max(0, min(col, self.n_cols - 1))
        self.n_assigned += 1
        if clamped_row != row or clamped_col != col:
            self.n_clamped += 1
        return clamped_row, clamped_col

    def coverage_report(self, label: str = "", max_clamped_frac: float = 0.01) -> float:
        """
        Print, and return, the fraction of assigned points that fell outside the
        bbox and had to be clamped to the boundary.

        Raises above `max_clamped_frac`: a large clamped share means the bbox
        does not belong to the data (the classic case being a hardcoded Bonn
        bbox applied to another city, which collapses every point into one
        corner cell without any error).
        """
        if self.n_assigned == 0:
            print(f"[CityGrid] coverage{' ' + label if label else ''}: no points assigned")
            return 0.0
        frac = self.n_clamped / self.n_assigned
        print(f"[CityGrid] coverage{' ' + label if label else ''}: "
              f"{self.n_clamped:,}/{self.n_assigned:,} points clamped to the grid "
              f"boundary ({100 * frac:.2f}%)")
        if frac > max_clamped_frac:
            raise ValueError(
                f"CityGrid: {100 * frac:.1f}% of points fall outside the grid bbox "
                f"{tuple(round(v, 4) for v in self._bbox_wgs84)}"
                f"{' for city ' + repr(self.city) if self.city else ''}. "
                f"Clamping would collapse them onto the boundary. This almost "
                f"always means the bbox is another city's."
            )
        return frac

    def coord_to_cell(self, lat: float, lon: float) -> str:
        """
        WGS84 (lat, lon) -> cell_id string "row_col".

        Returns "invalid" for NaN inputs rather than raising.
        """
        if math.isnan(lat) or math.isnan(lon):
            return "invalid"
        easting, northing = _to_utm(lon, lat)
        row, col = self._utm_to_rowcol(easting, northing)
        return f"{row}_{col}"

    def cell_centroid(self, cell_id: str) -> Tuple[float, float]:
        """
        cell_id "row_col" -> centroid in WGS84 (lat, lon).

        Returns (NaN, NaN) for "invalid" cell_id.
        """
        if cell_id == "invalid":
            return (float("nan"), float("nan"))
        row, col = map(int, cell_id.split("_"))
        e = self.origin_e + (col + 0.5) * self.cell_size
        n = self.origin_n - (row + 0.5) * self.cell_size
        lon, lat = _to_wgs84(e, n)
        return lat, lon

    def cell_bounds_wgs84(self, cell_id: str) -> dict:
        """
        Return the four WGS84 corner coordinates of a cell.
        Useful for drawing cell polygons in Folium.
        """
        if cell_id == "invalid":
            return {}
        row, col = map(int, cell_id.split("_"))
        e0 = self.origin_e + col       * self.cell_size
        e1 = self.origin_e + (col + 1) * self.cell_size
        n0 = self.origin_n - row       * self.cell_size  # north edge
        n1 = self.origin_n - (row + 1) * self.cell_size  # south edge
        corners = [_to_wgs84(e, n) for e, n in [(e0, n0), (e1, n0), (e1, n1), (e0, n1)]]
        return {
            "sw": (corners[3][1], corners[3][0]),   # (lat, lon)
            "ne": (corners[1][1], corners[1][0]),
            "corners_latlon": [(lon, lat) for lon, lat in corners],
        }

    # ------------------------------------------------------------------
    # Path tracing
    # ------------------------------------------------------------------

    def trace_path(self, points_lonlat: List[Tuple[float, float]]) -> List[str]:
        """
        Enumerate all unique cells touched by a polyline.

        Parameters
        ----------
        points_lonlat : list of (lon, lat) pairs
            Either a road-network linestring from PathGenerator
            or a two-point straight line [(o_lon, o_lat), (d_lon, d_lat)].

        Returns
        -------
        List of cell_id strings in traversal order, with consecutive
        duplicates removed.  An empty list is returned for empty input.

        Implementation
        --------------
        For each segment we densely sample at sub-cell resolution
        (every cell_size/2 metres along the segment) and collect unique
        cells in order.  This correctly handles segments that are much
        longer than 500 m without missing intermediate cells.
        """
        if not points_lonlat or len(points_lonlat) < 1:
            return []

        cells: List[str] = []
        step = self.cell_size / 2.0  # sample every 50 m -- sub-cell resolution

        # Project all points to UTM once
        utm_points = [_to_utm(lon, lat) for lon, lat in points_lonlat]

        for i in range(len(utm_points)):
            e0, n0 = utm_points[i]
            # Single point edge case
            if i == 0:
                row, col = self._utm_to_rowcol(e0, n0)
                _append_cell(cells, f"{row}_{col}")
                continue

            e1, n1 = utm_points[i - 1]
            seg_len = math.hypot(e0 - e1, n0 - n1)

            if seg_len == 0:
                row, col = self._utm_to_rowcol(e0, n0)
                _append_cell(cells, f"{row}_{col}")
                continue

            n_steps = max(1, int(math.ceil(seg_len / step)))
            for k in range(n_steps + 1):
                t = k / n_steps
                e = e1 + t * (e0 - e1)
                n = n1 + t * (n0 - n1)
                row, col = self._utm_to_rowcol(e, n)
                _append_cell(cells, f"{row}_{col}")

        return cells

    # ------------------------------------------------------------------
    # DataFrame snapping
    # ------------------------------------------------------------------

    def snap_activities_df(self, activities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'cell_id' column to activities_df.

        Works on TRAVEL rows too (those have no lat/lon -- cell_id = "invalid").
        Original DataFrame is not modified; a copy with the new column is returned.
        """
        df = activities_df.copy()

        if "lat" not in df.columns or "lon" not in df.columns:
            print("[CityGrid] snap_activities_df: no lat/lon columns found -- all cells = 'invalid'")
            df["cell_id"] = "invalid"
            return df

        df["cell_id"] = df.apply(
            lambda row: self.coord_to_cell(
                float(row["lat"]) if pd.notna(row["lat"]) else float("nan"),
                float(row["lon"]) if pd.notna(row["lon"]) else float("nan"),
            ),
            axis=1,
        )
        n_invalid = (df["cell_id"] == "invalid").sum()
        if n_invalid > 0:
            print(f"[CityGrid] snap_activities_df: {n_invalid} rows mapped to 'invalid' (NaN coords or TRAVEL rows)")
        # Guard the wrong-city-bbox case while the coordinates are still here to
        # check; after this they are dropped and only cell_id is written out.
        self.coverage_report(label="(activities)")
        n_cells = df.loc[df["cell_id"] != "invalid", "cell_id"].nunique()
        print(f"[CityGrid] snap_activities_df: {n_cells:,} distinct cells occupied")
        if n_cells <= 1 and len(df) > 100:
            raise ValueError(
                f"CityGrid: {len(df):,} activities collapsed into {n_cells} cell(s). "
                f"The grid bbox {tuple(round(v, 4) for v in self._bbox_wgs84)} does "
                f"not cover this city's coordinates."
            )
        return df

    def snap_travels_df(
        self,
        travels_df: pd.DataFrame,
        path_generator=None,
    ) -> pd.DataFrame:
        """
        Add spatial grid columns to travels_df.

        New columns
        -----------
        origin_cell  : cell_id of trip origin
        dest_cell    : cell_id of trip destination
        path_cells   : JSON-encoded list of cell_ids along the route
        path_source  : "road_network" | "straight_line" -- mirrors what the
                       Folium map actually draws, so the cell trace is always
                       consistent with the displayed geometry.

        Parameters
        ----------
        path_generator : PathGenerator instance or None
            If provided, road-network linestrings are used for cell tracing
            (same logic as the Folium trajectory map).  Falls back to
            straight-line when no linestring is available.
        """
        df = travels_df.copy()

        origin_cells = []
        dest_cells   = []
        path_cells_list = []
        path_sources = []

        for _, row in df.iterrows():
            o_lat = _safe_float(row.get("origin_lat"))
            o_lon = _safe_float(row.get("origin_lon"))
            d_lat = _safe_float(row.get("dest_lat"))
            d_lon = _safe_float(row.get("dest_lon"))

            # Origin and destination cells
            o_cell = self.coord_to_cell(o_lat, o_lon) if _valid(o_lat, o_lon) else "invalid"
            d_cell = self.coord_to_cell(d_lat, d_lon) if _valid(d_lat, d_lon) else "invalid"
            origin_cells.append(o_cell)
            dest_cells.append(d_cell)

            # Path tracing -- road network first, straight line fallback
            ls = None
            if path_generator is not None:
                ls_id = row.get("linestring_id")
                mode  = row.get("mode", "car")
                o_id  = row.get("origin_id")
                d_id  = row.get("dest_id")
                if ls_id is not None and pd.notna(ls_id) and pd.notna(o_id) and pd.notna(d_id):
                    try:
                        ls = path_generator.get_linestring(int(o_id), int(d_id), mode)
                    except Exception:
                        ls = None

            if ls and len(ls) >= 2:
                # PathGenerator returns (lon, lat, dist) tuples
                points = [(pt[0], pt[1]) for pt in ls]
                cells  = self.trace_path(points)
                source = "road_network"
            elif _valid(o_lat, o_lon) and _valid(d_lat, d_lon):
                # Straight-line fallback: trace between the two endpoint coordinates
                cells  = self.trace_path([(o_lon, o_lat), (d_lon, d_lat)])
                source = "straight_line"
            else:
                cells  = []
                source = "no_geometry"

            path_cells_list.append(json.dumps(cells))
            path_sources.append(source)

        df["origin_cell"] = origin_cells
        df["dest_cell"]   = dest_cells
        df["path_cells"]  = path_cells_list
        df["path_source"] = path_sources

        # Summary stats
        n_road   = path_sources.count("road_network")
        n_sl     = path_sources.count("straight_line")
        n_none   = path_sources.count("no_geometry")
        print(
            f"[CityGrid] snap_travels_df: {len(df)} trips snapped\n"
            f"  road_network: {n_road}  |  straight_line: {n_sl}  |  no_geometry: {n_none}"
        )
        return df

    # ------------------------------------------------------------------
    # Aggregate grid statistics  (used by grid-adapted analysis functions)
    # ------------------------------------------------------------------

    def visit_counts(self, activities_df_snapped: pd.DataFrame) -> pd.Series:
        """
        Return a Series mapping cell_id -> total visit count,
        excluding TRAVEL rows and 'invalid' cells.
        """
        df = activities_df_snapped
        mask = (df["activity_type"] != "TRAVEL") & (df["cell_id"] != "invalid")
        return df.loc[mask, "cell_id"].value_counts()

    def flow_matrix(self, travels_df_snapped: pd.DataFrame) -> pd.DataFrame:
        """
        Return an O-D flow DataFrame with columns [origin_cell, dest_cell, trips].
        Excludes rows where either cell is 'invalid'.
        """
        df = travels_df_snapped[
            (travels_df_snapped["origin_cell"] != "invalid") &
            (travels_df_snapped["dest_cell"]   != "invalid")
        ]
        flows = (
            df.groupby(["origin_cell", "dest_cell"])
              .size()
              .reset_index(name="trips")
              .sort_values("trips", ascending=False)
        )
        return flows

    def cell_centroids_for_cells(self, cell_ids) -> pd.DataFrame:
        """
        Given an iterable of cell_id strings, return a DataFrame with
        columns [cell_id, lat, lon] for plotting centroid markers.
        """
        records = []
        for cid in cell_ids:
            if cid == "invalid":
                continue
            lat, lon = self.cell_centroid(cid)
            records.append({"cell_id": cid, "lat": lat, "lon": lon})
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CityGrid(cell_size={self.cell_size}m, "
            f"{self.n_rows}x{self.n_cols} cells, "
            f"origin_E={self.origin_e:.0f}, origin_N={self.origin_n:.0f})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_cell(cells: List[str], cell_id: str) -> None:
    """Append cell_id only if it differs from the last entry (dedup consecutive)."""
    if not cells or cells[-1] != cell_id:
        cells.append(cell_id)


def _safe_float(v) -> float:
    """Convert to float, returning NaN on failure."""
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return float("nan")


def _valid(lat: float, lon: float) -> bool:
    """True iff both coordinates are non-NaN finite numbers."""
    return (
        not math.isnan(lat) and not math.isnan(lon)
        and math.isfinite(lat) and math.isfinite(lon)
    )