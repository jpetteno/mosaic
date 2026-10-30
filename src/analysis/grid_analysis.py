"""
GRID ANALYSIS
=============
Grid-adapted analysis functions for the simulation pipeline.

This module is the grid layer's analysis counterpart to sim.py.
It operates exclusively on already-computed DataFrames -- the simulation
itself is never touched.

Called from sim.py main() when --use_grid is set:

    from grid_analysis import run_grid_pipeline
    if args.use_grid:
        run_grid_pipeline(activities_df, travels_df, path_generator, args.output)

All outputs land in {output_dir}/grid_data/ and are fully independent
of the standard sim.py outputs.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional

import folium
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grid_layer import CityGrid, DEFAULT_CITY_BBOX

# ---------------------------------------------------------------------------
# Consistent colour palette (mirrors sim.py)
# ---------------------------------------------------------------------------

ACTIVITY_COLORS = {
    'HOME':    '#1f77b4',
    'WORK':    '#ff7f0e',
    'FOOD':    '#2ca02c',
    'ERRAND':  '#d62728',
    'LEISURE': '#9467bd',
    'TRAVEL':  '#7f7f7f',
}

DEST_COLORS = {
    'HOME':    '#6ec6f0',
    'WORK':    '#ff7f0e',
    'FOOD':    '#ffe135',
    'ERRAND':  '#d62728',
    'LEISURE': '#a020f0',
}


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_grid_pipeline(
    activities_df: pd.DataFrame,
    travels_df: pd.DataFrame,
    path_generator,
    output_dir: str,
) -> None:
    """
    Full grid analysis pipeline.  Call this from sim.py main() after all
    standard outputs have been saved.

    Parameters
    ----------
    activities_df : DataFrame
        activity_log with lat/lon columns (straight from run_simulations).
    travels_df : DataFrame
        travel_log with origin_lat/lon, dest_lat/lon, linestring_id columns.
    path_generator : PathGenerator or None
        Road-network path library -- same instance used by sim.py maps.
    output_dir : str
        Root output directory (same as --output).  Grid outputs go to
        {output_dir}/grid_data/.
    """

    print("\n" + "=" * 70)
    print("GRID LAYER -- SPATIAL ABSTRACTION PIPELINE")
    print("=" * 70)

    grid_dir = os.path.join(output_dir, "grid_data")
    os.makedirs(os.path.join(grid_dir, "data"),  exist_ok=True)
    os.makedirs(os.path.join(grid_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(grid_dir, "maps"),  exist_ok=True)

    # Per-city bbox, never a hardcoded one -- see grid_layer's module docstring.
    from config import MAP_FOLDER
    grid = CityGrid(city=MAP_FOLDER)
    print(f"  Grid bbox ({MAP_FOLDER}): {[round(x,4) for x in grid._bbox_wgs84]}")

    # -- Step 1: Snap DataFrames -----------------------------------------------
    print("\n[1/4] Snapping coordinates to grid cells...")
    acts_grid  = grid.snap_activities_df(activities_df)
    travs_grid = grid.snap_travels_df(travels_df, path_generator=path_generator)

    # -- Step 2: Save data -----------------------------------------------------
    print("\n[2/4] Saving grid data tables...")
    _save_data(acts_grid, travs_grid, grid, grid_dir)

    # -- Step 3: Plots ---------------------------------------------------------
    print("\n[3/4] Creating grid plots...")
    _plot_cell_visit_frequency(acts_grid, grid, grid_dir)
    _plot_top_cells_bar(acts_grid, grid_dir)
    _plot_trip_cell_distance(travs_grid, grid_dir)
    _plot_flow_heatmap(travs_grid, grid_dir)

    # -- Step 4: Folium maps ---------------------------------------------------
    print("\n[4/4] Creating grid Folium maps...")
    _map_cell_visit_intensity(acts_grid, grid, grid_dir)
    _map_od_flow(travs_grid, grid, grid_dir)
    _map_sample_cell_trajectories(travs_grid, grid, grid_dir)

    print("\n" + "=" * 70)
    print(f"[OK] Grid pipeline complete -- outputs in {grid_dir}/")
    print("=" * 70)


# =============================================================================
# DATA SAVING
# =============================================================================

def _save_data(
    acts_grid: pd.DataFrame,
    travs_grid: pd.DataFrame,
    grid: CityGrid,
    grid_dir: str,
) -> None:

    # Full annotated logs (original columns preserved + grid columns appended).
    # Exact POI coordinates are dropped at write time, exactly as in sim.py:
    # the in-memory frames keep them so the plots and Folium maps below can
    # still be drawn, but the released spatial resolution is the grid cell.
    # Cell centroids written further down are a 500 m grid position, not a
    # building location, and are kept.
    _COORD_COLS = ("lat", "lon", "origin_lat", "origin_lon", "dest_lat", "dest_lon")
    acts_path  = os.path.join(grid_dir, "data", "activity_log_grid.csv")
    travs_path = os.path.join(grid_dir, "data", "travel_log_grid.csv")
    acts_grid.drop(columns=[c for c in _COORD_COLS if c in acts_grid.columns]).to_csv(
        acts_path, index=False
    )
    travs_grid.drop(columns=[c for c in _COORD_COLS if c in travs_grid.columns]).to_csv(
        travs_path, index=False
    )
    print(f"  [OK] activity_log_grid.csv  ({len(acts_grid):,} rows)")
    print(f"  [OK] travel_log_grid.csv    ({len(travs_grid):,} rows)")

    # Aggregate: cell visit counts
    visit_counts = grid.visit_counts(acts_grid).reset_index()
    visit_counts.columns = ["cell_id", "visit_count"]
    # Attach centroids for convenience
    centroids = visit_counts["cell_id"].apply(
        lambda cid: pd.Series(grid.cell_centroid(cid), index=["lat", "lon"])
    )
    visit_counts = pd.concat([visit_counts, centroids], axis=1)
    vc_path = os.path.join(grid_dir, "data", "cell_visit_counts.csv")
    visit_counts.to_csv(vc_path, index=False)
    print(f"  [OK] cell_visit_counts.csv  ({len(visit_counts):,} cells visited)")

    # Aggregate: O-D flow matrix
    flow_df = grid.flow_matrix(travs_grid)
    flow_path = os.path.join(grid_dir, "data", "flow_matrix.csv")
    flow_df.to_csv(flow_path, index=False)
    print(f"  [OK] flow_matrix.csv        ({len(flow_df):,} O-D pairs)")


# =============================================================================
# PLOTS
# =============================================================================

def _plot_cell_visit_frequency(
    acts_grid: pd.DataFrame,
    grid: CityGrid,
    grid_dir: str,
) -> None:
    """
    Spatial heatmap: 2-D grid matrix coloured by visit count.
    Each pixel = one 500m cell.  Colour intensity = log(visits+1).
    """
    print("  1/4: Cell visit frequency spatial map...")

    visit_counts = grid.visit_counts(acts_grid)
    if len(visit_counts) == 0:
        print("       No visited cells, skipping.")
        return

    # Build 2-D array
    matrix = np.zeros((grid.n_rows, grid.n_cols), dtype=float)
    for cell_id, count in visit_counts.items():
        try:
            row, col = map(int, cell_id.split("_"))
            matrix[row, col] = count
        except (ValueError, IndexError):
            continue

    # Log-scale for visibility (many cells will have low counts)
    log_matrix = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(12, 14))
    im = ax.imshow(
        log_matrix,
        origin="upper",
        cmap="YlOrRd",
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log(visit count + 1)", fontsize=10)

    ax.set_title(
        "Cell Visit Frequency -- 500m Grid\n(colour = log-scaled visit count)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Grid column (W -> E)", fontsize=11)
    ax.set_ylabel("Grid row (N -> S)", fontsize=11)

    n_visited = (matrix > 0).sum()
    n_total   = grid.n_rows * grid.n_cols
    ax.text(
        0.01, 0.01,
        f"Visited cells: {n_visited:,} / {n_total:,}  ({n_visited/n_total*100:.1f}%)",
        transform=ax.transAxes,
        fontsize=9, color="black",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
    )

    plt.tight_layout()
    out = os.path.join(grid_dir, "plots", "01_cell_visit_frequency.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"     Saved {out}")


def _plot_top_cells_bar(acts_grid: pd.DataFrame, grid_dir: str) -> None:
    """
    Horizontal bar chart of the top 30 most-visited cells.
    Cell IDs on y-axis, visit count on x-axis.
    """
    print("  2/4: Top cells bar chart...")

    non_travel = acts_grid[
        (acts_grid["activity_type"] != "TRAVEL") & (acts_grid["cell_id"] != "invalid")
    ]
    if len(non_travel) == 0:
        print("       No data, skipping.")
        return

    top = non_travel["cell_id"].value_counts().head(30)

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(
        range(len(top)),
        top.values,
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Total visits", fontsize=11)
    ax.set_title("Top 30 Most-Visited Grid Cells (500m x 500m)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate count on each bar
    for i, v in enumerate(top.values):
        ax.text(v + top.values.max() * 0.005, i, str(v), va="center", fontsize=7)

    plt.tight_layout()
    out = os.path.join(grid_dir, "plots", "02_top_cells_bar.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"     Saved {out}")


def _plot_trip_cell_distance(travs_grid: pd.DataFrame, grid_dir: str) -> None:
    """
    Distribution of trip length expressed in grid cells
    (Manhattan distance between origin_cell and dest_cell).

    This is a grid-space analogue of the km distance distribution --
    each unit = 500 m, so it's directly interpretable.
    """
    print("  3/4: Trip cell-distance distribution...")

    valid = travs_grid[
        (travs_grid["origin_cell"] != "invalid") &
        (travs_grid["dest_cell"]   != "invalid")
    ].copy()

    if len(valid) == 0:
        print("       No valid O-D pairs, skipping.")
        return

    def manhattan(row):
        try:
            r1, c1 = map(int, row["origin_cell"].split("_"))
            r2, c2 = map(int, row["dest_cell"].split("_"))
            return abs(r1 - r2) + abs(c1 - c2)
        except Exception:
            return np.nan

    valid["cell_distance"] = valid.apply(manhattan, axis=1)
    valid = valid.dropna(subset=["cell_distance"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear histogram
    ax = axes[0]
    ax.hist(valid["cell_distance"], bins=60, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(valid["cell_distance"].mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {valid['cell_distance'].mean():.1f} cells ({valid['cell_distance'].mean()*100:.0f} m)")
    ax.set_xlabel("Trip distance (500m cells, Manhattan)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Trip Cell-Distance Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Log-log (to compare with Gonzalez 2008 power-law shape)
    ax = axes[1]
    d = valid["cell_distance"].values
    d = d[d > 0]
    bins = np.logspace(0, np.log10(d.max()), 40)
    hist, edges = np.histogram(d, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    ax.loglog(centers[mask], hist[mask], "o", markersize=4, alpha=0.6, label="Data")
    # Reference power law
    ref = centers[mask] ** -1.75
    ref = ref / ref[0] * hist[mask][0]
    ax.loglog(centers[mask], ref, "g--", linewidth=2, label="Reference alpha=1.75 (Gonzalez 2008)")
    ax.set_xlabel("Distance (cells)", fontsize=11)
    ax.set_ylabel("P(d)", fontsize=11)
    ax.set_title("Log-Log Trip Cell-Distance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.suptitle("Grid-Space Trip Distance  (1 cell = 500 m)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(grid_dir, "plots", "03_trip_cell_distance.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"     Saved {out}")


def _plot_flow_heatmap(travs_grid: pd.DataFrame, grid_dir: str) -> None:
    """
    O-D flow heatmap: matrix of trip counts between the top 40 origin and
    destination cells.  Reveals dominant corridors at cell resolution.
    """
    print("  4/4: O-D flow heatmap...")

    valid = travs_grid[
        (travs_grid["origin_cell"] != "invalid") &
        (travs_grid["dest_cell"]   != "invalid")
    ]
    if len(valid) == 0:
        print("       No valid O-D pairs, skipping.")
        return

    # Top cells by involvement (origin or destination)
    all_cells = pd.concat([valid["origin_cell"], valid["dest_cell"]])
    top_cells = all_cells.value_counts().head(40).index.tolist()

    sub = valid[
        valid["origin_cell"].isin(top_cells) &
        valid["dest_cell"].isin(top_cells)
    ]

    pivot = (
        sub.groupby(["origin_cell", "dest_cell"])
           .size()
           .unstack(fill_value=0)
           .reindex(index=top_cells, columns=top_cells, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, label="Trip count")

    tick_labels = [f"({r},{c})" for cell in top_cells
                   for r, c in [tuple(map(int, cell.split("_")))]]
    ax.set_xticks(range(len(top_cells)))
    ax.set_yticks(range(len(top_cells)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6, fontfamily="monospace")
    ax.set_yticklabels(tick_labels, fontsize=6, fontfamily="monospace")
    ax.set_xlabel("Destination cell (row, col)", fontsize=10)
    ax.set_ylabel("Origin cell (row, col)", fontsize=10)
    ax.set_title("O-D Flow Heatmap -- Top 40 Cells", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(grid_dir, "plots", "04_od_flow_heatmap.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"     Saved {out}")


# =============================================================================
# FOLIUM MAPS
# =============================================================================

# Basemap -- light/white so grid cell colours read clearly
_BASEMAP = "CartoDB positron"

# Matplotlib colormap used for all cell fill colours (import once)
import matplotlib.cm as _cm
import matplotlib.colors as _mcolors

def _count_to_hex(count: float, max_count: float, cmap_name: str = "YlOrRd") -> str:
    """
    Map a visit count to a hex colour string using a log-scaled colormap.
    Returns a fully opaque hex colour -- opacity is controlled separately
    via Folium's fill_opacity parameter.
    """
    norm  = np.log1p(count) / np.log1p(max_count)   # 0..1, log-scaled
    rgba  = _cm.get_cmap(cmap_name)(norm)
    return _mcolors.to_hex(rgba)


def _draw_cell_rect(
    m: folium.Map,
    grid: CityGrid,
    cell_id: str,
    color: str,
    fill_opacity: float,
    stroke: bool = True,
    tooltip: str = "",
    popup: str = "",
) -> None:
    """
    Draw a single 500m grid cell as a filled Rectangle on a Folium map.

    Parameters
    ----------
    fill_opacity : float
        0.0 = fully transparent, 1.0 = fully opaque fill.
    stroke : bool
        Whether to draw the cell border (thin, same colour).
    """
    bounds = grid.cell_bounds_wgs84(cell_id)
    if not bounds:
        return
    sw = bounds["sw"]   # (lat, lon)
    ne = bounds["ne"]   # (lat, lon)

    folium.Rectangle(
        bounds=[sw, ne],
        color=color if stroke else "none",
        weight=0.4 if stroke else 0,
        fill=True,
        fill_color=color,
        fill_opacity=fill_opacity,
        tooltip=tooltip or None,
        popup=folium.Popup(popup, max_width=220) if popup else None,
    ).add_to(m)


def _map_cell_visit_intensity(
    acts_grid: pd.DataFrame,
    grid: CityGrid,
    grid_dir: str,
    max_cells: int = 3000,
) -> None:
    """
    Folium map: every visited cell rendered as a filled rectangle on a
    light basemap.  Fill colour is YlOrRd, log-scaled by visit count.
    Cell borders are drawn in the same colour at low weight so the grid
    structure is visible without dominating.

    If more than max_cells cells are visited, only the top max_cells by
    visit count are drawn (with a console note) to keep the HTML fast.
    """
    print("  1/3: Cell visit intensity map...")

    visit_counts = grid.visit_counts(acts_grid)
    if len(visit_counts) == 0:
        print("       No visited cells, skipping.")
        return

    df = visit_counts.reset_index()
    df.columns = ["cell_id", "count"]

    # Cap for performance
    n_total = len(df)
    if n_total > max_cells:
        print(f"       {n_total} visited cells -- drawing top {max_cells} by count")
        df = df.nlargest(max_cells, "count")

    # Compute centroids for map centre
    centroids = df["cell_id"].apply(
        lambda cid: pd.Series(grid.cell_centroid(cid), index=["lat", "lon"])
    )
    df = pd.concat([df, centroids], axis=1).dropna(subset=["lat", "lon"])

    m = folium.Map(
        location=[df["lat"].mean(), df["lon"].mean()],
        zoom_start=13,
        tiles=_BASEMAP,
    )

    max_count = df["count"].max()

    for _, row in df.iterrows():
        hex_color = _count_to_hex(row["count"], max_count)
        _draw_cell_rect(
            m, grid, row["cell_id"],
            color=hex_color,
            fill_opacity=0.65,
            stroke=True,
            tooltip=f"Cell {row['cell_id']} -- {int(row['count'])} visits",
            popup=f"Cell: {row['cell_id']}<br>Visits: {int(row['count'])}",
        )

    _add_legend(
        m,
        title="Cell Visit Intensity",
        lines=[
            "Colour scale: YlOrRd (log-scaled)",
            "Yellow = few visits -> Red = many visits",
            f"Cells shown: {len(df):,} / {n_total:,}",
            "Grid cell size: 500 m x 500 m",
        ],
    )

    out = os.path.join(grid_dir, "maps", "01_cell_visit_intensity.html")
    m.save(out)
    print(f"     Saved {out}  ({len(df):,} cells drawn)")


def _map_od_flow(
    travs_grid: pd.DataFrame,
    grid: CityGrid,
    grid_dir: str,
    top_n: int = 100,
) -> None:
    """
    Folium map: top O-D cell pairs shown as weighted lines between cell
    centroids, with the involved cells drawn as filled rectangles on a
    light basemap.  Origin cells are blue, destination cells are orange,
    flow lines scale in width and opacity with trip count.
    """
    print("  2/3: O-D flow map...")

    flow_df = grid.flow_matrix(travs_grid)
    if len(flow_df) == 0:
        print("       No flow data, skipping.")
        return

    top_flows = flow_df.head(top_n)
    max_trips = top_flows["trips"].max()

    origin_cells = set(top_flows["origin_cell"])
    dest_cells   = set(top_flows["dest_cell"])
    all_cells    = origin_cells | dest_cells
    centroid_map = {cid: grid.cell_centroid(cid) for cid in all_cells}

    lats = [v[0] for v in centroid_map.values() if not np.isnan(v[0])]
    lons = [v[1] for v in centroid_map.values() if not np.isnan(v[1])]
    if not lats:
        print("       No valid centroids, skipping.")
        return

    m = folium.Map(
        location=[np.mean(lats), np.mean(lons)],
        zoom_start=13,
        tiles=_BASEMAP,
    )

    # Draw cell rectangles first (below the flow lines)
    for cid in dest_cells - origin_cells:
        _draw_cell_rect(m, grid, cid, color="#e07b00", fill_opacity=0.35, stroke=True,
                        tooltip=f"Destination cell: {cid}")
    for cid in origin_cells - dest_cells:
        _draw_cell_rect(m, grid, cid, color="#1a6faf", fill_opacity=0.35, stroke=True,
                        tooltip=f"Origin cell: {cid}")
    for cid in origin_cells & dest_cells:
        _draw_cell_rect(m, grid, cid, color="#7b3fa0", fill_opacity=0.35, stroke=True,
                        tooltip=f"Origin + destination cell: {cid}")

    # Draw flow lines on top
    for _, row in top_flows.iterrows():
        o_lat, o_lon = centroid_map.get(row["origin_cell"], (np.nan, np.nan))
        d_lat, d_lon = centroid_map.get(row["dest_cell"],   (np.nan, np.nan))
        if np.isnan(o_lat) or np.isnan(d_lat):
            continue

        t = row["trips"] / max_trips
        folium.PolyLine(
            [[o_lat, o_lon], [d_lat, d_lon]],
            color="#333333",
            weight=1.0 + 6.0 * t,
            opacity=0.25 + 0.65 * t,
            tooltip=f"{row['origin_cell']} -> {row['dest_cell']}: {row['trips']} trips",
        ).add_to(m)

    _add_legend(
        m,
        title=f"O-D Flow -- top {top_n} pairs",
        lines=[
            "# Blue  = origin cells",
            "# Orange = destination cells",
            "# Purple = origin + destination",
            "Lines: width & opacity proportional to trip count",
            "Grid cell size: 500 m x 500 m",
        ],
    )

    out = os.path.join(grid_dir, "maps", "02_od_flow.html")
    m.save(out)
    print(f"     Saved {out}  ({len(top_flows)} flow lines, {len(all_cells)} cells drawn)")


def _map_sample_cell_trajectories(
    travs_grid: pd.DataFrame,
    grid: CityGrid,
    grid_dir: str,
    n_sample: int = 50,
) -> None:
    """
    Folium map: for a sample of trips, every cell in path_cells is drawn
    as a filled rectangle.

    Opacity rules (applied per-trip, colour = destination activity):
      - Origin cell     : fill_opacity 0.85  (opaque)
      - Destination cell: fill_opacity 0.85  (opaque)
      - Intermediate    : fill_opacity 0.20  (translucent, shows grid structure)

    A thin connecting line between cell centroids is also drawn so the
    direction of travel is clear.  Solid = road-network trace,
    dashed = straight-line fallback.
    """
    print("  3/3: Sample cell trajectory map...")

    valid = travs_grid[
        travs_grid["path_cells"].notna() &
        (travs_grid["origin_cell"] != "invalid")
    ]
    if len(valid) == 0:
        print("       No path data, skipping.")
        return

    sample = valid.sample(min(n_sample, len(valid)), random_state=42)

    # Map centre from origin cells
    lats, lons = [], []
    for _, row in sample.iterrows():
        lat, lon = grid.cell_centroid(row["origin_cell"])
        if not np.isnan(lat):
            lats.append(lat); lons.append(lon)
    if not lats:
        print("       No valid centroids, skipping.")
        return

    m = folium.Map(
        location=[np.mean(lats), np.mean(lons)],
        zoom_start=13,
        tiles=_BASEMAP,
    )

    road_count     = 0
    straight_count = 0

    for _, trip in sample.iterrows():
        try:
            cells = json.loads(trip["path_cells"])
        except (json.JSONDecodeError, TypeError):
            continue
        if len(cells) < 1:
            continue

        colour = DEST_COLORS.get(trip.get("dest_activity"), "#888888")
        source = trip.get("path_source", "straight_line")
        dest_act = trip.get("dest_activity", "?")

        # -- Draw cell rectangles ------------------------------------------
        for i, cid in enumerate(cells):
            is_origin = (i == 0)
            is_dest   = (i == len(cells) - 1)

            if is_origin or is_dest:
                fill_opacity = 0.85
                stroke       = True
                tip = f"{'Origin' if is_origin else 'Destination'} -- {dest_act}"
            else:
                fill_opacity = 0.20
                stroke       = False   # no border on intermediate cells: cleaner look
                tip = f"Path cell -- {dest_act}"

            _draw_cell_rect(
                m, grid, cid,
                color=colour,
                fill_opacity=fill_opacity,
                stroke=stroke,
                tooltip=tip,
            )

        # -- Draw centroid connector line ----------------------------------
        latlon = []
        for cid in cells:
            lat, lon = grid.cell_centroid(cid)
            if not np.isnan(lat):
                latlon.append((lat, lon))

        if len(latlon) >= 2:
            folium.PolyLine(
                latlon,
                color=colour,
                weight=1.5,
                opacity=0.55,
                dash_array=None if source == "road_network" else "6 4",
                tooltip=f"{dest_act} | {source} | {len(cells)} cells",
            ).add_to(m)

        if source == "road_network":
            road_count += 1
        else:
            straight_count += 1

    _add_legend(
        m,
        title="Cell Trajectories",
        lines=[
            *[f'<span style="color:{c}">#</span> {a}' for a, c in DEST_COLORS.items()],
            "",
            "Opaque cells = origin / destination",
            "Translucent cells = path",
            "Solid line = road-network trace",
            "Dashed line = straight-line fallback",
            "",
            f"Road traces    : {road_count}",
            f"Straight traces: {straight_count}",
            "Grid cell size : 500 m x 500 m",
        ],
    )

    out = os.path.join(grid_dir, "maps", "03_sample_cell_trajectories.html")
    m.save(out)
    print(f"     Saved {out}  (road: {road_count}, straight: {straight_count})")


# =============================================================================
# UTILITIES
# =============================================================================

def _add_legend(m: folium.Map, title: str, lines: list) -> None:
    """Add a light legend box to a Folium map (matches white basemap)."""
    rows = "".join(f"{line}<br>" for line in lines)
    html = f"""
    <div style="position:fixed;bottom:25px;left:25px;z-index:1000;
                background:rgba(255,255,255,0.92);border:1px solid #cccccc;
                border-radius:8px;padding:12px 16px;font-family:monospace;
                font-size:12px;color:#222;box-shadow:0 2px 10px rgba(0,0,0,0.15);">
        <b style="color:#111">{title}</b><br>{rows}
    </div>"""
    m.get_root().html.add_child(folium.Element(html))