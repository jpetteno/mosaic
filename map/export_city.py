"""
EXPORT CITY OSM DATA TO SPATIAL CACHE
======================================
Queries a city OSM PostgreSQL database (populated via city_map.sql)
and exports to data/spatial_cache.pkl for use by region_data.py.

Run once per city:
    python export_city.py --city bonn
    python export_city.py --city hamburg
    python export_city.py --city bamberg

Each city requires its own database (named <city>_osm by convention) and its own
output directory (created automatically under data/<city>/).

Tables used (all created by city_map.sql):
    pois             -- all POIs with poi_category, x, y
    home_nodes       -- residential buildings as poi_category='home_residential'
    transport_nodes  -- PT stops (bus, tram, train, ferry)
    network_edges    -- road graph with mode-aware weights
    network_nodes    -- POI -> road node mapping
    pt_connectivity  -- stub table (empty); kept for schema compatibility
"""

import sys as _sys, os as _os
_map  = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_map)
_src  = _os.path.join(_root, 'src')
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _map, _root, _src, _p


import argparse
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine


# =============================================================================
# CITY CONFIGURATIONS
# =============================================================================
# Database credentials are read from custom_settings.py.
# The database name is derived as <city>_osm by convention.
# Override DB_NAME_MAP below if your databases use different names.

from custom_settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD

# Map city folder name -> database name.
# By default: <map_folder>_osm (e.g. "bonn" -> "bonn_osm").
# Add entries here only if your database name differs from this convention.
DB_NAME_MAP: dict = {
    # "my_city": "my_custom_db_name",
}


def _db_config(city: str) -> dict:
    """Build database connection config for the given city folder name."""
    dbname = DB_NAME_MAP.get(city, f"{city}_osm")
    return {
        "dbname":   dbname,
        "user":     DB_USER,
        "password": DB_PASSWORD,
        "host":     DB_HOST,
        "port":     DB_PORT,
    }


# POI categories to include -- loaded from custom_settings.py
# Users can comment out categories they do not want.
try:
    from config import POI_INCLUDE as _POI_INCLUDE
except ImportError:
    _POI_INCLUDE = None  # include all if not configured

# Always exclude noise categories regardless of POI_INCLUDE
EXCLUDE_CATEGORIES = {
    "uncategorized", "amenity_other", "tourism_other",
    "leisure_other", "shop_other",
}


# =============================================================================
# CONNECTION
# =============================================================================

def connect(city: str):
    """Return a SQLAlchemy engine -- required for pd.read_sql in pandas >= 2.0.

    The driver is pg8000 (pure Python, BSD 3-Clause), chosen so that this
    repository's dependency tree carries no copyleft licence. The loaders below
    use the connection only through pandas.read_sql, and all PostGIS work
    (ST_X/ST_Y) happens server-side, so no driver-specific API is involved.
    """
    cfg = _db_config(city)
    url = (
        f"postgresql+pg8000://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )
    return create_engine(url)


# =============================================================================
# LOADERS (identical logic to export_bonn.py, city-agnostic)
# =============================================================================

def load_pois(conn) -> pd.DataFrame:
    print("Loading POIs...")
    query = """
        SELECT
            osm_id          AS poi_id,
            poi_category,
            ST_X(geom)      AS x,
            ST_Y(geom)      AS y
        FROM pois
        WHERE poi_category IS NOT NULL
          AND is_private = FALSE
    """
    df = pd.read_sql(query, conn)
    before = len(df)
    df = df[~df["poi_category"].isin(EXCLUDE_CATEGORIES)].copy()
    if _POI_INCLUDE is not None:
        df = df[df["poi_category"].isin(_POI_INCLUDE)].copy()
    print(f"  {before:,} raw -> {len(df):,} after filtering categories")
    print(f"  Categories: {df['poi_category'].nunique()}")
    return df


def load_home_nodes(conn) -> pd.DataFrame:
    print("Loading home nodes...")
    query = """
        SELECT
            osm_id                   AS poi_id,
            'home_residential'::text AS poi_category,
            FALSE                    AS is_private,
            ST_X(geom)               AS x,
            ST_Y(geom)               AS y
        FROM home_nodes
    """
    df = pd.read_sql(query, conn)
    print(f"  {len(df):,} home nodes loaded")
    return df


def load_transport_nodes(conn) -> pd.DataFrame:
    print("Loading transport nodes...")
    query = """
        SELECT
            osm_id          AS stop_id,
            transport_type,
            ST_X(geom)      AS x,
            ST_Y(geom)      AS y
        FROM transport_nodes
    """
    df = pd.read_sql(query, conn)
    print(f"  {len(df):,} transport nodes loaded")
    if len(df) > 0:
        print(f"  Types: {df['transport_type'].value_counts().to_dict()}")
    return df


def load_pt_connectivity(conn) -> pd.DataFrame:
    """Load PT connectivity stub -- will be empty but keeps schema compatibility."""
    print("Loading PT connectivity (stub)...")
    query = """
        SELECT
            origin_stop_id,
            dest_stop_id,
            route_id,
            route_type,
            stop_distance,
            geom_distance_m
        FROM pt_connectivity
    """
    df = pd.read_sql(query, conn)
    print(f"  {len(df):,} PT connectivity pairs (0 expected -- stub table)")
    return df


def load_edges(conn) -> pd.DataFrame:
    print("Loading network edges...")
    query = """
        SELECT
            edge_id,
            road_osm_id,
            node_start          AS source,
            node_end            AS target,
            edge_length_meters  AS length_m,
            highway,
            maxspeed_final,
            oneway_final,
            lanes_final,
            bicycle_final,
            foot_final,
            bus_final,
            car_time_sec,
            bike_time_sec,
            foot_time_sec,
            is_pt_served,
            pt_stops_nearby
        FROM network_edges
    """
    df = pd.read_sql(query, conn)
    print(f"  {len(df):,} edges loaded")
    return df


def load_nearest_road(conn) -> pd.DataFrame:
    """Load POI -> road VERTEX mapping."""
    print("Loading POI -> road node mapping...")
    query = """
        SELECT
            poi_id,
            poi_category,
            road_id,
            distance_to_road_meters
        FROM network_nodes
    """
    df = pd.read_sql(query, conn)
    print(f"  {len(df):,} mappings loaded (resolved to vertex IDs)")
    print(f"  Sample road_ids: {df['road_id'].head(5).tolist()}")
    return df


def generalize_to_road_nodes(pois, nearest, node_coords_df):
    """
    Replace each POI's exact position with the position of the road-network
    node it is bound to.

    Every POI is attached to a road node during snapping, and all routing
    already runs node to node, so the road node is the position at which a POI
    actually enters the simulation. Keeping the building's own coordinate adds
    no routing precision but does pin every POI -- including the residential
    building each agent is assigned to -- to an individual real address.

    After this, POIs sharing a road node share a coordinate (median 6 per
    node), which is the intended generalization. It does not alter routing,
    but it does change distance-based scoring (air-distance matrix, POI
    density, U_dist), so a model trained before this change should be
    retrained.
    """
    coords = node_coords_df.set_index("node_id")[["x", "y"]]
    poi_to_road = nearest[["poi_id", "road_id"]].drop_duplicates("poi_id").set_index("poi_id")["road_id"]

    road_id = pois["poi_id"].map(poi_to_road)
    new_x = road_id.map(coords["x"])
    new_y = road_id.map(coords["y"])

    matched = new_x.notna() & new_y.notna()
    if matched.any():
        dlat = (pois.loc[matched, "y"] - new_y[matched]) * 111_000.0
        dlon = ((pois.loc[matched, "x"] - new_x[matched]) * 111_000.0
                * np.cos(np.radians(pois.loc[matched, "y"])))
        moved = np.sqrt(dlat ** 2 + dlon ** 2)
        median_m = float(moved.median())
    else:
        median_m = float("nan")

    # POIs with no road node keep their own coordinate; they cannot be routed
    # to in any case, so they never appear in a trajectory.
    pois = pois.copy()
    pois.loc[matched, "x"] = new_x[matched]
    pois.loc[matched, "y"] = new_y[matched]

    n_nodes = int(road_id[matched].nunique())
    print(f"  Generalized {int(matched.sum()):,} POI positions to {n_nodes:,} road nodes "
          f"(median shift {median_m:.1f} m; {int((~matched).sum()):,} unmatched kept as-is)")
    return pois


def reindex_poi_ids(pois, transport_nodes, nearest, pt_connectivity):
    """
    Replace OSM object ids with sequential internal ids.

    The exported caches are keyed by POI id. Keeping the OSM id there means
    every POI -- including the residential building each simulated agent is
    assigned to -- stays directly resolvable on openstreetmap.org, with
    whatever name, address and contact tags that object carries. Removing the
    name column alone does not prevent that lookup; the id itself is the link.

    Ids are assigned in sorted order so the mapping is deterministic for a
    given extract. The OSM-to-internal mapping is deliberately NOT written to
    disk -- persisting it next to the data would restore exactly the linkage
    this removes. Road vertex ids are untouched: pgRouting already generates
    its own sequential ids, independent of OSM.
    """
    id_cols = [
        (pois,            "poi_id"),
        (transport_nodes, "stop_id"),
        (nearest,         "poi_id"),
        (pt_connectivity, "origin_stop_id"),
        (pt_connectivity, "dest_stop_id"),
    ]

    present = [
        df[col].dropna() for df, col in id_cols
        if col in df.columns and len(df) > 0
    ]
    if not present:
        return pois, transport_nodes, nearest, pt_connectivity

    all_ids = np.sort(pd.unique(pd.concat(present, ignore_index=True)).astype("int64"))
    mapping = pd.Series(np.arange(1, len(all_ids) + 1, dtype="int64"), index=all_ids)

    for df, col in id_cols:
        if col in df.columns and len(df) > 0:
            df[col] = df[col].map(mapping).astype("int64")

    print(f"  Re-indexed {len(all_ids):,} OSM ids -> internal ids 1..{len(all_ids):,}")
    return pois, transport_nodes, nearest, pt_connectivity


def load_road_node_coords(conn) -> pd.DataFrame:
    print("Loading road node coordinates...")
    query = """
        SELECT
            id          AS node_id,
            ST_X(the_geom) AS x,
            ST_Y(the_geom) AS y
        FROM roads_net_vertices_pgr
    """
    df = pd.read_sql(query, conn)
    print(f"  {len(df):,} road node coordinates loaded")
    return df


# =============================================================================
# GRAPH BUILDER (unchanged from export_bonn.py)
# =============================================================================

def build_graph(edges_df: pd.DataFrame, node_coords_df: pd.DataFrame = None) -> nx.Graph:
    """Build NetworkX graph restricted to the largest connected component."""
    print("Building NetworkX graph...")

    node_coord_lookup = {}
    if node_coords_df is not None and len(node_coords_df) > 0:
        for _, row in node_coords_df.iterrows():
            node_coord_lookup[int(row["node_id"])] = (float(row["x"]), float(row["y"]))

    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(
            int(row["source"]),
            int(row["target"]),
            weight=float(row["length_m"]),
            car_time_sec=float(row["car_time_sec"]) if pd.notna(row["car_time_sec"]) else None,
            bike_time_sec=float(row["bike_time_sec"]) if pd.notna(row["bike_time_sec"]) else None,
            foot_time_sec=float(row["foot_time_sec"]) if pd.notna(row["foot_time_sec"]) else None,
            highway=row["highway"],
            bicycle_final=row["bicycle_final"],
            foot_final=row["foot_final"],
            is_pt_served=bool(row["is_pt_served"]),
        )

    print(f"  Full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    print(f"  Connected components: {len(components):,}")
    print(f"  Largest 5: {[len(c) for c in components[:5]]}")

    largest_cc = components[0]
    G = G.subgraph(largest_cc).copy()
    print(f"  Keeping largest component: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges "
          f"({G.number_of_nodes() / len(components[0]) * 100:.0f}% of all nodes)")

    coords_added = 0
    for nid in G.nodes():
        coords = node_coord_lookup.get(int(nid))
        if coords:
            G.nodes[nid]["x"] = coords[0]
            G.nodes[nid]["y"] = coords[1]
            coords_added += 1
    print(f"  Node coords added: {coords_added:,} / {G.number_of_nodes():,}")

    car_nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get("car_time_sec") and d.get("highway", "") not in (
            "footway", "steps", "pedestrian", "cycleway", "path"
        ):
            car_nodes.add(u); car_nodes.add(v)
    foot_nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get("foot_time_sec"):
            foot_nodes.add(u); foot_nodes.add(v)
    print(f"  Car-accessible nodes:  {len(car_nodes):,}")
    print(f"  Foot-accessible nodes: {len(foot_nodes):,}")

    return G


# =============================================================================
# RESNAP (unchanged from export_bonn.py)
# =============================================================================

def resnap_to_component(nearest: pd.DataFrame, graph: nx.Graph,
                         node_coords_df: pd.DataFrame) -> pd.DataFrame:
    """Resnap POIs whose snapped node fell outside the largest component."""
    print("\nResnapping POIs to largest connected component...")
    graph_nodes = set(graph.nodes())

    gc_lon = np.array([graph.nodes[n].get("x", 0.0) for n in graph_nodes])
    gc_lat = np.array([graph.nodes[n].get("y", 0.0) for n in graph_nodes])
    gc_ids = np.array(list(graph_nodes))

    def _resnap(road_id):
        if int(road_id) in graph_nodes:
            return int(road_id)
        row = node_coords_df[node_coords_df["node_id"] == road_id]
        if len(row) == 0:
            return int(gc_ids[0])
        lon = float(row["x"].iloc[0])
        lat = float(row["y"].iloc[0])
        dlat = np.radians(gc_lat - lat)
        dlon = np.radians(gc_lon - lon)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat)) * np.cos(np.radians(gc_lat))
             * np.sin(dlon / 2) ** 2)
        dists = 2 * 6371000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return int(gc_ids[np.argmin(dists)])

    outside = nearest[~nearest["road_id"].isin(graph_nodes)]
    print(f"  POI 0s outside component: {len(outside):,} of {len(nearest):,}")
    if len(outside) > 0:
        nearest = nearest.copy()
        mask = ~nearest["road_id"].isin(graph_nodes)
        nearest.loc[mask, "road_id"] = nearest.loc[mask, "road_id"].apply(_resnap)
        still_outside = nearest[~nearest["road_id"].isin(graph_nodes)]
        print(f"  After resnap, still outside: {len(still_outside):,}")
    return nearest


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export city OSM data to spatial cache.")
    parser.add_argument("--city", required=True,
        help="City folder name")
    parser.add_argument("--output", default=None,
        help="Output path for spatial_cache.pkl (default: data/formatted/<city>/spatial_cache.pkl)")
    args = parser.parse_args()
    city = args.city

    if args.output:
        output_file = args.output
        output_dir  = os.path.dirname(output_file)
    else:
        output_dir  = os.path.join("data", "formatted", city)
        output_file = os.path.join(output_dir, "spatial_cache.pkl")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"EXPORTING {city.upper()} OSM DATA")
    print("=" * 60)

    conn = connect(city)

    pois            = load_pois(conn)
    home_nodes      = load_home_nodes(conn)
    transport_nodes = load_transport_nodes(conn)
    pt_connectivity = load_pt_connectivity(conn)
    edges           = load_edges(conn)
    nearest         = load_nearest_road(conn)
    node_coords_df  = load_road_node_coords(conn)

    conn.dispose()

    graph   = build_graph(edges, node_coords_df)
    nearest = resnap_to_component(nearest, graph, node_coords_df)

    pois_combined = pd.concat([pois, home_nodes], ignore_index=True)
    print(f"\nMerged pois: {len(pois):,} POIs + {len(home_nodes):,} home nodes "
          f"= {len(pois_combined):,} total")

    pois_combined = generalize_to_road_nodes(pois_combined, nearest, node_coords_df)

    pois_combined, transport_nodes, nearest, pt_connectivity = reindex_poi_ids(
        pois_combined, transport_nodes, nearest, pt_connectivity
    )

    data = {
        "city":             city,
        "pois":             pois_combined,
        "transport_nodes":  transport_nodes,
        "pt_connectivity":  pt_connectivity,
        "edges":            edges,
        "nodes":            nearest,
        "graph":            graph,
    }

    print(f"\nSaving to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"  Saved ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print(f"EXPORT COMPLETE -- {city.upper()}")
    print("=" * 60)
    print(f"\n  POIs (excl. home):     {len(pois):,}")
    print(f"  Home nodes:            {len(home_nodes):,}")
    print(f"  Total pois merged:     {len(pois_combined):,}")
    print(f"  Transport nodes:       {len(transport_nodes):,}")
    print(f"  Network edges:         {len(edges):,}")
    print(f"  POI->road map:         {len(nearest):,}")
    print(f"\nNext: python filter_connected_pois.py --city {city}")
    print(f"Then: python setup.py --city {city}")


if __name__ == "__main__":
    main()