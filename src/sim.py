"""
UNIFIED SIMULATION & ANALYSIS PIPELINE
=======================================
One comprehensive tool that does everything:
1. Runs simulations with trained model
2. Logs to 4 CSV tables (with sub-activity types)
3. Generates complete text summary
4. Creates aggregate visualizations
5. Creates individual visualizations  
6. Generates Folium interactive maps
7. Computes validation metrics

Usage:
    python simulate_and_analyze.py \
        --model models/ppo_final.zip \
        --n_sims 300 \
        --output results_v35 \
        --n_individual_plots 5 \
        --n_trajectory_maps 3 \
        --n_trajectories_sample 100
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p


import numpy as np
import matplotlib.gridspec as gridspec
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from grid_analysis import run_grid_pipeline
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sb3_contrib import MaskablePPO  #  FIXED: was "from stable_baselines3 import PPO"
from environment import MosaicEnv
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import Counter, defaultdict
from scipy.stats import entropy
import pickle
import json
import os
import argparse
from pyproj import Transformer

# Activity colors (consistent across all plots)
ACTIVITY_COLORS = {
    'HOME': '#1f77b4',
    'WORK': '#ff7f0e',
    'FOOD': '#2ca02c',
    'ERRAND': '#d62728',
    'LEISURE': '#9467bd',
    'TRAVEL': '#7f7f7f'
}

# v4 data is already in WGS84 (EPSG:4326) from the flex import.
# No coordinate transformation needed -- coordinates are lon/lat degrees directly.
# transformer kept for backward compatibility but not used.
try:
    from pyproj import Transformer as _Transformer
    transformer = _Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
except Exception:
    transformer = None

def _get_wgs84(row):
    """Return (lon, lat) from a POI row -- already WGS84 in v4."""
    x_col = next((c for c in ['x', 'lon', 'longitude'] if c in row.index), None)
    y_col = next((c for c in ['y', 'lat', 'latitude']  if c in row.index), None)
    if x_col and y_col:
        return float(row[x_col]), float(row[y_col])
    return 0.0, 0.0


# Coordinate columns that must never reach the written logs.
_ACTIVITY_COORD_COLS = ['lat', 'lon']
_TRAVEL_COORD_COLS   = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']


def _to_grid_cells(activities_df, travels_df):
    """
    Return (activities, travels) copies in which exact POI coordinates are
    replaced by grid cell ids.

    Exact coordinates are needed to run the simulation and to draw the Folium
    maps, but they resolve to individual real buildings -- including the
    residential building each synthetic agent is assigned to -- so they are
    dropped before anything is written to disk. The released spatial
    resolution is the grid cell (see grid_layer.CELL_SIZE_M).

    Activities gain  : cell_id
    Activities lose  : lat, lon
    Travels gain     : origin_cell, dest_cell
    Travels lose     : origin_lat, origin_lon, dest_lat, dest_lon
    """
    from grid_layer import CityGrid
    from config import MAP_FOLDER

    # bbox must come from the city being simulated, not a hardcoded one: the
    # grid clamps out-of-bbox points, so another city's bbox silently collapses
    # every activity into one corner cell.
    grid = CityGrid(city=MAP_FOLDER)

    acts = grid.snap_activities_df(activities_df)
    acts = acts.drop(columns=[c for c in _ACTIVITY_COORD_COLS if c in acts.columns])

    travs = travels_df.copy()
    for prefix in ('origin', 'dest'):
        lat_col, lon_col = f'{prefix}_lat', f'{prefix}_lon'
        if lat_col in travs.columns and lon_col in travs.columns:
            travs[f'{prefix}_cell'] = [
                grid.coord_to_cell(la, lo)
                for la, lo in zip(travs[lat_col], travs[lon_col])
            ]
        else:
            travs[f'{prefix}_cell'] = 'invalid'
    travs = travs.drop(columns=[c for c in _TRAVEL_COORD_COLS if c in travs.columns])

    return acts, travs

# Trajectory destination colors
DEST_COLORS = {
    'HOME':    '#6ec6f0',  # light blue
    'WORK':    '#ff7f0e',  # orange
    'FOOD':    '#ffe135',  # yellow
    'ERRAND':  '#d62728',  # red
    'LEISURE': '#a020f0',  # violet
}

# ============================================================================
# SIMULATION & LOGGING
# ============================================================================

def load_path_library():
    """Load PathGenerator for road-network trajectory geometry."""
    try:
        from path_generator import PathGenerator
        from config import SPATIAL_CACHE_FILE
        lib = PathGenerator.load(SPATIAL_CACHE_FILE)
        if lib is not None and lib.available:
            print("[OK] PathGenerator loaded -- road-network trajectories enabled")
            return lib
        else:
            print("[WARN] PathGenerator unavailable -- trajectories will use straight lines")
            return None
    except Exception as e:
        print(f"[WARN] PathGenerator error: {e} -- trajectories will use straight lines")
        return None
def run_simulations(model_path, n_simulations, output_dir, path_generator=None):
    """
    Run simulations and create 4 CSV tables with sub-activity logging.
    
    Returns:
        Tuple of (activities_df, travels_df, hourly_df, metadata_df)
    """
    print("="*70)
    print(f"RUNNING {n_simulations} SIMULATIONS")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = MaskablePPO.load(model_path)  #  FIXED: was PPO.load(model_path)
    print("[OK] Model loaded")
    
    # Load PathGenerator for road-network trajectories (passed in or loaded here)
    if path_generator is None:
        path_generator = load_path_library()
    
    # Storage
    all_activities = []
    all_travels = []
    all_hourly_stamps = []
    all_metadata = []
    
    travel_id_counter = 0
    activity_id_counter = 0
    
    # Ensure trajectory tracking is enabled for simulation mode
    import config as _cfg
    _cfg.FAST_TRAINING_MODE = False
    _cfg.TRACK_TRAJECTORIES = True

    # Build the env ONCE -- all heavy data (OSM, PathGenerator, graphs) is loaded
    # here and reused across every simulation via reset(seed=...).
    print("\nInitializing environment (one-time load)...")
    env = MosaicEnv(seed=42)
    print("[OK] Environment ready")

    print(f"\nRunning {n_simulations} simulations...")

    for sim_id in tqdm(range(n_simulations), desc="Simulations"):
        obs, info = env.reset(seed=42 + sim_id)
        action_masks = info.get("action_masks", None)  #  FIXED: grab mask from reset
        
        home_id = env.agent.home_location_id
        work_id = env.agent.work_location_id
        
        sim_activities = []
        sim_travels = []
        
        # Initial state
        activity_start_time = env.current_datetime
        activity_type = 'HOME'
        sub_activity_type = None
        location_id = home_id
        
        # Run episode
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action_masks = env.getvalidactionsmask()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)  #  FIXED: pass masks
            action = int(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            action_masks = env.getvalidactionsmask()  #  FIXED: grab mask for next step
            
            new_activity          = info['macro_activity']
            new_sub_activity      = info.get('sub_activity', None)
            new_location_id       = info['location_id']
            travel_time_min       = info['travel_time']

            # Use exact timestamps from env (immune to day-skip clock changes)
            dep_time_str = info.get('dep_time')
            arr_time_str = info.get('arr_time')
            act_end_str  = info.get('act_end_time')

            if dep_time_str and arr_time_str and act_end_str:
                travel_start       = datetime.fromisoformat(dep_time_str)
                travel_end         = datetime.fromisoformat(arr_time_str)
                new_activity_start = travel_end
                new_activity_end   = datetime.fromisoformat(act_end_str)
                old_activity_end   = travel_start
            else:
                # Fallback: reconstruct (old envs without dep/arr in info)
                current_time       = env.current_datetime
                new_activity_duration_min = info['duration']
                new_activity_start = current_time - timedelta(minutes=new_activity_duration_min)
                travel_end         = new_activity_start
                travel_start       = travel_end - timedelta(minutes=travel_time_min)
                old_activity_end   = travel_start
                new_activity_end   = current_time

            old_activity_duration_min = max(0.0, (old_activity_end - activity_start_time).total_seconds() / 60)

            # --- COORDINATE TRANSFORMATION (EPSG:3857 -> EPSG:4326) ---
            pois_table = env.city.spatial_env.pois
            loc_row = pois_table[pois_table['poi_id'] == location_id].iloc[0]
            new_loc_row = pois_table[pois_table['poi_id'] == new_location_id].iloc[0]
            curr_lon, curr_lat = _get_wgs84(loc_row)
            new_lon, new_lat   = _get_wgs84(new_loc_row)
            # -----------------------------------------------------------

            # Log completed activity
            if old_activity_duration_min > 0:
                sim_activities.append({
                    'sim_id': sim_id,
                    'activity_id': activity_id_counter,
                    'start_time': activity_start_time,
                    'end_time': travel_start,
                    'activity_type': activity_type,
                    'sub_activity_type': sub_activity_type,
                    'location_id': location_id,
                    'lat': curr_lat,
                    'lon': curr_lon,
                    'duration_min': old_activity_duration_min,
                    'travel_id': None
                })
                activity_id_counter += 1
            
            # Log travel
            if travel_time_min > 0:
                # Get road-network linestring via PathGenerator
                # mode comes from transport_mode in info dict
                ls_id = None
                if path_generator is not None:
                    _mode = info.get('transport_mode', 'car')
                    ls_id = path_generator.get_linestring_id(location_id, new_location_id, _mode)
                
                travel_entry = {
                    'travel_id': f'trip_{travel_id_counter:06d}',
                    'sim_id': sim_id,
                    'start_time': travel_start,
                    'end_time': travel_end,
                    'duration_min': travel_time_min,
                    'origin_id': location_id,
                    'dest_id': new_location_id,
                    'dest_activity': new_activity,
                    'origin_lat': curr_lat,
                    'origin_lon': curr_lon,
                    'dest_lat': new_lat,
                    'dest_lon': new_lon,
                    'distance_km': info['distance_km'],
                    'mode': info['transport_mode'],
                    'linestring_id': ls_id,
                }
                sim_travels.append(travel_entry)
                
                sim_activities.append({
                    'sim_id': sim_id,
                    'activity_id': activity_id_counter,
                    'start_time': travel_start,
                    'end_time': travel_end,
                    'activity_type': 'TRAVEL',
                    'sub_activity_type': None,
                    'location_id': None,
                    'duration_min': travel_time_min,
                    'travel_id': travel_entry['travel_id']
                })
                activity_id_counter += 1
                travel_id_counter += 1
            
            # Update state
            activity_start_time = new_activity_start  # real arrival time
            activity_type       = new_activity
            sub_activity_type   = new_sub_activity
            location_id         = new_location_id
        
        # Final activity
        end_of_week = env.episode_end_time
        final_duration_min = (end_of_week - activity_start_time).total_seconds() / 60

        # Transform final location
        f_loc_row = env.city.spatial_env.pois[
            env.city.spatial_env.pois['poi_id'] == env.agent.current_location_id
        ].iloc[0]
        f_lon, f_lat = _get_wgs84(f_loc_row)

        sim_activities.append({
            'sim_id': sim_id,
            'activity_id': activity_id_counter,
            'start_time': activity_start_time,
            'end_time': end_of_week,
            'activity_type': env.agent.current_location,
            'sub_activity_type': env.agent.current_subactivity,
            'location_id': env.agent.current_location_id,
            'lat': f_lat,
            'lon': f_lon,
            'duration_min': final_duration_min,
            'travel_id': None
        })
        activity_id_counter += 1
        
        # Generate hourly stamps across the WHOLE episode.
        # The window spans EPISODE_DAYS rather than a fixed 7 days. The
        # activity and travel logs above already cover the whole episode, so a
        # shorter window here truncates hourly_stamps.csv with no other symptom.
        start_of_sim = sim_activities[0]['start_time'].replace(hour=0, minute=0, second=0)
        stamps_end = env.episode_end_time

        current_hour = start_of_sim
        while current_hour < stamps_end:
            location_type = 'HOME'
            
            for activity in sim_activities:
                if activity['start_time'] <= current_hour < activity['end_time']:
                    location_type = activity['activity_type']
                    break
            
            all_hourly_stamps.append({
                'sim_id': sim_id,
                'timestamp': current_hour,
                'location_type': location_type
            })
            
            current_hour += timedelta(hours=1)
        
        all_activities.extend(sim_activities)
        all_travels.extend(sim_travels)
        
        # Metadata
        all_metadata.append({
            'sim_id': sim_id,
            'home_id': home_id,
            'work_id': work_id,
            'n_activities': len(sim_activities),
            'n_trips': len(sim_travels),
            'total_distance_km': sum(t['distance_km'] for t in sim_travels),
            'total_travel_time_min': sum(t['duration_min'] for t in sim_travels)
        })
    
    print(f"\n[OK] Completed {n_simulations} simulations")
    
    # Create DataFrames
    activities_df = pd.DataFrame(all_activities)
    travels_df = pd.DataFrame(all_travels)
    hourly_df = pd.DataFrame(all_hourly_stamps)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save tables
    os.makedirs(f'{output_dir}/data', exist_ok=True)

    print(f"\nSaving database tables to {output_dir}/data/...")
    # Exact POI coordinates are used for the simulation itself and for the
    # Folium maps built further below from the in-memory frames, but they are
    # never written to disk: the logs carry grid cell ids instead. See
    # _to_grid_cells() for the exact columns that are dropped and added.
    acts_out, travs_out = _to_grid_cells(activities_df, travels_df)

    acts_out.to_csv(f'{output_dir}/data/activity_log.csv', index=False)
    print(f"  [OK] activity_log.csv ({len(acts_out):,} rows, grid cells)")

    travs_out.to_csv(f'{output_dir}/data/travel_log.csv', index=False)
    print(f"  [OK] travel_log.csv ({len(travs_out):,} rows, grid cells)")
    
    hourly_df.to_csv(f'{output_dir}/data/hourly_stamps.csv', index=False)
    print(f"  [OK] hourly_stamps.csv ({len(hourly_df):,} rows)")
    
    metadata_df.to_csv(f'{output_dir}/data/simulation_metadata.csv', index=False)
    print(f"  [OK] simulation_metadata.csv ({len(metadata_df):,} rows)")

    _save_paths(travels_df, path_generator, output_dir)
    _save_poi_list(env, output_dir)

    return activities_df, travels_df, hourly_df, metadata_df


def _save_poi_list(env, output_dir):
    """
    Write the POI table the logs refer to, as poi_list.csv.

    activity_log.csv and travel_log.csv identify places by poi_id only, so
    without this table the released trajectories cannot be resolved to a
    location or a venue type at all. Positions are road-network node
    coordinates (see generalize_to_road_nodes in map/export_city.py), shared
    by a median of six POIs each, rather than individual building positions.
    """
    pois = getattr(getattr(env, "city", None), "spatial_env", None)
    pois = getattr(pois, "pois", None)
    if pois is None or len(pois) == 0:
        print("  [WARNING] POI table unavailable; poi_list.csv not written")
        return

    keep = [c for c in ("poi_id", "poi_category", "x", "y") if c in pois.columns]
    out = pois[keep].drop_duplicates("poi_id").sort_values("poi_id")
    out = out.rename(columns={"x": "lon", "y": "lat"})
    out.to_csv(f'{output_dir}/data/poi_list.csv', index=False)
    print(f"  [OK] poi_list.csv ({len(out):,} POIs, road-node positions)")


def _save_paths(travels_df, path_generator, output_dir):
    """
    Write the road-network geometry of every distinct trip to paths.csv.

    Without this the generated trajectories cannot be reconstructed from the
    released files at all: travel_log.csv stores only the origin and
    destination, and the geometry lives solely in the path library, which is
    an internal cache rather than part of the dataset. Trips are deduplicated
    on (origin_id, dest_id, mode) because many agents reuse the same route.

    Geometry is written as a WKT LINESTRING in WGS84, so it loads directly in
    geopandas, QGIS or PostGIS. Coordinates are road-network node positions.
    """
    if path_generator is None or travels_df.empty:
        return

    cols = [c for c in ('origin_id', 'dest_id', 'mode') if c in travels_df.columns]
    if len(cols) < 3:
        return

    rows = []
    for origin_id, dest_id, mode in travels_df[cols].drop_duplicates().itertuples(index=False):
        try:
            ls = path_generator.get_linestring(int(origin_id), int(dest_id), str(mode))
        except Exception:
            ls = None
        if not ls or len(ls) < 2:
            continue
        wkt = "LINESTRING (" + ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat, *_ in ls) + ")"
        rows.append({
            "origin_id": int(origin_id),
            "dest_id":   int(dest_id),
            "mode":      str(mode),
            "n_points":  len(ls),
            "geometry_wkt": wkt,
        })

    paths_df = pd.DataFrame(rows)
    paths_df.to_csv(f'{output_dir}/data/paths.csv', index=False)
    print(f"  [OK] paths.csv ({len(paths_df):,} distinct routes)")


# ============================================================================
# TEXT SUMMARY
# ============================================================================

def generate_text_summary(activities_df, travels_df, hourly_df, metadata_df, output_dir, args):
    """Generate comprehensive text summary of simulation results"""
    
    print("\n" + "="*70)
    print("GENERATING TEXT SUMMARY")
    print("="*70)
    
    summary_lines = []
    summary_lines.append("="*70)
    summary_lines.append("SIMULATION SUMMARY REPORT")
    summary_lines.append("="*70)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Model: {args.model}")
    summary_lines.append(f"Number of simulations: {args.n_sims}")
    summary_lines.append("")
    
    # Activity statistics
    summary_lines.append("="*70)
    summary_lines.append("ACTIVITY STATISTICS")
    summary_lines.append("="*70)
    summary_lines.append(f"Total activities logged: {len(activities_df):,}")
    summary_lines.append("")
    
    time_by_type = activities_df.groupby('activity_type')['duration_min'].sum()
    total_time = time_by_type.sum()
    summary_lines.append('Activity Type Distribution (by time):')
    for activity, mins in time_by_type.sort_values(ascending=False).items():
        pct = mins / total_time * 100 if total_time > 0 else 0
        summary_lines.append(f"  {activity:<10} {mins/60:>8.1f}h  {pct:>5.2f}%")
    summary_lines.append("")
    
    # Activity durations
    summary_lines.append("Activity Duration Statistics (hours):")
    for activity in ['HOME', 'WORK', 'FOOD', 'ERRAND', 'LEISURE']:
        durations = activities_df[activities_df['activity_type'] == activity]['duration_min'] / 60
        if len(durations) == 0:
            continue
        summary_lines.append(f"  {activity:10} Mean: {durations.mean():5.2f}h  "
                            f"Std: {durations.std():5.2f}h  "
                            f"Min: {durations.min():5.2f}h  "
                            f"Max: {durations.max():5.2f}h")
    summary_lines.append("")
    
    # Sub-activity statistics
    if 'sub_activity_type' in activities_df.columns:
        summary_lines.append("="*70)
        summary_lines.append("SUB-ACTIVITY BREAKDOWN")
        summary_lines.append("="*70)
        
        for activity in ['FOOD', 'ERRAND', 'LEISURE']:
            sub_acts = activities_df[activities_df['activity_type'] == activity]['sub_activity_type'].value_counts()
            if len(sub_acts) > 0:
                summary_lines.append(f"\n{activity}:")
                for sub_act, count in sub_acts.head(10).items():
                    if pd.notna(sub_act):
                        summary_lines.append(f"  {sub_act:35} {count:6,}")
        summary_lines.append("")
    
    # Travel statistics
    summary_lines.append("="*70)
    summary_lines.append("TRAVEL STATISTICS")
    summary_lines.append("="*70)
    summary_lines.append(f"Total trips: {len(travels_df):,}")
    
    if len(travels_df) > 0:
        summary_lines.append(f"\nDistance Statistics:")
        summary_lines.append(f"  Mean: {travels_df['distance_km'].mean():6.2f} km")
        summary_lines.append(f"  Std:  {travels_df['distance_km'].std():6.2f} km")
        summary_lines.append(f"  Min:  {travels_df['distance_km'].min():6.2f} km")
        summary_lines.append(f"  Max:  {travels_df['distance_km'].max():6.2f} km")
        
        summary_lines.append(f"\nDuration Statistics:")
        summary_lines.append(f"  Mean: {travels_df['duration_min'].mean():6.1f} min")
        summary_lines.append(f"  Std:  {travels_df['duration_min'].std():6.1f} min")
        summary_lines.append(f"  Min:  {travels_df['duration_min'].min():6.1f} min")
        summary_lines.append(f"  Max:  {travels_df['duration_min'].max():6.1f} min")
        
        trips_per_person = travels_df.groupby('sim_id').size()
        summary_lines.append(f"\nTrips per Person:")
        summary_lines.append(f"  Mean: {trips_per_person.mean():6.1f}")
        summary_lines.append(f"  Std:  {trips_per_person.std():6.1f}")
    summary_lines.append("")
    
    # Per-person statistics
    summary_lines.append("="*70)
    summary_lines.append("PER-PERSON STATISTICS")
    summary_lines.append("="*70)
    summary_lines.append(f"Total simulated people: {len(metadata_df)}")
    summary_lines.append(f"\nAverage per person:")
    summary_lines.append(f"  Activities: {metadata_df['n_activities'].mean():6.1f}")
    summary_lines.append(f"  Trips:      {metadata_df['n_trips'].mean():6.1f}")
    summary_lines.append(f"  Distance:   {metadata_df['total_distance_km'].mean():6.1f} km")
    summary_lines.append(f"  Travel time:{metadata_df['total_travel_time_min'].mean():6.1f} min")
    summary_lines.append("")
    
    # Temporal patterns
    summary_lines.append("="*70)
    summary_lines.append("TEMPORAL PATTERNS")
    summary_lines.append("="*70)
    
    hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
    hourly_df['hour'] = hourly_df['timestamp'].dt.hour
    hourly_df['day'] = hourly_df['timestamp'].dt.day_name()
    
    summary_lines.append("\nMost common activity by hour (all days averaged):")
    hour_activities = hourly_df.groupby('hour')['location_type'].agg(lambda x: x.value_counts().index[0])
    for hour, activity in hour_activities.items():
        summary_lines.append(f"  {hour:02d}:00  {activity}")
    summary_lines.append("")
    
    # Write to file
    output_path = f'{output_dir}/summary_statistics.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"[OK] Saved: {output_path}")
    
    return summary_lines


# ============================================================================
# AGGREGATE VISUALIZATIONS
# ============================================================================

def create_hourly_cumulative_plot(hourly_df, output_dir):
    """Stacked area chart of population activity over time"""
    
    print("\n" + "="*70)
    print("Creating aggregate visualizations...")
    print("  1/8: Hourly cumulative...")
    
    hourly_df = hourly_df.copy()
    hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
    hourly_df['hour_of_week'] = (hourly_df['timestamp'] - hourly_df['timestamp'].min()).dt.total_seconds() / 3600
    
    pivot = hourly_df.pivot_table(
        index='hour_of_week',
        columns='location_type',
        values='sim_id',
        aggfunc='count',
        fill_value=0
    )
    
    for activity in ['HOME', 'WORK', 'TRAVEL', 'FOOD', 'ERRAND', 'LEISURE']:
        if activity not in pivot.columns:
            pivot[activity] = 0
    
    activity_order = ['HOME', 'WORK', 'TRAVEL', 'FOOD', 'ERRAND', 'LEISURE']
    pivot = pivot[[col for col in activity_order if col in pivot.columns]]
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    pivot.plot.area(
        ax=ax,
        color=[ACTIVITY_COLORS[act] for act in pivot.columns],
        alpha=0.8,
        linewidth=0
    )
    
    ax.set_xlabel('Hour of Week', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of People', fontsize=13, fontweight='bold')
    ax.set_title('Population Activity Distribution Over Week\n(Cumulative stacked area chart)', 
                 fontsize=15, fontweight='bold')
    
    for day in range(8):
        hour = day * 24
        ax.axvline(hour, color='black', alpha=0.2, linestyle='--', linewidth=1)
        if day < 7:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax.text(hour + 12, ax.get_ylim()[1] * 0.95, day_names[day], 
                   ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, 168)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/aggregate/01_hourly_cumulative.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_24x7_heatmap(hourly_df, output_dir):
    """24 hours x 7 days heatmap showing most popular activity"""
    
    print("  2/8: 24x7 heatmap...")
    
    hourly_df = hourly_df.copy()
    hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
    hourly_df['hour'] = hourly_df['timestamp'].dt.hour
    hourly_df['day'] = hourly_df['timestamp'].dt.dayofweek
    
    # Find most common activity for each hour/day combo
    activity_matrix = np.zeros((24, 7), dtype=int)
    activity_labels = ['HOME', 'WORK', 'TRAVEL', 'FOOD', 'ERRAND', 'LEISURE']
    
    for hour in range(24):
        for day in range(7):
            mask = (hourly_df['hour'] == hour) & (hourly_df['day'] == day)
            if mask.sum() > 0:
                most_common = hourly_df[mask]['location_type'].value_counts().index[0]
                activity_matrix[hour, day] = activity_labels.index(most_common)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # Create colormap
    colors = [ACTIVITY_COLORS[act] for act in activity_labels]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    im = ax.imshow(activity_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=len(activity_labels)-1)
    
    # Set ticks
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=11)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(24)], fontsize=9)
    
    ax.set_xlabel('Day of Week', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hour of Day', fontsize=13, fontweight='bold')
    ax.set_title('Most Common Activity by Hour and Day\n(24x7 Grid)', fontsize=15, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(7) - 0.5, minor=True)
    ax.set_yticks(np.arange(24) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=ACTIVITY_COLORS[act], label=act) 
                      for act in activity_labels]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
             fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/aggregate/02_activity_heatmap_24x7.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_activity_duration_plots(activities_df, output_dir):
    """Box plots of activity durations by type"""
    
    print("  3/8: Activity durations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    activities = ['HOME', 'WORK', 'FOOD', 'ERRAND', 'LEISURE', 'TRAVEL']
    
    for idx, activity in enumerate(activities):
        ax = axes[idx]
        data = activities_df[activities_df['activity_type'] == activity]['duration_min'] / 60
        
        if len(data) > 0:
            ax.boxplot([data], labels=[activity], vert=True)
            ax.set_ylabel('Duration (hours)', fontsize=11)
            ax.set_title(f'{activity}\nMean: {data.mean():.2f}h', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Activity Duration Distributions', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/aggregate/03_activity_durations_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_activity_pie_chart(activities_df, output_dir):
    """Pie chart of activity distribution"""
    
    print("  4/8: Activity pie chart...")
    
    # Calculate time spent (not count)
    time_by_activity = activities_df.groupby('activity_type')['duration_min'].sum() / 60
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [ACTIVITY_COLORS.get(act, '#cccccc') for act in time_by_activity.index]
    
    wedges, texts, autotexts = ax.pie(
        time_by_activity.values,
        labels=time_by_activity.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax.set_title('Time Distribution by Activity Type\n(Percentage of Total Hours)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/aggregate/04_activity_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_weekly_distribution_comparison(activities_df, output_dir):
    """Bar chart comparing actual vs target weekly distributions"""
    
    print("  5/8: Weekly distribution comparison...")
    
    # Calculate actual percentages
    time_by_activity = activities_df.groupby('activity_type')['duration_min'].sum()
    total_time = time_by_activity.sum()
    actual_pct = (time_by_activity / total_time * 100).to_dict()
    
    # Typical targets (adjust based on your expectations)
    target_pct = {
        'HOME': 55.0,
        'WORK': 25.0,
        'FOOD': 5.0,
        'ERRAND': 3.0,
        'LEISURE': 8.0,
        'TRAVEL': 4.0
    }
    
    activities = ['HOME', 'WORK', 'FOOD', 'ERRAND', 'LEISURE', 'TRAVEL']
    actual_vals = [actual_pct.get(act, 0) for act in activities]
    target_vals = [target_pct.get(act, 0) for act in activities]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(activities))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, actual_vals, width, label='Actual', 
                   color=[ACTIVITY_COLORS[act] for act in activities], alpha=0.8)
    bars2 = ax.bar(x + width/2, target_vals, width, label='Target', 
                   color='gray', alpha=0.5)
    
    ax.set_xlabel('Activity Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Week (%)', fontsize=13, fontweight='bold')
    ax.set_title('Weekly Time Distribution: Actual vs Target', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(activities, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/aggregate/05_weekly_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_temporal_patterns(activities_df, output_dir):
    """Show when different activities happen during the day"""
    
    print("  6/8: Temporal patterns...")
    
    # Extract hour from start_time
    activities_df = activities_df.copy()
    activities_df['start_time'] = pd.to_datetime(activities_df['start_time'])
    activities_df['hour'] = activities_df['start_time'].dt.hour
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    activities = ['HOME', 'WORK', 'FOOD', 'ERRAND', 'LEISURE', 'TRAVEL']
    
    for idx, activity in enumerate(activities):
        ax = axes[idx]
        data = activities_df[activities_df['activity_type'] == activity]['hour']
        
        if len(data) > 0:
            ax.hist(data, bins=24, range=(0, 24), edgecolor='black', 
                   color=ACTIVITY_COLORS[activity], alpha=0.7)
            ax.set_xlabel('Hour of Day', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{activity} Start Times', fontsize=11, fontweight='bold')
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Temporal Patterns: When Do Activities Start?', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/aggregate/06_temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_sub_activity_breakdown(activities_df, output_dir):
    """Show distribution of sub-activities"""
    
    print("  7/8: Sub-activity breakdown...")
    
    if 'sub_activity_type' not in activities_df.columns:
        print("      No sub-activity data available, skipping...")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, activity in enumerate(['FOOD', 'ERRAND', 'LEISURE']):
        ax = axes[idx]
        
        sub_acts = activities_df[activities_df['activity_type'] == activity]['sub_activity_type'].value_counts().head(10)
        
        if len(sub_acts) > 0:
            sub_acts.plot(kind='barh', ax=ax, color=ACTIVITY_COLORS[activity], alpha=0.7)
            ax.set_xlabel('Count', fontsize=10)
            ax.set_title(f'Top 10 {activity} Sub-Activities', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Sub-Activity Distribution', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/aggregate/07_sub_activity_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_travel_analysis(travels_df, output_dir):
    """Travel analysis with 4 subplots"""
    
    print("  8/8: Travel analysis...")
    
    if len(travels_df) == 0:
        print("      No travel data, skipping...")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distance distribution
    ax = axes[0, 0]
    ax.hist(travels_df['distance_km'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(travels_df['distance_km'].mean(), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {travels_df["distance_km"].mean():.2f} km')
    ax.set_xlabel('Distance (km)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Trip Distance Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Duration distribution
    ax = axes[0, 1]
    ax.hist(travels_df['duration_min'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(travels_df['duration_min'].mean(), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {travels_df["duration_min"].mean():.1f} min')
    ax.set_xlabel('Duration (minutes)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Trip Duration Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Trips per person
    ax = axes[1, 0]
    trips_per_person = travels_df.groupby('sim_id').size()
    ax.hist(trips_per_person, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(trips_per_person.mean(), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {trips_per_person.mean():.1f}')
    ax.set_xlabel('Number of Trips per Person', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Trips per Person Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance vs Duration
    ax = axes[1, 1]
    ax.scatter(travels_df['distance_km'], travels_df['duration_min'], 
              alpha=0.3, s=10, color='purple')
    ax.set_xlabel('Distance (km)', fontsize=11)
    ax.set_ylabel('Duration (minutes)', fontsize=11)
    ax.set_title('Distance vs Duration', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/aggregate/08_travel_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# INDIVIDUAL VISUALIZATIONS
# ============================================================================

def create_individual_text_summary(activities_df, travels_df, sim_id, output_dir):
    """Create a detailed monthly text summary for one person."""
    
    sim_acts = activities_df[activities_df['sim_id'] == sim_id].copy()
    sim_trav = travels_df[travels_df['sim_id'] == sim_id].copy()
    
    sim_acts['start_time'] = pd.to_datetime(sim_acts['start_time'])
    sim_acts['end_time'] = pd.to_datetime(sim_acts['end_time'])
    sim_acts['duration_min'] = pd.to_numeric(sim_acts['duration_min'], errors='coerce')
    sim_trav['duration_min'] = pd.to_numeric(sim_trav['duration_min'], errors='coerce')
    sim_trav['distance_km'] = pd.to_numeric(sim_trav['distance_km'], errors='coerce')
    
    sim_acts = sim_acts[sim_acts['duration_min'] > 0].sort_values('start_time')
    
    lines = []
    sep = "=" * 70
    thin = "-" * 70
    
    # -- Header ------------------------------------------------------------
    lines.append(sep)
    lines.append(f"  MONTHLY ACTIVITY REPORT -- Person {sim_id}")
    lines.append(sep)
    
    if sim_acts.empty:
        lines.append("  No activity data found.")
        _write(lines, output_dir, sim_id)
        return
    
    date_min = sim_acts['start_time'].min()
    date_max = sim_acts['end_time'].max()
    total_days = (date_max - date_min).days + 1
    lines.append(f"  Period : {date_min.strftime('%Y-%m-%d')} -> {date_max.strftime('%Y-%m-%d')}  ({total_days} days)")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # -- Time distribution -------------------------------------------------
    lines.append(sep)
    lines.append("  TIME DISTRIBUTION")
    lines.append(sep)
    time_by_type = sim_acts.groupby('activity_type')['duration_min'].sum()
    total_time = time_by_type.sum()
    for act in ['HOME', 'WORK', 'TRAVEL', 'FOOD', 'ERRAND', 'LEISURE']:
        mins = time_by_type.get(act, 0)
        hrs = mins / 60
        pct = mins / total_time * 100 if total_time > 0 else 0
        bar = '#' * int(pct / 2)  # 50 chars = 100%
        lines.append(f"  {act:<8} {hrs:6.1f}h  {pct:5.1f}%  {bar}")
    lines.append("")
    
    # -- Activity counts & durations ---------------------------------------
    lines.append(sep)
    lines.append("  ACTIVITY STATISTICS")
    lines.append(sep)
    lines.append(f"  {'Type':<10} {'Count':>6}  {'Mean dur':>9}  {'Min':>7}  {'Max':>7}")
    lines.append(thin)
    for act in ['HOME', 'WORK', 'FOOD', 'ERRAND', 'LEISURE']:
        subset = sim_acts[sim_acts['activity_type'] == act]['duration_min']
        if len(subset) == 0:
            continue
        lines.append(
            f"  {act:<10} {len(subset):>6}  "
            f"{subset.mean()/60:>7.2f}h  "
            f"{subset.min()/60:>5.2f}h  "
            f"{subset.max()/60:>5.2f}h"
        )
    lines.append("")
    
    # -- Travel statistics -------------------------------------------------
    lines.append(sep)
    lines.append("  TRAVEL STATISTICS")
    lines.append(sep)
    if not sim_trav.empty:
        lines.append(f"  Total trips      : {len(sim_trav)}")
        lines.append(f"  Total distance   : {sim_trav['distance_km'].sum():.1f} km")
        lines.append(f"  Mean trip dist   : {sim_trav['distance_km'].mean():.2f} km")
        lines.append(f"  Mean trip time   : {sim_trav['duration_min'].mean():.1f} min")
        lines.append(f"  Longest trip     : {sim_trav['distance_km'].max():.2f} km")
        lines.append(f"  Trips per day    : {len(sim_trav) / total_days:.1f}")
    else:
        lines.append("  No travel data.")
    lines.append("")
    
    # -- Sub-activity breakdown --------------------------------------------
    lines.append(sep)
    lines.append("  SUB-ACTIVITY BREAKDOWN")
    lines.append(sep)
    for act in ['FOOD', 'ERRAND', 'LEISURE']:
        subset = sim_acts[sim_acts['activity_type'] == act]
        if 'sub_activity_type' not in subset.columns or subset.empty:
            continue
        counts = subset['sub_activity_type'].value_counts().head(5)
        if counts.empty:
            continue
        lines.append(f"  {act}:")
        for subact, cnt in counts.items():
            if pd.notna(subact):
                lines.append(f"    {subact:<35} {cnt:>4}x")
    lines.append("")
    
    # -- Day-by-day diary --------------------------------------------------
    lines.append(sep)
    lines.append("  DAY-BY-DAY DIARY")
    lines.append(sep)
    
    sim_acts['date'] = sim_acts['start_time'].dt.date
    for day, day_acts in sim_acts.groupby('date'):
        dow = pd.Timestamp(day).strftime('%A')
        lines.append(f"\n  {dow} {day}")
        lines.append(thin)
        for _, row in day_acts.iterrows():
            act_type = row['activity_type']
            sub = f" [{row['sub_activity_type']}]" if pd.notna(row.get('sub_activity_type')) else ""
            t_start = row['start_time'].strftime('%H:%M')
            t_end   = row['end_time'].strftime('%H:%M')
            dur_h   = row['duration_min'] / 60
            lines.append(f"    {t_start}->{t_end}  {act_type:<8}{sub:<30}  {dur_h:.2f}h")
    
    lines.append("")
    lines.append(sep)
    
    # -- Write file --------------------------------------------------------
    os.makedirs(f'{output_dir}/summaries', exist_ok=True)
    out_path = f'{output_dir}/summaries/person_{sim_id}_monthly_report.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [OK] Saved {out_path}")


def _write(lines, output_dir, sim_id):
    os.makedirs(f'{output_dir}/summaries', exist_ok=True)
    with open(f'{output_dir}/summaries/person_{sim_id}_monthly_report.txt', 'w') as f:
        f.write('\n'.join(lines))


def create_individual_visual_summary(activities_df, sim_id, output_dir):
    """
    3-panel visual summary for one agent:
      Top-left  : Activity Clock (24h polar)
      Top-right : Transition Flow (Sankey-style chord)
      Bottom    : Calendar Heatmap (30-day grid)
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch
    from collections import Counter

    sim_data = activities_df[activities_df['sim_id'] == sim_id].copy()
    sim_data['start_time'] = pd.to_datetime(sim_data['start_time'])
    sim_data['end_time']   = pd.to_datetime(sim_data['end_time'])
    sim_data['duration_min'] = pd.to_numeric(sim_data['duration_min'], errors='coerce')
    sim_data = sim_data[sim_data['duration_min'] > 0].sort_values('start_time')

    if sim_data.empty:
        print(f"  No data for person {sim_id}, skipping visual summary.")
        return

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#0f0f1a')

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.2, 1],
        hspace=0.35, wspace=0.3
    )

    ax_clock    = fig.add_subplot(gs[0, 0], projection='polar')
    ax_sankey   = fig.add_subplot(gs[0, 1])
    ax_calendar = fig.add_subplot(gs[1, :])

    DARK_COLORS = {
        'HOME':    '#4a9eda',
        'WORK':    '#ff9500',
        'FOOD':    '#4cd964',
        'ERRAND':  '#ff3b30',
        'LEISURE': '#bf5af2',
        'TRAVEL':  '#8e8e93',
    }
    ACT_ORDER = ['HOME', 'WORK', 'FOOD', 'ERRAND', 'LEISURE', 'TRAVEL']

    # -- PANEL 1: Activity Clock (polar) -----------------------------------
    ax_clock.set_facecolor('#1a1a2e')
    non_home = sim_data[sim_data['activity_type'] != 'HOME']

    for act_type in ACT_ORDER:
        if act_type == 'HOME':
            continue
        subset = non_home[non_home['activity_type'] == act_type]
        if subset.empty:
            continue
        hours = subset['start_time'].dt.hour + subset['start_time'].dt.minute / 60
        hist, bin_edges = np.histogram(hours, bins=48, range=(0, 24))
        theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        width = 2 * np.pi / 48
        bars = ax_clock.bar(
            theta, hist,
            width=width * 0.85,
            color=DARK_COLORS[act_type],
            alpha=0.75,
            label=act_type,
            bottom=0
        )

    ax_clock.set_theta_zero_location('N')
    ax_clock.set_theta_direction(-1)
    ax_clock.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax_clock.set_xticklabels(
        [f'{h:02d}:00' for h in range(24)],
        fontsize=7, color='#cccccc'
    )
    ax_clock.tick_params(colors='#cccccc')
    ax_clock.yaxis.label.set_color('#cccccc')
    ax_clock.set_facecolor('#1a1a2e')
    for spine in ax_clock.spines.values():
        spine.set_edgecolor('#333355')
    ax_clock.set_title(
        f'Activity Clock -- Person {sim_id}',
        fontsize=13, fontweight='bold', color='white', pad=20
    )
    ax_clock.legend(
        loc='upper left', bbox_to_anchor=(1.12, 1.0),
        fontsize=9, framealpha=0.3,
        facecolor='#1a1a2e', labelcolor='white'
    )

    # -- PANEL 2: Transition Flow (chord/Sankey approximation) ------------
    ax_sankey.set_facecolor('#1a1a2e')
    ax_sankey.set_xlim(0, 1)
    ax_sankey.set_ylim(0, 1)
    ax_sankey.axis('off')

    # Count transitions
    transitions = Counter()
    acts_seq = sim_data['activity_type'].tolist()
    for i in range(len(acts_seq) - 1):
        a, b = acts_seq[i], acts_seq[i + 1]
        if a != b:
            transitions[(a, b)] += 1

    # Node positions (left column = origin, right = destination)
    acts_present = [a for a in ACT_ORDER if a in sim_data['activity_type'].values]
    n = len(acts_present)
    y_positions = {act: 0.1 + 0.8 * i / max(n - 1, 1)
                   for i, act in enumerate(acts_present)}

    max_trans = max(transitions.values()) if transitions else 1

    for (src, dst), count in transitions.items():
        if src not in y_positions or dst not in y_positions:
            continue
        lw = 0.5 + 8.0 * (count / max_trans)
        alpha = 0.3 + 0.5 * (count / max_trans)
        color = DARK_COLORS.get(src, '#888888')
        ax_sankey.annotate(
            '', xy=(0.72, y_positions[dst]),
            xytext=(0.28, y_positions[src]),
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=lw, alpha=alpha,
                connectionstyle='arc3,rad=0.2'
            )
        )

    # Node labels
    for act, y in y_positions.items():
        c = DARK_COLORS.get(act, '#888888')
        ax_sankey.add_patch(plt.Circle((0.25, y), 0.025, color=c, zorder=5))
        ax_sankey.text(0.19, y, act, ha='right', va='center',
                       fontsize=10, color='white', fontweight='bold')
        ax_sankey.add_patch(plt.Circle((0.75, y), 0.025, color=c, zorder=5))
        ax_sankey.text(0.81, y, act, ha='left', va='center',
                       fontsize=10, color='white', fontweight='bold')

    ax_sankey.set_title(
        'Activity Transitions',
        fontsize=13, fontweight='bold', color='white', pad=10
    )

    # -- PANEL 3: Calendar Heatmap -----------------------------------------
    ax_calendar.set_facecolor('#1a1a2e')

    sim_data['date'] = sim_data['start_time'].dt.date
    dates = sorted(sim_data['date'].unique())
    start_date = dates[0]

    # Build grid: rows = days of week (0=Mon), cols = week number
    n_weeks = (sim_data['start_time'].max() - sim_data['start_time'].min()).days // 7 + 2
    grid_act  = np.full((7, n_weeks), -1, dtype=int)   # dominant non-home activity
    grid_time = np.zeros((7, n_weeks))                  # total out-of-home hours

    act_to_idx = {a: i for i, a in enumerate(ACT_ORDER)}

    for date, day_data in sim_data.groupby('date'):
        dt = pd.Timestamp(date)
        week = (dt - pd.Timestamp(start_date)).days // 7
        dow  = dt.weekday()
        if week >= n_weeks:
            continue
        grid_time[dow, week] = day_data[
            day_data['activity_type'] != 'HOME'
        ]['duration_min'].sum() / 60

        non_home_day = day_data[~day_data['activity_type'].isin(['HOME', 'TRAVEL'])]
        if not non_home_day.empty:
            dominant = non_home_day.groupby('activity_type')['duration_min'].sum().idxmax()
            grid_act[dow, week] = act_to_idx[dominant]

    cell_w, cell_h = 0.85, 0.7
    days_labels   = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    color_list    = [DARK_COLORS[a] for a in ACT_ORDER]

    for dow in range(7):
        for week in range(n_weeks):
            x = week
            y = 6 - dow
            out_h = grid_time[dow, week]
            act_i = grid_act[dow, week]

            bg_color = '#2a2a3e' if out_h == 0 else '#1e1e30'
            rect = plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=bg_color, edgecolor='#333355', linewidth=0.5
            )
            ax_calendar.add_patch(rect)

            if act_i >= 0 and out_h > 0:
                fill_ratio = min(out_h / 10.0, 1.0)
                color = color_list[act_i]
                fill = plt.Rectangle(
                    (x, y), cell_w * fill_ratio, cell_h,
                    facecolor=color, alpha=0.65, linewidth=0
                )
                ax_calendar.add_patch(fill)
                ax_calendar.text(
                    x + cell_w / 2, y + cell_h / 2,
                    f'{out_h:.1f}h',
                    ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold'
                )

    ax_calendar.set_xlim(-0.5, n_weeks + 0.1)
    ax_calendar.set_ylim(-0.3, 7.2)
    ax_calendar.set_yticks([6 - i + cell_h / 2 for i in range(7)])
    ax_calendar.set_yticklabels(days_labels, fontsize=10, color='#cccccc')
    ax_calendar.set_xticks([w + cell_w / 2 for w in range(n_weeks)])
    ax_calendar.set_xticklabels(
        [f'Wk {w + 1}' for w in range(n_weeks)],
        fontsize=9, color='#cccccc'
    )
    ax_calendar.tick_params(left=False, bottom=False)
    for spine in ax_calendar.spines.values():
        spine.set_visible(False)
    ax_calendar.set_title(
        'Monthly Activity Calendar  (cell color = dominant activity, width = out-of-home hours)',
        fontsize=12, fontweight='bold', color='white', pad=10
    )

    # Calendar legend
    legend_patches = [
        mpatches.Patch(facecolor=DARK_COLORS[a], label=a, alpha=0.75)
        for a in ACT_ORDER if a not in ('HOME', 'TRAVEL')
    ]
    ax_calendar.legend(
        handles=legend_patches,
        loc='lower right', fontsize=9,
        framealpha=0.3, facecolor='#1a1a2e', labelcolor='white'
    )

    fig.suptitle(
        f'Agent Mobility Profile -- Person {sim_id}',
        fontsize=16, fontweight='bold', color='white', y=0.98
    )

    os.makedirs(f'{output_dir}/plots/individuals', exist_ok=True)
    out_path = f'{output_dir}/plots/individuals/visual_summary_person_{sim_id}.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved {out_path}")



# ============================================================================
# FOLIUM MAPS
# ============================================================================

def load_spatial_data():
    """Load spatial environment for map generation"""
    try:
        from region_data import get_spatial_env
        from config import SPATIAL_CACHE_FILE
        spatial_env = get_spatial_env(cache_file=SPATIAL_CACHE_FILE)
        return spatial_env
    except Exception as e:
        print(f" Could not load spatial data: {e}")
        return None


def create_all_pois_map(spatial_env, output_dir):
    """Create interactive map with all POIs color-coded by type"""

    print("\n" + "="*70)
    print("Creating Folium maps...")
    print("  1/4: All POIs...")

    if spatial_env is None:
        print("      No spatial data, skipping...")
        return

    pois = spatial_env.pois

    # Detect coordinate columns (v4 uses x/y, legacy uses lon/lat)
    _lon_col = next((c for c in ['x', 'lon', 'longitude'] if c in pois.columns), None)
    _lat_col = next((c for c in ['y', 'lat', 'latitude']  if c in pois.columns), None)
    if _lon_col is None or _lat_col is None:
        print("      Cannot find coordinate columns, skipping...")
        return

    # Sample POIs if too many
    pois_sample = pois.sample(min(5000, len(pois)), random_state=42)

    # v4 data already in WGS84 -- read directly
    c_lon = float(pois_sample[_lon_col].mean())
    c_lat = float(pois_sample[_lat_col].mean())

    m = folium.Map(location=[c_lat, c_lon], zoom_start=13)

    poi_colors = {
        'accommodation': 'blue', 'sustenance': 'green', 'education': 'orange',
        'work': 'red', 'shop': 'purple', 'errand': 'darkred',
        'entertainment': 'pink', 'tourism': 'lightblue', 'leisure': 'lightgreen'
    }

    for _, poi in pois_sample.iterrows():
        p_lon = float(poi[_lon_col])
        p_lat = float(poi[_lat_col])

        category = poi.get('poi_category', '') or ''
        color = 'gray'
        for key, col in poi_colors.items():
            if key in category:
                color = col
                break

        folium.CircleMarker(
            location=[p_lat, p_lon],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            popup=f"{poi.get('poi_category', 'Unknown')}"
        ).add_to(m)

    m.save(f'{output_dir}/maps/01_all_pois.html')


def create_activity_heatmap(activities_df, spatial_env, output_dir):
    """Create heatmap of general activity locations using pre-computed WGS84 lat/lon."""

    print("  2/4: Activity heatmap...")

    # Use the lat/lon columns already converted during simulation
    valid = activities_df[['lat', 'lon']].dropna()
    if len(valid) == 0:
        print("      No location data, skipping...")
        return

    heat_data = valid.values.tolist()
    center_lat = valid['lat'].mean()
    center_lon = valid['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
    m.save(f'{output_dir}/maps/02_activity_heatmap.html')


def create_frequented_pois_map(activities_df, spatial_env, output_dir):
    """Create map showing frequently visited POIs using pre-computed WGS84 lat/lon."""

    print("  3/4: Frequented POIs...")

    # Use the lat/lon columns stored during simulation
    top_pois = (
        activities_df.dropna(subset=['lat', 'lon'])
        .groupby(['location_id', 'lat', 'lon'])
        .size()
        .reset_index(name='count')
        .nlargest(100, 'count')
    )

    if len(top_pois) == 0:
        print("      No location data, skipping...")
        return

    center_lat = top_pois['lat'].mean()
    center_lon = top_pois['lon'].mean()
    max_visits = top_pois['count'].max()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for _, row in top_pois.iterrows():
        radius = 5 + (row['count'] / max_visits) * 15
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6,
            popup=f"Visits: {row['count']}"
        ).add_to(m)

    m.save(f'{output_dir}/maps/03_frequented_pois.html')


def create_sample_trajectories_map(travels_df, spatial_env, n_sample, output_dir, path_generator=None):
    """Create map with sample trajectories. Road-network paths are transformed
    from EPSG:3857 to WGS84; straight-line fallbacks use pre-computed columns."""

    print(f"  4/4: Sample trajectories (n={n_sample})...")

    if len(travels_df) == 0:
        print("     No data, skipping...")
        return

    sample = travels_df.sample(min(n_sample, len(travels_df)), random_state=42)

    # Map centre from pre-computed WGS84 columns
    valid_origins = sample.dropna(subset=['origin_lat', 'origin_lon'])
    if len(valid_origins) == 0:
        print("     No coordinates found, skipping...")
        return

    m = folium.Map(
        location=[valid_origins['origin_lat'].mean(), valid_origins['origin_lon'].mean()],
        zoom_start=13,
        tiles="CartoDB dark_matter",
    )

    road_count = 0
    straight_count = 0

    for _, trip in sample.iterrows():
        ls_id = trip.get('linestring_id')
        _mode = trip.get('mode', 'car')
        if path_generator and ls_id is not None:
            ls = path_generator.get_linestring(
                int(trip['origin_id']), int(trip['dest_id']), _mode
            )
        else:
            ls = None
        colour = DEST_COLORS.get(trip.get('dest_activity'), '#aaaaaa')

        if ls and len(ls) >= 2:
            # PathGenerator already outputs WGS84 (lon, lat, dist) -- just swap to (lat, lon)
            latlon = [(pt[1], pt[0]) for pt in ls]
            folium.PolyLine(latlon, color=colour, weight=2, opacity=0.7).add_to(m)
            road_count += 1
        else:
            # Straight-line fallback using pre-computed WGS84 columns
            if pd.notna(trip.get('origin_lat')) and pd.notna(trip.get('dest_lat')):
                folium.PolyLine(
                    [[trip['origin_lat'], trip['origin_lon']],
                     [trip['dest_lat'],   trip['dest_lon']]],
                    color=colour, weight=2, opacity=0.6, dash_array='6 4',
                ).add_to(m)
                straight_count += 1

    print(f"     Road paths: {road_count}  |  Straight-line fallbacks: {straight_count}")
    # Legend
    legend_html = """
    <div style="position:fixed;bottom:25px;left:25px;z-index:1000;
                background:rgba(10,10,15,0.92);border:1px solid rgba(255,255,255,0.15);
                border-radius:8px;padding:12px 16px;font-family:monospace;
                font-size:12px;color:#ddd;box-shadow:0 4px 20px rgba(0,0,0,0.5);">
        <b style="color:#fff">Destination</b><br>"""
    for act, col in DEST_COLORS.items():
        legend_html += f'<span style="color:{col}">&#9472;&#9472;</span> {act}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(f'{output_dir}/maps/04_sample_trajectories.html')
def create_individual_trajectory_map(travels_df, spatial_env, sim_id, output_dir, path_generator=None):
    """Create road-network trajectory map for one person.
    Road-network linestrings (EPSG:3857) are transformed to WGS84;
    straight-line fallbacks use pre-computed lat/lon columns."""

    sim_travels = travels_df[travels_df['sim_id'] == sim_id]
    if len(sim_travels) == 0:
        return

    # Map centre from pre-computed WGS84 columns
    valid = sim_travels.dropna(subset=['origin_lat', 'origin_lon'])
    if len(valid) == 0:
        return

    m = folium.Map(
        location=[valid['origin_lat'].mean(), valid['origin_lon'].mean()],
        zoom_start=14,
        tiles="CartoDB dark_matter",
    )

    road_count = 0
    straight_count = 0

    for trip_idx, (_, trip) in enumerate(sim_travels.iterrows()):
        colour = DEST_COLORS.get(trip.get('dest_activity'), '#aaaaaa')
        ls_id = trip.get('linestring_id')
        _mode = trip.get('mode', 'car')
        if path_generator and ls_id is not None:
            ls = path_generator.get_linestring(
                int(trip['origin_id']), int(trip['dest_id']), _mode
            )
        else:
            ls = None

        if ls and len(ls) >= 2:
            # PathGenerator already outputs WGS84 (lon, lat, dist) -- just swap to (lat, lon)
            latlon = [(pt[1], pt[0]) for pt in ls]
            folium.PolyLine(
                latlon, color=colour, weight=3, opacity=0.85,
                tooltip=f"Trip {trip_idx+1}: {trip['distance_km']:.2f} km | {trip['duration_min']:.0f} min",
            ).add_to(m)
            folium.CircleMarker(latlon[0], radius=5, color=colour, fill=True, fill_opacity=1.0).add_to(m)
            folium.Marker(
                latlon[-1],
                icon=folium.DivIcon(
                    html=f'<div style="width:10px;height:10px;background:{colour};border:2px solid white;margin:-5px"></div>',
                    icon_size=(10, 10), icon_anchor=(5, 5),
                ),
            ).add_to(m)
            road_count += 1
        else:
            # Straight-line fallback using pre-computed WGS84 columns
            if pd.notna(trip.get('origin_lat')) and pd.notna(trip.get('dest_lat')):
                folium.PolyLine(
                    [[trip['origin_lat'], trip['origin_lon']],
                     [trip['dest_lat'],   trip['dest_lon']]],
                    color=colour, weight=2, opacity=0.5, dash_array='6 4',
                ).add_to(m)
                folium.CircleMarker(
                    [trip['origin_lat'], trip['origin_lon']],
                    radius=4, color=colour, fill=True, fill_opacity=1.0
                ).add_to(m)
                straight_count += 1

    legend = f"""
    <div style="position:fixed;bottom:25px;left:25px;z-index:1000;
                background:rgba(10,10,15,0.92);border:1px solid rgba(255,255,255,0.15);
                border-radius:8px;padding:12px 16px;font-family:monospace;
                font-size:12px;color:#ddd;box-shadow:0 4px 20px rgba(0,0,0,0.5);">
        <b style="color:#fff">Person {sim_id} &mdash; Destination</b><br>"""
    for act, col in DEST_COLORS.items():
        legend += f'<span style="color:{col}">&#9472;&#9472;</span> {act}<br>'
    legend += f"<br><span style='color:#aaa'>Road paths: {road_count} | Fallback: {straight_count}</span></div>"
    m.get_root().html.add_child(folium.Element(legend))

    m.save(f'{output_dir}/maps/trajectory_person_{sim_id}.html')
    print(f"     Person {sim_id}: {road_count} road paths, {straight_count} fallbacks")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(model_path=None, n_simulations=None, output_dir=None, n_individual_plots=None, n_trajectory_maps=None, use_grid=False):
    # Called from run.py with explicit args, or from CLI via __main__
    if model_path is not None:
        import types
        args = types.SimpleNamespace(
            model=model_path,
            n_sims=n_simulations or 100,
            output=output_dir or 'output/sim/run',
            n_individual_plots=n_individual_plots or 5,
            n_trajectory_maps=n_trajectory_maps or 3,
            n_trajectories_sample=100,
            use_grid=use_grid,
            skip_validation=False,
        )
    else:
        args = None

    if args is None:
        parser = argparse.ArgumentParser(description="Unified simulation and analysis pipeline")
        parser.add_argument("--model", type=str, required=True, help="Path to trained model")
        parser.add_argument("--n_sims", type=int, default=300, help="Number of simulations")
        parser.add_argument("--output", type=str, default="simulation_results", help="Output directory")
        parser.add_argument("--n_individual_plots", type=int, default=5, help="Number of individual timelines")
        parser.add_argument("--n_trajectory_maps", type=int, default=3, help="Number of individual trajectory maps")
        parser.add_argument("--n_trajectories_sample", type=int, default=100, help="Number of trajectories for sample map")
        parser.add_argument("--skip_validation", action="store_true", help="Skip validation metrics")
        parser.add_argument("--use_grid", action="store_true", help="Run grid abstraction layer and save outputs to grid_data/")
        args = parser.parse_args()

    print("="*70)
    print("UNIFIED SIMULATION & ANALYSIS PIPELINE")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Simulations: {args.n_sims}")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Create output directories
    os.makedirs(f'{args.output}/data', exist_ok=True)
    os.makedirs(f'{args.output}/plots/aggregate', exist_ok=True)
    os.makedirs(f'{args.output}/plots/individuals', exist_ok=True)
    os.makedirs(f'{args.output}/maps', exist_ok=True)
    
    # Load PathGenerator once -- shared across simulation and map generation
    path_generator = load_path_library()

    # Step 1: Run simulations
    activities_df, travels_df, hourly_df, metadata_df = run_simulations(
        args.model, args.n_sims, args.output, path_generator=path_generator
    )
    
    # Step 2: Generate text summary
    generate_text_summary(activities_df, travels_df, hourly_df, metadata_df, args.output, args)
    
    # Step 3: Create aggregate visualizations
    print("\n" + "="*70)
    print("CREATING AGGREGATE VISUALIZATIONS")
    print("="*70)
    
    create_hourly_cumulative_plot(hourly_df, args.output)
    create_24x7_heatmap(hourly_df, args.output)
    create_activity_duration_plots(activities_df, args.output)
    create_activity_pie_chart(activities_df, args.output)
    create_weekly_distribution_comparison(activities_df, args.output)
    create_temporal_patterns(activities_df, args.output)
    create_sub_activity_breakdown(activities_df, args.output)
    create_travel_analysis(travels_df, args.output)
    
    print("[OK] Created 8 aggregate visualizations")
    
    # Step 4: Create individual visualizations
    print("=" * 70)
    print("CREATING INDIVIDUAL TEXT SUMMARIES")
    print("=" * 70)
    selected_sims = np.random.choice(
        metadata_df['sim_id'].unique(),
        min(args.n_individual_plots, len(metadata_df)),
        replace=False
    )
    for sim_id in tqdm(selected_sims, desc="Individual summaries"):
        create_individual_text_summary(activities_df, travels_df, sim_id, args.output)
        create_individual_visual_summary(activities_df, sim_id, args.output)

    print(f"[OK] Created {len(selected_sims)} individual summaries and plots")

    
    # Step 5: Create Folium maps
    spatial_env = load_spatial_data()
    
    if spatial_env is not None:
        create_all_pois_map(spatial_env, args.output)
        create_activity_heatmap(activities_df, spatial_env, args.output)
        create_frequented_pois_map(activities_df, spatial_env, args.output)
        create_sample_trajectories_map(travels_df, spatial_env, args.n_trajectories_sample, args.output, path_generator=path_generator)
        
        # Individual trajectory maps
        selected_trajectory_sims = np.random.choice(metadata_df['sim_id'].unique(),
                                                    min(args.n_trajectory_maps, len(metadata_df)),
                                                    replace=False)
        
        print(f"\nCreating {len(selected_trajectory_sims)} individual trajectory maps...")
        for sim_id in tqdm(selected_trajectory_sims, desc="Trajectory maps"):
            create_individual_trajectory_map(travels_df, spatial_env, sim_id, args.output, path_generator=path_generator)
        
        print(f"[OK] Created all maps")
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {args.output}/")
    print("\nGenerated files:")
    print("   Data:")
    print("     - activity_log.csv (with sub_activity_type)")
    print("     - travel_log.csv")
    print("     - hourly_stamps.csv")
    print("     - simulation_metadata.csv")
    print("   Summary:")
    print("     - summary_statistics.txt")
    print("   Aggregate plots (8):")
    print("     - Hourly cumulative, 24 heatmap, durations, pie, etc.")
    print(f"   Individual plots ({len(selected_sims)} timelines)")
    print(f"    Folium maps (4 aggregate + {args.n_trajectory_maps} individual)")
    print("\n" + "="*70)

    # -- Grid layer (optional) ------------------------------------------------
    if args.use_grid:
        run_grid_pipeline(
            activities_df=activities_df,
            travels_df=travels_df,
            path_generator=path_generator,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()