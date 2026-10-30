"""
DATA SETUP
==========
Checks that all required data files exist and generates any that are missing.
Run this before training, or let train.py call it automatically.

City is set by the MOBILITY_CITY environment variable (via run.py --train/--sim <city>).
MAP_FOLDER in custom_settings.py is the fallback default when running standalone.

Pipeline (in order):
  1. data/<city>/spatial_cache.pkl          -- raw OSM export (from pipeline.py)
  2. data/<city>/spatial_cache_filtered.pkl -- filter_connected_pois.py
  3. data/<city>/poi_density_cache.pkl      -- precompute_poi_density.py
  4. data/<city>/air_distance_cache.pkl     -- precompute_air_distances.py
  5. data/<city>/travel_times_cache.pkl     -- pre_compute_travel_times.py (optional)

Note: path_library*.pkl builds lazily on demand during simulation.
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p


import os
import sys
import subprocess
import time


def _get_paths():
    try:
        import config as _cfg
        return {
            "map_folder": _cfg.MAP_FOLDER,
            "raw":        _cfg.RAW_CACHE_FILE,
            "filtered":   _cfg.SPATIAL_CACHE_FILE,
            "density":    _cfg.POI_DENSITY_FILE,
            "air_dist":   _cfg.AIR_DISTANCE_FILE,
            "travel":     _cfg.TRAVEL_TIMES_CACHE_FILE,
        }
    except Exception as e:
        print(f"[ERROR] Could not import config.py: {e}")
        sys.exit(1)


def exists(path):
    return os.path.exists(path)


def size_mb(path):
    return os.path.getsize(path) / 1024 / 1024


def run(script, *args):
    cmd = [sys.executable, script] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\nv {script} completed in {elapsed:.0f}s")


def main(force=False):
    paths = _get_paths()

    print("=" * 60)
    print(f"DATA SETUP -- map: {paths['map_folder']}")
    print("=" * 60)

    # Step 0: raw spatial cache must exist (produced by pipeline.py)
    if not exists(paths["raw"]):
        print(f"\n[ERROR] Raw spatial cache not found: {paths['raw']}")
        print(f"  Run: python pipeline.py --city {paths['map_folder']}")
        sys.exit(1)
    print(f"v {paths['raw']}  ({size_mb(paths['raw']):.0f} MB)")

    # Step 1: filtered cache
    if force or not exists(paths["filtered"]):
        print(f"\nx {paths['filtered']} -- generating...")
        run("filter_connected_pois.py",
            "--input",  paths["raw"],
            "--output", paths["filtered"])
        if not exists(paths["filtered"]):
            print(f"[ERROR] filter_connected_pois.py did not produce {paths['filtered']}")
            sys.exit(1)
    print(f"v {paths['filtered']}  ({size_mb(paths['filtered']):.0f} MB)")

    # Step 2: POI density cache
    if force or not exists(paths["density"]):
        print(f"\nx {paths['density']} -- generating...")
        run("precompute_poi_density.py")
        if not exists(paths["density"]):
            print(f"[ERROR] precompute_poi_density.py did not produce {paths['density']}")
            sys.exit(1)
    print(f"v {paths['density']}  ({size_mb(paths['density']):.0f} MB)")

    # Step 3: air distance matrix
    if force or not exists(paths["air_dist"]):
        print(f"\nx {paths['air_dist']} -- generating...")
        run("precompute_air_distances.py")
        if not exists(paths["air_dist"]):
            print(f"[ERROR] precompute_air_distances.py did not produce {paths['air_dist']}")
            sys.exit(1)
    print(f"v {paths['air_dist']}  ({size_mb(paths['air_dist']):.0f} MB)")

    # Step 4: travel times cache (optional)
    try:
        import config as _cfg
        use_precomputed = bool(getattr(_cfg, "USE_PRECOMPUTED_TRAVEL_TIMES", False))
    except Exception:
        use_precomputed = False

    if not use_precomputed:
        print(f"\n- Skipping travel times cache (USE_PRECOMPUTED_TRAVEL_TIMES=False)")
        print(f"  PathGenerator handles routing on demand.")
    elif force or not exists(paths["travel"]):
        print(f"\nx {paths['travel']} -- generating...")
        print("  (This computes all-pairs travel times -- may take 10-30 min)")
        run("pre_compute_travel_times.py")
        if not exists(paths["travel"]):
            print(f"[ERROR] pre_compute_travel_times.py did not produce {paths['travel']}")
            sys.exit(1)
    else:
        print(f"v {paths['travel']}  ({size_mb(paths['travel']):.0f} MB)")

    print("\n" + "=" * 60)
    print(f"v All data files present -- ready to train ({paths['map_folder']})")
    print("  Note: path_library*.pkl builds lazily during simulation")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data files for training")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate all files even if they already exist")
    args = parser.parse_args()
    main(force=args.force)