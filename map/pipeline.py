"""
CITY DATA PIPELINE
==================
Transforms raw OSM data into simulation-ready spatial caches.

Usage (via run.py -- recommended):
    python run.py --pipeline bonn

Or directly:
    python pipeline.py bonn
    python pipeline.py bonn --from-step 3

Input  (user provides):
    data/raw/<city>/          <- create this folder
    data/raw/<city>/<any>.pbf <- drop PBF here (any filename, one file only)
    data/raw/<city>/config.json

config.json format:
    {
        "bbox": [min_lon, min_lat, max_lon, max_lat]
    }

    Get your bbox from BBBike (https://extract.bbbike.org) when you
    draw your bounding box. The PBF from BBBike is already clipped --
    the bbox here is used only by city_map.sql as a spatial filter.

Output (auto-generated):
    data/formatted/<city>/
        spatial_cache.pkl           -- step 3
        spatial_cache_filtered.pkl  -- step 4
        poi_density_cache.pkl       -- step 5
        air_distance_cache.pkl      -- step 6

Pipeline steps:
    1. Validate inputs (PBF + config.json present)
    2. Create PostgreSQL database + extensions
    3. osm2pgsql import -> PostgreSQL
    4. city_map.sql -> road network + POI tables
    5. export_city.py -> spatial_cache.pkl
    6. filter_connected_pois.py -> spatial_cache_filtered.pkl
    7. precompute_poi_density.py -> poi_density_cache.pkl
    8. precompute_air_distances.py -> air_distance_cache.pkl

Requirements:
    osm2pgsql, psql, createdb on PATH (or set full paths in TOOL_PATHS below)
    PostgreSQL running locally with PostGIS + pgRouting available
"""

import sys
import os
import json
import time
import glob
import subprocess

# Path setup
_map  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_map)
_src  = os.path.join(_root, "src")
for _p in [_root, _src] + [os.path.join(_src, d) for d in ("simulation", "spatial", "analysis")]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from custom_settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD

# =============================================================================
# TOOL PATHS
# =============================================================================
# On Linux/macOS these are usually on PATH -- set to just the tool name.
# On Windows, set the full path to each executable.
#
# Find on Windows:
#   Get-ChildItem "C:\Program Files\PostgreSQL" -Recurse -Filter psql.exe
#   Test-Path "C:\Program Files\osm2pgsql\osm2pgsql.exe"
#
# Linux/macOS defaults (uncomment if needed):
# OSM2PGSQL = "osm2pgsql"
# PSQL      = "psql"
# CREATEDB  = "createdb"

OSM2PGSQL = r"C:\Program Files\osm2pgsql\osm2pgsql.exe"
PSQL      = r"C:\Program Files\PostgreSQL\17\bin\psql.exe"
CREATEDB  = r"C:\Program Files\PostgreSQL\17\bin\createdb.exe"


# =============================================================================
# HELPERS
# =============================================================================

def banner(msg):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def run(cmd, env=None, check=True, description=""):
    if description:
        print(f"\n>> {description}")
    print(f"   {' '.join(str(c) for c in cmd)}\n")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    t0 = time.time()
    result = subprocess.run(cmd, env=merged)
    elapsed = time.time() - t0
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed (exit {result.returncode}) after {elapsed:.0f}s")
        sys.exit(result.returncode)
    print(f"\n[OK] Done in {elapsed:.0f}s")
    return result.returncode


def pg_env():
    # PAGER="" disables psql paging so output flows without user input
    return {"PGPASSWORD": DB_PASSWORD, "PAGER": ""}


def db_name(city):
    """Database name convention: <city>_osm"""
    return f"{city}_osm"


def raw_dir(city):
    return os.path.join(_root, "data", "raw", city)


def formatted_dir(city):
    d = os.path.join(_root, "data", "formatted", city)
    os.makedirs(d, exist_ok=True)
    return d



# =============================================================================
# STEP 0: PRE-FLIGHT CHECK
# =============================================================================

def step_preflight():
    """Verify required tools are installed before running anything."""
    banner("Step 0: Checking required tools")

    import shutil

    tools = {
        PSQL:      ("psql",      "PostgreSQL client",
                    "Linux: sudo apt install postgresql\n"
                    "    Mac:   brew install postgresql\n"
                    "    Win:   https://www.postgresql.org/download/windows/"),
        CREATEDB:  ("createdb",  "PostgreSQL client",
                    "Included with PostgreSQL -- see psql instructions above"),
        OSM2PGSQL: ("osm2pgsql", "osm2pgsql",
                    "Linux: sudo apt install osm2pgsql\n"
                    "    Mac:   brew install osm2pgsql\n"
                    "    Win:   https://osm2pgsql.org/doc/install.html"),
    }

    all_ok = True
    for executable, (name, label, install_hint) in tools.items():
        # Check if it exists either as a full path or on PATH
        found = os.path.isfile(executable) or shutil.which(executable) is not None
        if found:
            # Get version
            try:
                result = subprocess.run(
                    [executable, "--version"],
                    capture_output=True, text=True, timeout=5
                )
                version = result.stdout.strip().split("\n")[0]
            except Exception:
                version = "version unknown"
            print(f"  [OK] {label} ({version})")
        else:
            print(f"  [FAIL] {label} -- NOT FOUND")
            print(f"    Install: {install_hint}")
            all_ok = False

    if not all_ok:
        print(f"\n[ERROR] Missing required tools -- see install instructions above.")
        print(f"  After installing, re-run: python run.py --pipeline <city>")
        sys.exit(1)

    print(f"\n  All tools found. Starting pipeline...")

# =============================================================================
# STEP 1: VALIDATE
# =============================================================================

def step_validate(city):
    banner(f"Step 1: Validating inputs -- {city}")

    rdir = raw_dir(city)
    if not os.path.isdir(rdir):
        print(f"[ERROR] Raw data folder not found: {rdir}")
        print(f"  Create it and drop your PBF file inside:")
        print(f"  data/raw/{city}/<your_city>.pbf")
        sys.exit(1)

    # Find PBF
    pbfs = glob.glob(os.path.join(rdir, "*.pbf")) + glob.glob(os.path.join(rdir, "*.osm.pbf"))
    if not pbfs:
        print(f"[ERROR] No PBF file found in {rdir}")
        print(f"  Download from https://extract.bbbike.org")
        print(f"  Place the .pbf file inside data/raw/{city}/")
        sys.exit(1)
    if len(pbfs) > 1:
        print(f"[WARNING] Multiple PBF files found -- using: {os.path.basename(pbfs[0])}")
    pbf = pbfs[0]
    print(f"  PBF: {os.path.basename(pbf)}")

    # Find config.json
    cfg_path = os.path.join(rdir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"[ERROR] config.json not found: {cfg_path}")
        print(f"\n  Create data/raw/{city}/config.json with your bounding box:")
        print(f"  {{")
        print(f"    \"bbox\": [min_lon, min_lat, max_lon, max_lat]")
        print(f"  }}")
        print(f"\n  You can find your bbox coordinates from BBBike or Geofabrik")
        print(f"  when you download your PBF file.")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)

    if "bbox" not in cfg:
        print(f"[ERROR] config.json missing 'bbox' key")
        sys.exit(1)

    bbox = cfg["bbox"]
    if len(bbox) != 4:
        print(f"[ERROR] bbox must have 4 values: [min_lon, min_lat, max_lon, max_lat]")
        sys.exit(1)

    print(f"  bbox: {bbox}")
    print(f"  Output dir: data/formatted/{city}/")
    return pbf, bbox


# =============================================================================
# STEP 2: CREATE DATABASE
# =============================================================================

def step_create_db(city):
    banner(f"Step 2: Create database -- {db_name(city)}")

    # createdb -- don't exit if DB already exists
    rc = run(
        [CREATEDB, "--username", DB_USER, "--host", DB_HOST,
         "--port", str(DB_PORT), db_name(city)],
        env=pg_env(), check=False,
        description=f"Creating {db_name(city)}"
    )
    if rc == 0:
        print(f"  Database {db_name(city)} created.")
    else:
        print(f"  Database {db_name(city)} already exists -- continuing.")

    run(
        [PSQL, "--username", DB_USER, "--host", DB_HOST,
         "--port", str(DB_PORT), "--dbname", db_name(city),
         "--no-psqlrc", "--command",
         "CREATE EXTENSION IF NOT EXISTS postgis; "
         "CREATE EXTENSION IF NOT EXISTS pgrouting; "
         "CREATE EXTENSION IF NOT EXISTS hstore;"],
        env=pg_env(),
        description="Enabling PostGIS, pgRouting, hstore"
    )


# =============================================================================
# STEP 3: osm2pgsql IMPORT
# =============================================================================

def step_import(city, pbf):
    banner(f"Step 3: osm2pgsql import -- {city}")

    lua = os.path.join(_map, "map.lua")
    if not os.path.exists(lua):
        print(f"[ERROR] map.lua not found: {lua}")
        sys.exit(1)

    run(
        [OSM2PGSQL,
         "--database", db_name(city),
         "--user", DB_USER,
         "--host", DB_HOST,
         "--port", str(DB_PORT),
         "--output", "flex",
         "--style", lua,
         "--create",
         pbf],
        env=pg_env(),
        description=f"Importing {os.path.basename(pbf)} -> {db_name(city)}"
    )


# =============================================================================
# STEP 4: city_map.sql
# =============================================================================

def step_sql(city, bbox):
    banner(f"Step 4: city_map.sql -- {city}")

    sql = os.path.join(_map, "city_map.sql")
    if not os.path.exists(sql):
        print(f"[ERROR] city_map.sql not found: {sql}")
        sys.exit(1)

    min_lon, min_lat, max_lon, max_lat = bbox
    run(
        [PSQL,
         "--dbname", db_name(city),
         "--username", DB_USER,
         "--host", DB_HOST,
         "--port", str(DB_PORT),
         "--no-psqlrc",
         "--file", sql,
         "--variable", f"city={city}",
         "--variable", f"min_lon={min_lon}",
         "--variable", f"min_lat={min_lat}",
         "--variable", f"max_lon={max_lon}",
         "--variable", f"max_lat={max_lat}",
         "--echo-errors"],
        env=pg_env(),
        description=f"city_map.sql -> {db_name(city)}"
    )


# =============================================================================
# STEP 5: EXPORT
# =============================================================================

def step_export(city):
    banner(f"Step 5: export_city.py -- {city}")

    export_script = os.path.join(_map, "export_city.py")
    fdir = formatted_dir(city)
    output = os.path.join(fdir, "spatial_cache.pkl")

    run(
        [sys.executable, export_script, "--city", city, "--output", output],
        description=f"export_city.py -> {output}"
    )


# =============================================================================
# STEP 6: FILTER
# =============================================================================

def step_filter(city):
    banner(f"Step 6: filter_connected_pois.py -- {city}")

    filter_script = os.path.join(_map, "filter_connected_pois.py")
    fdir  = formatted_dir(city)
    raw   = os.path.join(fdir, "spatial_cache.pkl")
    filt  = os.path.join(fdir, "spatial_cache_filtered.pkl")

    run(
        [sys.executable, filter_script, "--input", raw, "--output", filt],
        description=f"filter -> {filt}"
    )


# =============================================================================
# STEP 7: POI DENSITY
# =============================================================================

def step_density(city):
    banner(f"Step 7: precompute_poi_density.py -- {city}")

    fdir   = formatted_dir(city)
    input_ = os.path.join(fdir, "spatial_cache_filtered.pkl")
    output = os.path.join(fdir, "poi_density_cache.pkl")
    script = os.path.join(_map, "precompute_poi_density.py")

    env = os.environ.copy()
    env["MOBILITY_SPATIAL_CACHE"] = input_
    env["MOBILITY_OUTPUT"]        = output
    env["PYTHONPATH"]             = os.pathsep.join([
        _root, _src,
        os.path.join(_src, "simulation"),
        os.path.join(_src, "spatial"),
        os.path.join(_src, "analysis"),
        _map,
    ] + [env.get("PYTHONPATH", "")])

    run([sys.executable, script], env=env, description=f"poi_density -> {output}")


# =============================================================================
# STEP 8: AIR DISTANCES
# =============================================================================

def step_air(city):
    banner(f"Step 8: precompute_air_distances.py -- {city}")

    fdir   = formatted_dir(city)
    input_ = os.path.join(fdir, "spatial_cache_filtered.pkl")
    output = os.path.join(fdir, "air_distance_cache.pkl")
    script = os.path.join(_map, "precompute_air_distances.py")

    env = os.environ.copy()
    env["MOBILITY_SPATIAL_CACHE"] = input_
    env["MOBILITY_OUTPUT"]        = output
    env["PYTHONPATH"]             = os.pathsep.join([
        _root, _src,
        os.path.join(_src, "simulation"),
        os.path.join(_src, "spatial"),
        os.path.join(_src, "analysis"),
        _map,
    ] + [env.get("PYTHONPATH", "")])

    run([sys.executable, script], env=env, description=f"air_distances -> {output}")


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(city, from_step=1):
    banner(f"CITY PIPELINE -- {city.upper()}  (steps {from_step}-8)")

    t_total = time.time()
    pbf, bbox = None, None

    step_preflight()  # always runs regardless of from_step

    if from_step <= 1:
        pbf, bbox = step_validate(city)
    if from_step <= 2:
        step_create_db(city)
    if from_step <= 3:
        if pbf is None:
            pbfs = glob.glob(os.path.join(raw_dir(city), "*.pbf"))
            pbf = pbfs[0] if pbfs else None
        step_import(city, pbf)
    if from_step <= 4:
        if bbox is None:
            cfg = json.load(open(os.path.join(raw_dir(city), "config.json")))
            bbox = cfg["bbox"]
        step_sql(city, bbox)
    if from_step <= 5:
        step_export(city)
    if from_step <= 6:
        step_filter(city)
    if from_step <= 7:
        step_density(city)
    if from_step <= 8:
        step_air(city)

    elapsed = time.time() - t_total
    banner(f"PIPELINE COMPLETE -- {city.upper()}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"\n  Output: data/formatted/{city}/")
    fdir = formatted_dir(city)
    for fname in ["spatial_cache.pkl", "spatial_cache_filtered.pkl",
                  "poi_density_cache.pkl", "air_distance_cache.pkl"]:
        fpath = os.path.join(fdir, fname)
        if os.path.exists(fpath):
            mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    [OK] {fname}  ({mb:.1f} MB)")
        else:
            print(f"    [FAIL] {fname}  (not found)")
    print(f"\n  Next: python run.py --train {city}")
    print(f"  Then: python run.py --train {city}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="City data pipeline")
    parser.add_argument("city", help="City folder name inside data/raw/")
    parser.add_argument("--from-step", type=int, default=1, metavar="N",
                        help="Start from step N (1-8, default: 1)")
    args = parser.parse_args()
    run_pipeline(args.city, args.from_step)