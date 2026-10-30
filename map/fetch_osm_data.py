"""
Fetch OSM data for a city via Geofabrik and prepare it for the MOSAIC pipeline.

Uses Geofabrik's official index (index-v1.json) to find the smallest region
extract that fully contains the given bounding box, downloads it, and clips
it down to the exact bbox with osmium.

Usage:
    python fetch_osm_data.py --bbox <min_lon>,<min_lat>,<max_lon>,<max_lat> --city <city_name>

Output:
    data/raw/<city>/<city>.osm.pbf
    data/raw/<city>/config.json

Requires:
    - requests, shapely (pip)
    - osmium-tool (system package, e.g. `apt install osmium-tool`), for clipping
      the region extract down to the exact bbox. Called via subprocess.

NOTE: This script has not been run against the live Geofabrik index or a real
download. Test it on a small bounding box first and confirm the output before
relying on it for a real city.
"""

import argparse
import json
import os
import subprocess
import sys

import requests
from shapely.geometry import box, shape

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB, for streaming large downloads

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
RAW_DATA_DIR = os.path.abspath(RAW_DATA_DIR)
CACHE_DIR = os.path.join(RAW_DATA_DIR, "_geofabrik_cache")


# ---------------------------------------------------------------------------
# Geofabrik index lookup
# ---------------------------------------------------------------------------

def fetch_geofabrik_index():
    print(f"Fetching Geofabrik index from {GEOFABRIK_INDEX_URL} ...")
    response = requests.get(GEOFABRIK_INDEX_URL, timeout=60)
    response.raise_for_status()
    return response.json()


def find_best_region(index_geojson, bbox_shape):
    """Find the smallest region that fully contains bbox_shape.

    Falls back to the smallest region that merely intersects bbox_shape,
    with a warning, if nothing fully contains it (e.g. bbox straddles a
    region border).

    Returns (feature, fully_contained: bool).
    """
    containing = []
    intersecting = []

    for feature in index_geojson["features"]:
        geom = feature.get("geometry")
        if geom is None:
            continue
        try:
            region_shape = shape(geom)
        except Exception:
            continue

        if not region_shape.intersects(bbox_shape):
            continue

        intersecting.append(feature)
        if region_shape.contains(bbox_shape):
            containing.append(feature)

    def area_of(feature):
        return shape(feature["geometry"]).area

    if containing:
        best = min(containing, key=area_of)
        return best, True

    if intersecting:
        best = min(intersecting, key=area_of)
        return best, False

    return None, False


# ---------------------------------------------------------------------------
# Download + clip
# ---------------------------------------------------------------------------

def download_region_pbf(feature, cache_dir):
    region_id = feature["properties"]["id"]
    pbf_url = feature["properties"]["urls"]["pbf"]

    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = region_id.replace("/", "_") + ".osm.pbf"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"  Using cached download: {cache_path}")
        return cache_path

    print(f"  Downloading {pbf_url} ...")
    with requests.get(pbf_url, stream=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(cache_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"    {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.0f}%)", end="\r")
    print()
    print(f"  Saved to {cache_path}")
    return cache_path


def check_osmium_available():
    import shutil
    if shutil.which("osmium") is None:
        print("[ERROR] osmium-tool not found on PATH.")
        print("  Install it with: apt install osmium-tool  (or your platform's equivalent)")
        sys.exit(1)


def clip_to_bbox(region_pbf_path, min_lon, min_lat, max_lon, max_lat, out_pbf_path):
    """Clip a region extract down to the exact bbox using osmium extract.

    NOTE: not tested. --strategy smart keeps ways/relations whose members
    fall outside the box but are referenced by something inside it, this
    avoids truncating geometry at the box edge. If osmium errors on strategy
    name or flags, check `osmium extract --help` against your installed version.
    """
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    cmd = [
        "osmium", "extract",
        "-b", bbox_str,
        "--strategy", "smart",
        "-o", out_pbf_path,
        "--overwrite",
        region_pbf_path,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] osmium extract failed:\n{result.stderr}")
        sys.exit(1)
    print(f"  Wrote {out_pbf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", required=True,
                         help="min_lon,min_lat,max_lon,max_lat (EPSG:4326)")
    parser.add_argument("--city", required=True,
                         help="Folder name to create under data/raw/")
    args = parser.parse_args()

    check_osmium_available()

    try:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in args.bbox.split(","))
    except ValueError:
        print("[ERROR] --bbox must be 4 comma-separated numbers: min_lon,min_lat,max_lon,max_lat")
        sys.exit(1)

    if min_lon >= max_lon or min_lat >= max_lat:
        print("[ERROR] bbox mins must be less than maxes.")
        sys.exit(1)
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        print("[ERROR] bbox values out of range for EPSG:4326 (lon in [-180,180], lat in [-90,90]).")
        sys.exit(1)

    bbox_shape = box(min_lon, min_lat, max_lon, max_lat)

    index_geojson = fetch_geofabrik_index()

    print("Searching for the smallest containing region...")
    feature, fully_contained = find_best_region(index_geojson, bbox_shape)

    if feature is None:
        print("[ERROR] No Geofabrik region intersects this bbox. Check your coordinates.")
        sys.exit(1)

    region_id = feature["properties"]["id"]
    if fully_contained:
        print(f"  Found: {region_id} (fully contains bbox)")
    else:
        print(f"[WARNING] No single region fully contains this bbox.")
        print(f"  Falling back to the smallest intersecting region: {region_id}")
        print(f"  Your bbox may straddle a region border, verify the clipped output covers your full area of interest.")

    region_pbf_path = download_region_pbf(feature, CACHE_DIR)

    city_dir = os.path.join(RAW_DATA_DIR, args.city)
    os.makedirs(city_dir, exist_ok=True)
    out_pbf_path = os.path.join(city_dir, f"{args.city}.osm.pbf")

    print("Clipping to exact bbox...")
    clip_to_bbox(region_pbf_path, min_lon, min_lat, max_lon, max_lat, out_pbf_path)

    config_path = os.path.join(city_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({"bbox": [min_lon, min_lat, max_lon, max_lat]}, f, indent=2)
    print(f"  Wrote {config_path}")

    print(f"\nDone. {args.city} is ready for: python run.py --pipeline {args.city}")


if __name__ == "__main__":
    main()