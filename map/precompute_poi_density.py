"""
PRECOMPUTE POI DENSITY CACHE
=============================
Calculates POI density (POIs per km^2 within 500m radius) for all POIs
and saves to cache file for fast loading during training/simulation.

OPTIMIZED: Uses vectorized numpy operations with batch processing.
This only needs to be run once per spatial dataset.

Usage:
    python precompute_poi_density.py

Output:
    data/poi_density_cache.pkl
"""

import sys as _sys, os as _os
_map  = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_map)
_src  = _os.path.join(_root, 'src')
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _map, _root, _src, _p


import numpy as np
import pandas as pd
import pickle
from region_data import get_spatial_env
import os as _os
from config import SPATIAL_CACHE_FILE as _DEFAULT_CACHE, POI_DENSITY_FILE as _DEFAULT_OUTPUT
SPATIAL_CACHE_FILE = _os.environ.get('MOBILITY_SPATIAL_CACHE', _DEFAULT_CACHE)
POI_DENSITY_FILE   = _os.environ.get('MOBILITY_OUTPUT', _DEFAULT_OUTPUT)
from tqdm import tqdm
import os


def compute_poi_density(spatial_env):
    """
    Compute POI density for all POIs using vectorized operations.
    
    OPTIMIZED: Uses numpy broadcasting and batch processing instead of nested loops.
    This is MUCH faster than the naive O(n^2) approach.
    
    For each POI, counts how many POIs of each type are within 500m radius,
    then calculates density as: count / area (POIs per km^2)
    
    Returns:
        dict: {poi_id: {'FOOD': density, 'ERRAND': density, ...}}
    """
    
    print("="*70)
    print("COMPUTING POI DENSITY (VECTORIZED)")
    print("="*70)
    
    pois = spatial_env.pois
    
    # Detect coordinate columns
    coord_cols = None
    for x_col, y_col in [('x', 'y'), ('lon', 'lat'), ('longitude', 'latitude')]:
        if x_col in pois.columns and y_col in pois.columns:
            coord_cols = (x_col, y_col)
            break
    
    if coord_cols is None:
        raise ValueError("Could not find coordinate columns in POI data")
    
    x_col, y_col = coord_cols
    
    print(f"\nFound {len(pois):,} POIs")
    print(f"Coordinate columns: {x_col}, {y_col}")
    
    # EXACT category -> macro map, inverted from
    # src/spatial/region_representation.py's `location_to_osm_categories`,
    # which is the single source of truth for this version. Keep the two in
    # sync; the coverage check below fails loudly if the data contains a
    # category this map does not know.
    #
    # The mapping must be exact rather than by substring: 'shop' matches both
    # shop_supermarket (ERRAND) and shop_mall (LEISURE), so a substring rule
    # cannot express this taxonomy and drifts from it silently.
    location_to_osm_categories = {
        'HOME': [
            'home_residential',
        ],
        'WORK': [
            'education_college', 'education_university', 'education_school',
            'education_library', 'education_kindergarten',
            'work_office', 'work_government', 'work_research',
            'work_industrial', 'work_retail',
        ],
        'FOOD': [
            'sustenance_restaurant', 'sustenance_fastfood',
            'sustenance_food_court',
        ],
        'ERRAND': [
            'errand_bank', 'errand_atm', 'errand_post', 'errand_pharmacy',
            'errand_townhall', 'errand_courthouse', 'errand_police',
            'healthcare_doctor', 'healthcare_dentist', 'healthcare_hospital',
            'healthcare_clinic',
            'shop_supermarket', 'shop_convenience', 'shop_bakery',
            'shop_hairdresser',
        ],
        'LEISURE': [
            'entertainment_cinema', 'entertainment_theatre',
            'entertainment_nightclub', 'entertainment_community',
            'entertainment_arts', 'entertainment_casino',
            'tourism_museum', 'tourism_gallery', 'tourism_attraction',
            'tourism_viewpoint',
            'shop_mall', 'shop_clothing', 'shop_books', 'shop_department',
            'leisure_fitness', 'leisure_sports', 'leisure_pool',
            'leisure_park', 'leisure_garden', 'leisure_playground',
            'social_worship', 'social_centre',
            'sustenance_cafe', 'sustenance_bar', 'sustenance_pub',
            'sustenance_biergarten',
        ],
    }
    category_to_activity = {
        cat: macro
        for macro, cats in location_to_osm_categories.items()
        for cat in cats
    }

    # Add activity_type column to POIs
    def get_activity_type(poi_category):
        """Map POI category to activity type by EXACT category name."""
        if pd.isna(poi_category):
            return 'OTHER'
        return category_to_activity.get(str(poi_category).strip().lower(), 'OTHER')


    cat_col = None
    for col in ['poi_category', 'category', 'type', 'sub_type']:
        if col in pois.columns:
            cat_col = col
            break
    
    if cat_col is None:
        print("[WARNING] Warning: No category column found, all POIs will be 'OTHER'")
        pois['activity_type'] = 'OTHER'
    else:
        pois['activity_type'] = pois[cat_col].apply(get_activity_type)
    
    print(f"\nActivity type distribution:")
    for activity, count in pois['activity_type'].value_counts().items():
        print(f"  {activity:10} {count:6,} POIs")

    # Coverage check: any category present in the data but absent from the map
    # above contributes to no macro, so it is reported by name, and a large
    # unmapped share is an error rather than a warning. Without this a drifted
    # map simply produces quietly understated densities.
    if cat_col is not None:
        _unmapped = pois.loc[pois['activity_type'] == 'OTHER', cat_col].astype(str)
        _other_share = len(_unmapped) / max(len(pois), 1)
        if len(_unmapped):
            print(f"\n[!] {len(_unmapped):,} POIs ({_other_share*100:.1f}%) have a "
                  f"category not present in location_to_osm_categories:")
            for _c, _n in _unmapped.value_counts().head(20).items():
                print(f"      {_c:32} {_n:6,}")
            print("    These contribute to NO density macro. If any of them "
                  "should, add it to the map above AND to "
                  "src/spatial/region_representation.py.")
        if _other_share > 0.10:
            raise ValueError(
                f"{_other_share*100:.1f}% of POIs map to 'OTHER' -- the "
                f"category map has drifted from the data's taxonomy. Densities "
                f"built from this would under-count most macros. Reconcile the "
                f"map with region_representation.py before continuing."
            )
    
    # Parameters
    search_radius_km = 0.5  # 500 meters
    search_radius_m = search_radius_km * 1000.0
    area_km2 = np.pi * (search_radius_km ** 2)
    
    print(f"\nRadius: {search_radius_km*1000:.0f}m")
    print(f"Area: {area_km2:.4f} km^2")
    
    # Extract coordinates and types as numpy arrays for vectorized operations
    poi_ids = pois['poi_id'].values
    coords = pois[[x_col, y_col]].values.astype(float)  # Shape: (n_pois, 2)
    activity_types = pois['activity_type'].values

    # Units are decided ONCE here, from the values rather than the column name,
    # and asserted loudly. A cache whose units are misread produces a constant
    # density, which looks plausible in aggregate and is easy to miss.
    _absx = np.nanmax(np.abs(coords[:, 0]))
    _absy = np.nanmax(np.abs(coords[:, 1]))
    is_degrees = bool(_absx <= 180.0 and _absy <= 90.0)
    print(f"Coordinate units  : {'DEGREES (lon/lat)' if is_degrees else 'METRES (projected)'} "
          f"(max |x|={_absx:,.2f}, max |y|={_absy:,.2f})")
    if not is_degrees and (_absx < 1000.0 or _absy < 1000.0):
        raise ValueError(
            f"Coordinate units are ambiguous: max |x|={_absx}, max |y|={_absy}. "
            f"Neither clearly degrees (<=180/<=90) nor clearly projected metres "
            f"(>=1000). Refusing to guess -- a wrong guess here silently yields "
            f"a constant density for every POI."
        )
    
    # Pre-compute activity masks for faster lookup
    activity_masks = {
        'HOME': activity_types == 'HOME',
        'WORK': activity_types == 'WORK',
        'FOOD': activity_types == 'FOOD',
        'ERRAND': activity_types == 'ERRAND',
        'LEISURE': activity_types == 'LEISURE'
    }
    
    # Compute density for each POI using VECTORIZED operations
    print(f"\nComputing density for {len(pois):,} POIs (vectorized batches)...")
    
    density_cache = {}
    
    # Process in batches to avoid memory issues with large distance matrices
    batch_size = 1000
    n_pois = len(poi_ids)
    
    for batch_start in tqdm(range(0, n_pois, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, n_pois)
        batch_coords = coords[batch_start:batch_end]  # Shape: (batch_size, 2)
        
        # Vectorized distance computation: broadcast to compute all pairwise distances
        # Shape: (batch_size, n_pois)
        diff = batch_coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (batch, n_pois, 2)

        # The degrees-vs-metres decision comes from the coordinate VALUES (see
        # is_degrees above), never from the column name: the column search
        # matches ('x', 'y') first, and those hold lon/lat in degrees in every
        # spatial cache here. Treating degrees as metres would put every pair
        # inside the 500 m radius and reduce each density to the city-wide
        # category total. Note this frame has no projected columns, so the unit
        # test is the only place the distinction can be made.
        if is_degrees:
            # Geographic coordinates - equirectangular approximation, which is
            # well under 1% error over a single city at a 500 m threshold.
            lat_factor = 111000.0  # meters per degree latitude
            lon_factor = 111000.0 * np.cos(np.radians(np.mean(coords[:, 1])))  # adjust for latitude
            distances_m = np.sqrt((diff[:, :, 0] * lon_factor)**2 + (diff[:, :, 1] * lat_factor)**2)
        else:
            # Projected coordinates - already metres
            distances_m = np.sqrt(np.sum(diff ** 2, axis=2))  # (batch, n_pois)
        
        # Find nearby POIs within radius
        nearby_mask = distances_m <= search_radius_m  # (batch, n_pois)
        
        # Set diagonal to False (POI not nearby to itself)
        for i in range(batch_end - batch_start):
            nearby_mask[i, batch_start + i] = False
        
        # Count nearby POIs by type for each POI in batch
        for i, poi_id in enumerate(poi_ids[batch_start:batch_end]):
            nearby_counts = {}
            for activity, mask in activity_masks.items():
                # Count POIs that are both nearby AND of this activity type
                count = np.sum(nearby_mask[i] & mask)
                nearby_counts[activity] = count / area_km2
            
            density_cache[poi_id] = nearby_counts
    
    print(f"\n[OK] Computed density for {len(density_cache):,} POIs")
    
    # Print statistics
    print("\nDensity statistics (POIs per km^2):")
    
    for activity in ['FOOD', 'ERRAND', 'LEISURE', 'WORK', 'HOME']:
        all_densities = [d[activity] for d in density_cache.values()]
        
        if len(all_densities) > 0:
            _a = np.asarray(all_densities, dtype=float)
            _cv = float(_a.std() / max(abs(_a.mean()), 1e-12))
            print(f"  {activity:10} "
                  f"Mean: {_a.mean():8.2f}  "
                  f"Max: {_a.max():8.2f}  "
                  f"Min: {_a.min():8.2f}  "
                  f"CV: {_cv:.3f}")

    # A density that does not vary is not a density: any category with real
    # POIs must show spatial variation. A CV at the 1e-3 level means the
    # radius filter is not discriminating and the cache must not be used, so
    # this is a hard failure rather than a warning.
    print("\nSelf-check: spatial variation per category")
    degenerate = []
    for activity in ['FOOD', 'ERRAND', 'LEISURE', 'WORK', 'HOME']:
        _a = np.asarray([d[activity] for d in density_cache.values()], dtype=float)
        if _a.size == 0 or _a.mean() == 0.0:
            # Legitimately absent category (no POIs of this macro at all).
            print(f"  {activity:10} all zero -- no POIs of this macro (not checked)")
            continue
        _cv = float(_a.std() / max(abs(_a.mean()), 1e-12))
        if _cv < 0.01:
            degenerate.append((activity, _cv, _a.mean()))
            print(f"  {activity:10} [FAIL] CV={_cv:.6f} -- effectively constant")
        else:
            print(f"  {activity:10} [OK]   CV={_cv:.3f}")
    if degenerate:
        raise ValueError(
            "POI density is effectively CONSTANT for: "
            + ", ".join(f"{a} (CV={c:.2e}, mean={m:.2f})" for a, c, m in degenerate)
            + ".\nThe radius filter is matching every POI, so each 'density' "
              "is the city-wide category total / area. Check the "
              "degrees-vs-metres decision above before using this cache -- a "
              "constant density silently makes poi_selector's POPULARITY term "
              "inert, with no other visible symptom."
        )

    return density_cache


def save_density_cache(density_cache, output_file=None):
    if output_file is None:
        import os as _os2
        output_file = _os2.environ.get('MOBILITY_OUTPUT', POI_DENSITY_FILE)
    """Save density cache to file"""
    
    print(f"\nSaving density cache to {output_file}...")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(density_cache, f)
    
    print(f"[OK] Saved {len(density_cache):,} POI density records")
    
    # Print file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[OK] File size: {file_size_mb:.2f} MB")


def main():
    print("="*70)
    print("PRECOMPUTING POI DENSITY CACHE")
    print("="*70)
    
    # Load spatial data
    print("\nLoading spatial data...")
    spatial_env = get_spatial_env(cache_file=SPATIAL_CACHE_FILE)
    
    # Compute density
    density_cache = compute_poi_density(spatial_env)
    
    # Save to file
    save_density_cache(density_cache)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print("\nDensity cache ready for use.")
    print("Location selector will now load this file instantly.")
    print("\nTo regenerate (if spatial data changes):")
    print("  python precompute_poi_density.py")


if __name__ == "__main__":
    main()