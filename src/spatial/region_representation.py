from __future__ import annotations

"""
CITY STRUCTURE FOR RL MOBILITY SIMULATION
========================================

City structure adapter.

Wraps RegionData to expose the location and routing interface
expected by the environment and location selector. Supports both real
OSM spatial data (primary) and a synthetic grid city (legacy fallback).
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p



import random
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np


SyntheticPlaceKey = Tuple[str, int]          # e.g. ("house", 3)
PlaceId = Union[int, SyntheticPlaceKey]      # int for OSM POI id, tuple for synthetic


class RegionRepresentation:
    """Grid-based or OSM-based city with places and routes."""

    def __init__(
        self,
        grid_size: int = 100,
        num_agents: int = 30,
        km_per_unit: float = 0.05,
        walking_speed_kmh: float = 5.0,
        cycling_speed_kmh: float = 15.0,
        driving_speed_kmh: float = 40.0,
        seed: int = 42,
        use_spatial: bool = False,
        spatial_env: Any = None,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.grid_size = grid_size
        self.km_per_unit = km_per_unit

        self.walking_speed_kmh = walking_speed_kmh
        self.cycling_speed_kmh = cycling_speed_kmh
        self.driving_speed_kmh = driving_speed_kmh

        self.use_spatial = use_spatial
        self.spatial_env = spatial_env

        # Mode thresholds (in minutes of walking time)
        self.walk_threshold_minutes = 20.0
        self.bike_threshold_minutes = 60.0

        # Precomputed matrices
        self.precomputed_travel_times: Optional[Dict[Tuple[int, int], float]] = None
        self.precomputed_distances: Optional[Dict[Tuple[int, int], float]] = None

        # Precomputed arrays (fast path)
        self.precomputed_travel_times_array: Optional[np.ndarray] = None
        self.precomputed_distances_array: Optional[np.ndarray] = None
        self.precomputed_poi_ids: Optional[List[int]] = None
        self.poi_id_to_idx: Optional[Dict[int, int]] = None
        self._spatial_coord_cols: Optional[Tuple[str, str]] = None
        self._poi_coord_cache: Optional[Dict[int, np.ndarray]] = None

        self._load_precomputed_times()

        if self.use_spatial and self.spatial_env is not None:
            self._init_from_spatial()
        else:
            self._init_synthetic(num_agents)

    # -------------------------------------------------------------------------
    # Legacy wrappers (camelCase / no-underscore)
    # -------------------------------------------------------------------------
    def gettraveltime(self, origin_id: int, dest_id: int, mode: str = "car") -> tuple[float, str]:
        """
        Legacy wrapper expected by older env code.

        `mode` must be forwarded, not dropped: discarding it here silently
        times every trip as a car trip even when the caller passed the agent's
        real transport mode.
        """
        return self.get_travel_time(origin_id, dest_id, mode)

    def getdistancekm(self, origin_id: int, dest_id: int) -> float:
        """Legacy wrapper expected by older env code."""
        return self.get_distance_km(origin_id, dest_id)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def _load_precomputed_times(self) -> None:
        """Load pre-computed travel times if available."""
        from config import USE_PRECOMPUTED_TRAVEL_TIMES, TRAVEL_TIMES_CACHE_FILE

        if not USE_PRECOMPUTED_TRAVEL_TIMES:
            return

        try:
            import pickle

            with open(TRAVEL_TIMES_CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)

            # New optimized array format
            if "travel_times_array" in cache_data:
                self.precomputed_travel_times_array = cache_data["travel_times_array"]
                self.precomputed_distances_array = cache_data["distances_array"]
                self.precomputed_poi_ids = list(cache_data["poi_ids"])
                self.poi_id_to_idx = {poi_id: idx for idx, poi_id in enumerate(self.precomputed_poi_ids)}
                print(f"[OK] Loaded precomputed arrays: {len(self.precomputed_poi_ids):,} POIs")
                return

            # Old dictionary format
            self.precomputed_travel_times = cache_data["travel_time_matrix"]
            self.precomputed_distances = cache_data["distance_matrix"]
            print(f"[OK] Loaded precomputed travel times: {len(self.precomputed_travel_times):,} entries")

        except FileNotFoundError:
            print(f"[WARN] Precomputed travel times not found at {TRAVEL_TIMES_CACHE_FILE}")
            print(" Run: python precompute_travel_times_air_optimized.py")
        except Exception as e:
            print(f"[WARN] Error loading precomputed times: {e}")

    def _init_synthetic(self, num_agents: int) -> None:
        """Initialize synthetic grid-based city (legacy)."""
        self.num_houses = max(1, round(num_agents * 400 / 1000)) * 3
        self.num_offices = max(1, round(num_agents * 15 / 1000)) * 3
        self.num_schools = max(1, round(num_agents * 2 / 1000)) * 3
        self.num_restaurants = max(3, round(num_agents * 5 / 1000)) * 3
        self.num_bars = max(1, round(num_agents * 4 / 1000)) * 3
        self.num_grocery_stores = max(1, round(num_agents * 0.4 / 1000)) * 3

        self.places = self._create_all_places_synthetic()
        self.routes = self._calculate_routes_synthetic()

        # Build activity_to_pois mapping for synthetic mode (for compatibility)
        # Map macro activity -> list of integer place IDs from the synthetic grid
        _SYNTHETIC_MACRO_MAP = {
            "HOME":    "house",
            "WORK":    "office",
            "FOOD":    "restaurant",
            "ERRAND":  "grocery",
            "LEISURE": "bar",
        }
        self.activity_to_pois: Dict[str, List[int]] = {
            macro: [int(p["id"]) for p in self.places.get(place_type, [])]
            for macro, place_type in _SYNTHETIC_MACRO_MAP.items()
        }

        print(f"Synthetic city created: {self.grid_size}x{self.grid_size} grid")
        print(f" Houses: {self.num_houses}")
        print(f" Offices: {self.num_offices}")
        print(f" Restaurants: {self.num_restaurants}")
        print(f" Bars: {self.num_bars}")
        print(f" Grocery stores: {self.num_grocery_stores}")

    def _init_from_spatial(self) -> None:
        """Initialize from OSM spatial data."""
        print("Initializing city from OSM spatial data...")

        # Categories are expected to match your SQL export naming convention.
        #
        # This table is the single source of truth for macro buckets, and every
        # string in it must be a category city_map.sql actually produces. A
        # category that appears in the data but in no bucket here is loaded via
        # POI_INCLUDE yet unreachable as a destination, with nothing to say so;
        # a string that matches nothing in the data is simply dead. Check both
        # directions against the cached POI table when editing.
        #
        # Cafe/bar/pub sit in LEISURE, matching semantic_enrichment.py.
        self.location_to_osm_categories: Dict[str, List[str]] = {
            "HOME": [
                "home_residential",
            ],
            "WORK": [
                "education_college",
                "education_university",
                "education_school",
                "education_library",
                "education_kindergarten",
                "work_office",
                "work_government",
                "work_research",
                "work_industrial",
                "work_retail",
            ],
            "FOOD": [
                "sustenance_restaurant",
                "sustenance_fastfood",
                "sustenance_food_court",
            ],
            "ERRAND": [
                "errand_bank",
                "errand_atm",
                "errand_post",
                "errand_pharmacy",
                "errand_townhall",
                "errand_courthouse",
                "errand_police",
                "healthcare_doctor",
                "healthcare_dentist",
                "healthcare_hospital",
                "healthcare_clinic",
                "shop_supermarket",
                "shop_convenience",
                "shop_bakery",
                "shop_hairdresser",
            ],
            "LEISURE": [
                # Entertainment
                "entertainment_cinema",
                "entertainment_theatre",
                "entertainment_nightclub",
                "entertainment_community",
                "entertainment_arts",
                "entertainment_casino",
                # Culture / tourism
                "tourism_museum",
                "tourism_gallery",
                "tourism_attraction",
                "tourism_viewpoint",
                # Shopping-as-leisure
                "shop_mall",
                "shop_clothing",
                "shop_books",
                "shop_department",
                # Fitness / sports
                "leisure_fitness",
                "leisure_sports",
                "leisure_pool",
                # Outdoors
                "leisure_park",
                "leisure_garden",
                "leisure_playground",
                # Social / community
                "social_worship",
                "social_centre",
                # Cafe/bar/pub + biergarten (SQL category, LEISURE destination)
                "sustenance_cafe",
                "sustenance_bar",
                "sustenance_pub",
                "sustenance_biergarten",
            ],
        }

        # Build activity -> list[poi_id]
        self.activity_to_pois: Dict[str, List[int]] = {}
        for activity, categories in self.location_to_osm_categories.items():
            pois: List[int] = []
            for cat in categories:
                if cat in self.spatial_env.pois_by_category:
                    # Expecting a DataFrame-like object with 'poi_id' column
                    pois.extend(self.spatial_env.pois_by_category[cat]["poi_id"].tolist())
            self.activity_to_pois[activity] = pois

        print("\nActivity POI mapping:")
        for activity, pois in self.activity_to_pois.items():
            print(f"  {activity:15} {len(pois):6,} POIs")

        missing = [k for k, v in self.activity_to_pois.items() if len(v) == 0]
        if missing:
            print(f"[WARNING] No POIs found for activities: {missing}")

        # Build coordinate cache once for fast fallback distance lookup.
        self._build_spatial_coord_cache()

    def _build_spatial_coord_cache(self) -> None:
        """Build POI coordinate cache for fast fallback distance computation."""
        self._spatial_coord_cols = None
        self._poi_coord_cache = {}

        if not (self.use_spatial and self.spatial_env is not None):
            return

        pois = self.spatial_env.pois

        for x_col, y_col in (("x", "y"), ("lon", "lat"), ("longitude", "latitude")):
            if x_col in pois.columns and y_col in pois.columns:
                self._spatial_coord_cols = (x_col, y_col)
                break

        if self._spatial_coord_cols is None or "poi_id" not in pois.columns:
            return

        x_col, y_col = self._spatial_coord_cols
        try:
            coords = pois[["poi_id", x_col, y_col]].dropna(subset=["poi_id", x_col, y_col])
            ids = coords["poi_id"].to_numpy(dtype=np.int64)
            xs  = coords[x_col].to_numpy(dtype=np.float64)
            ys  = coords[y_col].to_numpy(dtype=np.float64)
            self._poi_coord_cache = {
                int(pid): np.array([x, y], dtype=np.float64)
                for pid, x, y in zip(ids, xs, ys)
            }
        except Exception:
            self._poi_coord_cache = {}

    # -------------------------------------------------------------------------
    # Synthetic helpers
    # -------------------------------------------------------------------------
    def _create_all_places_synthetic(self) -> Dict[str, List[Dict[str, Any]]]:
        places: Dict[str, List[Dict[str, Any]]] = {
            "house": [],
            "office": [],
            "school": [],
            "restaurant": [],
            "bar": [],
            "grocery": [],
        }

        place_counts = {
            "house": self.num_houses,
            "office": self.num_offices,
            "school": self.num_schools,
            "restaurant": self.num_restaurants,
            "bar": self.num_bars,
            "grocery": self.num_grocery_stores,
        }

        # OPTIMIZATION: Use NumPy vectorized operations instead of while loop collision checking
        # Generate random positions more efficiently
        total_places = sum(place_counts.values())
        max_attempts = total_places * 10
        all_positions = []
        used_positions = set()
        
        # Generate all unique random positions upfront
        while len(all_positions) < min(total_places, self.grid_size * self.grid_size):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in used_positions:
                used_positions.add((x, y))
                all_positions.append((x, y))
        
        pos_idx = 0
        for place_type, count in place_counts.items():
            for i in range(count):
                if pos_idx < len(all_positions):
                    x, y = all_positions[pos_idx]
                    places[place_type].append({"id": i, "pos": np.array([x + 0.5, y + 0.5], dtype=np.float32)})
                    pos_idx += 1

        return places

    def _calculate_routes_synthetic(self) -> Dict[Tuple[SyntheticPlaceKey, SyntheticPlaceKey], Dict[str, Any]]:
        routes: Dict[Tuple[SyntheticPlaceKey, SyntheticPlaceKey], Dict[str, Any]] = {}

        all_places: List[Tuple[str, int, np.ndarray]] = []
        for place_type, place_list in self.places.items():
            for place in place_list:
                all_places.append((place_type, place["id"], place["pos"]))

        for type1, id1, pos1 in all_places:
            for type2, id2, pos2 in all_places:
                if (type1, id1) == (type2, id2):
                    continue
                distance = float(np.linalg.norm(pos1 - pos2))
                distance_km = distance * self.km_per_unit
                walk_time_minutes = (distance_km / self.walking_speed_kmh) * 60.0

                key = ((type1, id1), (type2, id2))
                routes[key] = {
                    "origin": pos1,
                    "destination": pos2,
                    "distance_km": distance_km,
                    "walk_time_minutes": walk_time_minutes,
                }

        return routes

    def get_route(self, origin_key: SyntheticPlaceKey, dest_key: SyntheticPlaceKey) -> Optional[Dict[str, Any]]:
        """Synthetic-only."""
        if self.use_spatial:
            return None
        return self.routes.get((origin_key, dest_key))

    # -------------------------------------------------------------------------
    # Travel time + distance (OSM / precomputed)
    # -------------------------------------------------------------------------
    def get_travel_time(self, origin_id: int, dest_id: int,
                        mode: str = "car") -> tuple[float, str]:
        """
        Return (travel_time_minutes, transport_mode) for the given mode.

        The returned time must reflect `mode`: timing everything at
        driving_speed_kmh turns a 2 km walk (~24 min at 5 km/h) into ~3 min.
        The precomputed branches below are car-only, having been built before
        per-mode assignment existed; the mode-aware coordinate fallback is what
        actually runs, since USE_PRECOMPUTED_TRAVEL_TIMES is off.
        """
        if origin_id == dest_id:
            return 0.0, mode

        # Precomputed arrays (car-only, see docstring)
        if self.precomputed_travel_times_array is not None and self.poi_id_to_idx is not None:
            try:
                oi = self.poi_id_to_idx[origin_id]
                di = self.poi_id_to_idx[dest_id]
                return float(self.precomputed_travel_times_array[oi, di]), "car"
            except KeyError:
                pass

        # Precomputed dict (car-only, see docstring)
        if self.precomputed_travel_times is not None:
            key = (origin_id, dest_id)
            if key in self.precomputed_travel_times:
                return float(self.precomputed_travel_times[key]), "car"

        # Fallback: compute from coordinates at the speed of the actual mode
        distance_km = self._calculate_distance_km(origin_id, dest_id)
        if distance_km <= 0.0:
            return 0.0, mode

        speed_kmh = {
            "foot":    self.walking_speed_kmh,
            "bicycle": self.cycling_speed_kmh,
            "car":     self.driving_speed_kmh,
        }.get(mode, self.driving_speed_kmh)

        minutes = (distance_km / speed_kmh) * 60.0
        return float(minutes), mode

    def _calculate_distance_km(self, origin_id: int, dest_id: int) -> float:
        """Fallback distance calculation if no precomputed matrix is available."""
        if not (self.use_spatial and self.spatial_env is not None):
            return 0.0

        if self._poi_coord_cache is None or self._spatial_coord_cols is None:
            self._build_spatial_coord_cache()

        if not self._poi_coord_cache or self._spatial_coord_cols is None:
            return 0.0

        oxy = self._poi_coord_cache.get(int(origin_id))
        dxy = self._poi_coord_cache.get(int(dest_id))
        if oxy is None or dxy is None:
            return 0.0

        x_col, _ = self._spatial_coord_cols

        # If lon/lat in degrees (rough approximation)
        #
        # "x" belongs in this tuple. map/export_city.py emits
        # `ST_X(geom) AS x` / `ST_Y(geom) AS y`, both plain WGS84 degrees, and
        # _build_spatial_coord_cache tries ("x", "y") FIRST, so real data always
        # resolves to x_col == "x". Omitting it sends real data to the "assume
        # metres" branch below, which treats a ~0.03-degree difference (a few
        # km) as 0.03 metres -- every distance ~100,000x too small, making
        # travel effectively free and instantaneous.
        if x_col in ("x", "lon", "longitude"):
            lon1, lat1 = oxy[0], oxy[1]
            lon2, lat2 = dxy[0], dxy[1]

            # If projected meters mistakenly stored in lon/lat columns
            if abs(lon1) > 180 or abs(lat1) > 90 or abs(lon2) > 180 or abs(lat2) > 90:
                meters = float(np.linalg.norm(oxy - dxy))
                return meters / 1000.0

            avg_lat = (lat1 + lat2) / 2.0
            lon_diff_km = abs(lon1 - lon2) * 111.0 * float(np.cos(np.radians(avg_lat)))
            lat_diff_km = abs(lat1 - lat2) * 111.0
            return float(np.sqrt(lon_diff_km**2 + lat_diff_km**2))

        # Otherwise assume meters
        meters = float(np.linalg.norm(oxy - dxy))
        return meters / 1000.0

    def get_distance_km(self, origin_id: int, dest_id: int) -> float:
        """Distance in km (prefers precomputed matrices)."""
        if origin_id == dest_id:
            return 0.0

        # Precomputed arrays
        if self.precomputed_distances_array is not None and self.poi_id_to_idx is not None:
            try:
                oi = self.poi_id_to_idx[origin_id]
                di = self.poi_id_to_idx[dest_id]
                return float(self.precomputed_distances_array[oi, di])
            except KeyError:
                pass

        # Precomputed dict
        if self.precomputed_distances is not None:
            key = (origin_id, dest_id)
            if key in self.precomputed_distances:
                return float(self.precomputed_distances[key])

        return self._calculate_distance_km(origin_id, dest_id)

    # -------------------------------------------------------------------------
    # POI sampling
    # -------------------------------------------------------------------------
    def get_random_place(self, location_type: str) -> PlaceId:
        """
        Return a random location for a given macro activity.

        For spatial mode: returns POI id (int).
        For synthetic mode: returns (place_type, place_id).
        """
        if self.use_spatial:
            pois = self.activity_to_pois.get(location_type, [])
            if pois:
                return int(random.choice(pois))
            # Spatial data not loaded / empty -- fall through to synthetic fallback
            if not self.places:
                raise ValueError(f"No POIs available for {location_type}")

        # synthetic -- macro activity names -> synthetic place type keys
        _MACRO_TO_SYNTHETIC: Dict[str, str] = {
            "HOME":    "house",
            "WORK":    "office",
            "FOOD":    "restaurant",
            "ERRAND":  "grocery",
            "LEISURE": "bar",
        }
        place_type = _MACRO_TO_SYNTHETIC.get(location_type.upper(), location_type.lower())
        if place_type not in self.places or not self.places[place_type]:
            raise ValueError(f"No synthetic places available for {location_type}")
        place = random.choice(self.places[place_type])
        return (place_type, int(place["id"]))

    def get_position(self, place_key: PlaceId) -> np.ndarray:
        """Synthetic-only helper; returns (0,0) for spatial mode."""
        if place_key is None or place_key == "traveling":
            return np.array([0.0, 0.0], dtype=np.float32)

        if self.use_spatial:
            return np.array([0.0, 0.0], dtype=np.float32)

        place_type, place_id = place_key
        for place in self.places.get(place_type, []):
            if int(place["id"]) == int(place_id):
                return place["pos"]
        return np.array([0.0, 0.0], dtype=np.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: SYNTHETIC CITY")
    print("=" * 60)
    city = RegionRepresentation(grid_size=100, num_agents=30, use_spatial=False)

    print("\n" + "=" * 60)
    print("TEST 2: OSM SPATIAL DATA")
    print("=" * 60)
    print("(Requires spatial cache + config to be set up)")