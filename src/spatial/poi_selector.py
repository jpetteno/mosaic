from __future__ import annotations

"""
LOCATION SELECTOR V5
====================
7-Dimensional behavioral location selection.

Dimensions
----------
1. FAVORITES      -- pre-assigned anchor POIs used as primary candidate filter
2. FAMILIARITY    -- explore-vs-exploit fallback when not using favorites
3. DISTANCE       -- context-aware exponential decay from home/work/current
4. TIME           -- sub-activity sampled proportional to time_alpha
5. VARIETY        -- two-level recency penalty (location + category cooldown)
6. POPULARITY     -- POI density scaled by sub-activity importance
7. CONTEXT        -- behavioral mode (routine/mixed/exploratory) modulates all

Integration notes
-----------------
- Replaces the old POISelector class entirely.
- All sub-activity keys are OSM-style (sustenance_restaurant, shop_supermarket).
- RegionDataAdapter wraps RegionRepresentation to expose the interface V5 expects.
- AgentState must have a `favorites` dict (populated by impute_favorites
  at episode reset) and helpers get_visited_pois() / get_familiar_locations().
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p



import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import config
from path_generator import PathGenerator
from semantic_enrichment import (
    ACTIVITY_PROBABILITIES,
    get_time_alpha,
    get_location_cooldown,
    get_category_cooldown,
    get_popularity_importance,
    get_sub_activities_for_macro,
)
from impute_favorites import get_favorites_for_activity


# =============================================================================
# HELPERS
# =============================================================================

def _haversine_vec(
    lon1: float, lat1: float,
    lons2: np.ndarray, lats2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine from one origin to N destinations (float32, km).

    Uses the same formula as precompute_air_distances so that home->dest
    distances are on the same numerical scale as the destxdest matrix.
    """
    R = 6371.0
    dlon = lons2 - lon1
    dlat = lats2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lats2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (R * c).astype(np.float32)


# =============================================================================
# CITY DATA ADAPTER
# Wraps RegionRepresentation to expose the interface V5 expects.
# =============================================================================

class RegionDataAdapter:
    """
    Thin adapter over RegionRepresentation that exposes the V5 location selector API.

    Methods required by V5:
        get_pois_by_macro(macro)        -> List[int]
        get_pois_by_subactivity(sub)    -> List[int]
        get_distances_to_poi(poi, home, work, current) -> Dict[str, float]
        get_distance(a, b)              -> float  (km)
        get_poi_density(poi, macro)     -> float
        poi_subtype_map                 -> Dict[int, str]
    """

    def __init__(self, city, path_generator: Optional["PathGenerator"] = None):
        self.city = city
        self._path_lib = path_generator          # mode-aware router (optional)
        self._poi_density_cache: Dict[int, Dict[str, float]] = {}
        self._poi_subtype_map: Dict[int, str] = {}
        self._subtype_to_pois: Dict[str, List[int]] = {}
        # Air distance matrix for fast location scoring
        self._air_dist_matrix: Optional[np.ndarray] = None
        self._air_poi_ids: Optional[np.ndarray] = None
        self._air_poi_id_to_idx: Dict[int, int] = {}
        self._air_dest_lons: Optional[np.ndarray] = None
        self._air_dest_lats: Optional[np.ndarray] = None
        self._air_home_ids: Optional[np.ndarray] = None
        self._air_home_lons: Optional[np.ndarray] = None
        self._air_home_lats: Optional[np.ndarray] = None
        self._air_home_id_to_idx: Dict[int, int] = {}
        # Pre-aligned density array: shape (N_dest_pois, 3) for FOOD/ERRAND/LEISURE.
        # Built once from the poi_density_cache dict so batch scoring uses a
        # single numpy index instead of an O(N) Python fromiter loop per step.
        self._density_arr: Optional[np.ndarray] = None
        self._load_poi_density_cache()
        self._load_air_distance_cache()
        self._build_subtype_map()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_poi_density_cache(self) -> None:
        try:
            from config import POI_DENSITY_FILE as _PDF
            with open(_PDF, "rb") as f:
                self._poi_density_cache = pickle.load(f)
        except Exception:
            self._poi_density_cache = {}

    def _load_air_distance_cache(self) -> None:
        try:
            import pickle
            from config import AIR_DISTANCE_FILE as _ADF
            with open(_ADF, "rb") as f:
                cache = pickle.load(f)
            self._air_dist_matrix    = cache["distances_km"]
            self._air_poi_ids        = cache["poi_ids"]
            self._air_poi_id_to_idx  = cache["poi_id_to_idx"]
            self._air_dest_lons      = cache["dest_lons"]
            self._air_dest_lats      = cache["dest_lats"]
            self._air_home_ids       = cache.get("home_ids")
            self._air_home_lons      = cache.get("home_lons")
            self._air_home_lats      = cache.get("home_lats")
            if self._air_home_ids is not None:
                self._air_home_id_to_idx = {
                    int(hid): int(i) for i, hid in enumerate(self._air_home_ids)
                }
            # Build density array aligned with the dest-poi index.
            if self._air_poi_ids is not None:
                _DENSITY_MACROS = ["FOOD", "ERRAND", "LEISURE"]
                _n = len(self._air_poi_ids)
                _arr = np.zeros((_n, len(_DENSITY_MACROS)), dtype=np.float32)
                for _pid, _col_idx in self._air_poi_id_to_idx.items():
                    _d = self._poi_density_cache.get(_pid, {})
                    for _mi, _mac in enumerate(_DENSITY_MACROS):
                        _arr[_col_idx, _mi] = float(_d.get(_mac, 0.0))
                self._density_arr = _arr
                self._density_macro_idx = {m: i for i, m in enumerate(_DENSITY_MACROS)}
            print(f"[POISelector] Air distance matrix loaded: "
                  f"{self._air_dist_matrix.shape[0]:,} POIs")
        except Exception as e:
            print(f"[POISelector] Air distance cache not loaded: {e}")

    def _build_subtype_map(self) -> None:
        """Build poi_id -> sub_activity_type map from spatial env if available."""
        if not hasattr(self.city, "spatial_env") or self.city.spatial_env is None:
            return
        pois = self.city.spatial_env.pois
        if pois is None:
            return
        try:
            # Vectorised: ~40x faster than iterrows() over 40k rows.
            # Try columns in priority order; export_bonn exports "poi_category".
            id_col  = next((c for c in ("poi_id", "poiid", "id")
                            if c in pois.columns), None)
            sub_col = next((c for c in ("sub_type", "poi_category", "category")
                            if c in pois.columns), None)
            if id_col is None or sub_col is None:
                return
            df   = pois[[id_col, sub_col]].dropna(subset=[id_col, sub_col])
            ids  = df[id_col].to_numpy(dtype=np.int64)
            subs = df[sub_col].astype(str).to_numpy()
            # Mixed-use buildings (a shop or office inside a residential
            # building) appear TWICE with the same poi_id -- once as
            # "home_residential", once under their real venue category. In raw
            # table order whichever row comes last wins, which can overwrite the
            # venue identity with "home_residential" and lose that POI as a
            # candidate. Processing home_residential rows first (a stable sort
            # keeps everything else in its original order) guarantees the venue
            # category always wins for a duplicate id.
            pairs = sorted(zip(ids, subs), key=lambda ps: ps[1] == "home_residential", reverse=True)
            for pid, sub in pairs:
                if sub and sub != "nan":
                    ipid = int(pid)
                    self._poi_subtype_map[ipid] = sub
                    self._subtype_to_pois.setdefault(sub, []).append(ipid)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # V5 interface
    # ------------------------------------------------------------------

    @property
    def poi_subtype_map(self) -> Dict[int, str]:
        return self._poi_subtype_map

    def get_pois_by_macro(self, macro_activity: str) -> List[int]:
        pois = self.city.activity_to_pois.get(macro_activity, [])
        return [int(p) for p in pois]

    def get_pois_by_subactivity(self, sub_activity: str) -> List[int]:
        """Return POI IDs whose sub_type matches sub_activity (OSM key)."""
        if self._subtype_to_pois:
            return list(self._subtype_to_pois.get(sub_activity, []))
        # Fallback: infer macro from sub_activity, return all macro POIs
        macro = _sub_to_macro(sub_activity)
        if macro:
            return self.get_pois_by_macro(macro)
        return []

    def get_travel_times_to_poi(
        self,
        poi_id: int,
        home_id: int,
        work_id: int,
        current_id: int,
        mode: str = "car",
    ) -> Dict[str, float]:
        """Return travel times in minutes from home/work/current to poi."""
        return {
            "home":    self._get_time(home_id,    poi_id, mode),
            "work":    self._get_time(work_id,    poi_id, mode),
            "current": self._get_time(current_id, poi_id, mode),
        }

    # Mode speeds for air-distance-to-time conversion (km/h)
    _MODE_SPEED_KMH = {"car": 30.0, "bicycle": 15.0, "foot": 5.0}

    def _get_time(self, origin: int, dest: int, mode: str) -> float:
        """
        Air-distance-based travel time estimate in minutes for location scoring.
        Fast: O(1) matrix lookup or vectorized haversine.
        Actual road-network time is computed separately in step() after selection.
        """
        km = self._get_air_distance_km(origin, dest)
        if km >= 999.0:
            return 999.0
        speed = self._MODE_SPEED_KMH.get(mode, 30.0)
        walk_t = (km / speed) * 60.0
        if mode == "foot" and walk_t > 25.0 and self._path_lib is not None:
            try:
                pt_t = self._path_lib.get_pt_travel_time(origin, dest)
                if pt_t is not None:
                    return min(walk_t, float(pt_t))
            except Exception:
                pass
        return walk_t

    def _get_air_distance_km(self, poi_a: int, poi_b: int) -> float:
        """Air distance in km between two POI IDs. Uses matrix if available."""
        # Both in matrix: O(1) lookup
        idx_a = self._air_poi_id_to_idx.get(int(poi_a))
        idx_b = self._air_poi_id_to_idx.get(int(poi_b))
        if idx_a is not None and idx_b is not None and self._air_dist_matrix is not None:
            return float(self._air_dist_matrix[idx_a, idx_b])

        # One is a home node: compute haversine on the fly
        if self._path_lib is not None:
            try:
                return float(self._path_lib.get_air_distance_km(poi_a, poi_b))
            except Exception:
                pass

        # Final fallback
        try:
            return float(self.city.getdistancekm(int(poi_a), int(poi_b)))
        except Exception:
            return 999.0

    def get_air_distance_km(self, a: int, b: int) -> float:
        """Air distance in km -- matrix lookup first, then haversine fallback."""
        return self._get_air_distance_km(a, b)

    # Legacy wrappers
    def get_distances_to_poi(self, poi_id, home_id, work_id, current_id):
        return self.get_travel_times_to_poi(poi_id, home_id, work_id, current_id, "car")

    def get_distance(self, a: int, b: int) -> float:
        return self._get_air_distance_km(a, b)

    def get_poi_density(self, poi_id: int, macro_activity: str) -> float:
        """Return nearby POI density for a POI (from precomputed cache)."""
        densities = self._poi_density_cache.get(int(poi_id), {})
        return float(densities.get(macro_activity, 0.0))


def _sub_to_macro(sub: str) -> str:
    """Infer macro from OSM sub-activity prefix."""
    for macro, subs in ACTIVITY_PROBABILITIES.items():
        if sub in subs:
            return macro
    return ""


# =============================================================================
# BEHAVIORAL CONTEXT (Dimension 7)
# =============================================================================

@dataclass
class BehavioralContext:
    current_time: datetime
    hour: float
    day_of_week: int
    is_weekday: bool
    is_weekend: bool
    phase: str
    mode: str
    temperature: float
    exploration_rate: float
    current_location_type: str
    time_since_home: float
    time_since_work: Optional[float]


def _determine_phase(hour: float, is_weekday: bool) -> str:
    if is_weekday:
        if 6 <= hour < 9:   return "morning_commute"
        if 9 <= hour < 12:  return "morning_work"
        if 12 <= hour < 14: return "lunch_hours"
        if 14 <= hour < 17: return "afternoon_work"
        if 17 <= hour < 19: return "evening_commute"
        if 19 <= hour < 22: return "evening_home"
        return "night"
    else:
        if 8 <= hour < 12:  return "weekend_morning"
        if 12 <= hour < 18: return "weekend_afternoon"
        if 18 <= hour < 24: return "weekend_evening"
        return "night"


def _determine_mode(phase: str, is_weekend: bool, loc_type: str, hour: float) -> str:
    if phase in ("morning_commute", "morning_work", "lunch_hours", "evening_commute"):
        return "routine"
    if phase in ("weekend_afternoon", "weekend_evening"):
        return "exploratory"
    if is_weekend and loc_type == "HOME" and 10 <= hour < 22:
        return "exploratory"
    return "mixed"


def create_context(
    current_time: datetime,
    current_location_type: str,
    last_home_departure: Optional[datetime] = None,
    last_work_departure: Optional[datetime] = None,
) -> BehavioralContext:
    hour       = current_time.hour + current_time.minute / 60.0
    dow        = current_time.weekday()
    is_weekday = dow < 5
    is_weekend = dow >= 5

    time_since_home = 0.0
    if current_location_type != "HOME" and last_home_departure is not None:
        time_since_home = (current_time - last_home_departure).total_seconds() / 3600.0

    time_since_work: Optional[float] = None
    if last_work_departure is not None:
        time_since_work = (current_time - last_work_departure).total_seconds() / 3600.0

    phase = _determine_phase(hour, is_weekday)
    mode  = _determine_mode(phase, is_weekend, current_location_type, hour)

    if mode == "routine":
        temperature     = 0.4
        exploration_rate = 0.30
    elif mode == "exploratory":
        temperature     = 0.8
        exploration_rate = 0.45
    else:
        temperature     = 0.5
        exploration_rate = 0.35

    return BehavioralContext(
        current_time=current_time,
        hour=hour,
        day_of_week=dow,
        is_weekday=is_weekday,
        is_weekend=is_weekend,
        phase=phase,
        mode=mode,
        temperature=temperature,
        exploration_rate=exploration_rate,
        current_location_type=current_location_type,
        time_since_home=time_since_home,
        time_since_work=time_since_work,
    )


# =============================================================================
# DIMENSION 3: TRAVEL TIME UTILITY  (replaces distance utility)
# =============================================================================
# All values in MINUTES.
# Hard ceilings are mode-aware -- a 45-min car trip is acceptable,
# a 45-min walk for dinner is not.

_TIME_BASE_WEIGHTS = {
    "FOOD":    {"home": 0.65, "work": 0.25, "current": 0.10},
    "ERRAND":  {"home": 0.75, "work": 0.15, "current": 0.10},
    "LEISURE": {"home": 0.45, "work": 0.15, "current": 0.40},
}

# lambda_base: decay constant in minutes (higher = more tolerant of long trips)
# t_max: hard ceiling in minutes per mode -- trips beyond this return 0.0
# perceived_multiplier: psychological cost factor on top of raw time
_TIME_PARAMS = {
    "FOOD": {
        "lambda_base": 8.0,
        "t_max":       {"car": 30.0, "bicycle": 25.0, "foot": 20.0},
        "perceived":   {"car": 1.0,  "bicycle": 1.2,  "foot": 1.6},
    },
    "ERRAND": {
        "lambda_base": 6.0,
        "t_max":       {"car": 25.0, "bicycle": 20.0, "foot": 15.0},
        "perceived":   {"car": 1.0,  "bicycle": 1.2,  "foot": 1.7},
    },
    "LEISURE": {
        "lambda_base": 15.0,
        "t_max":       {"car": 45.0, "bicycle": 35.0, "foot": 25.0},
       "perceived":   {"car": 1.0,  "bicycle": 1.2,  "foot": 1.8},
    },
}

_MODE_LAMBDA_SCALE = {"car": 1.0, "bicycle": 0.85, "foot": 0.65}


def _distance_utility(
    macro: str,
    distances: Dict[str, float],   # now travel times in minutes
    ctx: BehavioralContext,
    transport_mode: str = "car",
) -> float:
    """
    Utility based on travel time in minutes.
    'distances' dict contains minutes (not km) -- named for backward compat.
    """
    bw = dict(_TIME_BASE_WEIGHTS.get(macro, {"home": 0.5, "work": 0.3, "current": 0.2}))
    w_home, w_work, w_cur = bw["home"], bw["work"], bw["current"]

    # Context modulations (same logic as before, just on time not distance)
    if ctx.current_location_type == "WORK":
        w_work *= 3.0
    if ctx.is_weekday and 8 <= ctx.hour < 18:
        w_work *= 2.5
    if ctx.time_since_work is not None and ctx.time_since_work < 2.0:
        w_work *= 1.0 + (1.0 - ctx.time_since_work / 2.0)
    if ctx.phase == "evening_home":
        w_home *= 1.5
    if ctx.current_location_type == "OTHER":
        w_cur *= 2.0
    if ctx.is_weekend:
        w_work *= 0.5
        w_cur  *= 1.3

    total = w_home + w_work + w_cur
    w_home /= total; w_work /= total; w_cur /= total

    t_eff = (w_home * distances.get("home",    0.0) +
             w_work * distances.get("work",    0.0) +
             w_cur  * distances.get("current", 0.0))

    p = _TIME_PARAMS.get(macro, {
        "lambda_base": 10.0,
        "t_max":   {"car": 40.0, "bicycle": 30.0, "foot": 20.0},
        "perceived": {"car": 1.0, "bicycle": 1.1, "foot": 1.2},
    })

    # Apply perceived cost multiplier
    perceived = p["perceived"].get(transport_mode, 1.0)
    t_perceived = t_eff * perceived

    # Hard ceiling -- mode-aware
    t_max = p["t_max"].get(transport_mode, 40.0)
    if ctx.is_weekend:   t_max *= 1.4   # more willing to travel on weekends
    if ctx.mode == "exploratory": t_max *= 1.2

    if t_perceived > t_max:
        # Steep penalty beyond ceiling -- preserves ordering, avoids random fallback
        lam = p["lambda_base"]
        overshoot = t_perceived - t_max
        return float(np.exp(-t_max / lam) * np.exp(-overshoot / (lam * 0.15)))

    # Exponential decay on perceived time
    lam = p["lambda_base"]
    if ctx.is_weekend:            lam *= 1.5
    if ctx.mode == "exploratory": lam *= 1.3
    elif ctx.mode == "routine":   lam *= 0.9
    if ctx.hour >= 21:            lam *= 0.8

    lam *= _MODE_LAMBDA_SCALE.get(transport_mode, 1.0)

    return float(np.exp(-t_perceived / lam))


# =============================================================================
# DIMENSION 5: VARIETY UTILITY
# =============================================================================

_VARIETY_LAMBDA = 5.0


def _variety_utility(
    poi_id: int,
    sub_activity: str,
    recent_poi_timestamps: Dict[int, datetime],
    poi_subtype_map: Dict[int, str],
    current_time: datetime,
    last_category_visit: Optional[datetime] = None,
) -> float:
    if poi_id in recent_poi_timestamps:
        h  = (current_time - recent_poi_timestamps[poi_id]).total_seconds() / 3600.0
        cd = get_location_cooldown(sub_activity)
        U_loc = 1.0 - float(np.exp(-_VARIETY_LAMBDA * h / cd))
    else:
        U_loc = 1.0

    if last_category_visit is None:
        cat_visits = [
            t for pid, t in recent_poi_timestamps.items()
            if poi_subtype_map.get(pid) == sub_activity
        ]
        last_category_visit = max(cat_visits) if cat_visits else None

    if last_category_visit is not None:
        h  = (current_time - last_category_visit).total_seconds() / 3600.0
        cd = get_category_cooldown(sub_activity)
        U_cat = 1.0 - float(np.exp(-_VARIETY_LAMBDA * h / cd))
    else:
        U_cat = 1.0

    return float(U_loc * U_cat)


# =============================================================================
# DIMENSION 6: POPULARITY UTILITY
# =============================================================================

_POP_BETA = 0.5
_POP_NORM = 50.0


def _popularity_utility(
    macro: str,
    sub_activity: str,
    poi_density: float,
    ctx: BehavioralContext,
) -> float:
    U_raw      = min(1.0, (poi_density / _POP_NORM) ** _POP_BETA)
    importance = get_popularity_importance(sub_activity, macro)
    if ctx.mode == "exploratory":
        importance = min(1.0, importance * 1.2)
    elif ctx.mode == "routine":
        importance *= 0.8
    return float(importance * U_raw + (1.0 - importance))


# =============================================================================
# SOFTMAX
# =============================================================================

def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    t = max(temperature, 0.01)
    e = np.exp(scores / t)
    return e / e.sum()


# =============================================================================
# MAIN LOCATION SELECTOR CLASS
# =============================================================================

class POISelector:
    """
    V5 Location Selector -- 7-dimension algorithm with favorites filter.
    Drop-in replacement for old POISelector with same select_location() API.
    """

    def __init__(self, city, random_seed: Optional[int] = None, path_generator=None):
        self.city_adapter = RegionDataAdapter(city, path_generator=path_generator)
        self.city         = city
        self.random_state = np.random.RandomState(random_seed)
        # No floor is applied to the configured value: a max(..., 15) here would
        # override TOP_K_VENUES from custom_settings.py and make it inert.
        self.top_k        = int(getattr(config, "TOP_K_VENUES", 15))

    def select_location(
        self,
        activity_type: str,
        current_location_id: int,
        home_location_id: int,
        work_location_id: int,
        visited_locations: Set[int],
        sub_activity_type: Optional[str] = None,
        current_hour: float = 12.0,
        *,
        recent_poi_timestamps: Optional[Dict[int, datetime]] = None,
        agent_state=None,
        current_time: Optional[datetime] = None,
    ) -> int:
        if activity_type == "HOME":
            return int(home_location_id)
        if activity_type == "WORK":
            return int(work_location_id)

        if recent_poi_timestamps is None:
            recent_poi_timestamps = {}

        if current_time is None:
            h = int(current_hour)
            m = int((current_hour % 1) * 60)
            current_time = datetime(2024, 1, 1, h, m)

        # Read transport preference from agent state
        transport_mode = "car"
        if agent_state is not None:
            transport_mode = str(getattr(agent_state, "transport_preference", "car"))

        loc_type = getattr(agent_state, "current_location", "OTHER") if agent_state else "OTHER"
        ctx = create_context(current_time=current_time, current_location_type=loc_type)

        if sub_activity_type is None:
            sub_activity_type = self._sample_sub_activity(activity_type, ctx)

        candidates, _ = self._get_candidates(
            macro=activity_type,
            sub_activity=sub_activity_type,
            visited_locations=visited_locations,
            recent_poi_timestamps=recent_poi_timestamps,
            agent_state=agent_state,
            ctx=ctx,
        )

        if not candidates:
            all_pois = self.city_adapter.get_pois_by_macro(activity_type)
            if not all_pois:
                fallback = self.city.get_random_place(activity_type)
                return int(fallback[1]) if isinstance(fallback, tuple) else int(fallback)
            dists = [(p, self.city_adapter.get_air_distance_km(current_location_id, p))
                     for p in all_pois]
            dists.sort(key=lambda x: x[1])
            return int(dists[0][0])

        utilities = self._score_candidates(
            candidates=candidates,
            macro=activity_type,
            sub_activity=sub_activity_type,
            current_id=current_location_id,
            home_id=home_location_id,
            work_id=work_location_id,
            recent_poi_timestamps=recent_poi_timestamps,
            ctx=ctx,
            transport_mode=transport_mode,
        )

        return self._select_from_utilities(
            utilities, ctx,
            path_generator=self.city_adapter._path_lib,
            current_id=current_location_id,
            transport_mode=transport_mode,
        )

    # ------------------------------------------------------------------
    # Sub-activity sampling (Dimension 4)
    # ------------------------------------------------------------------

    def _sample_sub_activity(self, macro: str, ctx: BehavioralContext) -> str:
        subs = get_sub_activities_for_macro(macro)
        if not subs:
            return macro
        alphas = np.array([
            float(get_time_alpha(s, ctx.hour, ctx.day_of_week)) for s in subs
        ])
        if alphas.sum() == 0:
            alphas = np.ones(len(subs))
        probs = alphas / alphas.sum()
        return subs[int(self.random_state.choice(len(subs), p=probs))]

    # ------------------------------------------------------------------
    # Candidate selection (Dimensions 1 + 2)
    # ------------------------------------------------------------------

    def _get_candidates(
        self,
        macro: str,
        sub_activity: str,
        visited_locations: Set[int],
        recent_poi_timestamps: Dict[int, datetime],
        agent_state,
        ctx: BehavioralContext,
    ) -> Tuple[List[int], str]:
        # Fetch favorites once; reused for both the escape check and fav_set below.
        agent_favs = getattr(agent_state, "favorites", {}) if agent_state else {}
        favorites  = get_favorites_for_activity(macro, agent_favs)

        if favorites:
            if self.random_state.random() < 0.30:
                favorites = []  # Probabilistic exploration
            elif recent_poi_timestamps:
                # Pre-compute last category visit once for the variety escape check.
                last_cat: Optional[datetime] = None
                cat_times = [t for pid, t in recent_poi_timestamps.items()
                             if self.city_adapter.poi_subtype_map.get(pid) == sub_activity]
                if cat_times:
                    last_cat = max(cat_times)
                best_util = max(
                    _variety_utility(
                        f, sub_activity, recent_poi_timestamps,
                        self.city_adapter.poi_subtype_map, ctx.current_time,
                        last_category_visit=last_cat,
                    )
                    for f in favorites
                )
                if best_util < 0.15:
                    favorites = []  # Variety escape

        if favorites:
            return favorites, "favorites"

        # Exploration split: 25% new / 10% visited-non-fav / 65% familiar
        roll    = self.random_state.random()
        all_mac = self.city_adapter.get_pois_by_macro(macro)
        fav_set = set(get_favorites_for_activity(macro, agent_favs))  # reuses agent_favs

        if roll < 0.25:
            cands = [p for p in all_mac
                     if p not in visited_locations and p not in fav_set]
            return cands or all_mac, "explore_new"

        if roll < 0.35:
            cands = [p for p in visited_locations if p not in fav_set]
            return cands or [p for p in all_mac if p not in fav_set] or all_mac, \
                   "explore_visited_nonfav"

        cands = [p for p in visited_locations if p not in fav_set]
        if not cands:
            cands = [p for p in all_mac if p not in fav_set]
        return cands or all_mac, "explore_familiar"

    # ------------------------------------------------------------------
    # Scoring (Dimensions 3 + 5 + 6)
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: List[int],
        macro: str,
        sub_activity: str,
        current_id: int,
        home_id: int,
        work_id: int,
        recent_poi_timestamps: Dict[int, datetime],
        ctx: BehavioralContext,
        transport_mode: str = "car",
    ) -> Dict[int, float]:
        utilities: Dict[int, float] = {}

        # Pre-compute last category visit once -- reused for every candidate.
        last_category_visit: Optional[datetime] = None
        if recent_poi_timestamps:
            cat_times = [
                t for pid, t in recent_poi_timestamps.items()
                if self.city_adapter.poi_subtype_map.get(pid) == sub_activity
            ]
            if cat_times:
                last_category_visit = max(cat_times)

        # Batch-compute travel times via the air-distance matrix for all
        # candidates that are in the matrix (>99 % during training).
        # This replaces Nx3 individual dict-lookups with 3 numpy row-slices.
        adapter  = self.city_adapter
        mat      = adapter._air_dist_matrix
        id2idx   = adapter._air_poi_id_to_idx
        speed    = adapter._MODE_SPEED_KMH.get(transport_mode, 30.0)
        t_factor = 60.0 / speed  # km -> minutes

        home_idx = id2idx.get(int(home_id))
        work_idx = id2idx.get(int(work_id))
        cur_idx  = id2idx.get(int(current_id))

        # Home nodes (home_residential) are never in the dest matrix.
        # Look them up in the home-coord index built from the stored
        # home_lons / home_lats arrays so we can use vectorised haversine.
        home_midx = (adapter._air_home_id_to_idx.get(int(home_id))
                     if home_idx is None else None)
        cur_midx  = (adapter._air_home_id_to_idx.get(int(current_id))
                     if cur_idx  is None else None)

        # -- Precompute scalar params shared by every candidate in this call ------
        # _distance_utility and _popularity_utility each recompute these from
        # (ctx, macro, transport_mode) which are constant across all candidates.
        # Collapsing N re-computations to one saves ~15 Python function calls per step.
        _bw_def = {"home": 0.5, "work": 0.3, "current": 0.2}
        _bw     = _TIME_BASE_WEIGHTS.get(macro, _bw_def)
        _w_home = _bw["home"]
        _w_work = _bw["work"]
        _w_cur  = _bw["current"]
        if ctx.current_location_type == "WORK":           _w_work *= 3.0
        if ctx.is_weekday and 8 <= ctx.hour < 18:         _w_work *= 2.5
        if ctx.time_since_work is not None and ctx.time_since_work < 2.0:
            _w_work *= 1.0 + (1.0 - ctx.time_since_work / 2.0)
        if ctx.phase == "evening_home":                   _w_home *= 1.5
        if ctx.current_location_type == "OTHER":          _w_cur  *= 2.0
        if ctx.is_weekend:
            _w_work *= 0.5
            _w_cur  *= 1.3
        _wtot   = _w_home + _w_work + _w_cur
        _w_home /= _wtot;  _w_work /= _wtot;  _w_cur /= _wtot

        _dp        = _TIME_PARAMS.get(macro, {
            "lambda_base": 10.0,
            "t_max":   {"car": 40.0, "bicycle": 30.0, "foot": 20.0},
            "perceived": {"car": 1.0, "bicycle": 1.1, "foot": 1.2},
        })
        _perceived = _dp["perceived"].get(transport_mode, 1.0)
        _t_max     = _dp["t_max"].get(transport_mode, 40.0)
        if ctx.is_weekend:            _t_max *= 1.4
        if ctx.mode == "exploratory": _t_max *= 1.2
        _lam = float(_dp["lambda_base"])
        if ctx.is_weekend:            _lam *= 1.5
        if ctx.mode == "exploratory": _lam *= 1.3
        elif ctx.mode == "routine":   _lam *= 0.9
        if ctx.hour >= 21:            _lam *= 0.8

        _lam *= _MODE_LAMBDA_SCALE.get(transport_mode, 1.0)

        _imp = get_popularity_importance(sub_activity, macro)
        if ctx.mode == "exploratory": _imp = min(1.0, _imp * 1.2)
        elif ctx.mode == "routine":   _imp *= 0.8

        use_batch = (
            mat is not None
            and work_idx is not None
            and (home_idx is not None or home_midx is not None)
            and (cur_idx  is not None or cur_midx  is not None)
        )

        out_pois: List[int] = []

        if use_batch:
            in_idxs: List[int] = []
            in_pois: List[int] = []
            for poi_id in candidates:
                idx = id2idx.get(int(poi_id))
                if idx is not None:
                    in_idxs.append(idx)
                    in_pois.append(poi_id)
                else:
                    out_pois.append(poi_id)

            if in_pois:
                cols   = np.array(in_idxs)
                d_work = mat[work_idx, cols] * t_factor

                if home_idx is not None:
                    d_home = mat[home_idx, cols] * t_factor
                else:
                    d_home = _haversine_vec(
                        float(adapter._air_home_lons[home_midx]),
                        float(adapter._air_home_lats[home_midx]),
                        adapter._air_dest_lons[cols],
                        adapter._air_dest_lats[cols],
                    ) * t_factor

                if cur_idx is not None:
                    d_cur = mat[cur_idx, cols] * t_factor
                else:
                    d_cur = _haversine_vec(
                        float(adapter._air_home_lons[cur_midx]),
                        float(adapter._air_home_lats[cur_midx]),
                        adapter._air_dest_lons[cols],
                        adapter._air_dest_lats[cols],
                    ) * t_factor

                # Vectorised distance utility -- replaces N calls to _distance_utility
                _t_eff  = _w_home * d_home + _w_work * d_work + _w_cur * d_cur
                _t_perc = _t_eff * _perceived
                _over = np.maximum(0.0, _t_perc - _t_max)
                U_dist_arr = np.where(
                    _t_perc <= _t_max,
                    np.exp(-_t_perc / _lam),
                    np.exp(-_t_max / _lam) * np.exp(-_over / (_lam * 0.15))
                ).astype(np.float32)

                # Vectorised popularity utility -- replaces N calls to _popularity_utility
                _midx = getattr(adapter, "_density_macro_idx", {}).get(macro, -1)
                if adapter._density_arr is not None and _midx >= 0:
                    _den_arr = adapter._density_arr[cols, _midx]
                else:
                    _den_arr = np.fromiter(
                        (adapter._poi_density_cache.get(int(pid), {}).get(macro, 0.0)
                         for pid in in_pois),
                        dtype=np.float32, count=len(in_pois),
                    )
                _U_raw    = np.minimum(1.0, (_den_arr / _POP_NORM) ** _POP_BETA)
                U_pop_arr = (_imp * _U_raw + (1.0 - _imp)).astype(np.float32)

                # Variety is still per-POI (dict lookups into recent_poi_timestamps)
                for i, poi_id in enumerate(in_pois):
                    U_var = _variety_utility(
                        poi_id, sub_activity, recent_poi_timestamps,
                        adapter.poi_subtype_map, ctx.current_time,
                        last_category_visit=last_category_visit,
                    )
                    utilities[poi_id] = float(U_dist_arr[i]) * U_var * float(U_pop_arr[i])
        else:
            out_pois = list(candidates)

        # Fallback: individual lookups for any POI not in the air-distance matrix.
        for poi_id in out_pois:
            times  = adapter.get_travel_times_to_poi(
                poi_id, home_id, work_id, current_id, transport_mode
            )
            U_dist = _distance_utility(macro, times, ctx, transport_mode)
            U_var  = _variety_utility(
                poi_id, sub_activity, recent_poi_timestamps,
                adapter.poi_subtype_map, ctx.current_time,
                last_category_visit=last_category_visit,
            )
            density = adapter.get_poi_density(poi_id, macro)
            U_pop   = _popularity_utility(macro, sub_activity, density, ctx)
            utilities[poi_id] = U_dist * U_var * U_pop

        return utilities

    def _select_from_utilities(
        self,
        utilities: Dict[int, float],
        ctx: BehavioralContext,
        path_generator=None,
        current_id: Optional[int] = None,
        transport_mode: str = "car",
    ) -> int:
        n = len(utilities)
        k = min(self.top_k, n)
        items = list(utilities.items())
        if n > k:
            # O(N) partial sort to find top-k, then sort only those k items.
            all_scores = np.array([s for _, s in items], dtype=np.float64)
            top_idx    = np.argpartition(all_scores, -k)[-k:]
            top = [items[i] for i in top_idx]
        else:
            top = items
        top.sort(key=lambda x: x[1], reverse=True)
        pois   = [p for p, _ in top]
        scores = np.array([s for _, s in top], dtype=np.float64)
        if scores.max() == 0:
            return int(self.random_state.choice(pois))
        probs = _softmax(scores, ctx.temperature)
        chosen_idx = int(self.random_state.choice(len(pois), p=probs))

        if path_generator is not None and current_id is not None:
            for attempt in range(len(pois)):
                candidate = pois[(chosen_idx + attempt) % len(pois)]
                try:
                    result = path_generator.get_travel_time(
                        int(current_id), int(candidate), transport_mode
                    )
                    if result is not None:
                        return int(candidate)
                except Exception:
                    continue
            return int(pois[chosen_idx])

        return int(pois[chosen_idx])