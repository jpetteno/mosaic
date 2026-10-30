"""
ACTIVITY METADATA FOR RL MOBILITY SIMULATION
===========================================

Defines sub-activity types, durations, selection modes, *opening hours* and
*preferred time windows* (alpha weights).

Key features added:
- A2: Sub-activity opening/closing hours (filter closed sub-activities).
- A1: Preferred windows via alpha weighting (soft preferences, not rigid).
- Support for cooldowns/weekly caps by allowing an optional `agent_state`
  argument in sampling helpers.

Sub-activity keys must be the POI categories that city_map.sql actually
produces -- healthcare_doctor / healthcare_dentist for medical errands, and
sustenance_bar / sustenance_pub / sustenance_cafe for social venues. A key that
matches no real category selects no POI and the sub-activity simply never
occurs, with nothing to indicate why. The same keys appear in
ACTIVITY_PROBABILITIES, ACTIVITY_METADATA, OPEN_HOURS, PREFERRED_WINDOWS, the
cooldowns and the popularity importance, and must agree across all of them; see
region_representation.py's location_to_osm_categories for the macro buckets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _normalize_probs(p: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([max(0.0, float(v)) for v in p.values()], dtype=np.float64)
    s = float(vals.sum())
    if s <= 0.0:
        # fallback to uniform
        n = len(p)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in p.keys()}
    vals = vals / s
    return {k: float(vals[i]) for i, k in enumerate(p.keys())}


def _hour_in_interval(hour: float, start: float, end: float) -> bool:
    """
    Supports wrap-around intervals like 22 -> 4.
    """
    hour = float(hour) % 24.0
    start = float(start) % 24.0
    end = float(end) % 24.0
    if start <= end:
        return start <= hour < end
    # wrap-around
    return hour >= start or hour < end


def _is_open_from_windows(hour: float, windows: Sequence[Tuple[float, float]]) -> bool:
    if not windows:
        return True
    for s, e in windows:
        if _hour_in_interval(hour, s, e):
            return True
    return False


def _alpha_from_windows(hour: float, windows: Sequence[Tuple[float, float, float]], default: float) -> float:
    """
    Piecewise-constant alpha. If multiple windows match, uses max alpha.
    """
    if not windows:
        return float(default)
    best = None
    for s, e, a in windows:
        if _hour_in_interval(hour, s, e):
            best = float(a) if best is None else max(best, float(a))
    return float(default) if best is None else float(best)


# -----------------------------------------------------------------------------
# Sub-activity definitions
# -----------------------------------------------------------------------------

# FOOD covers meals only; cafe, bar, and pub venues are classified as LEISURE.
ACTIVITY_PROBABILITIES: Dict[str, Dict[str, float]] = {
    "FOOD": {
        "sustenance_restaurant": 0.45,
        "sustenance_fastfood": 0.35,
        "sustenance_food_court": 0.20,
    },
    "ERRAND": {
        "errand_bank": 0.20,
        "errand_atm": 0.18,
        "errand_post": 0.12,
        "errand_pharmacy": 0.10,
        "shop_supermarket": 0.15,
        "shop_convenience": 0.07,
        "shop_bakery": 0.06,
        "healthcare_doctor": 0.05,
        "healthcare_dentist": 0.03,
        "errand_townhall": 0.02,
        "errand_police": 0.01,
        "errand_courthouse": 0.01,
    },
    "LEISURE": {
        # Social/drinks
        "sustenance_cafe": 0.14,
        "sustenance_pub": 0.06,
        "sustenance_bar": 0.08,

        # Entertainment
        "entertainment_cinema": 0.06,
        "entertainment_theatre": 0.05,
        "entertainment_nightclub": 0.05,

        # Culture
        "tourism_museum": 0.04,
        "tourism_gallery": 0.04,
        "tourism_attraction": 0.06,
        "entertainment_arts": 0.03,

        # Shopping-as-leisure
        "shop_mall": 0.05,
        "shop_clothing": 0.03,
        "shop_books": 0.03,

        # Fitness/sports
        "leisure_fitness": 0.10,
        "leisure_sports": 0.07,

        # Outdoors
        "leisure_park": 0.10,
        "leisure_garden": 0.04,
        "leisure_playground": 0.03,

        # Social/community
        "social_worship": 0.03,
        "social_centre": 0.02,

        # Biergarten + casino
        "sustenance_biergarten": 0.04,
        "entertainment_casino": 0.01,

        # Community / outdoors
        "entertainment_community": 0.02,
        "tourism_viewpoint": 0.02,
    },
}

# Normalize
for _k in list(ACTIVITY_PROBABILITIES.keys()):
    ACTIVITY_PROBABILITIES[_k] = _normalize_probs(ACTIVITY_PROBABILITIES[_k])


# -----------------------------------------------------------------------------
# Metadata: duration, selection mode, density weight
# -----------------------------------------------------------------------------

ACTIVITY_METADATA: Dict[str, Dict[str, object]] = {
    # FOOD
    "sustenance_restaurant": {"duration": "MEDIUM", "selection": "CONVENIENCE", "density_weight": 2.0},
    "sustenance_fastfood": {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 1.0},
    "sustenance_food_court": {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 0.6},

    # ERRAND (CONVENIENCE)
    "errand_atm": {"duration": "QUICK", "selection": "CONVENIENCE", "density_weight": 0.1},
    "errand_bank": {"duration": "MEDIUM", "selection": "CONVENIENCE", "density_weight": 0.2},
    "errand_post": {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 0.2},
    "errand_pharmacy": {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 0.3},
    "shop_supermarket": {"duration": "MEDIUM", "selection": "CONVENIENCE", "density_weight": 0.5},
    "shop_convenience": {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 0.3},
    "shop_bakery": {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 0.4},

    # ERRAND (DESTINATION)
    "errand_townhall": {"duration": "MEDIUM", "selection": "DESTINATION", "density_weight": 0.0},
    "errand_police": {"duration": "MEDIUM", "selection": "DESTINATION", "density_weight": 0.0},
    "errand_courthouse": {"duration": "LONG", "selection": "DESTINATION", "density_weight": 0.0},
    "healthcare_doctor": {"duration": "MEDIUM", "selection": "DESTINATION", "density_weight": 0.0},
    "healthcare_dentist": {"duration": "MEDIUM", "selection": "DESTINATION", "density_weight": 0.0},

    # LEISURE - social/drinks
    "sustenance_cafe": {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 1.3},
    "sustenance_pub": {"duration": "LONG", "selection": "CONVENIENCE", "density_weight": 1.6},
    "sustenance_bar": {"duration": "LONG", "selection": "CONVENIENCE", "density_weight": 1.6},

    # LEISURE - entertainment / culture
    "entertainment_cinema": {"duration": "LONG", "selection": "CONVENIENCE", "density_weight": 1.5},
    "entertainment_theatre": {"duration": "LONGEST", "selection": "DESTINATION", "density_weight": 0.5},
    "entertainment_nightclub": {"duration": "LONGEST", "selection": "CONVENIENCE", "density_weight": 2.0},
    "tourism_museum": {"duration": "LONG", "selection": "DESTINATION", "density_weight": 0.5},
    "tourism_gallery": {"duration": "LONG", "selection": "DESTINATION", "density_weight": 0.3},
    "tourism_attraction": {"duration": "LONG", "selection": "DESTINATION", "density_weight": 0.3},
    "entertainment_arts": {"duration": "LONG", "selection": "DESTINATION", "density_weight": 0.4},

    # LEISURE - shopping
    "shop_mall": {"duration": "LONG", "selection": "CONVENIENCE", "density_weight": 0.8},
    "shop_clothing": {"duration": "MEDIUM", "selection": "CONVENIENCE", "density_weight": 1.2},
    "shop_books": {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 0.6},

    # LEISURE - fitness/sports/community
    "leisure_fitness": {"duration": "LONG", "selection": "DESTINATION", "density_weight": 0.0},
    "leisure_sports": {"duration": "LONG", "selection": "DESTINATION", "density_weight": 0.0},
    "entertainment_community": {"duration": "MEDIUM", "selection": "CONVENIENCE", "density_weight": 0.5},
    "tourism_viewpoint": {"duration": "SHORT", "selection": "DESTINATION", "density_weight": 0.0},

    # LEISURE - outdoors
    "leisure_park":       {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 0.0},
    "leisure_garden":     {"duration": "SHORT",  "selection": "CONVENIENCE", "density_weight": 0.0},
    "leisure_playground": {"duration": "SHORT",  "selection": "CONVENIENCE", "density_weight": 0.0},

    # LEISURE - social/community
    "social_worship": {"duration": "MEDIUM", "selection": "DESTINATION", "density_weight": 0.0},
    "social_centre":  {"duration": "SHORT", "selection": "CONVENIENCE", "density_weight": 0.3},

    # LEISURE - biergarten + casino
    "sustenance_biergarten": {"duration": "LONG", "selection": "CONVENIENCE", "density_weight": 1.2},
    "entertainment_casino":  {"duration": "LONGEST",   "selection": "DESTINATION", "density_weight": 0.5},
}

DURATION_RANGES = {
    "QUICK":   {"mean": 0.20, "std": 0.07, "min": 0.08, "max": 0.35},  # ATM, postbox, viewpoint (~12 min)
    "SHORT":   {"mean": 0.40, "std": 0.12, "min": 0.20, "max": 0.65},  # cafe, bakery, park (~24 min)
    "MEDIUM":  {"mean": 0.90, "std": 0.25, "min": 0.50, "max": 1.50},  # supermarket, restaurant (~54 min)
    "LONG":    {"mean": 2.00, "std": 0.50, "min": 1.00, "max": 3.50},  # cinema, museum, gym (~2h)
    "LONGEST": {"mean": 3.50, "std": 0.75, "min": 2.00, "max": 5.00},  # nightclub, theatre (~3.5h)
}


# -----------------------------------------------------------------------------
# Opening hours (A2) and preferred windows (A1)
# -----------------------------------------------------------------------------

# If a subactivity is not in OPEN_HOURS, it is treated as always open.
# Windows are (start_hour, end_hour) and can wrap around midnight.
OPEN_HOURS: Dict[str, List[Tuple[float, float]]] = {
    # FOOD (meals)
    "sustenance_restaurant": [(11.0, 14.5), (17.5, 22.0)],
    "sustenance_fastfood": [(10.0, 23.0)],
    "sustenance_food_court": [(10.0, 20.0)],

    # ERRAND
    "errand_atm": [(0.0, 24.0)],
    "shop_convenience": [(6.0, 23.0)],
    "shop_bakery": [(6.0, 18.0)],
    "shop_supermarket": [(8.0, 21.0)],
    "errand_pharmacy": [(8.0, 20.0)],
    "errand_bank": [(9.0, 17.0)],
    "errand_post": [(9.0, 18.0)],
    "errand_townhall": [(9.0, 16.0)],
    "errand_police": [(0.0, 24.0)],
    "errand_courthouse": [(9.0, 16.0)],
    "healthcare_doctor": [(9.0, 17.0)],
    "healthcare_dentist": [(9.0, 17.0)],

    # LEISURE - culture daytime
    "tourism_museum": [(10.0, 18.0)],
    "tourism_gallery": [(10.0, 18.0)],
    "tourism_attraction": [(9.0, 19.0)],
    "tourism_viewpoint": [(9.0, 19.0)],

    # LEISURE - entertainment
    "entertainment_cinema": [(11.0, 23.5)],
    "entertainment_theatre": [(18.0, 23.0)],
    "entertainment_nightclub": [(22.0, 4.0)],

    # LEISURE - social/drinks
    "sustenance_cafe": [(7.0, 20.0)],
    "sustenance_pub": [(16.0, 2.0)],
    "sustenance_bar": [(17.0, 2.0)],

    # LEISURE - shopping / fitness
    "shop_mall": [(10.0, 20.0)],
    "shop_clothing": [(10.0, 20.0)],
    "shop_books": [(10.0, 20.0)],
    "leisure_fitness": [(6.0, 23.0)],
    "leisure_sports": [(8.0, 22.0)],
    "entertainment_community": [(10.0, 20.0)],

    # LEISURE - outdoors (no entry = always open, but let's be explicit)
    "leisure_park":       [(6.0, 22.0)],
    "leisure_garden":     [(8.0, 20.0)],
    "leisure_playground": [(8.0, 20.0)],

    # LEISURE - social/community
    "social_worship": [(8.0, 13.0)],
    "social_centre":  [(9.0, 18.0)],

    # LEISURE - biergarten + casino
    "sustenance_biergarten": [(14.0, 23.0)],
    "entertainment_casino":  [(14.0, 4.0)],
}

# Preferred windows (piecewise alpha).
# If not provided, alpha defaults to 1.0 when open.
PREFERRED_WINDOWS: Dict[str, List[Tuple[float, float, float]]] = {
    # FOOD: lunch + dinner peaks
    "sustenance_restaurant": [(11.5, 14.0, 1.4), (18.0, 20.5, 1.4), (20.5, 22.0, 1.1)],
    "sustenance_fastfood": [(11.0, 14.5, 1.2), (18.0, 22.5, 1.2), (10.0, 11.0, 0.8)],
    "sustenance_food_court": [(11.0, 14.5, 1.2), (17.0, 19.5, 1.0)],

    # ERRAND: daytime preferred
    "errand_bank": [(9.0, 12.0, 1.2), (13.0, 16.0, 1.1)],
    "errand_post": [(9.0, 12.0, 1.2), (13.0, 17.0, 1.1)],
    "shop_supermarket": [(9.0, 12.0, 1.0), (16.0, 19.0, 1.2)],
    "shop_bakery": [(7.0, 10.0, 1.3), (10.0, 12.0, 0.9)],
    "errand_atm": [(0.0, 24.0, 1.0)],

    # LEISURE: diversify into day + evening
    "tourism_museum": [(10.0, 13.0, 1.2), (13.0, 17.0, 1.1)],
    "tourism_gallery": [(11.0, 14.0, 1.1), (14.0, 17.0, 1.1)],
    "entertainment_cinema": [(17.0, 22.5, 1.3), (13.0, 17.0, 0.9)],
    "entertainment_theatre": [(18.0, 22.5, 1.3)],
    "entertainment_nightclub": [(22.5, 3.0, 1.4)],
    "sustenance_cafe": [(7.0, 10.0, 1.3), (14.0, 18.0, 1.0)],
    "sustenance_pub": [(18.0, 1.5, 1.2)],
    "sustenance_bar": [(19.0, 2.0, 1.2)],
    "leisure_fitness": [(6.0, 9.0, 1.2), (17.0, 21.0, 1.1)],

    # LEISURE - outdoors: midday and afternoon peaks
    "leisure_park":    [(10.0, 13.0, 1.1), (14.0, 19.0, 1.2)],
    "leisure_garden":  [(10.0, 18.0, 1.1)],

    # LEISURE - social/community
    "social_worship":        [(9.0, 12.0, 1.4)],
    "sustenance_biergarten": [(16.0, 21.0, 1.3)],
    "entertainment_casino":  [(20.0, 2.0, 1.3)],
}

ALPHA_DEFAULT_OPEN = 1.0
ALPHA_DEFAULT_CLOSED = 0.0


# -----------------------------------------------------------------------------
# Caps / cooldown defaults
# -----------------------------------------------------------------------------

DEFAULT_FOOD_COOLDOWN_HOURS = 3.0
DEFAULT_WEEKLY_CAPS = {
    "entertainment_cinema": 2,
    "entertainment_nightclub": 1,
    "entertainment_theatre": 1,
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def is_subactivity_open(sub_activity_type: str, hour: float, day_of_week: Optional[int] = None) -> bool:
    windows = OPEN_HOURS.get(sub_activity_type, [])
    return _is_open_from_windows(hour, windows)


def get_time_alpha(sub_activity_type: str, hour: float, day_of_week: Optional[int] = None) -> float:
    if not is_subactivity_open(sub_activity_type, hour, day_of_week):
        return 0.0
    return _alpha_from_windows(hour, PREFERRED_WINDOWS.get(sub_activity_type, []), default=ALPHA_DEFAULT_OPEN)


def get_duration_for_subactivity(sub_activity_type: str, random_state=None) -> float:
    if sub_activity_type not in ACTIVITY_METADATA:
        return 0.75
    duration_class = str(ACTIVITY_METADATA[sub_activity_type]["duration"])
    cfg = DURATION_RANGES.get(duration_class)
    if cfg is None:
        return 0.75
    rs = np.random if random_state is None else random_state
    dur = float(rs.normal(cfg["mean"], cfg["std"]))
    return float(np.clip(dur, cfg["min"], cfg["max"]))


def get_selection_mode(sub_activity_type: str) -> str:
    if sub_activity_type not in ACTIVITY_METADATA:
        return "CONVENIENCE"
    return str(ACTIVITY_METADATA[sub_activity_type]["selection"])


def get_density_weight(sub_activity_type: str) -> float:
    if sub_activity_type not in ACTIVITY_METADATA:
        return 0.5
    return float(ACTIVITY_METADATA[sub_activity_type]["density_weight"])


def sample_sub_activity(
    activity_type: str,
    random_state=None,
    *,
    hour: Optional[float] = None,
    day_of_week: Optional[int] = None,
    agent_state: Optional[object] = None,
    last_sub_activity: Optional[str] = None,  # For backwards compatibility
) -> Optional[str]:
    """
    Context-aware sampling:
    - A2: filters closed subactivities if hour provided
    - A1: reweights by alpha if hour provided
    - B1/B2: can consult agent_state for cooldown/caps (if it provides the methods)
    """
    if activity_type not in ACTIVITY_PROBABILITIES:
        return None

    base = ACTIVITY_PROBABILITIES[activity_type]
    if not base:
        return None

    # Determine candidates
    candidates = list(base.keys())

    # Apply opening hours (A2)
    if hour is not None:
        candidates = [s for s in candidates if is_subactivity_open(s, float(hour), day_of_week)]

    if not candidates:
        # fallback: ignore opening hours
        candidates = list(base.keys())

    # Apply cooldown/caps via agent_state if present (B1/B2)
    if agent_state is not None:
        # FOOD cooldown
        # FOOD cooldown
        if activity_type == "FOOD" and hasattr(agent_state, "is_food_on_cooldown"):
            if bool(agent_state.is_food_on_cooldown(float(hour) if hour is not None else None)):
                # Block FOOD entirely when on cooldown
                return None

        # Weekly caps for leisure subactivities
        if activity_type == "LEISURE" and hasattr(agent_state, "is_weekly_capped"):
            candidates2 = []
            for s in candidates:
                if not bool(agent_state.is_weekly_capped(s)):
                    candidates2.append(s)
            if candidates2:
                candidates = candidates2

    # Build weights
    weights = []
    for s in candidates:
        w = float(base.get(s, 0.0))
        if hour is not None:
            w *= float(get_time_alpha(s, float(hour), day_of_week))
        weights.append(max(0.0, w))

    # Normalize and sample
    weights_arr = np.array(weights, dtype=np.float64)
    ssum = float(weights_arr.sum())
    if ssum <= 0.0:
        # fallback to uniform over candidates
        weights_arr[:] = 1.0 / len(candidates)
    else:
        weights_arr /= ssum

    rs = np.random if random_state is None else random_state
    idx = int(rs.choice(len(candidates), p=weights_arr))
    return str(candidates[idx])


# =============================================================================
# V5 LOCATION SELECTOR INTERFACE
# These functions are required by the V5 location selector and provide
# sub-activity-level cooldown, popularity importance, and macro listing.
# All keys are OSM-style (sustenance_restaurant, shop_supermarket, etc.)
# =============================================================================

# Per-sub-activity location-level cooldown (hours).
# How long before the SAME POI feels fresh again.
_LOCATION_COOLDOWN_HOURS: Dict[str, float] = {
    "sustenance_restaurant": 18.0,
    "sustenance_fastfood":    6.0,
    "sustenance_food_court":  6.0,
    "errand_bank":           48.0,
    "errand_atm":            24.0,
    "errand_post":           48.0,
    "errand_pharmacy":       24.0,
    "shop_supermarket":      36.0,
    "shop_convenience":      12.0,
    "shop_bakery":           12.0,
    "healthcare_doctor":         72.0,
    "healthcare_dentist":       168.0,
    "errand_townhall":       72.0,
    "errand_police":        168.0,
    "errand_courthouse":    168.0,
    "sustenance_cafe":            8.0,
    "sustenance_pub":            24.0,
    "sustenance_bar":            24.0,
    "entertainment_cinema":  72.0,
    "entertainment_theatre": 72.0,
    "entertainment_nightclub": 48.0,
    "tourism_museum":        72.0,
    "tourism_gallery":       72.0,
    "tourism_attraction":    48.0,
    "entertainment_arts":    48.0,
    "shop_mall":             48.0,
    "shop_clothing":         48.0,
    "shop_books":            48.0,
    "leisure_fitness":       24.0,
    "leisure_sports":        24.0,
    "entertainment_community": 48.0,
    "tourism_viewpoint":     48.0,

    # New
    "leisure_park":          12.0,
    "leisure_garden":        24.0,
    "leisure_playground":    24.0,
    "social_worship":        96.0,   # ~weekly
    "social_centre":         48.0,
    "sustenance_biergarten": 48.0,
    "entertainment_casino":  72.0,
}

# Per-sub-activity category-level cooldown (hours).
# How long before the SAME CATEGORY of activity feels fresh again.
_CATEGORY_COOLDOWN_HOURS: Dict[str, float] = {
    "sustenance_restaurant":  5.0,
    "sustenance_fastfood":    4.0,
    "sustenance_food_court":  4.0,
    "errand_bank":           24.0,
    "errand_atm":            12.0,
    "errand_post":           24.0,
    "errand_pharmacy":       12.0,
    "shop_supermarket":      24.0,
    "shop_convenience":       6.0,
    "shop_bakery":            6.0,
    "healthcare_doctor":         48.0,
    "healthcare_dentist":       120.0,
    "errand_townhall":       48.0,
    "errand_police":        120.0,
    "errand_courthouse":    120.0,
    "sustenance_cafe":            3.0,
    "sustenance_pub":            12.0,
    "sustenance_bar":            12.0,
    "entertainment_cinema":  36.0,
    "entertainment_theatre": 36.0,
    "entertainment_nightclub": 24.0,
    "tourism_museum":        36.0,
    "tourism_gallery":       36.0,
    "tourism_attraction":    24.0,
    "entertainment_arts":    24.0,
    "shop_mall":             24.0,
    "shop_clothing":         24.0,
    "shop_books":            24.0,
    "leisure_fitness":       12.0,
    "leisure_sports":        12.0,
    "entertainment_community": 24.0,
    "tourism_viewpoint":     24.0,

    # New
    "leisure_park":           4.0,
    "leisure_garden":        12.0,
    "leisure_playground":    12.0,
    "social_worship":        72.0,
    "social_centre":         24.0,
    "sustenance_biergarten": 24.0,
    "entertainment_casino":  48.0,
}

# Per-sub-activity popularity importance [0, 1].
# How much POI density matters in location scoring.
_POPULARITY_IMPORTANCE: Dict[str, float] = {
    "sustenance_restaurant": 0.7,
    "sustenance_fastfood":   0.5,
    "sustenance_food_court": 0.5,
    "errand_bank":           0.1,
    "errand_atm":            0.1,
    "errand_post":           0.1,
    "errand_pharmacy":       0.2,
    "shop_supermarket":      0.3,
    "shop_convenience":      0.2,
    "shop_bakery":           0.3,
    "healthcare_doctor":         0.0,
    "healthcare_dentist":        0.0,
    "errand_townhall":       0.0,
    "errand_police":         0.0,
    "errand_courthouse":     0.0,
    "sustenance_cafe":           0.7,
    "sustenance_pub":            0.6,
    "sustenance_bar":            0.6,
    "entertainment_cinema":  0.6,
    "entertainment_theatre": 0.4,
    "entertainment_nightclub": 0.8,
    "tourism_museum":        0.5,
    "tourism_gallery":       0.4,
    "tourism_attraction":    0.5,
    "entertainment_arts":    0.4,
    "shop_mall":             0.5,
    "shop_clothing":         0.5,
    "shop_books":            0.4,
    "leisure_fitness":       0.2,
    "leisure_sports":        0.2,
    "entertainment_community": 0.3,
    "tourism_viewpoint":     0.3,

    # New
    "leisure_park":          0.0,
    "leisure_garden":        0.0,
    "leisure_playground":    0.0,
    "social_worship":        0.0,
    "social_centre":         0.3,
    "sustenance_biergarten": 0.6,
    "entertainment_casino":  0.5,
}


def get_sub_activities_for_macro(macro_activity: str) -> List[str]:
    """Return list of sub-activity keys for a macro activity (OSM keys)."""
    return list(ACTIVITY_PROBABILITIES.get(macro_activity, {}).keys())


def get_location_cooldown(sub_activity_type: str) -> float:
    """Hours before the same POI feels fresh again."""
    return float(_LOCATION_COOLDOWN_HOURS.get(sub_activity_type, 24.0))


def get_category_cooldown(sub_activity_type: str) -> float:
    """Hours before the same sub-activity category feels fresh again."""
    return float(_CATEGORY_COOLDOWN_HOURS.get(sub_activity_type, 12.0))


def get_popularity_importance(sub_activity_type: str, macro_activity: str = "") -> float:
    """Importance weight for POI density [0, 1]."""
    return float(_POPULARITY_IMPORTANCE.get(sub_activity_type, 0.4))


def validate_metadata() -> None:
    for activity_type, probs in ACTIVITY_PROBABILITIES.items():
        total = float(sum(probs.values()))
        assert abs(total - 1.0) < 1e-6, f"{activity_type} probabilities sum to {total}"

        for sub in probs.keys():
            assert sub in ACTIVITY_METADATA, f"{sub} missing metadata"

    for sub, meta in ACTIVITY_METADATA.items():
        assert str(meta["duration"]) in DURATION_RANGES, f"{sub} has invalid duration class"
        assert str(meta["selection"]) in ("CONVENIENCE", "DESTINATION"), f"{sub} invalid selection"
        dw = float(meta["density_weight"])
        assert 0.0 <= dw <= 3.0, f"{sub} invalid density_weight: {dw}"


if __name__ == "__main__":
    validate_metadata()
    print("[OK] Activity metadata validation passed")