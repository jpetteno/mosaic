"""
FAVORITE PLACES IMPUTATION MODULE
==================================

Assigns realistic "favorite" POIs at agent initialization based on literature
about anchor points, spatial distributions, and preferential locations.

This is a STATIC assignment - favorites do not evolve during simulation.
They represent pre-existing preferences (like a person who already knows
their favorite spots when moving to a neighborhood).

Literature Foundation:
----------------------
- Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A.-L. (2008). 
  "Understanding individual human mobility patterns." Nature, 453(7196), 779-782.
  -> ~80% of trips within 10km
  -> Distance follows truncated power-law distribution
  -> People have 3-5 regularly visited locations per activity type

- Pappalardo, L., et al. (2015). "Returners and explorers dichotomy in 
  human mobility." Nature Communications, 6, 8166.
  -> Top 3-5 locations account for 60-80% of visits
  -> Spatial clustering around home/work anchors

- Song, C., et al. (2010). "Limits of Predictability in Human Mobility."
  Science, 327(5968), 1018-1021.
  -> Regular locations form spatial clusters
  -> Radius of gyration: 5-20 km typical for urban residents

Spatial Distribution Rules:
---------------------------
1. **75% close anchors** (<5 km from home OR work)
2. **15% medium range** (5-10 km from home/work)
3. **10% far exploration** (>10 km from both)
4. **Spatial clustering**: Favorites tend to cluster in same neighborhoods

Number of Favorites by Activity:
---------------------------------
- FOOD: 3 favorites (breakfast/lunch/dinner rotation)
- ERRAND: 1 favorite (primary supermarket)
- LEISURE: 2 favorites (regular sports/culture venues)

Total: 6 favorites per agent

Usage:
------
    from impute_favorites import assign_favorite_places, get_favorites_for_activity
    
    favorites = assign_favorite_places(
        agent_home_id=12345,
        agent_work_id=67890,
        city_data=city_adapter,
        random_state=np.random.RandomState(42)
    )
    
    # Returns:
    # {
    #     "FOOD": [poi1, poi2, poi3],
    #     "ERRAND": [poi6],
    #     "LEISURE": [poi7, poi8]
    # }
    
    # Use as FILTER (not bonus):
    fav_pois = get_favorites_for_activity(macro_activity, favorites)
    if fav_pois:
        candidates = fav_pois  # Use ONLY favorites
    else:
        candidates = all_pois  # Fallback to exploration
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from dataclasses import dataclass


# =============================================================================
# LITERATURE-BASED PARAMETERS
# =============================================================================

# Number of favorite locations per activity type.
NUM_FAVORITES = {
    "FOOD": 3,       # breakfast/lunch/dinner rotation
    "ERRAND": 1,     # primary supermarket
    "LEISURE": 2,    # regular sports/culture venues
}

# Distance distribution (Gonzalez et al. 2008)
DISTANCE_DISTRIBUTION = {
    "close": {
        "probability": 0.75,      # 75% of favorites are close
        "max_distance_km": 5.0,   # Within 5km of home/work
    },
    "medium": {
        "probability": 0.15,      # 15% are medium range
        "min_distance_km": 5.0,
        "max_distance_km": 10.0,
    },
    "far": {
        "probability": 0.10,      # 10% are far (exploration)
        "min_distance_km": 10.0,
        "max_distance_km": 50.0,  # Cap at 50km (still within city)
    }
}

# Spatial clustering parameter (Pappalardo et al. 2015)
# After selecting first favorite, subsequent favorites have X% chance
# to be near existing favorites (clustering effect)
CLUSTERING_PROBABILITY = 0.60  # 60% chance to cluster near existing favorite
CLUSTERING_RADIUS_KM = 2.0     # Within 2km of existing favorite


# =============================================================================
# MAIN IMPUTATION FUNCTION
# =============================================================================

def assign_favorite_places(
    agent_home_id: int,
    agent_work_id: int,
    city_data,  # CityData adapter with get_pois_by_macro, get_distance methods
    random_state,
    enforce_supermarket: bool = False
) -> Dict[str, List[int]]:
    """
    Assign favorite POIs for an agent based on literature-grounded distributions.

    Args:
        agent_home_id: POI ID of agent's home
        agent_work_id: POI ID of agent's work location
        city_data: City data adapter with POI access methods
        random_state: numpy RandomState for reproducibility
        enforce_supermarket: If True, force the ERRAND favorite to be a
            supermarket. Off by default -- it made the overwhelming majority
            of ERRAND visits a supermarket regardless of what else is nearby;
            the ERRAND favorite is now drawn from the general ERRAND pool,
            same as FOOD/LEISURE favorites.
        
    Returns:
        Dict mapping activity type to list of favorite POI IDs
        {
            "FOOD": [poi1, poi2, poi3, poi4, poi5],
            "ERRAND": [poi6],
            "LEISURE": [poi7, poi8, poi9]
        }
    """
    favorites = {}
    
    for macro_activity in ["FOOD", "ERRAND", "LEISURE"]:
        # Get target number of favorites (now fixed values)
        n_favorites = NUM_FAVORITES[macro_activity]
        
        # Get candidate POIs for this activity
        if enforce_supermarket and macro_activity == "ERRAND":
            # For ERRAND, prefer supermarkets; fall back to all ERRAND POIs
            all_candidates = city_data.get_pois_by_subactivity("shop_supermarket")
            if not all_candidates:
                all_candidates = city_data.get_pois_by_macro(macro_activity)
        else:
            all_candidates = city_data.get_pois_by_macro(macro_activity)
        
        if not all_candidates:
            favorites[macro_activity] = []
            continue

        # Precompute anchor distances once per macro to avoid repeated
        # city_data.get_distance(home/work, poi) calls in each sampling round.
        min_anchor_distance_km = {}
        for poi_id in all_candidates:
            if poi_id == agent_home_id or poi_id == agent_work_id:
                continue
            d_home = city_data.get_distance(agent_home_id, poi_id)
            d_work = city_data.get_distance(agent_work_id, poi_id)
            min_anchor_distance_km[poi_id] = min(d_home, d_work)
        
        # Assign favorites according to distance distribution
        selected_favorites = []
        
        for i in range(n_favorites):
            if i == 0:
                # First favorite: sample from distance distribution
                poi = _sample_by_distance_distribution(
                    all_candidates,
                    agent_home_id,
                    agent_work_id,
                    city_data,
                    random_state,
                    exclude=selected_favorites,
                    min_anchor_distance_km=min_anchor_distance_km,
                )
            else:
                # Subsequent favorites: consider clustering
                if random_state.random() < CLUSTERING_PROBABILITY:
                    # Cluster near existing favorite
                    poi = _sample_near_existing(
                        all_candidates,
                        selected_favorites,
                        city_data,
                        random_state,
                        exclude=selected_favorites
                    )
                else:
                    # Independent sample from distance distribution
                    poi = _sample_by_distance_distribution(
                        all_candidates,
                        agent_home_id,
                        agent_work_id,
                        city_data,
                        random_state,
                        exclude=selected_favorites,
                        min_anchor_distance_km=min_anchor_distance_km,
                    )
            
            if poi is not None:
                selected_favorites.append(poi)
        
        favorites[macro_activity] = selected_favorites
    
    return favorites


def _sample_by_distance_distribution(
    candidates: List[int],
    home_id: int,
    work_id: int,
    city_data,
    random_state,
    exclude: List[int] = None,
    min_anchor_distance_km: Dict[int, float] = None,
) -> int:
    """
    Sample a POI according to distance distribution (75/15/10).
    
    Args:
        candidates: List of candidate POI IDs
        home_id: Home POI ID
        work_id: Work POI ID
        city_data: City data adapter
        random_state: Random state
        exclude: POIs to exclude from selection
        
    Returns:
        Selected POI ID, or None if no valid candidates
    """
    if exclude is None:
        exclude = []
    
    # Filter out excluded POIs
    available = [c for c in candidates if c not in exclude and c != home_id and c != work_id]
    
    if not available:
        return None
    
    # Sample distance range: close (75%), medium (15%), far (10%)
    rand = random_state.random()
    if rand < 0.75:
        distance_range = "close"
    elif rand < 0.90:
        distance_range = "medium"
    else:
        distance_range = "far"
    
    # Get parameters for this range
    params = DISTANCE_DISTRIBUTION[distance_range]
    min_dist = params.get("min_distance_km", 0.0)
    max_dist = params["max_distance_km"]
    
    # Filter candidates by distance from home OR work
    valid_candidates = []
    for poi_id in available:
        if min_anchor_distance_km is not None and poi_id in min_anchor_distance_km:
            d_min_anchor = min_anchor_distance_km[poi_id]
        else:
            # Fallback path keeps exact original behavior.
            d_home = city_data.get_distance(home_id, poi_id)
            d_work = city_data.get_distance(work_id, poi_id)
            d_min_anchor = min(d_home, d_work)
        
        if min_dist <= d_min_anchor <= max_dist:
            valid_candidates.append(poi_id)
    
    # Fallback: if no candidates in desired range, try any available
    if not valid_candidates:
        valid_candidates = available
    
    if not valid_candidates:
        return None
    
    # Random selection from valid candidates
    return random_state.choice(valid_candidates)


def _sample_near_existing(
    candidates: List[int],
    existing_favorites: List[int],
    city_data,
    random_state,
    exclude: List[int] = None
) -> int:
    """
    Sample a POI near an existing favorite (clustering).
    
    Args:
        candidates: List of candidate POI IDs
        existing_favorites: Already selected favorites
        city_data: City data adapter
        random_state: Random state
        exclude: POIs to exclude
        
    Returns:
        Selected POI ID, or None if no valid candidates
    """
    if exclude is None:
        exclude = []
    
    if not existing_favorites:
        return None
    
    # Filter available candidates
    available = [c for c in candidates if c not in exclude]
    
    if not available:
        return None
    
    # Find candidates within CLUSTERING_RADIUS_KM of any existing favorite
    nearby_candidates = []
    for poi_id in available:
        for fav_id in existing_favorites:
            distance = city_data.get_distance(fav_id, poi_id)
            if distance <= CLUSTERING_RADIUS_KM:
                nearby_candidates.append(poi_id)
                break  # Found one close favorite, that's enough
    
    # Fallback: if no nearby candidates, return random from available
    if not nearby_candidates:
        nearby_candidates = available
    
    return random_state.choice(nearby_candidates)


# =============================================================================
# FILTER FUNCTION (UPDATED - REPLACES get_favorite_bonus)
# =============================================================================

def get_favorites_for_activity(
    macro_activity: str,
    favorites: Dict[str, List[int]]
) -> List[int]:
    """
    Get list of favorite POI IDs for a given activity.
    
    This is the NEW filter function that replaces get_favorite_bonus.
    Use this to filter the candidate set to ONLY favorites.
    
    Args:
        macro_activity: "FOOD", "ERRAND", or "LEISURE"
        favorites: Agent's favorites dict {activity: [poi_ids]}
        
    Returns:
        List of favorite POI IDs for this activity (empty if no favorites)
        
    Example:
        favorites = {"FOOD": [1, 2, 3], "ERRAND": [6], "LEISURE": [7, 8]}
        
        fav_pois = get_favorites_for_activity("FOOD", favorites)
        # Returns: [1, 2, 3]
        
        if fav_pois:
            candidates = fav_pois  # Use ONLY favorites as candidates
        else:
            candidates = all_pois  # No favorites, explore
    """
    return favorites.get(macro_activity, [])


def is_favorite(poi_id: int, macro_activity: str, favorites: Dict[str, List[int]]) -> bool:
    """
    Check if a POI is a favorite for the given activity.
    
    Args:
        poi_id: POI identifier
        macro_activity: Activity type
        favorites: Agent's favorites dict
        
    Returns:
        True if POI is a favorite, False otherwise
    """
    activity_favorites = favorites.get(macro_activity, [])
    return poi_id in activity_favorites


# =============================================================================
# BACKWARDS COMPATIBILITY (DEPRECATED)
# =============================================================================

def get_favorite_bonus(poi_id: int, macro_activity: str, favorites: Dict[str, List[int]]) -> float:
    """
    DEPRECATED: Use get_favorites_for_activity() instead.
    
    This function is kept for backwards compatibility but should not be used
    in the new filter-based architecture.
    
    Returns:
        1.0 if POI is a favorite, 0.0 otherwise
    """
    return 1.0 if is_favorite(poi_id, macro_activity, favorites) else 0.0


# =============================================================================
# ANALYSIS & VALIDATION
# =============================================================================

def analyze_favorite_distribution(
    favorites: Dict[str, List[int]],
    home_id: int,
    work_id: int,
    city_data
) -> Dict:
    """
    Analyze the distribution of assigned favorites.
    
    Args:
        favorites: Assigned favorites dict
        home_id: Home POI ID
        work_id: Work POI ID
        city_data: City data adapter
        
    Returns:
        Analysis dict with distance distributions and counts
    """
    analysis = {
        "total_favorites": 0,
        "by_activity": {},
        "distance_distribution": {"close": 0, "medium": 0, "far": 0},
        "distance_percentages": {},
    }
    
    all_favorite_pois = []
    
    for activity, pois in favorites.items():
        analysis["by_activity"][activity] = len(pois)
        analysis["total_favorites"] += len(pois)
        all_favorite_pois.extend(pois)
        
        # Analyze distances for this activity
        for poi_id in pois:
            d_home = city_data.get_distance(home_id, poi_id)
            d_work = city_data.get_distance(work_id, poi_id)
            d_min = min(d_home, d_work)
            
            if d_min < 5.0:
                analysis["distance_distribution"]["close"] += 1
            elif d_min < 10.0:
                analysis["distance_distribution"]["medium"] += 1
            else:
                analysis["distance_distribution"]["far"] += 1
    
    # Calculate percentages
    total = analysis["total_favorites"]
    if total > 0:
        analysis["distance_percentages"] = {
            "close": analysis["distance_distribution"]["close"] / total * 100,
            "medium": analysis["distance_distribution"]["medium"] / total * 100,
            "far": analysis["distance_distribution"]["far"] / total * 100,
        }
    
    return analysis


def validate_favorite_distribution(favorites_analysis: Dict) -> Dict[str, bool]:
    """
    Validate that favorite distribution matches literature expectations.
    
    Args:
        favorites_analysis: Output from analyze_favorite_distribution
        
    Returns:
        Dict of validation checks (True = passes)
    """
    validations = {}
    
    # Check distance distribution (+/-15% tolerance)
    dist_pct = favorites_analysis["distance_percentages"]
    validations["close_pct_valid"] = 60.0 <= dist_pct.get("close", 0) <= 90.0
    validations["medium_pct_valid"] = 5.0 <= dist_pct.get("medium", 0) <= 30.0
    validations["far_pct_valid"] = 0.0 <= dist_pct.get("far", 0) <= 20.0
    
    # Check total count (3 + 1 + 2 = 6 total)
    total = favorites_analysis["total_favorites"]
    validations["total_favorites_count"] = total == 6  # Exact match expected
    
    # Check per-activity counts
    by_activity = favorites_analysis["by_activity"]
    validations["food_count"] = by_activity.get("FOOD", 0) == 3
    validations["errand_count"] = by_activity.get("ERRAND", 0) == 1
    validations["leisure_count"] = by_activity.get("LEISURE", 0) == 2
    
    return validations


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FAVORITE PLACES IMPUTATION - FILTER-BASED ARCHITECTURE")
    print("="*70)
    print()
    
    print("This module assigns realistic favorite POIs based on:")
    print("  - Distance distributions (Gonzalez et al. 2008): 75/15/10")
    print("  - Spatial clustering (Pappalardo et al. 2015): 60% cluster")
    print("  - Fixed counts per activity:")
    print("    - FOOD: 3 favorites")
    print("    - ERRAND: 1 favorite (supermarket)")
    print("    - LEISURE: 2 favorites")
    print()
    print("Usage:")
    print("  favorites = assign_favorite_places(...)")
    print()
    print("  # In location selector - FILTER, not bonus:")
    print("  fav_pois = get_favorites_for_activity(macro_activity, favorites)")
    print("  if fav_pois:")
    print("      candidates = fav_pois  # Use ONLY favorites")
    print("  else:")
    print("      candidates = all_pois  # Fallback to exploration")
    print()
    print("[OK] Module ready for filter-based architecture.")