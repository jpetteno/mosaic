from __future__ import annotations

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p


import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List

import config
from config import MACRO_COOLDOWN_HOURS
from config import DOUBLE_NEED_PROBS
from semantic_enrichment import sample_sub_activity, get_duration_for_subactivity, DEFAULT_WEEKLY_CAPS


class AgentState:
    """
    Event-based agent state -- V2 (redesign).

    Key changes vs V1:
    - Oracle dependency removed entirely.
    - Daily action budget = sum(daily_need_counts) + 1 discretionary slot.
    - Per-macro cooldowns: FOOD 6h, ERRAND 4h, LEISURE 2h.
      Cooldown starts at activity END (fast_forward totime), not start.
    - Needs can be required 1 or 2 times per day (conditional probability).
    - homeby22 need removed.
    - daily_macro_counts tracks how many times each macro was done today.
    - slots_remaining gives the observation builder the urgency signal.
    - last_food_end_time is set on FOOD completion only, never on LEISURE.
    - weekly_subactivity_counts is tracked for every macro, not only LEISURE.
    """

    # Canonical need keys in fixed order for observation building.
    NEEDS_CANONICAL = ["wenttowork", "eatoutside", "runerrand", "leisure"]

    # Map need key -> macro activity
    NEED_TO_MACRO: Dict[str, str] = {
        "wenttowork": "WORK",
        "eatoutside":  "FOOD",
        "runerrand":   "ERRAND",
        "leisure":     "LEISURE",
    }

    def __init__(self, agentid: int = 0, randomseed: Optional[int] = None):
        self.agent_id = agentid
        self.random_state = np.random.RandomState(randomseed)

        self.track_trajectories = bool(getattr(config, "TRACK_TRAJECTORIES", True))

        # --- Location state ---
        self.current_location: str = "HOME"
        self.current_location_id: Optional[int] = None
        self.current_subactivity: Optional[str] = None

        self.arrival_time: Optional[datetime] = None
        self.activity_duration_hours: Optional[float] = None
        self.can_leave_time: Optional[datetime] = None

        self.home_location_id: Optional[int] = None
        self.work_location_id: Optional[int] = None

        # --- Transport preference (assigned at episode reset, stable for whole episode) ---
        self.transport_preference: str = "car"

        # --- Needs ---
        self.daily_needs: Dict[str, int] = {}
        self.needs_accomplished: Dict[str, int] = {}

        # --- Time accounting ---
        self.time_at_locations_minutes: Dict[str, float] = {
            loc: 0.0 for loc in config.LOCATION_TYPES
        }
        self.time_traveling_minutes: float = 0.0

        # --- Visited POIs ---
        self.visited_locations: Dict[str, Set[int]] = {
            loc: set() for loc in config.LOCATION_TYPES
        }

        # --- Trajectories ---
        if self.track_trajectories:
            self.trajectories: Optional[List[Dict]] = []
            self.trajectory_counter: int = 0
        else:
            self.trajectories = None
            self.trajectory_counter = 0

        # --- Episode / day counters ---
        self.current_day: int = 0
        self.decision_count: int = 0

        # --- Daily macro counts (how many times each macro done today) ---
        self.daily_macro_counts: Dict[str, int] = {
            "WORK": 0, "FOOD": 0, "ERRAND": 0, "LEISURE": 0
        }

        # --- Action budget ---
        self.daily_budget: int = 1
        self.slots_remaining: int = 1

        # --- Per-macro cooldown end times ---
        self.macro_cooldown_end: Dict[str, Optional[datetime]] = {
            "FOOD":    None,
            "ERRAND":  None,
            "LEISURE": None,
        }

        # --- Weekly caps (rare leisure sub-activities) ---
        self.weekly_subactivity_counts: Dict[str, int] = {}
        self.weekly_caps: Dict[str, int] = dict(
            getattr(config, "WEEKLY_SUB_ACTIVITY_CAPS", DEFAULT_WEEKLY_CAPS)
        )

        # --- Favorite POIs ---
        self.favorites: Dict[str, list] = {}

        # --- Recent POI timestamps ---
        self.recent_poi_timestamps: Dict[int, datetime] = {}

        # --- Weekly aggregate counters ---
        self.weekly_non_home_activities: int = 0

        # --- Weekly distribution cache ---
        self._weekly_distribution_cache: Optional[Dict[str, float]] = None
        self._cache_valid: bool = False

    # =========================================================================
    # NEEDS GENERATION
    # =========================================================================

    def get_needs_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Return needs probabilities from config (oracle removed)."""
        src = getattr(config, "NEEDS_PROBABILITIES", {})
        normalized: Dict[str, Dict[str, float]] = {}
        for daytype, probs in dict(src).items():
            p = dict(probs)
            if "eatoutside" not in p and "eat_outside" in p:
                p["eatoutside"] = p.pop("eat_outside")
            if "runerrand" not in p and "run_errand" in p:
                p["runerrand"] = p.pop("run_errand")
            normalized[daytype] = p
        return normalized

    def generate_daily_needs(self, dayofweek: int) -> None:
        """
        Sample today's needs.

        Each need is first sampled as present/absent (Bernoulli).
        If present, a second Bernoulli decides whether it is needed
        once or twice (conditional double-need probability).

        Budget = sum of all required counts + 1 discretionary slot.
        HOME never consumes a slot.
        """
        daytype = "weekend" if bool(config.is_weekend(dayofweek)) else "weekday"
        probs = self.get_needs_probabilities().get(daytype, {})

        # --- Work: always 1 on weekdays, 0 on weekends ---
        wenttowork = 0 if daytype == "weekend" else 1

        # --- FOOD ---
        eatoutside_base = int(
            self.random_state.binomial(1, float(probs.get("eatoutside", 0.0)))
        )
        if eatoutside_base:
            double_p = DOUBLE_NEED_PROBS["eatoutside"][daytype]
            eatoutside = 2 if self.random_state.random() < double_p else 1
        else:
            eatoutside = 0

        # --- ERRAND ---
        runerrand_base = int(
            self.random_state.binomial(1, float(probs.get("runerrand", 0.0)))
        )
        if runerrand_base:
            double_p = DOUBLE_NEED_PROBS["runerrand"][daytype]
            runerrand = 2 if self.random_state.random() < double_p else 1
        else:
            runerrand = 0

        # --- LEISURE ---
        leisure_base = int(
            self.random_state.binomial(1, float(probs.get("leisure", 0.0)))
        )
        if leisure_base:
            double_p = DOUBLE_NEED_PROBS["leisure"][daytype]
            leisure = 2 if self.random_state.random() < double_p else 1
        else:
            leisure = 0

        self.daily_needs = {
            "wenttowork": wenttowork,
            "eatoutside":  eatoutside,
            "runerrand":   runerrand,
            "leisure":     leisure,
        }
        self.needs_accomplished = {k: 0 for k in self.daily_needs}

        # --- Compute budget ---
        total_required = sum(self.daily_needs.values())
        self.daily_budget = total_required + 1  # +1 discretionary
        self.slots_remaining = self.daily_budget

    # =========================================================================
    # COOLDOWN CHECKS
    # =========================================================================

    def is_on_cooldown(self, macro: str, current_time: datetime) -> bool:
        """Return True if macro activity is still in cooldown at current_time."""
        end = self.macro_cooldown_end.get(macro)
        if end is None:
            return False
        return current_time < end

    def is_food_on_cooldown_at(self, current_time: datetime) -> bool:
        return self.is_on_cooldown("FOOD", current_time)

    # =========================================================================
    # WEEKLY CAP CHECK
    # =========================================================================

    def is_weekly_capped(self, subactivity: str) -> bool:
        cap = self.weekly_caps.get(subactivity)
        if cap is None:
            return False
        return int(self.weekly_subactivity_counts.get(subactivity, 0)) >= int(cap)

    # =========================================================================
    # ACTIVITY LIFECYCLE
    # =========================================================================

    def start_activity(
        self,
        destination: str,
        destinationid: int,
        currenttime: datetime,
        traveltimeminutes: float = 0.0,
        subactivitytype: Optional[str] = None,
    ) -> None:
        self._cache_valid = False

        self.current_location = destination
        self.current_location_id = int(destinationid)
        self.arrival_time = currenttime

        self.visited_locations[destination].add(int(destinationid))
        self.recent_poi_timestamps[int(destinationid)] = currenttime

        # --- Weekly non-home counter ---
        if destination in ("WORK", "FOOD", "ERRAND", "LEISURE"):
            self.weekly_non_home_activities += 1

        # --- Duration sampling ---
        if destination == "HOME":
            hour = currenttime.hour + currenttime.minute / 60.0
            wakeup = 1.0 if 5.5 <= hour <= 23.5 else 2.0
            self.current_subactivity = None
            self.activity_duration_hours = wakeup
            self.can_leave_time = currenttime + timedelta(hours=wakeup)

        elif destination == "WORK":
            duration = float(
                config.sample_activity_duration("WORK", self.random_state, currenttime.weekday())
            )
            self.current_subactivity = None
            self.activity_duration_hours = duration
            self.can_leave_time = currenttime + timedelta(hours=duration)

        else:
            # FOOD / ERRAND / LEISURE -- sub-activity drives duration
            if subactivitytype is None:
                hour = currenttime.hour + currenttime.minute / 60.0
                subactivitytype = sample_sub_activity(
                    destination,
                    self.random_state,
                    hour=float(hour),
                    day_of_week=int(currenttime.weekday()),
                    agent_state=self,
                )
            self.current_subactivity = subactivitytype
            duration = (
                float(get_duration_for_subactivity(subactivitytype, self.random_state))
                if subactivitytype is not None
                else 0.75
            )
            self.activity_duration_hours = duration
            self.can_leave_time = currenttime + timedelta(hours=duration)

        self.time_traveling_minutes += float(traveltimeminutes)

        # --- Update needs & slot budget ---
        self._check_activity_needs()

        # HOME consumes a slot if needs are still unmet -- discourages mid-day
        # HOME bouncing. Returning HOME after all needs are satisfied is free,
        # which correctly represents legitimate end-of-day behaviour.
        needs_unmet = any(
            self.needs_accomplished.get(k, 0) < v
            for k, v in self.daily_needs.items()
            if v > 0
        )
        if destination != "HOME" or (needs_unmet and self.slots_remaining > 0):
            self.slots_remaining = max(0, self.slots_remaining - 1)

        if destination != "HOME":
            self.daily_macro_counts[destination] = (
                self.daily_macro_counts.get(destination, 0) + 1
            )

    def _check_activity_needs(self) -> None:
        """Increment needs_accomplished for the current location."""
        loc = self.current_location
        if loc == "WORK":
            self.needs_accomplished["wenttowork"] = (
                self.needs_accomplished.get("wenttowork", 0) + 1
            )
        elif loc == "FOOD":
            self.needs_accomplished["eatoutside"] = (
                self.needs_accomplished.get("eatoutside", 0) + 1
            )
        elif loc == "ERRAND":
            self.needs_accomplished["runerrand"] = (
                self.needs_accomplished.get("runerrand", 0) + 1
            )
        elif loc == "LEISURE":
            self.needs_accomplished["leisure"] = (
                self.needs_accomplished.get("leisure", 0) + 1
            )

    # =========================================================================
    # FAST FORWARD (activity completion)
    # =========================================================================

    def fast_forward(self, fromtime: datetime, totime: datetime) -> None:
        """
        Advance clock to end of current activity.

        Sets macro cooldown end time for FOOD/ERRAND/LEISURE.
        Updates weekly sub-activity counts for ALL non-HOME macros.
        """
        minutes = (totime - fromtime).total_seconds() / 60.0
        self.time_at_locations_minutes[self.current_location] += float(minutes)
        self._cache_valid = False

        macro = self.current_location

        # --- Set cooldown end at activity completion ---
        if macro in MACRO_COOLDOWN_HOURS:
            cooldown_h = MACRO_COOLDOWN_HOURS[macro]
            self.macro_cooldown_end[macro] = totime + timedelta(hours=cooldown_h)

        # --- Weekly sub-activity counts ---
        if macro in ("FOOD", "ERRAND", "LEISURE") and self.current_subactivity:
            s = str(self.current_subactivity)
            self.weekly_subactivity_counts[s] = (
                self.weekly_subactivity_counts.get(s, 0) + 1
            )

    # =========================================================================
    # OBSERVATION HELPERS
    # =========================================================================

    def get_visited_pois(self, macro_activity: str) -> list:
        return list(self.visited_locations.get(macro_activity, set()))

    def get_familiar_locations(
        self, macro_activity: str, sub_activity: str = None
    ) -> list:
        return list(self.visited_locations.get(macro_activity, set()))

    def get_time_until_can_leave_hours(self, currenttime: datetime) -> float:
        if self.can_leave_time is None or currenttime >= self.can_leave_time:
            return 0.0
        return float(
            max(0.0, (self.can_leave_time - currenttime).total_seconds() / 3600.0)
        )

    def get_need_feasibility(self, need: str, current_time: datetime) -> bool:
        dayofweek = current_time.weekday()
        hour = current_time.hour + current_time.minute / 60.0
        macro = self.NEED_TO_MACRO.get(need)
        if macro is None:
            return False
        return bool(config.is_location_open(macro, int(hour), dayofweek))

    def get_needs_satisfaction_score(self) -> float:
        needs = np.array(
            [float(self.daily_needs.get(k, 0)) for k in self.NEEDS_CANONICAL],
            dtype=np.float64,
        )
        done = np.array(
            [float(self.needs_accomplished.get(k, 0)) for k in self.NEEDS_CANONICAL],
            dtype=np.float64,
        )
        under = np.maximum(0.0, needs - done)
        over  = np.maximum(0.0, done  - needs)

        underw, overw = 2.0, 1.0
        distance = underw * np.sum(under) + overw * np.sum(over)
        total = np.sum(needs) + underw * len(needs) * 0.5 + overw
        if total == 0:
            return 1.0
        return float(1.0 - distance / total)

    def all_needs_satisfied(self) -> bool:
        for k in self.NEEDS_CANONICAL:
            required = int(self.daily_needs.get(k, 0))
            done     = int(self.needs_accomplished.get(k, 0))
            if done < required:
                return False
        return True

    def get_weekly_distribution(self) -> Dict[str, float]:
        if self._cache_valid and self._weekly_distribution_cache is not None:
            return self._weekly_distribution_cache

        total = sum(self.time_at_locations_minutes.values()) + self.time_traveling_minutes
        if total <= 0:
            dist: Dict[str, float] = {loc: 0.0 for loc in config.LOCATION_TYPES}
            dist["TRAVEL"] = 0.0
        else:
            dist = {
                loc: float(self.time_at_locations_minutes[loc]) / total
                for loc in config.LOCATION_TYPES
            }
            dist["TRAVEL"] = float(self.time_traveling_minutes) / total

        self._weekly_distribution_cache = dist
        self._cache_valid = True
        return dist

    # =========================================================================
    # TRAJECTORY LOGGING
    # =========================================================================

    def log_trajectory(
        self,
        originid: int,
        destinationid: int,
        departuretime: datetime,
        arrivaltime: datetime,
        traveltimeminutes: float,
        distance_km: float = 0.0,
        activity_type: str = "",
        sub_activity_type: str = "",
        linestring_id=None,
        transport_mode: str = "",
    ) -> None:
        if not self.track_trajectories or self.trajectories is None:
            return
        self.trajectories.append(
            {
                "trajectory_id":  self.trajectory_counter,
                "dep_loc":        int(originid),
                "dep_time":       departuretime,
                "arr_loc":        int(destinationid),
                "arr_time":       arrivaltime,
                "travel_minutes": float(traveltimeminutes),
                "distance_km":    float(distance_km),
                "activity_type":  activity_type,
                "sub_activity":   sub_activity_type,
                "linestring_id":  linestring_id,
                "transport_mode": transport_mode or self.transport_preference,
            }
        )
        self.trajectory_counter += 1

    # =========================================================================
    # RESET
    # =========================================================================

    def reset_for_new_day(self, dayofweek: int) -> None:
        """Called at day boundary. Regenerates needs and resets daily counters."""
        self.current_day += 1
        self.generate_daily_needs(dayofweek)
        self.daily_macro_counts = {"WORK": 0, "FOOD": 0, "ERRAND": 0, "LEISURE": 0}

    def reset_for_new_episode(self, starttime: datetime, homeid: int, workid: int,
                              transport_preference: str = "car") -> None:
        self.home_location_id = int(homeid)
        self.work_location_id = int(workid)
        self.transport_preference = str(transport_preference)

        self.current_location    = "HOME"
        self.current_location_id = int(homeid)
        self.current_subactivity = None

        self.arrival_time         = starttime
        self.activity_duration_hours = None
        self.can_leave_time       = starttime

        self.daily_needs        = {}
        self.needs_accomplished = {}

        self.time_at_locations_minutes = {loc: 0.0 for loc in config.LOCATION_TYPES}
        self.time_traveling_minutes    = 0.0
        self._cache_valid              = False

        self.visited_locations = {loc: set() for loc in config.LOCATION_TYPES}
        self.visited_locations["HOME"].add(int(homeid))

        if self.track_trajectories:
            self.trajectories = []
        else:
            self.trajectories = None
        self.trajectory_counter = 0

        self.current_day    = 0
        self.decision_count = 0

        self.daily_macro_counts = {"WORK": 0, "FOOD": 0, "ERRAND": 0, "LEISURE": 0}
        self.daily_budget    = 1
        self.slots_remaining = 1

        self.macro_cooldown_end     = {"FOOD": None, "ERRAND": None, "LEISURE": None}
        self.favorites              = {}
        self.weekly_subactivity_counts = {}
        self.recent_poi_timestamps  = {}
        self.weekly_non_home_activities = 0