from __future__ import annotations

"""
SINGLE WORKER MOBILITY ENVIRONMENT  V2
========================================

Event-based Gymnasium environment for RL mobility simulation.

Event-based Gymnasium environment for the RL mobility simulation.
One environment instance = one synthetic agent over EPISODE_DAYS days.
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p



import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple

import config
from region_data import get_spatial_env
from agent_state import AgentState
from reward_function import RewardFunction
from state_builder import StateBuilder
from region_representation import RegionRepresentation
from poi_selector import POISelector
from semantic_enrichment import sample_sub_activity
from impute_favorites import assign_favorite_places
from path_generator import PathGenerator

# Avoids re-creating this dict literal on every action in every step().
_MACRO_TO_NEED: Dict[str, str] = {
    "WORK":    "wenttowork",
    "FOOD":    "eatoutside",
    "ERRAND":  "runerrand",
    "LEISURE": "leisure",
}


class MosaicEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, seed: int = 0):
        super().__init__()
        self.seedvalue   = seed
        self.randomstate = np.random.RandomState(seed)

        # --- Spatial data ---
        self.spatialenv = None
        if bool(getattr(config, "USE_SPATIAL_DATA", False)):
            self._load_spatial_data()

        # --- City ---
        # All three mode speeds are passed explicitly. RegionRepresentation's
        # class defaults (15.0 cycling / 40.0 driving) would otherwise apply,
        # and the driving default disagrees with TRANSPORT_MODE_SPEEDS["car"].
        self.city = RegionRepresentation(
            grid_size=100,
            num_agents=30,
            walking_speed_kmh=float(config.TRANSPORT_MODE_SPEEDS.get("foot", 5.0)),
            cycling_speed_kmh=float(config.TRANSPORT_MODE_SPEEDS.get("bicycle", 15.0)),
            driving_speed_kmh=float(config.TRANSPORT_MODE_SPEEDS.get("car", 30.0)),
            seed=seed,
            use_spatial=bool(getattr(config, "USE_SPATIAL_DATA", False)),
            spatial_env=self.spatialenv,
        )

        # --- Path library -- always loaded, used for mode-aware routing ---
        # Also provides trajectory geometry when TRACK_TRAJECTORIES = True.
        self._path_lib: Optional[PathGenerator] = None
        try:
            self._path_lib = PathGenerator.load(
                getattr(config, "SPATIAL_CACHE_FILE", "data/spatial_cache_filtered.pkl")
            )
        except Exception as e:
            print(f"[ENV] Warning: PathGenerator not loaded: {e}")

        # --- Sub-components (placeholders; recreated on reset) ---
        self.agent            = AgentState()
        self.rewardcalc       = RewardFunction()
        self.obsbuilder       = StateBuilder()
        self.locationselector = POISelector(
            self.city, random_seed=seed, path_generator=self._path_lib
        )

        # --- Gym spaces ---
        self.actionspace = spaces.Discrete(int(config.NUM_ACTIONS))
        self.observationspace = spaces.Box(
            low=-1.0,          # hour_sin/cos can be negative
            high=1.0,
            shape=(int(config.OBS_DIM),),
            dtype=np.float32,
        )

        # --- Episode time trackers ---
        self.currentdatetime:   datetime = datetime.now()
        self.episodestarttime:  datetime = datetime.now()
        self.episodeendtime:    datetime = datetime.now()

    # ------------------------------------------------------------------
    # Gym properties
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Box:
        return self.observationspace

    @property
    def action_space(self) -> spaces.Discrete:
        return self.actionspace

    # Keep camelCase aliases used by existing code
    @property
    def current_datetime(self) -> datetime:
        return self.currentdatetime

    @current_datetime.setter
    def current_datetime(self, v: datetime) -> None:
        self.currentdatetime = v

    @property
    def episode_start_time(self) -> datetime:
        return self.episodestarttime

    @property
    def episode_end_time(self) -> datetime:
        return self.episodeendtime

    # ------------------------------------------------------------------
    # Spatial data loading
    # ------------------------------------------------------------------

    def _load_spatial_data(self) -> None:
        try:
            self.spatialenv = get_spatial_env(
                cache_file=config.SPATIAL_CACHE_FILE
            )
        except Exception as e:
            # The fallback to the synthetic grid city must be VISIBLE: a silent
            # one leaves the model training against a toy city that looks
            # superficially fine. One real cause has been a UnicodeEncodeError
            # raised while printing inside RegionData on a non-UTF-8 Windows
            # console, which has nothing to do with the spatial data itself.
            print(f"[WARNING] Failed to load spatial data ({e!r}); "
                  f"falling back to the synthetic city.")
            self.spatialenv = None

    # Kept for callers that use old name
    def loadspatialdata(self) -> None:
        self._load_spatial_data()

    # ------------------------------------------------------------------
    # TRANSPORT PREFERENCE ASSIGNMENT
    # ------------------------------------------------------------------

    # Transport mode prior -- loaded from custom_settings.py via config
    @property
    def _TRANSPORT_PRIOR(self):
        from config import TRANSPORT_MODE_PRIOR
        return TRANSPORT_MODE_PRIOR

    # Feasibility thresholds (minutes)
    _FOOT_COMMUTE_MAX   = 45.0   # foot+PT ceiling -- above this, can't be foot
    _BIKE_COMMUTE_MAX   = 30.0   # bike ceiling -- above this, can't be bike
    _CAR_TOO_CLOSE      = 8.0    # car feels wrong -- switch to bike
    _FOOT_TRIVIAL       = 5.0    # so close car is absurd -- switch to foot

    def _assign_transport_preference(
        self, home_id: int, work_id: int
    ) -> str:
        """
        Draw transport preference from population prior, then apply
        bidirectional feasibility override based on home->work commute time.

        Upward overrides (too far):
          foot  -> bicycle  if foot > 45 min but bike <= 30 min
          foot  -> car      if foot > 45 min AND bike > 30 min
          bicycle -> car    if bike > 30 min

        Downward overrides (too close):
          car   -> bicycle  if car < 8 min
          car   -> foot     if car < 5 min
        """
        # Draw from prior
        modes  = list(self._TRANSPORT_PRIOR.keys())
        probs  = list(self._TRANSPORT_PRIOR.values())
        pref   = str(self.randomstate.choice(modes, p=probs))

        # Use air distance for feasibility check -- fast, no Dijkstra
        # Detour factor 1.3 corrects air distance to road distance estimate
        try:
            air_km = self.locationselector.city_adapter.get_air_distance_km(
                home_id, work_id
            )
            DETOUR = 1.3
            foot_t = (air_km * DETOUR / 5.0)  * 60.0
            bike_t = (air_km * DETOUR / 15.0) * 60.0
            car_t  = (air_km * DETOUR / 30.0) * 60.0
        except Exception:
            return pref

        # --- Upward overrides ---
        if pref == "foot":
            if foot_t > self._FOOT_COMMUTE_MAX:
                if bike_t <= self._BIKE_COMMUTE_MAX:
                    pref = "bicycle"
                else:
                    pref = "car"
        if pref == "bicycle":
            if bike_t > self._BIKE_COMMUTE_MAX:
                pref = "car"

        # --- Downward overrides ---
        if pref == "car":
            if car_t < self._FOOT_TRIVIAL:
                pref = "foot"
            elif car_t < self._CAR_TOO_CLOSE:
                pref = "bicycle"

        return pref

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        if seed is not None:
            self.seedvalue   = seed
            self.randomstate = np.random.RandomState(seed)
        else:
            seed = int(np.random.randint(0, 1_000_000))
        super().reset(seed=seed)

        # Episode window
        self.episodestarttime = datetime(2024, 1, 1, 0, 0)
        self.episodeendtime   = self.episodestarttime + timedelta(
            days=int(getattr(config, "EPISODE_DAYS", 7))
        )
        self.currentdatetime = self.episodestarttime
        self._last_dep_time: Optional[datetime] = None
        self._last_arr_time: Optional[datetime] = None
        self._last_act_end:  Optional[datetime] = None

        # Home / work assignment
        homeid = self.city.get_random_place("HOME")
        workid = self.city.get_random_place("WORK")
        homeint = int(homeid) if isinstance(homeid, int) else int(homeid[1])
        workint = int(workid) if isinstance(workid, int) else int(workid[1])

        # Assign transport preference (prior + feasibility check).
        # Always go through _assign_transport_preference(): it draws from
        # TRANSPORT_MODE_PRIOR and then applies the commute-distance
        # feasibility override. An inline draw here would bypass both.
        transport_pref = self._assign_transport_preference(homeint, workint)

        # Fresh agent + reward calculator
        self.agent = AgentState(agentid=0, randomseed=seed)
        self.agent.reset_for_new_episode(
            self.episodestarttime, homeint, workint,
            transport_preference=transport_pref,
        )
        self.agent.generate_daily_needs(self.currentdatetime.weekday())

        # Impute static favorite POIs  assigned once per episode, never change
        self.agent.favorites = assign_favorite_places(
            agent_home_id=homeint,
            agent_work_id=workint,
            city_data=self.locationselector.city_adapter,
            random_state=self.randomstate,
        )

        self.rewardcalc = RewardFunction()
        self.rewardcalc.reset()

        obs  = self.obsbuilder.build_observation(self.agent, self.currentdatetime)
        info = self._base_info()
        return obs, info

    # ------------------------------------------------------------------
    # MIDNIGHT SKIP HELPER
    # ------------------------------------------------------------------

    def _check_midnight_skip(self) -> float:
        """
        If the agent is at HOME and the clock is in the dead-of-night window
        (00:0005:00), fire the end-of-day bonus and advance to 5AM next day.

        Covers two cases:
          1. Agent arrives home past midnight (HOME action completes after 00:00)
          2. Agent is already home and takes any action past midnight
             (skip fires before the action is processed)

        Returns the bonus earned (0.0 if no skip happened).
        """
        if self.agent is None:
            return 0.0
        if self.agent.current_location != "HOME":
            return 0.0
        if not (0 <= self.currentdatetime.hour < 5):
            return 0.0

        # Fire end-of-day bonus before resetting needs
        bonus = self.rewardcalc.compute_daily_reward(self.agent)

        # Advance clock to 5AM same calendar day
        # (we're already past midnight so "same day" is the correct next morning)
        self.currentdatetime = self.currentdatetime.replace(hour=5, minute=0, second=0, microsecond=0)

        # Reset agent for new day
        self.agent.reset_for_new_day(self.currentdatetime.weekday())
        self.agent.current_location    = "HOME"
        self.agent.current_location_id = self.agent.home_location_id
        self.agent.can_leave_time      = self.currentdatetime

        return float(bonus)

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.agent is not None

        # Fire midnight skip before any action is processed.
        # Covers: agent already home past midnight choosing any action,
        # and agent arriving home past midnight (HOME action will re-trigger
        # the existing day-done check below, which is safe  bonus only fires
        # once because reset_for_new_day() increments current_day).
        midnight_bonus = self._check_midnight_skip()

        valid_mask = self.get_valid_actions_mask()
        action_valid = bool(valid_mask[int(action)] > 0.0)

        # If the agent picks an invalid action, redirect to HOME silently.
        # This should be rare with MaskablePPO; the redirect prevents crashes.
        if not action_valid:
            action = int(config.ACTION_GO_HOME)

        destination = config.action_to_location_type(int(action))
        current_hour = self.currentdatetime.hour + self.currentdatetime.minute / 60.0
        dayofweek    = int(self.currentdatetime.weekday())

        # ------------------------------------------------------------------
        # Sub-activity sampling (single source of truth  env samples, not agent)
        # ------------------------------------------------------------------
        subactivitytype: Optional[str] = None
        if destination in ("FOOD", "ERRAND", "LEISURE"):
            subactivitytype = sample_sub_activity(
                destination,
                self.randomstate,
                hour=float(current_hour),
                day_of_week=dayofweek,
                agent_state=self.agent,
            )

        # ------------------------------------------------------------------
        # Location selection
        # ------------------------------------------------------------------
        # Normalise PlaceId: synthetic city returns (place_type, id) tuples;
        # spatial city returns plain ints.  Everything downstream expects int.
        def _place_int(place_id) -> int:
            if isinstance(place_id, tuple):
                return int(place_id[1])
            return int(place_id) if place_id is not None else 0

        cur_loc_int  = _place_int(self.agent.current_location_id)
        home_loc_int = _place_int(self.agent.home_location_id)
        work_loc_int = _place_int(self.agent.work_location_id)

        raw_destinationid = self.locationselector.select_location(
            activity_type=destination,
            current_location_id=cur_loc_int,
            home_location_id=home_loc_int,
            work_location_id=work_loc_int,
            visited_locations=self.agent.visited_locations[destination],
            sub_activity_type=subactivitytype,
            current_hour=float(current_hour),
            recent_poi_timestamps=self.agent.recent_poi_timestamps,
            agent_state=self.agent,
            current_time=self.currentdatetime,
        )
        destinationid = _place_int(raw_destinationid)

        # ------------------------------------------------------------------
        # Travel -- mode-aware, advance clock by travel time FIRST
        # Air distance x detour factor used for training speed.
        # Dijkstra only runs for geometry in simulation mode.
        # ------------------------------------------------------------------
        transportmode = str(getattr(self.agent, "transport_preference", "car"))
        # transportmode must be passed through: without it every trip is timed
        # as a car trip regardless of the agent's preference.
        traveltimemin, _ = self.city.gettraveltime(
            cur_loc_int, destinationid, transportmode
        )
        distancekm = self.city.getdistancekm(
            cur_loc_int, destinationid
        )

        arrival_time = self.currentdatetime + timedelta(minutes=float(traveltimemin))
        self._last_dep_time = self.currentdatetime
        self._last_arr_time = arrival_time

        # ------------------------------------------------------------------
        # Trajectory logging -- only during simulation (TRACK_TRAJECTORIES=True)
        # PathGenerator Dijkstra only called here, never during training.
        # ------------------------------------------------------------------
        if int(destinationid) != int(cur_loc_int) and self.agent.track_trajectories and self.agent.trajectories is not None:
            ls_id = None
            if self._path_lib is not None and self._path_lib.available:
                ls_id = self._path_lib.get_linestring_id(
                    cur_loc_int, int(destinationid), transportmode
                )
            self.agent.log_trajectory(
                originid          = int(cur_loc_int),
                destinationid     = int(destinationid),
                departuretime     = self.currentdatetime,
                arrivaltime       = arrival_time,
                traveltimeminutes = float(traveltimemin),
                distance_km       = float(distancekm),
                activity_type     = destination,
                sub_activity_type = subactivitytype or "",
                linestring_id     = ls_id,
                transport_mode    = transportmode,
            )

        # ------------------------------------------------------------------
        # Activity start  duration sampled here, can_leave_time set
        # ------------------------------------------------------------------
        self.agent.start_activity(
            destination,
            destinationid,
            arrival_time,
            float(traveltimemin),
            subactivitytype=subactivitytype,
        )

        # Immediate reward uses DEPARTURE time for phase calculation 
        # the decision was made now, before travel.
        immediate_reward = self.rewardcalc.compute_immediate_need_reward(
            self.agent, destination, current_time=self.currentdatetime
        )

        # ------------------------------------------------------------------
        # Activity completion  advance clock by duration
        # ------------------------------------------------------------------
        dur_hours = float(self.agent.activity_duration_hours or 0.0)
        activity_end = arrival_time + timedelta(hours=dur_hours)
        self._last_act_end = activity_end

        self.agent.fast_forward(arrival_time, activity_end)
        self.currentdatetime = activity_end
        self.agent.decision_count += 1

        # ------------------------------------------------------------------
        # Day-done check: HOME + (all needs satisfied OR slots exhausted)
        #  fire end-of-day bonus and skip clock to 5AM next day
        # ------------------------------------------------------------------
        daily_bonus = 0.0
        day_done = (
            destination == "HOME"
            and (
                self.agent.all_needs_satisfied()
                or self.agent.slots_remaining <= 0
            )
        )

        # Also trigger on natural midnight crossing
        current_day_idx = int(
            (self.currentdatetime - self.episodestarttime).total_seconds() // 86400
        )
        natural_midnight = current_day_idx > self.agent.current_day

        if day_done or natural_midnight:
            # Fire end-of-day bonus BEFORE resetting needs
            daily_bonus = self.rewardcalc.compute_daily_reward(self.agent)

            if day_done and not natural_midnight:
                # Skip clock to 5AM next day
                next_day = (self.currentdatetime + timedelta(days=1)).date()
                self.currentdatetime = datetime(
                    next_day.year, next_day.month, next_day.day, 5, 0
                )

            # Reset agent for new day
            self.agent.reset_for_new_day(self.currentdatetime.weekday())
            # Ensure agent is at home after skip
            if day_done:
                self.agent.current_location    = "HOME"
                self.agent.current_location_id = self.agent.home_location_id
                self.agent.can_leave_time       = self.currentdatetime

        # ------------------------------------------------------------------
        # Episode termination
        # ------------------------------------------------------------------
        terminated = self.currentdatetime >= self.episodeendtime
        truncated  = False

        # Weekly reward is always 0 in V2 but kept for API compat
        weekly_reward = self.rewardcalc.compute_weekly_reward(self.agent) if terminated else 0.0

        total_reward = float(immediate_reward + daily_bonus + weekly_reward + midnight_bonus)

        # ------------------------------------------------------------------
        # Observation
        # ------------------------------------------------------------------
        obs  = self.obsbuilder.build_observation(self.agent, self.currentdatetime)
        info = self._build_info(
            destination=destination,
            destinationid=int(destinationid),
            subactivitytype=subactivitytype,
            dur_hours=dur_hours,
            traveltimemin=float(traveltimemin),
            transportmode=transportmode,
            distancekm=float(distancekm),
            immediate_reward=immediate_reward,
            daily_bonus=daily_bonus,
            weekly_reward=weekly_reward,
            action_valid=action_valid,
        )

        if terminated:
            info["episodesummary"] = self._episode_summary()

        return obs, total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # ACTION MASKING
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Required by sb3-contrib MaskablePPO."""
        return self.get_valid_actions_mask()

    def get_valid_actions_mask(self) -> np.ndarray:
        """
        Returns binary mask of shape (NUM_ACTIONS,).

        Masking layers (applied in order; any False  action masked):
          1. Operating hours   macro not open right now
          2. Cooldown          macro still in post-activity cooldown
          3. Need saturation   daily need count already met for this macro
          4. Slot exhaustion   no budget slots remain (HOME always valid)

        HOME is never masked.
        """
        mask     = np.ones(int(config.NUM_ACTIONS), dtype=np.float32)
        hour     = int(self.currentdatetime.hour)
        dow      = int(self.currentdatetime.weekday())
        slots    = int(self.agent.slots_remaining)

        for a in range(int(config.NUM_ACTIONS)):
            macro = config.action_to_location_type(a)

            if macro == "HOME":
                continue  # HOME never masked

            # 1. Operating hours
            if not bool(config.is_location_open(macro, hour, dow)):
                mask[a] = 0.0
                continue

            # 2. Cooldown (FOOD, ERRAND, LEISURE only)
            if self.agent.is_on_cooldown(macro, self.currentdatetime):
                mask[a] = 0.0
                continue

            # 3. Need saturation  macro done >= required count today
            need_key = _MACRO_TO_NEED.get(macro)
            if need_key is not None:
                required     = int(self.agent.daily_needs.get(need_key, 0))
                accomplished = int(self.agent.needs_accomplished.get(need_key, 0))
                done_count   = int(self.agent.daily_macro_counts.get(macro, 0))

                # WORK: mask as soon as daily need is satisfied -- no extra visit
                # allowed (going to work twice in one day is not realistic).
                # FOOD/ERRAND/LEISURE: allow exactly one discretionary extra trip
                # beyond the required count (+1 slot feature).
                if macro == "WORK":
                    if accomplished >= required and required > 0:
                        mask[a] = 0.0
                        continue
                else:
                    # Mask if: need is fully met AND done more times than required
                    if accomplished >= required and done_count > required:
                        mask[a] = 0.0
                        continue

            # 4. Slot exhaustion  no slots left, only HOME allowed
            if slots <= 0:
                mask[a] = 0.0
                continue

        return mask

    # Alias used by some callers
    def getvalidactionsmask(self) -> np.ndarray:
        return self.get_valid_actions_mask()

    # ------------------------------------------------------------------
    # INFO / SUMMARY HELPERS
    # ------------------------------------------------------------------

    def _base_info(self) -> Dict[str, Any]:
        return {
            "datetime":     self.currentdatetime.isoformat(),
            "location":     self.agent.current_location,
            "locationid":   self.agent.current_location_id,
            "decisioncount": self.agent.decision_count,
            "slots_remaining": self.agent.slots_remaining,
            "daily_budget":    self.agent.daily_budget,
        }

    def _build_info(
        self,
        destination: str,
        destinationid: int,
        subactivitytype: Optional[str],
        dur_hours: float,
        traveltimemin: float,
        transportmode: str,
        distancekm: float,
        immediate_reward: float,
        daily_bonus: float,
        weekly_reward: float,
        action_valid: bool,
    ) -> Dict[str, Any]:
        return {
            # Timing
            "datetime":       self.currentdatetime.isoformat(),
            "dep_time":       self._last_dep_time.isoformat() if self._last_dep_time else None,
            "arr_time":       self._last_arr_time.isoformat() if self._last_arr_time else None,
            "act_end_time":   self._last_act_end.isoformat()  if self._last_act_end  else None,
            # Activity
            "location":       self.agent.current_location,
            "locationid":     destinationid,
            "location_id":    destinationid,
            "macroactivity":  destination,
            "macro_activity": destination,
            "subactivity":    subactivitytype,
            "sub_activity":   subactivitytype,
            "duration":       dur_hours * 60.0,          # minutes
            "duration_hours": dur_hours,
            # Travel
            "traveltime":     traveltimemin,
            "travel_time":    traveltimemin,
            "transportmode":  transportmode,
            "transport_mode": transportmode,
            "distancekm":     distancekm,
            "distance_km":    distancekm,
            # Counts
            "decisioncount":  int(self.agent.decision_count),
            "decision_count": int(self.agent.decision_count),
            # Budget
            "slots_remaining":  int(self.agent.slots_remaining),
            "daily_budget":     int(self.agent.daily_budget),
            # Reward breakdown
            "immediate_reward":  float(immediate_reward),
            "daily_bonus":       float(daily_bonus),
            "weekly_reward":     float(weekly_reward),
            # Legacy keys (kept so sim.py / logging don't break)
            "tcreward":          0.0,
            "tc_reward":         0.0,
            "immediate_need_reward": float(immediate_reward),
            "nsreward":          float(daily_bonus),
            "ns_reward":         float(daily_bonus),
            "wdpenalty":         0.0,
            "wd_penalty":        0.0,
            # Diagnostics
            "action_valid":      action_valid,
        }

    def _episode_summary(self) -> Dict[str, Any]:
        breakdown = self.rewardcalc.get_reward_breakdown()
        return {
            "totalreward":              breakdown["total"],
            "rewardbreakdown":          breakdown,
            "totaldecisions":           int(self.agent.decision_count),
            "finalweeklydistribution":  self.agent.get_weekly_distribution(),
            "finalneedssatisfaction":   float(self.agent.get_needs_satisfaction_score()),
            "trajectories":             list(self.agent.trajectories)
                                        if self.agent.trajectories is not None else [],
            "totaltraveltimeminutes":   float(self.agent.time_traveling_minutes),
            "weekly_non_home_activities": int(self.agent.weekly_non_home_activities),
        }

    # Keep old name for callers
    def getepisodesummary(self) -> Dict[str, Any]:
        return self._episode_summary()

    def buildobs(self, agent: Any, t: datetime) -> np.ndarray:
        return self.obsbuilder.build_observation(agent, t)

    # ------------------------------------------------------------------
    # RENDER / CLOSE
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return
        needs_str = " | ".join(
            f"{k}:{self.agent.needs_accomplished.get(k,0)}/{self.agent.daily_needs.get(k,0)}"
            for k in self.agent.NEEDS_CANONICAL
        )
        print("-" * 70)
        print(f"  {self.currentdatetime.strftime('%A %Y-%m-%d %H:%M')}")
        print(f"  Location : {self.agent.current_location}  (id={self.agent.current_location_id})")
        print(f"  Needs    : {needs_str}")
        print(f"  Slots    : {self.agent.slots_remaining}/{self.agent.daily_budget}")
        print(f"  Decisions: {self.agent.decision_count}")
        print("-" * 70)

    def close(self) -> None:
        pass