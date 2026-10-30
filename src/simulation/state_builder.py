from __future__ import annotations

"""
OBSERVATION BUILDER -- V2
========================
Builds the 25-dimensional observation vector for the RL agent.

Dimension breakdown
-------------------
Block A  Temporal         4 dims   hour_sin, hour_cos, day_of_week, is_weekend
Block B  Location         5 dims   current location one-hot (HOME/WORK/FOOD/ERRAND/LEISURE)
Block C  Time state       2 dims   time_at_location, time_until_can_leave   (norm. by 12h)
Block D  Slot budget      1 dim    slots_remaining  (norm. by max daily budget)
Block E  Phase            1 dim    temporal phase 0-5  (norm. by NUM_PHASES-1)
Block F  Need status      4 dims   per-need: 0=unmet, 0.5=N/A today, 1=met
Block G  Need feasibility 4 dims   per-need binary: 1=still achievable now
Block H  Daily counts     4 dims   WORK/FOOD/ERRAND/LEISURE done today (norm. by 3)
                         --
Total                    25 dims

All values in [0, 1] except hour trig ([-1, 1]) -- PPO handles that fine.
Feasibility flags are time-window aware via config.is_location_open().
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p



import numpy as np
from datetime import datetime

import config
from config import (
    LOCATION_TYPES,
    NUM_LOCATION_TYPES,
    NUM_PHASES,
    OBS_DIM,
    get_phase,
    is_weekend,
)
from agent_state import AgentState

# Fixed canonical order -- must match NEEDS_CANONICAL in AgentState.
NEEDS_ORDER = ["wenttowork", "eatoutside", "runerrand", "leisure"]

# Maximum expected daily budget for normalisation.
# Budget = sum(need_counts) + 1.  Theoretical max: WORK(1)+FOOD(2)+ERRAND(2)+LEISURE(2)+1 = 9
MAX_DAILY_BUDGET = 9


class StateBuilder:
    """Build 25-dim observation for event-based single worker (V2)."""

    def __init__(self) -> None:
        self.obs_dim = OBS_DIM
        self.location_to_idx = {loc: i for i, loc in enumerate(LOCATION_TYPES)}

    def build_observation(
        self,
        agent: AgentState,
        current_time: datetime,
    ) -> np.ndarray:
        """
        Build and return observation array of shape (25,).

        All values normalised to approximately [0, 1] except hour_sin/cos in [-1,1].
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0

        hour = current_time.hour + current_time.minute / 60.0
        dow  = current_time.weekday()
        wknd = bool(is_weekend(dow))

        # =================================================================
        # BLOCK A -- Temporal  (4 dims)
        # =================================================================
        angle = 2.0 * np.pi * hour / 24.0
        obs[idx] = np.sin(angle)   # hour_sin
        idx += 1
        obs[idx] = np.cos(angle)   # hour_cos
        idx += 1
        obs[idx] = dow / 6.0       # day_of_week  0=Mon ... 1=Sun
        idx += 1
        obs[idx] = 1.0 if wknd else 0.0
        idx += 1

        # =================================================================
        # BLOCK B -- Current location one-hot  (5 dims)
        # =================================================================
        loc_idx = self.location_to_idx.get(agent.current_location, 0)
        obs[idx + loc_idx] = 1.0
        idx += NUM_LOCATION_TYPES

        # =================================================================
        # BLOCK C -- Time state  (2 dims)
        # =================================================================
        if agent.arrival_time is not None:
            time_at = (current_time - agent.arrival_time).total_seconds() / 3600.0
            obs[idx] = float(np.clip(time_at / 12.0, 0.0, 1.0))
        # else stays 0.0
        idx += 1

        time_left = agent.get_time_until_can_leave_hours(current_time)
        obs[idx] = float(np.clip(time_left / 12.0, 0.0, 1.0))
        idx += 1

        # =================================================================
        # BLOCK D -- Slot budget  (1 dim)
        # =================================================================
        obs[idx] = float(
            np.clip(agent.slots_remaining / MAX_DAILY_BUDGET, 0.0, 1.0)
        )
        idx += 1

        # =================================================================
        # BLOCK E -- Phase  (1 dim)
        # =================================================================
        phase = get_phase(hour, wknd)
        obs[idx] = float(phase) / float(NUM_PHASES - 1)   # normalised to [0,1]
        idx += 1

        # =================================================================
        # BLOCK F -- Need status  (4 dims, fixed order)
        # =================================================================
        for need in NEEDS_ORDER:
            required    = int(agent.daily_needs.get(need, 0))
            accomplished = int(agent.needs_accomplished.get(need, 0))

            if required == 0:
                obs[idx] = 0.5   # need does not apply today
            else:
                # Progress ratio: 0.0 -> 1.0 as accomplished / required
                obs[idx] = float(np.clip(accomplished / required, 0.0, 1.0))
            idx += 1

        # =================================================================
        # BLOCK G -- Need feasibility  (4 dims, same order as F)
        # =================================================================
        for need in NEEDS_ORDER:
            required = int(agent.daily_needs.get(need, 0))
            done     = int(agent.needs_accomplished.get(need, 0))

            if required == 0 or done >= required:
                # N/A or already satisfied -- feasibility irrelevant
                obs[idx] = 0.5
            else:
                obs[idx] = 1.0 if agent.get_need_feasibility(need, current_time) else 0.0
            idx += 1

        # =================================================================
        # BLOCK H -- Daily macro counts  (4 dims)
        # =================================================================
        for macro in ["WORK", "FOOD", "ERRAND", "LEISURE"]:
            count = int(agent.daily_macro_counts.get(macro, 0))
            obs[idx] = float(np.clip(count / 3.0, 0.0, 1.0))
            idx += 1

        # =================================================================
        # SANITY CHECK (disabled in fast training mode)
        # =================================================================
        if not getattr(config, "FAST_TRAINING_MODE", False):
            assert idx == self.obs_dim, (
                f"ObservationBuilder dimension mismatch: wrote {idx}, expected {self.obs_dim}"
            )
            assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"

        return obs

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def get_observation_info(self, obs: np.ndarray) -> dict:
        """
        Parse observation array into a human-readable dict.
        Useful for debugging; not called during training.
        """
        idx = 0
        info: dict = {}

        # Temporal
        h_sin = float(obs[idx]);  idx += 1
        h_cos = float(obs[idx]);  idx += 1
        hour = np.arctan2(h_sin, h_cos) / (2.0 * np.pi) * 24.0
        if hour < 0:
            hour += 24.0
        info["hour"]        = round(float(hour), 2)
        info["day_of_week"] = round(float(obs[idx]) * 6.0);  idx += 1
        info["is_weekend"]  = bool(obs[idx] > 0.5);           idx += 1

        # Location
        loc_oh = obs[idx: idx + NUM_LOCATION_TYPES]
        info["current_location"] = LOCATION_TYPES[int(np.argmax(loc_oh))]
        idx += NUM_LOCATION_TYPES

        # Time state
        info["time_at_location_h"]    = round(float(obs[idx]) * 12.0, 2);  idx += 1
        info["time_until_can_leave_h"] = round(float(obs[idx]) * 12.0, 2); idx += 1

        # Budget
        info["slots_remaining_norm"] = float(obs[idx]);  idx += 1
        info["slots_remaining_approx"] = round(float(obs[idx - 1]) * MAX_DAILY_BUDGET, 1)

        # Phase
        info["phase_norm"]  = float(obs[idx])
        info["phase_index"] = round(float(obs[idx]) * (NUM_PHASES - 1))
        idx += 1

        # Need status
        info["need_status"] = {}
        for need in NEEDS_ORDER:
            info["need_status"][need] = float(obs[idx]);  idx += 1

        # Need feasibility
        info["need_feasibility"] = {}
        for need in NEEDS_ORDER:
            info["need_feasibility"][need] = float(obs[idx]);  idx += 1

        # Daily macro counts
        info["daily_macro_counts"] = {}
        for macro in ["WORK", "FOOD", "ERRAND", "LEISURE"]:
            info["daily_macro_counts"][macro] = round(float(obs[idx]) * 3.0, 1)
            idx += 1

        return info


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from datetime import datetime

    print("=" * 60)
    print("TESTING StateBuilder V2")
    print("=" * 60)

    builder = StateBuilder()
    agent   = AgentState(randomseed=42)

    t_start = datetime(2024, 1, 1, 0, 0)
    agent.reset_for_new_episode(t_start, home_id=1, work_id=2)
    agent.generate_daily_needs(0)   # Monday

    t_test = datetime(2024, 1, 1, 9, 30)
    agent.start_activity("WORK", 2, t_test, travel_time_minutes=20)

    obs = builder.build_observation(agent, t_test)

    print(f"\nShape : {obs.shape}  (expected ({OBS_DIM},))")
    print(f"Range : [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Finite: {np.all(np.isfinite(obs))}")

    info = builder.get_observation_info(obs)
    print(f"\nHour            : {info['hour']}")
    print(f"Day of week     : {info['day_of_week']:.0f}  (0=Mon)")
    print(f"Is weekend      : {info['is_weekend']}")
    print(f"Location        : {info['current_location']}")
    print(f"Phase index     : {info['phase_index']}  ({config.PHASE_NAMES[info['phase_index']]})")
    print(f"Slots remaining : ~{info['slots_remaining_approx']}")
    print(f"\nNeed status     : {info['need_status']}")
    print(f"Need feasibility: {info['need_feasibility']}")
    print(f"Daily counts    : {info['daily_macro_counts']}")

    assert obs.shape == (OBS_DIM,), "Shape mismatch"
    assert np.all(np.isfinite(obs)),    "Non-finite values"
    print("\n[OK] All assertions passed")