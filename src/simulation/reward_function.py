from __future__ import annotations

"""
REWARD CALCULATOR -- V2
======================

Simplified reward function.  Two components only:

  1. Immediate activity reward  -- fired on each activity completion.
     Value comes from PHASE_ACTIVITY_REWARD[phase][macro] in {-1, -0.5, 0, 0.5, 1}.

  2. End-of-day completeness bonus  -- fired once per day when all required
     needs have been satisfied.  Value = ALL_NEEDS_BONUS (default 2.0).

Two reward components:
  1. Immediate activity reward   -- phase-aware, fired on each activity completion
  2. End-of-day completeness bonus -- fired once per day when all needs are met
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p



from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

import config
from config import (
    PHASE_ACTIVITY_REWARD,
    ALL_NEEDS_BONUS,
    get_phase,
    is_weekend,
)


class RewardFunction:
    """
    Phase-aware immediate reward calculator (V2).

    Public API (kept compatible with environment call sites):
      compute_immediate_need_reward(agent, activity_type)   -> float
      compute_daily_reward(agent)                           -> float
      compute_weekly_reward(agent)                          -> float  (always 0)
      get_total_reward()                                    -> float
      get_reward_breakdown()                                -> dict
      reset()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_immediate_reward: float = 0.0
        self.total_bonus_reward:     float = 0.0

        # Logging lists -- kept lightweight; only populated when not in fast mode
        self._reward_log: List[Dict] = []

    # =========================================================================
    # IMMEDIATE REWARD  (called in env.step after activity completes)
    # =========================================================================

    def compute_immediate_need_reward(
        self, agent, activity_type: str, current_time: Optional[datetime] = None
    ) -> float:
        """
        Return the phase-aware reward for completing *activity_type*.

        Steps:
          1. Determine current phase from clock time.
          2. Look up PHASE_ACTIVITY_REWARD[phase][macro].
          3. If the activity satisfies a need, return the full table value.
          4. If the activity is the discretionary slot (accomplishes > required),
             halve the reward -- the agent is spending a free slot, good but not
             as valuable as satisfying a need.
          5. If the macro has no associated need (unexpected), return 0.

        HOME always returns 0 -- neutral action.
        """
        if activity_type == "HOME":
            # Sleep reward: arriving home during night hours is genuinely valuable.
            # Daytime HOME is neutral -- brief rest between activities.
            if current_time is not None:
                hour = current_time.hour + current_time.minute / 60.0
            else:
                t = getattr(agent, "arrival_time", None)
                hour = (t.hour + t.minute / 60.0) if t is not None else 12.0

            if hour >= 21.0 or hour < 6.0:
                # Prime sleep window -- reward going home to sleep
                sleep_reward = 0.5
                self.total_immediate_reward += sleep_reward
                return sleep_reward
            return 0.0

        # --- Determine phase ---
        if current_time is not None:
            hour = current_time.hour + current_time.minute / 60.0
            wknd = bool(is_weekend(current_time.weekday()))
        else:
            # Fallback: use agent arrival time if available
            t = getattr(agent, "arrival_time", None)
            if t is not None:
                hour = t.hour + t.minute / 60.0
                wknd = bool(is_weekend(t.weekday()))
            else:
                hour, wknd = 12.0, False   # neutral default

        phase = get_phase(hour, wknd)
        base_reward = float(
            PHASE_ACTIVITY_REWARD.get(phase, {}).get(activity_type, 0.0)
        )

        # --- Need accounting ---
        need_key = {
            "WORK":    "wenttowork",
            "FOOD":    "eatoutside",
            "ERRAND":  "runerrand",
            "LEISURE": "leisure",
        }.get(activity_type)

        if need_key is None:
            return 0.0

        required     = int(agent.daily_needs.get(need_key, 0))
        accomplished = int(agent.needs_accomplished.get(need_key, 0))

        if required == 0:
            # Need not required today -- light penalty to discourage unnecessary
            # activity, but still phase-aware so timing signal is preserved.
            # -0.3 is intentionally small: we don't want to suppress all
            # discretionary behaviour, just make unneeded activities slightly costly.
            reward = base_reward - 0.3
            reason = "discretionary_unnecessary"
        elif accomplished <= required:
            # This completion satisfies (part of) a need -- full phase reward
            reward = base_reward
            reason = "need_satisfied"
        else:
            # accomplished > required: need already met, extra use of this macro.
            # Still full phase reward -- the mask already limits over-use structurally.
            reward = base_reward
            reason = "discretionary_extra"

        reward = float(reward)
        self.total_immediate_reward += reward

        if not getattr(config, "FAST_TRAINING_MODE", True):
            self._reward_log.append({
                "activity":    activity_type,
                "phase":       phase,
                "phase_name":  config.PHASE_NAMES[phase],
                "base_reward": base_reward,
                "reward":      reward,
                "reason":      reason,
                "required":    required,
                "accomplished": accomplished,
            })

        return reward

    # =========================================================================
    # END-OF-DAY BONUS  (called in env.step at day boundary)
    # =========================================================================

    UNMET_NEED_PENALTY: float = -1.0  # Applied per unmet need at day boundary

    def compute_daily_reward(self, agent) -> float:
        """
        End-of-day reward:
          +ALL_NEEDS_BONUS  if every required need was satisfied today.
          UNMET_NEED_PENALTY * n_unmet  if any needs were missed.

        The penalty gives a gradient even on failed days, so the agent
        learns which needs to prioritise rather than giving up after
        missing one.  Bonus and penalty are mutually exclusive.

        Called once per day boundary, before reset_for_new_day().
        """
        if agent.all_needs_satisfied():
            bonus = float(ALL_NEEDS_BONUS)
            self.total_bonus_reward += bonus
            return bonus

        # Count unmet needs
        n_unmet = 0
        for k in agent.NEEDS_CANONICAL:
            required    = int(agent.daily_needs.get(k, 0))
            accomplished = int(agent.needs_accomplished.get(k, 0))
            if accomplished < required:
                n_unmet += 1

        penalty = float(self.UNMET_NEED_PENALTY) * float(n_unmet)
        self.total_bonus_reward += penalty  # tracked in same bucket for simplicity
        return penalty

    # =========================================================================
    # WEEKLY REWARD -- kept for API compatibility, always returns 0
    # =========================================================================

    def compute_weekly_reward(self, agent) -> float:
        """
        Weekly distribution is an evaluation metric, not a training signal.
        Retained for API compatibility with env.step().
        """
        return 0.0

    # =========================================================================
    # TOTALS / BREAKDOWN
    # =========================================================================

    def get_total_reward(self) -> float:
        return float(self.total_immediate_reward + self.total_bonus_reward)

    def get_reward_breakdown(self) -> Dict:
        return {
            "total":             self.get_total_reward(),
            "immediate_reward":  float(self.total_immediate_reward),
            "bonus_reward":      float(self.total_bonus_reward),
            # Legacy keys kept so downstream logging code doesn't break
            "nsreward":          float(self.total_bonus_reward),
            "tcreward":          0.0,
            "wdpenalty":         0.0,
            "nhpenalty":         0.0,
            "immediate_need_reward": float(self.total_immediate_reward),
            "unmet_penalty_per_need": float(self.UNMET_NEED_PENALTY),
            "num_reward_events": len(self._reward_log),
            "reward_log":        list(self._reward_log),
        }

    # Kept for backward compat with any logging that called this
    def compute_tc_for_activity(self, *args, **kwargs) -> float:
        return 0.0