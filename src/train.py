"""
TRAINING SCRIPT -- V2
====================
Train and evaluate the MaskablePPO mobility agent.

Usage:
  python train.py --train [--timesteps N] [--n_envs N] [--lr F] [--seed N]
  python train.py --eval path/to/model.zip [--seed N]
"""

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_src  = _here if _os.path.basename(_here) == 'src' else _os.path.dirname(_here)
_root = _os.path.dirname(_src)
for _p in [_root, _src] + [_os.path.join(_src, d) for d in ('simulation', 'spatial', 'analysis')]:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _here, _src, _root, _p


import os
import sys
import numpy as np
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from environment import MosaicEnv


# =============================================================================
# TENSORBOARD CUSTOM METRICS CALLBACK
# =============================================================================

class MobilityMetricsCallback(BaseCallback):
    """
    Logs mobility-specific metrics to TensorBoard at the end of each episode.

    Metrics logged under custom/:
      needs_satisfaction      -- NS score [0,1] from agent
      bonus_reward_rate       -- fraction of days all needs were met
      unmet_penalty_mean      -- mean daily unmet-need penalty
      immediate_reward_mean   -- mean immediate reward per step
      pct_home / pct_work / pct_travel / pct_food / pct_errand / pct_leisure
      non_home_activities     -- daily out-of-home count (slot budget check)
      slots_remaining_eod     -- slots left at episode end (should trend to 0)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards:       list = []
        self._ns_scores:             list = []
        self._bonus_rates:           list = []
        self._unmet_penalties:       list = []
        self._immediate_rewards:     list = []
        self._non_home_acts:         list = []
        self._slots_remaining:       list = []
        self._daily_activities:      list = []
        self._pct: dict              = {k: [] for k in ["HOME","WORK","TRAVEL","FOOD","ERRAND","LEISURE"]}

    def _on_step(self) -> bool:
        # SB3 provides dones and infos for every parallel env each step
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue

            summary = info.get("episodesummary", {})
            if not summary:
                continue

            # Needs satisfaction
            ns = float(summary.get("finalneedssatisfaction", 0.0))
            self._ns_scores.append(ns)

            # Reward breakdown
            bd = summary.get("rewardbreakdown", {})
            immediate = float(bd.get("immediate_reward",  0.0))
            bonus     = float(bd.get("bonus_reward",      0.0))
            self._immediate_rewards.append(immediate)

            # Bonus rate: bonus > 0 means at least one full day was completed
            # Use ratio: bonus / (ALL_NEEDS_BONUS * episode_days) as a rate proxy
            try:
                import config as _cfg
                max_bonus = float(_cfg.ALL_NEEDS_BONUS) * float(_cfg.EPISODE_DAYS)
                self._bonus_rates.append(bonus / max_bonus if max_bonus > 0 else 0.0)
                unmet_pen = float(bd.get("immediate_need_reward", 0.0))  # legacy key
                # Better: derive from bonus shortfall
                self._unmet_penalties.append(min(0.0, bonus - max_bonus))
            except Exception:
                self._bonus_rates.append(0.0)
                self._unmet_penalties.append(0.0)

            # Activity distribution
            wd = summary.get("finalweeklydistribution", {})
            for k in self._pct:
                self._pct[k].append(float(wd.get(k, 0.0)) * 100.0)

            # Non-home activities and slots
            self._non_home_acts.append(
                float(summary.get("weekly_non_home_activities", 0))
            )

            # Average daily activities = total non-home acts / episode days
            try:
                import config as _cfg
                ep_days = float(_cfg.EPISODE_DAYS)
                non_home = float(summary.get("weekly_non_home_activities", 0))
                self._daily_activities.append(non_home / ep_days if ep_days > 0 else 0.0)
            except Exception:
                self._daily_activities.append(0.0)
            self._slots_remaining.append(
                float(summary.get("slots_remaining", 0))
            )

        # Log aggregated metrics every 20 completed episodes
        n = len(self._ns_scores)
        if n > 0 and n % 20 == 0:
            self._flush_to_tensorboard()

        return True

    def _flush_to_tensorboard(self) -> None:
        step = self.num_timesteps

        def mean(lst):
            return float(sum(lst) / len(lst)) if lst else 0.0

        self.logger.record("custom/needs_satisfaction",    mean(self._ns_scores))
        self.logger.record("custom/bonus_reward_rate",     mean(self._bonus_rates))
        self.logger.record("custom/unmet_penalty_mean",    mean(self._unmet_penalties))
        self.logger.record("custom/immediate_reward_mean", mean(self._immediate_rewards))
        self.logger.record("custom/non_home_activities",   mean(self._non_home_acts))
        self.logger.record("custom/slots_remaining_eod",   mean(self._slots_remaining))
        self.logger.record("custom/avg_activities_per_day", mean(self._daily_activities))

        for k, vals in self._pct.items():
            self.logger.record(f"custom/pct_{k.lower()}", mean(vals))

        self.logger.dump(step)

        # Clear buffers
        self._ns_scores.clear()
        self._bonus_rates.clear()
        self._unmet_penalties.clear()
        self._immediate_rewards.clear()
        self._non_home_acts.clear()
        self._slots_remaining.clear()
        self._daily_activities.clear()
        for k in self._pct:
            self._pct[k].clear()



def make_env(seed: int):
    def _init():
        env = MosaicEnv(seed=seed)
        env = Monitor(env)
        return env
    return _init


def train(
    total_timesteps: int   = 1_000_000,
    n_envs:          int   = 8,
    learning_rate:   float = 3e-4,
    batch_size:      int   = 128,
    n_steps:         int   = 512,
    save_path:       str   = None,
    log_dir:         str   = None,
    seed:            int   = 42,
):
    import sys as _s; _root = _s.path[0] if _s.path else '.'
    if save_path is None:
        save_path = os.path.join(_root, 'output', 'model', 'model_0.zip')
    if log_dir is None:
        log_dir = os.path.join(_root, 'output', 'logs')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run_name  = os.path.basename(save_path).replace('.zip', '') if save_path else 'model'

    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    print("=" * 70)
    print(f"TRAINING: {run_name}")
    print("=" * 70)
    print(f"Total timesteps : {total_timesteps:,}")
    print(f"Parallel envs   : {n_envs}")
    print(f"Learning rate   : {learning_rate}")
    print(f"Batch size      : {batch_size}")
    print(f"Steps per update: {n_steps}")
    print(f"Device          : {device_name}")
    print("=" * 70)

    # Vectorised environments -- SubprocVecEnv for true multi-CPU training.
    # Switch to DummyVecEnv for easier debugging (single process).
    env = SubprocVecEnv([make_env(seed + i) for i in range(n_envs)])

    metrics_callback = MobilityMetricsCallback(verbose=0)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        verbose=1,
        seed=seed,
        device="auto",
    )

    print(f"\nUpdates per rollout : {n_steps * n_envs // batch_size}")
    print(f"Total updates       : "
          f"{total_timesteps // (n_steps * n_envs) * (n_steps * n_envs // batch_size)}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_callback],
        progress_bar=True,
    )

    model.save(save_path.replace('.zip', ''))

    print("\n" + "=" * 70)
    print(f"Training complete! Model saved to {save_path}")
    print("=" * 70)

    return model, env


def evaluate(model_path: str, n_episodes: int = 10, seed: int = 42):
    print("=" * 70)
    print(f"EVALUATING: {model_path}")
    print("=" * 70)

    env   = MosaicEnv(seed=seed)
    model = MaskablePPO.load(model_path)

    episode_rewards: list = []
    episode_lengths: list = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        length = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

        print(f"Episode {ep+1:2d}/{n_episodes}: "
              f"reward={total_reward:.2f}  length={length}")

        summary = info.get("episodesummary", {})
        if summary:
            wd   = summary.get("finalweeklydistribution", {})
            home_pct = wd.get("HOME", 0) * 100
            work_pct = wd.get("WORK", 0) * 100
            ns   = summary.get("finalneedssatisfaction", 0.0)
            tt   = summary.get("totaltraveltimeminutes", 0.0)
            acts = summary.get("weekly_non_home_activities", 0)
            print(f"           NS={ns:.3f}  home={home_pct:.1f}%  "
                  f"work={work_pct:.1f}%  travel={tt:.0f}min  "
                  f"non-home acts={acts}")

            # Reward breakdown
            bd = summary.get("rewardbreakdown", {})
            if bd:
                print(f"           immediate={bd.get('immediate_reward', 0):.2f}  "
                      f"bonus={bd.get('bonus_reward', 0):.2f}")

    print("\n" + "=" * 70)
    print(f"Mean reward : {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length : {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print("=" * 70)

    return episode_rewards, episode_lengths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train / evaluate PPO V2 mobility agent"
    )
    parser.add_argument("--train",     action="store_true",     help="Train new model")
    parser.add_argument("--eval",      type=str, default=None,  help="Evaluate model at path")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n_envs",    type=int, default=4)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    if args.train:
        model, env = train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            seed=args.seed,
        )
        print("\nTo evaluate:")
        print("  python train.py --eval path/to/model.zip")

    elif args.eval:
        evaluate(args.eval, n_episodes=10, seed=args.seed)

    else:
        print("Usage:")
        print("  Train: python train.py --train [--timesteps N] [--n_envs N] [--lr F] [--seed N]")
        print("  Eval:  python train.py --eval path/to/model.zip [--seed N]")