"""
Train one DQN Ego against a frozen deterministic PPO Partner.

This is a diagnostic experiment for separating DQN learning failure from the
non-stationarity of independent self-play. Periodic deterministic evaluations
and DQN checkpoints expose whether the greedy Ego policy ever learns.
"""

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed

from pantheonrl.common.agents import StaticModelAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
import overcookedgym  # noqa: F401  Registers OvercookedMultiEnv-v0.

from train_dqn_selfplay import (
    OPTIONAL_MODEL_ARGS,
    build_model_kwargs,
    effective_hyperparameters,
    replay_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DQN Ego against a frozen deterministic PPO Partner."
    )
    parser.add_argument("--layout", required=True, choices=LAYOUT_LIST)
    parser.add_argument(
        "--ppo-run-dir",
        type=Path,
        required=True,
        help="Completed PPO run containing partner_model.zip and config.json.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fixed_ppo_partner"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])

    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--target-update-interval", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.eval_freq <= 0 or args.eval_episodes <= 0:
        raise ValueError("--eval-freq and --eval-episodes must be positive")
    if not 0 < args.exploration_fraction <= 1:
        raise ValueError("--exploration-fraction must be in (0, 1]")
    for name in (
        "buffer_size",
        "learning_starts",
        "batch_size",
        "train_freq",
        "target_update_interval",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.gradient_steps is not None and args.gradient_steps < -1:
        raise ValueError("--gradient-steps must be at least -1")

    config_path = args.ppo_run_dir / "config.json"
    partner_path = args.ppo_run_dir / "partner_model.zip"
    if not config_path.is_file() or not partner_path.is_file():
        raise FileNotFoundError(
            "PPO run must contain config.json and partner_model.zip: "
            f"{args.ppo_run_dir}"
        )
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    if config.get("algo") != "ppo":
        raise ValueError(f"Expected PPO source run, got {config.get('algo')!r}")
    if config.get("layout") != args.layout:
        raise ValueError(
            f"Layout mismatch: source uses {config.get('layout')!r}, "
            f"but --layout is {args.layout!r}"
        )
    return config


def make_run_dir(args: argparse.Namespace) -> Path:
    source_name = args.ppo_run_dir.name.replace("/", "_")
    run_name = (
        f"steps_{args.timesteps}__exploration_fraction_"
        f"{args.exploration_fraction}__fixed_ppo_{source_name}"
    )
    return args.output_dir / args.layout / f"seed_{args.seed}" / run_name


def ppo_custom_objects(env: gym.Env) -> Dict[str, Any]:
    """Replace non-weight PPO metadata saved with pickle protocol 5."""
    return {
        "policy_class": PPO.policy_aliases["MlpPolicy"],
        "ep_info_buffer": deque(maxlen=100),
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "rollout_buffer_class": RolloutBuffer,
        "clip_range": get_schedule_fn(0.2),
        "lr_schedule": get_schedule_fn(0.0003),
    }


def load_frozen_partner(path: Path, env: gym.Env, device: str) -> PPO:
    return PPO.load(
        path,
        env=env.getDummyEnv(1),
        device=device,
        custom_objects=ppo_custom_objects(env),
    )


def make_env(
    layout: str,
    partner_path: Path,
    device: str,
    env_kwargs: Dict[str, Any],
) -> gym.Env:
    env = gym.make("OvercookedMultiEnv-v0", layout_name=layout, **env_kwargs)
    partner = load_frozen_partner(partner_path, env, device)
    env.add_partner_agent(StaticModelAgent(partner, deterministic=True))
    return env


def run_deterministic_evaluation(
    model: DQN, env: gym.Env, episodes: int
) -> Dict[str, Any]:
    results: List[Dict[str, float]] = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_return += float(reward)

        base_env = env.unwrapped.base_env
        sparse_return = float(base_env.cumulative_sparse_rewards)
        shaped_return = float(base_env.cumulative_shaped_rewards)
        results.append(
            {
                "total_return": total_return,
                "sparse_return": sparse_return,
                "shaped_return": shaped_return,
                "deliveries": (
                    sparse_return / float(env.unwrapped.mdp.delivery_reward)
                ),
            }
        )

    def values(key: str) -> List[float]:
        return [result[key] for result in results]

    return {
        "episodes": episodes,
        "mean_total_return": sum(values("total_return")) / episodes,
        "max_total_return": max(values("total_return")),
        "mean_sparse_return": sum(values("sparse_return")) / episodes,
        "max_sparse_return": max(values("sparse_return")),
        "mean_shaped_return": sum(values("shaped_return")) / episodes,
        "mean_deliveries": sum(values("deliveries")) / episodes,
        "episode_results": results,
    }


class DiagnosticEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        run_dir: Path,
        eval_freq: int,
        eval_episodes: int,
    ):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.history: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        result = run_deterministic_evaluation(
            self.model, self.eval_env, self.eval_episodes
        )
        result["timesteps"] = self.num_timesteps
        result["exploration_rate"] = float(self.model.exploration_rate)
        self.history.append(result)

        checkpoint = self.run_dir / "checkpoints" / (
            f"ego_model_{self.num_timesteps:07d}_steps"
        )
        self.model.save(checkpoint)
        write_evaluation_history(self.run_dir, self.history)
        print(
            "Deterministic evaluation at {} steps: total={:.3f}, "
            "sparse={:.3f}, deliveries={:.3f}, epsilon={:.3f}".format(
                self.num_timesteps,
                result["mean_total_return"],
                result["mean_sparse_return"],
                result["mean_deliveries"],
                result["exploration_rate"],
            ),
            flush=True,
        )
        return True

    def _on_training_end(self) -> None:
        if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
            result = run_deterministic_evaluation(
                self.model, self.eval_env, self.eval_episodes
            )
            result["timesteps"] = self.num_timesteps
            result["exploration_rate"] = float(self.model.exploration_rate)
            self.history.append(result)
            write_evaluation_history(self.run_dir, self.history)


def write_evaluation_history(
    run_dir: Path, history: List[Dict[str, Any]]
) -> None:
    with (run_dir / "checkpoint_evaluations.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(history, f, indent=2, sort_keys=True)
        f.write("\n")
    with (run_dir / "checkpoint_evaluations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fields = [
            "timesteps",
            "exploration_rate",
            "episodes",
            "mean_total_return",
            "max_total_return",
            "mean_sparse_return",
            "max_sparse_return",
            "mean_shaped_return",
            "mean_deliveries",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in history:
            writer.writerow({field: result[field] for field in fields})


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    source_config = validate_args(args)
    run_dir = make_run_dir(args)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()

    partner_path = args.ppo_run_dir / "partner_model.zip"
    env_kwargs = source_config.get("env_kwargs", {})
    shutil.copy2(partner_path, run_dir / "fixed_ppo_partner_model.zip")
    set_random_seed(args.seed)
    train_env = make_env(args.layout, partner_path, args.device, env_kwargs)
    eval_env = make_env(args.layout, partner_path, args.device, env_kwargs)
    model_kwargs = build_model_kwargs(args)
    model = DQN(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        tensorboard_log=str(run_dir / "logs"),
        **model_kwargs,
    )
    callback = DiagnosticEvalCallback(
        eval_env, run_dir, args.eval_freq, args.eval_episodes
    )
    initial_evaluation = run_deterministic_evaluation(
        model, eval_env, args.eval_episodes
    )
    initial_evaluation["timesteps"] = 0
    initial_evaluation["exploration_rate"] = float(model.exploration_rate)
    callback.history.append(initial_evaluation)
    write_evaluation_history(run_dir, callback.history)

    started = time.monotonic()
    status: Dict[str, Any] = {"status": "running"}
    write_json(run_dir / "training_status.json", status)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(run_dir / "ego_model")
        status = {
            "status": "completed",
            "actual_timesteps": model.num_timesteps,
            "updates": model._n_updates,
            "replay_transitions": replay_size(model),
            "wall_clock_seconds": time.monotonic() - started,
        }
    except Exception as exc:
        status = {
            "status": "failed",
            "actual_timesteps": model.num_timesteps,
            "wall_clock_seconds": time.monotonic() - started,
            "error": str(exc),
        }
        raise
    finally:
        write_json(run_dir / "training_status.json", status)
        train_env.close()
        eval_env.close()

    write_json(
        run_dir / "config.json",
        {
            "algo": "dqn",
            "layout": args.layout,
            "seed": args.seed,
            "self_play_type": "fixed_ppo_partner_diagnostic",
            "requested_timesteps": args.timesteps,
            "actual_timesteps": model.num_timesteps,
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "ppo_source_run": str(args.ppo_run_dir),
            "ppo_source_config": source_config,
            "fixed_partner_deterministic": True,
            "env_kwargs": env_kwargs,
            "model_kwargs": model_kwargs,
            "effective_hyperparameters": effective_hyperparameters(model),
            "output_dir": str(run_dir),
        },
    )
    print(f"Training complete: {run_dir}")


if __name__ == "__main__":
    main()
