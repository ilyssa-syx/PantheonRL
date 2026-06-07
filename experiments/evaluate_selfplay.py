"""
Deterministic self-play evaluation for trained PPO/A2C/DQN Overcooked models.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import Any, Dict, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from pantheonrl.common.agents import StaticModelAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
import overcookedgym  # noqa: F401  Registers OvercookedMultiEnv-v0.


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen PPO/A2C/DQN self-play models deterministically."
        )
    )
    parser.add_argument("--algo", required=True, choices=sorted(ALGOS))
    parser.add_argument("--layout", required=True, choices=LAYOUT_LIST)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing ego_model.zip and partner_model.zip.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device used to load the frozen models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to run-dir/evaluation.json.",
    )
    return parser.parse_args()


def run_episode(model: BaseAlgorithm, env: gym.Env) -> float:
    obs = env.reset()
    done = False
    episode_return = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_return += float(reward)
    return episode_return


def load_and_validate_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = args.run_dir / "config.json"
    status_path = args.run_dir / "training_status.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    if not status_path.is_file():
        raise FileNotFoundError(f"Missing training status: {status_path}")

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    with status_path.open(encoding="utf-8") as f:
        status = json.load(f)

    if status.get("status") != "completed":
        raise ValueError(
            f"Run is not marked completed: {status.get('status')!r}")
    if config.get("algo") != args.algo:
        raise ValueError(
            f"Algorithm mismatch: run uses {config.get('algo')!r}, "
            f"but --algo is {args.algo!r}")
    if config.get("layout") != args.layout:
        raise ValueError(
            f"Layout mismatch: run uses {config.get('layout')!r}, "
            f"but --layout is {args.layout!r}")
    return config


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    config = load_and_validate_config(args)
    algo_cls = ALGOS[args.algo]
    output = args.output or (args.run_dir / "evaluation.json")

    env = gym.make("OvercookedMultiEnv-v0", layout_name=args.layout)
    try:
        partner_env = env.getDummyEnv(1)
        partner_model = algo_cls.load(
            args.run_dir / "partner_model", env=partner_env, device=args.device)
        env.add_partner_agent(
            StaticModelAgent(partner_model, deterministic=True)
        )
        ego_model = algo_cls.load(
            args.run_dir / "ego_model", env=env, device=args.device)
        returns = [run_episode(ego_model, env) for _ in range(args.episodes)]
    finally:
        env.close()

    results = {
        "algo": args.algo,
        "layout": args.layout,
        "seed": config.get("seed"),
        "ego_seed": config.get("ego_seed"),
        "partner_seed": config.get("partner_seed"),
        "run_dir": str(args.run_dir),
        "training_model_kwargs": config.get("model_kwargs"),
        "training_effective_hyperparameters": config.get(
            "effective_hyperparameters"),
        "training_requested_timesteps": config.get(
            "requested_timesteps", config.get("timesteps")),
        "training_actual_ego_timesteps": config.get("actual_ego_timesteps"),
        "training_actual_partner_timesteps": config.get(
            "actual_partner_timesteps"),
        "episodes": args.episodes,
        "deterministic": True,
        "device": str(ego_model.device),
        "episode_returns": returns,
        "mean_return": mean(returns),
        "std_return": pstdev(returns),
        "median_return": median(returns),
        "min_return": min(returns),
        "max_return": max(returns),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return"])
        for episode, episode_return in enumerate(returns):
            writer.writerow([episode, episode_return])


if __name__ == "__main__":
    main()
