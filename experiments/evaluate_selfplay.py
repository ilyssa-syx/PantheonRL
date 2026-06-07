"""
Deterministic self-play evaluation for trained PPO/A2C Overcooked models.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import Dict, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from pantheonrl.common.agents import StaticModelAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
import overcookedgym  # noqa: F401  Registers OvercookedMultiEnv-v0.


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen PPO/A2C self-play models deterministically."
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


def main() -> None:
    args = parse_args()
    algo_cls = ALGOS[args.algo]
    output = args.output or (args.run_dir / "evaluation.json")

    env = gym.make("OvercookedMultiEnv-v0", layout_name=args.layout)
    partner_env = env.getDummyEnv(1)

    partner_model = algo_cls.load(args.run_dir / "partner_model", env=partner_env)
    env.add_partner_agent(
        StaticModelAgent(partner_model, deterministic=True)
    )
    ego_model = algo_cls.load(args.run_dir / "ego_model", env=env)

    returns = [run_episode(ego_model, env) for _ in range(args.episodes)]
    results = {
        "algo": args.algo,
        "layout": args.layout,
        "run_dir": str(args.run_dir),
        "episodes": args.episodes,
        "deterministic": True,
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
