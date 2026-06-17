"""Deterministic evaluation for trained Discrete SAC Overcooked models."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym

from pantheonrl.algos.discrete_sac import DiscreteSAC
from pantheonrl.common.agents import StaticModelAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
import overcookedgym  # noqa: F401  Registers OvercookedMultiEnv-v0.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Discrete SAC self-play models."
    )
    parser.add_argument("--layout", required=True, choices=LAYOUT_LIST)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def run_episode(model: DiscreteSAC, env: gym.Env) -> Dict[str, float]:
    obs = env.reset()
    done = False
    episode_return = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_return += float(reward)

    base_env = env.unwrapped.base_env
    sparse_return = float(base_env.cumulative_sparse_rewards)
    built_in_shaped_return = float(base_env.cumulative_shaped_rewards)
    custom_shaped_return = float(
        env.unwrapped.cumulative_custom_shaped_rewards
    )
    delivery_reward = float(env.unwrapped.mdp.delivery_reward)
    return {
        "total_return": episode_return,
        "sparse_return": sparse_return,
        "built_in_shaped_return": built_in_shaped_return,
        "custom_shaped_return": custom_shaped_return,
        "shaped_return": built_in_shaped_return + custom_shaped_return,
        "deliveries": sparse_return / delivery_reward,
    }


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    with (args.run_dir / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    with (args.run_dir / "training_status.json").open(encoding="utf-8") as f:
        status = json.load(f)
    if status.get("status") != "completed":
        raise ValueError(
            f"Run is not marked completed: {status.get('status')!r}"
        )
    if config.get("algo") != "discrete_sac":
        raise ValueError(f"Expected discrete_sac, got {config.get('algo')!r}")
    if config.get("layout") != args.layout:
        raise ValueError(
            f"Expected layout {args.layout!r}, got {config.get('layout')!r}"
        )
    return config


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    config = load_config(args)
    output = args.output or (args.run_dir / "evaluation.json")

    env = gym.make(
        "OvercookedMultiEnv-v0",
        layout_name=args.layout,
        **config.get("env_kwargs", {}),
    )
    try:
        partner_model = DiscreteSAC.load(
            args.run_dir / "partner_model",
            env=env.getDummyEnv(1),
            device=args.device,
        )
        env.add_partner_agent(
            StaticModelAgent(partner_model, deterministic=True)
        )
        ego_model = DiscreteSAC.load(
            args.run_dir / "ego_model", env=env, device=args.device
        )
        episodes = [run_episode(ego_model, env) for _ in range(args.episodes)]
    finally:
        env.close()

    results = {
        "algo": "discrete_sac",
        "layout": args.layout,
        "seed": config.get("seed"),
        "run_dir": str(args.run_dir),
        "episodes": args.episodes,
        "deterministic": True,
        "device": str(ego_model.device),
        "ego_ent_coef": ego_model.get_ent_coef(),
        "partner_ent_coef": partner_model.get_ent_coef(),
        "episode_returns": [item["total_return"] for item in episodes],
        "episode_sparse_returns": [
            item["sparse_return"] for item in episodes
        ],
        "episode_shaped_returns": [
            item["shaped_return"] for item in episodes
        ],
        "episode_deliveries": [item["deliveries"] for item in episodes],
    }
    returns = results["episode_returns"]
    sparse_returns = results["episode_sparse_returns"]
    shaped_returns = results["episode_shaped_returns"]
    deliveries = results["episode_deliveries"]
    results.update(
        {
            "mean_return": mean(returns),
            "std_return": pstdev(returns),
            "median_return": median(returns),
            "min_return": min(returns),
            "max_return": max(returns),
            "mean_sparse_return": mean(sparse_returns),
            "mean_shaped_return": mean(shaped_returns),
            "mean_deliveries": mean(deliveries),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    with output.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "total_return",
                "sparse_return",
                "built_in_shaped_return",
                "custom_shaped_return",
                "shaped_return",
                "deliveries",
            ]
        )
        for index, item in enumerate(episodes):
            writer.writerow(
                [
                    index,
                    item["total_return"],
                    item["sparse_return"],
                    item["built_in_shaped_return"],
                    item["custom_shaped_return"],
                    item["shaped_return"],
                    item["deliveries"],
                ]
            )


if __name__ == "__main__":
    main()
