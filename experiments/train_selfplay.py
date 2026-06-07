"""
Unified independent self-play training for Overcooked experiments.

This is an experiment-ready version of PantheonRL's minimal Overcooked
example. It keeps the same core recipe:

1. Build an Overcooked multi-agent environment.
2. Wrap a learning partner with OnPolicyAgent.
3. Train the ego model against that partner.

The additions are command-line configuration, seed control, model/log saving,
and config recording for reproducible PPO/A2C experiments.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import set_random_seed

from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
import overcookedgym  # noqa: F401  Registers OvercookedMultiEnv-v0.


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO/A2C independent self-play on Overcooked."
    )
    parser.add_argument(
        "--algo",
        required=True,
        choices=sorted(ALGOS),
        help="Stable-Baselines3 algorithm to train.",
    )
    parser.add_argument(
        "--layout",
        default="simple",
        choices=LAYOUT_LIST,
        help="Overcooked layout name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Run seed. Partner seed is derived as seed + partner-seed-offset.",
    )
    parser.add_argument(
        "--partner-seed-offset",
        type=int,
        default=1000,
        help="Offset used to derive the partner seed from the run seed.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total ego-environment timesteps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/selfplay"),
        help="Root directory for models, logs, and config.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="Optional entropy coefficient. Omit to use the SB3 default.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional SB3 learning rate. Omit to use the SB3 default.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional discount factor. Omit to use the SB3 default.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Optional rollout length. Omit to use the SB3 default.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device passed to SB3, for example auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity.",
    )
    return parser.parse_args()


def build_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "device": args.device,
        "verbose": args.verbose,
    }
    if args.ent_coef is not None:
        kwargs["ent_coef"] = args.ent_coef
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate
    if args.gamma is not None:
        kwargs["gamma"] = args.gamma
    if args.n_steps is not None:
        kwargs["n_steps"] = args.n_steps
    return kwargs


def make_run_dir(args: argparse.Namespace) -> Path:
    suffix = ""
    if args.ent_coef is not None:
        suffix = f"_ent_coef_{args.ent_coef:g}"
    run_dir = args.output_dir / args.algo / args.layout / f"seed_{args.seed}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def save_config(
    run_dir: Path,
    args: argparse.Namespace,
    partner_seed: int,
    model_kwargs: Dict[str, Any],
) -> None:
    config = {
        "algo": args.algo,
        "layout": args.layout,
        "seed": args.seed,
        "ego_seed": args.seed,
        "partner_seed": partner_seed,
        "partner_seed_offset": args.partner_seed_offset,
        "timesteps": args.timesteps,
        "policy": "MlpPolicy",
        "partner_wrapper": "OnPolicyAgent",
        "self_play_type": "independent_self_play",
        "model_kwargs": model_kwargs,
        "output_dir": str(run_dir),
        "saved_files": {
            "ego_model": "ego_model.zip",
            "partner_model": "partner_model.zip",
            "config": "config.json",
            "training_status": "training_status.json",
        },
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def save_training_status(run_dir: Path, args: argparse.Namespace) -> None:
    status = {
        "status": "completed",
        "algo": args.algo,
        "layout": args.layout,
        "seed": args.seed,
        "timesteps": args.timesteps,
    }
    with (run_dir / "training_status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    algo_cls = ALGOS[args.algo]
    partner_seed = args.seed + args.partner_seed_offset
    run_dir = make_run_dir(args)
    model_kwargs = build_model_kwargs(args)

    set_random_seed(args.seed)

    env = gym.make("OvercookedMultiEnv-v0", layout_name=args.layout)
    
    if not hasattr(env, "seed"):

        env.seed = lambda seed=None: None

    partner_env = env.getDummyEnv(1)

    if not hasattr(partner_env, "seed"):

        partner_env.seed = lambda seed=None: None

    print(f"Training {args.algo.upper()} on layout={args.layout}, seed={args.seed}")
    print(f"Writing outputs to: {run_dir}")

    partner_model = algo_cls(
        "MlpPolicy",
        partner_env,
        seed=partner_seed,
        tensorboard_log=str(run_dir / "logs" / "partner"),
        **model_kwargs,
    )
    partner_agent = OnPolicyAgent(
        partner_model,
        tensorboard_log=str(run_dir / "logs" / "partner_agent"),
        tb_log_name=f"{args.algo.upper()}Partner",
    )
    env.add_partner_agent(partner_agent)

    ego_model = algo_cls(
        "MlpPolicy",
        env,
        seed=args.seed,
        tensorboard_log=str(run_dir / "logs" / "ego"),
        **model_kwargs,
    )

    save_config(run_dir, args, partner_seed, model_kwargs)
    ego_model.learn(total_timesteps=args.timesteps)

    ego_model.save(run_dir / "ego_model")
    partner_model.save(run_dir / "partner_model")
    save_training_status(run_dir, args)
    print("Training complete.")


if __name__ == "__main__":
    main()
