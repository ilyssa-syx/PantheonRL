"""
Train one DQN independent self-play experiment on Overcooked.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed

from pantheonrl.common.agents import OffPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
import overcookedgym  # noqa: F401  Registers OvercookedMultiEnv-v0.


OPTIONAL_MODEL_ARGS = (
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "gamma",
    "train_freq",
    "gradient_steps",
    "target_update_interval",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DQN independent self-play on Overcooked."
    )
    parser.add_argument("--layout", required=True, choices=LAYOUT_LIST)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partner-seed-offset", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.1,
        help="Applied to both Ego and Partner DQN models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/selfplay"),
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


def validate_args(args: argparse.Namespace) -> None:
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
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


def build_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "device": args.device,
        "verbose": args.verbose,
        "exploration_fraction": args.exploration_fraction,
    }
    for name in OPTIONAL_MODEL_ARGS:
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    return kwargs


def make_run_name(args: argparse.Namespace) -> str:
    parts = [
        f"steps_{args.timesteps}",
        f"partner_offset_{args.partner_seed_offset}",
        f"exploration_fraction_{args.exploration_fraction}",
    ]
    for name in OPTIONAL_MODEL_ARGS:
        value = getattr(args, name)
        if value is not None:
            parts.append(f"{name}_{value}")
    return "__".join(parts)


def get_run_dir(args: argparse.Namespace) -> Path:
    return (
        args.output_dir
        / "dqn"
        / args.layout
        / f"seed_{args.seed}"
        / make_run_name(args)
    )


def prepare_run_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    run_dir = get_run_dir(args)
    working_dir = run_dir.with_name(f".{run_dir.name}.tmp")

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists; refusing to overwrite: {run_dir}")
    if working_dir.exists():
        raise FileExistsError(
            "A temporary run directory already exists. Inspect or remove it "
            f"before retrying: {working_dir}")

    working_dir.mkdir(parents=True)
    (working_dir / "logs").mkdir()
    return run_dir, working_dir


def effective_hyperparameters(model: DQN) -> Dict[str, Any]:
    return {
        "learning_rate": model.learning_rate,
        "buffer_size": model.buffer_size,
        "learning_starts": model.learning_starts,
        "batch_size": model.batch_size,
        "tau": model.tau,
        "gamma": model.gamma,
        "train_freq": {
            "frequency": model.train_freq.frequency,
            "unit": model.train_freq.unit.value,
        },
        "gradient_steps": model.gradient_steps,
        "target_update_interval": model.target_update_interval,
        "exploration_fraction": model.exploration_fraction,
        "exploration_initial_eps": model.exploration_initial_eps,
        "exploration_final_eps": model.exploration_final_eps,
    }


def save_config(
    output_dir: Path,
    final_run_dir: Path,
    args: argparse.Namespace,
    partner_seed: int,
    model_kwargs: Dict[str, Any],
    effective_params: Optional[Dict[str, Any]] = None,
    actual_ego_timesteps: Optional[int] = None,
    actual_partner_timesteps: Optional[int] = None,
    wall_clock_seconds: Optional[float] = None,
) -> None:
    config = {
        "algo": "dqn",
        "layout": args.layout,
        "seed": args.seed,
        "ego_seed": args.seed,
        "partner_seed": partner_seed,
        "partner_seed_offset": args.partner_seed_offset,
        "requested_timesteps": args.timesteps,
        "actual_ego_timesteps": actual_ego_timesteps,
        "actual_partner_timesteps": actual_partner_timesteps,
        "wall_clock_seconds": wall_clock_seconds,
        "policy": "MlpPolicy",
        "partner_wrapper": "OffPolicyAgent",
        "self_play_type": "independent_self_play",
        "model_kwargs": model_kwargs,
        "effective_hyperparameters": effective_params,
        "output_dir": str(final_run_dir),
        "saved_files": {
            "ego_model": "ego_model.zip",
            "partner_model": "partner_model.zip",
            "config": "config.json",
            "training_status": "training_status.json",
        },
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def save_training_status(
    output_dir: Path,
    args: argparse.Namespace,
    status_name: str,
    actual_ego_timesteps: Optional[int] = None,
    actual_partner_timesteps: Optional[int] = None,
    ego_updates: Optional[int] = None,
    partner_updates: Optional[int] = None,
    ego_replay_transitions: Optional[int] = None,
    partner_replay_transitions: Optional[int] = None,
    wall_clock_seconds: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    status = {
        "status": status_name,
        "algo": "dqn",
        "layout": args.layout,
        "seed": args.seed,
        "exploration_fraction": args.exploration_fraction,
        "requested_timesteps": args.timesteps,
        "actual_ego_timesteps": actual_ego_timesteps,
        "actual_partner_timesteps": actual_partner_timesteps,
        "ego_updates": ego_updates,
        "partner_updates": partner_updates,
        "ego_replay_transitions": ego_replay_transitions,
        "partner_replay_transitions": partner_replay_transitions,
        "wall_clock_seconds": wall_clock_seconds,
    }
    if error is not None:
        status["error"] = error
    with (output_dir / "training_status.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(status, f, indent=2, sort_keys=True)
        f.write("\n")


def replay_size(model: Optional[DQN]) -> Optional[int]:
    if model is None or model.replay_buffer is None:
        return None
    return model.replay_buffer.size()


def main() -> None:
    args = parse_args()
    validate_args(args)
    partner_seed = args.seed + args.partner_seed_offset
    run_dir, working_dir = prepare_run_dirs(args)
    model_kwargs = build_model_kwargs(args)
    start_time = time.monotonic()

    set_random_seed(args.seed)

    print(
        f"Training DQN on layout={args.layout}, seed={args.seed}, "
        f"exploration_fraction={args.exploration_fraction}"
    )
    print(f"Working directory: {working_dir}")
    print(f"Completed run directory: {run_dir}")

    save_config(working_dir, run_dir, args, partner_seed, model_kwargs)
    save_training_status(working_dir, args, "running")

    env = None
    ego_model = None
    partner_model = None
    try:
        env = gym.make("OvercookedMultiEnv-v0", layout_name=args.layout)
        partner_model = DQN(
            "MlpPolicy",
            env.getDummyEnv(1),
            seed=partner_seed,
            tensorboard_log=str(working_dir / "logs" / "partner"),
            **model_kwargs,
        )
        partner_agent = OffPolicyAgent(
            partner_model,
            tensorboard_log=str(working_dir / "logs" / "partner_agent"),
            tb_log_name="DQNPartner",
        )
        partner_agent.set_total_training_timesteps(args.timesteps)
        env.add_partner_agent(partner_agent)

        ego_model = DQN(
            "MlpPolicy",
            env,
            seed=args.seed,
            tensorboard_log=str(working_dir / "logs" / "ego"),
            **model_kwargs,
        )
        ego_model.learn(total_timesteps=args.timesteps)
        partner_agent.finish_training(env.get_player_observation(1))

        actual_ego_timesteps = ego_model.num_timesteps
        actual_partner_timesteps = partner_model.num_timesteps
        wall_clock_seconds = time.monotonic() - start_time

        ego_model.tensorboard_log = str(run_dir / "logs" / "ego")
        partner_model.tensorboard_log = str(run_dir / "logs" / "partner")
        ego_model.save(working_dir / "ego_model")
        partner_model.save(working_dir / "partner_model")
        save_config(
            working_dir,
            run_dir,
            args,
            partner_seed,
            model_kwargs,
            effective_hyperparameters(ego_model),
            actual_ego_timesteps,
            actual_partner_timesteps,
            wall_clock_seconds,
        )
        save_training_status(
            working_dir,
            args,
            "completed",
            actual_ego_timesteps,
            actual_partner_timesteps,
            ego_model._n_updates,
            partner_model._n_updates,
            replay_size(ego_model),
            replay_size(partner_model),
            wall_clock_seconds,
        )
    except Exception as exc:
        save_training_status(
            working_dir,
            args,
            "failed",
            getattr(ego_model, "num_timesteps", None),
            getattr(partner_model, "num_timesteps", None),
            getattr(ego_model, "_n_updates", None),
            getattr(partner_model, "_n_updates", None),
            replay_size(ego_model),
            replay_size(partner_model),
            time.monotonic() - start_time,
            str(exc),
        )
        raise
    finally:
        if env is not None:
            env.close()

    working_dir.rename(run_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
