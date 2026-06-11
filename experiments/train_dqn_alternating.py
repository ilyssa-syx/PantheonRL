"""
Train one Alternating DQN self-play experiment on Overcooked.

Only the latest Ego and Partner models are saved. Replay buffers remain
in-memory during an uninterrupted run and are intentionally not checkpointed.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.utils import get_linear_fn, set_random_seed

from pantheonrl.common.agents import StaticModelAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
import overcookedgym  # noqa: F401  Registers OvercookedMultiEnv-v0.

from train_dqn_selfplay import (
    OPTIONAL_MODEL_ARGS,
    build_model_kwargs,
    effective_hyperparameters,
    replay_size,
)


@dataclass
class GlobalExplorationSchedule:
    """Map a phase-local SB3 progress value to global per-agent progress."""

    initial_eps: float
    final_eps: float
    exploration_fraction: float
    phase_target_timesteps: int
    total_agent_timesteps: int

    def __call__(self, phase_progress_remaining: float) -> float:
        completed_steps = (
            1.0 - phase_progress_remaining
        ) * self.phase_target_timesteps
        completed_fraction = min(
            max(completed_steps / self.total_agent_timesteps, 0.0), 1.0
        )
        if completed_fraction > self.exploration_fraction:
            return self.final_eps
        return self.initial_eps + completed_fraction * (
            self.final_eps - self.initial_eps
        ) / self.exploration_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DQN with alternating frozen-partner self-play."
    )
    parser.add_argument("--layout", required=True, choices=LAYOUT_LIST)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partner-seed-offset", type=int, default=1000)
    parser.add_argument("--timesteps-per-agent", type=int, default=500_000)
    parser.add_argument("--phase-timesteps", type=int, default=50_000)
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.1,
        help="Applied to both models over each model's full timestep budget.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/alternating_dqn"),
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
    if args.timesteps_per_agent <= 0:
        raise ValueError("--timesteps-per-agent must be positive")
    if args.phase_timesteps <= 0:
        raise ValueError("--phase-timesteps must be positive")
    if args.timesteps_per_agent % args.phase_timesteps != 0:
        raise ValueError(
            "--timesteps-per-agent must be divisible by --phase-timesteps"
        )
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


def make_run_name(args: argparse.Namespace) -> str:
    parts = [
        f"steps_per_agent_{args.timesteps_per_agent}",
        f"phase_steps_{args.phase_timesteps}",
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
        / args.layout
        / f"seed_{args.seed}"
        / make_run_name(args)
    )


def atomic_write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    temporary_path.replace(path)


def atomic_save_model(model: DQN, path: Path) -> None:
    temporary_path = path.with_name(f".{path.stem}.tmp.zip")
    if temporary_path.exists():
        temporary_path.unlink()
    model.save(temporary_path)
    temporary_path.replace(path)


def recover_pending_phase(
    run_dir: Path,
    completed_phases: int,
    history: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    commit_path = run_dir / ".phase_commit.json"
    if not commit_path.is_file():
        for role in ("ego", "partner"):
            stale_path = run_dir / f".{role}_model.next.zip"
            if stale_path.exists():
                stale_path.unlink()
        return completed_phases, history

    commit = load_json(commit_path)
    record = commit["phase_record"]
    role = record["role"]
    next_model_path = run_dir / f".{role}_model.next.zip"
    if next_model_path.exists():
        next_model_path.replace(run_dir / f"{role}_model.zip")

    phase_index = int(record["phase_index"])
    if not any(int(item["phase_index"]) == phase_index for item in history):
        history.append(record)
        history.sort(key=lambda item: int(item["phase_index"]))
        atomic_write_json(run_dir / "phase_history.json", history)
    return max(completed_phases, phase_index + 1), history


def stage_phase_model(
    run_dir: Path,
    role: str,
    model: DQN,
    phase_record: Dict[str, Any],
) -> None:
    next_model_path = run_dir / f".{role}_model.next.zip"
    if next_model_path.exists():
        next_model_path.unlink()
    model.save(next_model_path)
    atomic_write_json(
        run_dir / ".phase_commit.json",
        {"phase_record": phase_record},
    )
    next_model_path.replace(run_dir / f"{role}_model.zip")


def config_identity(
    args: argparse.Namespace, model_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "algo": "dqn",
        "self_play_type": "alternating_self_play",
        "layout": args.layout,
        "seed": args.seed,
        "partner_seed": args.seed + args.partner_seed_offset,
        "partner_seed_offset": args.partner_seed_offset,
        "timesteps_per_agent": args.timesteps_per_agent,
        "phase_timesteps": args.phase_timesteps,
        "model_kwargs": model_kwargs,
    }


def save_config(
    run_dir: Path,
    args: argparse.Namespace,
    model_kwargs: Dict[str, Any],
    effective_params: Optional[Dict[str, Any]] = None,
    ego_timesteps: Optional[int] = None,
    partner_timesteps: Optional[int] = None,
) -> None:
    config = config_identity(args, model_kwargs)
    config.update(
        {
            "policy": "MlpPolicy",
            "frozen_partner_wrapper": "StaticModelAgent",
            "phase_order": ["ego", "partner"],
            "replay_buffer_policy": (
                "retained_in_memory_between_phases_not_checkpointed"
            ),
            "latest_models_only": True,
            "effective_hyperparameters": effective_params,
            "actual_ego_timesteps": ego_timesteps,
            "actual_partner_timesteps": partner_timesteps,
            "output_dir": str(run_dir),
            "saved_files": {
                "ego_model": "ego_model.zip",
                "partner_model": "partner_model.zip",
                "config": "config.json",
                "training_status": "training_status.json",
                "phase_history": "phase_history.json",
            },
        }
    )
    atomic_write_json(run_dir / "config.json", config)


def save_status(
    run_dir: Path,
    args: argparse.Namespace,
    status_name: str,
    completed_phases: int,
    ego_model: Optional[DQN],
    partner_model: Optional[DQN],
    wall_clock_seconds: float,
    error: Optional[str] = None,
) -> None:
    status = {
        "status": status_name,
        "algo": "dqn",
        "self_play_type": "alternating_self_play",
        "layout": args.layout,
        "seed": args.seed,
        "completed_phases": completed_phases,
        "total_phases": 2 * args.timesteps_per_agent // args.phase_timesteps,
        "next_role": "ego" if completed_phases % 2 == 0 else "partner",
        "actual_ego_timesteps": getattr(ego_model, "num_timesteps", None),
        "actual_partner_timesteps": getattr(
            partner_model, "num_timesteps", None
        ),
        "ego_updates": getattr(ego_model, "_n_updates", None),
        "partner_updates": getattr(partner_model, "_n_updates", None),
        "ego_replay_transitions_in_memory": replay_size(ego_model),
        "partner_replay_transitions_in_memory": replay_size(partner_model),
        "wall_clock_seconds": wall_clock_seconds,
        "latest_models_only": True,
    }
    if status_name == "completed":
        status["next_role"] = None
    if error is not None:
        status["error"] = error
    atomic_write_json(run_dir / "training_status.json", status)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def prepare_run(
    args: argparse.Namespace, model_kwargs: Dict[str, Any]
) -> Tuple[Path, int, List[Dict[str, Any]]]:
    run_dir = get_run_dir(args)
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        (run_dir / "logs" / "ego").mkdir(parents=True)
        (run_dir / "logs" / "partner").mkdir(parents=True)
        return run_dir, 0, []

    config_path = run_dir / "config.json"
    status_path = run_dir / "training_status.json"
    history_path = run_dir / "phase_history.json"
    if not config_path.is_file() or not status_path.is_file():
        raise FileExistsError(
            "Run directory exists without resumable metadata: "
            f"{run_dir}"
        )
    existing_config = load_json(config_path)
    expected_identity = config_identity(args, model_kwargs)
    mismatches = {
        key: (existing_config.get(key), value)
        for key, value in expected_identity.items()
        if existing_config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing run config does not match arguments: {mismatches}"
        )

    status = load_json(status_path)
    if status.get("status") == "completed":
        raise FileExistsError(f"Run is already completed: {run_dir}")
    for model_name in ("ego_model.zip", "partner_model.zip"):
        if not (run_dir / model_name).is_file():
            raise FileNotFoundError(
                f"Cannot resume without latest model: {run_dir / model_name}"
            )
    history = load_json(history_path) if history_path.is_file() else []
    completed_phases, history = recover_pending_phase(
        run_dir, int(status.get("completed_phases", 0)), history
    )
    return run_dir, completed_phases, history


def make_role_env(layout: str, role: str, frozen_model: DQN) -> gym.Env:
    ego_agent_idx = 0 if role == "ego" else 1
    env = gym.make(
        "OvercookedMultiEnv-v0",
        layout_name=layout,
        ego_agent_idx=ego_agent_idx,
    )
    env.add_partner_agent(StaticModelAgent(frozen_model, deterministic=True))
    return env


def initialize_models(
    args: argparse.Namespace,
    run_dir: Path,
    model_kwargs: Dict[str, Any],
) -> Tuple[DQN, DQN]:
    ego_env = gym.make(
        "OvercookedMultiEnv-v0", layout_name=args.layout, ego_agent_idx=0
    )
    partner_env = gym.make(
        "OvercookedMultiEnv-v0", layout_name=args.layout, ego_agent_idx=1
    )
    try:
        ego_model = DQN(
            "MlpPolicy",
            ego_env,
            seed=args.seed,
            tensorboard_log=str(run_dir / "logs" / "ego"),
            **model_kwargs,
        )
        partner_model = DQN(
            "MlpPolicy",
            partner_env,
            seed=args.seed + args.partner_seed_offset,
            tensorboard_log=str(run_dir / "logs" / "partner"),
            **model_kwargs,
        )
    finally:
        ego_env.close()
        partner_env.close()
    return ego_model, partner_model


def load_models(
    args: argparse.Namespace, run_dir: Path
) -> Tuple[DQN, DQN]:
    ego_model = DQN.load(run_dir / "ego_model", device=args.device)
    partner_model = DQN.load(run_dir / "partner_model", device=args.device)
    ego_model.tensorboard_log = str(run_dir / "logs" / "ego")
    partner_model.tensorboard_log = str(run_dir / "logs" / "partner")
    return ego_model, partner_model


def train_phase(
    args: argparse.Namespace,
    phase_index: int,
    role: str,
    active_model: DQN,
    frozen_model: DQN,
) -> Dict[str, Any]:
    role_phase_index = phase_index // 2
    role_seed = (
        args.seed if role == "ego" else args.seed + args.partner_seed_offset
    )
    phase_seed = role_seed + 10_000 * role_phase_index
    set_random_seed(phase_seed)

    before_steps = active_model.num_timesteps
    before_updates = active_model._n_updates
    phase_target = before_steps + args.phase_timesteps
    active_model.exploration_schedule = GlobalExplorationSchedule(
        active_model.exploration_initial_eps,
        active_model.exploration_final_eps,
        active_model.exploration_fraction,
        phase_target,
        args.timesteps_per_agent,
    )

    env = make_role_env(args.layout, role, frozen_model)
    started = time.monotonic()
    try:
        active_model.set_env(env, force_reset=True)
        active_model.learn(
            total_timesteps=args.phase_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"{role}_phase_{role_phase_index:02d}",
        )
    finally:
        env.close()
        active_model.exploration_schedule = get_linear_fn(
            active_model.exploration_initial_eps,
            active_model.exploration_final_eps,
            active_model.exploration_fraction,
        )

    actual_steps = active_model.num_timesteps - before_steps
    if actual_steps != args.phase_timesteps:
        raise RuntimeError(
            f"{role} phase executed {actual_steps} steps, expected "
            f"{args.phase_timesteps}"
        )
    return {
        "phase_index": phase_index,
        "role": role,
        "physical_player_index": 0 if role == "ego" else 1,
        "phase_seed": phase_seed,
        "timesteps_before": before_steps,
        "timesteps_after": active_model.num_timesteps,
        "updates_before": before_updates,
        "updates_after": active_model._n_updates,
        "replay_transitions_in_memory": replay_size(active_model),
        "exploration_rate_after": active_model.exploration_rate,
        "wall_clock_seconds": time.monotonic() - started,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    model_kwargs = build_model_kwargs(args)
    run_dir, completed_phases, history = prepare_run(args, model_kwargs)
    session_started = time.monotonic()
    previous_wall_time = 0.0

    status_path = run_dir / "training_status.json"
    if status_path.is_file():
        previous_wall_time = float(
            load_json(status_path).get("wall_clock_seconds", 0.0)
        )

    fresh_run = completed_phases == 0 and not (run_dir / "ego_model.zip").is_file()
    if fresh_run:
        ego_model, partner_model = initialize_models(args, run_dir, model_kwargs)
        save_config(
            run_dir,
            args,
            model_kwargs,
            effective_hyperparameters(ego_model),
            ego_model.num_timesteps,
            partner_model.num_timesteps,
        )
        atomic_save_model(ego_model, run_dir / "ego_model.zip")
        atomic_save_model(partner_model, run_dir / "partner_model.zip")
        atomic_write_json(run_dir / "phase_history.json", history)
    else:
        ego_model, partner_model = load_models(args, run_dir)

    total_phases = 2 * args.timesteps_per_agent // args.phase_timesteps
    print(
        f"Alternating DQN: layout={args.layout}, seed={args.seed}, "
        f"phases={completed_phases}/{total_phases}"
    )
    print(f"Run directory: {run_dir}")

    try:
        save_status(
            run_dir,
            args,
            "running",
            completed_phases,
            ego_model,
            partner_model,
            previous_wall_time,
        )
        commit_path = run_dir / ".phase_commit.json"
        if commit_path.exists():
            commit_path.unlink()
        for phase_index in range(completed_phases, total_phases):
            role = "ego" if phase_index % 2 == 0 else "partner"
            active_model, frozen_model = (
                (ego_model, partner_model)
                if role == "ego"
                else (partner_model, ego_model)
            )
            print(
                f"Phase {phase_index + 1}/{total_phases}: train {role}, "
                "freeze the other model",
                flush=True,
            )
            phase_record = train_phase(
                args, phase_index, role, active_model, frozen_model
            )
            stage_phase_model(run_dir, role, active_model, phase_record)
            if not any(
                int(item["phase_index"]) == phase_index for item in history
            ):
                history.append(phase_record)
            atomic_write_json(run_dir / "phase_history.json", history)
            completed_phases = phase_index + 1
            wall_time = previous_wall_time + time.monotonic() - session_started
            save_status(
                run_dir,
                args,
                "running",
                completed_phases,
                ego_model,
                partner_model,
                wall_time,
            )
            (run_dir / ".phase_commit.json").unlink()

        wall_time = previous_wall_time + time.monotonic() - session_started
        save_config(
            run_dir,
            args,
            model_kwargs,
            effective_hyperparameters(ego_model),
            ego_model.num_timesteps,
            partner_model.num_timesteps,
        )
        save_status(
            run_dir,
            args,
            "completed",
            completed_phases,
            ego_model,
            partner_model,
            wall_time,
        )
    except Exception as exc:
        save_status(
            run_dir,
            args,
            "failed",
            completed_phases,
            ego_model,
            partner_model,
            previous_wall_time + time.monotonic() - session_started,
            str(exc),
        )
        raise

    print("Alternating DQN training complete.")


if __name__ == "__main__":
    main()
