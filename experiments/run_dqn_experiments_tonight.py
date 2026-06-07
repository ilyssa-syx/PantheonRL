"""
Run tonight's 5-layout x 3-seed x 3-exploration-fraction DQN matrix.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

from train_dqn_selfplay import get_run_dir, make_run_name


LAYOUTS = ["simple", "unident_s", "random1", "random0", "random3"]
SEEDS = [0, 1, 2]
EXPLORATION_FRACTIONS = [0.1, 0.3, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete 45-run DQN experiment matrix on CPU."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/selfplay"))
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--partner-seed-offset", type=int, default=1000)
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only simple/seed0/fraction0.1 with 1,000 timesteps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the matrix when one training or evaluation command fails.",
    )
    return parser.parse_args()


def completed_training(run_dir: Path) -> bool:
    status_path = run_dir / "training_status.json"
    if not status_path.is_file():
        return False
    with status_path.open(encoding="utf-8") as f:
        status = json.load(f)
    return (
        status.get("status") == "completed"
        and (run_dir / "ego_model.zip").is_file()
        and (run_dir / "partner_model.zip").is_file()
    )


def completed_evaluation(run_dir: Path, episodes: int) -> bool:
    evaluation_path = run_dir / "evaluation.json"
    if not evaluation_path.is_file():
        return False
    with evaluation_path.open(encoding="utf-8") as f:
        evaluation = json.load(f)
    return (
        evaluation.get("algo") == "dqn"
        and evaluation.get("episodes") == episodes
        and evaluation.get("deterministic") is True
        and evaluation.get("device") == "cpu"
    )


def write_batch_status(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": results}, f, indent=2, sort_keys=True)
        f.write("\n")
    temporary_path.replace(path)


def run_command(command: List[str], dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def run_matrix(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_dqn_selfplay.py"
    evaluate_script = script_dir / "evaluate_selfplay.py"
    status_name = (
        "dqn_smoke_status.json" if args.smoke_test
        else "dqn_batch_status.json"
    )
    status_path = args.output_dir / status_name
    results: List[Dict[str, Any]] = []

    if args.smoke_test:
        matrix = [("simple", 0, 0.1)]
        timesteps = 1_000
    else:
        matrix = [
            (layout, seed, fraction)
            for layout in LAYOUTS
            for seed in SEEDS
            for fraction in EXPLORATION_FRACTIONS
        ]
        timesteps = args.timesteps

    total = len(matrix)
    failures = 0
    for index, (layout, seed, fraction) in enumerate(matrix, start=1):
        run_args = argparse.Namespace(
            output_dir=args.output_dir,
            layout=layout,
            seed=seed,
            partner_seed_offset=args.partner_seed_offset,
            timesteps=timesteps,
            exploration_fraction=fraction,
            **{name: None for name in (
                "learning_rate",
                "buffer_size",
                "learning_starts",
                "batch_size",
                "gamma",
                "train_freq",
                "gradient_steps",
                "target_update_interval",
            )},
        )
        run_dir = get_run_dir(run_args)
        result: Dict[str, Any] = {
            "layout": layout,
            "seed": seed,
            "exploration_fraction": fraction,
            "timesteps": timesteps,
            "run_name": make_run_name(run_args),
            "run_dir": str(run_dir),
            "training": "pending",
            "evaluation": "pending",
        }
        results.append(result)

        print(
            f"\n[{index}/{total}] layout={layout}, seed={seed}, "
            f"exploration_fraction={fraction}",
            flush=True,
        )
        started = time.monotonic()
        try:
            if completed_training(run_dir):
                result["training"] = "skipped_completed"
            else:
                train_command = [
                    sys.executable,
                    str(train_script),
                    "--layout", layout,
                    "--seed", str(seed),
                    "--partner-seed-offset", str(args.partner_seed_offset),
                    "--timesteps", str(timesteps),
                    "--exploration-fraction", str(fraction),
                    "--output-dir", str(args.output_dir),
                    "--device", "cpu",
                    "--verbose", str(args.verbose),
                ]
                run_command(train_command, args.dry_run)
                result["training"] = (
                    "dry_run" if args.dry_run else "completed")

            if completed_evaluation(run_dir, args.evaluation_episodes):
                result["evaluation"] = "skipped_completed"
            else:
                evaluate_command = [
                    sys.executable,
                    str(evaluate_script),
                    "--algo", "dqn",
                    "--layout", layout,
                    "--run-dir", str(run_dir),
                    "--episodes", str(args.evaluation_episodes),
                    "--device", "cpu",
                ]
                run_command(evaluate_command, args.dry_run)
                result["evaluation"] = (
                    "dry_run" if args.dry_run else "completed")
        except Exception as exc:
            failures += 1
            result["error"] = str(exc)
            if result["training"] == "pending":
                result["training"] = "failed"
            elif result["evaluation"] == "pending":
                result["evaluation"] = "failed"
            if args.stop_on_error:
                result["wall_clock_seconds"] = time.monotonic() - started
                write_batch_status(status_path, results)
                raise
        finally:
            result["wall_clock_seconds"] = time.monotonic() - started
            if not args.dry_run:
                write_batch_status(status_path, results)

    print(f"\nFinished {total} runs with {failures} failures.")
    return 1 if failures else 0


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.evaluation_episodes <= 0:
        raise ValueError("--evaluation-episodes must be positive")
    raise SystemExit(run_matrix(args))


if __name__ == "__main__":
    main()
