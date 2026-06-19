"""Run PPO-only Overcooked self-play experiments with custom dense reward."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from overcookedgym.overcooked_utils import LAYOUT_LIST
from train_ppo_a2c import make_run_name


DEFAULT_LAYOUTS = ["simple", "unident_s", "random1", "random0", "random3"]
DEFAULT_SEEDS = [0, 1, 2]
CUSTOM_SHAPING_GAMMA = 0.99
CUSTOM_SHAPING_SCALE = 1.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO-only Overcooked self-play experiments."
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=DEFAULT_LAYOUTS,
        choices=LAYOUT_LIST,
        help="Layouts to train. Defaults to the 5 fair-comparison layouts.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to train for each layout.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/selfplay")
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument("--partner-seed-offset", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument(
        "--custom-shaping-version",
        type=int,
        default=1,
        choices=[1, 2],
        help="Custom dense reward version passed to train_ppo_a2c.py.",
    )
    parser.add_argument(
        "--no-custom-dense-reward",
        action="store_true",
        help="Disable the custom progress-score shaping. It is enabled by default.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only train; do not run deterministic evaluation afterward.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only simple/seed0 with 1,000 timesteps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the matrix when one command fails.",
    )
    return parser.parse_args()


def get_run_dir(args: argparse.Namespace) -> Path:
    return (
        args.output_dir
        / "ppo"
        / args.layout
        / f"seed_{args.seed}"
        / make_run_name(args)
    )


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
        evaluation.get("algo") == "ppo"
        and evaluation.get("episodes") == episodes
        and evaluation.get("deterministic") is True
        and "mean_deliveries" in evaluation
    )


def write_status(path: Path, results: List[Dict[str, Any]]) -> None:
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


def add_optional_arg(command: List[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([f"--{name.replace('_', '-')}", str(value)])


def matrix_from_args(args: argparse.Namespace) -> List[tuple]:
    if args.smoke_test:
        return [("simple", 0)]
    return [(layout, seed) for layout in args.layouts for seed in args.seeds]


def run_matrix(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_ppo_a2c.py"
    evaluate_script = script_dir / "eval_ppo_a2c.py"
    status_name = (
        "ppo_dense_smoke_status.json"
        if args.smoke_test
        else "ppo_dense_batch_status.json"
    )
    status_path = args.output_dir / status_name
    timesteps = 1_000 if args.smoke_test else args.timesteps
    use_custom_dense = not args.no_custom_dense_reward
    matrix = matrix_from_args(args)
    results: List[Dict[str, Any]] = []
    failures = 0

    for index, (layout, seed) in enumerate(matrix, start=1):
        run_args = argparse.Namespace(
            algo="ppo",
            layout=layout,
            seed=seed,
            partner_seed_offset=args.partner_seed_offset,
            timesteps=timesteps,
            output_dir=args.output_dir,
            ent_coef=args.ent_coef,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_steps=args.n_steps,
            custom_dense_reward=use_custom_dense,
            custom_shaping_gamma=CUSTOM_SHAPING_GAMMA,
            custom_shaping_scale=CUSTOM_SHAPING_SCALE,
            custom_shaping_version=args.custom_shaping_version,
        )
        run_dir = get_run_dir(run_args)
        result: Dict[str, Any] = {
            "layout": layout,
            "seed": seed,
            "timesteps": timesteps,
            "custom_dense_reward": use_custom_dense,
            "custom_shaping_gamma": CUSTOM_SHAPING_GAMMA,
            "custom_shaping_scale": CUSTOM_SHAPING_SCALE,
            "custom_shaping_version": args.custom_shaping_version,
            "run_name": make_run_name(run_args),
            "run_dir": str(run_dir),
            "training": "pending",
            "evaluation": "skipped" if args.skip_eval else "pending",
        }
        results.append(result)

        print(f"\n[{index}/{len(matrix)}] layout={layout}, seed={seed}")
        started = time.monotonic()
        try:
            if completed_training(run_dir):
                result["training"] = "skipped_completed"
            else:
                train_command = [
                    sys.executable,
                    str(train_script),
                    "--algo", "ppo",
                    "--layout", layout,
                    "--seed", str(seed),
                    "--partner-seed-offset", str(args.partner_seed_offset),
                    "--timesteps", str(timesteps),
                    "--output-dir", str(args.output_dir),
                    "--device", args.device,
                    "--verbose", str(args.verbose),
                ]
                if use_custom_dense:
                    train_command.extend(
                        [
                            "--custom-dense-reward",
                            "--custom-shaping-gamma",
                            str(CUSTOM_SHAPING_GAMMA),
                            "--custom-shaping-scale",
                            str(CUSTOM_SHAPING_SCALE),
                            "--custom-shaping-version",
                            str(args.custom_shaping_version),
                        ]
                    )
                for name in ("ent_coef", "learning_rate", "gamma", "n_steps"):
                    add_optional_arg(train_command, name, getattr(args, name))
                run_command(train_command, args.dry_run)
                result["training"] = (
                    "dry_run" if args.dry_run else "completed"
                )

            if not args.skip_eval:
                if completed_evaluation(run_dir, args.evaluation_episodes):
                    result["evaluation"] = "skipped_completed"
                else:
                    evaluate_command = [
                        sys.executable,
                        str(evaluate_script),
                        "--algo", "ppo",
                        "--layout", layout,
                        "--run-dir", str(run_dir),
                        "--episodes", str(args.evaluation_episodes),
                        "--device", args.device,
                    ]
                    run_command(evaluate_command, args.dry_run)
                    result["evaluation"] = (
                        "dry_run" if args.dry_run else "completed"
                    )
        except Exception as exc:
            failures += 1
            result["error"] = str(exc)
            if result["training"] == "pending":
                result["training"] = "failed"
            elif result["evaluation"] == "pending":
                result["evaluation"] = "failed"
            if args.stop_on_error:
                result["wall_clock_seconds"] = time.monotonic() - started
                write_status(status_path, results)
                raise
        finally:
            result["wall_clock_seconds"] = time.monotonic() - started
            if not args.dry_run:
                write_status(status_path, results)

    print(f"\nFinished {len(matrix)} runs with {failures} failures.")
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
