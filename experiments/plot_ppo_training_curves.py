#!/usr/bin/env python3
"""Plot PPO TensorBoard training rewards as a seed-by-layout grid."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LAYOUTS = ("simple", "unident_s", "random0", "random1", "random3")


def load_scalar(event_dir: Path, tag: str):
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        return [], []
    events = accumulator.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/selfplay/ppo"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="rollout/ep_rew_mean")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()

    fig, axes = plt.subplots(len(args.seeds), len(LAYOUTS), figsize=(20, 10), sharex=True)
    fig.suptitle(f"PPO training reward: {args.run_name}", fontsize=14)

    for row, seed in enumerate(args.seeds):
        for col, layout in enumerate(LAYOUTS):
            ax = axes[row][col]
            event_root = args.results_dir / layout / f"seed_{seed}" / args.run_name / "logs" / "ego"
            event_dirs = sorted(path.parent for path in event_root.glob("*/events.out.tfevents*"))
            if not event_dirs:
                ax.text(0.5, 0.5, "Missing log", ha="center", va="center")
            else:
                steps, values = load_scalar(event_dirs[-1], args.tag)
                if steps:
                    ax.plot(steps, values, linewidth=1.2, color="#1976d2")
                else:
                    ax.text(0.5, 0.5, f"Missing {args.tag}", ha="center", va="center")
            if row == 0:
                ax.set_title(layout)
            if col == 0:
                ax.set_ylabel(f"seed {seed}\\nreward")
            if row == len(args.seeds) - 1:
                ax.set_xlabel("training steps")
            ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
