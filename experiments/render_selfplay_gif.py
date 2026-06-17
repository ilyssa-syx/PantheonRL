"""
Run a frozen PantheonRL Overcooked self-play pair and render one episode.

The input is a completed run directory containing config.json,
ego_model.zip, and partner_model.zip. No trajectory file is required.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OVERCOOKED_AI_ROOT = (
    PROJECT_ROOT / "overcookedgym" / "human_aware_rl" / "overcooked_ai"
)
if str(OVERCOOKED_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERCOOKED_AI_ROOT))

import gym
import cloudpickle
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

import overcookedgym  # noqa: F401  Registers OvercookedMultiEnv-v0.
from pantheonrl.common.agents import StaticModelAgent


if sys.version_info < (3, 8):
    try:
        import pickle5

        cloudpickle.loads = pickle5.loads
    except ImportError:
        pass


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}

TERRAIN_COLORS = {
    " ": "#d9c7aa",
    "X": "#80654a",
    "O": "#d19a36",
    "T": "#c84d4d",
    "P": "#555b66",
    "D": "#4f92c7",
    "S": "#ece8dc",
}

TERRAIN_LABELS = {
    "X": "COUNTER",
    "O": "ONION",
    "T": "TOMATO",
    "P": "POT",
    "D": "SERVE",
    "S": "DISH",
}

PLAYER_COLORS = ("#35a36b", "#3385cc")
OBJECT_COLORS = {
    "onion": "#f5df7a",
    "tomato": "#e65454",
    "dish": "#f7f7f2",
    "soup": "#e8a63a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a completed PantheonRL self-play run, execute one episode, "
            "and render it as a GIF."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing config.json and ego/partner_model.zip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to RUN_DIR/replay.gif.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Playback frames per second.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Render every N environment steps. Use 1 for every step.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=84,
        help="Rendered tile size in pixels.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using deterministic actions.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional directory where individual PNG frames are written.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF writing when --frames-dir is enough.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict[str, Any]:
    required = ("config.json", "ego_model.zip", "partner_model.zip")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Run directory is missing required files: " + ", ".join(missing)
        )
    with (run_dir / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("algo") not in ALGOS:
        raise ValueError("Unsupported algorithm: {!r}".format(config.get("algo")))
    if not config.get("layout"):
        raise ValueError("config.json does not contain a layout")
    return config


def load_model(
    algo_cls: Type[BaseAlgorithm],
    path: Path,
    env: gym.Env,
    device: str,
) -> BaseAlgorithm:
    return algo_cls.load(
        path,
        env=env,
        device=device,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "LinBiolinum_RB.otf" if bold else "LinBiolinum_R.otf"
    path = Path("/usr/share/fonts/opentype/linux-libertine") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2 - top
    draw.text((x, y), text, font=font, fill=fill)


def draw_object(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    name: str,
    radius: int,
    font: ImageFont.ImageFont,
) -> None:
    color = OBJECT_COLORS.get(name, "#cf70ba")
    x, y = center
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline="#292929",
        width=max(1, radius // 6),
    )
    label = {"onion": "O", "tomato": "T", "dish": "D", "soup": "S"}.get(
        name, name[:1].upper()
    )
    centered_text(
        draw,
        (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
        label,
        font,
        "#202020",
    )


def draw_player(
    draw: ImageDraw.ImageDraw,
    player: Any,
    index: int,
    tile_size: int,
    top: int,
    font: ImageFont.ImageFont,
) -> None:
    x, y = player.position
    cx = x * tile_size + tile_size / 2
    cy = top + y * tile_size + tile_size / 2
    radius = int(tile_size * 0.29)
    color = PLAYER_COLORS[index % len(PLAYER_COLORS)]
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=color,
        outline="#ffffff",
        width=max(2, tile_size // 18),
    )
    centered_text(
        draw,
        (
            int(cx - radius),
            int(cy - radius),
            int(cx + radius),
            int(cy + radius),
        ),
        "P{}".format(index),
        font,
        "#ffffff",
    )

    dx, dy = player.orientation
    tip = (cx + dx * radius * 1.45, cy + dy * radius * 1.45)
    side_x, side_y = -dy * radius * 0.35, dx * radius * 0.35
    base_x, base_y = cx + dx * radius * 0.75, cy + dy * radius * 0.75
    draw.polygon(
        [
            tip,
            (base_x + side_x, base_y + side_y),
            (base_x - side_x, base_y - side_y),
        ],
        fill="#ffffff",
        outline="#202020",
    )

    if player.held_object is not None:
        draw_object(
            draw,
            (cx + radius * 0.75, cy - radius * 0.75),
            player.held_object.name,
            max(8, tile_size // 9),
            font,
        )


def render_frame(
    mdp: Any,
    state: Any,
    step: int,
    horizon: int,
    total_return: float,
    sparse_return: float,
    tile_size: int,
) -> Image.Image:
    terrain = mdp.terrain_mtx
    rows = len(terrain)
    cols = len(terrain[0])
    header = max(82, tile_size)
    image = Image.new(
        "RGB", (cols * tile_size, header + rows * tile_size), "#16191e"
    )
    draw = ImageDraw.Draw(image)
    title_font = load_font(max(18, tile_size // 4), bold=True)
    small_font = load_font(max(12, tile_size // 7))
    object_font = load_font(max(11, tile_size // 7), bold=True)

    draw.text(
        (12, 8),
        "{}  |  step {:03d}/{:03d}".format(mdp.layout_name, step, horizon),
        font=title_font,
        fill="#ffffff",
    )
    deliveries = sparse_return / float(mdp.delivery_reward)
    draw.text(
        (12, 42),
        "return: {:.1f}   sparse: {:.1f}   deliveries: {:.0f}".format(
            total_return, sparse_return, deliveries
        ),
        font=small_font,
        fill="#d8dde6",
    )

    for y, row in enumerate(terrain):
        for x, terrain_type in enumerate(row):
            box = (
                x * tile_size,
                header + y * tile_size,
                (x + 1) * tile_size,
                header + (y + 1) * tile_size,
            )
            draw.rectangle(
                box,
                fill=TERRAIN_COLORS.get(terrain_type, "#80654a"),
                outline="#24272d",
                width=max(1, tile_size // 30),
            )
            label = TERRAIN_LABELS.get(terrain_type)
            if label:
                centered_text(draw, box, label, small_font, "#202020")

    for position, obj in state.objects.items():
        x, y = position
        center = (
            x * tile_size + tile_size / 2,
            header + y * tile_size + tile_size / 2,
        )
        draw_object(draw, center, obj.name, max(10, tile_size // 5), object_font)
        if obj.name == "soup" and obj.state is not None:
            soup_type, num_items, cook_time = obj.state
            text = "{} {}/{} t={}".format(
                soup_type[:1].upper(),
                num_items,
                mdp.num_items_for_soup,
                cook_time,
            )
            draw.text(
                (x * tile_size + 4, header + y * tile_size + tile_size - 18),
                text,
                font=small_font,
                fill="#202020",
            )

    for index, player in enumerate(state.players):
        draw_player(draw, player, index, tile_size, header, object_font)

    return image


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.frame_skip <= 0:
        raise ValueError("--frame-skip must be positive")
    if args.tile_size < 40:
        raise ValueError("--tile-size must be at least 40")
    if args.no_gif and args.frames_dir is None:
        raise ValueError("--no-gif requires --frames-dir")

    run_dir = args.run_dir.resolve()
    config = load_config(run_dir)
    algo_cls = ALGOS[config["algo"]]
    output = (args.output or (run_dir / "replay.gif")).resolve()
    frames_dir = args.frames_dir.resolve() if args.frames_dir is not None else None
    deterministic = not args.stochastic

    env = gym.make("OvercookedMultiEnv-v0", layout_name=config["layout"])
    frames = []
    try:
        partner_env = env.getDummyEnv(1)
        partner_model = load_model(
            algo_cls, run_dir / "partner_model", partner_env, args.device
        )
        env.add_partner_agent(
            StaticModelAgent(partner_model, deterministic=deterministic)
        )
        ego_model = load_model(algo_cls, run_dir / "ego_model", env, args.device)

        obs = env.reset()
        base_env = env.unwrapped.base_env
        mdp = env.unwrapped.mdp
        horizon = int(base_env.horizon)
        total_return = 0.0
        step = 0
        frames.append(
            render_frame(
                mdp,
                base_env.state,
                step,
                horizon,
                total_return,
                base_env.cumulative_sparse_rewards,
                args.tile_size,
            )
        )

        done = False
        while not done:
            action, _ = ego_model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            step += 1
            total_return += float(reward)
            if step % args.frame_skip == 0 or done:
                frames.append(
                    render_frame(
                        mdp,
                        base_env.state,
                        step,
                        horizon,
                        total_return,
                        base_env.cumulative_sparse_rewards,
                        args.tile_size,
                    )
                )
    finally:
        env.close()

    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            frame.save(frames_dir / "frame_{:04d}.png".format(index))

    if not args.no_gif:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame_duration_ms = max(1, round(1000 / args.fps))
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            disposal=2,
        )

    outputs = []
    if not args.no_gif:
        outputs.append(str(output))
    if frames_dir is not None:
        outputs.append(str(frames_dir))
    print(
        "Saved {} frames to {} (steps={}, return={:.1f}, sparse={:.1f})".format(
            len(frames),
            ", ".join(outputs),
            step,
            total_return,
            float(base_env.cumulative_sparse_rewards),
        )
    )


if __name__ == "__main__":
    main()
