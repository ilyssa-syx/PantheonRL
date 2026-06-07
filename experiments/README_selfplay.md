# PPO/A2C Independent Self-Play Experiments

This folder contains the first unified Overcooked self-play entrypoints for
the current experiment plan:

- `train_selfplay.py`: trains PPO or A2C ego and partner models together.
- `evaluate_selfplay.py`: evaluates frozen ego/partner models with
  `model.predict(obs, deterministic=True)`.

Run commands from the `PantheonRL` directory.

## Basic Training

```bash
python experiments/train_selfplay.py \
  --algo ppo \
  --layout simple \
  --seed 0 \
  --timesteps 500000 \
  --output-dir results/selfplay
```

```bash
python experiments/train_selfplay.py \
  --algo a2c \
  --layout simple \
  --seed 0 \
  --timesteps 500000 \
  --output-dir results/selfplay
```

The run seed is used as the ego seed. The partner seed is derived as
`seed + 1000`, matching the experiment plan.

Each run writes:

- `ego_model.zip`
- `partner_model.zip`
- `config.json`
- `training_status.json`
- TensorBoard logs under `logs/`

Runs are written to a configuration-specific directory such as:

```text
results/selfplay/ppo/simple/seed_0/steps_500000__partner_offset_1000/
```

Training first writes to a neighboring hidden `.tmp` directory. The completed
directory is published only after training, final partner rollout processing,
model saving, and metadata writing all succeed. Existing completed or temporary
run directories are never overwritten.

`config.json` and `training_status.json` record both the requested timestep
budget and the actual Ego and Partner timesteps. On-policy algorithms finish a
complete rollout before stopping, so actual timesteps can exceed the requested
budget.

## Entropy Sensitivity

Use `--ent-coef` for the random0 sensitivity runs:

```bash
python experiments/train_selfplay.py \
  --algo ppo \
  --layout random0 \
  --seed 0 \
  --timesteps 500000 \
  --ent-coef 0.01 \
  --output-dir results/selfplay
```

```bash
python experiments/train_selfplay.py \
  --algo a2c \
  --layout random0 \
  --seed 0 \
  --timesteps 500000 \
  --ent-coef 0.05 \
  --output-dir results/selfplay
```

Omit `--ent-coef` to use the Stable-Baselines3 default.

## Deterministic Evaluation

```bash
python experiments/evaluate_selfplay.py \
  --algo ppo \
  --layout simple \
  --run-dir \
  results/selfplay/ppo/simple/seed_0/steps_500000__partner_offset_1000 \
  --episodes 100
```

This writes `evaluation.json` and `evaluation.csv` in the run directory.
Evaluation refuses to run when `--algo` or `--layout` does not match the
completed run's `config.json`.
