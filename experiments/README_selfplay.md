# PPO/A2C/DQN Independent Self-Play Experiments

This folder contains the first unified Overcooked self-play entrypoints for
the current experiment plan:

- `train_selfplay.py`: trains PPO or A2C ego and partner models together.
- `train_dqn_selfplay.py`: trains one DQN ego/partner experiment.
- `evaluate_selfplay.py`: evaluates frozen ego/partner models with
  `model.predict(obs, deterministic=True)`.
- `run_dqn_experiments_tonight.py`: runs the complete 45-run DQN matrix and
  evaluates every completed run.
- `train_dqn_alternating.py`: alternates training one DQN while freezing the
  other DQN.
- `run_dqn_alternating_experiments.py`: runs the complete 45-run Alternating
  DQN matrix.

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
  --episodes 100 \
  --device cpu
```

This writes `evaluation.json` and `evaluation.csv` in the run directory.
Evaluation refuses to run when `--algo` or `--layout` does not match the
completed run's `config.json`. The outputs include total return, sparse return,
shaped return, and number of soup deliveries for every episode.

## DQN Training

The DQN entrypoint applies the same `exploration_fraction` to Ego and Partner:

```bash
python experiments/train_dqn_selfplay.py \
  --layout simple \
  --seed 0 \
  --timesteps 500000 \
  --exploration-fraction 0.1 \
  --device cpu \
  --output-dir results/selfplay
```

The Partner uses `OffPolicyAgent`. Its replay-buffer insertion, target-network
updates, epsilon schedule, `learning_starts`, and final pending transition are
kept in sync with the Ego DQN lifecycle.

Completed runs use configuration-specific directories such as:

```text
results/selfplay/dqn/simple/seed_0/steps_500000__partner_offset_1000__exploration_fraction_0.1/
```

## Tonight's DQN Matrix

The batch launcher runs these combinations sequentially on CPU:

```text
5 layouts x 3 seeds x 3 exploration_fraction values = 45 runs

layouts: simple, unident_s, random1, random0, random3
seeds: 0, 1, 2
exploration_fraction: 0.1, 0.3, 0.5
```

Every completed training run is followed by 100 deterministic evaluation
episodes:

```bash
python experiments/run_dqn_experiments_tonight.py \
  --output-dir results/selfplay
```

Before launching the matrix, run the end-to-end smoke test:

```bash
python experiments/run_dqn_experiments_tonight.py \
  --smoke-test \
  --output-dir results/selfplay_smoke
```

The launcher skips completed training and evaluation outputs, continues after
individual failures by default, and records progress in
`results/selfplay/dqn_batch_status.json`. Smoke-test progress is written to
`dqn_smoke_status.json` instead.

## Alternating DQN

Alternating DQN freezes one deterministic model while training the other, then
switches physical player roles. The default run gives each model 500,000
training steps in 50,000-step phases:

```bash
python experiments/train_dqn_alternating.py \
  --layout simple \
  --seed 0 \
  --timesteps-per-agent 500000 \
  --phase-timesteps 50000 \
  --exploration-fraction 0.1 \
  --device cpu
```

Only the latest `ego_model.zip` and `partner_model.zip` are saved and replaced
after completed phases. Replay buffers are retained in memory during an
uninterrupted run but are not checkpointed, keeping the output small. An
interrupted run resumes from the latest two models and the next unfinished
phase.

Run the complete 45-run matrix and 100-episode deterministic evaluations with:

```bash
python experiments/run_dqn_alternating_experiments.py
```

Before the full matrix, verify the pipeline with:

```bash
python experiments/run_dqn_alternating_experiments.py \
  --smoke-test \
  --output-dir results/alternating_dqn_smoke
```
