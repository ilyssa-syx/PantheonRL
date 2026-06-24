#!/usr/bin/env bash
set -euo pipefail

cd /workspace/HARL

export PYTHONPATH=/workspace/HARL:/workspace/PantheonRL/overcookedgym/human_aware_rl/overcooked_ai:${PYTHONPATH:-}

PY=/workspace/.venvs/pantheonrl-viz/bin/python
LOG_DIR=/workspace/PantheonRL/results/selfplay/happo

LAYOUTS=(simple unident_s random0 random1 random3)
SEEDS=(0 1 2)

REQUESTED_STEPS=500000
# SB3 PPO rolls requested 500k up to 245 rollouts * 2048 steps = 501760.
TOTAL_STEPS=501760
ROLLOUT_THREADS=1
EPISODE_LENGTH=2048
EVAL_EPISODES=1
PARTNER_SEED_OFFSET=1000
ROLLOUT_UPDATES=$((TOTAL_STEPS / (ROLLOUT_THREADS * EPISODE_LENGTH)))

SWITCH_GLOBAL_STEP=250000
SWITCH_PER_ENV_STEP=$((SWITCH_GLOBAL_STEP / ROLLOUT_THREADS))

COMMON_ARGS=(
  --algo happo
  --env overcooked
  --num_env_steps "$TOTAL_STEPS"
  --n_rollout_threads "$ROLLOUT_THREADS"
  --episode_length "$EPISODE_LENGTH"
  --eval_episodes "$EVAL_EPISODES"
  --n_eval_rollout_threads 1
  --eval_interval "$ROLLOUT_UPDATES"
  --log_interval "$ROLLOUT_UPDATES"
  --cuda True
  --agent_seed_offset "$PARTNER_SEED_OFFSET"
  --hidden_sizes "[64,64]"
  --activation_func tanh
  --use_feature_normalization True
  --use_hidden_layernorm True
  --lr 0.0003
  --critic_lr 0.0003
  --ppo_epoch 10
  --critic_epoch 10
  --actor_num_mini_batch 32
  --critic_num_mini_batch 32
  --entropy_coef 0.0
  --value_loss_coef 0.5
  --max_grad_norm 0.5
  --use_huber_loss False
  --use_clipped_value_loss False
  --use_valuenorm False
  --log_dir "$LOG_DIR"
)

for layout in "${LAYOUTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    "$PY" examples/train.py "${COMMON_ARGS[@]}" \
      --exp_name steps_${REQUESTED_STEPS}__partner_offset_${PARTNER_SEED_OFFSET} \
      --layout_name "$layout" \
      --seed "$seed" \
      --custom_dense_reward False

    "$PY" examples/train.py "${COMMON_ARGS[@]}" \
      --exp_name steps_${REQUESTED_STEPS}__partner_offset_${PARTNER_SEED_OFFSET}__custom_dense__sg_0.99__ss_5.0__v2extra_25.0__switch_${SWITCH_GLOBAL_STEP} \
      --layout_name "$layout" \
      --seed "$seed" \
      --custom_dense_reward True \
      --custom_shaping_gamma 0.99 \
      --custom_shaping_version 1 \
      --custom_shaping_scale 5.0 \
      --custom_shaping_extra_scale_v2 25.0 \
      --custom_shaping_version_switch_step "$SWITCH_PER_ENV_STEP"

    "$PY" examples/train.py "${COMMON_ARGS[@]}" \
      --exp_name steps_${REQUESTED_STEPS}__partner_offset_${PARTNER_SEED_OFFSET}__custom_dense__sg_0.99__ss_5.0_new \
      --layout_name "$layout" \
      --seed "$seed" \
      --custom_dense_reward True \
      --custom_shaping_gamma 0.99 \
      --custom_shaping_version 1 \
      --custom_shaping_scale 5.0
  done
done
