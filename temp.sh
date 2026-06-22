python experiments/run_ppo_dense_experiments.py \
  --layouts simple unident_s random1 random0 random3 \
  --seeds 0 1 2 \
  --timesteps 500000 \
  --evaluation-episodes 1 \
  --run-name-suffix _new \
  --custom-shaping-scale 5.0 \
  --device cuda \
  --verbose 0

python experiments/run_ppo_dense_experiments.py \
  --layouts simple unident_s random1 random0 random3 \
  --seeds 0 1 2 \
  --timesteps 500000 \
  --evaluation-episodes 1 \
  --custom-shaping-scale 5.0 \
  --custom-shaping-extra-scale-v2 25.0 \
  --custom-shaping-version-switch-step 250000 \
  --device cuda \
  --verbose 0
