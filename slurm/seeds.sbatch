#!/bin/bash
#SBATCH --job-name=iomt_seeds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-9
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
# Seed study, design fixed before running:
#   seeds {42, 123, 2024, 2025, 2026} x scaling {global, none} x 3 tasks x 4 models = 120 fits.
#   Only the models' random_state changes; data, features, row order (shuffle 42), hyperparameters
#   and CPUs (32, same as E1/ablation) are identical. Seed 42 is rerun with v2 to check equivalence
#   with the existing E1 (global) and ablation (none) runs. Results are NOT used to pick a seed.
source slurm/env.sh
SEEDS=(42 123 2024 2025 2026)
VARIANTS=(global none)
i=$SLURM_ARRAY_TASK_ID
S=${SEEDS[$((i % 5))]}
V=${VARIANTS[$((i / 5))]}
echo "model_seed=$S scaling=$V"
python src/04_run_experiment.py --train "$PAPER_TRAIN" --test "$PAPER_TEST" \
    --scaling "$V" --seed "$S" --results_dir results/seeds
echo "end=$(date '+%F %T')"
