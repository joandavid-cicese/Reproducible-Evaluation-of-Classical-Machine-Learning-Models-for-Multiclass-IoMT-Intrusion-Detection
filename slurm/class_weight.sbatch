#!/bin/bash
#SBATCH --job-name=iomt_classweight
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# D6 / R3-6: sensitivity of Random Forest to class_weight='balanced'. Design fixed before running:
#   RF only (model selected by CV), global scaling, the same 5 model seeds as the seed study,
#   same data, row order (shuffle 42), hyperparameters and CPUs (32). Compared PAIRED BY SEED with
#   results/seeds/global_seed*. Sensitivity analysis only: it does not replace the main results
#   (reference-study defaults) and it is not used to select a model.
source slurm/env.sh
SEEDS=(42 123 2024 2025 2026)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "model_seed=$S scaling=global class_weight=balanced model=RandomForest"
python src/04_run_experiment.py --train "$PAPER_TRAIN" --test "$PAPER_TEST" \
    --scaling global --seed "$S" --models RandomForest --class_weight balanced \
    --results_dir results/class_weight
echo "end=$(date '+%F %T')"
