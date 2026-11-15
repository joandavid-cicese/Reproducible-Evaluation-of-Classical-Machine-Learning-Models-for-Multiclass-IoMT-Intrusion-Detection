#!/bin/bash
#SBATCH --job-name=iomt_convergence
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Convergence sensitivity (R3-8). Baseline (LR 100 / MLP 200 iterations) is E1.
#   task 0: LR max_iter=500 and MLP max_iter=500
#   task 1: LR max_iter=1000
source slurm/env.sh
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python src/04_run_experiment.py --train "$PAPER_TRAIN" --test "$PAPER_TEST" \
        --models LogisticRegression DNN_MLP --lr_max_iter 500 --mlp_max_iter 500 --results_dir results/convergence
else
    python src/04_run_experiment.py --train "$PAPER_TRAIN" --test "$PAPER_TEST" \
        --models LogisticRegression --lr_max_iter 1000 --results_dir results/convergence
fi
echo "end=$(date '+%F %T')"
