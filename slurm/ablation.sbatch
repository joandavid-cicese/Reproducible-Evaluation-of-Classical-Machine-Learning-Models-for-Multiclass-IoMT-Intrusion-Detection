#!/bin/bash
#SBATCH --job-name=iomt_ablation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# Ablation: the four scaling variants on the rebuilt CSVs (one array task per variant).
source slurm/env.sh
VARIANTS=(global per_file none leaky)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
echo "variant=$V"
python src/04_run_experiment.py --train "$REBUILT_TRAIN" --test "$REBUILT_TEST" \
    --scaling "$V" --results_dir results/ablation
echo "end=$(date '+%F %T')"
