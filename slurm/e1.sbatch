#!/bin/bash
#SBATCH --job-name=iomt_e1
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# E1: rerun the paper configuration on the paper CSVs and compare with the paper JSON.
source slurm/env.sh
python src/04_run_experiment.py --train "$PAPER_TRAIN" --test "$PAPER_TEST" \
    --scaling global --results_dir results/e1 --reference_json "$REF_JSON"
echo "end=$(date '+%F %T')"
