#!/bin/bash
#SBATCH --job-name=iomt_classweight_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Tables for D6: balanced vs default RF, paired by seed -> results/report_class_weight/
source slurm/env.sh
python src/10_class_weight.py --results_root results
