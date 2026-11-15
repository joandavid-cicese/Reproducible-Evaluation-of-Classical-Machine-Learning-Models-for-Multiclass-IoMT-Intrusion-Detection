#!/bin/bash
#SBATCH --job-name=iomt_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
source slurm/env.sh
python src/00_smoke_test.py
