#!/bin/bash
#SBATCH --job-name=iomt_tables
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Report + manuscript tables. Uses the most recent COMPLETE run of each configuration;
# cancelled/crashed runs are listed in the report and ignored.
source slurm/env.sh
E1_DIR=$(python src/08_report.py --results_root results --print_run e1)
AB_DIR=$(python src/08_report.py --results_root results --print_run ablation_global)
for d in $E1_DIR $AB_DIR; do
    [ -f "$d/bootstrap_ci.csv" ] || python src/06_bootstrap_ci.py --run_dir "$d" --n_boot 1000
done
python src/08_report.py --results_root results --reference_json "$REF_JSON" --logs_dir logs
python src/09_paper_tables.py --results_root results --out results/paper_tables
