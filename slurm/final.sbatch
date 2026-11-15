#!/bin/bash
#SBATCH --job-name=iomt_final
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# Runs after everything else: CV summary, bootstrap CIs, final report.
source slurm/env.sh
set +e   # produce the report even if one piece is missing
python src/05_cross_validation.py --results_dir results/cv --summarize
# Bootstrap on the runs that 08/09 will actually use (most recent COMPLETE run, never a cancelled one)
E1_DIR=$(python src/08_report.py --results_root results --print_run e1)
AB_DIR=$(python src/08_report.py --results_root results --print_run ablation_global)
for d in $E1_DIR $AB_DIR; do
    python src/06_bootstrap_ci.py --run_dir "$d" --n_boot 1000
done
python src/08_report.py --results_root results --reference_json "$REF_JSON" --logs_dir logs
python src/09_paper_tables.py --results_root results --out results/paper_tables
echo "end=$(date '+%F %T')"
