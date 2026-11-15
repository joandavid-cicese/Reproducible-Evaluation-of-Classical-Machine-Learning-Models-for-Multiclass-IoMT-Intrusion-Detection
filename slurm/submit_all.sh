#!/bin/bash
# Usage (from anywhere):
#   bash slurm/submit_all.sh smoke    -> only the smoke test
#   bash slurm/submit_all.sh all      -> the whole pipeline with dependencies
#   bash slurm/submit_all.sh seeds    -> seed study (5 seeds x global/none), then report + tables
#   bash slurm/submit_all.sh tables   -> only report (08) + manuscript tables (09)
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------- EDIT THESE ----------------
PARTITION="joand_gpu"     # partition of the node
ACCOUNT=""              # only if your cluster requires --account
EXTRA=""                # e.g. "--gres=gpu:1" ONLY if the partition refuses jobs without a GPU
CV_CONCURRENT=12        # max CV array tasks running at the same time
# --------------------------------------------

COMMON="--parsable --partition=$PARTITION ${ACCOUNT:+--account=$ACCOUNT} $EXTRA"
mkdir -p logs results
MODE=${1:-all}

if [ "$MODE" = "smoke" ]; then
    J=$(sbatch $COMMON slurm/smoke.sbatch)
    echo "smoke test: job $J  ->  tail -f logs/iomt_smoke_${J}.out"
    exit 0
fi

if [ "$MODE" = "seeds" ]; then
    S=$(sbatch $COMMON slurm/seeds.sbatch)
    T=$(sbatch $COMMON --dependency=afterany:$S slurm/tables.sbatch)
    echo "seeds=$S tables=$T (afterany seeds)" | tee -a logs/submitted_jobs.txt
    exit 0
fi

if [ "$MODE" = "tables" ]; then
    T=$(sbatch $COMMON slurm/tables.sbatch)
    echo "tables=$T" | tee -a logs/submitted_jobs.txt
    exit 0
fi

E1=$(sbatch $COMMON slurm/e1.sbatch)
RB=$(sbatch $COMMON slurm/rebuild.sbatch)
AB=$(sbatch $COMMON --dependency=afterok:$RB slurm/ablation.sbatch)
CO=$(sbatch $COMMON slurm/convergence.sbatch)
CV=$(sbatch $COMMON --array=0-119%$CV_CONCURRENT slurm/cv.sbatch)
FI=$(sbatch $COMMON --dependency=afterany:$E1:$AB:$CO:$CV slurm/final.sbatch)

{
  echo "submitted $(date '+%F %T')"
  echo "e1=$E1 rebuild=$RB ablation=$AB (afterok rebuild) convergence=$CO cv=$CV final=$FI"
} | tee -a logs/submitted_jobs.txt
echo "Progress: bash slurm/status.sh"
