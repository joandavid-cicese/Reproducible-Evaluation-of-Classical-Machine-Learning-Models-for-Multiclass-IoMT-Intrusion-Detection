#!/bin/bash
# Sourced by every job. Edit ONLY the module lines if your cluster needs them.
# module purge
# module load python/3.12          # the same module you used to create .venv

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
set -eo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

REF_JSON=reference/multitask_results_20251206_115420.json
PAPER_TRAIN=data/paper/CICIoMT2024_train.csv
PAPER_TEST=data/paper/CICIoMT2024_test.csv
REBUILT_TRAIN=rebuilt/CICIoMT2024_train_src.csv
REBUILT_TEST=rebuilt/CICIoMT2024_test_src.csv

echo "job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date '+%F %T')"
python -c "import sys, sklearn, numpy, pandas; print('python', sys.version.split()[0], '| sklearn', sklearn.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__)"
