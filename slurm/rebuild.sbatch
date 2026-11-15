#!/bin/bash
#SBATCH --job-name=iomt_rebuild
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Rebuild D_train / D_test with `source_file` and validate them against the paper CSVs.
# If the validation fails, this job fails and the ablation (afterok) never starts.
source slurm/env.sh
mkdir -p rebuilt
python src/02_combine_train.py --input_dir data/raw/train --output_file "$REBUILT_TRAIN" | tee logs/rebuild_train_${SLURM_JOB_ID}.txt
python src/03_combine_test.py  --input_dir data/raw/test  --output_file "$REBUILT_TEST"  | tee logs/rebuild_test_${SLURM_JOB_ID}.txt
if grep -q "Unrecognized file pattern" logs/rebuild_train_${SLURM_JOB_ID}.txt logs/rebuild_test_${SLURM_JOB_ID}.txt; then
    echo "ERROR: unrecognized file names (see the [WARNING] lines above)"; exit 1
fi
python src/07_validate_rebuilt.py --paper_train "$PAPER_TRAIN" --paper_test "$PAPER_TEST" \
    --rebuilt_train "$REBUILT_TRAIN" --rebuilt_test "$REBUILT_TEST" \
    --reference_json "$REF_JSON" --out results/validation/validation.json
echo "end=$(date '+%F %T')"
