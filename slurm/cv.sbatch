#!/bin/bash
#SBATCH --job-name=iomt_cv
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-119
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
# Stratified 10-fold CV on D_train only. One array task = one (task, model, fold):
#   index = task_idx*40 + model_idx*10 + fold   (3 tasks x 4 models x 10 folds = 120)
source slurm/env.sh
TASKS=(binary 6class 19class)
MODELS=(RandomForest AdaBoost LogisticRegression DNN_MLP)
i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$((i / 40))]}
MODEL=${MODELS[$(((i / 10) % 4))]}
FOLD=$((i % 10))
echo "task=$TASK model=$MODEL fold=$FOLD"
python src/05_cross_validation.py --train "$PAPER_TRAIN" --results_dir results/cv --n_splits 10 \
    --tasks "$TASK" --models "$MODEL" --fold "$FOLD" --no_summary
echo "end=$(date '+%F %T')"
