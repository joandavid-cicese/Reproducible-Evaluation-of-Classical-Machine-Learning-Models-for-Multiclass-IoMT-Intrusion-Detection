"""
05_cross_validation.py
----------------------
Stratified K-fold cross-validation on D_train ONLY (D_test is never loaded).

The scaler is part of a scikit-learn Pipeline, so StandardScaler is fit on the
training folds of each split and applied to the validation fold:

    Pipeline([("scaler", StandardScaler()), ("clf", model)])

Each (task, model, fold) result is written to its own JSON file, which makes the
script resumable and allows running folds in parallel (e.g. a SLURM job array):

    # all tasks, models and folds sequentially
    python 05_cross_validation.py --train CICIoMT2024_train.csv --results_dir results_cv

    # a single fold (e.g. SLURM_ARRAY_TASK_ID in 0..9)
    python 05_cross_validation.py --train CICIoMT2024_train.csv --results_dir results_cv \
        --tasks 19class --models AdaBoost --fold 3

    # aggregate everything already computed (mean ± std per task/model)
    python 05_cross_validation.py --results_dir results_cv --summarize

Models, hyperparameters, data loading and label encoding are imported from
04_run_experiment.py so both scripts use exactly the same configuration.
"""

import argparse
import glob
import importlib.util
import json
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Reuse the exact configuration of 04_run_experiment.py
_spec = importlib.util.spec_from_file_location(
    "exp04", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_run_experiment.py"))
exp04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp04)

SEED = exp04.SEED


def fold_path(results_dir, task, model, fold):
    return os.path.join(results_dir, "folds", f"{task}__{model}__fold{fold:02d}.json")


def run(args):
    os.makedirs(os.path.join(args.results_dir, "folds"), exist_ok=True)
    df, n_dropped = exp04.load_dataset(args.train, "Train")
    feats = exp04.feature_columns(df)
    X = df[feats].to_numpy(dtype=np.float64)
    labels = {t: df[c] for t, c in exp04.TASKS.items()}
    del df
    folds_to_run = [args.fold] if args.fold is not None else list(range(args.n_splits))

    for task in args.tasks:
        le = LabelEncoder()
        y = le.fit_transform(labels[task])
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
        splits = list(skf.split(X, y))

        for name in args.models:
            for k in folds_to_run:
                out = fold_path(args.results_dir, task, name, k)
                if os.path.exists(out) and not args.overwrite:
                    logger.info(f"[{task}] {name} fold {k}: already done, skipping.")
                    continue
                tr_idx, va_idx = splits[k]
                model = exp04.get_configured_models()[name]
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
                logger.info(f"[{task}] {name} fold {k + 1}/{args.n_splits}: "
                            f"train={len(tr_idx)} val={len(va_idx)}")
                train_time, warn_list = exp04.fit_recording_warnings(pipe, X[tr_idx], y[tr_idx])
                y_pred = pipe.predict(X[va_idx])
                metrics = exp04.compute_metrics(y[va_idx], y_pred)
                metrics["train_time_sec"] = train_time
                record = {
                    "task": task, "model": name, "fold": k, "n_splits": args.n_splits,
                    "metrics": metrics,
                    "diagnostics": exp04.model_diagnostics(pipe.named_steps["clf"]),
                    "warnings": warn_list,
                    "class_names": le.classes_.tolist(),
                    "train_rows_dropped": n_dropped,
                    "environment": exp04.environment_info(),
                }
                with open(out, "w") as f:
                    json.dump(record, f, indent=2, default=float)
                logger.info(f"[{task}] {name} fold {k}: macroF1={metrics['f1_macro']:.4f} "
                            f"train={train_time:.0f}s")


def summarize(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "folds", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        row = {"task": r["task"], "model": r["model"], "fold": r["fold"], "n_splits": r["n_splits"]}
        row.update({k: v for k, v in r["metrics"].items()})
        row["n_warnings"] = len(r["warnings"])
        rows.append(row)
    if not rows:
        logger.info("No fold results found.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "cv_all_folds.csv"), index=False)
    cols = ["accuracy", "prec_macro", "recall_macro", "f1_macro", "f1_weighted", "mcc", "train_time_sec"]
    agg = df.groupby(["task", "model"])[cols].agg(["mean", "std"])
    agg["folds_done"] = df.groupby(["task", "model"])["fold"].count()
    agg["warnings_total"] = df.groupby(["task", "model"])["n_warnings"].sum()
    agg.to_csv(os.path.join(results_dir, "cv_summary.csv"))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(agg.round(4))
    incomplete = agg[agg["folds_done"] < df["n_splits"].max()]
    if len(incomplete):
        logger.warning(f"Incomplete task/model combinations:\n{incomplete['folds_done']}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", help="Consolidated D_train CSV")
    p.add_argument("--results_dir", default="./results_cv")
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--fold", type=int, default=None, help="Run only this fold (0-based)")
    p.add_argument("--tasks", nargs="+", choices=list(exp04.TASKS), default=list(exp04.TASKS))
    p.add_argument("--models", nargs="+", choices=exp04.MODEL_NAMES, default=exp04.MODEL_NAMES)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summarize", action="store_true", help="Only aggregate existing fold results")
    p.add_argument("--no_summary", action="store_true",
                   help="Do not aggregate after running (use in parallel SLURM array tasks)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.summarize:
        summarize(a.results_dir)
    else:
        if not a.train:
            raise SystemExit("--train is required unless --summarize is used.")
        run(a)
        if not a.no_summary:
            summarize(a.results_dir)
