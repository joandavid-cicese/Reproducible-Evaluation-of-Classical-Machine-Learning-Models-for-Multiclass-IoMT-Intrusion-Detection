"""
04_run_experiment.py
--------------------
Train each model on D_train and evaluate it ONCE on the held-out D_test.

Reproduces the configuration that produced the results reported in the paper
(original script: ml_experiment_make.py, AdaBoost with SAMME). Revision changes:

  * --scaling selects the preprocessing variant used in the ablation study:
        global   : StandardScaler fit on D_train only, applied to D_test   (A0, paper)
        per_file : each source CSV standardized with its own mean/std      (A2)
        none     : no scaling                                               (A3)
        leaky    : StandardScaler fit on D_train + D_test (deliberate leak) (A4)
  * Warnings are recorded instead of silenced (ConvergenceWarning, etc.).
  * Diagnostics per model: n_iter_ (LR, MLP), final loss (MLP), number of fitted
    estimators (AdaBoost), training and prediction time.
  * y_true / y_pred are saved for every model and task (.npz) so all metrics can
    be recomputed from the raw predictions.
  * Software versions are stored in the results file.
  * --reference_json compares the new results with a previous results file.
  * --lr_max_iter / --mlp_max_iter: convergence sensitivity analysis only; the defaults
    (100 / 200) are the configuration of the reference study used in the paper.
  * --seed: random_state of the models only (data shuffle stays at 42). Default 42 = paper.
    Used to measure seed-to-seed variability.
  * Models are fitted inside joblib's threading backend so that warnings raised during
    fit (e.g. LogisticRegression's ConvergenceWarning, which otherwise happens in a worker
    process and only reaches the log) are recorded. Results are identical.

Label encoding: LabelEncoder (alphabetical order), fit on D_train only. The class
order used in the confusion matrices is stored as "class_names" in the results.

Usage:
    python 04_run_experiment.py --train CICIoMT2024_train.csv --test CICIoMT2024_test.csv \
        --scaling global --results_dir results --reference_json multitask_results_20251206_115420.json
"""

import argparse
import json
import logging
import os
import platform
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_config
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMNS = ["Class", "Category", "Attack"]
META_COLUMNS = ["source_file"]          # present only if the CSVs were rebuilt with it
TASKS = {"binary": "Class", "6class": "Category", "19class": "Attack"}
MODEL_NAMES = ["RandomForest", "AdaBoost", "LogisticRegression", "DNN_MLP"]
SCALING_VARIANTS = ["global", "per_file", "none", "leaky"]
SEED = 42


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_dataset(file_path, dataset_type="Train"):
    """Identical to the original loader: label clean-up, inf -> NaN, dropna, shuffle(42)."""
    logger.info(f"Loading {dataset_type} data from: {file_path}")
    df = pd.read_csv(file_path)
    for col in ["Category", "Attack"]:
        if col in df.columns:
            df[col] = df[col].replace(["–", "-"], "Benign")
    n_before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    logger.info(f"{dataset_type}: {len(df)} rows after cleaning ({n_dropped} dropped).")
    return df, n_dropped


def feature_columns(df):
    return [c for c in df.columns if c not in TARGET_COLUMNS + META_COLUMNS]


def per_file_standardize(X, groups):
    """Standardize each group (source CSV) with its own mean and std (ddof=0, like StandardScaler)."""
    g = X.groupby(groups.values)
    mu = g.transform("mean")
    sd = g.transform("std", ddof=0)
    sd = sd.mask(sd == 0, 1.0)          # StandardScaler convention for constant features
    return ((X - mu) / sd).to_numpy(dtype=np.float64)


def scale(X_train, X_test, variant, src_train=None, src_test=None):
    if variant == "global":
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test)
    if variant == "none":
        return X_train.to_numpy(dtype=np.float64), X_test.to_numpy(dtype=np.float64)
    if variant == "leaky":
        scaler = StandardScaler().fit(pd.concat([X_train, X_test], axis=0))
        return scaler.transform(X_train), scaler.transform(X_test)
    if variant == "per_file":
        if src_train is None or src_test is None:
            raise ValueError("per_file scaling needs a 'source_file' column. Rebuild D_train/D_test "
                             "with the updated 02_combine_train.py / 03_combine_test.py.")
        return per_file_standardize(X_train, src_train), per_file_standardize(X_test, src_test)
    raise ValueError(f"Unknown scaling variant: {variant}")


# ----------------------------------------------------------------------------
# Models (configuration of the reference study, as in ml_experiment_make.py)
# ----------------------------------------------------------------------------
def _sklearn_version():
    return tuple(int(p) for p in sklearn.__version__.split(".")[:2])


def get_configured_models(lr_max_iter=100, mlp_max_iter=200, seed=SEED, class_weight=None):
    # class_weight: None = reference study (default). 'balanced' is a sensitivity analysis only (R3-6),
    # applied to RF and LR, the two models whose scikit-learn class accepts it directly.
    # AdaBoost: the results in the paper were produced with algorithm='SAMME'.
    # From scikit-learn 1.6 SAMME is the only algorithm and the parameter is deprecated;
    # from 1.8 the parameter no longer exists. Passing it only when needed keeps SAMME everywhere.
    ada_kwargs = dict(estimator=DecisionTreeClassifier(max_depth=None), n_estimators=50,
                      learning_rate=1.0, random_state=seed)
    if _sklearn_version() < (1, 6):
        ada_kwargs["algorithm"] = "SAMME"

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, criterion="gini", min_samples_split=2, min_samples_leaf=1,
            min_weight_fraction_leaf=0.0, max_features="sqrt", min_impurity_decrease=0.0,
            bootstrap=True, oob_score=False, warm_start=False, ccp_alpha=0.0,
            n_jobs=-1, random_state=seed, class_weight=class_weight),
        "AdaBoost": AdaBoostClassifier(**ada_kwargs),
        "LogisticRegression": LogisticRegression(
            penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True,
            intercept_scaling=1, solver="lbfgs", max_iter=lr_max_iter, warm_start=False,
            n_jobs=-1, random_state=seed, class_weight=class_weight),
        "DNN_MLP": MLPClassifier(
            hidden_layer_sizes=(32, 32, 32), solver="adam", alpha=0.0001, batch_size="auto",
            learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=mlp_max_iter,
            shuffle=True, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
            epsilon=1e-08, n_iter_no_change=10, random_state=seed),
    }


def model_diagnostics(model):
    d = {}
    if hasattr(model, "n_iter_"):
        n_iter = model.n_iter_
        d["n_iter"] = np.asarray(n_iter).tolist() if np.ndim(n_iter) else int(n_iter)
    if hasattr(model, "max_iter"):
        d["max_iter"] = model.max_iter
    if hasattr(model, "loss_"):
        d["final_loss"] = float(model.loss_)
    if isinstance(model, AdaBoostClassifier):
        d["n_estimators_fitted"] = len(model.estimators_)
        d["n_estimators_requested"] = model.n_estimators
    return d


def environment_info():
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "cpu_count_node": os.cpu_count(),
        "cpus_allocated": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "hostname": platform.node(),
    }


def fit_recording_warnings(model, X, y):
    """Fit inside joblib's threading backend so warnings from parallel workers are recorded.
    Returns (train_time_sec, list_of_warning_strings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with parallel_config(backend="threading"):
            t0 = time.time()
            model.fit(X, y)
            train_time = time.time() - t0
    return train_time, [f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "prec_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "prec_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def run(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.scaling
    if args.lr_max_iter != 100 or args.mlp_max_iter != 200:
        tag += f"_lrit{args.lr_max_iter}_mlpit{args.mlp_max_iter}"
    if args.class_weight != "none":
        tag += f"_cw{args.class_weight}"
    tag += f"_seed{args.seed}"   # always explicit, so runs with different seeds never share a name
    run_dir = os.path.join(args.results_dir, f"{tag}_{timestamp}")
    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    train_df, drop_tr = load_dataset(args.train, "Train")
    test_df, drop_te = load_dataset(args.test, "Test")
    feats = feature_columns(train_df)
    if feature_columns(test_df) != feats:
        raise ValueError("Train and test feature columns differ.")
    logger.info(f"{len(feats)} features. Scaling variant: {args.scaling}")

    src_train = train_df["source_file"] if "source_file" in train_df.columns else None
    src_test = test_df["source_file"] if "source_file" in test_df.columns else None
    X_train_s, X_test_s = scale(train_df[feats], test_df[feats], args.scaling, src_train, src_test)
    y_train_all = {t: train_df[c] for t, c in TASKS.items()}
    y_test_all = {t: test_df[c] for t, c in TASKS.items()}
    n_train, n_test = len(train_df), len(test_df)
    del train_df, test_df

    results = {
        "experiment_info": {
            "timestamp": timestamp, "scaling": args.scaling,
            "lr_max_iter": args.lr_max_iter, "mlp_max_iter": args.mlp_max_iter,
            "model_seed": args.seed, "data_shuffle_seed": SEED, "class_weight": args.class_weight,
            "requested": {"tasks": list(args.tasks), "models": list(args.models)},
            "status": "running",
            "train_samples": n_train, "test_samples": n_test,
            "train_rows_dropped": drop_tr, "test_rows_dropped": drop_te,
            "n_features": len(feats), "features": feats,
            "environment": environment_info(),
        },
        "results": {},
    }

    for task in args.tasks:
        logger.info(f"===== TASK {task} =====")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_all[task])
        known = set(le.classes_)
        mask = y_test_all[task].isin(known).to_numpy()
        if not mask.all():
            logger.warning(f"[{task}] {int((~mask).sum())} test rows with unseen classes dropped.")
        X_test_task = X_test_s[mask]
        y_test = le.transform(y_test_all[task][mask])
        class_names = le.classes_.tolist()
        results["results"][task] = {}

        models = get_configured_models(args.lr_max_iter, args.mlp_max_iter, args.seed,
                                       None if args.class_weight == "none" else args.class_weight)
        for name in args.models:
            model = models[name]
            logger.info(f"[{task}] Training {name}...")
            train_time, warn_list = fit_recording_warnings(model, X_train_s, y_train)
            t0 = time.time()
            y_pred = model.predict(X_test_task)
            pred_time = time.time() - t0

            metrics = compute_metrics(y_test, y_pred)
            metrics["train_time_sec"] = train_time
            metrics["predict_time_sec"] = pred_time

            results["results"][task][name] = {
                "metrics": metrics,
                "diagnostics": model_diagnostics(model),
                "warnings": warn_list,
                "class_names": class_names,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0),
            }
            np.savez_compressed(os.path.join(pred_dir, f"{task}_{name}.npz"),
                                y_true=y_test, y_pred=y_pred, class_names=np.array(class_names))
            logger.info(f"[{task}] {name}: acc={metrics['accuracy']:.4f} "
                        f"macroP={metrics['prec_macro']:.4f} macroR={metrics['recall_macro']:.4f} "
                        f"macroF1={metrics['f1_macro']:.4f} train={train_time:.0f}s "
                        f"warnings={len(warn_list)}")

            # Save after every model so a crash does not lose finished work
            with open(os.path.join(run_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2, default=float)

    results["experiment_info"]["status"] = "complete"
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved in {run_dir}")
    if args.reference_json:
        compare_with_reference(results, args.reference_json)


def compare_with_reference(results, reference_path):
    """Print metric differences and confusion-matrix equality against a previous results file."""
    with open(reference_path) as f:
        ref = json.load(f)
    logger.info(f"Comparison with reference: {reference_path}")
    keys = ["accuracy", "prec_macro", "recall_macro", "f1_macro"]
    for task, models in results["results"].items():
        for name, res in models.items():
            r = ref.get("results", {}).get(task, {}).get(name)
            if r is None:
                continue
            same_cm = res["confusion_matrix"] == r["confusion_matrix"]
            deltas = " ".join(f"{k}:{res['metrics'][k] - r['metrics'][k]:+.5f}" for k in keys)
            logger.info(f"  {task:8s} {name:18s} identical_CM={same_cm} {deltas}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="Consolidated D_train CSV")
    p.add_argument("--test", required=True, help="Consolidated D_test CSV")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--scaling", choices=SCALING_VARIANTS, default="global")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    p.add_argument("--reference_json", default=None, help="Previous results file to compare against")
    p.add_argument("--lr_max_iter", type=int, default=100, help="Sensitivity analysis only (default = reference study)")
    p.add_argument("--mlp_max_iter", type=int, default=200, help="Sensitivity analysis only (default = reference study)")
    p.add_argument("--class_weight", choices=["none", "balanced"], default="none",
                   help="Sensitivity analysis only (R3-6). Applies to RandomForest and LogisticRegression.")
    p.add_argument("--seed", type=int, default=SEED, help="Model random_state (default 42 = paper). Data shuffle stays at 42.")
    a = p.parse_args()
    if a.class_weight != "none":
        bad = [m for m in a.models if m not in ("RandomForest", "LogisticRegression")]
        if bad:
            p.error(f"--class_weight {a.class_weight} only applies to RandomForest and LogisticRegression; "
                    f"remove {bad} from --models")
    return a


if __name__ == "__main__":
    run(parse_args())
