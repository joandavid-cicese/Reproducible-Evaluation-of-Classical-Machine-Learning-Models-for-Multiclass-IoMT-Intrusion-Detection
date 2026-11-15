"""
06_bootstrap_ci.py
------------------
Percentile bootstrap confidence intervals for test-set metrics, computed from the
predictions saved by 04_run_experiment.py (predictions/<task>_<model>.npz).

The test set is resampled with replacement (same size) B times; for each resample
accuracy, macro precision, macro recall and macro F1 are recomputed.

Usage:
    python 06_bootstrap_ci.py --run_dir results/e1/global_20261106_101500 --n_boot 1000
Output:
    <run_dir>/bootstrap_ci.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

SEED = 42


def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(cm.sum(0) > 0, tp / cm.sum(0), 0.0)   # zero_division=0, as in 04
        rec = np.where(cm.sum(1) > 0, tp / cm.sum(1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return tp.sum() / cm.sum(), prec.mean(), rec.mean(), f1.mean()


def bootstrap(y_true, y_pred, n_classes, n_boot, rng):
    n = len(y_true)
    out = np.empty((n_boot, 4))
    codes = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = np.bincount(codes[idx], minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        out[b] = metrics_from_cm(cm)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    a = p.parse_args()
    rng = np.random.default_rng(SEED)
    rows = []
    for path in sorted(glob.glob(os.path.join(a.run_dir, "predictions", "*.npz"))):
        d = np.load(path, allow_pickle=False)
        task, model = os.path.basename(path)[:-4].split("_", 1)
        y_true, y_pred = d["y_true"], d["y_pred"]
        n_classes = len(d["class_names"])
        point = metrics_from_cm(confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)))
        boots = bootstrap(y_true, y_pred, n_classes, a.n_boot, rng)
        lo, hi = np.percentile(boots, [100 * a.alpha / 2, 100 * (1 - a.alpha / 2)], axis=0)
        for k, name in enumerate(["accuracy", "prec_macro", "recall_macro", "f1_macro"]):
            rows.append({"task": task, "model": model, "metric": name, "point": point[k],
                         "ci_low": lo[k], "ci_high": hi[k], "n_boot": a.n_boot})
        print(f"{task:8s} {model:18s} macroF1={point[3]:.4f} [{lo[3]:.4f}, {hi[3]:.4f}]")
    out = os.path.join(a.run_dir, "bootstrap_ci.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
