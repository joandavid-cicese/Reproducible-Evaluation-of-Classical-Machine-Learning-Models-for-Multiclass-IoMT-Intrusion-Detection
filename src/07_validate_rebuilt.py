"""
07_validate_rebuilt.py
----------------------
Checks that the rebuilt D_train / D_test (02/03, with `source_file`) contain exactly the
same data as the consolidated CSVs used in the paper (data/paper/). Exit code 1 if any
check fails, so a SLURM dependency (afterok) blocks the ablation.

Checks (after the same cleaning as 04: label clean-up, inf -> NaN, dropna):
  1. Feature columns identical and in the same order (`source_file` is the only extra column).
  2. Same number of rows (raw row counts are reported for information).
  3. Same label counts for Class, Category and Attack.
  4. Same multiset of rows (row hashes over features + labels, order-independent), comparing
     numbers at 10 significant digits; exact bitwise equality is reported for information.
  5. Test label counts equal the `support` values of the reference JSON (3 tasks).

Usage:
    python src/07_validate_rebuilt.py --paper_train data/paper/CICIoMT2024_train.csv \
        --paper_test data/paper/CICIoMT2024_test.csv \
        --rebuilt_train rebuilt/CICIoMT2024_train_src.csv \
        --rebuilt_test rebuilt/CICIoMT2024_test_src.csv \
        --reference_json reference/multitask_results_20251206_115420.json \
        --out results/validation/validation.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

TARGETS = ["Class", "Category", "Attack"]
META = ["source_file"]
TASKS = {"binary": "Class", "6class": "Category", "19class": "Attack"}


def load_clean(path):
    df = pd.read_csv(path)
    n_raw = len(df)
    for col in ["Category", "Attack"]:
        if col in df.columns:
            df[col] = df[col].replace(["–", "-"], "Benign")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df, n_raw


def features(df):
    return [c for c in df.columns if c not in TARGETS + META]


def round_sig(a, sig=10):
    """Round to `sig` significant digits (absorbs last-digit differences from CSV parsing)."""
    a = np.asarray(a, dtype=np.float64)
    out = a.copy()
    nz = np.isfinite(a) & (a != 0)
    mag = np.floor(np.log10(np.abs(a[nz])))
    scale = 10.0 ** (sig - 1 - mag)
    out[nz] = np.round(a[nz] * scale) / scale
    return out


def row_hash_multiset(df, cols, sig=None):
    d = df[cols].copy()
    if sig is not None:
        num = d.select_dtypes(include=[np.number]).columns
        d[num] = round_sig(d[num].to_numpy(), sig)
    h = pd.util.hash_pandas_object(d, index=False).to_numpy()
    return np.sort(h)


def compare(name, paper_path, rebuilt_path, report):
    p, p_raw = load_clean(paper_path)
    r, r_raw = load_clean(rebuilt_path)
    res = {"paper_rows_raw": p_raw, "rebuilt_rows_raw": r_raw,
           "paper_rows_clean": len(p), "rebuilt_rows_clean": len(r)}
    fp, fr = features(p), features(r)
    res["features_identical_order"] = fp == fr
    res["n_features"] = len(fp)
    if fp != fr:
        res["features_only_in_paper"] = sorted(set(fp) - set(fr))
        res["features_only_in_rebuilt"] = sorted(set(fr) - set(fp))
    res["extra_columns_rebuilt"] = [c for c in r.columns if c not in p.columns]
    res["rows_equal"] = len(p) == len(r)
    label_ok = True
    res["label_counts"] = {}
    for col in TARGETS:
        cp, cr = p[col].value_counts().to_dict(), r[col].value_counts().to_dict()
        same = cp == cr
        label_ok &= same
        res["label_counts"][col] = {"equal": same, "paper": cp, "rebuilt": cr}
    res["labels_equal"] = label_ok
    if fp == fr and len(p) == len(r):
        cols = fp + TARGETS
        res["rows_identical_exact"] = bool(np.array_equal(row_hash_multiset(p, cols),
                                                          row_hash_multiset(r, cols)))
        res["rows_identical_multiset"] = res["rows_identical_exact"] or bool(np.array_equal(
            row_hash_multiset(p, cols, sig=10), row_hash_multiset(r, cols, sig=10)))
    else:
        res["rows_identical_exact"] = False
        res["rows_identical_multiset"] = False
    res["passed"] = all([res["features_identical_order"], res["rows_equal"],
                         res["labels_equal"], res["rows_identical_multiset"]])
    report[name] = res
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper_train", required=True)
    ap.add_argument("--paper_test", required=True)
    ap.add_argument("--rebuilt_train", required=True)
    ap.add_argument("--rebuilt_test", required=True)
    ap.add_argument("--reference_json", required=True)
    ap.add_argument("--out", default="results/validation/validation.json")
    a = ap.parse_args()

    report = {}
    compare("train", a.paper_train, a.rebuilt_train, report)
    r_test = compare("test", a.paper_test, a.rebuilt_test, report)

    with open(a.reference_json) as f:
        ref = json.load(f)
    sup_ok = True
    report["test_support_vs_reference"] = {}
    for task, col in TASKS.items():
        cr = ref["results"][task]["RandomForest"]["classification_report"]
        ref_support = {k: int(v["support"]) for k, v in cr.items()
                       if isinstance(v, dict) and k not in ("macro avg", "weighted avg")}
        got = {str(k): int(v) for k, v in r_test[col].value_counts().items()}
        # binary labels may differ only in case (e.g. ATTACK/BENIGN)
        same = ref_support == got or {k.upper(): v for k, v in ref_support.items()} == \
            {k.upper(): v for k, v in got.items()}
        sup_ok &= same
        report["test_support_vs_reference"][task] = {"equal": same, "reference": ref_support, "rebuilt": got}

    report["passed"] = report["train"]["passed"] and report["test"]["passed"] and sup_ok
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, default=int)

    for part in ["train", "test"]:
        r = report[part]
        print(f"[{part}] rows clean paper={r['paper_rows_clean']} rebuilt={r['rebuilt_rows_clean']} "
              f"(raw {r['paper_rows_raw']}/{r['rebuilt_rows_raw']}) | features same order={r['features_identical_order']} "
              f"| labels equal={r['labels_equal']} | identical rows (10 sig. digits)={r['rows_identical_multiset']} "
              f"| bitwise={r['rows_identical_exact']}")
    for task, v in report["test_support_vs_reference"].items():
        print(f"[test support vs reference JSON] {task}: {v['equal']}")
    print("VALIDATION PASSED" if report["passed"] else "VALIDATION FAILED -> see " + a.out)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
