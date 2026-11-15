"""
09_paper_tables.py
------------------
Regenerates every table of the manuscript directly from the saved results, so that no
number is transcribed by hand. Each confusion matrix is rebuilt from the saved predictions
(y_true / y_pred), cross-checked against results.json, and its metrics are recomputed.

Steps
  1. Model selection by cross-validation ONLY (mean macro-F1 over the CV folds of D_train).
     Written to selection_by_cv.json. Note: this documents the selection rule; it does not
     certify that the test set was never inspected before the rule was formalized.
  2. Class mapping table (Table 2): presentation order, LabelEncoder index, train/test counts.
  3. Confusion matrices for every task/model in PRESENTATION order (rows = true class,
     columns = predicted class). Rows and columns are reordered together, programmatically.
  4. Main metrics table: test metrics + bootstrap 95% CI + CV mean ± sd, selected model marked.
  5. Per-class precision / recall / F1 / support (6class and 19class).
  6. Ablation (macro-F1 by scaling variant) and convergence tables.
  7. Reference comparison: published and preprint values of [5] and this work (no scaling =
     reconstruction from what [5] documents; global = this pipeline) shown as SEPARATE evidence,
     plus differences in percentage points and relative %, rounded to the published precision.
  8. Seed study (if results/seeds exists): mean ± sd per scaling, and paired differences
     global − no scaling per seed.
  Runs are grouped by full configuration (scaling, seed, max_iter); a configuration run more than
  once is reported in checks.json ("duplicate_runs"), never silently replaced.

Usage:
    python src/09_paper_tables.py --results_root results --out results/paper_tables
Outputs: CSV + LaTeX (.tex) per table, selection_by_cv.json and checks.json.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import importlib.util

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "rep08", os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_report.py"))
rep08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep08)

TASKS = ["binary", "6class", "19class"]
MODELS = ["RandomForest", "AdaBoost", "LogisticRegression", "DNN_MLP"]
MODEL_LABEL = {"RandomForest": "RF", "AdaBoost": "AdaBoost", "LogisticRegression": "LR", "DNN_MLP": "MLP"}

# Presentation order used in the manuscript (Table 2). Benign is shown last in 19class.
# Code indices follow sklearn's LabelEncoder (alphabetical); the mapping is in Table 2.
PRESENTATION_ORDER = {
    "binary": ["ATTACK", "BENIGN"],
    "6class": ["Benign", "DDoS", "DoS", "MQTT", "RECON", "SPOOFING"],
    "19class": ["ARP spoofing", "DDoS ICMP", "DDoS SYN", "DDoS TCP", "DDoS UDP",
                "DDoS connect flood", "DDoS publish flood", "DoS ICMP", "DoS SYN", "DoS TCP",
                "DoS UDP", "DoS connect flood", "DoS publish flood", "Malformed data", "OS scan",
                "Ping sweep", "Port scan", "Recon VulScan", "Benign"],
}
TASK_COLUMN = {"binary": "Class", "6class": "Category", "19class": "Attack"}


# ----------------------------------------------------------------- helpers
ORDER_ISSUES = []
PER_CLASS = {}   # per-class metrics recomputed from the saved predictions


def pres_order(task, enc):
    """Presentation order restricted to the classes present; unknown classes appended and reported."""
    order = [c for c in PRESENTATION_ORDER[task] if c in enc]
    extra = [c for c in enc if c not in PRESENTATION_ORDER[task]]
    missing = [c for c in PRESENTATION_ORDER[task] if c not in enc]
    if extra or missing:
        issue = {"task": task, "classes_not_in_presentation_order": extra, "expected_but_absent": missing}
        if issue not in ORDER_ISSUES:
            ORDER_ISSUES.append(issue)
    return order + extra


def latest(pattern):
    paths = sorted(glob.glob(pattern), key=lambda p: os.path.basename(os.path.dirname(p))[-15:])
    return paths[-1] if paths else None


def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    prec = np.where(cm.sum(0) > 0, tp / np.where(cm.sum(0) > 0, cm.sum(0), 1), 0.0)
    rec = np.where(cm.sum(1) > 0, tp / np.where(cm.sum(1) > 0, cm.sum(1), 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0.0)
    return {"accuracy": tp.sum() / cm.sum(), "prec_macro": prec.mean(), "recall_macro": rec.mean(),
            "f1_macro": f1.mean()}, prec, rec, f1


def to_latex(df, caption, label, floatfmt="{:.4f}", int_cols=()):
    cols = list(df.columns)
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}",
             "\\toprule", " & ".join(str(c).replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\",
             "\\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in int_cols and pd.notna(v):
                cells.append(f"{int(v):,}".replace(",", "{,}"))
            elif isinstance(v, (float, np.floating)):
                cells.append("--" if pd.isna(v) else floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}".replace(",", "{,}"))
            else:
                cells.append(str(v).replace("_", "\\_").replace("%", "\\%"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def save(df, out, name, caption=None, label=None, **kw):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    if caption:
        with open(os.path.join(out, f"{name}.tex"), "w") as f:
            f.write(to_latex(df, caption, label or f"tab:{name}", **kw))


# ----------------------------------------------------------------- steps
def select_by_cv(root, out):
    files = sorted(glob.glob(os.path.join(root, "cv", "folds", "*.json")))
    if not files:
        print("No CV folds found: selection skipped.")
        return None, None
    rows = []
    for f in files:
        r = json.load(open(f))
        rows.append({"task": r["task"], "model": r["model"], "fold": r["fold"],
                     "f1_macro": r["metrics"]["f1_macro"], "n_splits": r["n_splits"]})
    df = pd.DataFrame(rows)
    agg = df.groupby(["task", "model"]).agg(cv_f1_mean=("f1_macro", "mean"), cv_f1_sd=("f1_macro", "std"),
                                            folds=("fold", "count"), n_splits=("n_splits", "max")).reset_index()
    sel = {}
    for t in TASKS:
        a = agg[agg.task == t].sort_values("cv_f1_mean", ascending=False)
        if len(a) == 0:
            continue
        complete = bool((a.folds == a.n_splits).all())
        sel[t] = {"selected_model": a.iloc[0]["model"], "complete_cv": complete,
                  "ranking": [{"model": r.model, "cv_f1_mean": r.cv_f1_mean, "cv_f1_sd": r.cv_f1_sd,
                               "folds": int(r.folds)} for r in a.itertuples()]}
    digest = hashlib.sha256(b"".join(open(f, "rb").read() for f in files)).hexdigest()
    doc = {"rule": "Per task, the model with the highest mean macro-F1 over the stratified CV folds "
                   "of D_train is selected. Only CV results are read; no test-set result is used.",
           "note": "This file formalizes the selection rule. It does not certify that test results "
                   "were never inspected before the rule was adopted.",
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "cv_fold_files": len(files),
           "cv_fold_files_sha256": digest, "selection": sel}
    with open(os.path.join(out, "selection_by_cv.json"), "w") as f:
        json.dump(doc, f, indent=2)
    return {t: v["selected_model"] for t, v in sel.items()}, agg


def class_mapping(root, e1, out):
    val_path = os.path.join(root, "validation", "validation.json")
    val = json.load(open(val_path)) if os.path.exists(val_path) else None
    rows = []
    for t in ["6class", "19class"]:
        enc = e1["results"][t]["RandomForest"]["class_names"]
        for i, name in enumerate(pres_order(t, enc)):
            row = {"task": t, "presentation_index": i, "class": name, "labelencoder_index": enc.index(name)}
            if val:
                col = TASK_COLUMN[t]
                row["train_count"] = val["train"]["label_counts"][col]["paper"].get(name)
                row["test_count"] = val["test"]["label_counts"][col]["paper"].get(name)
            rows.append(row)
    df = pd.DataFrame(rows)
    save(df[df.task == "19class"].drop(columns=["task"]), out, "table2_class_mapping_19class",
         "Class mapping for the 19-class task. Presentation order (used in all tables) and the "
         "LabelEncoder index used in the code (alphabetical order). Counts after cleaning.",
         "tab:class_mapping", int_cols=("train_count", "test_count"))
    save(df[df.task == "6class"].drop(columns=["task"]), out, "class_mapping_6class")
    return df


def confusion_tables(run_dir, res, out, checks):
    recomputed = {}
    for t in TASKS:
        for m in MODELS:
            npz = os.path.join(run_dir, "predictions", f"{t}_{m}.npz")
            if not os.path.exists(npz):
                continue
            d = np.load(npz, allow_pickle=False)
            enc = [str(c) for c in d["class_names"]]
            k = len(enc)
            cm_enc = np.bincount(d["y_true"].astype(np.int64) * k + d["y_pred"].astype(np.int64),
                                 minlength=k * k).reshape(k, k)
            r = res["results"][t][m]
            ok_json = cm_enc.tolist() == r["confusion_matrix"]
            met, pc_p, pc_r, pc_f = metrics_from_cm(cm_enc)
            recomputed[(t, m)] = met
            PER_CLASS[(t, m)] = {c: {"precision": pc_p[i], "recall": pc_r[i], "f1-score": pc_f[i],
                                     "support": int(cm_enc[i].sum())} for i, c in enumerate(enc)}
            cr = r["classification_report"]
            ok_pc = all(abs(PER_CLASS[(t, m)][c][k] - cr[c][k]) < 1e-12 for c in enc
                        for k in ("precision", "recall", "f1-score"))
            ok_metrics = all(abs(met[x] - r["metrics"][x]) < 1e-12 for x in met)
            names = pres_order(t, enc)
            order = [enc.index(c) for c in names]
            cm_pres = cm_enc[np.ix_(order, order)]          # rows AND columns reordered together
            ok_reorder = abs(metrics_from_cm(cm_pres)[0]["f1_macro"] - met["f1_macro"]) < 1e-12
            checks.append({"task": t, "model": m, "n": int(cm_enc.sum()), "cm_matches_json": bool(ok_json),
                           "metrics_match_json": bool(ok_metrics), "per_class_match_json": bool(ok_pc),
                           "reorder_preserves_metrics": bool(ok_reorder)})
            idx = list(range(k))
            df = pd.DataFrame(cm_pres, columns=[str(i) for i in idx])
            df.insert(0, "True / Pred", [f"{i}" for i in idx] if t == "19class" else names)
            if t != "19class":
                df.columns = ["True / Pred"] + names
            save(df, out, f"cm_{t}_{m}",
                 f"Confusion matrix, {t}, {MODEL_LABEL[m]} (test set, n = {int(cm_enc.sum()):,}). Rows: true "
                 f"class; columns: predicted class; class order as in Table 2.", f"tab:cm_{t}_{m}",
                 int_cols=tuple(df.columns[1:]))
    return recomputed


def main_table(res, boot_csv, cv_agg, selected, out, recomputed=None):
    boot = pd.read_csv(boot_csv) if boot_csv and os.path.exists(boot_csv) else None
    rows = []
    for t in TASKS:
        for m in MODELS:
            r = res["results"][t].get(m)
            if r is None:
                continue
            mt = (recomputed or {}).get((t, m), r["metrics"])   # metrics rebuilt from predictions
            row = {"Task": t, "Model": MODEL_LABEL[m] + (" *" if selected and selected.get(t) == m else ""),
                   "Accuracy": mt["accuracy"], "Macro P": mt["prec_macro"], "Macro R": mt["recall_macro"],
                   "Macro F1": mt["f1_macro"]}
            if boot is not None:
                b = boot[(boot.task == t) & (boot.model == m) & (boot.metric == "f1_macro")]
                if len(b):
                    row["F1 95% CI"] = f"[{b.ci_low.iloc[0]:.4f}, {b.ci_high.iloc[0]:.4f}]"
            if cv_agg is not None:
                c = cv_agg[(cv_agg.task == t) & (cv_agg.model == m)]
                if len(c):
                    row["CV Macro F1"] = f"{c.cv_f1_mean.iloc[0]:.4f} ± {c.cv_f1_sd.iloc[0]:.4f}"
            rows.append(row)
    df = pd.DataFrame(rows)
    save(df, out, "main_results",
         "Test-set results (single evaluation on D_test; metrics recomputed from the saved predictions) "
         "and 10-fold stratified CV on D_train. "
         "* = model selected by CV macro-F1. CI: percentile bootstrap over test observations "
         "(1,000 resamples); it does not include seed-to-seed variability.", "tab:main_results")
    return df


def per_class(res, out):
    for t in ["6class", "19class"]:
        rows = []
        enc = res["results"][t]["RandomForest"]["class_names"]
        for i, c in enumerate(pres_order(t, enc)):
            row = {"#": i, "Class": c}
            for m in MODELS:
                cr = PER_CLASS.get((t, m), res["results"][t][m]["classification_report"])[c]
                row[f"{MODEL_LABEL[m]} P"] = cr["precision"]
                row[f"{MODEL_LABEL[m]} R"] = cr["recall"]
                row[f"{MODEL_LABEL[m]} F1"] = cr["f1-score"]
                row["Support"] = int(cr["support"])
            rows.append(row)
        df = pd.DataFrame(rows)
        cols = ["#", "Class", "Support"] + [c for c in df.columns if c not in ("#", "Class", "Support")]
        save(df[cols], out, f"per_class_{t}", f"Per-class precision, recall and F1 on the test set, {t}.",
             f"tab:per_class_{t}", int_cols=("Support", "#"))


def ablation_table(root, out, dup_log):
    sel, dups = rep08.collect(root, "ablation")
    dup_log.extend(dups)
    by_var = {cfg["scaling"]: res for _, cfg, res in sel if cfg["model_seed"] == 42}
    if not by_var:
        return
    names = {"global": "Global (fit on train)", "none": "No scaling", "leaky": "Fit on train+test (diagnostic)",
             "per_file": "Per file (label-dependent, diagnostic)"}
    rows = []
    for v in ["global", "none", "leaky", "per_file"]:
        if v not in by_var:
            continue
        for t in TASKS:
            row = {"Scaling": names[v], "Task": t}
            for m in MODELS:
                row[MODEL_LABEL[m]] = by_var[v]["results"][t][m]["metrics"]["f1_macro"]
            rows.append(row)
    save(pd.DataFrame(rows), out, "ablation_scaling",
         "Macro-F1 on the test set by scaling variant (same data, models and hyperparameters; seed 42). "
         "The last two variants are methodological diagnostics, not valid deployment pipelines.",
         "tab:ablation")


def convergence_table(root, e1_path, out, dup_log):
    sel, dups = rep08.collect(root, "convergence")
    dup_log.extend(dups)
    e1 = json.load(open(e1_path))
    items = [(e1_path, rep08.run_config(e1), e1)] + sel
    rows = []
    for p, cfg, r in items:
        cpus = r["experiment_info"].get("environment", {}).get("cpus_allocated")
        for t in TASKS:
            for m in ["LogisticRegression", "DNN_MLP"]:
                x = r["results"][t].get(m)
                if x is None:
                    continue
                d = x["diagnostics"]
                n_iter = int(max(np.atleast_1d(d["n_iter"])))
                rows.append({"Task": t, "Model": MODEL_LABEL[m], "max_iter": d["max_iter"], "Iterations used": n_iter,
                             "Reached max_iter": "yes" if n_iter >= d["max_iter"] else "no",
                             "Macro F1": x["metrics"]["f1_macro"], "Train time (s)": x["metrics"]["train_time_sec"],
                             "CPUs": cpus if cpus else "n/r",
                             "Warnings recorded in run": rep08.warnings_field(x, cfg)})
    df = pd.DataFrame(rows)
    df = df.sort_values(["Model", "Task", "max_iter"], key=lambda s: s.map({t: i for i, t in enumerate(TASKS)})
                        if s.name == "Task" else s)
    save(df, out, "convergence",
         "Convergence sensitivity. 'Reached max_iter' = optimizer stopped at the iteration limit. "
         "Warnings: recorded during fitting (code v2) or not recorded (v1; recovered separately from the "
         "SLURM logs, see report). Training times depend on the CPUs allocated to each job and are not "
         "directly comparable across rows with different allocations.", "tab:convergence", floatfmt="{:.4f}")


def reference_comparison(root, res, out, dup_log):
    path = os.path.join("reference", "baseline_published.csv")
    ours = {"This work, global scaling (E1, seed 42)": res}
    abl, dups = rep08.collect(root, "ablation")
    dup_log.extend(dups)
    for _, cfg, r in abl:
        if cfg["scaling"] == "none" and cfg["model_seed"] == 42:
            ours["This work, no scaling (reconstruction from what [5] documents, seed 42)"] = r
    seeds, dups = rep08.collect(root, "seeds")
    dup_log.extend(dups)
    rows, diffs = [], []
    base = pd.read_csv(path) if os.path.exists(path) else None
    if base is None:
        print(f"{path} not found: published values not included.")
    keys = set()
    if base is not None:
        b = base.dropna(subset=["value"])
        for x in b.itertuples():
            keys.add((x.task, x.model, x.metric))
            rows.append({"Task": x.task, "Model": MODEL_LABEL[x.model], "Metric": x.metric,
                         "Evidence": f"{x.source} ({x.source_version})", "Value": f"{x.value:.{int(x.decimals)}f}",
                         "Averaging": x.averaging})
    if not keys:
        keys = {(t, "RandomForest", "f1_macro") for t in TASKS}
    for (t, m, metric) in sorted(keys):
        for label, r in ours.items():
            rows.append({"Task": t, "Model": MODEL_LABEL[m], "Metric": metric, "Evidence": label,
                         "Value": f"{r['results'][t][m]['metrics'][metric]:.4f}", "Averaging": "macro"})
        for scaling in ["global", "none"]:
            vals = [r["results"][t][m]["metrics"][metric] for _, cfg, r in seeds if cfg["scaling"] == scaling]
            if len(vals) > 1:
                rows.append({"Task": t, "Model": MODEL_LABEL[m], "Metric": metric,
                             "Evidence": f"This work, {scaling}, {len(vals)} seeds (mean ± sd)",
                             "Value": f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}", "Averaging": "macro"})
    if base is not None:
        for x in base.dropna(subset=["value"]).itertuples():
            if "published" not in str(x.source_version).lower():
                continue
            dec = int(x.decimals)
            for label, r in ours.items():
                v = r["results"][x.task][x.model]["metrics"][x.metric]
                diffs.append({"Task": x.task, "Model": MODEL_LABEL[x.model], "Metric": x.metric,
                              "Published [5]": x.value, "This work": label, "Value": round(v, 4),
                              "Difference (pp)": round((v - x.value) * 100, max(dec - 2, 1)),
                              "Relative (%)": round((v - x.value) / x.value * 100, 1),
                              "Published averaging": x.averaging})
    order_ev = lambda e: (0 if "published" in e else 1 if "preprint" in e else 2 if "no scaling" in e else 3)
    df = pd.DataFrame(rows)
    if len(df):
        df["_o"] = df["Evidence"].map(order_ev)
        df = df.sort_values(["Task", "Model", "Metric", "_o"]).drop(columns="_o")
        save(df, out, "reference_comparison",
             "Reference values of [5] (published and preprint versions) and the results of this work, "
             "reported as separate evidence. The no-scaling run reconstructs only what [5] documents; it is "
             "not an exact reproduction of [5].", "tab:reference_comparison")
    if diffs:
        save(pd.DataFrame(diffs), out, "comparison_published",
             "Difference with the published values of [5], rounded to the precision of the published value.",
             "tab:comparison_published", floatfmt="{:.3f}")


def seeds_tables(root, out, dup_log):
    ind, summ, diff, eq = rep08.section_seeds(root, None)
    dup_log.extend(rep08.DUPLICATES)
    if ind is None:
        return
    ind.to_csv(os.path.join(out, "seeds_individual.csv"), index=False)
    t = summ.copy()
    t["Macro F1 (mean ± sd)"] = t.apply(lambda r: f"{r.f1_macro_mean:.4f} ± {r.f1_macro_std:.4f}", axis=1)
    t = t.rename(columns={"scaling": "Scaling", "task": "Task", "model": "Model", "f1_macro_min": "Min",
                          "f1_macro_max": "Max", "n_seeds": "Seeds"})
    t["Model"] = t["Model"].map(MODEL_LABEL)
    save(t[["Scaling", "Task", "Model", "Seeds", "Macro F1 (mean ± sd)", "Min", "Max"]], out, "seeds_summary",
         "Seed-to-seed variability on the official test set: model random_state varied, data and row order "
         "fixed. LR (lbfgs) is deterministic.", "tab:seeds")
    if diff is not None:
        d = diff.rename(columns={"task": "Task", "model": "Model", "mean": "Mean diff", "std": "SD",
                                 "min": "Min", "max": "Max", "n_seeds": "Seeds", "n_positive": "Seeds with diff > 0"})
        d["Model"] = d["Model"].map(MODEL_LABEL)
        save(d, out, "seeds_diff_global_minus_none",
             "Macro-F1 difference global scaling minus no scaling, paired by seed.", "tab:seeds_diff")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--e1_dir", default=None, help="Run to tabulate (default: latest results/e1/*)")
    ap.add_argument("--out", default="results/paper_tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # Remove this script's previous outputs so no stale table survives a re-run
    for f in glob.glob(os.path.join(a.out, "*.csv")) + glob.glob(os.path.join(a.out, "*.tex")) + \
            [os.path.join(a.out, x) for x in ("checks.json", "selection_by_cv.json")]:
        if os.path.exists(f):
            os.remove(f)
    e1_json = os.path.join(a.e1_dir, "results.json") if a.e1_dir else rep08.select_e1(a.results_root)
    if not e1_json:
        for r in rep08.INCOMPLETE:
            print("  INCOMPLETE RUN (not used):", r["run"], "missing:", r["missing"])
        sys.exit("No COMPLETE E1 run found under results/e1 (paper configuration). Nothing tabulated.")
    run_dir = os.path.dirname(e1_json)
    res = json.load(open(e1_json))
    ok, missing = rep08.completeness(res)
    if not ok:
        sys.exit(f"E1 run {e1_json} is incomplete (missing: {missing}). Nothing tabulated.")

    selected, cv_agg = select_by_cv(a.results_root, a.out)
    class_mapping(a.results_root, res, a.out)
    checks, dup_log = [], []
    recomputed = confusion_tables(run_dir, res, a.out, checks)
    main_table(res, os.path.join(run_dir, "bootstrap_ci.csv"), cv_agg, selected, a.out, recomputed)
    per_class(res, a.out)
    ablation_table(a.results_root, a.out, dup_log)
    convergence_table(a.results_root, e1_json, a.out, dup_log)
    reference_comparison(a.results_root, res, a.out, dup_log)
    seeds_tables(a.results_root, a.out, dup_log)
    uniq = {json.dumps(d, sort_keys=True): d for d in dup_log}
    with open(os.path.join(a.out, "checks.json"), "w") as f:
        json.dump({"source_run": e1_json, "checks": checks, "class_order_issues": ORDER_ISSUES,
                   "duplicate_runs": list(uniq.values()), "incomplete_runs_not_used": rep08.INCOMPLETE},
                  f, indent=2)
    for d in uniq.values():
        print("  DUPLICATE CONFIGURATION (most recent complete used):", d["config"], "->", d["used"])
    for r in rep08.INCOMPLETE:
        print("  INCOMPLETE RUN (not used):", r["run"], "missing:", r["missing"])
    bad = [c for c in checks if not (c["cm_matches_json"] and c["metrics_match_json"] and c["per_class_match_json"]
                                     and c["reorder_preserves_metrics"])]
    print(f"Source run: {e1_json}")
    print(f"Selection by CV: {selected}")
    sel_doc = json.load(open(os.path.join(a.out, "selection_by_cv.json"))) if selected else {"selection": {}}
    for t in TASKS:
        if t not in sel_doc["selection"]:
            print(f"  WARNING: no CV results for {t}; no model selected.")
        elif not sel_doc["selection"][t]["complete_cv"]:
            print(f"  WARNING: CV incomplete for {t}; selection is provisional.")
    print(f"Consistency checks: {len(checks) - len(bad)}/{len(checks)} passed")
    for issue in ORDER_ISSUES:
        print("  CLASS ORDER WARNING:", issue)
    for c in bad:
        print("  FAILED:", c)
    print(f"Tables written to {a.out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
