"""
08_report.py
------------
Aggregates every result available under results/ into tables (CSV), a JSON summary and a
Markdown report. It can be run at any time: missing phases are reported as pending.

    python src/08_report.py --results_root results \
        --reference_json reference/multitask_results_20251206_115420.json

Outputs (results/report/): REPORT.md, summary.json and one CSV per table.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

TASK_ORDER = ["binary", "6class", "19class"]
MODEL_ORDER = ["RandomForest", "AdaBoost", "LogisticRegression", "DNN_MLP"]
VARIANT_ORDER = ["global", "per_file", "none", "leaky"]
MAIN = ["accuracy", "prec_macro", "recall_macro", "f1_macro", "f1_weighted"]


# ---------------------------------------------------------------- helpers
def load_json(path):
    with open(path) as f:
        return json.load(f)


def runs(pattern):
    """results.json files matching pattern, oldest first (directory names end in a timestamp)."""
    return sorted(glob.glob(pattern), key=lambda p: os.path.basename(os.path.dirname(p))[-15:])


def md_table(df, floatfmt="{:.4f}"):
    if df is None or len(df) == 0:
        return "_pendiente_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def order(df):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if "task" in df:
        df["task"] = pd.Categorical(df["task"], TASK_ORDER, ordered=True)
    if "model" in df:
        df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    if "variant" in df:
        df["variant"] = pd.Categorical(df["variant"], VARIANT_ORDER, ordered=True)
    keys = [k for k in ["task", "model", "variant", "max_iter"] if k in df]
    return df.sort_values(keys).reset_index(drop=True)


def metric_rows(res, extra=None):
    rows = []
    for task, models in res["results"].items():
        for model, r in models.items():
            row = {"task": task, "model": model}
            row.update({k: r["metrics"].get(k) for k in MAIN})
            if extra:
                row.update(extra)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- run collection
CONFIG_KEYS = ["scaling", "model_seed", "lr_max_iter", "mlp_max_iter", "class_weight"]


def run_config(res):
    ei = res["experiment_info"]
    return {"scaling": ei.get("scaling", "global"), "model_seed": ei.get("model_seed", 42),
            "lr_max_iter": ei.get("lr_max_iter", 100), "mlp_max_iter": ei.get("mlp_max_iter", 200),
            "class_weight": ei.get("class_weight", "none"),
            "code_version": "v2" if "model_seed" in ei else "v1"}


def completeness(res):
    """(complete, missing). Runs from code >= v2.2 record what was requested and a final status;
    for older runs the expected set is inferred: every task x every model that appears in the run
    (04 loops models inside tasks, so a cancelled run always lacks some task/model pair)."""
    ei = res["experiment_info"]
    got = {(t, m) for t, ms in res["results"].items() for m in ms}
    req = ei.get("requested")
    if req:
        expected = {(t, m) for t in req["tasks"] for m in req["models"]}
    else:
        models = {m for _, m in got}
        expected = {(t, m) for t in TASK_ORDER for m in models}
    missing = sorted(expected - got)
    status_ok = ei.get("status", "complete") == "complete"
    return (not missing and status_ok and len(got) > 0), missing


INCOMPLETE = []


def collect(root, stage):
    """All runs of a stage (e1, ablation, convergence, seeds) grouped by full configuration.
    Returns (runs, duplicates). Per configuration the most recent COMPLETE run is used; incomplete
    runs (cancelled or crashed jobs) are reported and only used if no complete run exists.
    Duplicates are REPORTED, never silently dropped."""
    paths = runs(os.path.join(root, stage, "*", "results.json"))
    by_cfg = {}
    for p in paths:
        r = load_json(p)
        cfg = run_config(r)
        ok, missing = completeness(r)
        if not ok:
            rec = {"stage": stage, "run": p, "pairs_present": sum(len(v) for v in r["results"].values()),
                   "missing": "; ".join(f"{t}/{m}" for t, m in missing) or "status != complete"}
            if rec not in INCOMPLETE:
                INCOMPLETE.append(rec)
        key = tuple(cfg[k] for k in CONFIG_KEYS)
        by_cfg.setdefault(key, []).append((p, cfg, r, ok))
    selected, dups = [], []
    for k, v in by_cfg.items():
        complete = [x for x in v if x[3]]
        chosen = complete[-1] if complete else None
        if chosen:
            selected.append(chosen[:3])
        if len(v) > 1:
            dups.append({"stage": stage, "config": dict(zip(CONFIG_KEYS, k)),
                         "runs": [x[0] + ("" if x[3] else " (INCOMPLETE)") for x in v],
                         "used": chosen[0] if chosen else "none (no complete run)"})
    return selected, dups


E1_CONFIG = {"scaling": "global", "model_seed": 42, "lr_max_iter": 100, "mlp_max_iter": 200, "class_weight": "none"}


def select_e1(root):
    """Path of the E1 run to use: most recent complete run with the paper configuration."""
    sel, dups = collect(root, "e1")
    for d in dups:
        if d not in DUPLICATES:
            DUPLICATES.append(d)
    cands = [p for p, cfg, _ in sel if all(cfg[k] == v for k, v in E1_CONFIG.items())]
    return cands[-1] if cands else None


def warnings_field(r, cfg):
    if cfg["code_version"] == "v1":
        return "n/r (v1: no registradas; ver logs)"
    cats = Counter(w.split(":")[0] for w in r.get("warnings", []))
    return "; ".join(f"{k} x{v}" for k, v in cats.items()) if cats else "0"


# ---------------------------------------------------------------- sections
def section_e1(root, ref):
    path = select_e1(root)
    if not path:
        return None, None, None
    res = load_json(path)
    comp = []
    for task, models in res["results"].items():
        for model, r in models.items():
            rr = ref["results"].get(task, {}).get(model)
            if rr is None:
                continue
            row = {"task": task, "model": model,
                   "identical_CM": r["confusion_matrix"] == rr["confusion_matrix"]}
            for k in ["accuracy", "prec_macro", "recall_macro", "f1_macro"]:
                row[f"delta_{k}"] = r["metrics"][k] - rr["metrics"][k]
            comp.append(row)
    diag = []
    cfg = run_config(res)
    for task, models in res["results"].items():
        for model, r in models.items():
            d = r.get("diagnostics", {})
            n_iter = d.get("n_iter")
            diag.append({
                "task": task, "model": model,
                "train_time_sec": r["metrics"].get("train_time_sec"),
                "predict_time_sec": r["metrics"].get("predict_time_sec"),
                "n_iter": json.dumps(n_iter) if n_iter is not None else "",
                "max_iter": d.get("max_iter", ""),
                "hit_max_iter": (max(np.atleast_1d(n_iter)) >= d["max_iter"]) if n_iter is not None and "max_iter" in d else "",
                "adaboost_estimators_fitted": d.get("n_estimators_fitted", ""),
                "cpus_allocated": res["experiment_info"].get("environment", {}).get("cpus_allocated", "n/r"),
                "warnings_recorded": warnings_field(r, cfg),
            })
    return path, order(pd.DataFrame(comp)), order(pd.DataFrame(diag))


def section_ablation(root):
    sel, dups = collect(root, "ablation")
    DUPLICATES.extend(dups)
    latest = {cfg["scaling"]: (p, res) for p, cfg, res in sel if cfg["model_seed"] == 42}
    rows = []
    for v, (p, res) in latest.items():
        rows += metric_rows(res, {"variant": v})
    if not rows:
        return None, None, {}
    df = order(pd.DataFrame(rows))
    pivot = df.pivot_table(index=["task", "model"], columns="variant", values="f1_macro", observed=True)
    pivot = pivot.reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    return df, pivot, {v: p for v, (p, _) in latest.items()}


def section_convergence(root, e1_path):
    rows = []
    sel, dups = collect(root, "convergence")
    DUPLICATES.extend(dups)
    items = ([(e1_path, run_config(load_json(e1_path)), load_json(e1_path))] if e1_path else []) + sel
    for p, cfg, res in items:
        for task, models in res["results"].items():
            for model in ["LogisticRegression", "DNN_MLP"]:
                r = models.get(model)
                if r is None:
                    continue
                d = r.get("diagnostics", {})
                rows.append({"task": task, "model": model, "max_iter": d.get("max_iter"),
                             "n_iter": json.dumps(d.get("n_iter")),
                             "warnings_recorded": warnings_field(r, cfg),
                             "accuracy": r["metrics"]["accuracy"], "f1_macro": r["metrics"]["f1_macro"],
                             "train_time_sec": r["metrics"]["train_time_sec"]})
    if not rows:
        return None
    return order(pd.DataFrame(rows))


def section_cv(root):
    files = glob.glob(os.path.join(root, "cv", "folds", "*.json"))
    if not files:
        return None, None
    rows = []
    for f in files:
        r = load_json(f)
        row = {"task": r["task"], "model": r["model"], "fold": r["fold"], "n_splits": r["n_splits"]}
        row.update({k: r["metrics"].get(k) for k in MAIN + ["mcc", "train_time_sec"]})
        rows.append(row)
    folds = pd.DataFrame(rows)
    g = folds.groupby(["task", "model"])
    summ = g[MAIN].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["folds_done"] = g["fold"].count()
    summ["n_splits"] = g["n_splits"].max()
    summ = summ.reset_index()
    return order(folds), order(summ)


def section_bootstrap(root, e1_path, ablation_paths):
    rows = []
    sources = {}
    if e1_path:
        sources["e1"] = os.path.dirname(e1_path)
    if "global" in ablation_paths:
        sources["ablation_global"] = os.path.dirname(ablation_paths["global"])
    for name, d in sources.items():
        f = os.path.join(d, "bootstrap_ci.csv")
        if os.path.exists(f):
            b = pd.read_csv(f)
            b.insert(0, "source", name)
            rows.append(b)
    if not rows:
        return None
    b = pd.concat(rows)
    b = b[b["metric"] == "f1_macro"].drop(columns=["metric"])
    return order(b)


def section_per_class(e1_path):
    if not e1_path:
        return {}
    res = load_json(e1_path)
    out = {}
    for task in ["6class", "19class"]:
        frames = []
        for model in MODEL_ORDER:
            r = res["results"].get(task, {}).get(model)
            if r is None:
                continue
            cr = r["classification_report"]
            rows = [{"class": k, f"{model}_P": v["precision"], f"{model}_R": v["recall"],
                     f"{model}_F1": v["f1-score"], "support": int(v["support"])}
                    for k, v in cr.items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")]
            frames.append(pd.DataFrame(rows).set_index("class"))
        if frames:
            df = frames[0]
            for fr in frames[1:]:
                df = df.join(fr.drop(columns=["support"]))
            cols = ["support"] + [c for c in df.columns if c != "support"]
            out[task] = df[cols].reset_index()
    return out


def section_logs(logs_dir):
    """Count ConvergenceWarnings printed in SLURM logs, attributed to the model being trained.
    Needed for runs made before warnings were recorded in results.json."""
    if not logs_dir or not os.path.isdir(logs_dir):
        return None
    start_04 = re.compile(r"\[(binary|6class|19class)\] Training (\w+)\.\.\.")
    start_05 = re.compile(r"\[(binary|6class|19class)\] (\w+) fold (\d+)/(\d+): train=")
    header = re.compile(r"job=(\S+) array=(\S+) host=(\S+) cpus=(\S+)")
    variant = re.compile(r"Scaling variant: (\w+)")
    iters = re.compile(r"failed to converge after (\d+) iteration")
    counts = {}
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.out"))):
        job_type = os.path.basename(path).split("_")[1] if os.path.basename(path).startswith("iomt_") else "?"
        cpus, var, ctx = None, None, None
        with open(path, errors="replace") as f:
            for line in f:
                h = header.search(line)
                if h:
                    cpus = h.group(4)
                v = variant.search(line)
                if v:
                    var = v.group(1)
                m = start_05.search(line) or start_04.search(line)
                if m:
                    fold = int(m.group(3)) - 1 if m.re is start_05 else None
                    ctx = (os.path.basename(path), job_type, var, m.group(1), m.group(2), fold, cpus)
                    counts.setdefault(ctx, [0, None])
                if "ConvergenceWarning" in line and ctx is not None:
                    counts[ctx][0] += 1
                    it = iters.search(line)
                    if it:
                        counts[ctx][1] = int(it.group(1))
    rows = [{"log": k[0], "job": k[1], "scaling": k[2], "task": k[3], "model": k[4], "fold": k[5],
             "cpus_allocated": k[6], "convergence_warnings": v[0], "iterations_in_warning": v[1]}
            for k, v in counts.items()]
    return pd.DataFrame(rows) if rows else None


def section_seeds(root, e1_path):
    """Seed study: same data and data order (shuffle 42), model random_state varied.
    Returns (individual, summary_by_scaling, paired_differences_global_minus_none, equivalence)."""
    sel, dups = collect(root, "seeds")
    DUPLICATES.extend(dups)
    sel = [x for x in sel if x[1]["class_weight"] == "none"]   # class_weight runs live in their own stage
    if not sel:
        return None, None, None, None
    rows = []
    for p, cfg, res in sel:
        rows += metric_rows(res, {"scaling": cfg["scaling"], "seed": cfg["model_seed"], "run": p})
    ind = pd.DataFrame(rows)
    g = ind.groupby(["scaling", "task", "model"])
    summ = g[["accuracy", "f1_macro"]].agg(["mean", "std", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["n_seeds"] = g["seed"].nunique()
    summ["seeds"] = g["seed"].apply(lambda x: " ".join(str(v) for v in sorted(x)))
    summ = summ.reset_index()
    diff = None
    if {"global", "none"} <= set(ind["scaling"]):
        w = ind.pivot_table(index=["task", "model", "seed"], columns="scaling", values="f1_macro").dropna()
        w["diff_global_minus_none"] = w["global"] - w["none"]
        gd = w.groupby(level=["task", "model"])["diff_global_minus_none"]
        diff = pd.DataFrame({"mean": gd.mean(), "std": gd.std(), "min": gd.min(), "max": gd.max(),
                             "n_seeds": gd.count(),
                             "n_positive": gd.apply(lambda x: int((x > 0).sum()))}).reset_index()
        diff = order(diff)
    # Equivalence of the v2 seed-42 runs with the existing runs (E1 = global; ablation none = none)
    eq = []
    ref_runs = {}
    if e1_path:
        ref_runs["global"] = load_json(e1_path)
    abl, _ = collect(root, "ablation")
    for p, cfg, res in abl:
        if cfg["scaling"] == "none" and cfg["model_seed"] == 42:
            ref_runs["none"] = res
    for p, cfg, res in sel:
        if cfg["model_seed"] == 42 and cfg["scaling"] in ref_runs:
            ref = ref_runs[cfg["scaling"]]
            for t, ms in res["results"].items():
                for m, r in ms.items():
                    rr = ref["results"].get(t, {}).get(m)
                    if rr is not None:
                        eq.append({"scaling": cfg["scaling"], "task": t, "model": m,
                                   "identical_CM_with_existing_run": r["confusion_matrix"] == rr["confusion_matrix"],
                                   "delta_f1_macro": r["metrics"]["f1_macro"] - rr["metrics"]["f1_macro"]})
    return order(ind), order(summ), diff, (order(pd.DataFrame(eq)) if eq else None)


DUPLICATES = []


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--reference_json")
    ap.add_argument("--logs_dir", default="logs", help="SLURM logs, to recover warnings of older runs")
    ap.add_argument("--print_run", choices=["e1", "ablation_global"],
                    help="Only print the directory of the complete run that would be used, then exit")
    a = ap.parse_args()
    root = a.results_root
    if a.print_run:
        if a.print_run == "e1":
            p = select_e1(root)
        else:
            sel, _ = collect(root, "ablation")
            c = [x[0] for x in sel if x[1]["scaling"] == "global" and x[1]["model_seed"] == 42]
            p = c[-1] if c else None
        if p:
            print(os.path.dirname(p))
        sys.exit(0 if p else 1)
    if not a.reference_json:
        ap.error("--reference_json is required")
    out_dir = os.path.join(root, "report")
    os.makedirs(out_dir, exist_ok=True)
    # Remove this script's previous outputs so no stale table survives a re-run
    for f in glob.glob(os.path.join(out_dir, "*.csv")) + [os.path.join(out_dir, x) for x in ("REPORT.md", "summary.json")]:
        if os.path.exists(f):
            os.remove(f)
    ref = load_json(a.reference_json)

    e1_path, e1_comp, e1_diag = section_e1(root, ref)
    abl, abl_pivot, abl_paths = section_ablation(root)
    conv = section_convergence(root, e1_path)
    cv_folds, cv_summ = section_cv(root)
    boot = section_bootstrap(root, e1_path, abl_paths)
    per_class = section_per_class(e1_path)
    logs_tab = section_logs(a.logs_dir)
    seeds_all, seeds_summ, seeds_diff, seeds_eq = section_seeds(root, e1_path)
    val_path = os.path.join(root, "validation", "validation.json")
    val = load_json(val_path) if os.path.exists(val_path) else None
    env = load_json(e1_path)["experiment_info"].get("environment") if e1_path else None

    tables = {"e1_vs_reference": e1_comp, "e1_diagnostics": e1_diag, "ablation": abl,
              "ablation_f1_macro_pivot": abl_pivot, "convergence": conv,
              "cv_all_folds": cv_folds, "cv_summary": cv_summ, "bootstrap_f1_macro": boot,
              "log_convergence_warnings": logs_tab, "seeds_all": seeds_all, "seeds_summary": seeds_summ,
              "seeds_paired_diff_global_minus_none": seeds_diff, "seeds_v2_equivalence_seed42": seeds_eq,
              "duplicate_runs": pd.DataFrame(DUPLICATES) if DUPLICATES else None,
              "incomplete_runs": pd.DataFrame(INCOMPLETE) if INCOMPLETE else None}
    for k, v in per_class.items():
        tables[f"per_class_{k}"] = v
    for name, df in tables.items():
        if df is not None and len(df):
            df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    if e1_comp is None:
        e1_status = "pendiente"
    else:
        same = e1_comp[e1_comp["identical_CM"]]
        diff = e1_comp[~e1_comp["identical_CM"]]
        n_ref = sum(len(v) for v in ref["results"].values())
        e1_status = f"{len(same)}/{n_ref} matrices identicas al JSON del paper ({e1_path})"
        if len(e1_comp) < n_ref:
            e1_status += f"; FALTAN {n_ref - len(e1_comp)} combinaciones en la corrida E1"
        if len(diff):
            worst = diff.loc[diff["delta_f1_macro"].abs().idxmax()]
            e1_status += (f"; difieren: " + ", ".join(f"{r.task}/{r.model}" for r in diff.itertuples())
                          + f"; mayor |dF1| = {abs(worst['delta_f1_macro']):.4f} ({worst['task']}/{worst['model']})")
    status = {
        "e1": e1_status,
        "validation": "pendiente" if val is None else ("PASSED" if val.get("passed") else "FAILED"),
        "ablation_variants_done": sorted(abl_paths),
        "convergence_table_rows (incluye E1 como base)": 0 if conv is None else int(len(conv)),
        "cv_folds_done": 0 if cv_folds is None else int(len(cv_folds)),
        "cv_folds_expected": 120,
        "bootstrap": "pendiente" if boot is None else "hecho",
        "seeds": "no corridas" if seeds_summ is None else f"{int(seeds_summ['n_seeds'].max())} semillas",
        "seeds_v2_equivalence": "n/a" if seeds_eq is None else
            f"{int(seeds_eq['identical_CM_with_existing_run'].sum())}/{len(seeds_eq)} matrices identicas",
        "duplicate_configurations": len(DUPLICATES),
        "incomplete_runs (no usadas)": len(INCOMPLETE),
        "convergence_warnings_in_logs": 0 if logs_tab is None else int(logs_tab["convergence_warnings"].sum()),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"status": status, "environment": env, "e1_run": e1_path,
                   "ablation_runs": abl_paths}, f, indent=2, default=str)

    md = ["# Reporte de experimentos — revisión ARRAY-D-26-05137\n", "## Estado\n"]
    md += [f"- **{k}**: {v}" for k, v in status.items()]
    md.append("\n## 1. Entorno\n")
    md.append("```\n" + json.dumps(env, indent=2) + "\n```\n" if env else "_pendiente_\n")
    md.append("## 2. E1: reproducción del JSON del paper\n")
    md.append(md_table(e1_comp, "{:+.6f}"))
    md.append("## 3. Validación de los CSV reconstruidos\n")
    if val:
        for part in ["train", "test"]:
            r = val[part]
            md.append(f"- {part}: filas limpias paper={r['paper_rows_clean']} reconstruido={r['rebuilt_rows_clean']}, "
                      f"features en mismo orden={r['features_identical_order']}, etiquetas iguales={r['labels_equal']}, "
                      f"filas idénticas={r['rows_identical_multiset']}")
        md.append(f"- conteos de test vs JSON: " + ", ".join(
            f"{t}={v['equal']}" for t, v in val["test_support_vs_reference"].items()) + "\n")
    else:
        md.append("_pendiente_\n")
    md.append("## 4. Ablation (escalado)\n")
    md.append("Macro F1 por variante:\n\n" + md_table(abl_pivot))
    md.append("Todas las métricas:\n\n" + md_table(abl))
    md.append("## 5. Diagnósticos (E1)\n")
    md.append(md_table(e1_diag, "{:.1f}"))
    md.append("## 6. Convergencia (LR y MLP)\n")
    md.append(md_table(conv))
    md.append("## 7. Validación cruzada estratificada (solo D_train)\n")
    md.append(md_table(cv_summ))
    md.append("## 8. IC bootstrap 95% de macro F1 (test)\n")
    md.append(md_table(boot))
    md.append("## 9. Advertencias de convergencia encontradas en los logs\n")
    md.append("Los runs anteriores a la corrección no guardaban estas advertencias en results.json "
              "(LR ajusta en un proceso aparte); se recuperan de los logs de SLURM.\n\n")
    md.append(md_table(logs_tab[logs_tab["convergence_warnings"] > 0] if logs_tab is not None else None, "{:.0f}"))
    md.append("## 10. Variabilidad entre semillas\n")
    md.append("Varía solo random_state de los modelos; datos y orden de filas fijos (shuffle 42). "
              "LR con lbfgs es determinista, así que su desviación debe ser 0.\n\n")
    md.append(md_table(seeds_summ.drop(columns=["seeds"]) if seeds_summ is not None else None))
    md.append("Diferencia de macro F1 global − sin escalado, pareada por semilla:\n\n" + md_table(seeds_diff))
    md.append("Equivalencia de v2 (semilla 42) con las corridas existentes:\n\n" + md_table(seeds_eq, "{:+.6f}"))
    md.append("## 10b. Configuraciones duplicadas (se usa la más reciente completa)\n")
    md.append(md_table(pd.DataFrame(DUPLICATES) if DUPLICATES else None))
    md.append("## 10c. Corridas incompletas (canceladas o caídas; no se usan)\n")
    md.append(md_table(pd.DataFrame(INCOMPLETE) if INCOMPLETE else None))
    md.append("## 11. Métricas por clase (E1)\n")
    for task, df in per_class.items():
        md.append(f"### {task}\n\n" + md_table(df))
    report_path = os.path.join(out_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md))

    print("ESTADO:")
    for k, v in status.items():
        print(f"  {k}: {v}")
    if abl_pivot is not None:
        print("\nAblation, macro F1:\n" + abl_pivot.round(4).to_string(index=False))
    if cv_summ is not None:
        print("\nCV, macro F1:\n" + cv_summ[["task", "model", "folds_done", "f1_macro_mean", "f1_macro_std"]]
              .round(4).to_string(index=False))
    print(f"\nReporte: {report_path}")


if __name__ == "__main__":
    main()
