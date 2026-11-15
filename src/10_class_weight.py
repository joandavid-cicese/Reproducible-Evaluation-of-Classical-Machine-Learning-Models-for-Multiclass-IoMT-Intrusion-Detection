#!/usr/bin/env python3
"""
10_class_weight.py - Sensitivity of Random Forest to class_weight='balanced' (R3-6, decision D6).

Compares, PAIRED BY SEED, RF with class_weight='balanced' (results/class_weight) against the
default RF of the seed study (results/seeds, global scaling, class_weight none). Everything else
is identical: data, row order (shuffle 42), scaling (global), hyperparameters, CPUs, seeds.

This is a sensitivity analysis. It does NOT replace the main results (default hyperparameters of
the reference study) and it is NOT used to select a model or a configuration.

Outputs (results/report_class_weight/, kept apart from paper_tables/ because 09 cleans that folder):
  class_weight_summary.csv/.tex      macro F1 / macro recall / accuracy, mean +- sd over seeds,
                                     and the paired difference balanced - default
  class_weight_individual.csv        every seed, both variants
  class_weight_per_class_<task>.csv/.tex   per-class precision / recall / F1 (mean over seeds)
                                     and paired recall difference, in the manuscript class order

Usage:
  python src/10_class_weight.py --results_root results
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep08 = _load("rep08", "08_report.py")
rep09 = _load("rep09", "09_paper_tables.py")

MODEL = "RandomForest"
TASKS = ["binary", "6class", "19class"]
METRICS = ["f1_macro", "recall_macro", "prec_macro", "accuracy"]


def runs(root, stage, class_weight):
    sel, dups = rep08.collect(root, stage)
    for d in dups:
        print("  DUPLICATE CONFIGURATION (most recent complete used):", d["config"], "->", d["used"])
    out = {}
    for p, cfg, res in sel:
        if cfg["scaling"] != "global" or cfg["class_weight"] != class_weight:
            continue
        if not all(MODEL in res["results"].get(t, {}) for t in TASKS):
            continue
        out[cfg["model_seed"]] = (p, res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results_root, "report_class_weight")
    os.makedirs(out, exist_ok=True)

    bal = runs(a.results_root, "class_weight", "balanced")
    base = runs(a.results_root, "seeds", "none")
    seeds = sorted(set(bal) & set(base))
    for r in rep08.INCOMPLETE:
        print("  INCOMPLETE RUN (not used):", r["run"], "missing:", r["missing"])
    print(f"balanced runs: seeds {sorted(bal)}; default runs (seed study, global): seeds {sorted(base)}")
    if not seeds:
        sys.exit("No seed has both a balanced run (results/class_weight) and a default run (results/seeds).")
    if set(bal) - set(base):
        print(f"  WARNING: balanced seeds without a default counterpart are ignored: {sorted(set(bal) - set(base))}")

    # ---- overall metrics, paired by seed
    rows = []
    for s in seeds:
        for variant, src in (("default", base), ("balanced", bal)):
            p, res = src[s]
            for t in TASKS:
                m = res["results"][t][MODEL]["metrics"]
                rows.append({"task": t, "variant": variant, "seed": s, "run": p, **{k: m[k] for k in METRICS}})
    ind = pd.DataFrame(rows)
    ind.to_csv(os.path.join(out, "class_weight_individual.csv"), index=False)

    summ = []
    for t in TASKS:
        row = {"Task": t, "Seeds": len(seeds)}
        for k, lab in (("f1_macro", "Macro F1"), ("recall_macro", "Macro R"), ("accuracy", "Accuracy")):
            w = ind[ind.task == t].pivot(index="seed", columns="variant", values=k)
            d = w["balanced"] - w["default"]
            row[f"{lab} default"] = f"{w['default'].mean():.4f} ± {w['default'].std(ddof=1):.4f}" if len(seeds) > 1 \
                else f"{w['default'].mean():.4f}"
            row[f"{lab} balanced"] = f"{w['balanced'].mean():.4f} ± {w['balanced'].std(ddof=1):.4f}" if len(seeds) > 1 \
                else f"{w['balanced'].mean():.4f}"
            if k != "accuracy":
                row[f"Δ{lab} (bal − def)"] = f"{d.mean():+.4f} ± {d.std(ddof=1):.4f}" if len(seeds) > 1 else f"{d.mean():+.4f}"
                row[f"Δ{lab} > 0"] = f"{int((d > 0).sum())}/{len(d)}"
        if 42 in seeds:
            w = ind[(ind.task == t) & (ind.seed == 42)].set_index("variant")["f1_macro"]
            row["Macro F1 seed 42 (def → bal)"] = f"{w['default']:.4f} → {w['balanced']:.4f}"
        summ.append(row)
    summ = pd.DataFrame(summ)
    rep09.save(summ, out, "class_weight_summary",
               caption=f"Random Forest with class\\_weight='balanced' versus default (sensitivity analysis), "
                       f"global scaling, paired by model seed ({', '.join(map(str, seeds))}); mean ± sd over seeds.",
               label="tab:class_weight_summary")

    # ---- per class (classification_report stored in results.json), manuscript class order
    for t in ["6class", "19class"]:
        names = base[seeds[0]][1]["results"][t][MODEL]["class_names"]
        order = rep09.pres_order(t, names)
        recs = []
        for c in order:
            vals = {}
            for variant, src in (("def", base), ("bal", bal)):
                cr = [src[s][1]["results"][t][MODEL]["classification_report"][c] for s in seeds]
                vals[variant] = {k: np.array([x[k] for x in cr]) for k in ("precision", "recall", "f1-score")}
                support = int(cr[0]["support"])
            dr = vals["bal"]["recall"] - vals["def"]["recall"]
            recs.append({"Class": c, "Support": support,
                         "P def": vals["def"]["precision"].mean(), "P bal": vals["bal"]["precision"].mean(),
                         "R def": vals["def"]["recall"].mean(), "R bal": vals["bal"]["recall"].mean(),
                         "F1 def": vals["def"]["f1-score"].mean(), "F1 bal": vals["bal"]["f1-score"].mean(),
                         "ΔR": dr.mean(), "ΔR > 0": f"{int((dr > 0).sum())}/{len(dr)}"})
        pc = pd.DataFrame(recs)
        rep09.save(pc, out, f"class_weight_per_class_{t}",
                   caption=f"Per-class precision, recall and F1 of Random Forest, {t}, default versus "
                           f"class\\_weight='balanced' (mean over {len(seeds)} seeds; test set).",
                   label=f"tab:class_weight_per_class_{t}", int_cols=("Support",))

    print(summ.to_string(index=False))
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
