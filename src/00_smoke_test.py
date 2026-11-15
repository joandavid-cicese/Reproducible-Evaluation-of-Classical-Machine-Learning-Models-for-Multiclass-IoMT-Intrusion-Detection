"""
00_smoke_test.py
----------------
End-to-end check of the environment and of scripts 02-08 on a tiny synthetic dataset with
the same file naming and column structure as CICIoMT2024. Takes 1-3 minutes on one node.

    python src/00_smoke_test.py            # add --keep to keep the temporary directory

Prints the library versions and PASS/FAIL for each step. Exit code 0 only if all pass.
The temporary working directory is printed (kept on failure for inspection).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

TRAIN_FILES = {"ARP_Spoofing": 0, "Benign": 1, "TCP_IP-DDoS-ICMP1": 2, "TCP_IP-DDoS-ICMP2": 2,
               "Recon-Ping_Sweep": 3, "MQTT-DoS-Connect_Flood": 4, "TCP_IP-DoS-UDP1": 5}
TEST_FILES = {"ARP_Spoofing": 0, "Benign": 1, "TCP_IP-DDoS-ICMP1": 2, "Recon-Ping_Sweep": 3,
              "MQTT-DoS-Connect_Flood": 4, "TCP_IP-DoS-UDP1": 5}


def make_raw(folder, files, split, n, rng):
    os.makedirs(folder, exist_ok=True)
    for name, c in files.items():
        X = rng.normal(loc=c * 0.8, scale=1 + 0.2 * c, size=(n, 6)) * np.array([1, 10, 100, 1000, 1, 1])
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        df["flag_count"] = rng.integers(0, 3 + c, size=n)
        if name == "Benign":
            df.loc[0, "f0"] = np.inf
            df.loc[1, "f1"] = np.nan
        df.to_csv(os.path.join(folder, f"{name}_{split}.pcap.csv"), index=False)


def step(name, cmd, cwd, results, expect_fail=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    ok = (p.returncode != 0) if expect_fail else (p.returncode == 0)
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print("   stdout:", p.stdout[-1500:])
        print("   stderr:", p.stderr[-3000:])
    return p


def main():
    import sklearn, scipy
    print(f"python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | numpy {np.__version__} "
          f"| pandas {pd.__version__} | scipy {scipy.__version__} | cpus visible {len(os.sched_getaffinity(0))}")
    work = tempfile.mkdtemp(prefix="iomt_smoke_")
    print(f"working dir: {work}")
    rng = np.random.default_rng(0)
    make_raw(os.path.join(work, "data/raw/train"), TRAIN_FILES, "train", 400, rng)
    make_raw(os.path.join(work, "data/raw/test"), TEST_FILES, "test", 150, rng)
    res = []
    s = lambda f: os.path.join(SRC, f)

    os.makedirs(os.path.join(work, "rebuilt"), exist_ok=True)
    step("02 combine train", [PY, s("02_combine_train.py"), "--input_dir", "data/raw/train",
                              "--output_file", "rebuilt/train_src.csv"], work, res)
    step("03 combine test", [PY, s("03_combine_test.py"), "--input_dir", "data/raw/test",
                             "--output_file", "rebuilt/test_src.csv"], work, res)

    # "paper" CSVs: same rows without source_file and in a different order
    os.makedirs(os.path.join(work, "data/paper"), exist_ok=True)
    for part in ["train", "test"]:
        df = pd.read_csv(os.path.join(work, f"rebuilt/{part}_src.csv")).drop(columns=["source_file"])
        df.sample(frac=1, random_state=7).to_csv(os.path.join(work, f"data/paper/{part}.csv"), index=False)

    step("04 E1 (reference run)", [PY, s("04_run_experiment.py"), "--train", "data/paper/train.csv",
                                   "--test", "data/paper/test.csv", "--results_dir", "ref"], work, res)
    ref_dirs = os.listdir(os.path.join(work, "ref")) if os.path.isdir(os.path.join(work, "ref")) else []
    ref_json = os.path.join(work, "ref", ref_dirs[0], "results.json") if ref_dirs else "missing.json"
    p = step("04 E1 rerun vs reference (determinism)", [PY, s("04_run_experiment.py"), "--train", "data/paper/train.csv",
             "--test", "data/paper/test.csv", "--results_dir", "results/e1", "--reference_json", ref_json], work, res)
    det_ok = p.returncode == 0 and "identical_CM=False" not in p.stderr and p.stderr.count("identical_CM=True") == 12
    res.append(("04 determinism: 12/12 identical confusion matrices", det_ok))
    print(f"[{'PASS' if det_ok else 'FAIL'}] 04 determinism: 12/12 identical confusion matrices")

    step("07 validation (should pass)", [PY, s("07_validate_rebuilt.py"), "--paper_train", "data/paper/train.csv",
         "--paper_test", "data/paper/test.csv", "--rebuilt_train", "rebuilt/train_src.csv",
         "--rebuilt_test", "rebuilt/test_src.csv", "--reference_json", ref_json,
         "--out", "results/validation/validation.json"], work, res)
    bad = pd.read_csv(os.path.join(work, "rebuilt/test_src.csv")).iloc[5:]
    bad.to_csv(os.path.join(work, "rebuilt/test_bad.csv"), index=False)
    step("07 validation detects a corrupted file (should fail)", [PY, s("07_validate_rebuilt.py"),
         "--paper_train", "data/paper/train.csv", "--paper_test", "data/paper/test.csv",
         "--rebuilt_train", "rebuilt/train_src.csv", "--rebuilt_test", "rebuilt/test_bad.csv",
         "--reference_json", ref_json, "--out", "tmp_bad.json"], work, res, expect_fail=True)

    for v in ["global", "per_file", "none", "leaky"]:
        step(f"04 ablation --scaling {v}", [PY, s("04_run_experiment.py"), "--train", "rebuilt/train_src.csv",
             "--test", "rebuilt/test_src.csv", "--scaling", v, "--results_dir", "results/ablation"], work, res)
    step("04 convergence (--lr_max_iter 500 --mlp_max_iter 300)", [PY, s("04_run_experiment.py"),
         "--train", "data/paper/train.csv", "--test", "data/paper/test.csv", "--models", "LogisticRegression",
         "DNN_MLP", "--lr_max_iter", "500", "--mlp_max_iter", "300", "--results_dir", "results/convergence"], work, res)
    step("05 CV single fold (task/model/fold selection)", [PY, s("05_cross_validation.py"), "--train",
         "data/paper/train.csv", "--results_dir", "results/cv", "--n_splits", "3", "--tasks", "19class",
         "--models", "AdaBoost", "--fold", "1"], work, res)
    step("05 CV all (3 folds, resumable)", [PY, s("05_cross_validation.py"), "--train", "data/paper/train.csv",
         "--results_dir", "results/cv", "--n_splits", "3"], work, res)
    n_folds = len(os.listdir(os.path.join(work, "results/cv/folds"))) if os.path.isdir(os.path.join(work, "results/cv/folds")) else 0
    res.append(("05 CV produced 36 fold files", n_folds == 36))
    print(f"[{'PASS' if n_folds == 36 else 'FAIL'}] 05 CV produced 36 fold files ({n_folds})")

    for seed in ["42", "7"]:
        for v in ["global", "none"]:
            step(f"04 seed study (--seed {seed} --scaling {v})", [PY, s("04_run_experiment.py"), "--train",
                 "data/paper/train.csv", "--test", "data/paper/test.csv", "--seed", seed, "--scaling", v,
                 "--results_dir", "results/seeds"], work, res)
    for seed in ["42", "7"]:
        step(f"04 class_weight balanced RF (--seed {seed})", [PY, s("04_run_experiment.py"), "--train",
             "data/paper/train.csv", "--test", "data/paper/test.csv", "--seed", seed, "--models", "RandomForest",
             "--class_weight", "balanced", "--results_dir", "results/class_weight"], work, res)
    step("04 class_weight rejected for AdaBoost/MLP", [PY, s("04_run_experiment.py"), "--train",
         "data/paper/train.csv", "--test", "data/paper/test.csv", "--models", "AdaBoost",
         "--class_weight", "balanced", "--results_dir", "tmp_cw"], work, res, expect_fail=True)
    p = step("10 class_weight tables (paired by seed)", [PY, s("10_class_weight.py"), "--results_root", "results"],
             work, res)
    cwf = os.path.join(work, "results/report_class_weight/class_weight_summary.csv")
    cw_ok = os.path.exists(cwf) and (pd.read_csv(cwf)["Seeds"] == 2).all() and \
        os.path.exists(os.path.join(work, "results/report_class_weight/class_weight_per_class_19class.tex"))
    res.append(("10 class_weight: 2 seeds paired, per-class tables written", cw_ok))
    print(f"[{'PASS' if cw_ok else 'FAIL'}] 10 class_weight: 2 seeds paired, per-class tables written")
    p = step("04 forced non-convergence (warnings must be recorded)", [PY, s("04_run_experiment.py"),
             "--train", "data/paper/train.csv", "--test", "data/paper/test.csv", "--tasks", "19class",
             "--models", "LogisticRegression", "--lr_max_iter", "3", "--results_dir", "tmp_warn"], work, res)
    wj = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(work, "tmp_warn")) for f in fs if f == "results.json"]
    w_ok = bool(wj) and any("ConvergenceWarning" in w for w in
                            json.load(open(wj[0]))["results"]["19class"]["LogisticRegression"]["warnings"])
    res.append(("04 ConvergenceWarning recorded in results.json", w_ok))
    print(f"[{'PASS' if w_ok else 'FAIL'}] 04 ConvergenceWarning recorded in results.json")

    e1_dirs = os.listdir(os.path.join(work, "results/e1")) if os.path.isdir(os.path.join(work, "results/e1")) else ["x"]
    step("06 bootstrap", [PY, s("06_bootstrap_ci.py"), "--run_dir", os.path.join("results/e1", e1_dirs[0]),
                          "--n_boot", "100"], work, res)
    # Simulate a cancelled job: a NEWER E1 run and a newer convergence run with only part of the models.
    # 08/09 must ignore them and keep using the complete runs.
    good_e1 = os.path.join(work, "results/e1", e1_dirs[0], "results.json")
    if os.path.exists(good_e1):
        part = json.load(open(good_e1))
        part["experiment_info"]["status"] = "running"
        part["results"] = {"binary": {m: part["results"]["binary"][m]
                                      for m in ["RandomForest", "AdaBoost", "LogisticRegression"]}}
        os.makedirs(os.path.join(work, "results/e1/global_seed42_29991231_235959"), exist_ok=True)
        json.dump(part, open(os.path.join(work, "results/e1/global_seed42_29991231_235959/results.json"), "w"))
        conv = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(work, "results/convergence"))
                for f in fs if f == "results.json"][0]
        part = json.load(open(conv))
        part["experiment_info"].pop("status", None)          # pre-v2.2 run: completeness must be inferred
        part["experiment_info"].pop("requested", None)
        part["results"] = {"binary": {"LogisticRegression": part["results"]["binary"]["LogisticRegression"]}}
        d = os.path.join(work, "results/convergence/global_lrit500_mlpit300_seed42_29991231_235959")
        os.makedirs(d, exist_ok=True)
        json.dump(part, open(os.path.join(d, "results.json"), "w"))
    p = step("08 --print_run e1 (must skip the cancelled run)", [PY, s("08_report.py"), "--results_root", "results",
             "--print_run", "e1"], work, res)
    pr_ok = p.stdout.strip() == os.path.join("results/e1", e1_dirs[0])
    res.append(("08 --print_run e1 returns the complete run, not the newer cancelled one", pr_ok))
    print(f"[{'PASS' if pr_ok else 'FAIL'}] 08 --print_run e1 returns the complete run ({p.stdout.strip()})")
    os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    p = step("08 report", [PY, s("08_report.py"), "--results_root", "results", "--reference_json", ref_json,
                           "--logs_dir", "logs"], work, res)
    rep_ok = os.path.exists(os.path.join(work, "results/report/REPORT.md")) and "12/12 matrices" in p.stdout \
        and "2 semillas" in p.stdout
    inc_ok = "incomplete_runs (no usadas): 2" in p.stdout
    res.append(("08 cancelled runs detected (2) and not used", inc_ok))
    print(f"[{'PASS' if inc_ok else 'FAIL'}] 08 cancelled runs detected (2) and not used")
    res.append(("08 report: E1 12/12 identical and seeds aggregated", rep_ok))
    print(f"[{'PASS' if rep_ok else 'FAIL'}] 08 report: E1 12/12 identical and seeds aggregated")
    eqf = os.path.join(work, "results/report/seeds_v2_equivalence_seed42.csv")
    eq_ok = os.path.exists(eqf) and pd.read_csv(eqf).query("scaling == 'global'")["identical_CM_with_existing_run"].all()
    res.append(("08 seed-42 v2 run identical to E1 (global)", bool(eq_ok)))
    print(f"[{'PASS' if eq_ok else 'FAIL'}] 08 seed-42 v2 run identical to E1 (global)")
    dfd = os.path.join(work, "results/report/seeds_paired_diff_global_minus_none.csv")
    d_ok = os.path.exists(dfd) and int(pd.read_csv(dfd)["n_seeds"].min()) == 2
    res.append(("08 paired differences global-none computed for 2 seeds", d_ok))
    print(f"[{'PASS' if d_ok else 'FAIL'}] 08 paired differences global-none computed for 2 seeds")

    os.makedirs(os.path.join(work, "reference"), exist_ok=True)
    pd.DataFrame([{"source": "test", "source_version": "published (synthetic)", "task": "19class",
                   "model": "RandomForest", "metric": "f1_macro", "value": 0.551, "decimals": 3,
                   "averaging": "not stated", "status": "synthetic"},
                  {"source": "test", "source_version": "preprint (synthetic)", "task": "19class",
                   "model": "RandomForest", "metric": "f1_macro", "value": 0.9074, "decimals": 4,
                   "averaging": "not stated", "status": "synthetic"}]).to_csv(
        os.path.join(work, "reference/baseline_published.csv"), index=False)
    p = step("09 paper tables (12/12 consistency checks)", [PY, s("09_paper_tables.py"), "--results_root", "results",
             "--out", "results/paper_tables"], work, res)
    t_ok = "12/12 passed" in p.stdout and all(os.path.exists(os.path.join(work, "results/paper_tables", f)) for f in
              ["selection_by_cv.json", "cm_19class_RandomForest.tex", "main_results.tex",
               "table2_class_mapping_19class.tex", "comparison_published.csv", "ablation_scaling.tex",
               "convergence.tex", "per_class_19class.tex", "reference_comparison.tex", "seeds_summary.tex",
               "seeds_diff_global_minus_none.tex", "seeds_individual.csv"])
    res.append(("09 all tables written", t_ok))
    print(f"[{'PASS' if t_ok else 'FAIL'}] 09 all tables written")

    n_fail = sum(not ok for _, ok in res)
    print(f"\n{len(res) - n_fail}/{len(res)} checks passed.")
    if n_fail == 0:
        if "--keep" not in sys.argv:
            shutil.rmtree(work, ignore_errors=True)
        print("SMOKE TEST PASSED - the environment is ready.")
        sys.exit(0)
    print(f"SMOKE TEST FAILED - files kept in {work}")
    sys.exit(1)


if __name__ == "__main__":
    main()
