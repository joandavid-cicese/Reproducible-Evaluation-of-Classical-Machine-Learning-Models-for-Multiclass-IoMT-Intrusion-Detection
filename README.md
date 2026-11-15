# Reproducible Evaluation of Classical Machine Learning Models for Multiclass IoMT Intrusion Detection on CICIoMT2024

This repository contains the code used to produce every result of the article of the same
title (submitted to *Array*; the full reference will be added on publication).

The study evaluates four classical classifiers on the Wi-Fi and MQTT data of
[CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html):

- Random Forest (RF);
- AdaBoost;
- Logistic Regression (LR);
- a multilayer perceptron (MLP).

They use the hyperparameters of the reference study of the dataset authors and are evaluated at
three granularities: binary (attack vs. benign), 6 classes (attack categories), and 19 classes
(attack types). No hyperparameter is tuned.

The repository contains **code only**. The dataset is not redistributed, and the result tables
are not versioned: their values are reported in the article, and the scripts below regenerate
them from your own runs.

---

## Contents

1. [What the code does](#1-what-the-code-does)
2. [Quick start](#2-quick-start)
3. [Requirements](#3-requirements)
4. [Step 1 – Download the data](#4-step-1--download-the-data)
5. [Step 2 – Build D_train and D_test](#5-step-2--build-d_train-and-d_test)
6. [Step 3 – Run the experiments](#6-step-3--run-the-experiments)
7. [Step 4 – Report and tables](#7-step-4--report-and-tables)
8. [Where each result of the article comes from](#8-where-each-result-of-the-article-comes-from)
9. [Notes on reproducibility](#9-notes-on-reproducibility)
10. [Troubleshooting](#10-troubleshooting)
11. [Repository layout](#11-repository-layout)
12. [License and citation](#12-license-and-citation)

---

## 1. What the code does

The evaluation protocol has three stages.

**Consolidation.** The 51 training CSV files and the 21 test CSV files of the official split are
merged into two tables, `D_train` and `D_test`. Each CSV file contains a single class, so the
models are trained on the merged training table.

**Model selection on the training data only.** The four models are compared with stratified
10-fold cross-validation on `D_train`. Each fold uses a scikit-learn
`Pipeline(StandardScaler, model)`, so the scaler is refitted on the nine training folds of every
split. The model with the highest mean cross-validation Macro F1-score is selected.

**Test evaluation.** The scaler and every model are refitted on the complete `D_train`.
`D_test` is transformed with the mean and standard deviation learned from `D_train` and used only
to report performance. It plays no role in scaling, training, or model selection.

The same code also runs the sensitivity analyses reported in the article:

- **scaling ablation:** no scaling, a scaler fitted on train + test, and a scaler fitted per file;
- **seed study:** five random seeds;
- **convergence:** higher iteration limits for LR and the MLP;
- **class weighting:** RF with `class_weight='balanced'`.

None of these analyses is used to select a model or to change the main configuration.

## 2. Quick start

```bash
git clone https://github.com/joandavid-cicese/Reproducible-Evaluation-of-Classical-Machine-Learning-Models-for-Multiclass-IoMT-Intrusion-Detection.git
cd Reproducible-Evaluation-of-Classical-Machine-Learning-Models-for-Multiclass-IoMT-Intrusion-Detection

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install requests beautifulsoup4      # needed only by the download script

python src/00_smoke_test.py              # 1–3 min on synthetic data
```

The smoke test runs the whole pipeline end to end on a small synthetic dataset: build, cross-
validation, main run, ablation, seeds, class weighting, report, and tables. It must end with
`SMOKE TEST PASSED`. If it does, the environment is ready for the real data.

## 3. Requirements

**Software.**

- Python 3.11. The reported runs used 3.11.9; the smoke test also passes with 3.13.
- `requirements.txt` pins the versions used in the article: scikit-learn 1.7.2, NumPy 2.3.5,
  pandas 2.3.3, and SciPy 1.16.3.
- `requirements-lock.txt` lists the complete environment of the reported runs.

> **Use scikit-learn 1.7.2.** From version 1.8, `AdaBoostClassifier` no longer accepts the
> `algorithm` parameter. The code uses SAMME in every version, but results obtained with other
> versions are not guaranteed to match the article.

**Hardware.** The experiments were run as SLURM jobs on a CPU node; no GPU is used. The table
lists the resources requested by each job (`slurm/*.sbatch`). The time limits were enough for
the reported runs. The training times in Section 4.6 of the article were measured in jobs with 32
CPU cores.

| Job | What it runs | CPUs | Memory | Time limit |
|---|---|---|---|---|
| `build` | `02` + `03` (consolidation) | 4 | 64 GB | 3 h |
| `e1` | main run + bootstrap CI | 32 | 128 GB | 12 h |
| `cv` (array of 120) | one CV fold per task, model and fold | 8 | 20 GB | 8 h |
| `ablation` | scaling variants | 32 | 128 GB | 12 h |
| `seeds` | 5 seeds × {global, none} | 32 | 96 GB | 12 h |
| `convergence` | higher `max_iter` for LR and MLP | 16 | 64 GB | 24 h |
| `class_weight` | RF balanced, 5 seeds | 32 | 96 GB | 4 h |
| `final`, `tables` | report and tables | 8 / 4 | 32 / 16 GB | 4 h / 1 h |

You also need enough disk space for the raw CSV files plus one consolidated copy of each split.

## 4. Step 1 – Download the data

```bash
python src/01_download_datasets.py --out_dir data/raw
```

The script reads the file listing of
`CICIoMT2024/Dataset/WiFI_and_MQTT/attacks/CSV/` on the server of the Canadian Institute for
Cybersecurity and saves every CSV of its `train/` and `test/` folders into `data/raw/train` and
`data/raw/test`. Files that already exist are skipped, so the script can be re-run after an
interrupted download. It warns if a folder does not end up with the expected number of files (51
for training, 21 for testing).

If the server address changes, pass the new address of the `attacks/CSV/` folder with
`--base_url`. You can also download the files manually from the
[dataset page](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) and place them in the same
two folders.

Please cite the dataset paper: Dadkhah et al., *Internet of Things* 28 (2024) 101351.

## 5. Step 2 – Build D_train and D_test

```bash
python src/02_combine_train.py --input_dir data/raw/train --output_file data/CICIoMT2024_train.csv
python src/03_combine_test.py  --input_dir data/raw/test  --output_file data/CICIoMT2024_test.csv
```

Each script concatenates the CSV files of one split and adds four columns:

- `Class`: binary label;
- `Category`: 6-class label;
- `Attack`: 19-class label;
- `source_file`: name of the original file.

The labels are derived from the file names. `source_file` is metadata: it is used only by the
per-file scaling diagnostic and is **never** a model feature.

The models use the 45 numerical features of the dataset. Rows with missing or infinite values
are removed from each split separately when the experiments load the data. The released files
contain none, so no row is removed.

Expected sizes: 7,160,831 training rows and 1,614,182 test rows.

## 6. Step 3 – Run the experiments

### Option A: with SLURM (as in the article)

1. Edit `slurm/submit_all.sh` and set `PARTITION` (and `ACCOUNT`, if your cluster requires it).
   The script refuses to run while `PARTITION="CHANGE_ME"`.
2. If your cluster needs it, uncomment the `module load` lines at the top of `slurm/env.sh`.
   Every job sources that file and activates `.venv`.
3. Submit:

```bash
bash slurm/submit_all.sh smoke    # optional: smoke test as a job
bash slurm/submit_all.sh all      # whole pipeline
bash slurm/status.sh              # progress, errors in the logs, and a short report
```

`all` first submits the build job, which runs `02` and `03`. Every experiment then starts only if
the build succeeded: main run, ablation, convergence, the 120 CV folds, seeds, and class
weighting. The report and the tables are generated when the experiments finish. Job IDs are
appended to `logs/submitted_jobs.txt`, and every job writes its log to `logs/`.

Other modes:

- `bash slurm/submit_all.sh seeds`: seed study, then report and tables.
- `bash slurm/submit_all.sh tables`: report and tables only, from existing results.

### Option B: without SLURM

Run the commands from the repository root with the virtual environment active. Each block is
independent once `D_train` and `D_test` exist.

```bash
TR=data/CICIoMT2024_train.csv
TE=data/CICIoMT2024_test.csv

# (a) Model selection: stratified 10-fold CV on D_train only (Sections 3.3 and 4.1).
#     Resumable; use --fold k to run a single fold, e.g. on different machines.
python src/05_cross_validation.py --train $TR --results_dir results/cv --n_splits 10

# (b) Main run: final fit on D_train, evaluation on D_test, saved predictions (Section 4).
python src/04_run_experiment.py --train $TR --test $TE --scaling global --results_dir results/e1
#     Bootstrap 95% CI of the test Macro F1-score (1,000 resamples)
python src/06_bootstrap_ci.py --run_dir results/e1/<run_folder> --n_boot 1000

# (c) Ablation of the scaling step (Section 4.3)
for v in global none leaky per_file; do
  python src/04_run_experiment.py --train $TR --test $TE --scaling $v --results_dir results/ablation
done

# (d) Seed study (Section 4.3)
for s in 42 123 2024 2025 2026; do for v in global none; do
  python src/04_run_experiment.py --train $TR --test $TE --scaling $v --seed $s --results_dir results/seeds
done; done

# (e) Convergence sensitivity (Section 4.6)
python src/04_run_experiment.py --train $TR --test $TE --models LogisticRegression DNN_MLP \
    --lr_max_iter 500 --mlp_max_iter 500 --results_dir results/convergence
python src/04_run_experiment.py --train $TR --test $TE --models LogisticRegression \
    --lr_max_iter 1000 --results_dir results/convergence

# (f) Class weighting, RF only (Section 4.5)
for s in 42 123 2024 2025 2026; do
  python src/04_run_experiment.py --train $TR --test $TE --scaling global --seed $s \
      --models RandomForest --class_weight balanced --results_dir results/class_weight
done
```

The main run takes several hours on 32 cores. Section 4.6 of the article reports the training
time of each model.

**Scaling variants of `04_run_experiment.py`:**

| `--scaling` | What it does | Role |
|---|---|---|
| `global` | scaler fitted on `D_train`, applied to `D_test` | main configuration |
| `none` | no scaling | sensitivity analysis; closest to the procedure documented in the reference study |
| `leaky` | scaler fitted on `D_train ∪ D_test` | deliberate leak, to measure its effect |
| `per_file` | scaler fitted separately on each CSV file | diagnostic only (see Section 9) |

**Output of each run:** a folder `results/<experiment>/<scaling>[_options]_seed<s>_<timestamp>/`
with:

- `results.json`: metrics, confusion matrices, iterations, training times, warnings, software
  versions, and allocated CPUs;
- `predictions/<task>_<model>.npz`: saved `y_true` and `y_pred`;
- `bootstrap_ci.csv`: written by `06`.

## 7. Step 4 – Report and tables

```bash
python src/05_cross_validation.py --results_dir results/cv --summarize   # cv_summary.csv
python src/08_report.py   --results_root results                         # results/report/REPORT.md
python src/09_paper_tables.py --results_root results --out results/paper_tables
python src/10_class_weight.py --results_root results --out results/class_weight_tables
```

**`08_report.py`** collects every run into a Markdown report. For each configuration it uses the
most recent *complete* run and lists incomplete or duplicate runs separately.

**`09_paper_tables.py`** regenerates the tables of the article as CSV and LaTeX. It does four
things:

- applies the selection rule from the cross-validation results only;
- rebuilds every confusion matrix from the saved predictions;
- recomputes the metrics;
- checks them against `results.json`.

It prints `Consistency checks: 12/12 passed` when all 12 model–task combinations agree. Any
mismatch is printed as `FAILED` and recorded in `checks.json`.

**`10_class_weight.py`** compares RF with and without class weighting, paired by seed.

## 8. Where each result of the article comes from

| Article | Produced by | File |
|---|---|---|
| Class mapping and counts (Table 2) | `09` | `table2_class_mapping_19class.*`, `class_mapping_6class.csv` |
| CV selection and test results of all candidates (Table 4) | `05`, `04`, `06`, `09` | `main_results.*`, `selection_by_cv.json` |
| Comparison with published results (Tables 5–7) | `09` + `reference/baseline_published.csv` | `reference_comparison.*`, `comparison_published.csv` |
| Ablation and seed study (Section 4.3) | `04`, `09` | `ablation_scaling.*`, `seeds_summary.*`, `seeds_individual.csv`, `seeds_diff_global_minus_none.*` |
| Confusion matrices (Table 8, Appendix C) | `09` | `cm_<task>_<model>.*` |
| Per-class metrics (Table 9, Appendix D) | `09` | `per_class_6class.*`, `per_class_19class.*` |
| Class weighting (Tables 9 and 10) | `10` | `class_weight_summary.*`, `class_weight_per_class_<task>.*` |
| Convergence and training time (Table 11) | `04`, `09` | `convergence.*` |

`09` writes its files to `results/paper_tables/`, and `10` to the folder given with `--out`. The
published values of the reference study are in `reference/baseline_published.csv`, transcribed
from the published article and checked by the authors. The values of other studies in Tables 5–7
were taken from those articles and are not generated by this code.

**Where the hyperparameters are.** They are defined in `get_configured_models()` of
`src/04_run_experiment.py` and listed in Appendix B of the article. `05_cross_validation.py`
imports them from there, so cross-validation and the final fit always use the same
configuration. The main runs use `random_state=42`.

## 9. Notes on reproducibility

- **Determinism.** Within the pinned environment the runs are deterministic: repeating a
  configuration gives identical confusion matrices.
- **Comparison with the original run.** Before this code was finalized, the main configuration
  was run in an environment that was not recorded. The rerun reproduced exactly:
  - RF in all tasks;
  - LR in the binary and 19-class tasks;
  - AdaBoost in the binary task.

  The other results differed:
  - LR at 6 classes, by less than 1e-5;
  - AdaBoost in the multiclass tasks and the MLP, by up to 0.024 in Macro F1-score.

  The cause of these differences could not be determined. All reported results come from the
  pinned environment.
- **Seeds.** `--seed` changes only the `random_state` of the models. The order of the training
  rows is fixed (shuffle with seed 42), so a seed run measures only the randomness of the model.
- **Tree models and scaling.** RF and AdaBoost results change slightly between the `global`,
  `none`, and `leaky` variants, although decision trees are insensitive to monotonic rescaling in
  theory. A plausible explanation, which was not tested, is that scikit-learn converts the
  features to `float32` for trees, so ties and split thresholds can differ after scaling.
- **The `per_file` variant is not a valid configuration.** Each CSV file contains a single class,
  so per-file statistics depend on the label. It is reported only as a diagnostic.
- **Options used only for sensitivity analyses:** `--scaling`, `--seed`, `--lr_max_iter`,
  `--mlp_max_iter`, and `--class_weight`. None of them is used to select a model.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Results differ slightly from the article | check the versions: `python -c "import sklearn, numpy, pandas; print(sklearn.__version__, numpy.__version__, pandas.__version__)"` must print `1.7.2 2.3.5 2.3.3`; otherwise `pip install -r requirements.txt` |
| `Edit PARTITION in slurm/submit_all.sh first.` | set `PARTITION` (and `ACCOUNT` if needed) in `slurm/submit_all.sh` |
| A job is killed for memory (`oom`, `Killed`) | raise `--mem` in the corresponding `.sbatch`; the values above were sufficient for the reported runs |
| `ConvergenceWarning` for LR | expected: with the reference setting `max_iter=100`, LR stops at the limit in all tasks; the warnings are recorded in `results.json` and analysed in Section 4.6 |
| `09` says `CV incomplete … selection is provisional` | not all 120 folds finished: check `results/cv/folds/` and rerun the missing folds with `--fold k` |
| A run was interrupted | `08` and `09` ignore incomplete runs and use the most recent complete one; delete the incomplete folder or rerun the configuration |
| Want to recompute a CV fold | `python src/05_cross_validation.py --train $TR --results_dir results/cv --fold k --overwrite` |

## 11. Repository layout

```
├── src/
│   ├── 00_smoke_test.py          end-to-end test on synthetic data
│   ├── 01_download_datasets.py   download the 51 + 21 CSV files
│   ├── 02_combine_train.py       build D_train
│   ├── 03_combine_test.py        build D_test
│   ├── 04_run_experiment.py      final fit + test evaluation (all variants)
│   ├── 05_cross_validation.py    stratified 10-fold CV on D_train
│   ├── 06_bootstrap_ci.py        bootstrap CI from saved predictions
│   ├── 07_validate_rebuilt.py    optional: check that two consolidated copies are identical
│   ├── 08_report.py              Markdown report of all runs
│   ├── 09_paper_tables.py        tables of the article + consistency checks
│   └── 10_class_weight.py        class-weighting comparison, paired by seed
├── slurm/                        one .sbatch per job, env.sh, submit_all.sh, status.sh
├── reference/
│   └── baseline_published.csv    published values of the reference study
├── requirements.txt              pinned versions
├── requirements-lock.txt         full environment of the reported runs
├── CITATION.cff
└── LICENSE
```

Folders created when you run the code, which are not versioned: `data/`, `results/`, `logs/`,
and `.venv/`.

## 12. License and citation

MIT License (see `LICENSE`). If you use this code, please cite the article (see `CITATION.cff`)
and the CICIoMT2024 dataset paper.
