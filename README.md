# Model Complexity Versus Transportability in Ovarian Cancer Risk Prediction

Repository for the IEEE BIBM 2026 Undergraduate & High School Symposium
(UGHS) submission: *"Model Complexity Versus Transportability in Ovarian
Cancer Risk Prediction"* (5 pages, IEEE conference format).

An eight-model benchmark (CA125 rule, ROMA, CPH-I, logistic regression, a
hand-computable integer scorecard, XGBoost, CatBoost, Random Forest)
trained frozen on a Chinese cohort (Changzhou, n = 349) and externally
validated on two Asian cohorts (West China, n = 380; Japan, n = 177), with
complete-case and imputed co-primary analyses, meta-analytic pooling,
weighted-harm decision analysis, calibration reporting, robustness
experiments, and an illustrative concept-drift simulation.

## Repository layout

```
bmib_hs/        BIBM UGHS submission: main.tex, compiled main.pdf,
                figures/ (Figures 1-3)
src/            all analysis code (see "How to Reproduce")
data/raw/       the three public cohort files as downloaded
data/processed/ analysis-ready feature/label files
results/        every CSV behind the paper's tables and figures
reproduce.sh    one-shot pipeline for all numbers and figures
```

## Environment setup

Python 3.13 with the pinned packages in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LaTeX (for the paper): Tectonic
(`cd bmib_hs && tectonic -X compile main.tex`) or Overleaf. The paper uses
the standard `IEEEtran` conference class with Times-compatible fonts
(`newtxtext`/`newtxmath`).

## Data access

All three cohorts are public and de-identified; no new data collection was
performed.

| Cohort | Role | Repository | URL |
|---|---|---|---|
| Chinese (n = 349; 171 cancer / 178 benign) | training | Mendeley Data | https://doi.org/10.17632/th7fztbrv9.11 |
| West China (n = 380; 188 cancer / 192 benign cysts) | external (co-primary) | figshare | https://doi.org/10.6084/m9.figshare.28831256 |
| Japan (n = 177; 27 cancer / 150 healthy) | specificity stress test | Karger figshare | https://doi.org/10.6084/m9.figshare.24235450 |

Download the files into `data/raw/` (the repository already contains them);
`data/processed/` holds the harmonized feature/label files produced by the
preprocessing step below.

## How to Reproduce

All commands run from the repository root. The full pipeline takes ~45
minutes, dominated by the simulation study (~25 min) and repeated
cross-validation (~10 min). A one-shot driver is `bash reproduce.sh`; the
individual steps are:

### 1. Preprocessing (raw → processed)

`data/processed/` ships precomputed with the repository (the exact files
used for every number in the paper):

```bash
python src/clean_data.py      # raw Chinese CSV -> chinese_train_cleaned.csv
python src/impute_data.py     # class-stratified KNN imputation
```

### 2. Numerical results behind the tables

```bash
# Table II, West (imputed) and Japan columns + pairwise bootstrap tests
python src/clinical_comparators.py

# Internal validity: repeated 5-fold CV, selection stability, nested CV
python src/repeated_cv.py

# Pooled random-effects meta-analysis (scorecard-vs-ROMA +0.043 comparison)
python src/meta_analysis.py

# Brier scores and calibration-in-the-large
python src/calibration_metrics.py

# Weighted-harm decision analysis (Table III base)
python src/decision_harm_analysis.py

# Robustness: noise types, outliers, learning curves
python src/robustness_extended.py

# Simulation study: regime-map data + factor decomposition (~25 min)
python src/simulation_study.py

# Setting B tree tuning grid (nested CV)
python src/tree_tuning_grid.py

# BIBM-specific artifacts: complete-case CIs, Setting B ensembles, MICE
# sensitivity, recalibrated-formula harms, probability-scale calibration
# slopes/ECE, and Figures 1-3 in bmib_hs/figures/
python src/make_bmib_artifacts.py
```

### 3. Paper

```bash
cd bmib_hs
tectonic -X compile main.tex
```

Expected output: exactly 5 pages (the UGHS high-school limit), 0 errors,
0 overfull warnings, two-column IEEE conference layout.

## Table/Figure mapping

| Paper item | Produced by |
|---|---|
| Table I (integer scorecard) | fitted by `src/bench_lib.py`; bins in `bmib_hs/main.tex` |
| Table II (AUROC + 95% CI) | `src/clinical_comparators.py` + `src/make_bmib_artifacts.py` (West CC and Setting B columns) |
| Table III (minimum harm) | `src/decision_harm_analysis.py` + `src/make_bmib_artifacts.py` (recalibrated formulas) |
| Figure 1 (complete-case forest plot) | `src/make_bmib_artifacts.py` |
| Figure 2 (decision curve analysis) | `src/make_bmib_artifacts.py` |
| Figure 3 (regime map) | `src/simulation_study.py` + `src/make_bmib_artifacts.py` |
| Calibration slopes/ECE | `src/make_bmib_artifacts.py` → `results/bmib_calibration_slope.csv` |
| MICE sensitivity | `src/make_bmib_artifacts.py` → `results/bmib_mice.csv` |

## License and citation

MIT (see LICENSE). If you use the data, cite the originating datasets and
this work:

```bibtex
@software{ovarian_complexity_transportability,
  title = {Model Complexity Versus Transportability in Ovarian Cancer Risk
           Prediction},
  year = {2026}
}
```
