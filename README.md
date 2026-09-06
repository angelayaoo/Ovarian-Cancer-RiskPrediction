# Model Complexity Versus Transportability in Ovarian Cancer Risk Prediction

Repository for the ML4H 2026 Proceedings-track submission: *"Model Complexity
Versus Transportability in Ovarian Cancer Risk Prediction: A Cross-Cohort
Benchmark with Decision Analysis."*

An eight-model benchmark (CA125 rule, ROMA, CPH-I, unbinned logistic
regression, a seven-weight integer scorecard, XGBoost, CatBoost, Random
Forest) trained frozen on a Chinese cohort (n = 349) and externally
validated on two Asian cohorts (West China, n = 380; Japan, n = 177), with
meta-analytic pooling, weighted-harm decision analysis, robustness
experiments, and a synthetic concept-drift simulation.

## Repository layout

```
paper/          ML4H submission: main.tex, ref.bib, jmlr class files,
                figures/ (Figures 1-5 + grayscale versions), compiled main.pdf
src/            all analysis code (see "How to Reproduce")
data/raw/       the three public cohort files as downloaded
data/processed/ analysis-ready feature/label files
results/        every CSV behind the paper's tables and figures
journal/        secondary BMC-journal manuscript artifacts (optional)
```

## Environment setup

Python 3.13 with the pinned packages in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LaTeX (for the paper): any of Tectonic (`tectonic -X compile main.tex`),
Overleaf (upload `paper/`), or TeX Live. The ML4H-modified `jmlr.cls` ships
inside `paper/`.

## Data access

All three cohorts are public and de-identified; no new data collection was
performed.

| Cohort | Role | Repository | URL |
|---|---|---|---|
| Chinese (n = 349; 171 cancer / 178 benign) | training | Mendeley Data | https://doi.org/10.17632/th7fztbrv9.11 |
| West China (n = 380; 188 cancer / 192 benign cysts) | external | figshare | https://doi.org/10.6084/m9.figshare.28831256 |
| Japan (n = 177; 27 cancer / 150 healthy) | external stress test | Karger figshare | https://doi.org/10.6084/m9.figshare.24235450 |

Download the files into `data/raw/` (the repository already contains them);
`data/processed/` holds the harmonized feature/label files produced by the
preprocessing step below.

## How to Reproduce

All commands run from the repository root. Runtime notes: the full pipeline
takes ~45 minutes, dominated by the simulation study (~25 min) and repeated
cross-validation (~10 min).

### 1. Preprocessing (raw → processed)

`data/processed/` ships precomputed with the repository (the exact files used
for every number in the paper). The Chinese-cohort preprocessing is fully
reproducible and verified byte-identical:

```bash
python src/clean_data.py      # raw Chinese CSV -> chinese_train_cleaned.csv
python src/impute_data.py     # class-stratified KNN imputation
```

### 2. Numerical results behind Tables 1-4

```bash
# Table 1  (AUROC by model and cohort, 0% noise; noise grid + pairwise tests)
python src/clinical_comparators.py

# Internal validity: repeated 5-fold CV, 1-SE selection stability, nested CV
python src/repeated_cv.py

# Table 2  (random-effects pooling across the two external cohorts)
python src/meta_analysis.py

# Calibration, PR-AUC, NRI/IDI (cited in Results)
python src/calibration_metrics.py

# Table 3  (weighted-harm decision analysis) + consequence-curve data
python src/decision_harm_analysis.py

# DCA with bootstrap bands + operating points
python src/decision_analysis.py

# Robustness: noise types, outliers, cutpoint perturbation, learning curves
python src/robustness_extended.py

# Table 4  (factor decomposition) + regime-map data + gamma sweep (~25 min)
python src/simulation_study.py

# Optional sensitivities cited in the paper
python src/select_config.py
python src/tree_tuning_grid.py
```

### 3. Figures 1-5

```bash
python src/make_ml4h_figures.py
```

writes `paper/figures/fig_ladder_col.png`, `fig_consequences.png`,
`fig_noise_col.png`, `fig_learning.png`, `fig_regime_col.png` and their
grayscale companions. A one-shot driver for the whole pipeline is
`bash reproduce.sh`.

### 4. Paper

```bash
cd paper
tectonic -X compile main.tex      # or build in Overleaf / LaTeX Workshop
```

Expected output: 9-10 pages (8-page body limit for Proceedings), 0 errors,
0 overfull warnings, two-column layout.

## Table/Figure mapping

| Paper item | Produced by |
|---|---|
| Table 1 (AUROC by model/cohort) | `src/clinical_comparators.py` |
| Table 2 (pooled differences) | `src/meta_analysis.py` |
| Table 3 (weighted harm) | `src/decision_harm_analysis.py` |
| Table 4 (factor decomposition) | `src/simulation_study.py` → `results/factor_decomposition.csv` |
| Figure 1 (complexity ladder) | `src/make_ml4h_figures.py` |
| Figure 2 (consequence curves) | `src/make_ml4h_figures.py` |
| Figure 3 (noise degradation) | `src/make_ml4h_figures.py` |
| Figure 4 (learning curves) | `src/make_ml4h_figures.py` |
| Figure 5 (regime map) | `src/make_ml4h_figures.py` |

## License and citation

MIT (see LICENSE). If you use the data, cite the originating datasets and
this work:

```bibtex
@software{ovarian_complexity_transportability,
  title = {Model Complexity Versus Transportability in Ovarian Cancer Risk
           Prediction: A Cross-Cohort Benchmark with Decision Analysis},
  year = {2026}
}
```
