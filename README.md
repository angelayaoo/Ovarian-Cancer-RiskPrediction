# Model Complexity Versus Transportability in Ovarian Cancer Risk Prediction

**Jiayi Yao** (BASIS Independent Bellevue, Bellevue, WA, USA),
**Zooey Lane Go Hua** (Issaquah High School, Issaquah, WA, USA),
**Vritika S Sharma** (Issaquah High School, Issaquah, WA, USA)

Submission to the IEEE BIBM 2026 Undergraduate & High School Symposium
(UGHS)

## Overview

Eight models (a CA125 cutoff rule, the clinical formulas ROMA and CPH-I, a
logistic regression, a hand-computable integer scorecard, and three tree
ensembles) are trained once on a Chinese cohort (Changzhou, n = 349) and
applied unchanged to two independent Asian cohorts (West China, n = 380;
Japan, n = 177). Complete-case (measured-HE4) and imputed co-primary
external benchmarks, a random-effects meta-analysis, weighted-harm
decision analysis, calibration reporting, robustness experiments, and an
illustrative concept-drift simulation show that a seven-parameter integer
scorecard matches more complex models' discrimination while remaining
transparent, auditable, and computable by hand.

## Environment setup

```bash
git clone https://github.com/angelayaoo/Ovarian-Cancer-RiskSLIM.git
cd Ovarian-Cancer-RiskSLIM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact environment used to produce every
reported number (Python 3.13; scikit-learn 1.7.2, XGBoost 3.3.0,
CatBoost 1.2.10, numpy 2.3.5, pandas 2.3.3, matplotlib 3.10.6,
scipy 1.16.3, statsmodels 0.14.5, pillow 12.3.0). All analyses use
`random_state=42` (`src/bench_lib.py`, `SEED = 42`).

## Execution

```bash
bash reproduce.sh
```

runs the full pipeline end-to-end (~45 min; the simulation study is the
slow part) and writes every CSV behind Tables I--III and Figures 1--3.
To rebuild the paper PDF:

```bash
cd bmib_hs
tectonic -X compile main.tex
```

Expected output: exactly 5 pages, 0 errors, 0 overfull warnings.

## Directory tree

```
.
├── bmib_hs/                  # BIBM UGHS paper (main.tex, main.pdf, figures/)
│   └── figures/              # fig1_auroc.png, fig2_cons_col.png,
│                             # fig3_regime_col.png (generated)
├── src/                      # all analysis code (see reproduce.sh)
├── data/
│   ├── raw/                  # the three public cohort files as downloaded
│   └── processed/            # harmonized feature/label files (precomputed)
├── results/                  # every CSV behind the tables and figures
├── reproduce.sh              # one-shot pipeline
├── requirements.txt          # pinned dependencies
├── LICENSE                   # MIT
└── README.md
```

## Data availability

All three cohorts are public and de-identified by their originators; no new
data collection was performed. The raw files ship in `data/raw/` exactly as
published (they are not downloaded automatically):

| Cohort | Role | Repository | URL |
|---|---|---|---|
| Chinese (n = 349; 171 cancer / 178 benign) | training | Mendeley Data | https://doi.org/10.17632/th7fztbrv9.11 |
| West China (n = 380; 188 cancer / 192 benign cysts) | external (co-primary) | figshare | https://doi.org/10.6084/m9.figshare.28831256 |
| Japan (n = 177; 27 cancer / 150 healthy) | specificity stress test | Karger figshare | https://doi.org/10.6084/m9.figshare.24235450 |

`data/processed/` contains the harmonized four-feature files used for every
analysis; `src/clean_data.py` and `src/impute_data.py` regenerate them from
the raw files.

## Table/Figure mapping

| Paper item | Produced by |
|---|---|
| Table I (integer scorecard) | fitted by `src/bench_lib.py` |
| Table II (AUROC + 95% CI) | `src/clinical_comparators.py` + `src/make_bmib_artifacts.py` |
| Table III (minimum harm) | `src/decision_harm_analysis.py` + `src/make_bmib_artifacts.py` |
| Figures 1--3 | `src/make_bmib_artifacts.py` (+ `src/simulation_study.py` for Fig. 3 data) |
| Calibration slopes/ECE | `src/make_bmib_artifacts.py` → `results/bmib_calibration_slope.csv` |
| MICE sensitivity | `src/make_bmib_artifacts.py` → `results/bmib_mice.csv` |

## Review policy

The BIBM 2026 UGHS review process is single-blind (reviewers anonymous,
authors visible), so public attribution in this repository is permitted.

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
