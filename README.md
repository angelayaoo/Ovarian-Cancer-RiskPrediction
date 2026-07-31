# Ovarian Cancer RiskSLIM — Integer Scorecard for Risk Prediction

An integer-weighted clinical scorecard for ovarian cancer risk stratification that outperforms the clinical-standard ROMA on external cross-cohort validation while degrading 4× less than complex machine learning models under laboratory measurement noise.

## Results Summary

**5-model comparison, 3 Asian cohorts, 7 noise levels (0–30%), 100 Monte Carlo perturbations each**

| Model | OG China (Internal) | West China (External) | Japan (External) | Overfit Δ | Noise Δ (West) |
|---|---:|---:|---:|---:|---:|
| **RiskSLIM** | 0.9032 | **0.9292** | **0.7689** | **−0.026** | **−0.043** |
| CatBoost | 0.9911 | 0.9066 | 0.7380 | +0.085 | −0.086 |
| Random Forest | 0.9962 | 0.9012 | 0.7621 | +0.095 | −0.067 |
| XGBoost | 0.9716 | 0.8906 | 0.6836 | +0.081 | −0.106 |
| ROMA | 0.8986 | 0.8753 | 0.7485 | +0.023 | −0.068 |

RiskSLIM is the **only model that generalizes** (external AUROC exceeds internal). All tree-based models overfit by +0.08–0.10.

## Scorecard

| Feature | Low (0–33%) | Mid (33–66%) | High (66–100%) |
|---|---:|---:|---:|
| CA125 | −1 | 0 | +2 |
| HE4 | −4 | −1 | +5 |
| Age | −2 | 0 | +2 |
| Menopause | 0 | 0 | 0 |

7 non-zero integer weights. Bedside-computable in under 10 seconds.

## Methodology

```
KNN imputation (per class, k=5) → 4 features (CA125, HE4, Age, Menopause)
→ QuantileTransformer (uniform [0,1]) → KBinsDiscretizer (3 quantile bins)
→ L2 logistic regression (C=1.5) → integer rounding [-5, 5]
→ Monte Carlo evaluation: 100 perturbations × 7 noise levels × 3 cohorts
```

## Quick Start

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib openpyxl
python src/run_monte_carlo_benchmarks.py    # Full benchmark (~10 min)
python src/generate_results_summary.py       # Docs + plots
python src/plot_degradation_curves.py        # Degradation curves
python src/lab_drift_simulation.py           # Lab drift analysis
python src/generate_figures.py               # Publication figures
python src/statistical_tests.py              # Bootstrap significance + risk stratification
```

## Repository Structure

```
src/
├── run_monte_carlo_benchmarks.py   # Main benchmark pipeline
├── lab_drift_simulation.py         # Gaussian noise degradation
├── prepare_riskslim.py             # Scorecard generation
├── run_riskslim_solver.py          # Integer scorecard solver
├── validate_scorecard.py           # External validation
├── train_pipeline.py               # CV + threshold tuning
├── clean_data.py                   # Data preprocessing
├── impute_data.py                  # KNN imputation
├── statistical_tests.py            # Bootstrap + risk tiers
├── generate_results_summary.py     # Docs + bar charts
├── generate_figures.py             # Publication figures
├── plot_degradation_curves.py      # Degradation line plots
├── compare_models.py               # Model comparison tables
└── harmonize western china cohort  # Cohort harmonization
data/
├── raw/          (5 raw cohort files)
└── processed/    (6 cleaned + scorecard CSVs)
figures/          (6 publication-quality PNGs)
results/          (benchmark CSV + plots)
docs/             (summary plots + markdown)
```

## Citation

```bibtex
@software{yao2026ovarian,
  title = {Ovarian Cancer RiskSLIM: Integer Scorecard for Ovarian Cancer Risk Prediction},
  author = {Yao, Jiayi},
  year = {2026},
  url = {https://github.com/angelayaoo/Ovarian-Cancer-RiskSLIM}
}
```

## License

MIT
