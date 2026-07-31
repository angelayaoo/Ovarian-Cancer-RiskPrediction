
Ovarian Cancer Machine Learning & Diagnostic Benchmarks

This repository contains the clinical data pipeline and model benchmarking suite for evaluating ovarian cancer risk models. It compares the standard ROMA calculation against XGBoost and RiskSLIM models under simulated assay noise and cross-cohort evaluation.

File & Directory Guide


Main Pipeline & Benchmarks (\src/\)
* \
un_monte_carlo_benchmarks.py\ - Runs the main 1,000-iteration noise test (5% to 20% measurement error) across the primary China, West China, and Japanese cohorts.
* \	rain_model.py\ - Trains the core XGBoost and RiskSLIM classification models.
* \	rain_pipeline.py\ - End-to-end script handling preprocessing, feature building, and model training.
* \evaluate_all_cohorts.py\ - Evaluates baseline performance across all cohort validation splits.
* \lab_drift_simulation.py\ - Evaluates model performance when continuous lab features experience systematic drift.
* \generate_master_table.py\ - Pulls evaluation results together to build summary output tables.


Data Processing & Utilities (\src/\)
* \clean_data.py\ - Filters and standardizes raw input data files.
* \impute_data.py\ - Fills missing biomarker and laboratory values.
* \prepare_riskslim.py\ - Discretizes continuous features into bins required by RiskSLIM.
* \
un_riskslim_solver.py\ - Optimization solver that builds the integer point scorecards.


Data Directories
* \data/processed/\ - Cleaned and formatted CSVs (\chinese_discovery_processed.csv\, \west_china_processed.csv\, \japan_processed.csv\).

---

How to Run
To run the primary noise benchmark script:
\\\ash
python src/run_monte_carlo_benchmarks.py

Master Results Table

| Cohort                      |   Samples |   ROMA AUROC |   XGBoost AUROC |   RiskSLIM AUROC |
|:----------------------------|----------:|-------------:|----------------:|-----------------:|
| Chinese Primary (Discovery) |       235 |       0.0994 |          0.96   |            0.887 |
| West China Validation       |       188 |       0.5353 |          0.5247 |            0.38  |
| Japanese Cohort             |        27 |     nan      |        nan      |          nan     |

## Master Benchmarks Table (Empirical Fixed-Model Noise Evaluation)

| Cohort | Noise Level | ROMA Formula (AUROC) | XGBoost (AUROC) | Optimized RiskSLIM (AUROC) |
| :--- | :---: | :---: | :---: | :---: |
| **OG China (Primary)** | 5% | 0.8989 (±0.003) | 0.9413 (±0.006) | 0.8928 (±0.005) |
| **OG China (Primary)** | 10% | 0.8960 (±0.006) | 0.9281 (±0.008) | 0.8907 (±0.006) |
| **OG China (Primary)** | 15% | 0.8925 (±0.008) | 0.9162 (±0.010) | 0.8865 (±0.008) |
| **OG China (Primary)** | 20% | 0.8865 (±0.010) | 0.9086 (±0.011) | 0.8839 (±0.009) |
| **West China** | 5% | 0.5352 (±0.002) | 0.4729 (±0.004) | 0.6203 (±0.006) |
| **West China** | 10% | 0.5352 (±0.004) | 0.4736 (±0.006) | 0.6203 (±0.011) |
| **West China** | 15% | 0.5343 (±0.005) | 0.4735 (±0.007) | 0.6190 (±0.010) |
| **West China** | 20% | 0.5319 (±0.007) | 0.4768 (±0.008) | 0.6191 (±0.015) |
| **Japanese** | 5% | nan (±nan) | nan (±nan) | nan (±nan) |
| **Japanese** | 10% | nan (±nan) | nan (±nan) | nan (±nan) |
| **Japanese** | 15% | nan (±nan) | nan (±nan) | nan (±nan) |
| **Japanese** | 20% | nan (±nan) | nan (±nan) | nan (±nan) |

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib openpyxl

# Run the full Monte Carlo benchmark
python src/run_monte_carlo_benchmarks.py

# Generate results plots
python src/generate_results_summary.py
python src/plot_degradation_curves.py

# Run lab drift simulation
python src/lab_drift_simulation.py
```

## Citation

If you use this work, please cite:

```bibtex
@software{yao2026ovarian,
  title = {Ovarian Cancer RiskSLIM: Integer Scorecard for Ovarian Cancer Risk Prediction},
  author = {Yao, Jiayi},
  year = {2026},
  url = {https://github.com/yaojiayi2020/Ovarian-Cancer-RiskSLIM}
}
```
