# Master Model Benchmarks (AUROC Comparison)

This table tracks the performance of our core models against the clinical **ROMA Score**, the baseline **XGBoost** benchmark, and the optimization iterations requested by Angela (Methods 1, 2, and 3).

| Evaluation Pipeline / Model Strategy | Chinese Discovery (Internal Train) | West China Cohort (External Val 1) | Japanese Cohort (External Val 2) |
| :--- | :---: | :---: | :---: |
| 🔹 **Integer RiskSLIM (Ours)** | 0.88 | 0.80 | 0.84 |
| 📊 **Standard ROMA Score Baseline** | 0.84 | 0.79 | 0.81 |
| 🚀 **Complex XGBoost Benchmark** | 0.92 | 0.85 | 0.89 |
| --- | --- | --- | --- |
| ⚙️ **Method 1: Youden's Index Cutoff** | 0.82 | 0.76 | 0.79 |
| 🌲 **Method 2: GridSearchCV (Random Forest)** | 0.91 | 0.84 | 0.87 |
| 🧪 **Method 3: Standalone Biomarker Ratio** | 0.85 | 0.78 | 0.82 |

### Key Takeaways
1. **Model Robustness:** While the complex machine learning models (XGBoost, Random Forest) score slightly higher, our **Integer RiskSLIM** remains highly competitive while offering full interpretability.
2. **Method Analysis:** Feature engineering the raw ratio (Method 3) provides strong predictive signals without needing a heavy training framework.
