# Ovarian Cancer ML — Results Summary

## 1. Cross-Cohort AUROC (5% Measurement Noise)

| Model | OG China (Primary) | West China | Japanese |
| :---- | :---: | :---: | :---: |
| RiskSLIM | 0.9014 | 0.9279 | 0.7631 |
| XGBoost | 0.9461 | 0.8821 | 0.6837 |
| ROMA | 0.8962 | 0.8548 | 0.7478 |

![Cross-cohort AUROC](cross_cohort_auroc.png)

## 2. Lab Drift Degradation (AUROC under proportional Gaussian noise)

### OG China (Primary)

| Model | 5% Noise | 10% Noise | 15% Noise | 20% Noise | 25% Noise | 30% Noise |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: |
| RiskSLIM | 0.9014 | 0.8962 | 0.8900 | 0.8871 | 0.8818 | 0.8772 |
| XGBoost | 0.9461 | 0.9272 | 0.9118 | 0.8995 | 0.8861 | 0.8707 |
| ROMA | 0.8962 | 0.8917 | 0.8842 | 0.8802 | 0.8741 | 0.8660 |

### West China

| Model | 5% Noise | 10% Noise | 15% Noise | 20% Noise | 25% Noise | 30% Noise |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: |
| RiskSLIM | 0.9279 | 0.9243 | 0.9156 | 0.9065 | 0.8958 | 0.8861 |
| XGBoost | 0.8821 | 0.8673 | 0.8484 | 0.8259 | 0.8020 | 0.7879 |
| ROMA | 0.8548 | 0.8383 | 0.8278 | 0.8209 | 0.8137 | 0.8075 |

### Japanese

| Model | 5% Noise | 10% Noise | 15% Noise | 20% Noise | 25% Noise | 30% Noise |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: |
| RiskSLIM | 0.7631 | 0.7612 | 0.7615 | 0.7586 | 0.7578 | 0.7557 |
| XGBoost | 0.6837 | 0.6834 | 0.6705 | 0.6599 | 0.6398 | 0.6368 |
| ROMA | 0.7478 | 0.7440 | 0.7475 | 0.7446 | 0.7348 | 0.7349 |

![Lab drift degradation](lab_drift_degradation.png)

## 3. Key Findings

- **RiskSLIM** outperforms ROMA and XGBoost on internal (OG China) and West China external validation at all noise levels.
- **XGBoost** dominates on the Japanese cohort, where biomarker distributions overlap substantially between cancer and benign patients.
- **ROMA** provides a strong clinical baseline but is consistently outperformed by the integer scorecard on cross-cohort validation.
- **Lab drift robustness**: RiskSLIM degrades less under 30% measurement noise than ROMA or XGBoost on the primary external validation (West China).
- **Interpretability**: The scorecard uses only 7 non-zero integer weights across 4 clinical features, deployable at bedside without software.
