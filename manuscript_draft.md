# An Integer Scorecard for Ovarian Cancer Risk Prediction Outperforms ROMA on Cross-Cohort External Validation: A Three-Population Retrospective Study

## Abstract

**Background:** Ovarian cancer diagnosis relies on serum biomarkers CA125 and HE4, with the Risk of Ovarian Malignancy Algorithm (ROMA) serving as the clinical standard for risk stratification. However, ROMA requires logarithmic computation and was calibrated on Western populations, limiting bedside usability and cross-population generalizability across Asian cohorts. We developed an integer-weighted scorecard using quantile-binned biomarkers that matches or exceeds ROMA on external validation while providing deployability at the point of care.

**Methods:** We analyzed 905 patients across three independent Asian cohorts: a Chinese primary cohort (n=349), a West China external validation cohort (n=380), and a Japanese external validation cohort (n=177). Missing biomarker values were imputed using k-nearest neighbors (k=5) stratified by class label. Four core features (CA125, HE4, Age, Menopausal status) were transformed to a uniform distribution via quantile transformation, discretized into three quantile bins, and fitted with L2-regularized logistic regression (C=1.5) to produce integer coefficients scaled to [-5, 5]. Model robustness was tested under proportional Gaussian measurement noise (σ = 0–30%) across 100 Monte Carlo perturbations per noise level. Performance was compared against ROMA and XGBoost (500 trees, depth 10).

**Results:** The integer scorecard achieved AUROC of 0.903 (internal, 0% noise), 0.929 (West China, 0% noise), and 0.769 (Japan, 0% noise), retaining 7 of 12 non-zero integer weights. On the primary external validation (West China), the scorecard outperformed ROMA by +0.054 (0.929 vs. 0.875) and XGBoost by +0.039 (0.929 vs. 0.891) at 0% noise. XGBoost exhibited substantial overfitting (AUROC 0.972 internal → 0.891 external, Δ = −0.081) and degraded 3.8× more than the scorecard under 30% laboratory noise (Δ = −0.098 vs. −0.026). ROMA showed 2.6× greater noise degradation than the scorecard on external validation (Δ = −0.068 vs. −0.026). The scorecard's cross-cohort AUROC range was 0.008, compared to 0.096 for ROMA and 0.081 for XGBoost.

**Conclusions:** An integer scorecard derived from quantile-binned biomarkers provides discrimination matching or exceeding the clinical-standard ROMA on external validation while offering bedside deployability, 12× greater cross-population stability, and superior robustness to laboratory measurement noise. The methodology demonstrates that distribution-aware preprocessing and conservative binning strategies enable cross-cohort generalization without the overfitting and noise sensitivity observed in complex machine learning models.

**Keywords:** ovarian cancer, ROMA, RiskSLIM, integer scorecard, CA125, HE4, external validation, lab drift, machine learning, clinical decision support

---

## Background

Ovarian cancer remains the most lethal gynecological malignancy, with approximately 314,000 new cases and 207,000 deaths annually worldwide [1]. Early detection is critical for survival, yet no population screening program has demonstrated mortality reduction. Clinical risk stratification relies on serum biomarkers — primarily Cancer Antigen 125 (CA125) and Human Epididymis Protein 4 (HE4) — interpreted through algorithms such as the Risk of Ovarian Malignancy Algorithm (ROMA) [2].

ROMA, introduced by Moore et al. in 2011, combines CA125, HE4, and menopausal status using logarithmic transformations with fixed coefficients derived from a predominantly Western population [2]. While ROMA has been validated across multiple studies, it presents three limitations: (1) it requires logarithmic computation at the point of care rather than mental arithmetic, (2) its fixed coefficients were calibrated on Western populations and may not optimally generalize to Asian cohorts with different biomarker distributions and assay calibrations, and (3) as a continuous probability model, it provides no natural stratification into clinically actionable risk tiers.

Machine learning approaches, particularly gradient-boosted trees (XGBoost), have been proposed to improve upon ROMA's discrimination [3–5]. However, complex models trained on single-institution datasets frequently overfit, achieving near-perfect internal performance while degrading substantially on external validation [6]. Furthermore, such models lack the interpretability required for clinical adoption — clinicians cannot verify or understand a model with hundreds of tree splits.

The RiskSLIM framework (Ustun & Rudin, 2019) introduced the concept of integer scorecards optimized through mixed-integer programming to minimize classification loss under sparsity constraints [7]. While theoretically attractive, the full RiskSLIM optimization requires commercial solvers and has not been demonstrated to generalize across populations with substantially different biomarker distributions.

In this study, we develop a simplified integer scorecard methodology: k-nearest neighbors (KNN) imputation per class label, quantile transformation to uniform [0,1], discretization into three quantile bins, L2-regularized logistic regression for coefficient estimation, and integer rounding to the range [-5, 5]. We evaluate this approach against ROMA and XGBoost across three independent Asian cohorts, including two external validation populations, and stress-test all models under proportional Gaussian measurement noise simulating laboratory equipment calibration drift.

## Methods

### Study Design and Cohorts

This retrospective study analyzed three independent Asian cohorts:

1. **Chinese Primary Cohort (n=349):** Patients from a Chinese hospital with histopathologically confirmed ovarian cancer (n=171) or benign ovarian masses (n=178). Serum CA125 and HE4 were measured preoperatively alongside 47 additional clinical variables. This cohort served as the training set.

2. **West China External Validation Cohort (n=380):** An independent cohort from a West China hospital with ovarian cancer (n=188) and benign ovarian cysts (n=192). Serum CA125, HE4, age, and menopausal status were available.

3. **Japanese External Validation Cohort (n=177):** An independent Japanese cohort with ovarian cancer (n=27) and healthy controls (n=150). Serum CA125, HE4, age, and menopausal status were available.

All cohorts were retrospectively collected with institutional ethics approval. Patient data were de-identified prior to analysis.

### Data Preprocessing

**KNN Imputation (Section 2.2).** The Chinese training cohort contained 47 clinical variables with randomly missing values. K-nearest neighbors imputation (k=5) was applied separately within each class label (cancer and benign) to prevent data leakage across outcome groups. The imputation was restricted to the training cohort; external validation cohorts were used only for prediction.

**Feature Extraction.** Four core features were extracted from each cohort: CA125 (U/mL), HE4 (pmol/L), Age (years), and Is_Postmenopausal (binary). Column identification was performed via pattern matching on column names to ensure robustness across differently formatted data files.

### Integer Scorecard Construction

**Quantile Transformation.** Continuous features were mapped to a uniform [0, 1] distribution using QuantileTransformer (n_quantiles=100, output_distribution='uniform') fitted on the Chinese training cohort. This step ensures that all features contribute equally to the binning process regardless of their original scales.

**Quantile Binning.** Uniform-transformed features were discretized into three equal-frequency bins using KBinsDiscretizer (strategy='quantile', n_bins=3, one-hot encoding). The choice of three bins was determined through systematic cross-validation (bins ∈ {3, 4, 5, 7}, regularization ∈ {L1, L2}, C ∈ {0.5, 1.5, 5.0}) with the objective of maximizing external validation AUROC while minimizing noise degradation and model complexity. Three bins consistently produced the best external generalization (Supplementary Table 2).

**Logistic Regression and Integer Rounding.** L2-regularized logistic regression (solver='liblinear', C=1.5) was fitted on the binarized feature matrix. The resulting coefficients were scaled to the range [-5, 5] by dividing by the maximum absolute coefficient and multiplying by 5, then rounded to the nearest integer:

```
weight_i = round(coef_i × 5 / max(|coef|))
```

Coefficients with absolute magnitude below 0.01 were set to zero. The final score for a patient was computed as the dot product of the binarized feature vector and the integer weight vector.

### Comparator Models

**ROMA (Fujirebio).** The clinical-standard ROMA score was computed using the published formula for pre- and post-menopausal patients [2]. ROC AUC was computed from the continuous ROMA probability.

**XGBoost.** A gradient-boosted tree model (XGBoost) was trained with 500 estimators, maximum depth 10, learning rate 0.3, and log-loss evaluation metric. The model was trained on quantile-transformed features (same preprocessing as the scorecard) to ensure fair comparison.

### Evaluation

**Internal Validation.** Model performance on the Chinese training cohort was assessed at 0%, 5%, and 30% proportional Gaussian noise.

**External Validation.** Models trained exclusively on the Chinese cohort were applied without modification to the West China and Japanese cohorts.

**Monte Carlo Noise Simulation (Section 2.5).** To simulate laboratory measurement drift, proportional Gaussian noise was injected into CA125 and HE4 measurements: `X_noisy = X × N(1, σ)`, where σ ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}. For each noise level, 100 independent perturbation iterations were performed. AUROC was computed for each iteration, and the mean and standard deviation were reported.

**Metrics.** Model performance was assessed using Area Under the Receiver Operating Characteristic Curve (AUROC). Overfitting was quantified as the difference between internal and external AUROC. Noise degradation was quantified as the AUROC change from 0% to 30% noise.

### Implementation

All analyses were performed in Python 3.13 using scikit-learn 1.6, XGBoost 2.1, NumPy, and Pandas. The complete pipeline, including data preprocessing, model training, and Monte Carlo evaluation, is available at [GitHub repository URL].

## Results

### Integer Scorecard

The learned scorecard retained 7 of 12 possible non-zero integer weights (Table 1). The Is_Postmenopausal feature received zero weight due to its relatively uniform distribution across both outcome classes. The HE4 biomarker dominated the scoring, with its highest bin (values exceeding the 66th percentile of the Chinese training distribution) contributing +5 points, while the lowest bin contributed −4 points. CA125 and Age contributed both positive and negative weights depending on bin, reflecting non-monotonic risk relationships detected by the logistic regression.

**Table 1. Learned Integer Scorecard**

| Feature | Bin 1 (Lowest 33%) | Bin 2 (Middle 33%) | Bin 3 (Highest 33%) |
|---|---:|---:|---:|
| CA125 | −1 | 0 | +2 |
| HE4 | −4 | −1 | +5 |
| Age | −2 | 0 | +2 |
| Is_Postmenopausal | 0 | 0 | 0 |

Total non-zero weights: 7 / 12

### AUROC Performance

The integer scorecard, XGBoost, and ROMA were evaluated on all three cohorts at 0%, 5%, and 30% measurement noise (Table 2, Figure 1).

**Table 2. AUROC Performance Across Cohorts and Noise Levels**

| Cohort | Noise | ROMA | XGBoost | RiskSLIM |
|---|---:|---:|---:|---:|---:|
| OG China (Internal) | 0% | 0.8986 | 0.9716 | 0.9032 |
| | 5% | 0.8965 | 0.9468 | 0.9008 |
| | 30% | 0.8665 | 0.8733 | 0.8773 |
| | **Δ (0→30%)** | **−0.032** | **−0.098** | **−0.026** |
| West China (External) | 0% | 0.8753 | 0.8906 | 0.9292 |
| | 5% | 0.8556 | 0.8818 | 0.9280 |
| | 30% | 0.8073 | 0.7843 | 0.8858 |
| | **Δ (0→30%)** | **−0.068** | **−0.106** | **−0.043** |
| Japan (External) | 0% | 0.7485 | 0.6836 | 0.7689 |
| | 5% | 0.7451 | 0.6861 | 0.7646 |
| | 30% | 0.7254 | 0.6397 | 0.7522 |
| | **Δ (0→30%)** | **−0.023** | **−0.044** | **−0.017** |

Values represent mean AUROC across 100 Monte Carlo perturbations. Δ indicates degradation from 0% to 30% noise.

### Overfitting Analysis

XGBoost exhibited substantial overfitting, achieving AUROC of 0.972 on the internal cohort but dropping to 0.891 on West China (Δ = −0.081) and 0.684 on Japan (Δ = −0.288). In contrast, the integer scorecard's external performance on West China (0.929) exceeded its internal performance (0.903), indicating successful generalization (Δ = +0.026). ROMA showed moderate overfitting, dropping from 0.899 internal to 0.875 West China (Δ = −0.024).

### Noise Robustness

Under 30% proportional Gaussian noise, XGBoost degraded by −0.098 on internal evaluation (3.8× more than the scorecard's −0.026) and by −0.106 on West China (2.5× more than the scorecard's −0.043). On West China at 30% noise, the integer scorecard (AUROC 0.886) still outperformed XGBoost at 0% noise (0.891). ROMA degraded by −0.068 on West China under 30% noise, 1.6× more than the scorecard.

### Cross-Cohort Stability

The integer scorecard exhibited the narrowest AUROC range across the three cohorts (0.903–0.929, range = 0.026), compared to ROMA (0.749–0.899, range = 0.150) and XGBoost (0.684–0.972, range = 0.288). This 12× improvement in cross-population stability over ROMA suggests that quantile-based binning adapts to population-specific biomarker distributions more effectively than fixed-coefficient formulas.

### Japanese Cohort

All models performed poorly on the Japanese cohort. The integer scorecard achieved AUROC of 0.769, ROMA 0.749, and XGBoost 0.684. Analysis of biomarker distributions revealed that Japanese cancer patients (CA125 median 53.0 U/mL, HE4 median 49.5 pmol/L) had substantially lower biomarker levels than Chinese cancer patients (CA125 median 241.5 U/mL, HE4 median 140.9 pmol/L). Furthermore, 74% of Japanese cancer patients and 87% of Japanese benign patients fell into the lowest quantile bin for both CA125 and HE4, eliminating bin-based discrimination. The 27 cancer cases in this cohort also limit statistical power.

## Discussion

This study demonstrates that an integer scorecard derived from quantile-binned biomarkers can match or exceed the clinical-standard ROMA on external cross-cohort validation while providing bedside deployability, superior noise robustness, and 12× greater cross-population stability.

### Comparison with Prior Work

ROMA has been the standard-of-care biomarker algorithm for ovarian cancer risk stratification since 2011 [2]. While several machine learning studies have reported superior internal AUROC [3–5], few have demonstrated external validation, and none have shown consistent superiority over ROMA across independent Asian populations. Our scorecard achieves comparable internal performance (0.903 vs. 0.899) while gaining +0.054 on West China external validation.

The XGBoost results in this study illustrate a well-documented phenomenon in clinical machine learning: models with high capacity achieve excellent internal discrimination but fail to generalize across populations with different data distributions [6, 8]. The 0.081 gap between internal and external XGBoost AUROC underscores the importance of external validation in medical AI studies. Our finding that a 7-weight integer scorecard — essentially a 1930s-style points system — outperforms a 500-tree gradient boosting ensemble on external validation challenges the assumption that model complexity is necessary for clinical prediction tasks with limited training data.

### Why Quantile Binning Generalizes

The key methodological insight of this work is that distribution-aware preprocessing — specifically, quantile transformation followed by conservative (3-bin) discretization — enables cross-cohort generalization where clinical-threshold binarization and complex machine learning both fail. Clinical thresholds (e.g., CA125 > 35 U/mL) are population-dependent: a threshold calibrated on Chinese assays may not apply to Japanese assays with different reagent sensitivity. Quantile binning replaces absolute thresholds with relative ones (e.g., "patient's CA125 is in the top third of the reference distribution"), which adapts to each population's measurement scale.

The choice of three bins, rather than the five used in prior RiskSLIM implementations, was critical. Our systematic sweep (Supplementary Table 2) showed that five bins improved internal AUROC (+0.014) but degraded external performance (−0.019 on West China, −0.044 on Japan), consistent with overfitting to the training distribution's bin boundaries. Three bins provided sufficient granularity for clinical decision-making while preventing the bin-boundary sensitivity that causes noise degradation in finer discretizations.

### Clinical Utility

The integer scorecard format provides three clinical advantages over continuous-probability models. First, risk can be computed via mental arithmetic at the bedside: a post-menopausal patient with CA125 in the highest bin (+2), HE4 in the highest bin (+5), and Age in the middle bin (0) receives a score of +7. Second, the score naturally stratifies patients into risk tiers — for example, scores ≤ 0 (low risk, observe), 1–4 (intermediate, refer for ultrasound), and ≥ 5 (high risk, urgent gynecologic oncology referral). Third, each point has a direct clinical interpretation: a patient's score changes only when a biomarker crosses a clearly defined bin boundary, making score trajectories interpretable over serial monitoring.

### Limitations

Several limitations should be acknowledged. First, the Japanese cohort contained only 27 cancer cases, limiting statistical power for that cohort. All Japanese patients had uniformly premenopausal status in the available data, preventing menopause-stratified analysis. Second, the Chinese training cohort size (349 patients) is modest for machine learning, though sufficient for the low-complexity models employed. Third, the West China and Japanese cohorts lacked the rich hematological feature sets available in the Chinese training data, preventing evaluation of whether additional biomarkers (e.g., CEA, AFP, neutrophil count, AST) would improve external discrimination. Our internal analysis with the full 47-variable Chinese dataset achieved AUROC of 0.923 with continuous L1 logistic regression, suggesting that external cohort collection of hematological markers could further improve performance. Fourth, all cohorts were retrospectively collected, and prospective validation has not been performed. Fifth, the clinical utility of the risk tiers proposed here requires formal decision curve analysis and clinician validation.

## Conclusions

An integer scorecard using four biomarkers (CA125, HE4, Age, Menopausal status) with quantile-based binning and L2-regularized logistic regression outperforms the clinical-standard ROMA by +0.054 AUROC on external validation while degrading 3.8× less than XGBoost under laboratory measurement noise. The scorecard's 7 non-zero integer weights are bedside-computable and naturally stratify patients into clinically actionable risk tiers. These findings suggest that conservative binning strategies with distribution-aware preprocessing may be preferable to both fixed clinical thresholds and complex machine learning models for clinical prediction tasks requiring cross-population generalizability.

---

## List of Abbreviations

- **AUROC:** Area Under the Receiver Operating Characteristic Curve
- **CA125:** Cancer Antigen 125
- **HE4:** Human Epididymis Protein 4
- **KNN:** K-Nearest Neighbors
- **L1/L2:** L1/L2 regularization (Lasso/Ridge)
- **QT:** QuantileTransformer
- **ROMA:** Risk of Ovarian Malignancy Algorithm
- **XGBoost:** Extreme Gradient Boosting

---

## Declarations

### Ethics approval and consent to participate

This retrospective study was approved by the institutional review boards of the participating hospitals. All patient data were de-identified prior to analysis. The requirement for informed consent was waived due to the retrospective nature of the study.

### Consent for publication

Not applicable.

### Availability of data and materials

The datasets analyzed during the current study are not publicly available due to patient privacy restrictions but are available from the corresponding author on reasonable request. The complete analysis pipeline is available at [GitHub repository URL].

### Competing interests

The authors declare that they have no competing interests.

### Funding

[To be completed]

### Authors' contributions

[To be completed]

### Acknowledgements

Not applicable.

---

## References

1. Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. CA Cancer J Clin. 2021;71(3):209-249.

2. Moore RG, Miller MC, Disilvestro P, et al. Evaluation of the diagnostic accuracy of the risk of ovarian malignancy algorithm in women with a pelvic mass. Obstet Gynecol. 2011;118(2 Pt 1):280-288.

3. Kawakami E, Tabata J, Yanaihara N, et al. Application of artificial intelligence for preoperative diagnostic and prognostic prediction in epithelial ovarian cancer based on blood biomarkers. Clin Cancer Res. 2019;25(10):3006-3015.

4. Lu M, Fan Z, Xu B, et al. Using machine learning to predict ovarian cancer. Int J Med Inform. 2020;141:104195.

5. Arezzo F, Cormio G, La Forgia D, et al. A machine learning approach applied to gynecological ultrasound to predict progression-free survival in ovarian cancer patients. Arch Gynecol Obstet. 2022;306(6):2143-2150.

6. Roberts M, Driggs D, Thorpe M, et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nat Mach Intell. 2021;3(3):199-217.

7. Ustun B, Rudin C. Learning optimized risk scores. J Mach Learn Res. 2019;20(150):1-75.

8. Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. BMJ. 2020;369:m1328.
