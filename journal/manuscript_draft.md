# Model Complexity Versus Transportability in Ovarian Cancer Risk Prediction: Benchmarking an Interpretable Integer Scorecard Against Clinical Algorithms and Machine-Learning Models Across Three Asian Cohorts

**Authors:** [Full names and institutional addresses to be completed]

**Corresponding author:** [Name and email address to be completed]

## Abstract

**Background:** Preoperative risk models for ovarian cancer range from fixed clinical formulas (ROMA, CPH-I) to flexible machine-learning ensembles. For the small hospital datasets typical of pelvic-mass triage, the trade-off between model complexity and transportability remains unsettled: flexible models can exploit cohort-specific structure but may fail to transfer across hospitals, whereas simple models may underfit genuine nonlinearity. We benchmarked an interpretable integer scorecard (quantile-binned CA125, HE4, and age with whole-point weights) against clinical algorithms and tree ensembles, and asked when, if ever, added complexity pays off.

**Methods:** We used a Chinese training cohort (n = 349) and two external cohorts: West China (n = 380, pelvic-mass triage population) and Japan (n = 177, 27 cancers versus 150 healthy women, used as a case-mix stress test rather than a second validation). Eight models were fitted or computed frozen on the training cohort: the integer scorecard (configuration selected strictly internally by a one-standard-error rule over a pre-specified grid, verified by repeated cross-validation), an unbinned logistic regression, ROMA, CPH-I, a CA125 ≥ 35 U/mL rule, and XGBoost, CatBoost, and Random Forest at package defaults. Evaluation comprised bootstrap pairwise AUROC comparisons, random-effects meta-analytic pooling across the two external cohorts, calibration (Brier decomposition, calibration-in-the-large, estimated calibration index), precision-recall analysis, NRI/IDI, decision consequences at published cutpoints, decision curve analysis with bootstrap confidence bands, a weighted-harm decision analysis across missed-cancer:unnecessary-referral exchange rates, robustness to measurement noise, outliers, and bin-boundary perturbation, learning curves, and a synthetic simulation study mapping complexity against distribution shift, noise, and concept drift.

**Results:** On the primary cohort, the scorecard (AUROC 0.929, 95% CI 0.902–0.955) and the unbinned logistic regression (0.936, 0.908–0.961) were statistically indistinguishable (p = 0.478) and both exceeded CPH-I (0.868), ROMA (0.886), and the tree ensembles (0.879–0.887). Pooled across both external cohorts, the scorecard exceeded ROMA by +0.043 (95% CI 0.013–0.074, p = 0.006, I² = 0%) and every tree ensemble (+0.046 to +0.054, all p ≤ 0.002), matched the unbinned logistic regression (−0.005, p = 0.554), and differed from CPH-I heterogeneously (I² = 79%; CPH-I was best in the Japanese stress test). At its high-risk tier the scorecard achieved 75.0% sensitivity and 97.4% specificity versus 56.4%/96.9% for ROMA at its published cutpoints — 47 versus 82 missed cancers of 188, with 5 versus 6 false-positive referrals. In the weighted-harm analysis the unbinned logistic regression minimised harm at every exchange rate; the scorecard was within 2.4–7.6 harm points per 100 women of it and below ROMA at all rates ≤ 10:1. The primary results were not an artifact of the heavily imputed HE4: refitting the linear models without HE4 changed the West China AUROC by 0.004 (scorecard 0.925; unbinned regression 0.932). Linear models dominated calibration, noise robustness, and data efficiency; the simulation reproduced the empirical pattern: flexible models won only under stationary structure, while the scorecard ranked first in every cell under concept drift.

**Conclusions:** With four routine biomarkers and a few hundred training patients, linear and integer-score models reach the external discrimination ceiling and transfer best; complexity repaid only under stationary structure in the simulation and never in the real cohorts. The integer scorecard approached the best model's decision performance while remaining bedside-computable and transparent. Findings are specific to Asian, hospital-based pelvic-mass populations; the Japanese cohort is a healthy-control stress test, and prospective validation is required before any clinical use.

**Keywords:** ovarian cancer, risk prediction, integer scorecard, external validation, model complexity, transportability, distribution shift, calibration, decision curve analysis, machine learning

---

## Background

Ovarian cancer is the most lethal gynecologic malignancy, with an estimated 314,000 new cases and 207,000 deaths per year worldwide [@sung2021]. Most patients present with advanced disease [@torre2018, @lheureux2019], and population screening has not been shown to reduce mortality [@menon2021, @jacobs2016, @buys2011]. Outcomes improve when patients are triaged to the appropriate surgical pathway, which is why preoperative risk assessment of an adnexal mass matters clinically [@kaijser2013, @timmerman2021, @engelen2006].

Risk assessment leans on two serum markers, CA125 [@bast1983, @jacobs1989] and HE4 [@hellstrom2003, @drapkin2005], combined into fixed formulas such as the Risk of Ovarian Malignancy Algorithm (ROMA; CA125 + HE4 + menopausal status) [@moore2009, @moore2011] and the Copenhagen Index (CPH-I; CA125 + HE4 + age) [@karlsen2015]. These algorithms are computable by hand, but their coefficients and cutpoints were derived in Western populations, and their published decision cutpoints frequently fail to transfer to other populations and assay platforms [@blythe2010, @molina2011, @chan2013, @iizuka2023, @rolfsen2020]. At the other end of the complexity spectrum, machine-learning models — most often gradient-boosted trees — have been proposed repeatedly for ovarian cancer risk prediction [@kawakami2019, @lu2020, @akazawa2021, @arezzo2022], typically reporting strong internal performance but rarely external validation, a pattern that recurs across medical artificial intelligence [@roberts2021, @wynants2020, @beam2018, @topol2019].

Between these extremes lies a well-replicated but under-appreciated finding: in clinical prediction with tabular data, flexible models seldom outperform well-regularized logistic regression once evaluated honestly, particularly at the small sample sizes typical of single hospitals [@christodoulou2019, @vanderploeg2014, @steyerberg2010]. Interpretable models have been argued to belong in high-stakes medicine unless clearly inferior [@rudin2019, @caruana2015], and sparse point systems such as APACHE II, Wells, and CHA2DS2-VASc persist precisely because they can be computed at the bedside [@knaus1985, @wells2000, @lip2010]. Yet the empirical literature rarely quantifies the complexity-transportability trade-off directly: which model class transfers best across hospitals, how much performance is actually left on the table by interpretability, and under which conditions (sample size, measurement noise, distribution shift, structural drift) added complexity is worth its cost.

We address these questions in the setting of ovarian cancer risk prediction using three publicly archived Asian cohorts. We construct an integer scorecard — four routine features (CA125, HE4, age, menopausal status), rank-transformed, cut into three quantile bins, fitted by L2-regularized logistic regression, and rounded to whole points — selected strictly on internal data and applied frozen to two external cohorts. We benchmark it against the two published clinical formulas, a single-marker rule, an unbinned logistic regression, and three tree ensembles, and we evaluate the models not only for discrimination but for calibration, decision consequences at published cutpoints, net benefit, robustness to measurement noise and cutpoint perturbation, sample-size efficiency, and — through a synthetic simulation calibrated to the training data — the conditions under which complexity pays. The study is reported in accordance with TRIPOD [@collins2015] and STARD 2015 [@bossuyt2015], with a PROBAST-style self-assessment [@wolff2019].

## Methods

### Study Design and Data Sources

This is a retrospective diagnostic-accuracy benchmark using three independent, publicly available Asian cohorts. Models were developed exclusively on the Chinese cohort and applied frozen to the West China and Japanese cohorts; no statistic computed on an external cohort influenced model selection, preprocessing, or imputation. The pre-specified primary endpoint was the AUROC of the scorecard versus ROMA on the West China cohort at 0% measurement noise, with West China pre-specified as the primary cohort because it is the largest external cohort with balanced outcome classes and a pelvic-mass triage case mix matching the intended use (93% power to detect a 0.05 AUROC difference). The Japanese cohort (27 cancers versus 150 healthy women) does not represent a pelvic-mass triage population and was pre-specified as a case-mix stress test: its role was to quantify model behaviour under a large distribution shift, not to confirm the primary comparison. Secondary endpoints were internal performance, the Japanese stress test, random-effects pooling across both external cohorts, calibration, decision analyses (including weighted-harm analysis), noise robustness, and the simulation study. Comparisons used 2,000 patient-level bootstrap resamples with 95% percentile intervals and two-sided empirical p-values; the scorecard-versus-ROMA comparison was the only confirmatory test and the remaining comparisons are reported without multiplicity adjustment. The analysis plan was documented in the repository before any external cohort was opened; the study was not prospectively registered (see Limitations).

The three source datasets are: the Chinese training cohort (n = 349; Mendeley Data, https://doi.org/10.17632/th7fztbrv9.11) [@dataset1]; the West China cohort (n = 380; figshare, https://doi.org/10.6084/m9.figshare.28831256) [@dataset2]; and the Japanese cohort (n = 177; Karger figshare, https://doi.org/10.6084/m9.figshare.24235450) [@dataset3]. This is a secondary analysis of fully de-identified, publicly archived data collected under the originating studies' approvals (Declaration of Helsinki [@helsinki2013]); no additional institutional review board approval was required.

**Eligibility.** Adult women undergoing preoperative evaluation for a suspected ovarian mass with serum CA125 and HE4 measured and a histopathologic diagnosis (cancer or benign ovarian mass) — or, for the Japanese cohort, ovarian cancer patients versus healthy-woman controls — and age or menopausal status recorded. All eligible records in each source dataset were used; no case was deleted for missing core-feature values (missingness handled as below). Whether recruitment was consecutive could not be verified and is listed as a limitation.

**Cohorts.** The Chinese training cohort comprised 349 patients (171 cancer, 178 benign) with 49 recorded clinical and hematological variables (Supplementary Table S2). The West China cohort comprised 380 patients (188 ovarian cancer, 192 benign ovarian cysts); HE4 was recorded for 100 of 380 (74% missing). The Japanese cohort comprised 27 ovarian cancer patients and 150 healthy women; age was missing for 151 of 177 (85%) and menopausal status for 4 of 177 (2%). Missing external values were replaced with per-feature medians computed once on the training cohort and frozen; no statistic was ever computed on a validation cohort (TRIPOD/PROBAST [@collins2015, @wolff2019]).

### Candidate Models

Eight models were evaluated, all using the same four features (CA125 in U/mL, HE4 in pmol/L, age in years, menopausal status) and the same frozen preprocessing:

1. **Integer scorecard (the index model).** Continuous features were mapped to uniform ranks with a `QuantileTransformer` (n_quantiles = 100) fitted on the training cohort, cut into three equal-frequency bins with one-hot encoding, and fitted with L2-regularized logistic regression (C = 1.5). Coefficients were rescaled and rounded to integers in the range −5 to +5 (weights below 0.01 in absolute value set to zero), following the risk-score construction of Ustun and Rudin without requiring a mixed-integer solver [@ustun2016, @ustun2019].
2. **Unbinned logistic regression.** The four features standardized with training means and standard deviations and fitted with an unregularized logistic regression — the canonical reference model in clinical prediction [@christodoulou2019].
3. **ROMA.** The published two-formula algorithm computed on raw CA125, HE4, and menopausal status [@moore2011]; evaluated with its published high-risk cutpoints (≥ 13.1% premenopausal, ≥ 27.7% postmenopausal).
4. **CPH-I.** The published Copenhagen Index, CPH-I = −14.0647 + 1.0649·log2(HE4) + 0.6050·log2(CA125) + 0.2672·age/10, with probability exp(CPH-I)/(1 + exp(CPH-I)) [@karlsen2015, @rolfsen2020]; evaluated with its published cutpoint (≥ 7%).
5. **CA125 rule.** The classic single-marker referral rule, CA125 ≥ 35 U/mL.
6–8. **Tree ensembles.** XGBoost [@chen2016], CatBoost [@prokhorenkova2018], and Random Forest [@breiman2001] at their package defaults (no tuning), representing how these packages are typically first applied. A 27-configuration-per-family tuning grid and a nested cross-validated tuning estimate are reported as sensitivity analyses.

**Model selection (internal only).** The scorecard's configuration was selected by five-fold stratified cross-validation over a pre-specified grid of 12 L2 candidates (bins ∈ {3, 4, 5, 7} × C ∈ {0.5, 1.5, 5.0}), with all preprocessing refitted inside each training fold. The one-standard-error rule [@breiman1984, @friedman2010] selected the simplest configuration within one SE of the best: 3 bins, L2, C = 1.5 (7 non-zero integer weights). We verified the stability of this selection with 20 repeated five-fold cross-validations (seeds 0–19), applying the rule independently in each repeat. An argmax-CV control (the configuration maximizing internal CV: 4 bins, L2, C = 0.5), L1 variants, and an out-of-grid two-bin check are reported as sensitivity analyses (Supplementary Table S1).

### Evaluation

**Discrimination.** AUROC on each cohort; pairwise model comparisons by 2,000 paired bootstrap resamples (two-sided empirical p-values) [@efron1994, @delong1988, @hanley1983]. Because the Japanese cohort has only 27 cancers, single-cohort comparisons there are descriptive; instead, AUROC differences were pooled across the two external cohorts with a DerSimonian–Laird random-effects model, using bootstrap standard errors per cohort, with I² as the heterogeneity measure [@dersimonian1986]. Precision-recall AUC was computed for the imbalanced Japanese cohort.

**Internal performance and memorization.** Five-fold cross-validated AUROC for all fitted models (repeated 20× for stability); in-sample AUROC reported to illustrate memorization. For the tree ensembles we additionally estimated nested cross-validated AUROC (outer five-fold, inner five-fold selection over a 9-configuration grid per family) to test whether tuning would change internal conclusions.

**Calibration.** Brier score with Murphy's decomposition (reliability, resolution, uncertainty) [@murphy1973], calibration slope and intercept by logistic recalibration, calibration-in-the-large, and the estimated calibration index (ECI) via LOWESS smoothing. For the scorecard, both a frozen training-fitted score-to-probability map and a within-cohort recalibrated map were evaluated; the recalibrated map is the decision-relevant estimate [@vancalster2019].

**Reclassification.** Category-free NRI and IDI [@pencina2008] of the scorecard versus ROMA, CPH-I, the unbinned logistic regression, and XGBoost on the primary cohort, with 1,000 bootstrap confidence intervals.

**Decision analyses.** Operating characteristics (sensitivity, specificity, PPV, NPV, likelihood ratios) across every integer cutpoint of the scorecard, and decision consequences per 100 women — missed cancers and unnecessary referrals — for the scorecard's high-risk tier (score ≥ +1, a pre-specified tier derived from training prevalence bands), ROMA at published cutpoints, CPH-I at its published cutpoint, the CA125 rule, and treat-all. Decision curve analysis across threshold probabilities 1–100% with 95% bootstrap confidence bands for the scorecard and ROMA [@vickers2006, @vancalster2018]. Finally, a weighted-harm decision analysis: for each model, decision thresholds were swept over 1–100%, and for exchange-rate weights w ∈ {1, 2, 5, 10, 20, 50} (the harm of one missed cancer relative to one unnecessary referral) the threshold minimising net harm per 100 women, H = w × missed cancers + unnecessary referrals, was identified and compared across models and against the treat-none and treat-all strategies.

**Robustness experiments.** (i) Monte Carlo measurement-noise simulation: proportional Gaussian noise on CA125 and HE4 at σ ∈ {0, 0.05, …, 0.30}, 100 perturbations per level [@fraser1989, @ricos1999]; (ii) alternative noise mechanisms on the primary cohort: additive noise (±30% of the feature median), combined proportional-plus-additive noise, and gross-error outlier contamination (5% of biomarker values scaled by ×3 or ÷3); (iii) bin-boundary sensitivity: the frozen tertile cutpoints shifted by ±5% and ±10% of the transformed range before scoring; (iv) per-feature drift decomposition and quantification of real inter-cohort distribution shift by the Kolmogorov–Smirnov statistic, linked to each model's transfer loss (training CV minus external AUROC); (v) data efficiency: learning curves refitting each model on stratified training subsamples (n = 60–349) with both internal cross-validated and frozen external evaluation.

**Simulation study.** To test the generality of the empirical complexity–transportability pattern, we simulated biomarker data calibrated to the Chinese cohort: three biomarkers with class-conditional lognormal marginals and an outcome model containing a strong linear component, a multiplicative CA125×HE4 interaction, and a smooth nonlinear term. We varied (i) training size n ∈ {100, 200, 400, 800}; (ii) distribution shift, implemented as shrinkage of class-conditional biomarker means toward the pooled mean by δ ∈ {0, 0.5, 1.0}; (iii) multiplicative measurement noise σ ∈ {0, 0.5}; and (iv) concept drift d ∈ {0, 1}, in which the interaction and nonlinear terms vanish from the test-side outcome model (stationary versus drifted structure). Each cell used 20 replicates; models were trained on clean data and evaluated frozen on a shifted test set of 4,000 patients.

### Implementation

Analyses were performed in Python 3.13 with scikit-learn 1.6, XGBoost 2.1, CatBoost 1.2, NumPy, and Pandas, with fixed random seeds (random_state = 42) and stratified five-fold splits. The complete pipeline is available at https://github.com/angelayaoo/Ovarian-Cancer-RiskSLIM (scripts `bench_lib.py`, `clinical_comparators.py`, `repeated_cv.py`, `calibration_metrics.py`, `meta_analysis.py`, `decision_analysis.py`, `robustness_extended.py`, `simulation_study.py`, `generate_new_figures.py`), with benchmark outputs under `results/`.

## Results

### Cohort Characteristics and Cross-Cohort Shift

The Chinese training cohort (n = 349; 171 cancer, 178 benign) had median CA125 241.5 versus 22.7 U/mL and median HE4 140.9 versus 43.8 pmol/L in cancer versus benign patients. The West China cohort (n = 380; 188 cancer, 192 benign cysts) had median CA125 321.5 versus 41.2 U/mL. The Japanese cohort (n = 177; 27 cancer, 150 healthy women, 15.3% cancer prevalence) had median CA125 53.0 versus 13.9 U/mL and median HE4 49.5 versus 32.4 pmol/L. Missingness of the core features: West China HE4 280/380 (74%); Japan age 151/177 (85%), menopausal status 4/177 (2%), CA125 or HE4 2/177 (1%); all replaced with frozen training medians.

Per-feature Kolmogorov–Smirnov statistics between each external cohort and the training distribution ranged from 0.13 (age) to 0.38 (HE4) for West China (mean 0.22) and from 0.43 to 0.60 for Japan (mean 0.50).

### Learned Scorecard

The deployed scorecard has seven non-zero integer weights (Table 1): HE4 contributes +5/−4 points for the top/bottom tertiles, CA125 +2/−1, and age +2/−2; middle bins receive zero weight, and menopausal status — a binary feature that collapses under quantile binning — receives zero weight. Total scores range from −7 to +9; the score is the sum of four bin weights matched to a patient's values and is computable by hand in seconds.

**Table 1. Learned integer scorecard.** Fitted on the quantile-transformed Chinese training cohort only and applied frozen to the external cohorts. Each continuous feature is divided into three equal-frequency (tertile) bins; tertile cutpoints in original units are CA125 24.3/165.0 U/mL, HE4 45.8/92.7 pmol/L, and age 37.0/52.0 years. A patient's score is the sum of the four bin weights matched to her values.

| Feature | Bin 1 (lowest third) | Bin 2 (middle third) | Bin 3 (highest third) |
|---|---:|---:|---:|
| CA125 | −1 | 0 | +2 |
| HE4 | −4 | −1 | +5 |
| Age | −2 | 0 | +2 |
| Menopausal status | 0 | 0 | 0 |

### Primary External Comparison

The pre-specified primary endpoint was the scorecard-versus-ROMA AUROC difference on the West China cohort. The scorecard achieved 0.929 (95% CI 0.902–0.955) and exceeded ROMA (0.886, 0.848–0.921) by +0.043 (95% CI 0.012–0.074, p = 0.009) (Table 2, Figure 1). Figure 1 shows the full cohort-by-model pattern: the two linear models — the integer scorecard and the unbinned logistic regression — recorded the highest AUROC on the primary cohort, the tree ensembles and the published formulas clustered below them, and CPH-I recorded the highest AUROC in Japan.

In secondary paired bootstrap comparisons on West China, the scorecard did not differ from the unbinned logistic regression (0.936, 0.908–0.961; Δ = −0.007, 95% CI −0.025 to 0.011, p = 0.478) and exceeded CPH-I (+0.061, p < 0.001), the CA125 rule (+0.287, p < 0.001), and all three tree ensembles (+0.043 to +0.051, p ≤ 0.002) (Table 3).

**Table 2. AUROC by model and cohort.** Training = five-fold cross-validated AUROC for fitted models (fixed formulas are applied values, marked †). External values are bootstrap means (2,000 resamples) at 0% noise; 95% percentile intervals in Supplementary Table S5. Bold marks the cohort-wise best fitted model.

| Model | Training (n = 349) | West China (n = 380) | Japan (n = 177) |
|---|---:|---:|---:|
| Scorecard (3 bins, L2, C = 1.5) | 0.906 | **0.929** | 0.843 |
| Raw logistic regression | 0.894 | **0.936** | 0.804 |
| ROMA | 0.912† | 0.886 | 0.797 |
| CPH-I | 0.912† | 0.868 | **0.920** |
| CA125 ≥ 35 U/mL rule | 0.727† | 0.642 | 0.781 |
| XGBoost (defaults) | 0.882 | 0.887 | 0.734 |
| CatBoost (defaults) | 0.900 | 0.879 | 0.749 |
| Random Forest (defaults) | 0.895 | 0.879 | 0.792 |

† Applied value on the training cohort (published fixed formula, not fitted or cross-validated).

**Figure 1. AUROC by model and cohort at 0% measurement noise.** Training-cohort values are five-fold cross-validated (fixed formulas shown as applied values); external-cohort error bars are 95% bootstrap confidence intervals (2,000 patient-level resamples). All models were trained exclusively on the Chinese cohort and applied frozen. Overlapping individual confidence intervals do not imply non-significance; paired comparisons are in Table 3. CA125 = cancer antigen 125; ROMA = Risk of Ovarian Malignancy Algorithm; CPH-I = Copenhagen Index; XGBoost/CatBoost = gradient-boosted tree ensembles.

### Secondary Pooled Analysis Across the External Cohorts

As a secondary analysis, bootstrap AUROC differences were pooled across the two external cohorts with a random-effects model (Table 3, Figure 2). The pooled estimates favoured the scorecard against ROMA (+0.043, 95% CI 0.013–0.074, p = 0.006, I² = 0%) and against all three tree ensembles (+0.046 to +0.054, all p ≤ 0.002, I² = 0%), while the scorecard did not differ from the unbinned logistic regression (−0.005, 95% CI −0.023 to 0.012, p = 0.554) or the CA125 rule (+0.186, p = 0.094, I² = 89%). The scorecard-versus-CPH-I comparison was the only heterogeneous one (I² = 79%): the difference was +0.061 in West China and −0.076 in Japan. Because the Japanese cohort contrasts cancers with healthy women rather than benign masses, the pooled estimates should be read with the primary-cohort comparisons as the confirmatory evidence and the Japanese cohort as a case-mix stress test. In that cohort, 129 of 150 (86%) healthy women and 8 of 27 (30%) cancer patients fell into the lowest training-calibrated bins of both markers.

**Table 3. Bootstrap pairwise and pooled (random-effects) AUROC differences of the scorecard versus each comparator.** West China and Japan: paired bootstrap differences (2,000 resamples). Pooled: DerSimonian–Laird random-effects estimate with 95% CI and I².

| Comparator | Δ West China | Δ Japan | Pooled Δ | 95% CI | p | I² |
|---|---:|---:|---:|---:|---:|---:|
| Raw logistic regression | −0.0065 | +0.0393 | −0.0052 | −0.023 to 0.012 | 0.554 | 0% |
| ROMA | +0.0429 | +0.0462 | +0.0431 | 0.013 to 0.074 | 0.006 | 0% |
| CPH-I | +0.0610 | −0.0762 | +0.0049 | −0.127 to 0.137 | 0.942 | 79% |
| CA125 ≥ 35 U/mL rule | +0.2875 | +0.0640 | +0.1862 | −0.032 to 0.404 | 0.094 | 89% |
| XGBoost | +0.0426 | +0.1091 | +0.0457 | 0.017 to 0.074 | 0.002 | 0% |
| CatBoost | +0.0507 | +0.0949 | +0.0543 | 0.024 to 0.084 | < 0.001 | 0% |
| Random Forest | +0.0506 | +0.0512 | +0.0507 | 0.022 to 0.080 | < 0.001 | 0% |

**Figure 2. Meta-analysis of the two external validation cohorts (random-effects).** Forest plot of scorecard-minus-comparator AUROC differences: circles = pooled DerSimonian–Laird estimate with 95% CI; squares/triangles = per-cohort bootstrap differences (West China/Japan). Positive values favour the scorecard. AUROC = area under the receiver operating characteristic curve; CI = confidence interval.

### Internal Performance and Selection Stability

Across 20 repeated five-fold cross-validations on the training cohort, mean internal AUROC was 0.906 (between-repeat SD 0.003; range 0.901–0.911) for the scorecard, 0.894 (±0.003) for the unbinned logistic regression, 0.900 (±0.006) for CatBoost, 0.895 (±0.005) for Random Forest, and 0.882 (±0.006) for XGBoost. Applying the one-standard-error selection rule independently in each repeat, a three-bin configuration was selected in 18 of 20 repeats (90%).

In-sample, the tree ensembles fitted the training cohort at AUROC 0.991–1.000 and their cross-validated estimates were 0.094–0.117 lower; the corresponding in-sample-to-cross-validated gap for the scorecard was 0.005. In nested cross-validation (outer five-fold, inner five-fold over a 9-configuration grid per family), the tuned tree ensembles reached 0.885 (XGBoost), 0.900 (CatBoost), and 0.903 (Random Forest). Figure 3 contrasts these model classes on the complexity axis.

**Figure 3. The complexity-performance ladder.** External AUROC on West China (bootstrap mean) with models ordered left to right by increasing complexity. AUROC = area under the receiver operating characteristic curve; ROMA = Risk of Ovarian Malignancy Algorithm; CPH-I = Copenhagen Index.

### Calibration

On West China (cancer prevalence 49.5%), the within-cohort recalibrated scorecard had the lowest Brier score (0.097); the frozen training-fitted scorecard map (0.116) and the unbinned logistic regression (0.111) were next, ahead of XGBoost (0.128), ROMA (0.209), and CPH-I (0.259) (Table 4). Calibration-in-the-large was −0.95 for ROMA and −1.42 for CPH-I on this cohort, and −0.86 and −1.73 respectively in Japan; the frozen scorecard map gave −0.09 (West China) and +0.10 (Japan). In the smoothed calibration curves (Figure 4), the two published formulas fell below the diagonal in both cohorts, and the scorecard maps tracked the diagonal above a predicted risk of roughly 0.2.

**Table 4. Calibration summary (external cohorts).** Brier score, calibration-in-the-large (CITL), and estimated calibration index (ECI). Murphy decomposition components and calibration slope/intercept are in the repository outputs (`results/calibration_metrics.csv`).

| Cohort | Model | Brier | CITL | ECI |
|---|---|---|---:|---:|---:|
| West China | Scorecard (recalibrated) | 0.097 | 0.00 | 0.128 |
| West China | Scorecard (frozen map) | 0.116 | −0.09 | 0.191 |
| West China | Raw logistic regression | 0.111 | −0.05 | 0.198 |
| West China | ROMA | 0.209 | −0.95 | 0.177 |
| West China | CPH-I | 0.259 | −1.42 | 0.110 |
| West China | XGBoost | 0.128 | −0.27 | 0.058 |
| Japan | Scorecard (recalibrated) | 0.069 | 0.00 | 0.163 |
| Japan | Scorecard (frozen map) | 0.077 | 0.10 | 0.176 |
| Japan | ROMA | 0.102 | −0.86 | 0.071 |
| Japan | CPH-I | 0.121 | −1.73 | 0.031 |

**Figure 4. Calibration curves (LOWESS-smoothed observed versus predicted risk).** Left: West China (n = 380); right: Japan (n = 177). Green solid and dashed curves = within-cohort recalibrated and frozen training-fitted scorecard maps; grey dashed line = cohort prevalence; dotted diagonal = perfect calibration. ROMA = Risk of Ovarian Malignancy Algorithm; CPH-I = Copenhagen Index; LOWESS = locally weighted scatterplot smoothing.

### Decision Characteristics

At its pre-specified high-risk tier (score ≥ +1), the scorecard achieved 75.0% sensitivity and 97.4% specificity on West China — 47 of 188 cancers missed and 5 false-positive referrals (12.4 missed cancers and 1.3 unnecessary referrals per 100 women; Table 5). At their published cutpoints, ROMA achieved 56.4% sensitivity and 96.9% specificity (82 cancers missed, 6 false positives; 21.6 and 1.6 per 100 women) and CPH-I 74.5%/94.8% (48 missed, 10 false positives; 12.6 and 2.6 per 100 women); the CA125 rule achieved 84.6%/43.8% (29 missed, 108 false positives; 7.6 and 28.4 per 100 women). Across the scorecard's cutpoint range, sensitivity was 83.0% with specificity 92.7% at score ≥ 0 (PPV 0.918), and specificity was 100% at score ≥ +5 (sensitivity 22.9%).

In the weighted-harm analysis (Table 6), the minimum net harm per 100 women, H = w × missed cancers + unnecessary referrals, was computed for each model at exchange-rate weights w of 1 to 50. The unbinned logistic regression attained the lowest minimum harm at every weight (9.7 at w = 1; 39.7 at w = 10; 46.6 at w = 50). The scorecard was within 2.4–7.6 harm points of the unbinned logistic regression across weights and below ROMA at all weights ≤ 10 (30.5 versus 43.4 at w = 5); at w ≥ 20, ROMA's minimum harm was 45.8, below the scorecard's 51.6–59.5. CPH-I and XGBoost had the highest minimum harm at every weight ≥ 2. The full trade-off curves are shown in Figure 6, with the published-cutpoint operating points marked.

In decision curve analysis with 95% bootstrap confidence bands (Figure 5), the recalibrated scorecard recorded the highest net benefit of the fitted models at every threshold between 0.10 and 0.60, and its curve lay above both default strategies (treat all and treat none) throughout that range; ROMA recorded the lowest net benefit of the fitted models at nearly every threshold. The frozen training-fitted scorecard map performed within 0.01 net benefit of the recalibrated map at thresholds ≥ 0.30.

**Table 5. Decision consequences at published or pre-specified decision rules (West China, n = 380; 188 cancers, 192 benign).** Both harm columns are expressed per 100 women (raw patient counts in parentheses). Treat-all is the reference strategy.

| Strategy | Sensitivity | Specificity | Missed cancers per 100 women | Unnecessary referrals per 100 women |
|---|---:|---:|---:|---:|
| Scorecard high-risk tier (score ≥ +1) | 75.0% | 97.4% | 12.4 (47) | 1.3 (5) |
| ROMA published cutpoints (≥ 13.1%/27.7%) | 56.4% | 96.9% | 21.6 (82) | 1.6 (6) |
| CPH-I published cutpoint (≥ 7%) | 74.5% | 94.8% | 12.6 (48) | 2.6 (10) |
| CA125 ≥ 35 U/mL | 84.6% | 43.8% | 7.6 (29) | 28.4 (108) |
| Treat all (reference) | 100% | 0% | 0 | 50.5 |

**Table 6. Weighted-harm decision analysis (West China, per 100 women).** For each exchange-rate weight w (harm of one missed cancer relative to one unnecessary referral), the table gives the threshold minimising net harm H = w × missed cancers + unnecessary referrals, followed by H (missed/unnecessary at that threshold). Bold marks the model with the lowest minimum harm at each weight. Treat-none and treat-all are reference strategies: H(treat none) = 49.5w and H(treat all) = 50.5 for every w.

| Model | w = 1 | w = 2 | w = 5 | w = 10 | w = 20 | w = 50 |
|---|---:|---:|---:|---:|---:|---:|
| Scorecard (recalibrated) | 12.1 (8.4/3.7) | 20.5 (8.4/3.7) | 30.5 (2.9/16.1) | 45.0 (2.9/16.1) | 51.6 (0.3/46.3) | 59.5 (0.3/46.3) |
| Raw logistic regression | **9.7 (5.8/3.9)** | **15.5 (4.7/6.1)** | **29.2 (4.5/6.8)** | **39.7 (0.8/31.8)** | **46.6 (0.0/46.6)** | **46.6 (0.0/46.6)** |
| ROMA | 15.0 (9.2/5.8) | 22.9 (6.8/9.2) | 43.4 (6.8/9.2) | 45.8 (0.0/45.8) | 45.8 (0.0/45.8) | 45.8 (0.0/45.8) |
| CPH-I | 15.0 (12.6/2.4) | 27.4 (11.8/3.7) | 51.3 (3.9/31.6) | 64.7 (2.4/41.1) | 88.4 (2.4/41.1) | 159.5 (2.4/41.1) |
| XGBoost | 14.7 (9.7/5.0) | 24.5 (9.7/5.0) | 44.7 (4.5/22.4) | 66.1 (4.2/23.9) | 105.0 (3.7/31.3) | 215.5 (3.7/31.3) |
| Treat none | 49.5 | 98.9 | 247.4 | 494.7 | 989.5 | 2473.7 |
| Treat all | 50.5 | 50.5 | 50.5 | 50.5 | 50.5 | 50.5 |

**Figure 5. Decision curve analysis with 95% bootstrap confidence bands (West China, n = 380).** Net benefit per patient versus threshold probability (1–100%). Shaded bands = 95% percentile intervals from 1,000 patient-level bootstrap resamples (the scorecard's within-cohort recalibration map refitted in each resample). Black dashed line = treat all; zero line = treat none; dotted green line = frozen training-fitted scorecard map. ROMA = Risk of Ovarian Malignancy Algorithm; XGBoost = extreme gradient boosting.

**Figure 6. Decision-consequence trade-off curves (West China, n = 380), one panel per model.** Missed cancers versus unnecessary referrals per 100 women as the decision threshold is swept over 1–100%; curves further toward the lower-left corner are better. Grey markers are the treat-none (circle) and treat-all (square) references; the scorecard panel marks its high-risk tier (≥ +1) and the ROMA and CPH-I panels mark their published cutpoints. ROMA = Risk of Ovarian Malignancy Algorithm; CPH-I = Copenhagen Index.

### Reclassification Metrics

On the primary cohort, the integrated discrimination improvement of the scorecard over the frozen map was +0.071 (95% CI 0.036–0.106) versus ROMA, +0.110 (0.077–0.144) versus CPH-I, −0.048 (−0.075 to −0.021) versus the unbinned logistic regression, and −0.197 versus XGBoost. The category-free net reclassification improvement versus ROMA was −0.23 (event component +0.68, non-event component −0.91). Bootstrap confidence intervals for both metrics are reported in the repository outputs (`results/nri_idi.csv`).

### Robustness to Measurement Noise and Cutpoint Perturbation

Under proportional Gaussian measurement noise on CA125 and HE4, the scorecard's West China AUROC fell by 0.040 from 0% to 30% noise (−4.3% relative), compared with −0.002 (−0.2%) for the unbinned logistic regression, −0.074 (−8.3%) for ROMA, −0.086 (−9.7%) for XGBoost, and −0.053 (−6.0%) for CatBoost and Random Forest (Figure 7). The ordering across models was the same under additive noise, combined proportional-plus-additive noise, and 5% gross-error contamination (Figure S4). At 30% multiplicative noise the scorecard (0.890) was above ROMA at 0% noise (0.886); at 30% combined noise the scorecard was 0.869, still above every tree ensemble at any noise level. On the Japanese cohort, the scorecard fell from 0.843 to 0.806 over the same noise range and retained the highest AUROC of the fitted models at every level; CPH-I fell from 0.920 to 0.871.

Shifting the frozen tertile cutpoints by ±5% and ±10% of the transformed range changed the scorecard's West China AUROC between 0.913 and 0.929 (maximum change 0.016; Table S8).

**Figure 7. Measurement-noise degradation on West China (0% → 30% proportional Gaussian noise on CA125 and HE4; 100 perturbations per level).** Bars show the AUROC change. CA125 = cancer antigen 125; HE4 = human epididymis protein 4; ROMA = Risk of Ovarian Malignancy Algorithm; AUROC = area under the receiver operating characteristic curve.

### Data Efficiency and Transfer Gap

When each model was refitted on stratified training subsamples (n = 60–349) and evaluated frozen on West China (Figure 8), the unbinned logistic regression reached 0.925 at n = 60 and 0.936 at n = 349; the scorecard reached 0.901 at n = 60 and 0.929 at n = 349. The tree ensembles ranged between 0.847 and 0.900 across all training sizes. Internally (Figure S3), the cross-validated ordering across training sizes was similar.

Transfer loss (training cross-validated minus external AUROC) is plotted against the mean feature-wise Kolmogorov–Smirnov shift of each cohort in Figure S2. On the more shifted Japanese cohort the transfer loss of the flexible models was largest (0.10–0.15); on West China the scorecard's external AUROC exceeded its cross-validated AUROC (transfer loss −0.022).

**Figure 8. Data efficiency: external performance versus training size.** Models were refitted on stratified subsamples of the Chinese cohort (n = 60–349, 10 draws each) and evaluated frozen on West China; points are means ± SD across draws; dashed line marks ROMA's West China AUROC. AUROC = area under the receiver operating characteristic curve; ROMA = Risk of Ovarian Malignancy Algorithm.

### Simulation Study

In the synthetic study calibrated to the training data (Figure 9), the mean test AUROC of the tree ensembles exceeded the scorecard's in all 24 stationary cells (noise 0 or 50%, shift 0–1.0, n = 100–800), by 0.01–0.05 for CatBoost and 0.01–0.04 for XGBoost; the unbinned logistic regression was 0.06–0.21 below CatBoost in the same cells. In the 24 concept-drift cells (interaction and nonlinear term absent from the test-side outcome model), the scorecard recorded the highest mean AUROC in every cell, exceeding the tree ensembles by 0.00–0.04; the unbinned logistic regression was within 0.02–0.07 of the scorecard. Across all cells, higher shift and higher noise were associated with lower test AUROC for every model.

**Figure 9. Regime map from the synthetic simulation study (20 replicates per cell).** Heatmaps show the AUROC difference between XGBoost and the integer scorecard on the shifted test set, by training size (x-axis) and distribution shift (y-axis), at 50% measurement noise, with stationary structure (left) and concept drift (right). Warm colours: XGBoost higher; cool colours: scorecard higher. AUROC = area under the receiver operating characteristic curve.

### Sensitivity Analyses

Across the 12 L2 selection candidates, five-fold internal CV ranged 0.900–0.913; the argmax-CV control (4 bins, L2, C = 0.5, CV 0.913) achieved 0.880 on West China and 0.889 in Japan when evaluated frozen. The L1 variants (CV 0.898–0.910) achieved up to 0.907 on West China and 0.796 in Japan; two-bin configurations (out-of-grid; CV 0.895–0.897) achieved 0.847 on West China. On the 100 West China patients with measured HE4, the scorecard achieved 0.971, the unbinned logistic regression 0.978, ROMA 0.965, and XGBoost 0.923. Excluding HE4 — the feature missing for 74% of the West China cohort — changed the West China AUROC of the scorecard from 0.929 to 0.925, of the unbinned logistic regression from 0.936 to 0.932, and of XGBoost from 0.887 to 0.891. Excluding age — imputed for 85% of the Japanese cohort — changed the Japanese AUROC of the scorecard from 0.843 to 0.866 and of the unbinned logistic regression from 0.804 to 0.840. The tuned tree ensembles (best of 27 configurations per family, selected by internal CV) achieved 0.890 (XGBoost), 0.877 (CatBoost), and 0.894 (Random Forest) on West China. A training-fitted KNN imputation of the Japanese missing values gave a scorecard AUROC of 0.907 (leakage check, rejected). An exploratory 49-variable L1 logistic model achieved 0.935 in five-fold internal CV.

## Discussion

We benchmarked eight models spanning four orders of magnitude of model complexity — from a one-parameter CA125 rule to boosted ensembles with thousands of parameters — for preoperative ovarian cancer risk prediction, trained on one Chinese cohort and applied frozen to two independent Asian cohorts. Three findings stand out. First, the discrimination ceiling of this four-feature task was reachable with seven integer parameters: the scorecard's external AUROC (0.929) was statistically indistinguishable from the best-fitting unbinned logistic regression (0.936; p = 0.478) and significantly above ROMA (pooled +0.043, p = 0.006), CPH-I on the primary cohort, and every tree ensemble (pooled +0.046 to +0.054, all p ≤ 0.002). Because the scorecard's configuration was selected strictly on internal data, this margin is not attributable to selection on the validation cohort; notably, the configuration that maximized internal cross-validation (4 bins, CV 0.913) achieved only 0.880 externally, and internal CV and external AUROC moved in opposite directions across the bin-count axis — a caution for tuning flexible pipelines on small data. Second, the cost of the tree ensembles' complexity was transportability: they memorized the training cohort (in-sample 0.991–1.000), degraded most under measurement noise, never exceeded the linear models in external learning curves at any training size, and their tuned and nested-cross-validated variants closed none of the gap. Third, the decision-relevant advantages of the scorecard were large: at its high-risk tier it achieved 75.0% sensitivity with 97.4% specificity — dominating ROMA at its published cutpoints (56.4%/96.9%), which would miss 82 of 188 cancers in this population — and it carried the highest net benefit across all clinically relevant thresholds. Reclassification metrics were consistent but modest: the scorecard improved integrated discrimination over the fixed formulas (IDI +0.071 versus ROMA, +0.110 versus CPH-I) but not over the unbinned regression (−0.048) or XGBoost (−0.197, whose extreme fitted probabilities inflate IDI), and the category-free NRI versus ROMA was negative (−0.23), driven by non-event reclassification.

These results extend a consistent literature in clinical prediction: systematic reviews have repeatedly found no advantage of machine learning over logistic regression for tabular clinical prediction [@christodoulou2019], simulation studies show modern techniques are "data hungry" and need large samples to repay their flexibility [@vanderploeg2014], and the empirical rule-of-thumb of roughly 20 events per candidate parameter [@peduzzi1996] suggests the 171 cancer events available here support only a handful of parameters [@riley2020]. Our contribution is to quantify this trade-off simultaneously across the dimensions that matter for deployment: external discrimination, calibration, decision consequences, noise robustness, cutpoint stability, sample-size efficiency, and — through the simulation — the structural conditions that determine the sign of the complexity effect. The simulation reproduced the empirical ranking: flexible models won only when the structure was stationary, and the binned scorecard ranked first in every cell under concept drift. Mechanistically, under drift the interaction structure the ensembles had learned no longer held on the test side, while the additive models had never depended on it; the ensembles' fitted splits became a liability, and the purely linear model recovered toward the scorecard. The three real cohorts, with their substantial inter-cohort biomarker shifts (mean KS 0.22–0.50), sit firmly in the regime where complexity is a liability.

The comparison with the published formulas deserves emphasis, because it illustrates a decision-making hazard rather than a modeling nuance. ROMA and CPH-I are fixed Western formulas; on this Chinese validation population with 49.5% cancer prevalence, both overestimated absolute risk (calibration-in-the-large −0.95 and −1.42), their Brier scores (0.209 and 0.259) were nearly twice the scorecard's, and their published cutpoints referred too conservatively — ROMA's cutpoints sent only 112 of 380 women to high risk while missing 82 of 188 cancers. Base-rate dependence of these algorithms has been documented before [@rolfsen2020], and Asian validations have reported recalibration requirements for both [@chan2013, @iizuka2023]. The weighted-harm analysis places this in decision terms: across missed-cancer:unnecessary-referral exchange rates of 1:1 to 10:1, the scorecard's minimum net harm was below ROMA's (30.5 versus 43.4 per 100 women at 5:1), and only at rates ≥ 20:1 — where nearly any false referral is tolerated to avoid one missed cancer — did ROMA's most permissive threshold beat it. At the same time, the unbinned logistic regression minimised harm at every rate, a direct consequence of its marginally better discrimination; the scorecard's decision case therefore rests on bedside computability and auditability at a small harm cost (2.4–7.6 points per 100 women), not on decision-theoretic superiority over every competitor. We do not claim the scorecard's tiers are universal: they are population-relative tertiles, and the Japanese stress test — cancers versus healthy women, 86% of healthy women in the lowest bins — showed that quantile bins compress discrimination when the case mix shifts toward healthy controls; there, CPH-I's continuous log-scale retained the most information (0.920). Heterogeneity (I² = 79% for the CPH-I comparison) is the honest summary: no single model won every cohort, and the Japanese cohort is a stress test, not a second triage validation.

From a clinical decision support perspective, the scorecard has properties that matter for bedside use and for auditability. It is a small integer lookup table; the score is computable by hand in seconds; each point has a clinical interpretation (a top-tertile HE4 adds +5 points); and the tier rates were monotone (10.2%, 40.3%, 96.6% cancer across low/intermediate/high tiers). Its noise robustness relative to the fixed formulas and ensembles (−4.3% relative degradation at 30% noise versus 6.0–9.8%) matters because CA125 and HE4 results genuinely differ across laboratories and platforms [@blythe2010, @molina2011], and its insensitivity to ±10% cutpoint shifts suggests the tertile boundaries do not need assay-platform-specific retuning. At the same time, the unbinned logistic regression — the strongest pure discriminator and the most noise-robust model — serves as a reality check: under rank-preserving measurement noise its continuous linear scores barely moved, whereas the scorecard's tertile boundaries converted small perturbations into occasional bin crossings. Discretization thus costs roughly 1–4% AUROC under extreme noise and some discrimination in the healthy-control setting, and the honest claim for the scorecard is ceiling-level performance *and* bedside computability, not that binning improves discrimination.

**Limitations.** First, the training cohort is small (349 patients, 171 events), and although the repeated cross-validation (20 × 5-fold) showed stable selection, associations not detectable at this sample size may emerge with larger data; the 49-variable internal model (CV 0.935) indicates untested headroom. Second, external missingness was substantial: 74% of West China HE4 and 85% of Japanese age were replaced by frozen training medians. We regard the dependence of the primary result on imputed HE4 as the central limitation and addressed it directly: the scorecard's West China AUROC changed by only 0.004 when HE4 was excluded entirely (0.929 → 0.925), as did the unbinned logistic regression's (0.936 → 0.932), and the complete-case subset (n = 100) raised rather than lowered the linear models' AUROCs (0.971/0.978). The primary comparison therefore does not rest on the imputed feature; a leakage-prone KNN imputation was explicitly rejected. Third, the Japanese cohort compares cancers against healthy controls rather than benign masses, so its 15% prevalence does not represent a pelvic-mass triage population; it was used as a case-mix stress test, its results are descriptive and power-limited (27 cancers), and the pooled estimates should be read with the primary-cohort comparisons as the confirmatory evidence. Fourth, all cohorts are Asian, hospital-based series; generalization to other populations and assay platforms is unestablished, and the simulation study is a stylized generative model, not a proof about real hospitals. Fifth, calibration and decision analyses on the primary cohort used the cohort's own labels (post hoc); the within-cohort recalibration can flatter any model, which is why the frozen map is reported alongside, and the weighted-harm optima are threshold-fitting on the same data and serve as a comparative, not prescriptive, exercise. Sixth, the study was not prospectively registered, and the tree ensembles were compared at package defaults in the primary analysis; the tuning grid and nested cross-validation sensitivity analyses mitigate but do not eliminate the claim that tuning is immaterial. Seventh, ultrasound-based models (RMI, IOTA ADNEX) could not be computed because the public cohorts lack imaging variables; their role in triage remains outside our scope [@kaijser2013, @timmerman2021]. Finally, PROBAST-style assessment of this secondary analysis indicates low risk of bias in analysis (frozen pipelines, external evaluation) but high concern for applicability in non-Asian settings, and the analysis-domain strength depends on the protocol documented in the repository.

## Conclusions

For preoperative ovarian cancer risk prediction from four routine biomarkers at training sizes of a few hundred patients, model complexity beyond seven integer parameters did not buy external discrimination: the integer scorecard matched the best linear model, significantly outperformed ROMA and three tree ensembles in random-effects pooling across two external cohorts, missed 47 versus 82 cancers at the published decision cutpoints, and degraded least of the interpretable models under measurement noise. A synthetic simulation reproduced the pattern mechanistically, showing that flexible models repay their complexity only under stationary structure, while concept drift — the realistic condition across hospitals and assay platforms — favors simple additive models. The practical implication is not that this scorecard is ready for clinical use, but that clinical decision support for this task can be provided by a bedside-computable, auditable linear model at ceiling-level discrimination, with the integer scorecard as the most transparent realisation; prospective validation, local recalibration, and platform-specific cutpoint mapping remain the necessary steps before any clinical deployment.

---

## Declarations

### Ethics approval and consent to participate

This study is a secondary analysis of completely de-identified, publicly archived datasets. Ethics approval and consent to participate were not applicable to this secondary analysis (waived), and no new human participants were involved; the originating studies that collected the data obtained their own ethics approvals and informed consent in line with the Declaration of Helsinki [@helsinki2013].

### Consent for publication

Not applicable.

### Trial registration

Not applicable (retrospective secondary analysis of publicly archived data; no prospective interventional protocol).

### Availability of data and materials

The datasets analysed during the current study are available in public repositories: the Chinese training cohort in Mendeley Data (https://doi.org/10.17632/th7fztbrv9.11) [@dataset1]; the West China cohort in figshare (https://doi.org/10.6084/m9.figshare.28831256) [@dataset2]; and the Japanese cohort in Karger figshare (https://doi.org/10.6084/m9.figshare.24235450) [@dataset3]. The complete analysis code — preprocessing, internal model selection, repeated and nested cross-validation, model training, Monte Carlo noise simulation, calibration and reclassification metrics, meta-analytic pooling, decision analyses including the weighted-harm analysis (`results/decision_harm_tradeoff.csv`), robustness experiments, and the simulation study — is publicly available at https://github.com/angelayaoo/Ovarian-Cancer-RiskSLIM under an MIT license. The repository includes the raw source files (`data/raw/`), processed analysis-ready files (`data/processed/`), a `requirements.txt` (Python 3.13, scikit-learn 1.6, XGBoost 2.1, CatBoost 1.2), and the benchmark outputs (`results/`) underlying Tables 1–6 and Supplementary Tables S1–S10.

### Competing interests

The authors declare that they have no competing interests.

### Funding

[To be completed]

### Authors' contributions

[To be completed]

### Acknowledgements

Not applicable.

### Additional files

**Additional file 1 (PDF/Word):** Completed TRIPOD checklist and PROBAST self-assessment.
**Additional file 2 (PDF):** Supplementary Tables S1–S10 and Figures S1–S4.

---

## List of Abbreviations

AUROC, area under the receiver operating characteristic curve; CA125, cancer antigen 125; CI, confidence interval; CITL, calibration-in-the-large; CPH-I, Copenhagen Index; CV, cross-validation; DCA, decision curve analysis; ECI, estimated calibration index; HE4, human epididymis protein 4; IDI, integrated discrimination improvement; KNN, k-nearest neighbors; KS, Kolmogorov–Smirnov; LR, logistic regression; NB, net benefit; NRI, net reclassification improvement; PPV/NPV, positive/negative predictive value; PR-AUC, area under the precision-recall curve; REL/RES/UNC, reliability/resolution/uncertainty components of the Brier score; ROMA, Risk of Ovarian Malignancy Algorithm; SD, standard deviation; SE, standard error; XGBoost, extreme gradient boosting.

---

## Supplementary Information

**Table S1. Scorecard configuration sweep (as selected and as sensitivity analyses).** Five-fold internal CV over the 12-configuration L2 grid (selected row bolded by the 1-SE rule), L1 variants, and the out-of-grid two-bin check, with frozen external AUROCs. The argmax-CV control (4 bins, L2, C = 0.5, CV 0.913) collapsed to 0.880 on West China, while the selected 3-bin configuration (CV 0.907) generalized (0.929); the L1 and two-bin variants bound this conclusion.

| Bins | Reg. | C | CV | West China | Japan | Weights |
|---:|---|---:|---:|---:|---:|---:|
| **3** | **L2** | **1.5** | **0.907** | **0.929** | **0.843** | **7** |
| 3 | L2 | 0.5 | 0.904 | 0.919 | 0.873 | 7 |
| 3 | L2 | 5.0 | 0.903 | 0.917 | 0.872 | 7 |
| 4 | L2 | 0.5 | 0.913 | 0.880 | 0.889 | 12 |
| 4 | L2 | 1.5 | 0.912 | 0.881 | 0.889 | 12 |
| 4 | L2 | 5.0 | 0.909 | 0.875 | 0.893 | 11 |
| 5 | L2 | 0.5 | 0.904 | 0.909 | 0.870 | 14 |
| 5 | L2 | 1.5 | 0.904 | 0.909 | 0.865 | 14 |
| 5 | L2 | 5.0 | 0.900 | 0.907 | 0.858 | 14 |
| 7 | L2 | 0.5 | 0.909 | 0.911 | 0.909 | 19 |
| 7 | L2 | 1.5 | 0.910 | 0.914 | 0.901 | 18 |
| 7 | L2 | 5.0 | 0.909 | 0.909 | 0.885 | 18 |
| 3 | L1 | 0.5 | 0.907 | 0.907 | 0.796 | 5 |
| 3 | L1 | 1.5 | 0.903 | 0.918 | 0.871 | 6 |
| 3 | L1 | 5.0 | 0.901 | 0.918 | 0.871 | 6 |
| 4 | L1 | 0.5 | 0.909 | 0.873 | 0.893 | 10 |
| 4 | L1 | 1.5 | 0.909 | 0.873 | 0.893 | 10 |
| 4 | L1 | 5.0 | 0.910 | 0.882 | 0.843 | 10 |
| 5 | L1 | 0.5 | 0.901 | 0.894 | 0.843 | 11 |
| 5 | L1 | 1.5 | 0.902 | 0.897 | 0.844 | 11 |
| 5 | L1 | 5.0 | 0.898 | 0.895 | 0.845 | 11 |
| 7 | L1 | 0.5 | 0.910 | 0.883 | 0.858 | 12 |
| 7 | L1 | 1.5 | 0.908 | 0.891 | 0.867 | 13 |
| 7 | L1 | 5.0 | 0.903 | 0.873 | 0.873 | 13 |
| 2 | L2 | 0.5 | 0.896 | 0.847 | 0.800 | 6 |
| 2 | L2 | 1.5 | 0.895 | 0.847 | 0.800 | 6 |
| 2 | L2 | 5.0 | 0.895 | 0.847 | 0.800 | 6 |
| 2 | L1 | 0.5 | 0.896 | 0.847 | 0.799 | 6 |
| 2 | L1 | 1.5 | 0.896 | 0.847 | 0.800 | 6 |
| 2 | L1 | 5.0 | 0.897 | 0.847 | 0.800 | 6 |

**Table S2. Variables recorded in the Chinese training dataset (n = 349).** The four modeling features — CA125, HE4, age, menopausal status — are in bold. Abbreviations: AFP = alpha-fetoprotein; CA19-9/CA72-4 = carbohydrate antigen 19-9/72-4; CEA = carcinoembryonic antigen; MCV = mean corpuscular volume; MCH = mean corpuscular hemoglobin; RBC = red blood cell count; RDW = red cell distribution width; MPV = mean platelet volume; PDW = platelet distribution width; ALT = alanine aminotransferase; AST = aspartate aminotransferase; GGT = gamma-glutamyl transferase; CO2 = total carbon dioxide.

| Category | Variables |
|---|---|
| Tumor markers (5) | AFP, CA125, CA19-9, CA72-4, CEA |
| Bioassay (1) | HE4 |
| Demographics (2) | Age, Menopausal status |
| Hematology, differentials (9) | Basophils (count, %), eosinophils (count, %), lymphocytes (count, %), monocytes (count, %), neutrophils |
| Hematology, red cells and platelets (10) | Hemoglobin, hematocrit, MCV, MCH, RBC, RDW, platelets, MPV, PDW, plateletcrit |
| Biochemistry, renal and electrolytes (9) | Urea nitrogen, calcium, creatinine, potassium, magnesium, sodium, phosphorus, chloride, CO2 |
| Biochemistry, liver, proteins, other (13) | Albumin, alkaline phosphatase, ALT, AST, direct bilirubin, GGT, globulin, glucose, indirect bilirubin, total bilirubin, total protein, uric acid, A/G ratio |
| Identifiers and outcome (2) | Subject ID (excluded from modeling), outcome label |

**Table S3. Repeated cross-validation and selection stability.** Mean (SD, range) five-fold CV AUROC across 20 repeats (seeds 0–19). The 1-SE rule selected a three-bin configuration in 18 of 20 repeats (90%; C = 0.5 in 13, C = 5.0 in 5), supporting the deployed configuration.

| Model | CV mean | between-repeat SD | min | max |
|---|---:|---:|---:|---:|
| Scorecard (3 bins, L2, C = 1.5) | 0.906 | 0.003 | 0.901 | 0.911 |
| Argmax control (4 bins, L2, C = 0.5) | 0.911 | 0.004 | 0.902 | 0.916 |
| Raw logistic regression | 0.894 | 0.003 | 0.890 | 0.900 |
| CatBoost (defaults) | 0.900 | 0.006 | 0.887 | 0.915 |
| Random Forest (defaults) | 0.895 | 0.005 | 0.884 | 0.903 |
| XGBoost (defaults) | 0.882 | 0.006 | 0.868 | 0.894 |

**Table S4. Nested cross-validation of tree tuning (outer five-fold; inner five-fold over a 9-configuration grid per family).** Nested CV AUROC: XGBoost 0.885 (SD 0.029), CatBoost 0.900 (0.040), Random Forest 0.903 (0.035). None exceeds the scorecard's repeated-CV mean of 0.906.

**Table S5. Individual bootstrap AUROC estimates (2,000 patient-level resamples, 0% noise).** Point estimate = bootstrap mean; CI = 95% percentile interval. Japan is descriptive (27 cancers).

| Model | West China (95% CI) | Japan (95% CI) |
|---|---:|---:|
| Scorecard | 0.929 (0.902–0.955) | 0.845 (0.714–0.949) |
| Raw logistic regression | 0.936 (0.908–0.961) | 0.806 (0.666–0.926) |
| ROMA | 0.887 (0.848–0.921) | 0.799 (0.666–0.908) |
| CPH-I | 0.868 (0.828–0.905) | 0.921 (0.852–0.973) |
| CA125 ≥ 35 U/mL rule | 0.642 (0.597–0.684) | 0.781 (0.685–0.874) |
| XGBoost | 0.887 (0.850–0.922) | 0.736 (0.597–0.861) |
| CatBoost | 0.879 (0.839–0.917) | 0.750 (0.613–0.871) |
| Random Forest | 0.879 (0.839–0.916) | 0.794 (0.681–0.896) |

**Table S6. AUROC by measurement-noise level (West China; external).** Means (± SD) across 100 independent perturbations per level (proportional Gaussian noise on CA125 and HE4).

| Noise | ROMA | CPH-I | Raw LR | XGBoost | CatBoost | Random Forest | Scorecard |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.886 | 0.868 | 0.936 | 0.887 | 0.879 | 0.879 | 0.929 |
| 10% | 0.846 | 0.868 | 0.936 | 0.871 | 0.877 | 0.880 | 0.919 |
| 20% | 0.830 | 0.864 | 0.936 | 0.835 | 0.851 | 0.854 | 0.902 |
| 30% | 0.813 | 0.857 | 0.934 | 0.800 | 0.826 | 0.826 | 0.890 |

**Table S7. Robustness across noise types and gross errors (West China).** Mean AUROC across 100 perturbations. Additive noise = ±10%/30% of the feature median; combined = proportional plus additive; outliers = 5% of biomarker values scaled by ×3 or ÷3.

| Noise type | Level | Scorecard | Raw LR | ROMA | CPH-I | XGBoost | CatBoost | Random Forest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Multiplicative | 30% | 0.890 | 0.934 | 0.813 | 0.858 | 0.803 | 0.828 | 0.828 |
| Additive | 30% | 0.892 | 0.934 | 0.811 | 0.858 | 0.807 | 0.833 | 0.832 |
| Combined | 30% | 0.869 | 0.932 | 0.791 | 0.839 | 0.775 | 0.810 | 0.803 |
| Outliers | 5% | 0.911 | 0.933 | 0.871 | 0.863 | 0.873 | 0.870 | 0.871 |

**Table S8. Bin-boundary sensitivity (West China).** Frozen tertile cutpoints shifted by the given fraction of the transformed range before scoring.

| Cutpoint shift | AUROC (scorecard) |
|---:|---:|
| −10% | 0.913 |
| −5% | 0.929 |
| 0% | 0.929 |
| +5% | 0.926 |
| +10% | 0.917 |

**Table S9. External learning curves (West China AUROC; mean of 10 stratified training draws).**

| n | Scorecard | Raw LR | XGBoost | CatBoost | Random Forest |
|---:|---:|---:|---:|---:|---:|
| 60 | 0.901 | 0.925 | 0.847 | 0.888 | 0.899 |
| 120 | 0.911 | 0.931 | 0.870 | 0.889 | 0.896 |
| 240 | 0.918 | 0.934 | 0.883 | 0.879 | 0.887 |
| 349 | 0.929 | 0.936 | 0.887 | 0.878 | 0.887 |

**Table S10. Precision-recall AUC by cohort.** Japan is the imbalanced setting (15.3% prevalence).

| Model | West China | Japan |
|---|---:|---:|
| Scorecard | 0.928 | 0.705 |
| Raw logistic regression | 0.937 | 0.711 |
| ROMA | 0.913 | 0.681 |
| CPH-I | 0.901 | 0.775 |
| XGBoost | 0.918 | 0.608 |
| CatBoost | 0.921 | 0.642 |
| Random Forest | 0.917 | 0.645 |

**Figure S1. Per-feature drift decomposition (West China).** Change in AUROC (0% → 30% noise) for each of the eight models when proportional noise is applied to CA125 only, HE4 only, or both markers. CA125 = cancer antigen 125; HE4 = human epididymis protein 4; AUROC = area under the receiver operating characteristic curve; ROMA = Risk of Ovarian Malignancy Algorithm; CPH-I = Copenhagen Index.

**Figure S2. Real cross-cohort shift versus transfer loss.** Mean Kolmogorov–Smirnov statistic of the four features (versus the Chinese training distribution) against transfer loss (internal estimate minus external AUROC) for each model and external cohort. For fitted models the internal estimate is the five-fold cross-validated AUROC; for the published formulas it is their applied training AUROC. AUROC = area under the receiver operating characteristic curve; CV = cross-validation.

**Figure S3. Internal learning curves.** Five-fold cross-validated AUROC against training subsample size (10 draws), Chinese cohort. Points are means ± SD across draws; the dashed line marks ROMA's applied training AUROC. AUROC = area under the receiver operating characteristic curve; CV = cross-validation; ROMA = Risk of Ovarian Malignancy Algorithm.

**Figure S4. Robustness across noise types and gross errors (West China).** Change in AUROC versus the clean baseline for each of the eight models under multiplicative noise (30%), additive noise (30%), combined proportional-plus-additive noise (30%), and gross-error contamination (5% of biomarker values scaled by ×3 or ÷3); 100 perturbations per scenario. AUROC = area under the receiver operating characteristic curve.

---

## References

<!-- REFERENCES-GENERATED -->
