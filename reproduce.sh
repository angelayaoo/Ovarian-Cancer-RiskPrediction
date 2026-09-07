#!/usr/bin/env bash
# One-shot reproduction of all numbers and figures for the IEEE BIBM 2026
# Undergraduate & High School Symposium paper (bmib_hs/main.tex).
# Run from the repository root. Runtime ~45 min (simulation is the slow part).
set -euo pipefail

echo "[1/10] harmonize raw cohort files into data/processed (byte-for-byte)"
python src/clean_data.py

echo "[2/10] clinical comparators (Table II West/Japan columns, pairwise tests)"
python src/clinical_comparators.py

echo "[3/10] repeated CV + selection stability + nested CV (Section III-E)"
python src/repeated_cv.py

echo "[4/10] meta-analytic pooling (pooled +0.043 comparison)"
python src/meta_analysis.py

echo "[5/10] calibration, Brier, calibration-in-the-large"
python src/calibration_metrics.py

echo "[6/10] weighted-harm decision analysis (Table III base + Q1 operating points)"
python src/decision_harm_analysis.py

echo "[7/10] robustness experiments + learning curves"
python src/robustness_extended.py

echo "[8/10] simulation study (Fig. 3 + factor decomposition, ~25 min)"
python src/simulation_study.py

echo "[9/10] tree tuning grid + scorecard selection sweep"
python src/tree_tuning_grid.py
python src/select_config.py

echo "[10/10] BIBM-specific artifacts: complete-case CIs, Setting B ensembles,"
echo "       MICE sensitivity, recalibrated harms, calibration slopes,"
echo "       and Figures 1-3 in bmib_hs/figures/"
python src/make_bmib_artifacts.py

echo "Done. Compile the paper with: cd bmib_hs && tectonic main.tex"
