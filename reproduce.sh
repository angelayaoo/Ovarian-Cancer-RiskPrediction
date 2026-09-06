#!/usr/bin/env bash
# One-shot reproduction of all numbers and figures for the ML4H 2026 paper.
# Run from the repository root. Runtime ~45 min (simulation is the slow part).
set -euo pipefail

echo "[1/9] clinical comparators (Table 1 + noise grid)"
python src/clinical_comparators.py

echo "[2/9] repeated CV + selection stability + nested CV"
python src/repeated_cv.py

echo "[3/9] meta-analytic pooling (Table 2)"
python src/meta_analysis.py

echo "[4/9] calibration, PR-AUC, NRI/IDI"
python src/calibration_metrics.py

echo "[5/9] weighted-harm decision analysis (Table 3)"
python src/decision_harm_analysis.py

echo "[6/9] DCA with bootstrap bands"
python src/decision_analysis.py

echo "[7/9] robustness experiments + learning curves"
python src/robustness_extended.py

echo "[8/9] simulation study (Table 4 + regime map, ~25 min)"
python src/simulation_study.py

echo "[9/9] paper figures (Figures 1-5)"
python src/make_ml4h_figures.py

echo "Done. Paper: cd paper && tectonic -X compile main.tex"
