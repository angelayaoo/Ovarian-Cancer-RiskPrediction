"""
Gaussian Lab Drift Simulation (Section 2.5).

Imports the trained models and preprocessing from run_monte_carlo_benchmarks
to ensure consistent evaluation. Injects proportional Gaussian noise
(σ = 5%–30%) into CA125 and HE4 and plots AUROC degradation.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monte_carlo_benchmarks import (
    parse_cohort_data, compute_roma, PreTrainedRiskSLIM,
    KNNImputer, QuantileTransformer, XGBClassifier
)

DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "chinese_train_cleaned.csv")
WEST_PATH = os.path.join(DATA_DIR, "west_china_processed.csv")
JAPAN_PATH = os.path.join(DATA_DIR, "japan_processed.csv")


def inject_noise(X, sigma, rng):
    X_n = X.copy()
    n = len(X_n)
    X_n['CA125'] = np.maximum(X_n['CA125'].values * rng.normal(1.0, sigma, n), 0.1)
    X_n['HE4'] = np.maximum(X_n['HE4'].values * rng.normal(1.0, sigma, n), 0.1)
    return X_n


def main():
    print("=" * 70)
    print("  GAUSSIAN LAB DRIFT SIMULATION")
    print("  Proportional noise: CA125/HE4 × N(1, σ), σ = 5%–30%")
    print("=" * 70)

    df_train = pd.read_csv(TRAIN_PATH)
    numeric_cols = []
    for c in df_train.columns:
        if c in ['label', 'subject']: continue
        if 'unnamed' in str(c).lower(): continue
        s = pd.to_numeric(df_train[c].astype(str).str.replace(r'[><\t\s]', '', regex=True), errors='coerce')
        df_train[c] = s
        if s.notna().sum() > 10: numeric_cols.append(c)
    for lbl in [0, 1]:
        mask = df_train['label'] == lbl
        if mask.sum() < 3: continue
        Xg = df_train.loc[mask, numeric_cols].values.astype(float)
        if np.isnan(Xg).sum() > 0:
            imp = KNNImputer(n_neighbors=min(5, mask.sum()))
            df_train.loc[mask, numeric_cols] = imp.fit_transform(Xg)

    X_train, y_train = parse_cohort_data(df_train)
    print(f"\nTraining: {len(X_train)} patients")

    qt = QuantileTransformer(n_quantiles=min(100, len(X_train)), output_distribution='uniform', random_state=42)
    X_train_norm = pd.DataFrame(qt.fit_transform(X_train), columns=X_train.columns)

    rslim = PreTrainedRiskSLIM(n_bins=3)
    rslim.fit(X_train, y_train)

    xgb = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.3,
                        random_state=42, eval_metric='logloss')
    xgb.fit(X_train_norm, y_train)

    noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    N_ITER = 100

    cohorts = [
        (TRAIN_PATH, "OG China (Internal)"),
        (WEST_PATH, "West China"),
        (JAPAN_PATH, "Japan"),
    ]

    all_drift = {}
    for path, name in cohorts:
        df_val = pd.read_csv(path)
        X_val, y_val = parse_cohort_data(df_val)
        print(f"\n  {name} ({len(X_val)} patients, {y_val.sum()} cancer):")
        print(f"  {'Noise':>8} {'RiskSLIM':>12} {'XGBoost':>12} {'ROMA':>12}")
        print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*12}")

        drift = {}
        for sigma in noise_levels:
            rs_aucs, xg_aucs, ro_aucs = [], [], []
            for s in range(N_ITER):
                rng = np.random.RandomState(s * 100 + int(sigma * 100))
                X_n = inject_noise(X_val, sigma, rng)
                X_n_norm = pd.DataFrame(qt.transform(X_n), columns=X_n.columns)

                rs = roc_auc_score(y_val, rslim.predict_score(X_n))
                xg = roc_auc_score(y_val, xgb.predict_proba(X_n_norm)[:, 1])
                ro = roc_auc_score(y_val, compute_roma(X_n))

                rs_aucs.append(rs if rs >= 0.5 else 1 - rs)
                xg_aucs.append(xg if xg >= 0.5 else 1 - xg)
                ro_aucs.append(ro if ro >= 0.5 else 1 - ro)

            drift[sigma] = (np.mean(rs_aucs), np.mean(xg_aucs), np.mean(ro_aucs))
            print(f"  {sigma*100:>5.0f}%  {np.mean(rs_aucs):>12.4f} {np.mean(xg_aucs):>12.4f} {np.mean(ro_aucs):>12.4f}")

        d_rs = drift[0.30][0] - drift[0.05][0]
        d_xg = drift[0.30][1] - drift[0.05][1]
        d_ro = drift[0.30][2] - drift[0.05][2]
        print(f"\n  Δ (5→30%): RiskSLIM={d_rs:+.4f}  XGBoost={d_xg:+.4f}  ROMA={d_ro:+.4f}")
        all_drift[name] = drift

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False, dpi=200)
    colors = {'RiskSLIM': '#dc2626', 'XGBoost': '#ea580c', 'ROMA': '#16a34a'}
    markers = {'RiskSLIM': 'o', 'XGBoost': 's', 'ROMA': '^'}
    styles = {'RiskSLIM': '-', 'XGBoost': '--', 'ROMA': ':'}
    noise_pct = [n * 100 for n in noise_levels]

    for idx, (name, drift) in enumerate(all_drift.items()):
        ax = axes[idx]
        for model in ['RiskSLIM', 'XGBoost', 'ROMA']:
            i = 0 if model == 'RiskSLIM' else 1 if model == 'XGBoost' else 2
            vals = [drift[s][i] for s in noise_levels]
            ax.plot(noise_pct, vals,
                    marker=markers[model], markersize=6, linewidth=2.5,
                    linestyle=styles[model], color=colors[model], label=model)
        ax.set_title(name, fontweight='bold', fontsize=12)
        ax.set_xlabel("Measurement Noise (%)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Validation AUROC", fontsize=10)
        ax.set_xlim(3, 32)
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', frameon=True, fontsize=9)

    fig.suptitle("Model Robustness to Proportional Biomarker Measurement Drift",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = "lab_drift_degradation.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nPlot saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
