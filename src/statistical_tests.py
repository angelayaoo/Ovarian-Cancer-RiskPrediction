"""
Statistical tests and risk stratification for the paper.

1. Bootstrap 95% CI and pairwise DeLong-like comparison for all models
2. Risk tier stratification from integer scorecard
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import norm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monte_carlo_benchmarks import (
    parse_cohort_data, compute_roma, PreTrainedRiskSLIM, KNNImputer
)

DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "chinese_train_cleaned.csv")
WEST_PATH = os.path.join(DATA_DIR, "west_china_processed.csv")
JAPAN_PATH = os.path.join(DATA_DIR, "japan_processed.csv")


def load_and_train():
    """Load data, KNN impute, train all models, return predictions."""
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

    qt = QuantileTransformer(n_quantiles=min(100, len(X_train)), output_distribution='uniform', random_state=42)
    X_train_norm = pd.DataFrame(qt.fit_transform(X_train), columns=X_train.columns)

    # Train all models
    rslim = PreTrainedRiskSLIM(n_bins=3)
    rslim.fit(X_train, y_train)

    xgb = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.3, random_state=42, eval_metric='logloss')
    xgb.fit(X_train_norm, y_train)

    cat = CatBoostClassifier(n_estimators=500, depth=10, learning_rate=0.3, random_seed=42, verbose=0)
    cat.fit(X_train_norm, y_train)

    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train_norm, y_train)

    rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
    rf.fit(X_train_norm, y_train)

    models = {
        'RiskSLIM': lambda X: rslim.predict_score(X),
        'XGBoost': lambda X: xgb.predict_proba(pd.DataFrame(qt.transform(X), columns=X.columns))[:, 1],
        'CatBoost': lambda X: cat.predict_proba(pd.DataFrame(qt.transform(X), columns=X.columns))[:, 1],
        'Decision Tree': lambda X: dt.predict_proba(pd.DataFrame(qt.transform(X), columns=X.columns))[:, 1],
        'Random Forest': lambda X: rf.predict_proba(pd.DataFrame(qt.transform(X), columns=X.columns))[:, 1],
        'ROMA': lambda X: compute_roma(X),
    }

    return models, rslim, X_train, y_train


def bootstrap_pairwise_comparison(X_w, y_w, models, n_boot=2000, alpha=0.05):
    """Bootstrap-based pairwise AUROC comparison with 95% CI."""
    print("=" * 75)
    print("  1. BOOTSTRAP PAIRWISE AUROC COMPARISON (West China, 0% noise)")
    print("=" * 75)

    # Precompute predictions
    preds = {name: fn(X_w) for name, fn in models.items()}
    model_names = list(models.keys())
    n = len(y_w)

    results = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            diffs = []
            rng = np.random.RandomState(42)
            for _ in range(n_boot):
                idx = rng.choice(n, n, replace=True)
                a1 = roc_auc_score(y_w[idx], preds[m1][idx])
                a2 = roc_auc_score(y_w[idx], preds[m2][idx])
                a1 = a1 if a1 >= 0.5 else 1 - a1
                a2 = a2 if a2 >= 0.5 else 1 - a2
                diffs.append(a1 - a2)

            diffs = np.array(diffs)
            mean_diff = np.mean(diffs)
            ci_low = np.percentile(diffs, alpha / 2 * 100)
            ci_high = np.percentile(diffs, (1 - alpha / 2) * 100)
            p_val = np.mean(np.array(diffs) <= 0) if mean_diff > 0 else np.mean(np.array(diffs) >= 0)
            p_val = min(p_val, 1 - p_val) * 2  # two-sided
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            results.append((m1, m2, mean_diff, ci_low, ci_high, p_val, sig))

    print(f"  {'Model A':<15} {'Model B':<15} {'Δ AUROC':>10} {'95% CI':>20} {'p-value':>10} {'':<5}")
    print(f"  {'─'*15} {'─'*15} {'─'*10} {'─'*20} {'─'*10} {'─'*5}")
    for m1, m2, diff, lo, hi, p, sig in sorted(results, key=lambda x: x[5]):  # sort by p-value
        print(f"  {m1:<15} {m2:<15} {diff:>+10.4f} [{lo:.4f}, {hi:.4f}] {p:>10.4f} {sig:<5}")
    return results


def bootstrap_ci_individual(X_w, y_w, models, n_boot=2000):
    """Bootstrap 95% CI for each model individually."""
    print(f"\n{'='*75}")
    print("  2. INDIVIDUAL MODEL 95% BOOTSTRAP CI (West China)")
    print(f"{'='*75}")
    preds = {name: fn(X_w) for name, fn in models.items()}
    n = len(y_w)

    print(f"  {'Model':<20} {'AUROC':>8} {'95% CI':>22} {'SE':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*22} {'─'*8}")
    for name, p in preds.items():
        aucs = []
        rng = np.random.RandomState(42)
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            a = roc_auc_score(y_w[idx], p[idx])
            aucs.append(a if a >= 0.5 else 1 - a)
        aucs = np.array(aucs)
        print(f"  {name:<20} {np.mean(aucs):>8.4f} [{np.percentile(aucs, 2.5):.4f}, {np.percentile(aucs, 97.5):.4f}]   {np.std(aucs):>8.4f}")


def risk_stratification(rslim, X_w, y_w):
    """Stratify West China patients by RiskSLIM score into risk tiers."""
    print(f"\n{'='*75}")
    print("  3. RISK STRATIFICATION (West China, integer scorecard)")
    print(f"{'='*75}")
    scores = rslim.predict_score(X_w)

    # Define tiers
    tiers = [
        (-999, -2, "Very Low Risk"),
        (-1, 0, "Low Risk"),
        (1, 2, "Low-Intermediate"),
        (3, 4, "Intermediate"),
        (5, 6, "High-Intermediate"),
        (7, 999, "High Risk"),
    ]

    print(f"  {'Risk Tier':<22} {'Score Range':>12} {'N':>6} {'Cancer':>8} {'Rate':>8} {'Sens':>8} {'Spec':>8}")
    print(f"  {'─'*22} {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    total_cancer = y_w.sum()
    total_benign = (1 - y_w).sum()
    cum_tp = 0

    for lo, hi, label in tiers:
        mask = (scores >= lo) & (scores <= hi)
        n = mask.sum()
        cancer_n = y_w[mask].sum()
        rate = cancer_n / n * 100 if n > 0 else 0
        cum_tp += cancer_n
        sens = cum_tp / total_cancer * 100 if total_cancer > 0 else 0
        # For spec: patients ABOVE this tier are called positive
        above = scores > hi
        tn = ((1 - y_w) & ~above).sum()
        spec = tn / total_benign * 100 if total_benign > 0 else 0
        print(f"  {label:<22} {f'[{lo}, {hi}]':>12} {n:>6} {cancer_n:>8} {rate:>7.1f}% {sens:>7.1f}% {spec:>7.1f}%")


def main():
    models, rslim, X_train, y_train = load_and_train()

    # Load West China
    df_west = pd.read_csv(WEST_PATH)
    X_w, y_w = parse_cohort_data(df_west)
    print(f"West China: {len(X_w)} patients, {y_w.sum()} cancer\n")

    # 1. Pairwise bootstrap comparison
    pairwise = bootstrap_pairwise_comparison(X_w, y_w, models)

    # 2. Individual CIs
    bootstrap_ci_individual(X_w, y_w, models)

    # 3. Risk stratification
    risk_stratification(rslim, X_w, y_w)

    # 4. Scorecard weights
    print(f"\n{'='*75}")
    print("  4. SCORECARD WEIGHTS")
    print(f"{'='*75}")
    print(f"  {'Feature':<25} {'Bin 1 (Low)':>12} {'Bin 2 (Mid)':>12} {'Bin 3 (High)':>12}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*12}")
    feats = X_train.columns.tolist()
    for i, feat in enumerate(feats):
        w_str = []
        for b in range(3):
            idx = i * 3 + b
            w_str.append(f"{rslim.weights[idx]:+d}" if rslim.weights[idx] != 0 else "0")
        print(f"  {feat:<25} {w_str[0]:>12} {w_str[1]:>12} {w_str[2]:>12}")
    print(f"\n  Total non-zero weights: {np.sum(np.abs(rslim.weights) > 0)} / {len(rslim.weights)}")


if __name__ == "__main__":
    main()
