"""
RiskSLIM Integer Scorecard Solver.

Implements the integer logistic regression scorecard:
QuantileTransformer → 3 quantile bins → L2(C=1.5) → integer weights [-5, 5]
Menopause-stratified (separate scorecards for pre- and post-menopausal).
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, QuantileTransformer
from sklearn.impute import KNNImputer

PROCESSED_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "chinese_train_cleaned.csv")


def load_and_prepare():
    """Load Chinese training data, KNN impute, extract core features."""
    df = pd.read_csv(TRAIN_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # KNN imputation per class
    numeric_cols = []
    for c in df.columns:
        if c in ['label', 'subject']: continue
        if 'unnamed' in str(c).lower(): continue
        s = pd.to_numeric(df[c].astype(str).str.replace(r'[><\t\s]', '', regex=True), errors='coerce')
        df[c] = s
        if s.notna().sum() > 10: numeric_cols.append(c)

    for lbl in [0, 1]:
        mask = df['label'] == lbl
        if mask.sum() < 3: continue
        Xg = df.loc[mask, numeric_cols].values.astype(float)
        if np.isnan(Xg).sum() > 0:
            imp = KNNImputer(n_neighbors=min(5, mask.sum()))
            df.loc[mask, numeric_cols] = imp.fit_transform(Xg)

    # Parse labels
    tc = 'label' if 'label' in df.columns else df.columns[-1]
    y_raw = df[tc].values
    u = np.unique(y_raw)
    y = np.where(y_raw == u[1], 1, 0) if len(u) == 2 else y_raw

    # Extract core features
    ca = pd.to_numeric(df['ca125'], errors='coerce').values
    he = pd.to_numeric(df['he4'], errors='coerce').values
    ag = pd.to_numeric(df['age'], errors='coerce').values
    po = pd.to_numeric(df['menopause'], errors='coerce').fillna(0).values

    X = pd.DataFrame({
        'CA125': ca, 'HE4': he, 'Age': ag, 'Is_Postmenopausal': po
    }).fillna(X.median() if 'X' in dir() else 0)

    if np.corrcoef(X['CA125'].values, y)[0, 1] < 0:
        y = 1 - y

    return X, y


def optimize_scorecard(X, y, stratum_name, n_bins=3, C=1.5, max_weight=5):
    """Fit integer scorecard on binarized features."""
    scaler = QuantileTransformer(n_quantiles=min(100, len(X)), output_distribution='uniform', random_state=42)
    X_s = scaler.fit_transform(X)

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
    X_b = discretizer.fit_transform(X_s)

    model = LogisticRegression(solver='liblinear', C=C, penalty='l2', random_state=42)
    model.fit(X_b, y)

    coefs = model.coef_[0]
    nonzero = np.abs(coefs) > 0.01
    if np.any(nonzero):
        scale = max_weight / np.max(np.abs(coefs[nonzero]))
        weights = np.round(coefs * scale).astype(int)
    else:
        weights = np.zeros_like(coefs, dtype=int)

    print(f"\n  {stratum_name} RiskSLIM Scorecard (n={len(X)}, bins={n_bins}):")
    for i, feat in enumerate(X.columns):
        for b in range(n_bins):
            idx = i * n_bins + b
            if idx < len(weights) and weights[idx] != 0:
                print(f"    {feat} (bin {b+1}/{n_bins})  →  {weights[idx]:+d} pt")
    print(f"    Non-zero weights: {np.sum(np.abs(weights) > 0)} / {len(weights)}")

    return weights


def run_pipeline():
    print("=" * 70)
    print("  RiskSLIM Integer Scorecard Solver")
    print("  Methodology: KNN → QT → 3 quantile bins → L2 LR → Integer Weights")
    print("=" * 70)

    X, y = load_and_prepare()
    print(f"\n  Loaded: {len(X)} patients, Cancer={y.sum()}, Benign={(1-y).sum()}")

    # Pre-menopausal
    pre_mask = X['Is_Postmenopausal'] == 0
    pre_weights = optimize_scorecard(X[pre_mask], y[pre_mask], "Premenopausal")

    # Post-menopausal
    post_mask = X['Is_Postmenopausal'] == 1
    post_weights = optimize_scorecard(X[post_mask], y[post_mask], "Postmenopausal")

    # Generate scoring function
    print("\n" + "=" * 70)
    print("  GENERATED SCORING FUNCTION")
    print("=" * 70)
    print("\ndef calculate_risk_score(patient):")
    print("    \"\"\"")
    print("    Calculate ovarian cancer risk score.")
    print("    patient: dict with keys CA125, HE4, Age, Is_Postmenopausal")
    print("    Returns: integer risk score (higher = more likely cancer)")
    print("    \"\"\"")
    print("    score = 0")

    feat_names = X.columns.tolist()
    for gender, mask, label in [('pre', pre_mask, 'premenopausal'), ('post', post_mask, 'postmenopausal')]:
        w = pre_weights if gender == 'pre' else post_weights
        condition = "not" if gender == 'pre' else ""
        print(f"    if {condition} patient['Is_Postmenopausal']:  # {label}")
        for i, feat in enumerate(feat_names):
            for b in range(3):
                idx = i * 3 + b
                if idx < len(w) and w[idx] != 0:
                    sign = '+' if w[idx] > 0 else ''
                    print(f"        score += bin_check(patient['{feat}'], {b}) * {sign}{w[idx]}")
    print("    return score")

    print("\nDone.")


if __name__ == "__main__":
    run_pipeline()
