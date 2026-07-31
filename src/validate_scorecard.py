"""
External Scorecard Validation on West China and Japanese cohorts.

Applies the learned integer scorecard (trained on Chinese primary cohort)
to external validation cohorts with Monte Carlo noise simulation.
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, QuantileTransformer, StandardScaler
from sklearn.impute import KNNImputer

DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "chinese_train_cleaned.csv")
WEST_PATH = os.path.join(DATA_DIR, "west_china_processed.csv")
JAPAN_PATH = os.path.join(DATA_DIR, "japan_processed.csv")


def parse_cohort(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    tc = 'label' if 'label' in df.columns else df.columns[-1]
    y_raw = df[tc].values; u = np.unique(y_raw)
    y = np.where(y_raw == u[1], 1, 0) if len(u) == 2 else y_raw
    ca_cols = [c for c in df.columns if '125' in c and '19' not in c and '72' not in c]
    ca = pd.to_numeric(df[ca_cols[0]], errors='coerce').values
    he_cols = [c for c in df.columns if 'he4' in c]
    he = pd.to_numeric(df[he_cols[0]], errors='coerce').values
    ag = pd.to_numeric(df['age'], errors='coerce').values if 'age' in df.columns else np.full(len(df), 50.0)
    pc = [c for c in df.columns if 'menopause' in c or 'menopausal' in c]
    if not pc: pc = [c for c in df.columns if 'post' in c and 'roma' not in c]
    po = pd.to_numeric(df[pc[0]], errors='coerce').values if pc else np.zeros(len(df))
    if pc:
        pu = np.unique(po[~np.isnan(po)])
        if set(pu).issubset({1, 2}): po = np.where(po == 2, 1, 0)
    X = pd.DataFrame({'CA125': ca, 'HE4': he, 'Age': ag, 'Is_Postmenopausal': po}).fillna(0)
    if np.corrcoef(X['CA125'].values, y)[0, 1] < 0: y = 1 - y
    return X, y


class RiskSLIMScorecard:
    def __init__(self, n_bins=3, C=1.5):
        self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
        self.model = LogisticRegression(solver='liblinear', C=C, penalty='l2', random_state=42)

    def fit(self, X, y):
        X_s = self.scaler.fit_transform(X)
        X_b = self.discretizer.fit_transform(X_s)
        self.model.fit(X_b, y)
        coefs = self.model.coef_[0]
        nonzero = np.abs(coefs) > 0.01
        if np.any(nonzero):
            scale = 5.0 / np.max(np.abs(coefs[nonzero]))
            self.weights = np.round(coefs * scale).astype(int)
        else:
            self.weights = np.zeros_like(coefs, dtype=int)
        return self

    def predict_score(self, X):
        X_s = self.scaler.transform(X)
        X_b = self.discretizer.transform(X_s)
        return np.dot(X_b, self.weights)


def main():
    print("=" * 70)
    print("  EXTERNAL SCORECARD VALIDATION")
    print("=" * 70)

    df_train = pd.read_csv(TRAIN_PATH)
    X_train, y_train = parse_cohort(df_train)

    # KNN imputation
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

    X_train, y_train = parse_cohort(df_train)

    # Train scorecard
    print(f"\nTraining on Chinese cohort: {len(X_train)} patients")
    scorecard = RiskSLIMScorecard(n_bins=3, C=1.5)
    scorecard.fit(X_train, y_train)
    print(f"  Non-zero weights: {np.sum(np.abs(scorecard.weights) > 0)} / {len(scorecard.weights)}")

    # External validation
    for path, name in [(WEST_PATH, "West China"), (JAPAN_PATH, "Japan")]:
        if not os.path.exists(path):
            print(f"\n  {name}: file not found")
            continue

        df_val = pd.read_csv(path)
        X_val, y_val = parse_cohort(df_val)

        rs_scores = scorecard.predict_score(X_val)
        auc = roc_auc_score(y_val, rs_scores)
        auc = auc if auc >= 0.5 else 1 - auc
        print(f"\n  {name} ({len(X_val)} patients, {y_val.sum()} cancer):")
        print(f"    Clean AUROC: {auc:.4f}")

        # Noise evaluation
        for noise in [0.05, 0.10, 0.20, 0.30]:
            aucs = []
            for s in range(50):
                rng = np.random.RandomState(s)
                X_n = X_val.copy()
                X_n['CA125'] = np.maximum(X_n['CA125'].values * rng.normal(1.0, noise, len(X_n)), 0.1)
                X_n['HE4'] = np.maximum(X_n['HE4'].values * rng.normal(1.0, noise, len(X_n)), 0.1)
                a = roc_auc_score(y_val, scorecard.predict_score(X_n))
                aucs.append(a if a >= 0.5 else 1 - a)
            print(f"    Noise {int(noise*100):>2}%: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
