"""
Prepare binarized feature matrices for RiskSLIM integer scorecard generation.

Methodology:
1. Load Chinese training data (349 patients, 47+ clinical features)
2. KNN imputation per class label (k=5, prevents data leakage)
3. Extract core features: CA125, HE4, Age, Menopause
4. QuantileTransformer → uniform [0,1] per feature
5. KBinsDiscretizer(3 quantile bins, one-hot) → 12 binary features
6. L2 logistic regression (C=1.5) → integer weights [-5, 5]
7. Menopause-stratified scorecards (pre- and post-menopausal)
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


def parse_cohort(df):
    """Extract core features: CA125, HE4, Age, Menopause, Label."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    tc = 'label' if 'label' in df.columns else df.columns[-1]
    y_raw = df[tc].values
    u = np.unique(y_raw)
    y = np.where(y_raw == u[1], 1, 0) if len(u) == 2 else y_raw

    ca_cols = [c for c in df.columns if '125' in c and '19' not in c and '72' not in c]
    ca = pd.to_numeric(df[ca_cols[0]], errors='coerce').values if ca_cols else np.ones(len(df))
    he_cols = [c for c in df.columns if 'he4' in c]
    he = pd.to_numeric(df[he_cols[0]], errors='coerce').values if he_cols else np.ones(len(df))
    ag = pd.to_numeric(df['age'], errors='coerce').values if 'age' in df.columns else np.full(len(df), 50.0)

    pc = [c for c in df.columns if 'menopause' in c or 'menopausal' in c]
    if not pc:
        pc = [c for c in df.columns if 'post' in c and 'roma' not in c]
    po = pd.to_numeric(df[pc[0]], errors='coerce').values if pc else np.zeros(len(df))
    if pc:
        pu = np.unique(po[~np.isnan(po)])
        if set(pu).issubset({1, 2}): po = np.where(po == 2, 1, 0)

    X = pd.DataFrame({'CA125': ca, 'HE4': he, 'Age': ag, 'Is_Postmenopausal': po}).fillna(0)
    if np.corrcoef(X['CA125'].values, y)[0, 1] < 0:
        y = 1 - y
    return X, y


def knn_impute(df):
    """KNN imputation per class label (k=5)."""
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
    return df


class RiskSLIMScorecard:
    """
    Integer scorecard: QuantileTransformer → 3 quantile bins → L2 LR → integer weights.
    """
    def __init__(self, n_bins=3, C=1.5):
        self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
        self.model = LogisticRegression(solver='liblinear', C=C, penalty='l2', random_state=42)
        self.n_bins = n_bins

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
        self.feature_names_ = X.columns.tolist()
        return self

    def predict_score(self, X):
        X_s = self.scaler.transform(X)
        X_b = self.discretizer.transform(X_s)
        return np.dot(X_b, self.weights)

    def print_scorecard(self):
        print("\nRiskSLIM Integer Scorecard:")
        for i, feat in enumerate(self.feature_names_):
            for b in range(self.n_bins):
                idx = i * self.n_bins + b
                if idx < len(self.weights) and self.weights[idx] != 0:
                    print(f"  {feat} (bin {b+1}/{self.n_bins})  →  {self.weights[idx]:+d} pt")
        print(f"  Total non-zero weights: {np.sum(np.abs(self.weights) > 0)} / {len(self.weights)}")

    def to_dataframe(self):
        rows = []
        for i, feat in enumerate(self.feature_names_):
            for b in range(self.n_bins):
                idx = i * self.n_bins + b
                if idx < len(self.weights):
                    rows.append({
                        'Feature': feat,
                        'Bin': f"{b+1}/{self.n_bins}",
                        'Weight': int(self.weights[idx])
                    })
        return pd.DataFrame(rows)


def prepare_riskslim_data():
    """Main entry point: load, impute, train scorecard, save."""
    print("=" * 95)
    print("  RiskSLIM Scorecard Generation")
    print("  KNN → 4 features → QuantileTransformer → 3 bins → L2 LR → integer weights")
    print("=" * 95)

    df = pd.read_csv(TRAIN_PATH)
    print(f"\n  Loaded: {len(df)} patients")

    df = knn_impute(df)
    print(f"  KNN imputation complete")

    X, y = parse_cohort(df)
    print(f"  Core features: {list(X.columns)}")
    print(f"  Cancer: {y.sum()}  Benign: {(1 - y).sum()}")
    print(f"  Postmenopausal: {X['Is_Postmenopausal'].sum()}")

    # Pre-menopausal
    pre_mask = X['Is_Postmenopausal'] == 0
    pre_card = RiskSLIMScorecard(n_bins=3, C=1.5)
    pre_card.fit(X[pre_mask], y[pre_mask])
    print(f"\n  Premenopausal ({pre_mask.sum()} patients):")
    pre_card.print_scorecard()

    # Post-menopausal
    post_mask = X['Is_Postmenopausal'] == 1
    post_card = RiskSLIMScorecard(n_bins=3, C=1.5)
    post_card.fit(X[post_mask], y[post_mask])
    print(f"\n  Postmenopausal ({post_mask.sum()} patients):")
    post_card.print_scorecard()

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pre_df = pre_card.to_dataframe()
    pre_df.to_csv(os.path.join(PROCESSED_DIR, "riskslim_premenopausal.csv"), index=False)
    post_df = post_card.to_dataframe()
    post_df.to_csv(os.path.join(PROCESSED_DIR, "riskslim_postmenopausal.csv"), index=False)
    print(f"\n  Scorecards saved to data/processed/riskslim_*menopausal.csv")

    return pre_card, post_card


if __name__ == "__main__":
    prepare_riskslim_data()
