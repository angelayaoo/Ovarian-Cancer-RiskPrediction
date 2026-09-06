"""
Shared benchmark library for the model-complexity vs. transportability study.

Central model zoo: every model is trained once on the Chinese training cohort
(frozen) and applied unchanged to the external cohorts. Clinical fixed
formulas (ROMA, CPH-I, CA125 rule) are computed directly from raw features.

Models:
    Scorecard   - quantile-binned, L2 logistic regression with integer weights
    Raw LR      - standardized 4-feature logistic regression (unbinned)
    ROMA        - published Risk of Ovarian Malignancy Algorithm (Moore 2011)
    CPH-I       - published Copenhagen index (Karlsen 2015)
    CA125 rule  - single-marker rule: CA125 >= 35 U/mL
    XGBoost     - gradient-boosted trees, package defaults
    CatBoost    - ordered boosting, package defaults
    RandomForest- bagged trees, package defaults
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = 'data/processed'
RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
SEED = 42

for _d in (RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, 'chinese_train_cleaned.csv')
WEST_PATH = os.path.join(DATA_DIR, 'west_china_processed.csv')
JAPAN_PATH = os.path.join(DATA_DIR, 'japan_processed.csv')

MODEL_COLORS = {
    'Scorecard': '#2ca02c', 'Raw LR': '#8c564b', 'ROMA': '#d62728',
    'CPH-I': '#17becf', 'CA125 rule': '#7f7f7f',
    'XGBoost': '#ff7f0e', 'CatBoost': '#9467bd', 'Random Forest': '#1f77b4',
}
MODEL_LABELS = {
    'Scorecard': 'Integer scorecard', 'Raw LR': 'Logistic regression',
    'ROMA': 'ROMA', 'CPH-I': 'CPH-I', 'CA125 rule': 'CA125 \u2265 35 U/mL rule',
    'XGBoost': 'XGBoost', 'CatBoost': 'CatBoost', 'Random Forest': 'Random forest',
}

ROMA_CUT_PRE = 13.1   # published high-risk cutoff, premenopausal (%)
ROMA_CUT_POST = 27.7  # published high-risk cutoff, postmenopausal (%)
CPHI_CUT = 7.0        # published high-risk cutoff (%)
CA125_CUT = 35.0      # classic CA125 cutoff (U/mL)


def parse_cohort_data(df, fill_medians=None):
    """Parse a cohort file into the 4-feature matrix and binary outcome."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    target_col = 'label' if 'label' in df.columns else df.columns[-1]
    y_raw = df[target_col].values
    uniques = np.unique(y_raw)
    if len(uniques) == 2:
        y = np.where(y_raw == uniques[1], 1, 0)
    else:
        y = y_raw

    ca125_col = [c for c in df.columns if '125' in c][0] if any('125' in c for c in df.columns) else None
    he4_col = [c for c in df.columns if 'he4' in c][0] if any('he4' in c for c in df.columns) else None
    age_col = [c for c in df.columns if 'age' in c][0] if any('age' in c for c in df.columns) else None
    post_col = [c for c in df.columns if 'menopause' in c or 'menopausal' in c]
    if not post_col:
        post_col = [c for c in df.columns if 'post' in c and 'roma' not in c]
    post_col = post_col[0] if post_col else None

    ca125_vals = pd.to_numeric(df[ca125_col], errors='coerce').values if ca125_col else np.ones(len(df))
    he4_vals = pd.to_numeric(df[he4_col], errors='coerce').values if he4_col else np.ones(len(df))
    age_vals = pd.to_numeric(df[age_col], errors='coerce').values if age_col else np.full(len(df), 50.0)

    if post_col:
        p_raw = pd.to_numeric(df[post_col], errors='coerce').values
        p_uniques = np.unique(p_raw[~np.isnan(p_raw)])
        if set(p_uniques).issubset({1, 2}):
            post_vals = np.where(p_raw == 2, 1, 0)
        else:
            post_vals = np.where(p_raw == 1, 1, 0)
    else:
        post_vals = np.zeros(len(df))

    X = pd.DataFrame({
        'CA125': ca125_vals,
        'HE4': he4_vals,
        'Age': age_vals,
        'Is_Postmenopausal': post_vals,
    })
    if fill_medians is not None:
        X = X.fillna({c: fill_medians[c] for c in X.columns})
    else:
        X = X.fillna(X.median())

    if np.corrcoef(X['CA125'].values, y)[0, 1] < 0:
        y = 1 - y
    return X, y


def load_train_imputed():
    """Training cohort: class-stratified KNN imputation over all numeric vars."""
    df = pd.read_csv(TRAIN_PATH)
    numeric_cols = []
    for c in df.columns:
        if c in ['label', 'subject']:
            continue
        if 'unnamed' in str(c).lower():
            continue
        s = pd.to_numeric(df[c].astype(str).str.replace(r'[><\t\s]', '', regex=True), errors='coerce')
        df[c] = s
        if s.notna().sum() > 10:
            numeric_cols.append(c)
    for lbl in [0, 1]:
        mask = df['label'] == lbl
        if mask.sum() < 3:
            continue
        Xg = df.loc[mask, numeric_cols].values.astype(float)
        if np.isnan(Xg).sum() > 0:
            imp = KNNImputer(n_neighbors=min(5, mask.sum()))
            df.loc[mask, numeric_cols] = imp.fit_transform(Xg)
    X, y = parse_cohort_data(df)
    return X, y


def load_external(path, medians):
    return parse_cohort_data(pd.read_csv(path), fill_medians=medians)


def compute_roma(X):
    """Published ROMA formula (Moore et al. 2011); returns risk in %."""
    ca125 = np.maximum(X['CA125'].values, 1e-5)
    he4 = np.maximum(X['HE4'].values, 1e-5)
    post = X['Is_Postmenopausal'].values
    pi_pre = -12.0 + (2.38 * np.log(he4)) + (0.0626 * np.log(ca125))
    pi_post = -8.09 + (1.04 * np.log(he4)) + (0.732 * np.log(ca125))
    pi = np.where(post == 1, pi_post, pi_pre)
    return (np.exp(pi) / (1.0 + np.exp(pi))) * 100.0


def compute_cphi(X):
    """Published Copenhagen Index CPH-I (Karlsen et al. 2015); risk in [0,1].
    CPH-I = -14.0647 + 1.0649*log2(HE4) + 0.6050*log2(CA125) + 0.2672*age/10;
    HE4 in pmol/L, CA125 in U/mL, age in years. Single formula, no menopausal
    status. PP = exp(CPH-I) / (1 + exp(CPH-I)); published cutoff >= 0.07."""
    ca125 = np.maximum(X['CA125'].values, 1e-5)
    he4 = np.maximum(X['HE4'].values, 1e-5)
    age = X['Age'].values
    cphi = (-14.0647 + 1.0649 * np.log2(he4) + 0.6050 * np.log2(ca125)
            + 0.2672 * (age / 10.0))
    return np.exp(cphi) / (1.0 + np.exp(cphi))


def compute_ca125_rule(X):
    """Single-marker rule: CA125 >= 35 U/mL (binary score)."""
    return (X['CA125'].values >= CA125_CUT).astype(float)


class Scorecard:
    """Quantile-binned L2 logistic regression with integer point weights."""

    def __init__(self, n_bins=3, C=1.5, penalty='l2', seed=SEED):
        self.n_bins = n_bins
        self.C = C
        self.penalty = penalty
        self.seed = seed
        self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform',
                                          random_state=seed)
        from sklearn.preprocessing import KBinsDiscretizer
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense',
                                            strategy='quantile')
        self.model = LogisticRegression(solver='liblinear', C=C, penalty=penalty,
                                        random_state=seed)

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


class RawLR:
    """Standardized 4-feature logistic regression (unbinned)."""

    def __init__(self, C=1.0, seed=SEED):
        self.C = C
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit(self, X, y):
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


def fit_model_zoo(X_train, y_train, include_trees=True):
    """Fit every model once on the training cohort; return predict functions.

    Returns dict name -> callable(X) -> continuous score (higher = more risk).
    """
    qt = QuantileTransformer(n_quantiles=min(100, len(X_train)),
                             output_distribution='uniform', random_state=SEED)
    Xn = pd.DataFrame(qt.fit_transform(X_train), columns=X_train.columns)

    sc = Scorecard(n_bins=3).fit(X_train, y_train)
    lr = RawLR().fit(X_train, y_train)

    zoo = {
        'Scorecard': lambda X: sc.predict_score(X),
        'Raw LR': lambda X: lr.predict_proba(X),
        'ROMA': lambda X: compute_roma(X),
        'CPH-I': lambda X: compute_cphi(X),
        'CA125 rule': lambda X: compute_ca125_rule(X),
    }
    if include_trees:
        xgb = XGBClassifier(random_state=SEED, eval_metric='logloss')
        xgb.fit(Xn, y_train)
        cat = CatBoostClassifier(random_seed=SEED, verbose=0)
        cat.fit(Xn, y_train)
        rf = RandomForestClassifier(random_state=SEED)
        rf.fit(Xn, y_train)
        zoo['XGBoost'] = lambda X: xgb.predict_proba(
            pd.DataFrame(qt.transform(X), columns=X.columns))[:, 1]
        zoo['CatBoost'] = lambda X: cat.predict_proba(
            pd.DataFrame(qt.transform(X), columns=X.columns))[:, 1]
        zoo['Random Forest'] = lambda X: rf.predict_proba(
            pd.DataFrame(qt.transform(X), columns=X.columns))[:, 1]
    return zoo, sc, lr, qt


def auroc(y, s):
    a = roc_auc_score(y, s)
    return a if a >= 0.5 else 1.0 - a


def bootstrap_auroc(y, s, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    aucs = np.empty(n_boot)
    n = len(y)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        aucs[b] = auroc(y[idx], s[idx])
    return aucs


def bootstrap_pairwise(y, s1, s2, n_boot=2000, seed=SEED):
    """Bootstrap distribution of AUROC(s1) - AUROC(s2)."""
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    n = len(y)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs[b] = auroc(y[idx], s1[idx]) - auroc(y[idx], s2[idx])
    return diffs


def pairwise_summary(diffs):
    m = float(np.mean(diffs))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    pv = min(np.mean(diffs <= 0), np.mean(diffs >= 0)) * 2 if m != 0 else 1.0
    return m, lo, hi, float(min(pv, 1.0))
