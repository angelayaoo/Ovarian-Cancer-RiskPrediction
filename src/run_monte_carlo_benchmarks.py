import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, QuantileTransformer, StandardScaler
from sklearn.impute import KNNImputer

def parse_cohort_data(df):
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
    he4_col   = [c for c in df.columns if 'he4' in c][0] if any('he4' in c for c in df.columns) else None
    age_col   = [c for c in df.columns if 'age' in c][0] if any('age' in c for c in df.columns) else None
    post_col  = [c for c in df.columns if 'menopause' in c or 'menopausal' in c]
    if not post_col:
        post_col = [c for c in df.columns if 'post' in c and 'roma' not in c]
    post_col = post_col[0] if post_col else None

    ca125_vals = pd.to_numeric(df[ca125_col], errors='coerce').values if ca125_col else np.ones(len(df))
    he4_vals   = pd.to_numeric(df[he4_col], errors='coerce').values if he4_col else np.ones(len(df))
    age_vals   = pd.to_numeric(df[age_col], errors='coerce').values if age_col else np.full(len(df), 50.0)
    
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
        'Is_Postmenopausal': post_vals
    })
    X = X.fillna(X.median())

    if np.corrcoef(X['CA125'].values, y)[0, 1] < 0:
        y = 1 - y

    return X, y

def compute_roma(df_core):
    ca125 = np.maximum(df_core['CA125'].values, 1e-5)
    he4   = np.maximum(df_core['HE4'].values, 1e-5)
    post  = df_core['Is_Postmenopausal'].values
    
    pi_pre  = -12.0 + (2.38 * np.log(he4)) + (0.0626 * np.log(ca125))
    pi_post = -8.09 + (1.04 * np.log(he4)) + (0.732 * np.log(ca125))
    
    pi = np.where(post == 1, pi_post, pi_pre)
    return (np.exp(pi) / (1.0 + np.exp(pi))) * 100.0

class PreTrainedRiskSLIM:
    def __init__(self, n_bins=3):
        self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
        self.model = LogisticRegression(solver='liblinear', C=1.5, penalty='l2', random_state=42)

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

    def predict_score(self, X):
        X_s = self.scaler.transform(X)
        X_b = self.discretizer.transform(X_s)
        return np.dot(X_b, self.weights)

data_dir = 'data/processed'
cohort_files = {
    'OG China (Primary)': os.path.join(data_dir, 'chinese_train_cleaned.csv'),
    'West China': os.path.join(data_dir, 'west_china_processed.csv'),
    'Japanese': os.path.join(data_dir, 'japan_processed.csv')
}

primary_path = cohort_files['OG China (Primary)']
df_primary = pd.read_csv(primary_path)

# KNN imputation on all numeric features (per class, prevents leakage)
numeric_cols = []
for c in df_primary.columns:
    if c in ['label', 'subject']: continue
    if 'unnamed' in str(c).lower(): continue
    s = pd.to_numeric(df_primary[c].astype(str).str.replace(r'[><\t\s]', '', regex=True), errors='coerce')
    df_primary[c] = s
    if s.notna().sum() > 10: numeric_cols.append(c)

for lbl_val in [0, 1]:
    mask = df_primary['label'] == lbl_val
    if mask.sum() < 3: continue
    Xg = df_primary.loc[mask, numeric_cols].values.astype(float)
    if np.isnan(Xg).sum() > 0:
        imp = KNNImputer(n_neighbors=min(5, mask.sum()))
        df_primary.loc[mask, numeric_cols] = imp.fit_transform(Xg)

X_train, y_train = parse_cohort_data(df_primary)

qt = QuantileTransformer(n_quantiles=min(100, len(X_train)), output_distribution='uniform', random_state=42)
X_train_norm = pd.DataFrame(qt.fit_transform(X_train), columns=X_train.columns)

fixed_xgb = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.3,
    random_state=42,
    eval_metric='logloss'
)
fixed_xgb.fit(X_train_norm, y_train)

fixed_cat = CatBoostClassifier(n_estimators=500, depth=10, learning_rate=0.3,
                               random_seed=42, verbose=0)
fixed_cat.fit(X_train_norm, y_train)

fixed_rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
fixed_rf.fit(X_train_norm, y_train)

fixed_rslim = PreTrainedRiskSLIM(n_bins=3)
fixed_rslim.fit(X_train, y_train)

# Print learned RiskSLIM scorecard
print("\n" + "="*95)
print("LEARNED RISKSLIM INTEGER SCORECARD")
print("="*95)
n_features = len(X_train.columns)
n_bins = 3
for i, feat in enumerate(X_train.columns):
    for b in range(n_bins):
        idx = i * n_bins + b
        if idx < len(fixed_rslim.weights) and fixed_rslim.weights[idx] != 0:
            print(f"  {feat} (bin {b+1}/{n_bins})  →  {fixed_rslim.weights[idx]:+d} pt")
print(f"  Total non-zero weights: {np.sum(np.abs(fixed_rslim.weights) > 0)} / {len(fixed_rslim.weights)}")
print("="*95)

noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_PERTURBATIONS = 100
results = []

np.random.seed(42)

for cohort_name, path in cohort_files.items():
    if not os.path.exists(path):
        continue

    df_cohort = pd.read_csv(path)
    X_clean, y_eval = parse_cohort_data(df_cohort)

    if len(np.unique(y_eval)) < 2:
        continue

    for noise in noise_levels:
        roma_aucs, xgb_aucs, cat_aucs, rf_aucs, rslim_aucs = [], [], [], [], []

        for _ in range(N_PERTURBATIONS):
            noise_mat = np.random.normal(1.0, noise, (len(X_clean), 2))
            X_noisy = X_clean.copy()
            X_noisy['CA125'] = np.maximum(X_clean['CA125'].values * noise_mat[:, 0], 1e-5)
            X_noisy['HE4']   = np.maximum(X_clean['HE4'].values * noise_mat[:, 1], 1e-5)

            X_noisy_norm = pd.DataFrame(qt.transform(X_noisy), columns=X_noisy.columns)

            roma_score = compute_roma(X_noisy)
            xgb_score  = fixed_xgb.predict_proba(X_noisy_norm)[:, 1]
            cat_score  = fixed_cat.predict_proba(X_noisy_norm)[:, 1]
            rf_score   = fixed_rf.predict_proba(X_noisy_norm)[:, 1]
            rslim_score = fixed_rslim.predict_score(X_noisy)

            roma_auc = roc_auc_score(y_eval, roma_score)
            xgb_auc  = roc_auc_score(y_eval, xgb_score)
            cat_auc  = roc_auc_score(y_eval, cat_score)
            rf_auc   = roc_auc_score(y_eval, rf_score)
            rslim_auc = roc_auc_score(y_eval, rslim_score)

            roma_aucs.append(roma_auc if roma_auc >= 0.5 else 1 - roma_auc)
            xgb_aucs.append(xgb_auc if xgb_auc >= 0.5 else 1 - xgb_auc)
            cat_aucs.append(cat_auc if cat_auc >= 0.5 else 1 - cat_auc)
            rf_aucs.append(rf_auc if rf_auc >= 0.5 else 1 - rf_auc)
            rslim_aucs.append(rslim_auc if rslim_auc >= 0.5 else 1 - rslim_auc)

        results.append({
            'Cohort': cohort_name,
            'Noise': f"{int(noise*100)}%",
            'ROMA Formula': f"{np.mean(roma_aucs):.4f} (±{np.std(roma_aucs):.3f})",
            'XGBoost': f"{np.mean(xgb_aucs):.4f} (±{np.std(xgb_aucs):.3f})",
            'CatBoost': f"{np.mean(cat_aucs):.4f} (±{np.std(cat_aucs):.3f})",
            'Random Forest': f"{np.mean(rf_aucs):.4f} (±{np.std(rf_aucs):.3f})",
            'Optimized RiskSLIM': f"{np.mean(rslim_aucs):.4f} (±{np.std(rslim_aucs):.3f})"
        })

res_df = pd.DataFrame(results)

print("\n" + "="*95)
print("BENCHMARK NOISE RESULTS (5% - 30% NOISE, FROZEN EVALUATION)")
print("="*95)
print(res_df.to_string(index=False))
print("="*95 + "\n")

os.makedirs('results', exist_ok=True)
res_df.to_csv('results/monte_carlo_benchmarks.csv', index=False)
