import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
from xgboost import XGBClassifier

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def safe_read_csv(file_path):
    for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception:
            continue
    raise ValueError(f"Could not decode file {file_path}")


def clean_and_standardize_cohort(df, cohort_name="Cohort"):
    df = df.dropna(how='all').reset_index(drop=True)
    df.columns = [str(c).lower().strip() for c in df.columns]

    target_col = None
    priority_keywords = ['pathological diagnosis', 'diagnosis', 'outcome', 'histology', 'type', 'group', 'class']
    
    for priority_kw in priority_keywords:
        for col in df.columns:
            if priority_kw in col and not any(id_kw in col for id_kw in ['subject', 'sample', 'patient', 'menopausal', 'table', 'id', 'bleeding']):
                target_col = col
                break
        if target_col:
            break

    rename_dict = {}
    if target_col:
        rename_dict[target_col] = 'label'

    for col in df.columns:
        if col == target_col:
            continue
        if 'ca125' in col or 'ca-125' in col or 'ca_125' in col:
            rename_dict[col] = 'ca125'
        elif 'he4' in col or 'he-4' in col or 'he_4' in col:
            rename_dict[col] = 'he4'
        elif 'age' in col and 'stage' not in col:
            rename_dict[col] = 'age'
        elif 'meno' in col or 'menopause' in col:
            rename_dict[col] = 'menopause'

    df = df.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if col != 'label':
            df[col] = df[col].astype(str).str.replace('>', '', regex=False).str.replace('<', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'menopause' not in df.columns and 'age' in df.columns:
        df['menopause'] = (df['age'] >= 50).astype(int)

    if 'label' in df.columns:
        def map_label(val):
            s = str(val).lower().strip()
            if any(k in s for k in ['cancer', 'carcinoma', 'malignant', 'malignan', 'borderline', 'eoc', 'bot', 'case', 'tumor', 'tumour']):
                return 1
            if any(k in s for k in ['benign', 'normal', 'control', 'healthy', 'cyst', 'adenoma', 'endometrioma']):
                return 0
            try:
                num = float(s)
                # Mendeley raw type column: 0.0 = Epithelial Ovarian Cancer, 1.0 = Benign
                if num == 0.0:
                    return 1
                elif num == 1.0:
                    return 0
            except ValueError:
                pass
            return np.nan

        df['label'] = df['label'].apply(map_label)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

    feature_cols = [c for c in df.columns if c != 'label']
    if feature_cols and df[feature_cols].isnull().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

    if 'label' in df.columns:
        u, c = np.unique(df['label'], return_counts=True)
        print(f"[{cohort_name}] Target '{target_col}' -> Class Distribution (0=Benign, 1=Cancer): {dict(zip(u, c))}")

    return df


def binarize_features(df):
    df_bin = pd.DataFrame(index=df.index)
    if 'ca125' in df.columns:
        df_bin['ca125_gt_35']  = (df['ca125'] > 35).astype(int)
        df_bin['ca125_gt_100'] = (df['ca125'] > 100).astype(int)
    if 'he4' in df.columns:
        df_bin['he4_gt_70']  = (df['he4'] > 70).astype(int)
        df_bin['he4_gt_140'] = (df['he4'] > 140).astype(int)
    if 'age' in df.columns:
        df_bin['age_gt_50'] = (df['age'] > 50).astype(int)
    return df_bin


def calculate_clinical_roma(age, ca125, he4):
    ca125_safe = np.maximum(np.nan_to_num(ca125, nan=35.0), 0.1)
    he4_safe = np.maximum(np.nan_to_num(he4, nan=70.0), 0.1)
    age_safe = np.nan_to_num(age, nan=50.0)

    ln_he4 = np.log(he4_safe)
    ln_ca125 = np.log(ca125_safe)

    pi_pre = -12.0 + (2.38 * ln_he4) + (0.0626 * ln_ca125)
    pi_post = -8.09 + (1.04 * ln_he4) + (0.732 * ln_ca125)

    pi = np.where(age_safe >= 50, pi_post, pi_pre)
    roma = np.exp(pi) / (1.0 + np.exp(pi))
    return np.nan_to_num(roma, nan=0.5)


def run_pipeline():
    all_files = glob.glob(os.path.join("data", "**", "*.csv"), recursive=True)
    files = [f for f in all_files if not f.endswith("_processed.csv") and "processed" not in f]

    train_path = None
    for f in files:
        f_lower = os.path.basename(f).lower()
        if "chinese training" in f_lower or "original training" in f_lower or "mendeley" in f_lower:
            train_path = f
            break

    if not train_path:
        print("Training dataset not found in data/.")
        return

    print(f"Loading dataset: {train_path}\n")
    df_train = clean_and_standardize_cohort(safe_read_csv(train_path), cohort_name="Chinese Training Cohort")

    # --- 1. INTEGER POINT SCORE TRAINING ---
    X_bin = binarize_features(df_train)
    y = df_train['label'].values

    lr_bin = LogisticRegression(C=0.5, random_state=42)
    lr_bin.fit(X_bin, y)
    raw_weights = lr_bin.coef_[0] * 3.0
    integer_weights = dict(zip(X_bin.columns, np.round(raw_weights).astype(int)))

    print("\n==========================================")
    print(" 1. LEARNED CLINICAL INTEGER POINT SYSTEM")
    print("==========================================")
    for feature, points in integer_weights.items():
        print(f"  * {feature:<15} : {points:+d} pts")

    # --- 2. CROSS-VALIDATION WITH 95% CONFIDENCE INTERVALS ---
    print("\n==========================================")
    print(" 2. 5-FOLD CROSS-VALIDATION (AUROC & 95% CI)")
    print("==========================================")

    X_raw = df_train[['age', 'ca125', 'he4']]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_aucs, lr_aucs, roma_aucs = [], [], []

    for train_idx, val_idx in skf.split(X_raw, y):
        X_tr, X_va = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        xgb = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
        xgb.fit(X_tr, y_tr)
        xgb_preds = xgb.predict_proba(X_va)[:, 1]
        xgb_aucs.append(roc_auc_score(y_va, xgb_preds))

        lr = LogisticRegression(random_state=42)
        lr.fit(X_tr, y_tr)
        lr_preds = lr.predict_proba(X_va)[:, 1]
        lr_aucs.append(roc_auc_score(y_va, lr_preds))

        roma_preds = calculate_clinical_roma(X_va['age'].values, X_va['ca125'].values, X_va['he4'].values)
        roma_aucs.append(roc_auc_score(y_va, roma_preds))

    def print_ci(name, aucs):
        mean_auc = np.mean(aucs)
        std_err = np.std(aucs) / np.sqrt(len(aucs))
        ci_lower = max(0.0, mean_auc - 1.96 * std_err)
        ci_upper = min(1.0, mean_auc + 1.96 * std_err)
        print(f"{name:<25} | AUROC: {mean_auc:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")

    print_ci("Logistic Regression", lr_aucs)
    print_ci("XGBoost Classifier", xgb_aucs)
    print_ci("Clinical ROMA Baseline", roma_aucs)

    # --- 3. DECISION THRESHOLD TUNING ---
    print("\n==========================================")
    print(" 3. DECISION THRESHOLD TUNING (High Sensitivity Target)")
    print("==========================================")

    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    lr_final = LogisticRegression(random_state=42)
    lr_final.fit(X_train_split, y_train_split)
    probs = lr_final.predict_proba(X_test_split)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.5
    best_spec = 0.0
    target_sens = 0.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_split, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        if sens >= 0.90 and spec > best_spec:
            best_thresh = t
            best_spec = spec
            target_sens = sens

    print(f"Optimal Probability Cutoff Threshold : {best_thresh:.3f}")
    print(f"Achieved Sensitivity (Recall)        : {target_sens:.2%}")
    print(f"Corresponding Specificity            : {best_spec:.2%}")

    # --- 4. GAUSSIAN LAB DRIFT SIMULATION ---
    print("\n==========================================")
    print(" 4. GAUSSIAN LAB DRIFT SIMULATION (100 Iterations)")
    print("==========================================")

    xgb_final = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
    xgb_final.fit(X_train_split, y_train_split)

    sigmas = [0.05, 0.10, 0.15, 0.20]

    for sigma in sigmas:
        sim_lr_aucs, sim_xgb_aucs, sim_roma_aucs = [], [], []

        for _ in range(100):
            X_noisy = X_test_split.copy()
            X_noisy['ca125'] = np.maximum(X_noisy['ca125'] * (1.0 + np.random.normal(0, sigma, len(X_noisy))), 0.1)
            X_noisy['he4']   = np.maximum(X_noisy['he4']   * (1.0 + np.random.normal(0, sigma, len(X_noisy))), 0.1)

            sim_lr_aucs.append(roc_auc_score(y_test_split, lr_final.predict_proba(X_noisy)[:, 1]))
            sim_xgb_aucs.append(roc_auc_score(y_test_split, xgb_final.predict_proba(X_noisy)[:, 1]))

            r_preds = calculate_clinical_roma(X_noisy['age'].values, X_noisy['ca125'].values, X_noisy['he4'].values)
            sim_roma_aucs.append(roc_auc_score(y_test_split, r_preds))

        print(f"Noise s = {int(sigma*100):2d}% | Logistic: {np.mean(sim_lr_aucs):.4f} | XGBoost: {np.mean(sim_xgb_aucs):.4f} | ROMA: {np.mean(sim_roma_aucs):.4f}")


if __name__ == "__main__":
    run_pipeline()
