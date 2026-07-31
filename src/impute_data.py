"""
KNN Imputation — per class label, prevents data leakage.

Methodology (Section 2.2):
K-Nearest Neighbors (KNN) imputation applied to the training cohort
separately for cancer (label=1) and benign (label=0) patients.
Uses k=5 nearest neighbors within each class.
"""
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

PROCESSED_DIR = "data/processed"

COHORT_FILES = [
    "chinese_discovery_cleaned.csv",
    "west_china_cleaned.csv",
    "japanese_cleaned.csv"
]


def knn_impute_per_class(file_path, output_suffix="_final.csv"):
    """KNN imputation isolated per class label (k=5)."""
    if not os.path.exists(file_path):
        print(f"  Skipping {file_path}: not found")
        return None

    df = pd.read_csv(file_path)
    print(f"\n  Processing: {file_path} ({len(df)} rows)")

    # Ensure label is binary int
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    # Convert all non-label columns to numeric
    feature_cols = [c for c in df.columns if c != 'label']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[><\t\s]', '', regex=True), errors='coerce')

    # Drop columns that are entirely NaN
    empty_cols = [c for c in feature_cols if df[c].isnull().all()]
    if empty_cols:
        print(f"    Dropping empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)
        feature_cols = [c for c in feature_cols if c not in empty_cols]

    print(f"    Missing before imputation: {df[feature_cols].isnull().sum().sum()}")

    # Impute per class
    imputed_groups = []
    for lbl_val in [0, 1]:
        group = df[df['label'] == lbl_val].copy()
        if len(group) < 3:
            imputed_groups.append(group)
            continue

        labels = group['label']
        features = group[feature_cols].copy()

        # Fill entirely-NaN columns with 0
        all_nan = [c for c in feature_cols if features[c].isnull().all()]
        if all_nan:
            features[all_nan] = 0.0

        if features.isnull().sum().sum() > 0:
            imp = KNNImputer(n_neighbors=min(5, len(group)))
            imputed_vals = imp.fit_transform(features)
            features = pd.DataFrame(imputed_vals, columns=features.columns, index=features.index)

        features['label'] = labels.values
        imputed_groups.append(features)

    final_df = pd.concat(imputed_groups).sort_index()

    # Add CA125/HE4 ratio
    if 'ca125' in final_df.columns and 'he4' in final_df.columns:
        final_df['ca125_he4_ratio'] = final_df['ca125'] / (final_df['he4'] + 1e-5)

    print(f"    Missing after imputation: {final_df[feature_cols].isnull().sum().sum()}")

    output_path = os.path.join(
        os.path.dirname(file_path),
        os.path.basename(file_path).replace("_cleaned.csv", output_suffix)
    )
    final_df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path}")
    return final_df


def run_imputation():
    print("=" * 55)
    print("  KNN IMPUTATION (per class, k=5)")
    print("=" * 55)

    for file_name in COHORT_FILES:
        file_path = os.path.join(PROCESSED_DIR, file_name)
        knn_impute_per_class(file_path)

    print("\nAll cohorts imputed.")


if __name__ == "__main__":
    run_imputation()
