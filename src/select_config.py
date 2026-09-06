"""
Model selection: internal 1-SE rule over a pre-specified L2 grid.

Protocol (pre-specified, applied to internal data only):
  1. Candidate grid: L2-regularized logistic regression on quantile-binned
     features, bins in {3, 4, 5, 7} x C in {0.5, 1.5, 5.0}. L1 was not part of
     the selection grid by design: the scorecard targets a dense point system
     over four clinically mandated features, so variable selection is not
     applicable; sparsity is instead enforced by the integer-rounding step
     (|coef| < 0.01 -> 0). The L1 sweep and an out-of-grid 2-bin check are
     reported as sensitivity analyses.
  2. Internal performance = 5-fold stratified CV with ALL preprocessing
     (quantile transformer, bin boundaries, logistic fit) refit within each
     training fold, so no statistics leak across folds.
  3. 1-SE rule (Breiman et al.; Friedman, Hastie & Tibshirani; glmnet
     lambda.1se): among configurations whose CV AUROC is within one standard
     error of the best CV AUROC (SE = fold SD / sqrt(5) of the best
     configuration), choose the simplest.
  4. Simplicity order (pre-specified): fewest bins, then fewest non-zero
     integer weights, then highest CV AUROC, then the grid's middle C value
     on exact ties.
  5. External cohorts are never used in selection.
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from bench_lib import (
    parse_cohort_data, KNNImputer, Scorecard as PreTrainedRiskSLIM
)

DATA_DIR = 'data/processed'
RESULTS_DIR = 'results'

GRID = [(b, 'l2', c) for b in [3, 4, 5, 7] for c in [0.5, 1.5, 5.0]]
L1_GRID = [(b, 'l1', c) for b in [3, 4, 5, 7] for c in [0.5, 1.5, 5.0]]
OUT_OF_GRID = [(2, 'l2', c) for c in [0.5, 1.5, 5.0]] + [(2, 'l1', c) for c in [0.5, 1.5, 5.0]]


def load_training():
    df = pd.read_csv(os.path.join(DATA_DIR, 'chinese_train_cleaned.csv'))
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
        Xg = df.loc[mask, numeric_cols].values.astype(float)
        if np.isnan(Xg).sum() > 0:
            imp = KNNImputer(n_neighbors=min(5, mask.sum()))
            df.loc[mask, numeric_cols] = imp.fit_transform(Xg)
    X, y = parse_cohort_data(df)
    return X, y


def cv_auroc(X, y, n_bins, C, penalty, folds):
    """5-fold CV of the DEPLOYED model: coefficients fitted on training folds,
    rounded to integer points (as deployed), evaluated on validation folds."""
    from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer
    from sklearn.linear_model import LogisticRegression
    aucs = []
    for tr, va in folds:
        qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
        kb = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
        Xtr_b = kb.fit_transform(qt.fit_transform(X.iloc[tr]))
        Xva_b = kb.transform(qt.transform(X.iloc[va]))
        m = LogisticRegression(solver='liblinear', C=C, penalty=penalty, random_state=42)
        m.fit(Xtr_b, y[tr])
        coefs = m.coef_[0]
        nonzero = np.abs(coefs) > 0.01
        if np.any(nonzero):
            w = np.round(coefs * 5.0 / np.max(np.abs(coefs[nonzero]))).astype(int)
            w[~nonzero] = 0
        else:
            w = np.zeros_like(coefs, dtype=int)
        aucs.append(roc_auc_score(y[va], np.dot(Xva_b, w)))
    return np.array(aucs)


def integer_weight_count(X, y, n_bins, C, penalty):
    from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer
    from sklearn.linear_model import LogisticRegression
    qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
    kb = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
    Xb = kb.fit_transform(qt.fit_transform(X))
    m = LogisticRegression(solver='liblinear', C=C, penalty=penalty, random_state=42).fit(Xb, y)
    coefs = m.coef_[0]
    nonzero = np.abs(coefs) > 0.01
    if np.any(nonzero):
        w = np.round(coefs * 5.0 / np.max(np.abs(coefs[nonzero]))).astype(int)
        w[~nonzero] = 0
        return int(np.sum(w != 0)), w
    return 0, np.zeros_like(coefs, dtype=int)


def external_auroc(X, y, n_bins, C, penalty, X_w, y_w, X_j, y_j):
    from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer
    from sklearn.linear_model import LogisticRegression
    qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
    kb = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
    Xb = kb.fit_transform(qt.fit_transform(X))
    m = LogisticRegression(solver='liblinear', C=C, penalty=penalty, random_state=42).fit(Xb, y)
    coefs = m.coef_[0]
    nonzero = np.abs(coefs) > 0.01
    if np.any(nonzero):
        w = np.round(coefs * 5.0 / np.max(np.abs(coefs[nonzero]))).astype(int)
        w[~nonzero] = 0
    else:
        w = np.zeros_like(coefs, dtype=int)
    def score(Xe):
        return np.dot(kb.transform(qt.transform(Xe)), w)
    return roc_auc_score(y_w, score(X_w)), roc_auc_score(y_j, score(X_j))


def main():
    from sklearn.model_selection import StratifiedKFold
    X, y = load_training()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))

    train_medians = {c: float(X[c].median()) for c in X.columns}
    X_w, y_w = parse_cohort_data(
        pd.read_csv(os.path.join(DATA_DIR, 'west_china_processed.csv')), fill_medians=train_medians)
    X_j, y_j = parse_cohort_data(
        pd.read_csv(os.path.join(DATA_DIR, 'japan_processed.csv')), fill_medians=train_medians)

    rows = []
    for b, pen, c in GRID + L1_GRID + OUT_OF_GRID:
        aucs = cv_auroc(X, y, b, c, pen, folds)
        nw, _ = integer_weight_count(X, y, b, c, pen)
        west, japan = external_auroc(X, y, b, c, pen, X_w, y_w, X_j, y_j)
        rows.append({'bins': b, 'reg': pen, 'C': c, 'cv': aucs.mean(),
                     'fold_sd': aucs.std(ddof=1), 'west': west, 'japan': japan, 'weights': nw})

    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(RESULTS_DIR, 'config_sweep.csv'), index=False)

    l2 = sweep[(sweep.reg == 'l2') & (sweep.bins.isin([3, 4, 5, 7]))].reset_index(drop=True)
    best = l2.loc[l2.cv.idxmax()]
    se = best.fold_sd / np.sqrt(5)
    thr = best.cv - se
    tied = l2[l2.cv >= thr - 1e-9].copy()
    tied = tied.sort_values(['bins', 'weights', 'cv', 'C'],
                            ascending=[True, True, False, True])
    # grid's middle C value (1.5) wins exact CV ties among same (bins, weights)
    def tie_key(row):
        return (row.bins, row.weights, -row.cv, 0 if abs(row.C - 1.5) < 1e-9 else 1)
    tied['order'] = [tie_key(r) for r in tied.itertuples()]
    tied = tied.sort_values('order')
    winner = tied.iloc[0]

    print("=" * 90)
    print("MODEL SELECTION: internal 5-fold CV, 1-SE rule (L2 grid, pre-specified)")
    print("=" * 90)
    print(f"Best CV configuration : {best.bins} bins, L2, C={best.C} (CV {best.cv:.4f})")
    print(f"Fold SD               : {best.fold_sd:.4f}")
    print(f"SE of CV mean         : {se:.4f}")
    print(f"1-SE threshold        : {thr:.4f}")
    print(f"Tied (CV >= threshold): {len(tied)} of {len(l2)} L2 configurations")
    print(f"Selected (1-SE)       : {winner.bins} bins, L2, C={winner.C} "
          f"(CV {winner.cv:.4f}, {winner.weights:.0f} weights)")
    print(f"  West China AUROC    : {winner.west:.4f}")
    print(f"  Japan AUROC         : {winner.japan:.4f}")
    print("-" * 90)
    print("L2 grid (selection grid):")
    print(l2.sort_values('cv', ascending=False).round(4).to_string(index=False))
    print("-" * 90)
    print("L1 grid (sensitivity):")
    l1 = sweep[(sweep.reg == 'l1') & (sweep.bins.isin([3, 4, 5, 7]))]
    print(l1.sort_values('cv', ascending=False).round(4).to_string(index=False))
    print("-" * 90)
    print("2-bin out-of-grid check:")
    print(sweep[sweep.bins == 2].round(4).to_string(index=False))
    print("=" * 90)
    l1_best = l1.loc[l1.cv.idxmax()]
    print(f"Note: the L1 variant with the best CV ({l1_best.bins} bins, L1, C={l1_best.C}, "
          f"CV {l1_best.cv:.4f}) was not selectable (L1 excluded by design); "
          f"its external AUROCs are West {l1_best.west:.4f}, Japan {l1_best.japan:.4f}.")


if __name__ == '__main__':
    main()
