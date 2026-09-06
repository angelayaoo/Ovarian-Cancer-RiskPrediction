"""
Repeated cross-validation, selection stability, and nested CV for tree tuning.

1. Repeated stratified 5-fold CV (N_REPEAT = 20 repeats, seeds 0..19):
   - scorecard (3 bins, L2, C = 1.5), argmax-CV control (4 bins, L2, C = 0.5),
     raw LR, and the three tree ensembles at package defaults.
   - Reports mean (+- SD) CV AUROC across repeats and per-repeat spread.

2. Selection stability: for every repeat, the 12-configuration L2 grid is
   evaluated by 5-fold CV and the 1-SE rule is applied; the table reports how
   often each configuration is selected and its mean CV rank.

3. Nested CV for tree hyperparameter tuning: outer 5-fold; inner 5-fold
   selection over a 9-configuration grid per family (3 depths x 3 estimators);
   the inner-selected configuration is refit and evaluated on the outer fold.
   Quantile transformer refit within each fold; no external data involved.

Outputs:
    results/repeated_cv.csv
    results/selection_stability.csv
    results/nested_cv_trees.csv
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer, StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lib import SEED, RESULTS_DIR, auroc, Scorecard

N_REPEAT = 20
GRID = [(b, c) for b in [3, 4, 5, 7] for c in [0.5, 1.5, 5.0]]

TREE_INNER_GRID = {
    'XGBoost': {'n_estimators': [50, 100, 300], 'max_depth': [3, 6, 9]},
    'CatBoost': {'iterations': [200, 500, 1000], 'depth': [4, 6, 8]},
    'Random Forest': {'n_estimators': [100, 300, 500], 'max_depth': [4, 6, None]},
}


def cv_scorecard_config(X, y, n_bins, C, folds, seed):
    """5-fold CV AUROC of the deployed (integer-rounded) scorecard."""
    aucs = []
    for tr, va in folds:
        qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform',
                                 random_state=seed)
        kb = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
        Xtr_b = kb.fit_transform(qt.fit_transform(X.iloc[tr]))
        Xva_b = kb.transform(qt.transform(X.iloc[va]))
        m = LogisticRegression(solver='liblinear', C=C, penalty='l2', random_state=seed)
        m.fit(Xtr_b, y[tr])
        coefs = m.coef_[0]
        nz = np.abs(coefs) > 0.01
        if np.any(nz):
            w = np.round(coefs * 5.0 / np.max(np.abs(coefs[nz]))).astype(int)
            w[~nz] = 0
        else:
            w = np.zeros_like(coefs, dtype=int)
        aucs.append(auroc(y[va], np.dot(Xva_b, w)))
    return np.array(aucs)


def fit_tree(name, X, y, **params):
    X = np.asarray(X)
    y = np.asarray(y)
    if name == 'XGBoost':
        m = XGBClassifier(random_state=SEED, eval_metric='logloss', **params)
    elif name == 'CatBoost':
        m = CatBoostClassifier(random_seed=SEED, verbose=0, **params)
    else:
        m = RandomForestClassifier(random_state=SEED, **params)
    m.fit(X, y)
    return m


def tree_score(m, X):
    return m.predict_proba(np.asarray(X))[:, 1]


def main():
    import bench_lib as B
    X, y = B.load_train_imputed()

    # ------------------------------------------------------------------
    # 1. Repeated stratified 5-fold CV
    # ------------------------------------------------------------------
    rows = []
    for repeat in range(N_REPEAT):
        folds = list(StratifiedKFold(n_splits=5, shuffle=True,
                                     random_state=100 + repeat).split(X, y))
        # scorecard (selected config) and argmax control
        for label, (bins, C) in [('Scorecard (3 bins)', (3, 1.5)),
                                 ('Argmax control (4 bins)', (4, 0.5))]:
            aucs = cv_scorecard_config(X, y, bins, C, folds, 42)
            rows.append({'repeat': repeat, 'model': label,
                         'cv_auroc_mean': float(np.mean(aucs)),
                         'cv_auroc_sd': float(np.std(aucs, ddof=1))})
        # raw LR
        aucs = []
        for tr, va in folds:
            sc = StandardScaler().fit(X.iloc[tr])
            m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            m.fit(sc.transform(X.iloc[tr]), y[tr])
            aucs.append(auroc(y[va], m.predict_proba(sc.transform(X.iloc[va]))[:, 1]))
        rows.append({'repeat': repeat, 'model': 'Raw LR',
                     'cv_auroc_mean': float(np.mean(aucs)),
                     'cv_auroc_sd': float(np.std(aucs, ddof=1))})
        # trees at package defaults
        for name in ['XGBoost', 'CatBoost', 'Random Forest']:
            aucs = []
            for tr, va in folds:
                qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform',
                                         random_state=42)
                qtr = qt.fit_transform(X.iloc[tr])
                qva = qt.transform(X.iloc[va])
                m = fit_tree(name, qtr, y[tr])
                aucs.append(auroc(y[va], tree_score(m, qva)))
            rows.append({'repeat': repeat, 'model': name + ' (defaults)',
                         'cv_auroc_mean': float(np.mean(aucs)),
                         'cv_auroc_sd': float(np.std(aucs, ddof=1))})
    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(os.path.join(RESULTS_DIR, 'repeated_cv.csv'), index=False)

    summ = rep_df.groupby('model')['cv_auroc_mean'].agg(
        ['mean', 'std', 'min', 'max']).round(4)
    print("Repeated 5-fold CV (20 repeats x 5 folds; mean +- between-repeat SD):")
    print(summ.sort_values('mean', ascending=False).to_string())

    # ------------------------------------------------------------------
    # 2. Selection stability of the 1-SE rule
    # ------------------------------------------------------------------
    print("\nSelection stability (1-SE rule over the 12-config L2 grid per repeat):")
    sel_counts = {}
    rank_rows = []
    for repeat in range(N_REPEAT):
        folds = list(StratifiedKFold(n_splits=5, shuffle=True,
                                     random_state=100 + repeat).split(X, y))
        grid_res = {}
        for b, c in GRID:
            aucs = cv_scorecard_config(X, y, b, c, folds, 42)
            grid_res[(b, c)] = (float(np.mean(aucs)), float(np.std(aucs, ddof=1)))
        best_key = max(grid_res, key=lambda k: grid_res[k][0])
        se = grid_res[best_key][1] / np.sqrt(5)
        thr = grid_res[best_key][0] - se
        band = {k: v for k, v in grid_res.items() if v[0] >= thr - 1e-9}
        # simplicity order: fewest bins, then highest CV, then middle C
        def key(k):
            b, c = k
            return (b, -band[k][0], 0 if abs(c - 1.5) < 1e-9 else 1)
        winner = min(band, key=key)
        sel_counts[winner] = sel_counts.get(winner, 0) + 1
        for k, v in grid_res.items():
            rank_rows.append({'repeat': repeat, 'bins': k[0], 'C': k[1],
                              'cv': v[0], 'cv_se': v[1] / np.sqrt(5)})
    sel_rows = [{'bins': k[0], 'C': k[1], 'times_selected': v,
                 'pct': round(100 * v / N_REPEAT, 1)} for k, v in sel_counts.items()]
    sel_df = pd.DataFrame(sel_rows).sort_values('times_selected', ascending=False)
    sel_df.to_csv(os.path.join(RESULTS_DIR, 'selection_stability.csv'), index=False)
    print(sel_df.to_string(index=False))
    rank_df = pd.DataFrame(rank_rows)
    rank_sum = rank_df.groupby(['bins', 'C'])['cv'].agg(['mean', 'std']).round(4)
    rank_sum = rank_sum.sort_values('mean', ascending=False)
    print("\nMean CV per configuration across repeats:")
    print(rank_sum.to_string())

    # ------------------------------------------------------------------
    # 3. Nested CV for tree hyperparameter tuning
    # ------------------------------------------------------------------
    print("\nNested CV for tree tuning (outer 5-fold, inner 5-fold over 9 configs):")
    outer = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
    nested_rows = []
    for name, grid in TREE_INNER_GRID.items():
        combos = []
        keys = list(grid.keys())
        for a in grid[keys[0]]:
            for b in grid[keys[1]]:
                combos.append((a, b))
        outer_aucs = []
        for otr, ova in outer:
            inner = list(StratifiedKFold(n_splits=5, shuffle=True,
                                         random_state=42).split(X.iloc[otr], y[otr]))
            best_combo, best_cv = None, -np.inf
            for combo in combos:
                params = {keys[0]: combo[0], keys[1]: combo[1]}
                inner_aucs = []
                for itr, iva in inner:
                    qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform',
                                             random_state=42)
                    qtr = qt.fit_transform(X.iloc[otr].iloc[itr])
                    qva = qt.transform(X.iloc[otr].iloc[iva])
                    m = fit_tree(name, qtr, y[otr][itr], **params)
                    inner_aucs.append(auroc(y[otr][iva], tree_score(m, qva)))
                cv = float(np.mean(inner_aucs))
                if cv > best_cv:
                    best_cv, best_combo = cv, combo
            qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform',
                                     random_state=42)
            qtr = qt.fit_transform(X.iloc[otr])
            qva = qt.transform(X.iloc[ova])
            m = fit_tree(name, qtr, y[otr], **{keys[0]: best_combo[0], keys[1]: best_combo[1]})
            outer_aucs.append(auroc(y[ova], tree_score(m, qva)))
        nested_rows.append({'family': name, 'nested_cv_auroc': round(float(np.mean(outer_aucs)), 4),
                            'sd': round(float(np.std(outer_aucs, ddof=1)), 4)})
    nested_df = pd.DataFrame(nested_rows)
    nested_df.to_csv(os.path.join(RESULTS_DIR, 'nested_cv_trees.csv'), index=False)
    print(nested_df.to_string(index=False))


if __name__ == '__main__':
    main()
