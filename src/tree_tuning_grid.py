"""
Supplementary tree-hyperparameter tuning grid (sensitivity analysis).

Question asked: would tuning the tree ensembles by internal CV change the
conclusions? For each family we sweep 27 configurations, report the best
tuned internal CV, and evaluate the best-tuned models frozen on the
external cohorts. The external cohorts never enter selection.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lib import load_train_imputed as load_training
from bench_lib import parse_cohort_data

DATA_DIR = 'data/processed'

GRIDS = {
    'XGBoost': [dict(n_estimators=n, max_depth=d, learning_rate=l)
                for n in [100, 300, 500] for d in [3, 6, 10] for l in [0.03, 0.1, 0.3]],
    'CatBoost': [dict(iterations=n, depth=d, learning_rate=l)
                 for n in [500, 1000, 1500] for d in [4, 6, 10] for l in [0.03, 0.1, 0.3]],
    'RandomForest': [dict(n_estimators=n, max_depth=d, min_samples_leaf=m)
                     for n in [100, 500, 1000] for d in [None, 6, 10] for m in [1, 2, 4]],
}


def make_model(name, cfg):
    if name == 'XGBoost':
        return XGBClassifier(random_state=42, eval_metric='logloss', **cfg)
    if name == 'CatBoost':
        return CatBoostClassifier(random_seed=42, verbose=0, **cfg)
    return RandomForestClassifier(random_state=42, n_jobs=-1, **cfg)


def main():
    X, y = load_training()
    medians = {c: float(X[c].median()) for c in X.columns}
    X_w, y_w = parse_cohort_data(pd.read_csv(os.path.join(DATA_DIR, 'west_china_processed.csv')),
                                 fill_medians=medians)
    X_j, y_j = parse_cohort_data(pd.read_csv(os.path.join(DATA_DIR, 'japan_processed.csv')),
                                 fill_medians=medians)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))

    records = []
    for name, grid in GRIDS.items():
        print("=" * 70)
        print(f"{name}: {len(grid)} configurations, 5-fold internal CV")
        print("=" * 70)
        for cfg in grid:
            cv = []
            for tr, va in folds:
                qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
                qtr = qt.fit_transform(X.iloc[tr]); qva = qt.transform(X.iloc[va])
                m = make_model(name, cfg)
                m.fit(qtr, y[tr])
                cv.append(roc_auc_score(y[va], m.predict_proba(qva)[:, 1]))
            records.append(dict(model=name, cv=np.mean(cv), west=np.nan, japan=np.nan, **cfg))
        best = sorted(records[-len(grid):], key=lambda r: -r['cv'])[0]
        qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42)
        qtr = qt.fit_transform(X); qw = qt.transform(X_w); qj = qt.transform(X_j)
        m = make_model(name, {k: best[k] for k in grid[0]})
        m.fit(qtr, y)
        best['west'] = roc_auc_score(y_w, m.predict_proba(qw)[:, 1])
        best['japan'] = roc_auc_score(y_j, m.predict_proba(qj)[:, 1])
        print(f"  best tuned CV {best['cv']:.4f} with { {k: best[k] for k in grid[0]} }")
        print(f"    -> West China {best['west']:.4f}, Japan {best['japan']:.4f}")

    out = pd.DataFrame(records)
    out.to_csv(os.path.join('results', 'tree_tuning_grid.csv'), index=False)
    print("\nSaved results/tree_tuning_grid.csv")


if __name__ == '__main__':
    main()
