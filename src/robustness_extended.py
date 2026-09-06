"""
Extended robustness experiments.

1. Noise-type grid on West China: multiplicative, additive, and combined
   Gaussian noise at 10% and 30% (100 perturbations), plus gross-error
   outlier contamination (5% of biomarker values multiplied by 3 or divided
   by 3) — models: scorecard, raw LR, ROMA, CPH-I, XGBoost, CatBoost, RF.
2. Bin-boundary sensitivity: the scorecard's frozen tertile cutpoints are
   shifted by -10%/-5%/+5%/+10% of the training range for CA125 and HE4
   before scoring the external cohorts.
3. External learning curves: scorecard, raw LR and the trees are retrained on
   stratified subsamples (n = 60..349) of the training cohort and evaluated
   frozen on West China (10 draws per size); SD across draws.

Outputs:
    results/robustness_extended.csv
    results/bin_boundary_sensitivity.csv
    results/external_learning_curves.csv
    figures/fig_noise_types.png
    figures/fig_external_learning_curves.png
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lib import (SEED, RESULTS_DIR, FIGURES_DIR, MODEL_COLORS, MODEL_LABELS,
                       auroc, Scorecard, RawLR)

RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR
N_PERT = 100


def make_noisy(X, kind, level, seed=0):
    """kind: 'multiplicative', 'additive', 'combined', 'outliers'."""
    Xn = X.copy()
    rng = np.random.RandomState(seed)
    for f in ['CA125', 'HE4']:
        v = X[f].values.astype(float)
        if kind == 'multiplicative':
            Xn[f] = np.maximum(v * rng.normal(1.0, level, len(v)), 1e-5)
        elif kind == 'additive':
            sd = level * np.nanmedian(v)
            Xn[f] = np.maximum(v + rng.normal(0, sd, len(v)), 1e-5)
        elif kind == 'combined':
            Xn[f] = np.maximum(v * rng.normal(1.0, level, len(v))
                               + rng.normal(0, level * np.nanmedian(v), len(v)), 1e-5)
        elif kind == 'outliers':
            mask = rng.random(len(v)) < level
            factor = np.where(rng.random(len(v)) < 0.5, 3.0, 1.0 / 3.0)
            Xn[f] = np.where(mask, np.maximum(v * factor, 1e-5), v)
    return Xn


def fit_trees(Xn_tr, y_tr, seed=SEED):
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    xgb = XGBClassifier(random_state=seed, eval_metric='logloss').fit(Xn_tr, y_tr)
    cat = CatBoostClassifier(random_seed=seed, verbose=0).fit(Xn_tr, y_tr)
    rf = RandomForestClassifier(random_state=seed).fit(Xn_tr, y_tr)
    return xgb, cat, rf


def main():
    import bench_lib as B
    np.random.seed(SEED)

    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(B.WEST_PATH, med)
    X_j, y_j = B.load_external(B.JAPAN_PATH, med)

    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)
    models = ['Scorecard', 'Raw LR', 'ROMA', 'CPH-I', 'XGBoost', 'CatBoost',
              'Random Forest']

    # 1. noise-type grid (West China)
    rows = []
    scenarios = [('multiplicative', 0.10), ('multiplicative', 0.30),
                 ('additive', 0.10), ('additive', 0.30),
                 ('combined', 0.10), ('combined', 0.30),
                 ('outliers', 0.05)]
    for kind, level in scenarios:
        aucs = {m: [] for m in models}
        for rep in range(N_PERT):
            Xn = make_noisy(X_w, kind, level, seed=rep)
            for m in models:
                aucs[m].append(auroc(y_w, zoo[m](Xn)))
        rows.append({'Cohort': 'West China', 'Noise type': kind,
                     'Level': level,
                     **{m: round(float(np.mean(aucs[m])), 4) for m in models},
                     **{m + ' SD': round(float(np.std(aucs[m])), 4) for m in models}})
    rob_df = pd.DataFrame(rows)
    rob_df.to_csv(os.path.join(RESULTS, 'robustness_extended.csv'), index=False)
    print("robustness_extended.csv saved")
    print(rob_df.to_string(index=False))

    # degradation vs 0% baseline (from the main benchmark: West China 0% noise)
    base = {m: 0.0 for m in models}
    for cname, Xc, yc in [('West China', X_w, y_w), ('Japan', X_j, y_j)]:
        for m in models:
            base[(cname, m)] = auroc(yc, zoo[m](Xc))

    # 2. bin-boundary sensitivity (perturb frozen tertile cutpoints)
    print("\nBin-boundary sensitivity (shift CA125/HE4 cutpoints, West China):")
    from sklearn.preprocessing import KBinsDiscretizer
    bb_rows = []
    base_edges = [e.copy() for e in sc.discretizer.bin_edges_]
    for shift in [-0.10, -0.05, 0.00, 0.05, 0.10]:
        Xw_s = sc.scaler.transform(X_w)
        kb = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='quantile')
        kb.fit(sc.scaler.transform(X_tr))  # dummy fit to get structure
        kb.bin_edges_ = base_edges
        new_edges = [e.copy() for e in base_edges]
        for j in (0, 1):  # CA125, HE4
            e = base_edges[j]
            if len(e) >= 3:
                span = e[-1] - e[0]
                e2 = e.copy()
                e2[1:-1] = e[1:-1] + shift * span
                new_edges[j] = e2
        kb.bin_edges_ = new_edges
        Xb = kb.transform(Xw_s)
        score_shifted = np.dot(Xb, sc.weights)
        bb_rows.append({'Cutpoint shift': shift,
                        'AUROC (scorecard)': round(float(auroc(y_w, score_shifted)), 4)})
    bb_df = pd.DataFrame(bb_rows)
    bb_df.to_csv(os.path.join(RESULTS, 'bin_boundary_sensitivity.csv'), index=False)
    print(bb_df.to_string(index=False))

    # 3. external learning curves (train subsample, evaluate frozen on West China)
    print("\nExternal learning curves (train subsample -> West China):")
    sizes = [60, 90, 120, 180, 240, 349]
    N_DRAWS = 10
    lc_rows = []
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    for n in sizes:
        for mname in ['Scorecard', 'Raw LR', 'XGBoost', 'CatBoost', 'Random Forest']:
            ext_aucs = []
            for draw in range(N_DRAWS):
                rng = np.random.RandomState(SEED + draw)
                idx_pos = rng.choice(np.where(y_tr == 1)[0],
                                     size=max(3, int(round(n * (y_tr == 1).mean()))),
                                     replace=False)
                idx_neg = rng.choice(np.where(y_tr == 0)[0],
                                     size=n - len(idx_pos), replace=False)
                idx = np.concatenate([idx_pos, idx_neg])
                Xs, ys = X_tr.iloc[idx], y_tr[idx]
                if mname == 'Scorecard':
                    mdl = Scorecard(n_bins=3).fit(Xs, ys)
                    ext_aucs.append(auroc(y_w, mdl.predict_score(X_w)))
                elif mname == 'Raw LR':
                    mdl = RawLR().fit(Xs, ys)
                    ext_aucs.append(auroc(y_w, mdl.predict_proba(X_w)))
                else:
                    from sklearn.preprocessing import QuantileTransformer
                    q = QuantileTransformer(n_quantiles=min(100, len(Xs)),
                                            output_distribution='uniform',
                                            random_state=SEED)
                    Xs_n = q.fit_transform(Xs)
                    Xw_n = pd.DataFrame(q.transform(X_w), columns=X_w.columns)
                    if mname == 'XGBoost':
                        mdl = XGBClassifier(random_state=SEED, eval_metric='logloss')
                    elif mname == 'CatBoost':
                        mdl = CatBoostClassifier(random_seed=SEED, verbose=0)
                    else:
                        mdl = RandomForestClassifier(random_state=SEED)
                    mdl.fit(Xs_n, ys)
                    ext_aucs.append(auroc(y_w, mdl.predict_proba(Xw_n)[:, 1]))
            lc_rows.append({'n': n, 'Model': mname,
                            'External AUROC mean': round(float(np.mean(ext_aucs)), 4),
                            'External AUROC SD': round(float(np.std(ext_aucs)), 4)})
    lc_df = pd.DataFrame(lc_rows)
    lc_df.to_csv(os.path.join(RESULTS, 'external_learning_curves.csv'), index=False)
    print(lc_df.pivot(index='n', columns='Model', values='External AUROC mean').to_string())

    # figures
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ['Mult. 10%', 'Mult. 30%', 'Add. 10%', 'Add. 30%',
              'Comb. 10%', 'Comb. 30%', 'Outliers 5%']
    for m in models:
        vals = rob_df[m].values
        ax.plot(labels, vals, marker='o', label=MODEL_LABELS.get(m, m),
                color=MODEL_COLORS.get(m))
    ax.set_ylabel('AUROC (West China)')
    ax.set_title('Robustness across noise types and gross errors (West China, n = 380)',
                 fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'fig_noise_types.png'), facecolor='white', dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ['Scorecard', 'Raw LR', 'XGBoost', 'CatBoost', 'Random Forest']:
        sub = lc_df[lc_df['Model'] == m]
        ax.errorbar(sub['n'], sub['External AUROC mean'],
                    yerr=sub['External AUROC SD'], marker='o', capsize=2,
                    color=MODEL_COLORS[m], label=MODEL_LABELS.get(m, m))
    ax.axhline(0.886, color=MODEL_COLORS['ROMA'], ls='--', lw=1.2)
    ax.text(70, 0.889, 'ROMA (West China AUROC)', fontsize=8, color=MODEL_COLORS['ROMA'])
    ax.set_xlabel('Training samples')
    ax.set_ylabel('External AUROC on West China (mean \u00b1 SD, 10 draws)')
    ax.set_title('Data efficiency: external performance vs training size', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'fig_external_learning_curves.png'),
                facecolor='white', dpi=200)
    plt.close()
    print("\nfigures saved")


if __name__ == '__main__':
    main()
