"""
Meta-analytic pooling of external validation results.

Random-effects (DerSimonian-Laird) pooling of AUROC differences between the
scorecard and each comparator across the two external cohorts (the imputed
West China cohort and the Japanese stress-test cohort), plus pooling of each
model's own AUROC. Bootstrap-derived SEs from 2,000 patient-level resamples
per cohort. I-squared reported for heterogeneity. Tree ensembles use the
Setting B (internally CV-tuned) configurations, matching the primary
comparison in the paper.

Outputs:
    results/meta_analysis.csv
    figures/fig_meta_analysis.png
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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lib import (SEED, RESULTS_DIR, FIGURES_DIR, MODEL_COLORS, MODEL_LABELS,
                       bootstrap_pairwise, bootstrap_auroc)

RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR


def der_simonian_laird(effects, ses):
    """Random-effects meta-analysis (DerSimonian-Laird)."""
    effects = np.asarray(effects, dtype=float)
    ses = np.asarray(ses, dtype=float)
    k = len(effects)
    if k == 1:
        return effects[0], ses[0], 0.0, np.nan
    w_fixed = 1.0 / ses ** 2
    q = float(np.sum(w_fixed * (effects - np.sum(w_fixed * effects) / np.sum(w_fixed)) ** 2))
    df = k - 1
    c = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0.0, (q - df) / c)
    w_rand = 1.0 / (ses ** 2 + tau2)
    pooled = float(np.sum(w_rand * effects) / np.sum(w_rand))
    se_pooled = float(np.sqrt(1.0 / np.sum(w_rand)))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return pooled, se_pooled, tau2, float(i2)


def main():
    import bench_lib as B
    np.random.seed(SEED)

    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(B.WEST_PATH, med)
    X_j, y_j = B.load_external(B.JAPAN_PATH, med)

    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)
    tuned = {
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3,
                                 learning_rate=0.03, random_state=SEED,
                                 eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(iterations=500, depth=4,
                                       learning_rate=0.03, random_seed=SEED,
                                       verbose=0),
        'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=6,
                                                min_samples_leaf=2,
                                                random_state=SEED),
    }
    for mdl in tuned.values():
        mdl.fit(X_tr, y_tr)

    def predict(name, X):
        if name in tuned:
            return tuned[name].predict_proba(X)[:, 1]
        return zoo[name](X)

    models = ['Scorecard', 'Raw LR', 'ROMA', 'CPH-I', 'CA125 rule',
              'XGBoost', 'CatBoost', 'Random Forest']
    cohorts = [('West China', X_w, y_w), ('Japan', X_j, y_j)]

    # per-cohort bootstrap AUROC (SE) for each model
    auroc_se = {}
    for cname, Xc, yc in cohorts:
        for m in models:
            boot = bootstrap_auroc(yc, predict(m, Xc), n_boot=2000)
            auroc_se[(cname, m)] = (float(np.mean(boot)), float(np.std(boot, ddof=1)))

    rows = []
    # 1. pool each model's AUROC across cohorts
    for m in models:
        eff = [auroc_se[(c, m)][0] for c, _, _ in cohorts]
        ses = [auroc_se[(c, m)][1] for c, _, _ in cohorts]
        pooled, se, tau2, i2 = der_simonian_laird(eff, ses)
        rows.append({'Quantity': 'AUROC', 'Model': m,
                     'West China': round(eff[0], 4), 'Japan': round(eff[1], 4),
                     'Pooled (RE)': round(pooled, 4), 'SE': round(se, 4),
                     '95% CI low': round(pooled - 1.96 * se, 4),
                     '95% CI high': round(pooled + 1.96 * se, 4),
                     'I2': round(i2, 2)})

    # 2. pool scorecard-minus-comparator differences
    print("Pooled scorecard-minus-comparator AUROC differences (random effects):")
    diff_rows = []
    for m in models:
        if m == 'Scorecard':
            continue
        eff, ses = [], []
        for cname, Xc, yc in cohorts:
            diffs = bootstrap_pairwise(yc, zoo['Scorecard'](Xc), predict(m, Xc),
                                       n_boot=2000)
            eff.append(float(np.mean(diffs)))
            ses.append(float(np.std(diffs, ddof=1)))
        pooled, se, tau2, i2 = der_simonian_laird(eff, ses)
        z = pooled / se
        p = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(z)))
        diff_rows.append({'Comparator': m,
                          'dAUROC West China': round(eff[0], 4),
                          'dAUROC Japan': round(eff[1], 4),
                          'Pooled dAUROC (RE)': round(pooled, 4),
                          'SE': round(se, 4),
                          '95% CI low': round(pooled - 1.96 * se, 4),
                          '95% CI high': round(pooled + 1.96 * se, 4),
                          'z': round(z, 3), 'p': round(float(p), 4),
                          'tau2': round(tau2, 5), 'I2': round(i2, 2)})
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(RESULTS, 'meta_analysis.csv'), index=False)
    print(diff_df.to_string(index=False))

    # forest-style figure
    fig, ax = plt.subplots(figsize=(8, 5))
    d = diff_df.sort_values('Pooled dAUROC (RE)')
    ypos = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.errorbar(r['Pooled dAUROC (RE)'], i,
                    xerr=[[r['Pooled dAUROC (RE)'] - r['95% CI low']],
                          [r['95% CI high'] - r['Pooled dAUROC (RE)']]],
                    fmt='o', color=MODEL_COLORS.get(r['Comparator'], 'k'),
                    capsize=3)
        ax.errorbar(r['dAUROC West China'], i - 0.15, xerr=0, fmt='s', ms=4,
                    color='grey', alpha=0.8)
        ax.errorbar(r['dAUROC Japan'], i + 0.15, xerr=0, fmt='^', ms=4,
                    color='grey', alpha=0.8)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(d['Comparator'])
    ax.set_xlabel('AUROC difference (scorecard minus comparator)')
    ax.set_title('Meta-analysis of the two external cohorts (random effects;\n'
                 'Setting B tuned ensembles)', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', axis='x')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='k', ls='', label='Pooled (RE), 95% CI'),
        Line2D([0], [0], marker='s', color='grey', ls='', label='West China'),
        Line2D([0], [0], marker='^', color='grey', ls='', label='Japan')],
        fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'fig_meta_analysis.png'), facecolor='white', dpi=200)
    plt.close()
    print("fig_meta_analysis.png saved")


if __name__ == '__main__':
    main()
