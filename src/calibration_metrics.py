"""
Calibration, precision-recall, and reclassification metrics.

1. Calibration on the two external cohorts for the scorecard (frozen
   training-fitted map and within-cohort recalibrated map), raw LR, ROMA,
   CPH-I, XGBoost:
     - calibration slope / intercept (logistic recalibration)
     - calibration-in-the-large (CITL)
     - Brier score and its Murphy decomposition (reliability, resolution,
       uncertainty) computed on 10 equal-width bins of predicted risk
     - ECI (estimated calibration index, mean |observed - predicted| via loess)
2. Precision-recall AUROC for all models on all cohorts (imbalance-aware).
3. NRI and IDI: scorecard vs ROMA / Raw LR / CPH-I at decision thresholds
   10% and 50% on West China, with bootstrap CIs.

Outputs:
    results/calibration_metrics.csv
    results/pr_auc.csv
    results/nri_idi.csv
    figures/fig_calibration.png
    figures/fig_pr_curves.png
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
from sklearn.metrics import (brier_score_loss, roc_auc_score, average_precision_score,
                             precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lib import (SEED, RESULTS_DIR, FIGURES_DIR, MODEL_COLORS, MODEL_LABELS,
                       auroc, bootstrap_pairwise)

RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR


def murphy_decomposition(y, p, n_bins=10):
    """Brier = REL - RES + UNC (Murphy 1973)."""
    edges = np.linspace(0, 1, n_bins + 1)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.digitize(p, edges) - 1
    rel = res = 0.0
    n = len(y)
    prev = float(np.mean(y))
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ob = float(np.mean(y[m]))
        pr = float(np.mean(p[m]))
        rel += (pr - ob) ** 2 * m.sum()
        res += (ob - prev) ** 2 * m.sum()
    rel /= n
    res /= n
    unc = prev * (1 - prev)
    return rel, res, unc


def loess_eci(y, p, frac=0.6):
    """Estimated calibration index via LOWESS-style local smoothing."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    try:
        sm = lowess(ys, ps, frac=frac, return_sorted=True)
        return float(np.mean(np.abs(sm[:, 1] - ps)))
    except Exception:
        return float('nan')


def nri_idi(y, p_base, p_new, thresholds=(0.10, 0.50)):
    """Category-free and category-based NRI plus IDI."""
    y = np.asarray(y)
    p_base = np.asarray(p_base)
    p_new = np.asarray(p_new)
    ev = y == 1
    up_ev = np.mean(p_new[ev] > p_base[ev]) - np.mean(p_new[ev] < p_base[ev])
    up_ne = np.mean(p_new[~ev] < p_base[~ev]) - np.mean(p_new[~ev] > p_base[~ev])
    nri_cf = up_ev + up_ne
    out = {'NRI_events': up_ev, 'NRI_nonevents': up_ne, 'NRI_catfree': nri_cf}
    for pt in thresholds:
        cat = lambda p: np.where(p >= pt, 1, np.where(p >= pt / 4, 0, -1))
        c0, c1 = cat(p_base), cat(p_new)
        up_ev_t = np.mean(c1[ev] > c0[ev]) - np.mean(c1[ev] < c0[ev])
        up_ne_t = np.mean(c1[~ev] < c0[~ev]) - np.mean(c1[~ev] > c0[~ev])
        out[f'NRI_{int(pt*100)}pct'] = up_ev_t + up_ne_t
    idi = float(np.mean(p_new[ev]) - np.mean(p_new[~ev])) - \
          float(np.mean(p_base[ev]) - np.mean(p_base[~ev]))
    out['IDI'] = idi
    return out


def main():
    import bench_lib as B
    np.random.seed(SEED)

    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(B.WEST_PATH, med)
    X_j, y_j = B.load_external(B.JAPAN_PATH, med)

    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)

    # score-to-probability maps (frozen and within-cohort)
    s_tr = zoo['Scorecard'](X_tr)
    s_w = zoo['Scorecard'](X_w)
    s_j = zoo['Scorecard'](X_j)
    frozen_map = LogisticRegression().fit(s_tr.reshape(-1, 1), y_tr)
    recal_w = LogisticRegression().fit(s_w.reshape(-1, 1), y_w)
    recal_j = LogisticRegression().fit(s_j.reshape(-1, 1), y_j)

    cohort_preds = {
        'West China': {
            'Scorecard (recalibrated)': recal_w.predict_proba(s_w.reshape(-1, 1))[:, 1],
            'Scorecard (frozen map)': frozen_map.predict_proba(s_w.reshape(-1, 1))[:, 1],
            'Raw LR': zoo['Raw LR'](X_w),
            'ROMA': zoo['ROMA'](X_w) / 100.0,
            'CPH-I': zoo['CPH-I'](X_w),
            'XGBoost': zoo['XGBoost'](X_w),
        },
        'Japan': {
            'Scorecard (recalibrated)': recal_j.predict_proba(s_j.reshape(-1, 1))[:, 1],
            'Scorecard (frozen map)': frozen_map.predict_proba(s_j.reshape(-1, 1))[:, 1],
            'Raw LR': zoo['Raw LR'](X_j),
            'ROMA': zoo['ROMA'](X_j) / 100.0,
            'CPH-I': zoo['CPH-I'](X_j),
            'XGBoost': zoo['XGBoost'](X_j),
        },
    }

    # 1. calibration metrics
    cal_rows = []
    for cname, preds in cohort_preds.items():
        yc = y_w if cname == 'West China' else y_j
        for mname, p in preds.items():
            p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
            brier = brier_score_loss(yc, p)
            rel, res, unc = murphy_decomposition(yc, p)
            # slope / intercept via logistic recalibration
            lm = LogisticRegression().fit(p.reshape(-1, 1), yc)
            slope = float(lm.coef_[0][0])
            intercept = float(lm.intercept_[0])
            citl = float(np.log(np.mean(p) / (1 - np.mean(p))) -
                         np.log(np.mean(yc) / (1 - np.mean(yc))))
            eci = loess_eci(yc, p)
            cal_rows.append({'Cohort': cname, 'Model': mname,
                             'Brier': round(brier, 4), 'REL': round(rel, 4),
                             'RES': round(res, 4), 'UNC': round(unc, 4),
                             'Cal slope': round(slope, 3),
                             'Cal intercept': round(intercept, 3),
                             'CITL': round(citl, 3), 'ECI': round(eci, 4)})
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(os.path.join(RESULTS, 'calibration_metrics.csv'), index=False)
    print("calibration_metrics.csv saved")
    print(cal_df.to_string(index=False))

    # 2. PR-AUC
    pr_rows = []
    for cname, (Xc, yc) in {'Training (in-sample)': (X_tr, y_tr),
                            'West China': (X_w, y_w), 'Japan': (X_j, y_j)}.items():
        for mname in ['Scorecard', 'Raw LR', 'ROMA', 'CPH-I', 'CA125 rule',
                      'XGBoost', 'CatBoost', 'Random Forest']:
            s = zoo[mname](Xc)
            pr_rows.append({'Cohort': cname, 'Model': mname,
                            'PR-AUC': round(float(average_precision_score(yc, s)), 4),
                            'AUROC': round(float(auroc(yc, s)), 4)})
    pr_df = pd.DataFrame(pr_rows)
    pr_df.to_csv(os.path.join(RESULTS, 'pr_auc.csv'), index=False)
    print("\npr_auc.csv saved")
    print(pr_df.pivot(index='Model', columns='Cohort', values='PR-AUC').round(4).to_string())

    # 3. NRI / IDI (West China)
    print("\nNRI / IDI (West China), scorecard (frozen map) vs comparators:")
    nri_rows = []
    p_sc = frozen_map.predict_proba(s_w.reshape(-1, 1))[:, 1]
    for mname, p_other in [('ROMA', zoo['ROMA'](X_w) / 100.0),
                           ('CPH-I', zoo['CPH-I'](X_w)),
                           ('Raw LR', zoo['Raw LR'](X_w)),
                           ('XGBoost', zoo['XGBoost'](X_w))]:
        res = nri_idi(y_w, p_other, p_sc)
        res['Comparator'] = mname
        # bootstrap CI for NRI and IDI
        rng = np.random.RandomState(SEED)
        n = len(y_w)
        nri_b, idi_b = [], []
        for _ in range(1000):
            idx = rng.choice(n, n, replace=True)
            rb = nri_idi(y_w[idx], p_other[idx], p_sc[idx])
            nri_b.append(rb['NRI_catfree'])
            idi_b.append(rb['IDI'])
        res['NRI 95% CI low'] = round(float(np.percentile(nri_b, 2.5)), 3)
        res['NRI 95% CI high'] = round(float(np.percentile(nri_b, 97.5)), 3)
        res['IDI 95% CI low'] = round(float(np.percentile(idi_b, 2.5)), 3)
        res['IDI 95% CI high'] = round(float(np.percentile(idi_b, 97.5)), 3)
        nri_rows.append(res)
    nri_df = pd.DataFrame(nri_rows)
    nri_df.to_csv(os.path.join(RESULTS, 'nri_idi.csv'), index=False)
    print(nri_df.round(4).to_string(index=False))

    # 4. figures: calibration curves + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cname in zip(axes, ['West China', 'Japan']):
        yc = y_w if cname == 'West China' else y_j
        for mname, p in cohort_preds[cname].items():
            if 'recalibrated' in mname or 'frozen' in mname:
                style = '-'
            else:
                style = '--'
            order = np.argsort(p)
            # loess smooth observed rate
            from statsmodels.nonparametric.smoothers_lowess import lowess
            try:
                sm = lowess(yc[order], p[order], frac=0.6, return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], style, lw=1.6,
                        label=mname, color='#2ca02c' if 'Scorecard' in mname else None)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], 'k:', lw=1)
        prev = float(np.mean(yc))
        ax.axhline(prev, color='grey', lw=1, ls='--', alpha=0.6)
        ax.text(0.02, prev + 0.03, f'prevalence {prev:.2f}', fontsize=8, color='grey')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed proportion')
        ax.set_title(f'{cname} (n = {len(yc)})', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, linestyle='--')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('Calibration curves (LOWESS-smoothed observed vs predicted risk)',
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(os.path.join(FIGURES, 'fig_calibration.png'), facecolor='white', dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    for mname in ['Scorecard', 'Raw LR', 'ROMA', 'CPH-I', 'XGBoost',
                  'CatBoost', 'Random Forest']:
        p, r, _ = precision_recall_curve(y_j, zoo[mname](X_j))
        ax.plot(r, p, lw=1.8, label=MODEL_LABELS.get(mname, mname),
                color=MODEL_COLORS.get(mname))
    ax.set_xlabel('Recall (sensitivity)')
    ax.set_ylabel('Precision (positive predictive value)')
    ax.set_title('Precision-recall curves, Japanese cohort (n = 177, 27 cancers)',
                 fontweight='bold')
    ax.axhline(27 / 177, color='grey', ls=':', lw=1)
    ax.text(0.02, 27 / 177 + 0.02, 'prevalence 0.15', fontsize=8, color='grey')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'fig_pr_curves.png'), facecolor='white', dpi=200)
    plt.close()
    print("\nfigures saved")


if __name__ == '__main__':
    main()
