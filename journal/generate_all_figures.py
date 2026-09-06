"""
Consolidated publication figures (v2, clean rendering).

Regenerates every figure used in the manuscript with a consistent style:
  - horizontal sorted bars for multi-model comparisons (no rotated labels)
  - explicit headroom so value labels and CIs never clip
  - legends placed outside the plot area or in clear corners
  - consistent per-model colors and short axis labels
  - savefig with bbox_inches='tight' + padding

Figures (paths must match build_docx.py FIGURES dict):
  fig1_main_comparison.png        Figure 1  AUROC by model and cohort
  fig_meta_analysis.png           Figure 2  random-effects forest plot
  fig3_complexity_ladder.png      Figure 3  complexity-performance ladder
  fig_calibration.png             Figure 4  calibration curves
  fig_decision_analysis.png       Figure 5  DCA with bootstrap bands
  fig2_noise_degradation.png      Figure 6  noise degradation
  fig_external_learning_curves.png Figure 7 external learning curves
  fig_simulation_study.png        Figure 8  regime map
  fig7_perfeature_drift.png       Figure S1 per-feature drift
  fig8_shift_vs_degradation.png   Figure S2 shift vs transfer loss
  fig9_learning_curves.png        Figure S3 internal learning curves
  fig_noise_types.png             Figure S4 noise-type robustness
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
# Journal build: results/data live at the repo root; figures are written to
# the journal/figures folder alongside the manuscript.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(ROOT)

from bench_lib import (SEED, RESULTS_DIR, FIGURES_DIR, MODEL_COLORS, MODEL_LABELS,
                       bootstrap_auroc, auroc)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 110,
})

R = RESULTS_DIR
F = os.path.join('journal', 'figures')
os.makedirs(F, exist_ok=True)
os.makedirs(F, exist_ok=True)

MODELS = ['Scorecard', 'Raw LR', 'ROMA', 'CPH-I', 'CA125 rule',
          'XGBoost', 'CatBoost', 'Random Forest']
SHORT = {'Scorecard': 'Integer scorecard', 'Raw LR': 'Logistic regression',
         'ROMA': 'ROMA', 'CPH-I': 'CPH-I', 'CA125 rule': 'CA125 rule',
         'XGBoost': 'XGBoost', 'CatBoost': 'CatBoost',
         'Random Forest': 'Random forest'}


def save(fig, name):
    fig.savefig(os.path.join(F, name), bbox_inches='tight', pad_inches=0.25,
                facecolor='white', dpi=300)
    plt.close(fig)
    print('saved', name)


def hbars(ax, names, values, colors, ci=None, x0=0.0):
    """Horizontal bars sorted by value; ci = list of (lo, hi) or None."""
    order = np.argsort(values)
    ys = np.arange(len(names))
    ax.barh(ys, [values[i] - x0 for i in order], left=x0, height=0.62,
            color=[colors[i] for i in order], zorder=3)
    for k, i in enumerate(order):
        v = values[i]
        pad = 0.006 * max(1.0, abs(v) * 12)
        ha = 'left' if v >= x0 else 'right'
        ax.text(v + pad if v >= x0 else v - pad, k, f'{v:.2f}', va='center',
                ha=ha, fontsize=9, zorder=4)
        if ci is not None and ci[i] is not None:
            lo, hi = ci[i]
            ax.plot([lo, hi], [k, k], color='black', lw=1.2, zorder=5)
            ax.plot([lo, lo], [k - 0.12, k + 0.12], color='black', lw=1.2, zorder=5)
            ax.plot([hi, hi], [k - 0.12, k + 0.12], color='black', lw=1.2, zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[i] for i in order])
    ax.tick_params(axis='y', length=0)
    return order


def fig_consequence_curves(X_tr, y_tr, X_w, y_w, zoo):
    from sklearn.linear_model import LogisticRegression
    s_w = zoo['Scorecard'](X_w)
    recal = LogisticRegression().fit(s_w.reshape(-1, 1), y_w)
    probs = {
        'Scorecard (recalibrated)': recal.predict_proba(s_w.reshape(-1, 1))[:, 1],
        'Logistic regression': zoo['Raw LR'](X_w),
        'ROMA': zoo['ROMA'](X_w) / 100.0,
        'CPH-I': zoo['CPH-I'](X_w),
    }
    colors = {'Scorecard (recalibrated)': MODEL_COLORS['Scorecard'],
              'Logistic regression': MODEL_COLORS['Raw LR'],
              'ROMA': MODEL_COLORS['ROMA'], 'CPH-I': MODEL_COLORS['CPH-I']}

    def cons(y, p, pt):
        pred = np.asarray(p) >= pt
        y = np.asarray(y)
        return (float(np.sum((pred == 0) & (y == 1)) / len(y) * 100),
                float(np.sum((pred == 1) & (y == 0)) / len(y) * 100))

    thresholds = np.linspace(0.01, 1.00, 240)
    prev100 = float(np.mean(y_w)) * 100
    marks = {'Scorecard (recalibrated)': ((12.4, 1.3), '*', 'Tier \u2265 +1'),
             'Logistic regression': (None, None, None),
             'ROMA': ((21.6, 1.6), 'D', 'Published cutpoints'),
             'CPH-I': ((12.6, 2.6), 's', '\u2265 7% cutpoint')}
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.6))
    for ax, mname in zip(axes.ravel(), probs.keys()):
        p = probs[mname]
        pts = np.array([cons(y_w, p, pt) for pt in thresholds])
        ax.plot(pts[:, 0], pts[:, 1], ls='-', lw=2.2, color=colors[mname],
                zorder=3)
        ax.scatter([prev100], [0], s=45, marker='o', color='#888888', zorder=2)
        ax.scatter([0], [100 - prev100], s=45, marker='s', color='#888888',
                   zorder=2)
        ax.annotate('treat none', (prev100, 0), xytext=(prev100 - 13, 1.5),
                    fontsize=7, color='#555555',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
        ax.annotate('treat all', (0, 100 - prev100), xytext=(1.5, 100 - prev100 - 5),
                    fontsize=7, color='#555555',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
        pt0, marker, label = marks[mname]
        if pt0 is not None:
            ax.scatter([pt0[0]], [pt0[1]], s=70, marker=marker,
                       color=colors[mname], edgecolor='black', zorder=5)
            ax.annotate(label, pt0, xytext=(pt0[0] + 2.5, pt0[1] + 6.5),
                        fontsize=7.5, color='#333333',
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
        ax.set_xlim(-3, prev100 + 4)
        ax.set_ylim(-3, 100 - prev100 + 4)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_title(mname, fontsize=9.5, fontweight='bold')
        ax.grid(alpha=0.25, ls='--', zorder=0)
        if ax in axes[:, 0]:
            ax.set_ylabel('Unnecessary referrals per 100 women', fontsize=8)
        if ax in axes[1, :]:
            ax.set_xlabel('Missed cancers per 100 women', fontsize=8)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, 'fig6_consequence_curves.png')


def main():
    import bench_lib as B
    np.random.seed(SEED)

    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(B.WEST_PATH, med)
    X_j, y_j = B.load_external(B.JAPAN_PATH, med)
    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)

    # ---------------------------------------------------------------- data
    # external bootstrap means + CIs
    ext = {}
    for cname, Xc, yc in [('West China', X_w, y_w), ('Japan', X_j, y_j)]:
        for m in MODELS:
            boot = bootstrap_auroc(yc, zoo[m](Xc), n_boot=2000)
            ext[(cname, m)] = (float(np.mean(boot)),
                               float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5)))
    # training: repeated CV means for fitted models; applied values for formulas
    rep = pd.read_csv(os.path.join(R, 'repeated_cv.csv'))
    cv = {row['model']: float(row['cv_auroc_mean']) for _, row in
          rep[rep['repeat'] == 0].iterrows()}
    train_val = {
        'Scorecard': cv['Scorecard (3 bins)'], 'Raw LR': cv['Raw LR'],
        'XGBoost': cv['XGBoost (defaults)'], 'CatBoost': cv['CatBoost (defaults)'],
        'Random Forest': cv['Random Forest (defaults)'],
    }
    cl = pd.read_csv(os.path.join(R, 'clinical_comparators_noise.csv'))
    tr0 = cl[cl['Cohort'] == 'Training (in-sample)'].iloc[0]
    for m in ['ROMA', 'CPH-I', 'CA125 rule']:
        train_val[m] = float(tr0[m])

    # ================================================================ Figure 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, cname in zip(axes, ['Training', 'West China', 'Japan']):
        if cname == 'Training':
            vals = np.array([train_val[m] for m in MODELS])
            ci = [None] * len(MODELS)
        else:
            vals = np.array([ext[(cname, m)][0] for m in MODELS])
            ci = [ext[(cname, m)][1:] for m in MODELS]
        hbars(ax, [SHORT[m] for m in MODELS], vals, [MODEL_COLORS[m] for m in MODELS], ci,
              x0=0.5)
        ax.axvline(0.5, color='grey', ls='--', lw=0.9, zorder=1)
        lo = min([c[0] for c in ci if c is not None] + [vals.min()])
        hi = max([c[1] for c in ci if c is not None] + [vals.max()])
        ax.set_xlim(max(0.5, lo - 0.02), min(1.0, hi + 0.03))
        ax.set_xlabel('AUROC')
        if cname == 'Training':
            ax.set_title('Training cohort\n(five-fold CV; \u2020 applied)', fontweight='bold')
        elif cname == 'West China':
            ax.set_title('West China\n(external, n = 380)', fontweight='bold')
        else:
            ax.set_title('Japan\n(external, n = 177)', fontweight='bold')
        ax.grid(axis='x', alpha=0.25, ls='--', zorder=0)
    fig.tight_layout(w_pad=2.4)
    save(fig, 'fig1_main_comparison.png')

    # ================================================================ Figure 2
    meta = pd.read_csv(os.path.join(R, 'meta_analysis.csv'))
    meta['label'] = meta['Comparator'].map(SHORT)
    order_labels = ['ROMA', 'CPH-I', 'CA125 rule', 'Raw LR', 'XGBoost', 'CatBoost',
                    'Random Forest']
    meta = meta.set_index('Comparator').loc[order_labels].reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ys = np.arange(len(meta))
    ax.axvline(0, color='black', lw=1.1, zorder=2)
    ax.axvspan(-0.02, 0.02, color='grey', alpha=0.08, zorder=0)
    for k, row in meta.iterrows():
        cmp = row['Comparator']
        pooled = float(row['Pooled dAUROC (RE)'])
        lo, hi = float(row['95% CI low']), float(row['95% CI high'])
        ax.plot([lo, hi], [k, k], lw=2.2, color=MODEL_COLORS[cmp], zorder=4)
        ax.scatter([pooled], [k], s=70, color=MODEL_COLORS[cmp], zorder=5,
                   edgecolor='black', lw=0.8)
        ax.scatter([row['dAUROC West China']], [k - 0.19], s=34, marker='s',
                   color='#555555', zorder=4)
        ax.scatter([row['dAUROC Japan']], [k + 0.19], s=34, marker='^',
                   color='#555555', zorder=4)
        ax.text(0.412, k, f"p = {row['p']:.3f}", va='center', fontsize=8.5,
                color='#333333')
    ax.set_yticks(ys)
    ax.set_yticklabels(meta['label'])
    ax.set_xlim(-0.17, 0.47)
    ax.set_xlabel('\u0394 AUROC (scorecard minus comparator)')
    ax.set_ylim(-0.55, len(meta) - 0.45)
    ax.set_title('Random-effects pooling across the two external cohorts',
                 fontweight='bold')
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#777777',
               markeredgecolor='black', markersize=8, ls='', label='Pooled estimate (95% CI)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#555555', markersize=7,
               ls='', label='West China (n = 380)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#555555', markersize=7,
               ls='', label='Japan (n = 177)'),
    ]
    ax.legend(handles=handles, loc='lower right', frameon=True, framealpha=0.95)
    ax.grid(axis='x', alpha=0.25, ls='--', zorder=1)
    fig.tight_layout()
    save(fig, 'fig_meta_analysis.png')

    # ================================================================ Figure 3
    order = ['CA125\nrule', 'ROMA', 'CPH-I', 'Logistic\nregr.', 'Scorecard',
             'Random\nforest', 'XGBoost', 'CatBoost']
    keys = ['CA125 rule', 'ROMA', 'CPH-I', 'Raw LR', 'Scorecard',
            'Random Forest', 'XGBoost', 'CatBoost']
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = np.arange(len(order))
    for x, m in zip(xs, keys):
        v = ext[('West China', m)][0]
        ax.scatter(x, v, s=150, color=MODEL_COLORS[m], zorder=4,
                   edgecolor='black', lw=0.8)
        ax.text(x, v + 0.012, f'{v:.2f}', ha='center', va='bottom',
                fontsize=10, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylim(0.60, 1.00)
    ax.set_ylabel('External AUROC (West China)')
    ax.set_title('Complexity-performance ladder', fontweight='bold')
    ax.set_xlabel('Model (left to right: increasing complexity)')
    ax.grid(axis='y', alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    save(fig, 'fig3_complexity_ladder.png')
    fig_consequence_curves(X_tr, y_tr, X_w, y_w, zoo)

    # ================================================================ Figure 4
    s_tr = zoo['Scorecard'](X_tr)
    s_w = zoo['Scorecard'](X_w)
    s_j = zoo['Scorecard'](X_j)
    from sklearn.linear_model import LogisticRegression
    frozen_map = LogisticRegression().fit(s_tr.reshape(-1, 1), y_tr)
    recal_w = LogisticRegression().fit(s_w.reshape(-1, 1), y_w)
    recal_j = LogisticRegression().fit(s_j.reshape(-1, 1), y_j)

    def loess(yc, p, frac=0.8):
        from statsmodels.nonparametric.smoothers_lowess import lowess
        o = np.argsort(p)
        sm = lowess(yc[o], p[o], frac=frac, return_sorted=True)
        return sm[:, 0], np.clip(sm[:, 1], 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    calib_sets = [
        ('West China', y_w, {
            'Scorecard (recalibrated)': (recal_w.predict_proba(s_w.reshape(-1, 1))[:, 1],
                                         MODEL_COLORS['Scorecard'], '--'),
            'Scorecard (frozen map)': (frozen_map.predict_proba(s_w.reshape(-1, 1))[:, 1],
                                       MODEL_COLORS['Scorecard'], '-'),
            'Logistic regression': (zoo['Raw LR'](X_w), MODEL_COLORS['Raw LR'], '-.'),
            'ROMA': (zoo['ROMA'](X_w) / 100, MODEL_COLORS['ROMA'], '-'),
            'CPH-I': (zoo['CPH-I'](X_w), MODEL_COLORS['CPH-I'], '-'),
            'XGBoost': (zoo['XGBoost'](X_w), MODEL_COLORS['XGBoost'], ':'),
        }),
        ('Japan', y_j, {
            'Scorecard (recalibrated)': (recal_j.predict_proba(s_j.reshape(-1, 1))[:, 1],
                                         MODEL_COLORS['Scorecard'], '--'),
            'Scorecard (frozen map)': (frozen_map.predict_proba(s_j.reshape(-1, 1))[:, 1],
                                       MODEL_COLORS['Scorecard'], '-'),
            'Logistic regression': (zoo['Raw LR'](X_j), MODEL_COLORS['Raw LR'], '-.'),
            'ROMA': (zoo['ROMA'](X_j) / 100, MODEL_COLORS['ROMA'], '-'),
            'CPH-I': (zoo['CPH-I'](X_j), MODEL_COLORS['CPH-I'], '-'),
            'XGBoost': (zoo['XGBoost'](X_j), MODEL_COLORS['XGBoost'], ':'),
        }),
    ]
    for ax, (cname, yc, preds) in zip(axes, calib_sets):
        ax.plot([0, 1], [0, 1], color='black', ls='-', lw=1.0, zorder=1)
        prev = float(np.mean(yc))
        ax.axhline(prev, color='grey', ls=(0, (3, 3)), lw=1.1, zorder=1)
        ax.text(0.985, prev + 0.025, f'prevalence {prev:.2f}', ha='right', fontsize=8.5,
                color='#444444')
        for name, (p, color, style) in preds.items():
            p = np.clip(p, 0, 1)
            xs, ys = loess(yc, p)
            ax.plot(xs, ys, ls=style, lw=1.9, color=color, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed proportion')
        ax.set_title(f'{cname} (n = {len(yc)})', fontweight='bold')
        ax.grid(alpha=0.25, ls='--', zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Calibration curves (LOWESS-smoothed)', fontweight='bold', y=1.0)
    fig.subplots_adjust(bottom=0.17, top=0.9)
    save(fig, 'fig_calibration.png')

    # ================================================================ Figure 5
    p_sc_recal = recal_w.predict_proba(s_w.reshape(-1, 1))[:, 1]
    p_sc_frozen = frozen_map.predict_proba(s_w.reshape(-1, 1))[:, 1]
    p_roma = zoo['ROMA'](X_w) / 100
    thresholds = np.linspace(0.01, 0.60, 60)

    def nb(y, p, pt):
        pred = (np.asarray(p) >= pt).astype(int)
        tp = np.sum((pred == 1) & (np.asarray(y) == 1))
        fp = np.sum((pred == 1) & (np.asarray(y) == 0))
        n = len(y)
        return tp / n - fp / n * (pt / (1 - pt))

    prev = float(np.mean(y_w))
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    nb_sc = np.array([nb(y_w, p_sc_recal, t) for t in thresholds])
    nb_sc_fr = np.array([nb(y_w, p_sc_frozen, t) for t in thresholds])
    nb_roma = np.array([nb(y_w, p_roma, t) for t in thresholds])
    rng = np.random.RandomState(SEED)
    n = len(y_w)
    n_boot = 1000
    sc_boot = np.empty((n_boot, len(thresholds)))
    roma_boot = np.empty((n_boot, len(thresholds)))
    for k in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        m = LogisticRegression().fit(s_w[idx].reshape(-1, 1), y_w[idx])
        p_boot = m.predict_proba(s_w.reshape(-1, 1))[:, 1]
        sc_boot[k] = [nb(y_w[idx], p_boot[idx], t) for t in thresholds]
        roma_boot[k] = [nb(y_w[idx], p_roma[idx], t) for t in thresholds]
    sc_lo = np.percentile(sc_boot, 2.5, axis=0)
    sc_hi = np.percentile(sc_boot, 97.5, axis=0)
    roma_lo = np.percentile(roma_boot, 2.5, axis=0)
    roma_hi = np.percentile(roma_boot, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.axvspan(0.10, 0.50, color='#f0f0f0', zorder=0)
    ax.fill_between(thresholds, sc_lo, sc_hi, color=MODEL_COLORS['Scorecard'],
                    alpha=0.16, zorder=1)
    ax.fill_between(thresholds, roma_lo, roma_hi, color=MODEL_COLORS['ROMA'],
                    alpha=0.13, zorder=1)
    ax.plot(thresholds, nb_all, color='black', ls='--', lw=1.6,
            label='Treat all', zorder=3)
    ax.axhline(0, color='grey', lw=0.9, zorder=2)
    ax.plot(thresholds, nb_sc, color=MODEL_COLORS['Scorecard'], lw=2.2,
            label='Scorecard (recalibrated)', zorder=4)
    ax.plot(thresholds, nb_sc_fr, color=MODEL_COLORS['Scorecard'], ls=(0, (1, 2)),
            lw=1.6, label='Scorecard (frozen map)', zorder=4)
    ax.plot(thresholds, nb_roma, color=MODEL_COLORS['ROMA'], lw=2.2,
            label='ROMA', zorder=4)
    ax.text(0.115, nb_all[7] + 0.02, 'clinical range\n10\u201350%', fontsize=8.5,
            color='#555555')
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.30, 0.52)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit (per patient)')
    ax.set_title('Decision curve analysis, West China (n = 380; shaded = 95% '
                 'bootstrap bands)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    save(fig, 'fig_decision_analysis.png')

    # ================================================================ Figure 6
    cl = pd.read_csv(os.path.join(R, 'clinical_comparators_noise.csv'))
    sub = cl[cl['Cohort'] == 'West China']
    v0 = {m: float(sub[sub['Noise'] == '0%'][m].iloc[0]) for m in MODELS}
    v3 = {m: float(sub[sub['Noise'] == '30%'][m].iloc[0]) for m in MODELS}
    deg = np.array([v3[m] - v0[m] for m in MODELS])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    order = np.argsort(deg)  # least degradation at top
    ys = np.arange(len(MODELS))
    ax.barh(ys, deg[order], height=0.62,
            color=[MODEL_COLORS[MODELS[i]] for i in order], zorder=3)
    for k, i in enumerate(order):
        d = deg[i]
        ax.text(d + (0.002 if d >= 0 else -0.002), k, f'{d:+.3f}',
                ha='left' if d >= 0 else 'right', va='center', fontsize=9.5, zorder=4)
    ax.axvline(0, color='black', lw=1.0, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT[MODELS[i]] for i in order])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(-0.105, 0.012)
    ax.set_xlabel('\u0394 AUROC (0% \u2192 30% multiplicative noise)')
    ax.set_title('Measurement-noise degradation, West China (n = 380)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    save(fig, 'fig2_noise_degradation.png')

    # ================================================================ Figure 7
    lc = pd.read_csv(os.path.join(R, 'external_learning_curves.csv'))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for m in ['Scorecard', 'Raw LR', 'XGBoost', 'CatBoost', 'Random Forest']:
        sub = lc[lc['Model'] == m]
        ax.errorbar(sub['n'], sub['External AUROC mean'], yerr=sub['External AUROC SD'],
                    marker='o', ms=5, capsize=3, lw=1.8,
                    color=MODEL_COLORS[m], label=SHORT[m], zorder=3)
    roma_w = ext[('West China', 'ROMA')][0]
    ax.axhline(roma_w, color=MODEL_COLORS['ROMA'], ls='--', lw=1.3, zorder=2)
    ax.text(64, roma_w + 0.004, f'ROMA ({roma_w:.3f})', fontsize=8.5,
            color=MODEL_COLORS['ROMA'])
    ax.set_xlabel('Training samples (n)')
    ax.set_ylabel('External AUROC on West China (mean \u00b1 SD, 10 draws)')
    ax.set_title('Data efficiency: external performance vs training size',
                 fontweight='bold')
    ax.set_ylim(0.83, 0.965)
    ax.set_xticks(sorted(sub['n'].unique()))
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    save(fig, 'fig_external_learning_curves.png')

    # ================================================================ Figure 8
    sim = pd.read_csv(os.path.join(R, 'simulation_study.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, drift in zip(axes, [0.0, 1.0]):
        sub = sim[(sim['noise'] == 0.5) & (sim['drift'] == drift)]
        xgb = sub[sub['Model'] == 'XGBoost'].pivot(index='shift', columns='n_train',
                                                   values='Test AUROC mean').values
        sc = sub[sub['Model'] == 'Scorecard'].pivot(index='shift', columns='n_train',
                                                    values='Test AUROC mean').values
        diff = xgb - sc
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.10, vmax=0.10, aspect='auto')
        sizes = sorted(sub['n_train'].unique())
        shifts = sorted(sub['shift'].unique())
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)
        ax.set_yticks(range(len(shifts)))
        ax.set_yticklabels(shifts)
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                v = diff[i, j]
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=10,
                        color='white' if abs(v) > 0.05 else 'black')
        ax.set_xlabel('Training samples')
        ax.set_ylabel('Distribution shift')
        ax.set_title('Stationary structure' if drift == 0 else 'Concept drift',
                     fontweight='bold')
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05,
                        pad=0.16, aspect=40)
    cbar.set_label('XGBoost minus scorecard, test AUROC (50% measurement noise)')
    fig.suptitle('Regime map from the synthetic simulation study (20 replicates per cell)',
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'fig_simulation_study.png')

    # ================================================================ Figure S1
    # regenerate per-feature drift for the full 8-model set (fast)
    pf_rows = []
    for scen, feats in [('CA125 only', ['CA125']), ('HE4 only', ['HE4']),
                        ('Both markers', ['CA125', 'HE4'])]:
        for noise in [0.00, 0.10, 0.20, 0.30]:
            aucs = {m: [] for m in MODELS}
            rng = np.random.RandomState(SEED)
            for _ in range(100):
                Xn = X_w.copy()
                for f in feats:
                    Xn[f] = np.maximum(X_w[f].values * rng.normal(1.0, noise, len(X_w)),
                                       1e-5)
                for m in MODELS:
                    aucs[m].append(auroc(y_w, zoo[m](Xn)))
            pf_rows.append({'Cohort': 'West China', 'Scenario': scen, 'Noise': noise,
                            **{m: float(np.mean(aucs[m])) for m in MODELS}})
    pf = pd.DataFrame(pf_rows)
    pf.to_csv(os.path.join(R, 'perfeature_drift.csv'), index=False)

    # regenerate shift-vs-loss rows for the full 8-model set
    ks_df = pd.read_csv(os.path.join(R, 'distribution_shift.csv'))
    tr_cv = {'Scorecard': 0.906, 'Raw LR': 0.894, 'ROMA': 0.912, 'CPH-I': 0.912,
             'CA125 rule': 0.727, 'XGBoost': 0.882, 'CatBoost': 0.900,
             'Random Forest': 0.895}
    loss_rows = []
    for cname, Xc, yc in [('West China', X_w, y_w), ('Japan', X_j, y_j)]:
        mean_ks = float(ks_df[ks_df['Cohort'] == cname]['KS'].mean())
        for m in MODELS:
            ext_auc = auroc(yc, zoo[m](Xc))
            loss_rows.append({'Cohort': cname, 'Model': m,
                              'Transfer loss': round(tr_cv[m] - ext_auc, 3),
                              'Mean KS shift': round(mean_ks, 3),
                              'External AUROC': round(ext_auc, 3)})
    loss = pd.DataFrame(loss_rows)
    loss.to_csv(os.path.join(R, 'shift_vs_loss.csv'), index=False)

    scenarios = ['CA125 only', 'HE4 only', 'Both markers']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, scen in zip(axes, scenarios):
        sub = pf[pf['Scenario'] == scen]
        b0 = sub[sub['Noise'] == 0.00]
        b3 = sub[sub['Noise'] == 0.30]
        degs = np.array([float(b3[m].iloc[0] - b0[m].iloc[0]) for m in MODELS])
        hbars(ax, [SHORT[m] for m in MODELS], degs, [MODEL_COLORS[m] for m in MODELS],
              ci=None)
        ax.axvline(0, color='black', lw=1.0, zorder=2)
        span = max(abs(degs.min()), abs(degs.max()))
        ax.set_xlim(-span - 0.015, span + 0.015)
        ax.set_xlabel('\u0394 AUROC (0% \u2192 30% noise)')
        ax.set_title(f'Noise on: {scen}', fontweight='bold')
        ax.grid(axis='x', alpha=0.25, ls='--', zorder=0)
    fig.tight_layout(w_pad=2.2)
    save(fig, 'fig7_perfeature_drift.png')

    # ================================================================ Figure S2
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for m in MODELS:
        sub = loss[loss['Model'] == m]
        ax.scatter(sub['Mean KS shift'], sub['Transfer loss'], s=85,
                   color=MODEL_COLORS[m], zorder=4, edgecolor='black', lw=0.7)
    annot_off = {}
    for _, row in loss.iterrows():
        dx = 0.012 if row['Cohort'] == 'West China' else 0.012
        dy = 0.008 if row['Cohort'] == 'Japan' else -0.012
        ax.annotate(row['Cohort'], (row['Mean KS shift'], row['Transfer loss']),
                    textcoords='offset points', xytext=(12 if dx > 0 else -12, dy * 60),
                    fontsize=8.5, color='#333333')
    ax.axhline(0, color='grey', lw=0.9, ls=':')
    ax.set_xlabel('Mean Kolmogorov\u2013Smirnov shift vs training distribution')
    ax.set_ylabel('Transfer loss (training CV \u2212 external AUROC)')
    ax.set_title('Real cross-cohort shift versus transfer loss (0% noise)',
                 fontweight='bold')
    handles = [Patch(color=MODEL_COLORS[m], label=SHORT[m]) for m in MODELS]
    ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.95,
              fontsize=8.5)
    ax.grid(alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    save(fig, 'fig8_shift_vs_degradation.png')

    # ================================================================ Figure S3
    lc_int = pd.read_csv(os.path.join(R, 'learning_curves.csv'))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for m in ['Scorecard', 'Raw LR', 'XGBoost', 'CatBoost', 'Random Forest']:
        sub = lc_int[lc_int['Model'] == m]
        ax.errorbar(sub['n'], sub['CV AUROC mean'], yerr=sub['CV AUROC SD'],
                    marker='o', ms=5, capsize=3, lw=1.8,
                    color=MODEL_COLORS[m], label=SHORT[m], zorder=3)
    ax.axhline(0.912, color=MODEL_COLORS['ROMA'], ls='--', lw=1.3)
    ax.text(70, 0.9155, 'ROMA (applied, training AUROC)', fontsize=8.5,
            color=MODEL_COLORS['ROMA'])
    ax.set_xlabel('Training samples (n)')
    ax.set_ylabel('Five-fold CV AUROC (mean \u00b1 SD, 10 draws)')
    ax.set_title('Internal learning curves (Chinese training cohort)',
                 fontweight='bold')
    ax.set_xticks(sorted(sub['n'].unique()))
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    save(fig, 'fig9_learning_curves.png')

    # ================================================================ Figure S4
    from robustness_extended import make_noisy
    clean = {m: float(cl[(cl['Cohort'] == 'West China') & (cl['Noise'] == '0%')][m].iloc[0])
             for m in MODELS}
    panels = [('Multiplicative 30%', 'multiplicative', 0.30),
              ('Additive 30%', 'additive', 0.30),
              ('Combined 30%', 'combined', 0.30),
              ('Outliers (5% gross errors)', 'outliers', 0.05)]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for ax, (title, kind, level) in zip(axes.ravel(), panels):
        aucs = {m: [] for m in MODELS}
        for rep in range(100):
            Xn = make_noisy(X_w, kind, level, seed=rep)
            for m in MODELS:
                aucs[m].append(auroc(y_w, zoo[m](Xn)))
        degs = np.array([float(np.mean(aucs[m]) - clean[m]) for m in MODELS])
        hbars(ax, [SHORT[m] for m in MODELS], degs, [MODEL_COLORS[m] for m in MODELS])
        ax.axvline(0, color='black', lw=1.0, zorder=2)
        span = max(abs(degs.min()), abs(degs.max()))
        ax.set_xlim(-span - 0.015, span + 0.015)
        ax.set_xlabel('\u0394 AUROC vs clean baseline')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='x', alpha=0.25, ls='--', zorder=0)
    fig.suptitle('Robustness across noise mechanisms, West China (n = 380; '
                 '100 perturbations)', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, 'fig_noise_types.png')

    print('all figures regenerated')


if __name__ == '__main__':
    main()
