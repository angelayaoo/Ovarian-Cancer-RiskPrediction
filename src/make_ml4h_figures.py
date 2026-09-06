"""
Canonical ML4H 2026 paper figures (Figures 1-5).

Generates every figure used in paper/main.tex, in both the full-width and
single-column versions, directly into ../paper/figures/. Run from the
repository root after the numerical pipeline (see README, "How to
Reproduce").

Figure mapping (paper/main.tex):
    Figure 1  complexity ladder          figures/fig_ladder_col.png   (column)
    Figure 2  consequence trade-offs     figures/fig_consequences.png (full width)
    Figure 3  noise degradation          figures/fig_noise_col.png    (column)
    Figure 4  learning curves            figures/fig_learning.png     (full width)
    Figure 5  regime map (simulation)    figures/fig_regime_col.png   (column)

Grayscale versions (*-gray.png) are written alongside each file.
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
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from bench_lib import (SEED, RESULTS_DIR, MODEL_COLORS, bootstrap_auroc, auroc)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

MODELS = ['Scorecard', 'Raw LR', 'ROMA', 'CPH-I', 'CA125 rule',
          'XGBoost', 'CatBoost', 'Random Forest']
SHORT = {'Scorecard': 'Scorecard', 'Raw LR': 'Logistic regr.',
         'ROMA': 'ROMA', 'CPH-I': 'CPH-I', 'CA125 rule': 'CA125 rule',
         'XGBoost': 'XGBoost', 'CatBoost': 'CatBoost',
         'Random Forest': 'Rand. forest'}

plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
                     'legend.fontsize': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=300)
    plt.close(fig)
    from PIL import Image
    Image.open(path).convert('L').save(path.replace('.png', '-gray.png'))
    print('saved', name, '+ gray')


def cons(y, p, pt):
    pred = np.asarray(p) >= pt
    y = np.asarray(y)
    return (float(np.sum((pred == 0) & (y == 1)) / len(y) * 100),
            float(np.sum((pred == 1) & (y == 0)) / len(y) * 100))


# ---------------------------------------------------------------- Figure 1
def fig_ladder(X_w, y_w, zoo, column=False):
    order = ['CA125 rule', 'ROMA', 'CPH-I', 'Logistic regr.', 'Scorecard',
             'Rand. forest', 'XGBoost', 'CatBoost']   # bottom -> top
    keys = ['CA125 rule', 'ROMA', 'CPH-I', 'Raw LR', 'Scorecard',
            'Random Forest', 'XGBoost', 'CatBoost']
    fam = {'CA125 rule': 'formula', 'ROMA': 'formula', 'CPH-I': 'formula',
           'Raw LR': 'linear', 'Scorecard': 'linear',
           'Random Forest': 'tree', 'XGBoost': 'tree', 'CatBoost': 'tree'}
    size = (3.1, 2.9) if column else (5.7, 3.7)
    fig, ax = plt.subplots(figsize=size)
    ys = np.arange(len(order))
    for y, m in zip(ys, keys):
        boot = bootstrap_auroc(y_w, zoo[m](X_w), n_boot=2000)
        v = float(np.mean(boot))
        lo = float(np.percentile(boot, 2.5))
        hi = float(np.percentile(boot, 97.5))
        ax.barh(y, v - 0.5, left=0.5, height=0.6, color=MODEL_COLORS[m],
                zorder=3)
        ax.plot([lo, hi], [y, y], color='black', lw=1.0, zorder=4)
        ax.plot([lo, lo], [y - 0.09, y + 0.09], color='black', lw=1.0, zorder=4)
        ax.plot([hi, hi], [y - 0.09, y + 0.09], color='black', lw=1.0, zorder=4)
        ax.text(hi + 0.006, y, f'{v:.2f}', va='center', ha='left',
                fontsize=8 if not column else 6.5, zorder=5,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    ax.axvline(0.5, color='grey', ls='--', lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=8 if not column else 6.5)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.55, 1.00)
    ax.set_xlabel('External AUROC (West China)')
    ax.set_title('Seven parameters reach the discrimination ceiling',
                 fontweight='bold')
    from matplotlib.patches import Patch
    handles = [
        Patch(color=MODEL_COLORS['Scorecard'], label='Linear models'),
        Patch(color=MODEL_COLORS['ROMA'], label='Published formulas'),
        Patch(color=MODEL_COLORS['XGBoost'], label='Tree ensembles'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=7, framealpha=0.95)
    ax.grid(axis='x', alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    save(fig, 'fig_ladder_col.png' if column else 'fig_ladder.png')


# ---------------------------------------------------------------- Figure 2
def fig_consequences(X_tr, y_tr, X_w, y_w, zoo):
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
    marks = {'Scorecard (recalibrated)': ((12.4, 1.3), '*', 'Tier \u2265 +1'),
             'Logistic regression': (None, None, None),
             'ROMA': ((21.6, 1.6), 'D', 'Published cutpoints'),
             'CPH-I': ((12.6, 2.6), 's', '\u2265 7% cutpoint')}
    thresholds = np.linspace(0.01, 1.00, 240)
    prev100 = float(np.mean(y_w)) * 100
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.6))
    for ax, mname in zip(axes.ravel(), probs.keys()):
        p = probs[mname]
        pts = np.array([cons(y_w, p, pt) for pt in thresholds])
        ax.plot(pts[:, 0], pts[:, 1], ls='-', lw=2.2, color=colors[mname], zorder=3)
        ax.scatter([prev100], [0], s=45, marker='o', color='#888888', zorder=2)
        ax.scatter([0], [100 - prev100], s=45, marker='s', color='#888888', zorder=2)
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
    save(fig, 'fig_consequences.png')


# ---------------------------------------------------------------- Figure 3
def fig_noise(column=False):
    cl = pd.read_csv(os.path.join(RESULTS_DIR, 'clinical_comparators_noise.csv'))
    sub = cl[cl['Cohort'] == 'West China']
    v0 = {m: float(sub[sub['Noise'] == '0%'][m].iloc[0]) for m in MODELS}
    v3 = {m: float(sub[sub['Noise'] == '30%'][m].iloc[0]) for m in MODELS}
    sd3 = {m: float(sub[sub['Noise'] == '30%'][m + ' SD'].iloc[0]) for m in MODELS}
    deg = np.array([v3[m] - v0[m] for m in MODELS])
    sds = np.array([sd3[m] for m in MODELS])
    size = (3.1, 2.7) if column else (5.7, 3.6)
    fig, ax = plt.subplots(figsize=size)
    o = np.argsort(deg)
    ys = np.arange(len(MODELS))
    ax.barh(ys, deg[o], height=0.5,
            color=[MODEL_COLORS[MODELS[i]] for i in o], zorder=3)
    ax.errorbar(deg[o], ys, xerr=sds[o], fmt='none',
                ecolor='black', lw=0.8, capsize=2, zorder=4)
    for k, i in enumerate(o):
        d = deg[i]
        ax.text(d, k + 0.34, f'{d:+.3f}', ha='center', va='center',
                fontsize=8 if not column else 6.5, zorder=5,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    ax.axvline(0, color='black', lw=0.9, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT[MODELS[i]] for i in o],
                       fontsize=8 if not column else 6.5)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(-0.13, 0.02)
    ax.set_xlabel('$\Delta$ AUROC (0% $\to$ 30% multiplicative noise)')
    ax.set_title('Measurement-noise degradation, West China', fontweight='bold')
    ax.grid(axis='x', alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    save(fig, 'fig_noise_col.png' if column else 'fig_noise.png')


# ---------------------------------------------------------------- Figure 4
def fig_learning():
    lc = pd.read_csv(os.path.join(RESULTS_DIR, 'external_learning_curves.csv'))
    models_l = ['Scorecard', 'Raw LR', 'XGBoost', 'CatBoost', 'Random Forest']
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 5.0))
    for ax, m in zip(axes.ravel(), models_l):
        sub = lc[lc['Model'] == m]
        ax.errorbar(sub['n'], sub['External AUROC mean'],
                    yerr=sub['External AUROC SD'], marker='o', ms=3.5,
                    capsize=2, lw=1.6, capthick=0.7, elinewidth=0.7,
                    color=MODEL_COLORS[m], zorder=3)
        ax.axhline(0.886, color=MODEL_COLORS['ROMA'], ls='--', lw=0.9, zorder=2)
        ax.set_xlim(50, 370)
        ax.set_ylim(0.83, 0.965)
        ax.set_xticks([60, 120, 180, 240, 300, 349])
        ax.tick_params(axis='x', labelsize=7)
        vmin, vmax = sub['External AUROC mean'].min(), sub['External AUROC mean'].max()
        ax.set_title(f'({chr(97+models_l.index(m))}) {SHORT[m]}: {vmin:.3f} \u2192 {vmax:.3f}',
                     fontsize=8.5, fontweight='bold')
        ax.grid(alpha=0.25, ls='--', zorder=0)
        if ax in axes[0, :]:
            ax.tick_params(axis='x', labelbottom=False)
        if ax in axes[:, 0]:
            ax.set_ylabel('External AUROC\n(West China)', fontsize=7.5)
        if ax in axes[1, :]:
            ax.set_xlabel('Training samples (n)', fontsize=7.5)
    axes.ravel()[-1].axis('off')
    from matplotlib.lines import Line2D
    axes[0, 0].legend(
        handles=[Line2D([0], [0], color=MODEL_COLORS['ROMA'], ls='--', lw=1.4,
                        label='ROMA (fixed formula)')],
        loc='lower left', fontsize=8, framealpha=0.9)
    fig.tight_layout(h_pad=1.8, w_pad=1.6)
    save(fig, 'fig_learning.png')


# ---------------------------------------------------------------- Figure 5
def fig_regime(column=True):
    sim = pd.read_csv(os.path.join(RESULTS_DIR, 'simulation_study.csv'))
    norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.10)
    if column:
        fig, axes = plt.subplots(2, 1, figsize=(3.1, 3.9))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.8))
        fig.subplots_adjust(left=0.10, right=0.98, wspace=0.32, bottom=0.34, top=0.87)
    for ax, drift in zip(axes, [0.0, 1.0]):
        sub = sim[(sim['noise'] == 0.5) & (sim['drift'] == drift)]
        xgb = sub[sub['Model'] == 'XGBoost'].pivot(index='shift', columns='n_train',
                                                   values='Test AUROC mean').values
        scv = sub[sub['Model'] == 'Scorecard'].pivot(index='shift', columns='n_train',
                                                     values='Test AUROC mean').values
        diff = xgb - scv
        nr, nc = diff.shape
        X, Y = np.meshgrid(np.arange(nc) + 1, np.arange(nr) + 1)
        ax.pcolormesh(X, Y, diff, cmap='RdBu_r', norm=norm,
                      edgecolors='white', linewidth=1.4 if not column else 1.0,
                      zorder=2)
        for i in range(nr):
            for j in range(nc):
                ax.text(j + 1, i + 1, f'{diff[i, j]:+.2f}', ha='center', va='center',
                        fontsize=8 if not column else 6, color='black', zorder=4,
                        path_effects=[pe.withStroke(linewidth=1.6 if not column else 1.0,
                                                    foreground='white')])
        ax.set_xlim(0.5 if not column else 0.6, nc + 1.5 if not column else nc + 1.4)
        ax.set_ylim(0.5 if not column else 0.6, nr + 1.5 if not column else nr + 1.4)
        ax.set_xticks(np.arange(nc) + 1)
        ax.set_xticklabels(['100', '200', '400', '800'], fontsize=8 if not column else 6)
        ax.set_yticks(np.arange(nr) + 1)
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=8 if not column else 6)
        ax.tick_params(length=0)
        ax.set_xlabel('Training samples', fontsize=8.5 if not column else 6.5)
        if column:
            ax.set_title(('Stationary' if drift == 0 else 'Concept drift'),
                         fontsize=7, fontweight='bold')
    if column:
        fig.text(0.02, 0.5, 'Distribution shift', rotation=90, va='center',
                 ha='center', fontsize=6.5)
        fig.suptitle('XGBoost $-$ scorecard test AUROC (50% noise)',
                     fontsize=7, y=0.97)
        fig.tight_layout(rect=[0.06, 0, 1, 0.95])
    else:
        cax = fig.add_axes([0.10, 0.10, 0.88, 0.055])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                          cax=cax, orientation='horizontal')
        cb.set_label('XGBoost $-$ scorecard test AUROC (50% noise)', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.5)
        fig.text(0.03, 0.55, 'Distribution shift', rotation=90, va='center',
                 ha='center', fontsize=8.5)
        for ax, lab in zip(axes, ['(a) Stationary structure', '(b) Concept drift']):
            ax.set_title(lab, fontsize=9.5, fontweight='bold')
    save(fig, 'fig_regime_col.png' if column else 'fig_regime.png')


def main():
    import bench_lib as B
    np.random.seed(SEED)
    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(B.WEST_PATH, med)
    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)

    fig_ladder(X_w, y_w, zoo, column=False)
    fig_consequences(X_tr, y_tr, X_w, y_w, zoo)
    fig_noise(column=False)
    fig_learning()
    fig_regime(column=True)
    print('all ML4H paper figures written to', OUT)


if __name__ == '__main__':
    main()
