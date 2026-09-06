"""
Decision-analytic evaluation: consequence trade-off curves and weighted-harm
analysis on the West China cohort.

For each model, sweeping the decision threshold over the full range gives the
(missed cancers per 100 women, unnecessary referrals per 100 women) trade-off
curve. A weighted-harm analysis then identifies, for a range of exchange-rate
weights w (harm of one missed cancer relative to one unnecessary referral),
the threshold that minimises net harm H = w x missed + unnecessary, and
compares models at that optimum.

Outputs:
    results/decision_harm_tradeoff.csv
    figures/fig6_consequence_curves.png
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lib import (SEED, RESULTS_DIR, FIGURES_DIR, MODEL_COLORS, MODEL_LABELS,
                       ROMA_CUT_PRE, ROMA_CUT_POST, CPHI_CUT)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def consequences(y, p, pt):
    """Missed cancers and unnecessary referrals per 100 women at threshold pt."""
    pred = np.asarray(p) >= pt
    y = np.asarray(y)
    missed = float(np.sum((pred == 0) & (y == 1)) / len(y) * 100)
    unnec = float(np.sum((pred == 1) & (y == 0)) / len(y) * 100)
    return missed, unnec


def main():
    import bench_lib as B
    np.random.seed(SEED)

    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(B.WEST_PATH, med)
    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)

    s_tr = zoo['Scorecard'](X_tr)
    s_w = zoo['Scorecard'](X_w)
    recal_map = LogisticRegression().fit(s_w.reshape(-1, 1), y_w)
    frozen_map = LogisticRegression().fit(s_tr.reshape(-1, 1), y_tr)

    model_probs = {
        'Scorecard (recalibrated)': recal_map.predict_proba(s_w.reshape(-1, 1))[:, 1],
        'Raw LR': zoo['Raw LR'](X_w),
        'ROMA': zoo['ROMA'](X_w) / 100.0,
        'CPH-I': zoo['CPH-I'](X_w),
        'XGBoost': zoo['XGBoost'](X_w),
    }

    thresholds = np.linspace(0.01, 0.60, 120)
    prev100 = float(np.mean(y_w)) * 100  # cancer prevalence per 100

    # weighted-harm analysis
    weights = [1, 2, 5, 10, 20, 50]
    rows = []
    for mname, p in model_probs.items():
        for w in weights:
            harms = []
            for pt in thresholds:
                missed, unnec = consequences(y_w, p, pt)
                harms.append(w * missed + unnec)
            harms = np.array(harms)
            k = int(np.argmin(harms))
            missed_o, unnec_o = consequences(y_w, p, thresholds[k])
            rows.append({'Model': mname, 'Weight w': w,
                         'Optimal threshold': round(float(thresholds[k]), 3),
                         'Missed per 100 women': round(missed_o, 1),
                         'Unnecessary per 100 women': round(unnec_o, 1),
                         'Net harm H': round(float(harms[k]), 1)})
    # default strategies (same for every model, reported once per weight)
    for w in weights:
        rows.append({'Model': 'Treat none (reference)', 'Weight w': w,
                     'Optimal threshold': None,
                     'Missed per 100 women': round(prev100, 1),
                     'Unnecessary per 100 women': 0.0,
                     'Net harm H': round(w * prev100, 1)})
        rows.append({'Model': 'Treat all (reference)', 'Weight w': w,
                     'Optimal threshold': None,
                     'Missed per 100 women': 0.0,
                     'Unnecessary per 100 women': round(100 - prev100, 1),
                     'Net harm H': round(100 - prev100, 1)})
    harm_df = pd.DataFrame(rows)
    harm_df.to_csv(os.path.join(RESULTS_DIR, 'decision_harm_tradeoff.csv'),
                   index=False)
    print(harm_df.to_string(index=False))

    # published-cutpoint operating points for annotation
    roma_rule = np.where(X_w['Is_Postmenopausal'].values == 1,
                         zoo['ROMA'](X_w) >= ROMA_CUT_POST,
                         zoo['ROMA'](X_w) >= ROMA_CUT_PRE)
    roma_m, roma_u = consequences(y_w, roma_rule, 0.5)
    cphi_m, cphi_u = consequences(y_w, zoo['CPH-I'](X_w) * 100 >= CPHI_CUT, 0.5)
    sc_m, sc_u = consequences(y_w, s_w >= 1, 0.5)
    print(f"\nPublished-cutpoint operating points (missed, unnecessary per 100):")
    print(f"  Scorecard tier >= +1 : {sc_m:.1f}, {sc_u:.1f}")
    print(f"  ROMA cutpoints        : {roma_m:.1f}, {roma_u:.1f}")
    print(f"  CPH-I >= 7%           : {cphi_m:.1f}, {cphi_u:.1f}")

    # trade-off curves figure
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ls_map = {'Scorecard (recalibrated)': '-', 'Raw LR': '--', 'ROMA': '-.',
              'CPH-I': ':', 'XGBoost': (0, (3, 1, 1, 1))}
    for mname, p in model_probs.items():
        pts = []
        for pt in thresholds:
            pts.append(consequences(y_w, p, pt))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], ls=ls_map[mname], lw=2.0,
                color=MODEL_COLORS.get(mname.split(' ')[0],
                                       MODEL_COLORS['Scorecard']),
                label=mname, zorder=3)
    # operating points
    ax.scatter([sc_m], [sc_u], s=90, marker='*', color=MODEL_COLORS['Scorecard'],
               edgecolor='black', zorder=5)
    ax.annotate('Scorecard\ntier \u2265 +1', (sc_m, sc_u), xytext=(sc_m + 1.5, sc_u + 1.5),
                fontsize=8.5, arrowprops=dict(arrowstyle='-', lw=0.8, color='#333'))
    ax.scatter([roma_m], [roma_u], s=80, marker='D', color=MODEL_COLORS['ROMA'],
               edgecolor='black', zorder=5)
    ax.annotate('ROMA published\ncutpoints', (roma_m, roma_u),
                xytext=(roma_m - 14, roma_u + 0.6), fontsize=8.5,
                arrowprops=dict(arrowstyle='-', lw=0.8, color='#333'))
    ax.scatter([cphi_m], [cphi_u], s=80, marker='s', color=MODEL_COLORS['CPH-I'],
               edgecolor='black', zorder=5)
    ax.annotate('CPH-I \u2265 7%', (cphi_m, cphi_u), xytext=(cphi_m + 1.5, cphi_u - 1.5),
                fontsize=8.5, arrowprops=dict(arrowstyle='-', lw=0.8, color='#333'))
    # default strategies
    ax.scatter([prev100], [0], s=80, marker='o', color='#444444', zorder=4)
    ax.annotate('Treat none', (prev100, 0), xytext=(prev100 - 12, 1.2), fontsize=8.5)
    ax.scatter([0], [100 - prev100], s=80, marker='o', color='#444444', zorder=4)
    ax.annotate('Treat all', (0, 100 - prev100), xytext=(1.2, 100 - prev100 - 2.5),
                fontsize=8.5)
    ax.set_xlabel('Missed cancers per 100 women')
    ax.set_ylabel('Unnecessary referrals per 100 women')
    ax.set_title('Decision-consequence trade-off curves, West China (n = 380)',
                 fontweight='bold')
    ax.set_xlim(-2, prev100 + 4)
    ax.set_ylim(-2, 100 - prev100 + 4)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, ls='--', zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig6_consequence_curves.png'),
                bbox_inches='tight', pad_inches=0.25, facecolor='white', dpi=300)
    plt.close()
    print("\nfig6_consequence_curves.png saved")


if __name__ == '__main__':
    main()
