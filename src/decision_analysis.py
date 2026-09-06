"""
Decision-analytic evaluation on the West China cohort (primary external).

1. Operating-point analysis for the scorecard across every integer cutpoint:
   sensitivity, specificity, PPV, NPV, likelihood ratios, cancer yield.
2. Published-cutoff decision consequences: scorecard high-risk tier (>= +1),
   ROMA published cutoffs, CPH-I >= 7%, CA125 >= 35 U/mL — per 100 women:
   referred cancers, missed cancers, unnecessary referrals.
3. Matched-sensitivity comparison: scorecard threshold matched to ROMA's
   sensitivity (and vice versa); difference in unnecessary referrals with a
   McNemar test on discordant pairs.
4. Decision curve analysis with 95% bootstrap confidence bands for the
   scorecard (recalibrated) and ROMA.

Outputs:
    results/decision_analysis.csv
    results/dca_bootstrap.csv
    figures/fig_decision_analysis.png
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
                       ROMA_CUT_PRE, ROMA_CUT_POST, CPHI_CUT, CA125_CUT)

RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR


def operating_characteristics(y, pred_binary):
    y = np.asarray(y)
    pred_binary = np.asarray(pred_binary)
    tp = ((pred_binary == 1) & (y == 1)).sum()
    fp = ((pred_binary == 1) & (y == 0)).sum()
    fn = ((pred_binary == 0) & (y == 1)).sum()
    tn = ((pred_binary == 0) & (y == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    lrp = sens / (1 - spec) if spec < 1 else np.inf
    lrn = (1 - sens) / spec if spec > 0 else np.inf
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                sens=float(sens), spec=float(spec), ppv=float(ppv), npv=float(npv),
                lr_plus=float(lrp), lr_minus=float(lrn))


def net_benefit(y, p, pt):
    pred = (np.asarray(p) >= pt).astype(int)
    tp = np.sum((pred == 1) & (np.asarray(y) == 1))
    fp = np.sum((pred == 1) & (np.asarray(y) == 0))
    n = len(y)
    return tp / n - fp / n * (pt / (1 - pt))


def mcnemar_p(b, c):
    """Two-sided McNemar exact p-value from discordant counts."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binomtest(min(b, c), n, 0.5).pvalue))


def main():
    import bench_lib as B
    np.random.seed(SEED)

    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(B.WEST_PATH, med)

    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)
    s_w = zoo['Scorecard'](X_w)

    # score-to-probability maps
    s_tr = zoo['Scorecard'](X_tr)
    frozen_map = LogisticRegression().fit(s_tr.reshape(-1, 1), y_tr)
    recal_map = LogisticRegression().fit(s_w.reshape(-1, 1), y_w)
    p_sc_frozen = frozen_map.predict_proba(s_w.reshape(-1, 1))[:, 1]
    p_sc_recal = recal_map.predict_proba(s_w.reshape(-1, 1))[:, 1]
    p_roma = zoo['ROMA'](X_w) / 100.0
    p_cphi = zoo['CPH-I'](X_w)
    p_xgb = zoo['XGBoost'](X_w)

    # 1. Operating points across all scorecard cutpoints
    rows = []
    cutpoints = sorted(np.unique(s_w.astype(int)))
    for c in cutpoints:
        oc = operating_characteristics(y_w, s_w >= c)
        rows.append({'Cutpoint': int(c),
                     'N referred': oc['tp'] + oc['fp'],
                     'Sensitivity': round(oc['sens'], 3), 'Specificity': round(oc['spec'], 3),
                     'PPV': round(oc['ppv'], 3), 'NPV': round(oc['npv'], 3),
                     'LR+': round(oc['lr_plus'], 2), 'LR-': round(oc['lr_minus'], 2)})
    op_df = pd.DataFrame(rows)
    op_df.to_csv(os.path.join(RESULTS, 'scorecard_operating_points.csv'), index=False)
    print("Scorecard operating points (West China, n = 380; 188 cancers):")
    print(op_df.to_string(index=False))

    # 2. Decision consequences per 100 women at published rules + scorecard tier
    def consequence_table():
        out = []
        # scorecard high-risk tier >= +1 (pre-specified tier)
        oc = operating_characteristics(y_w, s_w >= 1)
        out.append({'Strategy': 'Scorecard high-risk tier (score >= +1)',
                    'Sensitivity': round(oc['sens'], 3), 'Specificity': round(oc['spec'], 3),
                    'Missed cancers per 100': round((1 - oc['sens']) * 100, 1),
                    'Unnecessary referrals per 100': round((1 - oc['spec']) * 100, 1)})
        # ROMA published cutoffs
        roma_rule = np.where(X_w['Is_Postmenopausal'].values == 1,
                             zoo['ROMA'](X_w) >= ROMA_CUT_POST,
                             zoo['ROMA'](X_w) >= ROMA_CUT_PRE).astype(float)
        oc = operating_characteristics(y_w, roma_rule)
        out.append({'Strategy': 'ROMA published cutoffs (13.1%/27.7%)',
                    'Sensitivity': round(oc['sens'], 3), 'Specificity': round(oc['spec'], 3),
                    'Missed cancers per 100': round((1 - oc['sens']) * 100, 1),
                    'Unnecessary referrals per 100': round((1 - oc['spec']) * 100, 1)})
        # CPH-I cutoff
        oc = operating_characteristics(y_w, p_cphi * 100 >= CPHI_CUT)
        out.append({'Strategy': 'CPH-I cutoff (>= 7%)',
                    'Sensitivity': round(oc['sens'], 3), 'Specificity': round(oc['spec'], 3),
                    'Missed cancers per 100': round((1 - oc['sens']) * 100, 1),
                    'Unnecessary referrals per 100': round((1 - oc['spec']) * 100, 1)})
        # CA125 rule
        oc = operating_characteristics(y_w, zoo['CA125 rule'](X_w) >= 1)
        out.append({'Strategy': 'CA125 >= 35 U/mL',
                    'Sensitivity': round(oc['sens'], 3), 'Specificity': round(oc['spec'], 3),
                    'Missed cancers per 100': round((1 - oc['sens']) * 100, 1),
                    'Unnecessary referrals per 100': round((1 - oc['spec']) * 100, 1)})
        # treat-all reference
        out.append({'Strategy': 'Treat all (reference)',
                    'Sensitivity': 1.0, 'Specificity': 0.0,
                    'Missed cancers per 100': 0.0,
                    'Unnecessary referrals per 100': 50.5})
        return pd.DataFrame(out)

    cons_df = consequence_table()
    cons_df.to_csv(os.path.join(RESULTS, 'decision_consequences.csv'), index=False)
    print("\nDecision consequences per 100 women (West China):")
    print(cons_df.to_string(index=False))

    # 3. Matched-sensitivity comparison: scorecard threshold matched to ROMA sens
    roma_rule = np.where(X_w['Is_Postmenopausal'].values == 1,
                         zoo['ROMA'](X_w) >= ROMA_CUT_POST,
                         zoo['ROMA'](X_w) >= ROMA_CUT_PRE).astype(float)
    roma_sens = operating_characteristics(y_w, roma_rule)['sens']
    # find the scorecard cutpoint whose sensitivity is closest from below
    best_c, best_diff = None, np.inf
    for c in sorted(np.unique(s_w.astype(int))):
        sens_c = operating_characteristics(y_w, s_w >= c)['sens']
        if sens_c >= roma_sens - 0.02:
            diff = sens_c - roma_sens
            if abs(diff) < abs(best_diff):
                best_c, best_diff = c, diff
    sc_rule = (s_w >= best_c).astype(float)
    oc_sc = operating_characteristics(y_w, sc_rule)
    oc_roma = operating_characteristics(y_w, roma_rule)
    # McNemar on discordant pairs (false-positive reduction)
    b = int(np.sum((sc_rule == 0) & (roma_rule == 1) & (y_w == 0)))  # FP only ROMA
    c = int(np.sum((sc_rule == 1) & (roma_rule == 0) & (y_w == 0)))
    p_mc = mcnemar_p(b, c)
    print(f"\nMatched-sensitivity comparison (scorecard cutpoint {best_c}):")
    print(f"  ROMA sens {oc_roma['sens']:.3f}, spec {oc_roma['spec']:.3f}; "
          f"scorecard sens {oc_sc['sens']:.3f}, spec {oc_sc['spec']:.3f}")
    print(f"  Discordant benign pairs: ROMA-only FP = {b}, scorecard-only FP = {c}, "
          f"McNemar p = {p_mc:.4f}")

    # 4. DCA with bootstrap CI bands
    thresholds = np.linspace(0.01, 0.60, 60)
    prev = float(np.mean(y_w))
    nb_sc = np.array([net_benefit(y_w, p_sc_recal, t) for t in thresholds])
    nb_sc_fr = np.array([net_benefit(y_w, p_sc_frozen, t) for t in thresholds])
    nb_roma = np.array([net_benefit(y_w, p_roma, t) for t in thresholds])
    nb_xgb = np.array([net_benefit(y_w, p_xgb, t) for t in thresholds])
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))

    rng = np.random.RandomState(SEED)
    n = len(y_w)
    n_boot = 1000
    sc_boot = np.empty((n_boot, len(thresholds)))
    roma_boot = np.empty((n_boot, len(thresholds)))
    for k in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        # refit the recalibration map within the bootstrap sample (honest CI)
        m = LogisticRegression().fit(s_w[idx].reshape(-1, 1), y_w[idx])
        p_boot = m.predict_proba(s_w.reshape(-1, 1))[:, 1]
        sc_boot[k] = [net_benefit(y_w[idx], p_boot[idx], t) for t in thresholds]
        roma_boot[k] = [net_benefit(y_w[idx], p_roma[idx], t) for t in thresholds]
    sc_lo = np.percentile(sc_boot, 2.5, axis=0)
    sc_hi = np.percentile(sc_boot, 97.5, axis=0)
    roma_lo = np.percentile(roma_boot, 2.5, axis=0)
    roma_hi = np.percentile(roma_boot, 97.5, axis=0)

    dca_df = pd.DataFrame({
        'threshold': np.round(thresholds, 3),
        'treat_all': nb_all, 'scorecard_recal': nb_sc,
        'scorecard_recal_lo': sc_lo, 'scorecard_recal_hi': sc_hi,
        'ROMA': nb_roma, 'ROMA_lo': roma_lo, 'ROMA_hi': roma_hi,
    })
    dca_df.to_csv(os.path.join(RESULTS, 'dca_bootstrap.csv'), index=False)
    print("\ndca_bootstrap.csv saved")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(thresholds, sc_lo, sc_hi, color='#2ca02c', alpha=0.15)
    ax.fill_between(thresholds, roma_lo, roma_hi, color='#d62728', alpha=0.12)
    ax.plot(thresholds, nb_all, 'k--', lw=1.5, label='Treat all')
    ax.plot(thresholds, nb_sc, color='#2ca02c', lw=2,
            label='Scorecard (recalibrated)')
    ax.plot(thresholds, nb_sc_fr, color='#2ca02c', lw=1.2, ls=':',
            label='Scorecard (frozen map)')
    ax.plot(thresholds, nb_roma, color='#d62728', lw=2, label='ROMA')
    ax.plot(thresholds, nb_xgb, color='#ff7f0e', lw=1.4, ls='--', label='XGBoost')
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit (per patient)')
    ax.set_title('Decision curve analysis with 95% bootstrap bands (West China, n = 380)',
                 fontweight='bold')
    ax.set_xlim(0, 0.6)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'fig_decision_analysis.png'),
                facecolor='white', dpi=200)
    plt.close()
    print("fig_decision_analysis.png saved")


if __name__ == '__main__':
    main()
