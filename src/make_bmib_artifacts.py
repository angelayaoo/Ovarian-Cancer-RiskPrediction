"""BIBM 2026 UGHS paper: reproduce every BIBM-specific number and figure.

This script computes the analyses that are reported in bmib_hs/main.tex
but are not produced by the core pipeline scripts:

  results/bmib_complete_case.csv      complete-case (n=100) AUROCs + 95% CIs
  results/bmib_tuned_ensembles.csv    Setting B (nested-CV tuned) AUROCs + CIs
  results/bmib_mice.csv               MICE-imputed West China AUROCs (all 8 models)
  results/bmib_recalibrated_harms.csv recalibrated ROMA/CPH-I minimum harms
  results/bmib_calibration_slope.csv  probability-scale slope/intercept/ECE
  results/bmib_cc_calibration.csv     frozen calibration on the complete cases
  results/bmib_cc_operating.csv       complete-case published-cutpoint operating points
  results/bmib_japan_fpr.csv          false-positive rates among Japan's 150 healthy women
  results/bmib_cc_balance.csv         complete-case vs excluded outcome/covariate balance
  results/bmib_japan_descriptives.csv Japan lowest-bin shares and marker medians
  results/bmib_no_he4.csv             three-feature (no-HE4) refit AUROCs
  results/bmib_insample.csv           in-sample ensemble fits (vs CV in Section III-E)
  bmib_hs/figures/fig1_auroc.png      complete-case forest plot
  bmib_hs/figures/fig2_cons.png        decision curve analysis (net benefit)
  bmib_hs/figures/fig3_regime_col.png ensemble-average vs scorecard regime map

Runtime: a few minutes (bootstrap resampling is the slow part).
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
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B
from bench_lib import SEED, RESULTS_DIR, WEST_PATH, JAPAN_PATH, bootstrap_auroc

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'STIXGeneral',
                   'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8,
})
HS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'bmib_hs', 'figures')
os.makedirs(HS_DIR, exist_ok=True)

FAMILY = {'CA125 rule': '#7f7f7f', 'CPH-I': '#d62728', 'ROMA': '#d62728',
          'Scorecard': '#2ca02c', 'Raw LR': '#2ca02c',
          'CatBoost': '#ff7f0e', 'XGBoost': '#ff7f0e',
          'Random Forest': '#ff7f0e'}


def boot_row(y, p):
    boot = bootstrap_auroc(y, p, n_boot=2000)
    return float(np.mean(boot)), float(np.percentile(boot, 2.5)), \
        float(np.percentile(boot, 97.5))


def main():
    np.random.seed(SEED)
    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(WEST_PATH, med)
    X_j, y_j = B.load_external(JAPAN_PATH, med)
    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)

    raw_w = pd.read_csv(WEST_PATH)
    cc = raw_w['he4'].notna().values
    X_cc, y_cc = X_w[cc], y_w[cc]

    # ---------- Setting B (nested-CV tuned) ensembles ----------
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

    # ---------- complete-case + tuned + default AUROCs ----------
    order = ['CA125 rule', 'ROMA', 'CPH-I', 'Raw LR', 'Scorecard',
             'Random Forest', 'XGBoost', 'CatBoost']
    cc_rows, tuned_rows, default_rows = [], [], []
    for key in order:
        if key in tuned:
            p_t = tuned[key].predict_proba(X_cc)[:, 1]
            p_d = zoo[key](X_cc)
            v, lo, hi = boot_row(y_cc, p_t)
            tuned_rows.append((key, 'West CC', v, lo, hi))
            v, lo, hi = boot_row(y_cc, p_d)
            default_rows.append((key, 'West CC', v, lo, hi))
        else:
            p = zoo[key](X_cc)
            v, lo, hi = boot_row(y_cc, p)
            cc_rows.append((key, 'West CC', v, lo, hi))
        for label, X, y in [('West', X_w, y_w), ('Japan', X_j, y_j)]:
            if key in tuned:
                v, lo, hi = boot_row(y, tuned[key].predict_proba(X)[:, 1])
                tuned_rows.append((key, label, v, lo, hi))
                v, lo, hi = boot_row(y, zoo[key](X))
                default_rows.append((key, label, v, lo, hi))
            else:
                v, lo, hi = boot_row(y, zoo[key](X))
                cc_rows.append((key, label, v, lo, hi))
    pd.DataFrame(cc_rows, columns=['Model', 'Cohort', 'AUROC', 'CI low',
                                   'CI high']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_complete_case.csv'), index=False)
    pd.DataFrame(tuned_rows, columns=['Model', 'Cohort', 'AUROC', 'CI low',
                                      'CI high']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_tuned_ensembles.csv'), index=False)
    pd.DataFrame(default_rows, columns=['Model', 'Cohort', 'AUROC', 'CI low',
                                        'CI high']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_default_ensembles.csv'), index=False)

    # ---------- MICE sensitivity (all eight models, Setting B trees) ----------
    feat = raw_w[['ca125', 'he4', 'age', 'menopause']].copy()
    imp = IterativeImputer(max_iter=10, random_state=42)
    X_mice = imp.fit_transform(feat)
    X_w_mice = X_w.copy()
    X_w_mice['HE4'] = X_mice[:, 1]
    mice_rows = []
    for key in ['Scorecard', 'Raw LR', 'ROMA', 'CPH-I', 'CA125 rule',
                'XGBoost', 'CatBoost', 'Random Forest']:
        p = tuned[key].predict_proba(X_w_mice)[:, 1] if key in tuned \
            else zoo[key](X_w_mice)
        v, lo, hi = boot_row(y_w, p)
        mice_rows.append((key, v, lo, hi))
    pd.DataFrame(mice_rows, columns=['Model', 'AUROC', 'CI low', 'CI high']
                 ).to_csv(os.path.join(RESULTS_DIR, 'bmib_mice.csv'),
                          index=False)

    # ---------- recalibrated harms for all five Q2 models ----------
    thresholds = np.linspace(0.01, 0.60, 120)
    weights = [1, 5, 10, 50]

    def min_harm(p, ws):
        out = {}
        for w in ws:
            best = 1e9
            for t in thresholds:
                r = (p >= t).astype(int)
                missed = float(((r == 0) & (y_w == 1)).mean() * 100)
                unnec = float(((r == 1) & (y_w == 0)).mean() * 100)
                h = w * missed + unnec
                if h < best:
                    best = h
            out[w] = round(best, 1)
        return out

    def recalib(p):
        m = LogisticRegression().fit(np.asarray(p).reshape(-1, 1), y_w)
        return m.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]

    q2_probs = {
        'Scorecard': zoo['Scorecard'](X_w),
        'Logistic reg.': zoo['Raw LR'](X_w),
        'ROMA': zoo['ROMA'](X_w) / 100.0,
        'CPH-I': zoo['CPH-I'](X_w),
        'XGBoost': zoo['XGBoost'](X_w),
    }
    harm_rows = []
    for name, raw_p in q2_probs.items():
        harms = min_harm(recalib(raw_p), weights)
        harm_rows.append((name + ' (recalibrated)', harms[1], harms[5],
                          harms[10], harms[50]))
    pd.DataFrame(harm_rows, columns=['Model', 'w=1', 'w=5', 'w=10', 'w=50']
                 ).to_csv(os.path.join(RESULTS_DIR,
                                       'bmib_recalibrated_harms.csv'),
                          index=False)

    # ---------- probability-scale calibration ----------
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def ece(y, p, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        acc, counts = 0.0, 0
        for i in range(n_bins):
            m = (p > bins[i]) & (p <= bins[i + 1])
            if m.sum() > 0:
                acc += abs(float(y[m].mean()) - float(p[m].mean())) * m.sum()
                counts += m.sum()
        return acc / counts if counts else np.nan

    s_w = zoo['Scorecard'](X_w)
    recal_s = LogisticRegression().fit(s_w.reshape(-1, 1), y_w)
    probs = {
        'Scorecard (recalibrated)':
            recal_s.predict_proba(s_w.reshape(-1, 1))[:, 1],
        'Logistic reg.': zoo['Raw LR'](X_w),
        'ROMA': zoo['ROMA'](X_w) / 100.0,
        'CPH-I': zoo['CPH-I'](X_w),
    }
    cal_rows = []
    for name, p in probs.items():
        m = LogisticRegression().fit(logit(p).reshape(-1, 1), y_w)
        cal_rows.append((name, float(m.coef_[0][0]),
                         float(m.intercept_[0]), ece(y_w, p)))
    pd.DataFrame(cal_rows, columns=['Model', 'Cal slope', 'Cal intercept',
                                    'ECE']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_calibration_slope.csv'), index=False)

    # ---------- frozen calibration on the complete cases (n=100) ----------
    s_tr = zoo['Scorecard'](X_tr)
    frozen_map = LogisticRegression().fit(s_tr.reshape(-1, 1), y_tr)
    cc_probs = {
        'Scorecard (frozen map)':
            frozen_map.predict_proba(zoo['Scorecard'](X_cc).reshape(-1, 1))[:, 1],
        'Logistic reg.': zoo['Raw LR'](X_cc),
        'ROMA': zoo['ROMA'](X_cc) / 100.0,
        'CPH-I': zoo['CPH-I'](X_cc),
    }
    cc_cal_rows = []
    for name, p in cc_probs.items():
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        b = float(np.mean((y_cc - p) ** 2))
        m = LogisticRegression().fit(logit(p).reshape(-1, 1), y_cc)
        citl = float(np.log(np.mean(p) / (1 - np.mean(p)))
                     - np.log(np.mean(y_cc) / (1 - np.mean(y_cc))))
        cc_cal_rows.append((name, round(b, 4), round(float(m.coef_[0][0]), 3),
                            round(float(m.intercept_[0]), 3),
                            round(ece(y_cc, p), 4), round(citl, 3)))
    pd.DataFrame(cc_cal_rows, columns=['Model', 'Brier', 'Cal slope',
                                       'Cal intercept', 'ECE', 'CITL']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_cc_calibration.csv'), index=False)

    # ---------- complete-case operating points (n=100) ----------
    def op_row(name, pred):
        pred = np.asarray(pred)
        tp = float(((pred == 1) & (y_cc == 1)).sum())
        fp = float(((pred == 1) & (y_cc == 0)).sum())
        fn = float(((pred == 0) & (y_cc == 1)).sum())
        tn = float(((pred == 0) & (y_cc == 0)).sum())
        return (name, round(tp / (tp + fn), 3), round(tn / (tn + fp), 3),
                int(fn), int(tp + fn), int(fp), int(tn + fp))

    roma_cc = zoo['ROMA'](X_cc)
    roma_cc_rule = np.where(X_cc['Is_Postmenopausal'].values == 1,
                            roma_cc >= B.ROMA_CUT_POST,
                            roma_cc >= B.ROMA_CUT_PRE)
    op_rows = [
        op_row('Scorecard >= +1', zoo['Scorecard'](X_cc) >= 1),
        op_row('ROMA cutpoints', roma_cc_rule),
        op_row('CPH-I >= 7%', zoo['CPH-I'](X_cc) * 100 >= B.CPHI_CUT),
        op_row('CA125 >= 35 U/mL', X_cc['CA125'].values >= B.CA125_CUT),
    ]
    pd.DataFrame(op_rows, columns=['Strategy', 'Sensitivity', 'Specificity',
                                   'Missed (N)', 'Cancers (N)',
                                   'Unnecessary (N)', 'Benign (N)']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_cc_operating.csv'), index=False)

    # ---------- Japan false-positive rates among the 150 healthy women ------
    healthy = y_j == 0
    s_j = zoo['Scorecard'](X_j)
    roma_j = zoo['ROMA'](X_j)
    roma_j_rule = np.where(X_j['Is_Postmenopausal'].values == 1,
                           roma_j >= B.ROMA_CUT_POST,
                           roma_j >= B.ROMA_CUT_PRE)
    fpr_rows = []
    for name, pred in [
            ('Scorecard >= +1', s_j >= 1),
            ('ROMA cutpoints', roma_j_rule),
            ('CPH-I >= 7%', zoo['CPH-I'](X_j) * 100 >= B.CPHI_CUT),
            ('CA125 >= 35 U/mL', X_j['CA125'].values >= B.CA125_CUT)]:
        fp = int(((np.asarray(pred) == 1) & healthy).sum())
        fpr_rows.append((name, fp, round(fp / float(healthy.sum()) * 100, 1)))
    pd.DataFrame(fpr_rows, columns=['Strategy', 'False positives (N)',
                                    'FPR (%)']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_japan_fpr.csv'), index=False)

    # ---------- Japan descriptives (lowest-bin share and marker medians) ----
    lo_ca = X_j['CA125'].values < 24.3
    lo_he = X_j['HE4'].values < 45.8
    both_low = lo_ca & lo_he
    desc_rows = [
        ('Healthy in both lowest bins (%)',
         round(float(np.mean(both_low[healthy])) * 100, 1)),
        ('Cancer in both lowest bins (%)',
         round(float(np.mean(both_low[~healthy])) * 100, 1)),
        ('Healthy CA125 median (U/mL)', round(float(X_j.loc[healthy, 'CA125'].median()), 1)),
        ('Healthy HE4 median (pmol/L)', round(float(X_j.loc[healthy, 'HE4'].median()), 1)),
        ('West China CA125 median (U/mL)', round(float(X_w['CA125'].median()), 1)),
        ('West China HE4 median (pmol/L)', round(float(X_w['HE4'].median()), 1)),
    ]
    pd.DataFrame(desc_rows, columns=['Quantity', 'Value']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_japan_descriptives.csv'), index=False)

    # ---------- no-HE4 refits (3-feature models, unchanged elsewhere) -------
    feats3 = ['CA125', 'Age', 'Is_Postmenopausal']
    X_tr3, X_w3 = X_tr[feats3], X_w[feats3]
    sc3 = B.Scorecard(n_bins=3).fit(X_tr3, y_tr)
    lr3 = B.RawLR().fit(X_tr3, y_tr)
    qt3 = B.QuantileTransformer(n_quantiles=100, output_distribution='uniform',
                                random_state=SEED)
    xgb3 = XGBClassifier(random_state=SEED, eval_metric='logloss').fit(
        qt3.fit_transform(X_tr3), y_tr)
    nohe4_rows = [
        ('Scorecard', round(float(B.auroc(y_w, sc3.predict_score(X_w3))), 3)),
        ('Logistic reg.', round(float(B.auroc(y_w, lr3.predict_proba(X_w3))), 3)),
        ('XGBoost', round(float(B.auroc(y_w, xgb3.predict_proba(
            qt3.transform(X_w3))[:, 1])), 3)),
    ]
    pd.DataFrame(nohe4_rows, columns=['Model', 'AUROC (no HE4)']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_no_he4.csv'), index=False)

    # ---------- in-sample fits vs CV (Section III-E) ------------------------
    in_rows = []
    for key in ['XGBoost', 'CatBoost', 'Random Forest']:
        insample = float(B.auroc(y_tr, zoo[key](X_tr)))
        in_rows.append((key, round(insample, 3)))
    pd.DataFrame(in_rows, columns=['Model', 'In-sample AUROC']).to_csv(
        os.path.join(RESULTS_DIR, 'bmib_insample.csv'), index=False)

    # ---------- complete-case versus excluded outcome balance ----------
    from scipy.stats import fisher_exact
    n_cc_ca = int(y_cc.sum())
    n_cc_be = int(len(y_cc) - n_cc_ca)
    n_ex_ca = int(y_w.sum()) - n_cc_ca
    n_ex_be = int((y_w == 0).sum()) - n_cc_be
    _, p_fisher = fisher_exact([[n_cc_ca, n_cc_be], [n_ex_ca, n_ex_be]])
    bal_rows = [
        ('Complete cases (measured HE4)', n_cc_ca, n_cc_be,
         round(100 * n_cc_ca / len(y_cc), 1),
         round(float(X_cc['Age'].mean()), 1),
         round(float(X_cc['CA125'].median()), 1),
         round(float((X_cc['Is_Postmenopausal'] == 1).mean()) * 100, 1)),
        ('Excluded (imputed HE4)', n_ex_ca, n_ex_be,
         round(100 * n_ex_ca / (n_ex_ca + n_ex_be), 1),
         round(float(X_w.loc[~cc, 'Age'].mean()), 1),
         round(float(X_w.loc[~cc, 'CA125'].median()), 1),
         round(float((X_w.loc[~cc, 'Is_Postmenopausal'] == 1).mean()) * 100, 1)),
    ]
    pd.DataFrame(bal_rows, columns=['Subset', 'Cancers (N)', 'Benign (N)',
                                    'Cancer prevalence (%)', 'Age mean',
                                    'CA125 median', 'Postmenopausal (%)']
                 ).to_csv(os.path.join(RESULTS_DIR, 'bmib_cc_balance.csv'),
                          index=False)
    print(f'Fisher exact test (outcome vs HE4 measured): p = {p_fisher:.4f}')

    # ---------- West China paired scorecard-minus-comparator differences --
    def paired_diffs(X, y, out_path):
        rng = np.random.RandomState(SEED)
        n = len(y)
        s_sc = zoo['Scorecard'](X)
        diff_rows = []
        for key in ['Raw LR', 'ROMA', 'CPH-I', 'XGBoost', 'CatBoost',
                    'Random Forest']:
            p_c = tuned[key].predict_proba(X)[:, 1] if key in tuned \
                else zoo[key](X)
            diffs = []
            for _ in range(2000):
                idx = rng.randint(0, n, n)
                diffs.append(float(B.auroc(y[idx], s_sc[idx])
                                  - B.auroc(y[idx], p_c[idx])))
            diffs = np.array(diffs)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            p = 2 * min(float((diffs <= 0).mean()),
                        float((diffs >= 0).mean()))
            diff_rows.append((key, round(float(diffs.mean()), 4),
                              round(lo, 4), round(hi, 4), round(p, 4)))
        pd.DataFrame(diff_rows, columns=['Comparator', 'dAUROC', 'CI low',
                                         'CI high', 'p']).to_csv(
            os.path.join(RESULTS_DIR, out_path), index=False)

    paired_diffs(X_cc, y_cc, 'bmib_cc_paired.csv')
    paired_diffs(X_w, y_w, 'bmib_west_paired.csv')

    # ---------- Figure 1: complete-case forest plot ----------
    rows = [('CA125 rule', 'CA125 rule'), ('CPH-I', 'CPH-I'), ('ROMA', 'ROMA'),
            ('Scorecard', 'Scorecard'), ('Logistic regression', 'Raw LR'),
            ('CatBoost', 'CatBoost'), ('XGBoost', 'XGBoost'),
            ('Random forest', 'Random Forest')]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ys = np.arange(len(rows))
    for y, (disp, key) in zip(ys, rows):
        if key in tuned:
            p = tuned[key].predict_proba(X_cc)[:, 1]
        else:
            p = zoo[key](X_cc)
        v, lo, hi = boot_row(y_cc, p)
        ax.plot([lo, hi], [y, y], color='#333333', lw=1.4, zorder=3)
        ax.plot([lo, lo], [y - 0.13, y + 0.13], color='#333333', lw=1.2,
                zorder=3)
        ax.plot([hi, hi], [y - 0.13, y + 0.13], color='#333333', lw=1.2,
                zorder=3)
        ax.scatter([v], [y], s=44, color=FAMILY[key], edgecolor='black',
                   lw=0.7, zorder=4)
        ax.text(1.012, y, f'{v:.2f}', va='center', ha='left', fontsize=8,
                color='#222222', zorder=5)
    ax.axvline(0.5, color='grey', ls='--', lw=0.9, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    for lab, (disp, key) in zip(ax.get_yticklabels(), rows):
        lab.set_color(FAMILY[key])
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlim(0.46, 1.08)
    ax.set_xlabel('Area under the ROC curve', fontsize=8,
                  linespacing=1.3, labelpad=4)
    ax.grid(axis='x', alpha=0.25, ls='--', zorder=0)
    fig.subplots_adjust(left=0.38, right=0.97, top=0.97, bottom=0.16)
    fig.savefig(os.path.join(HS_DIR, 'fig1_auroc.png'),
                bbox_inches='tight', pad_inches=0.06, dpi=300)
    plt.close(fig)

    # ---------- Figure 2: decision curve analysis ----------
    def nb_curve(y, p, thresholds_):
        n = len(y)
        prev = float(np.mean(y))
        out = []
        for t in thresholds_:
            r = (p >= t).astype(int)
            tp = float(((r == 1) & (y == 1)).sum())
            fp = float(((r == 1) & (y == 0)).sum())
            out.append((tp - fp * t / (1 - t)) / n)
        return np.array(out)

    p_roma = zoo['ROMA'](X_w) / 100.0
    p_cphi = zoo['CPH-I'](X_w)
    p_lr = zoo['Raw LR'](X_w)
    recal_roma = LogisticRegression().fit(p_roma.reshape(-1, 1), y_w)
    recal_cphi = LogisticRegression().fit(p_cphi.reshape(-1, 1), y_w)
    recal_lr = LogisticRegression().fit(p_lr.reshape(-1, 1), y_w)
    dca_probs = {
        'Scorecard': recal_s.predict_proba(s_w.reshape(-1, 1))[:, 1],
        'Logistic reg.': recal_lr.predict_proba(p_lr.reshape(-1, 1))[:, 1],
        'ROMA (recal.)': recal_roma.predict_proba(p_roma.reshape(-1, 1))[:, 1],
        'CPH-I (recal.)': recal_cphi.predict_proba(p_cphi.reshape(-1, 1))[:, 1],
    }
    colors = {'Scorecard': '#2ca02c', 'Logistic reg.': '#8c564b',
              'ROMA (recal.)': '#d62728', 'CPH-I (recal.)': '#17becf'}
    styles = {'Scorecard': '-', 'Logistic reg.': '--',
              'ROMA (recal.)': '-.', 'CPH-I (recal.)': ':'}
    ts = np.linspace(0.01, 0.99, 200)
    prev = float(np.mean(y_w))
    fig, ax = plt.subplots(figsize=(3.5, 2.95))
    ax.axvspan(7, 30, color='#f2f2f2', zorder=0)
    for mname, p in dca_probs.items():
        nb = nb_curve(y_w, p, ts)
        ax.plot(ts * 100, nb, ls=styles[mname], lw=2.0, color=colors[mname],
                zorder=3, label=mname)
    nb_all = prev - (1 - prev) * ts / (1 - ts)
    ax.plot(ts * 100, nb_all, ls='--', lw=1.3, color='#555555', zorder=2,
            label='Treat all')
    ax.axhline(0, color='#555555', lw=0.8, zorder=1, label='Treat none')
    ax.set_xlabel('Threshold probability (%)', fontsize=8)
    ax.set_ylabel('Net benefit', fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.02, 0.52)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, ls='--', zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
              fontsize=8, frameon=False, columnspacing=1.8,
              handletextpad=0.6, handlelength=1.6)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.32)
    fig.savefig(os.path.join(HS_DIR, 'fig2_cons.png'),
                bbox_inches='tight', pad_inches=0.08, dpi=300)
    plt.close(fig)

    # ---------- Figure 3: regime map (ensemble average vs scorecard) -------
    sim = pd.read_csv(os.path.join(RESULTS_DIR, 'simulation_study.csv'))
    sub = sim[(sim['noise'] == 0.5) &
              (sim.Model.isin(['XGBoost', 'CatBoost', 'Random Forest',
                               'Scorecard']))]
    piv = sub.pivot_table(index=['n_train', 'shift', 'drift'],
                          columns='Model', values='Test AUROC mean')
    piv['ens'] = (piv['XGBoost'] + piv['CatBoost'] + piv['Random Forest']) / 3
    norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.10)
    fig, axes = plt.subplots(1, 2, figsize=(6.16, 2.68))
    for ax, drift, lab in zip(axes, [0.0, 1.0],
                              ['(a) Stationary structure',
                               '(b) Concept drift']):
        s = piv.xs(drift, level='drift')
        diff_df = (s['ens'] - s['Scorecard']).unstack('shift')
        diff = diff_df.T.values
        nr, nc = diff.shape
        X, Y = np.meshgrid(np.arange(nc) + 1, np.arange(nr) + 1)
        ax.pcolormesh(X, Y, diff, cmap='RdBu_r', norm=norm,
                      edgecolors='white', linewidth=1.2, zorder=2)
        for i in range(nr):
            for j in range(nc):
                ax.text(j + 1, i + 1, f'{diff[i, j]:+.2f}', ha='center',
                        va='center', fontsize=8, zorder=4,
                        path_effects=[pe.withStroke(linewidth=1.4,
                                                    foreground='white')])
        ax.set_title(lab, fontsize=9, fontweight='bold', pad=6)
        ax.set_xlim(0.6, nc + 1.4)
        ax.set_ylim(0.6, nr + 1.4)
        ax.set_xticks(np.arange(nc) + 1)
        ax.set_xticklabels(['100', '200', '400', '800'], fontsize=8)
        ax.set_yticks(np.arange(nr) + 1)
        ax.set_yticklabels(['0', '0.5', '1'], fontsize=8)
        ax.tick_params(length=0)
        ax.set_xlabel('Training samples', fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('Distribution shift', fontsize=8)
        else:
            ax.tick_params(axis='y', labelleft=False)
    cax = fig.add_axes([0.30, 0.03, 0.40, 0.06])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                      cax=cax, orientation='horizontal',
                      ticks=[-0.10, -0.05, 0.0, 0.05, 0.10])
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)
    cb.set_label('Favors scorecard     Ensemble average minus scorecard AUROC'
                 '     Favors ensembles', fontsize=8, labelpad=5)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.22, top=0.86,
                        wspace=0.22)
    fig.savefig(os.path.join(HS_DIR, 'fig3_regime_col.png'),
                bbox_inches='tight', pad_inches=0.06, dpi=300)
    plt.close(fig)

    print('BIBM artifacts written to', RESULTS_DIR, 'and', HS_DIR)


if __name__ == '__main__':
    main()
