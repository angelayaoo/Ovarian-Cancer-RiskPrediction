"""
Clinical comparators benchmark.

Adds three comparators to the main benchmark and evaluates them under the
same Monte Carlo noise grid:
    CPH-I       - published Copenhagen index (Van Gorp 2011): CA125 + HE4 + age
    CA125 rule  - single-marker rule (CA125 >= 35 U/mL)
    Raw LR      - standardized 4-feature logistic regression (unbinned)

Also computes bootstrap pairwise AUROC comparisons of the scorecard against
every clinical comparator on the West China cohort, and published-cutoff
operating characteristics for ROMA / CPH-I / CA125 rule.

Outputs:
    results/clinical_comparators_noise.csv
    results/clinical_comparators_pairwise.csv
    results/clinical_rules_operating.csv
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = 'results'
SEED = 42
N_PERT = 100
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def main():
    import bench_lib as B
    np.random.seed(SEED)

    X_tr, y_tr = B.load_train_imputed()
    med = {c: float(X_tr[c].median()) for c in X_tr.columns}
    X_w, y_w = B.load_external(B.WEST_PATH, med)
    X_j, y_j = B.load_external(B.JAPAN_PATH, med)

    zoo, sc, lr, qt = B.fit_model_zoo(X_tr, y_tr)
    models = ['Scorecard', 'Raw LR', 'ROMA', 'CPH-I', 'CA125 rule',
              'XGBoost', 'CatBoost', 'Random Forest']

    def make_noisy(X, noise):
        Xn = X.copy()
        rng = np.random.RandomState()
        for f in ['CA125', 'HE4']:
            Xn[f] = np.maximum(X[f].values * rng.normal(1.0, noise, len(X)), 1e-5)
        return Xn

    # 1. Noise grid for all models, all cohorts
    rows = []
    cohorts = {'Training (in-sample)': (X_tr, y_tr),
               'West China': (X_w, y_w),
               'Japan': (X_j, y_j)}
    for cname, (Xc, yc) in cohorts.items():
        for noise in NOISE_LEVELS:
            aucs = {m: [] for m in models}
            for _ in range(N_PERT):
                Xn = make_noisy(Xc, noise)
                for m in models:
                    aucs[m].append(B.auroc(yc, zoo[m](Xn)))
            rows.append({'Cohort': cname, 'Noise': f'{int(noise*100)}%',
                         **{m: round(float(np.mean(aucs[m])), 4) for m in models},
                         **{m + ' SD': round(float(np.std(aucs[m])), 4) for m in models}})
    noise_df = pd.DataFrame(rows)
    noise_df.to_csv(os.path.join(RESULTS, 'clinical_comparators_noise.csv'), index=False)
    print("clinical_comparators_noise.csv saved")
    print(noise_df[noise_df['Noise'] == '0%'].to_string(index=False))

    # 2. Bootstrap pairwise comparisons on West China (scorecard vs each)
    print("\nBootstrap pairwise AUROC comparisons (West China, 0% noise, 2000 resamples):")
    p_rows = []
    s_sc = zoo['Scorecard'](X_w)
    for m in models:
        if m == 'Scorecard':
            continue
        diffs = B.bootstrap_pairwise(y_w, s_sc, zoo[m](X_w), n_boot=2000)
        mean_d, lo, hi, pv = B.pairwise_summary(diffs)
        p_rows.append({'Comparator': m, 'dAUROC (scorecard - model)': round(mean_d, 4),
                       '95% CI low': round(lo, 4), '95% CI high': round(hi, 4),
                       'p': round(pv, 4)})
    pair_df = pd.DataFrame(p_rows)
    pair_df.to_csv(os.path.join(RESULTS, 'clinical_comparators_pairwise.csv'), index=False)
    print(pair_df.to_string(index=False))

    # 3. Published-cutoff operating characteristics (West China)
    print("\nPublished-cutoff operating characteristics (West China, 0% noise):")
    o_rows = []
    for m, cut_fn in [('CA125 rule', lambda X: B.compute_ca125_rule(X)),
                      ('ROMA', lambda X: np.where(
                          X['Is_Postmenopausal'].values == 1,
                          B.compute_roma(X) >= B.ROMA_CUT_POST,
                          B.compute_roma(X) >= B.ROMA_CUT_PRE).astype(float)),
                      ('CPH-I', lambda X: (B.compute_cphi(X) * 100 >= B.CPHI_CUT).astype(float))]:
        pred = cut_fn(X_w)
        tp = ((pred == 1) & (y_w == 1)).sum()
        fp = ((pred == 1) & (y_w == 0)).sum()
        fn = ((pred == 0) & (y_w == 1)).sum()
        tn = ((pred == 0) & (y_w == 0)).sum()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        o_rows.append({'Rule': m, 'Cutoff': {'CA125 rule': '>= 35 U/mL',
                                             'ROMA': '>= 13.1% (pre) / 27.7% (post)',
                                             'CPH-I': '>= 7%'}[m],
                       'Sensitivity': round(float(sens), 3),
                       'Specificity': round(float(spec), 3),
                       'PPV': round(float(ppv), 3), 'NPV': round(float(npv), 3),
                       'Referred (N)': int(tp + fp),
                       'Missed cancers (N)': int(fn),
                       'Unnecessary referrals (N)': int(fp)})
    op_df = pd.DataFrame(o_rows)
    op_df.to_csv(os.path.join(RESULTS, 'clinical_rules_operating.csv'), index=False)
    print(op_df.to_string(index=False))

    # 4. Bootstrap pairwise: CPH-I vs ROMA (head-to-head of the two fixed formulas)
    diffs = B.bootstrap_pairwise(y_w, zoo['CPH-I'](X_w), zoo['ROMA'](X_w), n_boot=2000)
    mean_d, lo, hi, pv = B.pairwise_summary(diffs)
    print(f"\nCPH-I - ROMA (West China): dAUROC {mean_d:+.4f} "
          f"[{lo:.4f}, {hi:.4f}], p = {pv:.4f}")


if __name__ == '__main__':
    main()
