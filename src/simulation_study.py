"""
Synthetic simulation study: when does model complexity pay off?

A generative model calibrated to the Chinese training cohort: three
biomarkers (CA125, HE4 on log scale; age), an outcome with a strong linear
signal plus a multiplicative interaction and a smooth nonlinear term, so
flexible models have learnable structure beyond the linear part. Each
scenario is defined by:
    n_train  - training set size
    shift    - distribution shift of test biomarkers (mean shift in SD units
               plus variance inflation)
    noise    - multiplicative Gaussian measurement noise on test biomarkers
    R        - replicates per cell

Five models are fit per replicate (scorecard, raw LR, XGBoost, CatBoost,
Random Forest) and evaluated on a large shifted test set. The output is the
test AUROC per model per cell, from which the complexity-vs-transportability
regime map is drawn.

Outputs:
    results/simulation_study.csv
    figures/fig_simulation_study.png
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lib import (SEED, RESULTS_DIR, FIGURES_DIR, MODEL_COLORS, MODEL_LABELS,
                       auroc, Scorecard, RawLR)

RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR

R = 20
N_TEST = 4000
SIZES = [100, 200, 400, 800]
SHIFTS = [0.0, 0.5, 1.0]
NOISES = [0.0, 0.5]
DRIFTS = [0.0, 1.0]


def calibrate_generator(X_tr, y_tr):
    """Estimate class-conditional biomarker parameters from the training data."""
    out = {}
    for f, logscale in [('CA125', True), ('HE4', True), ('Age', False)]:
        v = X_tr[f].values.astype(float)
        v = np.log(v) if logscale else v
        means, sds = [], []
        for cls in (0, 1):
            m = v[y_tr == cls]
            means.append(float(np.nanmean(m)))
            sds.append(float(np.nanstd(m)) + 1e-6)
        out[f] = {'mean': np.array(means), 'sd': np.array(sds)}
    return out


def generate_data(par, n, shift, noise, drift, rng, prevalence=0.35):
    """Generate features and labels.

    Base features are class-mixture lognormals calibrated to the training
    cohort; the label is Bernoulli with logit = linear(z) + gamma*z1*z2 +
    nu*sin(2*z1), where z are features standardized on the training marginals
    (frozen). Distribution shift shrinks the class-conditional biomarker means
    toward the pooled mean by factor `shift` and inflates variances, so
    discrimination decreases monotonically with shift. Concept drift removes
    the interaction/nonlinear term from the test-side outcome model by factor
    `drift` (0 = stationary, 1 = the structure a flexible model learned is no
    longer present). Measurement noise is additive Gaussian on the log scale
    (multiplicative on the raw scale).
    """
    z1_c = par['CA125']['mean']   # benign, cancer (log CA125)
    z2_c = par['HE4']['mean']
    z3_c = par['Age']['mean']
    z1_s = par['CA125']['sd']
    z2_s = par['HE4']['sd']
    z3_s = par['Age']['sd']
    z_c = {'CA125': (z1_c, z1_s), 'HE4': (z2_c, z2_s), 'Age': (z3_c, z3_s)}

    def sample_base(n, shrink, noise):
        """Features sampled from class mixtures; shrink collapses the class
        means toward the pooled mean (separation shrinkage); noise is additive
        Gaussian on the log/raw scale of CA125/HE4."""
        y_ = rng.binomial(1, prevalence, n)
        z = {}
        for f, (means, sds) in z_c.items():
            pooled = prevalence * means[1] + (1 - prevalence) * means[0]
            m = pooled + (1 - shrink) * (means - pooled)
            z[f] = np.array([rng.normal(m[y], sds[y] * (1 + shrink)) for y in y_])
        if noise > 0:
            for f in ['CA125', 'HE4']:
                z[f] += rng.normal(0, noise * np.std(z[f]), n)
        return y_, z

    # training features: clean, no shrinkage, no noise
    y_tr_, z_tr = sample_base(n, 0.0, 0.0)
    train_mean = {f: np.mean(z_tr[f]) for f in z_tr}
    train_sd = {f: np.std(z_tr[f]) + 1e-9 for f in z_tr}

    def zscore(z):
        return {f: (z[f] - train_mean[f]) / train_sd[f] for f in z}

    def eta_of(zs, drift):
        s1, s2, s3 = zs['CA125'], zs['HE4'], zs['Age']
        return (-0.65 + 1.8 * s1 + 1.0 * s2 + 0.5 * s3
                + (1 - drift) * (1.6 * s1 * s2 + 1.2 * np.sin(1.7 * s1)))

    zs_tr = zscore(z_tr)
    p = 1.0 / (1.0 + np.exp(-eta_of(zs_tr, 0.0)))
    y = rng.binomial(1, p).astype(int)

    def to_df(z):
        return pd.DataFrame({
            'CA125': np.exp(z['CA125']), 'HE4': np.exp(z['HE4']),
            'Age': z['Age'],
            'Is_Postmenopausal': (z['Age'] > train_mean['Age']).astype(float),
        })

    X = to_df(z_tr)

    # test set: shrunk (shift) biomarkers + noise + concept drift; labels from
    # the drifted eta applied to frozen-standardized features
    y_t_, z_te = sample_base(N_TEST, shift, noise)
    zs_te = zscore(z_te)
    p_t = 1.0 / (1.0 + np.exp(-eta_of(zs_te, drift)))
    y_t = rng.binomial(1, p_t).astype(int)
    X_t = to_df(z_te)
    return X, y, X_t, y_t


def fit_predict(model, X_tr, y_tr, X_te):
    if model == 'Scorecard':
        m = Scorecard(n_bins=3).fit(X_tr, y_tr)
        return m.predict_score(X_te)
    if model == 'Raw LR':
        m = RawLR().fit(X_tr, y_tr)
        return m.predict_proba(X_te)
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import QuantileTransformer
    q = QuantileTransformer(n_quantiles=min(100, len(X_tr)),
                            output_distribution='uniform', random_state=SEED)
    Xn_tr = q.fit_transform(X_tr)
    Xn_te = pd.DataFrame(q.transform(X_te), columns=X_te.columns)
    if model == 'XGBoost':
        m = XGBClassifier(random_state=SEED, eval_metric='logloss')
    elif model == 'CatBoost':
        m = CatBoostClassifier(random_seed=SEED, verbose=0, iterations=500)
    else:
        m = RandomForestClassifier(random_state=SEED)
    m.fit(Xn_tr, y_tr)
    return m.predict_proba(Xn_te)[:, 1]


def main():
    import bench_lib as B
    X_tr, y_tr = B.load_train_imputed()
    par = calibrate_generator(X_tr, y_tr)

    models = ['Scorecard', 'Raw LR', 'XGBoost', 'CatBoost', 'Random Forest']
    rows = []
    for n in SIZES:
        for shift in SHIFTS:
            for noise in NOISES:
                for drift in DRIFTS:
                    res = {m: [] for m in models}
                    for rep in range(R):
                        rng = np.random.RandomState(int(100000 * drift)
                                                    + 1000 * (n // 100)
                                                    + 100 * int(shift * 10)
                                                    + 10 * int(noise * 10) + rep)
                        Xa, ya, Xb, yb = generate_data(par, n, shift, noise, drift, rng)
                        for m in models:
                            res[m].append(auroc(yb, fit_predict(m, Xa, ya, Xb)))
                    for m in models:
                        rows.append({'n_train': n, 'shift': shift, 'noise': noise,
                                     'drift': drift, 'Model': m,
                                     'Test AUROC mean': round(float(np.mean(res[m])), 4),
                                     'Test AUROC SD': round(float(np.std(res[m])), 4)})
    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(os.path.join(RESULTS, 'simulation_study.csv'), index=False)
    print("simulation_study.csv saved")

    # pivot: mean AUROC at drift = 0, noise = 0
    sub = sim_df[(sim_df['noise'] == 0.0) & (sim_df['drift'] == 0.0)]
    piv = sub.pivot_table(index=['n_train', 'shift'], columns='Model',
                          values='Test AUROC mean').round(4)
    print("noise=0, drift=0:")
    print(piv.to_string())
    sub = sim_df[(sim_df['noise'] == 0.5) & (sim_df['drift'] == 1.0)]
    piv = sub.pivot_table(index=['n_train', 'shift'], columns='Model',
                          values='Test AUROC mean').round(4)
    print("noise=0.5, drift=1:")
    print(piv.to_string())

    # Table 4 support: factor decomposition of the model gap
    piv = sim_df.pivot_table(index=['n_train', 'shift', 'noise', 'drift'],
                             columns='Model', values='Test AUROC mean')
    piv['gap_xgb'] = piv['XGBoost'] - piv['Scorecard']
    piv['gap_cat'] = piv['CatBoost'] - piv['Scorecard']
    g = piv.reset_index()
    rows = []
    for factor in ['drift', 'shift', 'n_train', 'noise']:
        m = g.groupby(factor)['gap_xgb'].mean()
        rows.append({'factor': factor, 'levels': '..'.join(str(x) for x in m.index),
                     'gap_min': round(m.min(), 4), 'gap_max': round(m.max(), 4),
                     'spread': round(m.max() - m.min(), 4)})
    fdf = pd.DataFrame(rows)
    fdf.to_csv(os.path.join(RESULTS, 'factor_decomposition.csv'), index=False)
    print("\nfactor_decomposition.csv saved")
    print(fdf.to_string(index=False))

    # figure: regime map (heatmap of XGBoost minus scorecard AUROC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, drift in zip(axes, [0.0, 1.0]):
        sub = sim_df[(sim_df['noise'] == 0.5) & (sim_df['drift'] == drift)]
        xgb = sub[sub['Model'] == 'XGBoost'].pivot(index='shift', columns='n_train',
                                                   values='Test AUROC mean').values
        sc = sub[sub['Model'] == 'Scorecard'].pivot(index='shift', columns='n_train',
                                                    values='Test AUROC mean').values
        diff = xgb - sc
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.10, vmax=0.10, aspect='auto')
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES)
        ax.set_yticks(range(len(SHIFTS)))
        ax.set_yticklabels(SHIFTS)
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                ax.text(j, i, f'{diff[i, j]:+.2f}', ha='center', va='center', fontsize=9)
        ax.set_xlabel('Training samples')
        ax.set_ylabel('Distribution shift (separation shrinkage)')
        ax.set_title(f'XGBoost minus scorecard AUROC, noise 50%'
                     + ('\nstationary structure' if drift == 0 else '\nconcept drift'),
                     fontweight='bold')
        fig.colorbar(im, ax=ax, label='\u0394 AUROC')
    fig.suptitle('Regime map: where does model complexity pay off? (synthetic study, '
                 '20 replicates)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'fig_simulation_study.png'),
                facecolor='white', dpi=200)
    plt.close()
    print("fig_simulation_study.png saved")


if __name__ == '__main__':
    main()
