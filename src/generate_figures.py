"""
Generate publication-quality figures for the paper:
1. Bootstrap pairwise comparison figure
2. Risk stratification table with clinical action colors
3. Model comparison bar chart (all 6 models, 3 cohorts)
4. Scorecard weight heatmap
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monte_carlo_benchmarks import parse_cohort_data, PreTrainedRiskSLIM, KNNImputer

DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "chinese_train_cleaned.csv")
WEST_PATH = os.path.join(DATA_DIR, "west_china_processed.csv")
JAPAN_PATH = os.path.join(DATA_DIR, "japan_processed.csv")
RESULTS_CSV = "results/monte_carlo_benchmarks.csv"
OUTPUT_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    def clean(v): return float(str(v).split(' ')[0])
    cols = [c for c in df.columns if c not in ['Cohort', 'Noise']]
    for c in cols: df[c] = df[c].apply(clean)
    df['Noise_Pct'] = df['Noise'].str.rstrip('%').astype(int)
    return df


# ═══════════════════════════════════════════════
# FIGURE 1: Bootstrap Pairwise Comparison
# ═══════════════════════════════════════════════
def fig1_bootstrap_pairwise():
    # Data from statistical_tests.py output
    comparisons = [
        ('RiskSLIM', 'XGBoost', 0.0385, 0.0000, '***'),
        ('RiskSLIM', 'Decision\nTree', 0.1643, 0.0000, '***'),
        ('RiskSLIM', 'ROMA', 0.0540, 0.0020, '**'),
        ('RiskSLIM', 'Random\nForest', 0.0280, 0.0100, '*'),
        ('RiskSLIM', 'CatBoost', 0.0226, 0.0310, '*'),
    ]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    models_b = [c[1] for c in comparisons]
    diffs = [c[2] for c in comparisons]
    pvals = [c[3] for c in comparisons]
    sigs = [c[4] for c in comparisons]

    colors = ['#1a9850' if d > 0 else '#d73027' for d in diffs]
    bars = ax.barh(range(len(models_b)), diffs, color=colors, height=0.5, edgecolor='white', linewidth=0.5)

    for i, (bar, d, p, s) in enumerate(zip(bars, diffs, pvals, sigs)):
        label = f'+{d:.3f} {s}' if d > 0 else f'{d:.3f} {s}'
        if p < 0.001:
            label += '\n(p<0.001)'
        elif p < 0.01:
            label += '\n(p<0.01)'
        else:
            label += f'\n(p={p:.3f})'
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(range(len(models_b)))
    ax.set_yticklabels(models_b, fontsize=10)
    ax.set_xlabel('Δ AUROC (RiskSLIM − Comparator)', fontsize=11)
    ax.set_title('Bootstrap Pairwise Comparison (West China, 0% Noise)', fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlim(-0.01, 0.22)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    # Legend
    green_patch = mpatches.Patch(color='#1a9850', label='RiskSLIM superior')
    ax.legend(handles=[green_patch], loc='lower right', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_bootstrap_pairwise.png'), facecolor='white')
    plt.close()
    print("  Saved: fig1_bootstrap_pairwise.png")


# ═══════════════════════════════════════════════
# FIGURE 2: Risk Stratification Table
# ═══════════════════════════════════════════════
def fig2_risk_stratification():
    tiers = [
        ('Very Low', '≤ −2', 157, 10.2, '#2166ac', 'Observe'),
        ('Low', '−1 to 0', 77, 40.3, '#92c5de', 'Ultrasound'),
        ('Low-Int', '1 – 2', 58, 93.1, '#f4a582', 'Refer GYN'),
        ('Int', '3 – 4', 45, 97.8, '#ca0020', 'Urgent GYN'),
        ('High-Int', '5 – 6', 6, 100.0, '#a50f15', 'Oncology'),
        ('High', '≥ 7', 37, 100.0, '#67000d', 'Surgery'),
    ]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis('off')

    col_labels = ['Risk Tier', 'Score', 'N', 'Cancer %', 'Clinical Action']
    col_widths = [0.18, 0.15, 0.12, 0.15, 0.40]
    rows = []
    cell_colors = []
    for tier, score, n, rate, color, action in tiers:
        rows.append([tier, score, str(n), f'{rate:.1f}%', action])
        cell_colors.append([color, color, color, color, color])

    table = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for i in range(len(rows)):
        for j in range(5):
            cell = table[(i+1, j)]
            cell.set_facecolor(cell_colors[i][j])
            # White text on dark backgrounds
            if any(c in cell_colors[i][j] for c in ['#2166ac', '#ca0020', '#a50f15', '#67000d']):
                cell.get_text().set_color('white')
                cell.get_text().set_fontweight('bold')

    # Header
    for j in range(5):
        cell = table[(0, j)]
        cell.set_facecolor('#333333')
        cell.get_text().set_color('white')
        cell.get_text().set_fontweight('bold')
        cell.set_fontsize(10)

    ax.set_title('Risk Stratification by Integer Score (West China, n=380)', fontweight='bold', fontsize=13, pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_risk_stratification.png'), facecolor='white')
    plt.close()
    print("  Saved: fig2_risk_stratification.png")


# ═══════════════════════════════════════════════
# FIGURE 3: Model Comparison Bar Chart (0% noise, all cohorts)
# ═══════════════════════════════════════════════
def fig3_model_comparison():
    df = load_data()
    sub = df[df['Noise_Pct'] == 0]
    cohorts = ['OG China (Primary)', 'West China', 'Japanese']
    models = ['ROMA Formula', 'XGBoost', 'CatBoost', 'Decision Tree', 'Random Forest', 'Optimized RiskSLIM']
    labels = ['ROMA', 'XGBoost', 'CatBoost', 'Decision\nTree', 'Random\nForest', 'RiskSLIM']
    colors = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#1f77b4', '#2ca02c']

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cohorts))
    width = 0.12

    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        vals = [sub[sub['Cohort'] == c][model].values[0] for c in cohorts]
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor='white', linewidth=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=5.5, rotation=90)

    ax.set_ylabel('AUROC', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['OG China\n(Internal)', 'West China\n(External)', 'Japan\n(External)'], fontsize=9)
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9, ncol=2)
    ax.set_title('Model Comparison at 0% Measurement Noise', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_model_comparison.png'), facecolor='white')
    plt.close()
    print("  Saved: fig3_model_comparison.png")


# ═══════════════════════════════════════════════
# FIGURE 4: Overfitting + Noise Degradation
# ═══════════════════════════════════════════════
def fig4_overfitting_noise():
    df = load_data()
    models = ['ROMA Formula', 'XGBoost', 'CatBoost', 'Decision Tree', 'Random Forest', 'Optimized RiskSLIM']
    labels = ['ROMA', 'XGBoost', 'CatBoost', 'Decision\nTree', 'Random\nForest', 'RiskSLIM']
    colors = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#1f77b4', '#2ca02c']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Subplot 1: Overfitting (Internal - External)
    int_vals = [df[(df['Cohort']=='OG China (Primary)')&(df['Noise_Pct']==0)][m].values[0] for m in models]
    ext_vals = [df[(df['Cohort']=='West China')&(df['Noise_Pct']==0)][m].values[0] for m in models]
    gaps = [i - e for i, e in zip(int_vals, ext_vals)]

    bar_colors1 = ['#d73027' if g > 0.02 else '#1a9850' for g in gaps]
    bars1 = ax1.barh(range(len(models)), gaps, color=bar_colors1, height=0.5, edgecolor='white')
    for bar, g, m in zip(bars1, gaps, labels):
        label = f'{g:+.3f}'
        x_pos = bar.get_width() + 0.003 if g > 0 else bar.get_width() - 0.02
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=9, fontweight='bold')
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel('Internal − External AUROC', fontsize=10)
    ax1.set_title('Generalization Gap', fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()

    # Subplot 2: Noise Degradation (West China, 0% → 30%)
    w0 = [df[(df['Cohort']=='West China')&(df['Noise_Pct']==0)][m].values[0] for m in models]
    w30 = [df[(df['Cohort']=='West China')&(df['Noise_Pct']==30)][m].values[0] for m in models]
    degrad = [w3 - w0 for w0, w3 in zip(w0, w30)]

    bar_colors2 = ['#d73027' if d < -0.06 else '#fc8d59' if d < -0.04 else '#1a9850' for d in degrad]
    bars2 = ax2.barh(range(len(models)), degrad, color=bar_colors2, height=0.5, edgecolor='white')
    for bar, d in zip(bars2, degrad):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{d:+.3f}', va='center', fontsize=9, fontweight='bold')
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel('AUROC Change (0% → 30% Noise)', fontsize=10)
    ax2.set_title('Noise Degradation (West China)', fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()

    fig.suptitle('Model Robustness: Generalization Gap vs. Lab Drift Sensitivity', fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_overfitting_noise.png'), facecolor='white')
    plt.close()
    print("  Saved: fig4_overfitting_noise.png")


# ═══════════════════════════════════════════════
# FIGURE 5: Scorecard Heatmap
# ═══════════════════════════════════════════════
def fig5_scorecard_heatmap():
    weights = np.array([
        [-1,  0, +2],   # CA125
        [-4, -1, +5],   # HE4
        [-2,  0, +2],   # Age
        [ 0,  0,  0],   # Menopause
    ])
    features = ['CA125', 'HE4', 'Age', 'Menopause']
    bins = ['Low (0-33%)', 'Mid (33-66%)', 'High (66-100%)']

    fig, ax = plt.subplots(figsize=(6, 3.5))
    cmap = LinearSegmentedColormap.from_list('risk', ['#2166ac', '#f7f7f7', '#b2182b'], N=11)
    im = ax.imshow(weights, cmap=cmap, aspect='auto', vmin=-5, vmax=5)

    for i in range(len(features)):
        for j in range(len(bins)):
            w = weights[i, j]
            color = 'white' if abs(w) >= 3 else 'black'
            text = f'{w:+d}' if w != 0 else '0'
            ax.text(j, i, text, ha='center', va='center', fontsize=12, fontweight='bold', color=color)

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, fontsize=9)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_title('RiskSLIM Integer Scorecard', fontweight='bold', fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Risk Points', fontsize=10)
    cbar.set_ticks([-5, -3, -1, 1, 3, 5])

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_scorecard_heatmap.png'), facecolor='white')
    plt.close()
    print("  Saved: fig5_scorecard_heatmap.png")


if __name__ == "__main__":
    print("Generating publication figures...")
    fig1_bootstrap_pairwise()
    fig2_risk_stratification()
    fig3_model_comparison()
    fig4_overfitting_noise()
    fig5_scorecard_heatmap()
    print(f"\nAll figures saved in {OUTPUT_DIR}/")
