import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load benchmark results
results_path = 'results/monte_carlo_benchmarks.csv'
if not os.path.exists(results_path):
    print("Error: monte_carlo_benchmarks.csv not found in results directory. Run the benchmark script first.")
    exit(1)

df = pd.read_csv(results_path)

# Clean and format data for plotting
def clean_metric(val):
    if pd.isna(val):
        return 0.0
    return float(str(val).split(' ')[0])

# Process metrics
df['ROMA_mean'] = df['ROMA Formula'].apply(clean_metric)
df['XGB_mean'] = df['XGBoost'].apply(clean_metric)
df['RiskSLIM_mean'] = df['Optimized RiskSLIM'].apply(clean_metric)

# Extract numeric noise level for X-axis
df['Noise_Pct'] = df['Noise'].str.rstrip('%').astype(int)

# Set high-resolution plotting style
plt.figure(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

cohorts = df['Cohort'].unique()
colors = {'ROMA Formula': '#e74c3c', 'XGBoost': '#3498db', 'Optimized RiskSLIM': '#2ecc71'}
markers = {'ROMA Formula': 's', 'XGBoost': '^', 'Optimized RiskSLIM': 'o'}

# Plot subplots or faceted view for cohorts to keep it clean and comprehensive
fig, axes = plt.subplots(1, len(cohorts), figsize=(18, 5), sharey=True, dpi=300)

for idx, cohort in enumerate(cohorts):
    ax = axes[idx] if len(cohorts) > 1 else axes
    sub_df = df[df['Cohort'] == cohort]
    
    x = sub_df['Noise_Pct']
    roma = sub_df['ROMA_mean']
    xgb = sub_df['XGB_mean']
    rslim = sub_df['RiskSLIM_mean']
    
    ax.plot(x, rslim, marker='o', color='#2ecc71', linewidth=2.5, label='Optimized RiskSLIM')
    ax.plot(x, xgb, marker='^', color='#3498db', linewidth=2.5, label='XGBoost')
    ax.plot(x, roma, marker='s', color='#e74c3c', linewidth=2.5, linestyle='--', label='ROMA Formula')
    
    ax.set_title(f"{cohort}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Noise Level (%)", fontsize=10)
    if idx == 0:
        ax.set_ylabel("AUROC Performance", fontsize=10)
    ax.set_ylim(0.45, 0.95)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(frameon=True, fontsize=9)

plt.suptitle("Model AUROC Degradation Curves Across Cohorts (5% - 30% Noise)", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
output_file = 'results/degradation_curves.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Success! High-resolution graph successfully saved to '{output_file}'.")
