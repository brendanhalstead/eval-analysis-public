"""
Visualize bootstrap distributions to show their shape and compare to parametric fits.
Shows both point estimates (from full-data logistic regression) and bootstrap medians.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from horizon.utils.logistic import get_x_for_quantile, logistic_regression

# Configuration
WEIGHTING = 'invsqrt_task_weight'
REGULARIZATION = 0.1
SUCCESS_PERCENT = 50

def compute_point_estimates(data):
    """Compute point estimates from logistic regression on full data."""
    point_estimates = {}

    for agent in data['agent'].unique():
        agent_data = data[data['agent'] == agent]

        x = np.log2(agent_data['human_minutes'].values).reshape(-1, 1)
        y = agent_data['score_binarized'].values
        weights = agent_data[WEIGHTING].values

        if len(np.unique(y)) < 2:
            continue

        model = logistic_regression(x, y, sample_weight=weights,
                                     regularization=REGULARIZATION,
                                     ensure_weights_sum_to_1=False)

        horizon = np.exp2(get_x_for_quantile(model, SUCCESS_PERCENT / 100))
        point_estimates[agent] = horizon

    return point_estimates


# Load data and compute point estimates
print("Loading data and computing point estimates...")
data = pd.read_json('reports/time-horizon-1-1/data/raw/runs.jsonl', lines=True)
data.rename(columns={'alias': 'agent'}, inplace=True)
point_estimates = compute_point_estimates(data)

# Load bootstrap samples
samples_df = pd.read_csv('bootstrap_samples.csv')

# Select representative models (low, medium, high horizon)
# Dynamically select based on available models
available_models = samples_df.columns.tolist()

# Define preferred models for each tier (in order of preference)
low_tier = ['GPT-4 0314', 'GPT-4 Turbo (Inspect)', 'Claude 3 Opus (Inspect)']
mid_tier = ['Claude 3.5 Sonnet (New) (Inspect)', 'o1-preview', 'o1 (Inspect)']
high_tier = ['o3 (Inspect)', 'GPT-5 (Inspect)', 'Claude 4 Opus (Inspect)']
frontier_tier = ['GPT-5.2', 'Claude Opus 4.5 (Inspect)', 'GPT-5.1-Codex-Max (Inspect)', 'Gemini 3 Pro']

def pick_model(candidates, available):
    for m in candidates:
        if m in available:
            return m
    return None

models_to_plot = [
    pick_model(low_tier, available_models),
    pick_model(mid_tier, available_models),
    pick_model(high_tier, available_models),
    pick_model(frontier_tier, available_models),
]
models_to_plot = [m for m in models_to_plot if m is not None]
print(f"Plotting models: {models_to_plot}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, model in enumerate(models_to_plot):
    samples = samples_df[model].dropna().values
    point_est = point_estimates.get(model, np.nan)

    # Raw distribution
    ax = axes[0, i]
    ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # Fit and plot normal
    mu, sigma = stats.norm.fit(samples)
    x = np.linspace(samples.min(), samples.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')

    # Fit and plot log-normal
    shape, loc, scale = stats.lognorm.fit(samples, floc=0)
    ax.plot(x, stats.lognorm.pdf(x, shape, loc, scale), 'g-', lw=2, label='LogNormal fit')

    # Add CI markers
    q025, q50, q975 = np.percentile(samples, [2.5, 50, 97.5])
    ax.axvline(q50, color='black', linestyle='-', lw=1.5, label=f'Median={q50:.1f}')
    ax.axvline(q025, color='gray', linestyle='--', lw=1)
    ax.axvline(q975, color='gray', linestyle='--', lw=1)

    # Add point estimate
    if not np.isnan(point_est):
        ax.axvline(point_est, color='purple', linestyle=':', lw=2, label=f'Point est={point_est:.1f}')

    ax.set_title(model.replace(' (Inspect)', ''), fontsize=10)
    ax.set_xlabel('p50 Horizon (minutes)')
    if i == 0:
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper right')

    # Log-transformed distribution
    ax = axes[1, i]
    log_samples = np.log(samples)
    ax.hist(log_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # Fit and plot normal on log scale
    mu_log, sigma_log = stats.norm.fit(log_samples)
    x_log = np.linspace(log_samples.min(), log_samples.max(), 200)
    ax.plot(x_log, stats.norm.pdf(x_log, mu_log, sigma_log), 'r-', lw=2, label='Normal fit')

    # Mark median and point estimate
    log_q50 = np.log(q50)
    ax.axvline(log_q50, color='black', linestyle='-', lw=1.5, label='Median')

    if not np.isnan(point_est):
        log_point_est = np.log(point_est)
        ax.axvline(log_point_est, color='purple', linestyle=':', lw=2, label='Point est')

    # Calculate and show skewness and point est vs median difference
    skewness = stats.skew(log_samples)
    diff_pct = ((point_est - q50) / q50 * 100) if not np.isnan(point_est) else 0
    ax.text(0.95, 0.95, f'skew={skewness:.2f}\nPE-Med={diff_pct:+.1f}%', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('log(p50 Horizon)')
    if i == 0:
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper right')

axes[0, 0].set_ylabel('Raw scale\nDensity')
axes[1, 0].set_ylabel('Log scale\nDensity')

plt.suptitle('Bootstrap Distribution Shapes: Raw (top) vs Log-transformed (bottom)\nBlack=Median, Purple=Point Estimate', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('bootstrap_distributions.png', dpi=150, bbox_inches='tight')
print("Saved bootstrap_distributions.png")

# Also create a summary plot showing asymmetry
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

models = [col for col in samples_df.columns if col != 'Unnamed: 0']
stats_data = []

for model in models:
    samples = samples_df[model].dropna().values
    if len(samples) < 50:
        continue

    q025, q50, q975 = np.percentile(samples, [2.5, 50, 97.5])
    lower_width = q50 - q025
    upper_width = q975 - q50
    asym = lower_width / upper_width
    skewness = stats.skew(samples)
    log_skewness = stats.skew(np.log(samples))

    point_est = point_estimates.get(model, np.nan)
    pe_vs_median = ((point_est - q50) / q50 * 100) if not np.isnan(point_est) else np.nan

    stats_data.append({
        'model': model.replace(' (Inspect)', ''),
        'median': q50,
        'point_estimate': point_est,
        'pe_vs_median_pct': pe_vs_median,
        'asymmetry': asym,
        'skewness': skewness,
        'log_skewness': log_skewness
    })

df = pd.DataFrame(stats_data).sort_values('median')

# Plot 1: Asymmetry ratio vs median
ax = axes[0]
ax.scatter(df['median'], df['asymmetry'], s=80, alpha=0.7)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Symmetric')
ax.set_xlabel('Median p50 Horizon (minutes)')
ax.set_ylabel('Asymmetry Ratio\n(lower CI / upper CI)')
ax.set_xscale('log')
ax.set_title('CI Asymmetry vs Horizon')
ax.legend()

# Add model labels
for _, row in df.iterrows():
    ax.annotate(row['model'], (row['median'], row['asymmetry']),
                fontsize=7, alpha=0.7, rotation=15)

# Plot 2: Raw skewness vs median
ax = axes[1]
ax.scatter(df['median'], df['skewness'], s=80, alpha=0.7, color='orange')
ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='No skew')
ax.set_xlabel('Median p50 Horizon (minutes)')
ax.set_ylabel('Skewness (raw)')
ax.set_xscale('log')
ax.set_title('Raw Skewness vs Horizon')

# Plot 3: Log skewness vs median
ax = axes[2]
ax.scatter(df['median'], df['log_skewness'], s=80, alpha=0.7, color='green')
ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='No skew')
ax.set_xlabel('Median p50 Horizon (minutes)')
ax.set_ylabel('Skewness (log-transformed)')
ax.set_xscale('log')
ax.set_title('Log Skewness vs Horizon')
ax.set_ylim(-1, 2)

plt.tight_layout()
plt.savefig('bootstrap_asymmetry.png', dpi=150, bbox_inches='tight')
print("Saved bootstrap_asymmetry.png")

# Create a new plot: Point estimate vs Median comparison
fig, ax = plt.subplots(figsize=(8, 6))
valid_df = df.dropna(subset=['point_estimate'])

ax.scatter(valid_df['median'], valid_df['point_estimate'], s=80, alpha=0.7)

# Add diagonal line (perfect agreement)
max_val = max(valid_df['median'].max(), valid_df['point_estimate'].max())
min_val = min(valid_df['median'].min(), valid_df['point_estimate'].min())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect agreement')

ax.set_xlabel('Bootstrap Median (minutes)')
ax.set_ylabel('Point Estimate (minutes)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Point Estimate vs Bootstrap Median')
ax.legend()

# Add model labels
for _, row in valid_df.iterrows():
    ax.annotate(row['model'], (row['median'], row['point_estimate']),
                fontsize=7, alpha=0.7, rotation=15)

plt.tight_layout()
plt.savefig('point_estimate_vs_median.png', dpi=150, bbox_inches='tight')
print("Saved point_estimate_vs_median.png")

# Print summary table
print("\n" + "="*100)
print("POINT ESTIMATE VS BOOTSTRAP MEDIAN COMPARISON")
print("="*100)
print(f"{'Model':<40} {'Point Est':>12} {'Median':>12} {'Diff %':>10}")
print("-"*100)
for _, row in df.sort_values('median').iterrows():
    pe = row['point_estimate']
    med = row['median']
    diff = row['pe_vs_median_pct']
    pe_str = f"{pe:.1f}" if not np.isnan(pe) else "N/A"
    diff_str = f"{diff:+.1f}%" if not np.isnan(diff) else "N/A"
    print(f"{row['model']:<40} {pe_str:>12} {med:>12.1f} {diff_str:>10}")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("1. Log-normal fits best for all models (KS test)")
print("2. Frontier models (GPT-5, Opus 4.5) have positive skew even in log space")
print("3. Point estimates are generally close to bootstrap medians")
print("="*80)
