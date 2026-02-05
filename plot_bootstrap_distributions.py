"""
Visualize bootstrap distributions to show their shape and compare to parametric fits.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load bootstrap samples
samples_df = pd.read_csv('bootstrap_samples.csv')

# Select representative models (low, medium, high horizon)
models_to_plot = [
    'GPT-4 0314',  # Low horizon
    'Claude 3.5 Sonnet (New) (Inspect)',  # Medium horizon
    'o3 (Inspect)',  # High horizon
    'Claude Opus 4.5 (Inspect)',  # Highest horizon
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, model in enumerate(models_to_plot):
    samples = samples_df[model].dropna().values

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
    ylim = ax.get_ylim()
    ax.axvline(q50, color='black', linestyle='-', lw=1.5, label=f'Median={q50:.1f}')
    ax.axvline(q025, color='gray', linestyle='--', lw=1)
    ax.axvline(q975, color='gray', linestyle='--', lw=1)

    ax.set_title(model.replace(' (Inspect)', ''), fontsize=10)
    ax.set_xlabel('p50 Horizon (minutes)')
    if i == 0:
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    # Log-transformed distribution
    ax = axes[1, i]
    log_samples = np.log(samples)
    ax.hist(log_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # Fit and plot normal on log scale
    mu_log, sigma_log = stats.norm.fit(log_samples)
    x_log = np.linspace(log_samples.min(), log_samples.max(), 200)
    ax.plot(x_log, stats.norm.pdf(x_log, mu_log, sigma_log), 'r-', lw=2, label='Normal fit')

    # Mark median
    log_q50 = np.log(q50)
    ax.axvline(log_q50, color='black', linestyle='-', lw=1.5)

    # Calculate and show skewness
    skewness = stats.skew(log_samples)
    ax.text(0.95, 0.95, f'skew={skewness:.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('log(p50 Horizon)')
    if i == 0:
        ax.set_ylabel('Density')

axes[0, 0].set_ylabel('Raw scale\nDensity')
axes[1, 0].set_ylabel('Log scale\nDensity')

plt.suptitle('Bootstrap Distribution Shapes: Raw (top) vs Log-transformed (bottom)', fontsize=14, y=1.02)
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

    stats_data.append({
        'model': model.replace(' (Inspect)', ''),
        'median': q50,
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

# Print the key insight
print("\n" + "="*80)
print("KEY INSIGHT: Log-normal is a good fit for most models, but frontier models")
print("(GPT-5, Opus 4.5) show additional positive skew even after log transform.")
print("="*80)
