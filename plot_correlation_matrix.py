"""
Generate correlation matrix visualizations from bootstrap samples.

Usage:
    python plot_correlation_matrix.py [--success-percent 50]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_data(success_percent):
    """Load correlation matrix and bootstrap samples."""
    suffix = '' if success_percent == 50 else f'_p{success_percent}'

    corr_df = pd.read_csv(f'correlation_matrix{suffix}.csv', index_col=0)
    samples_df = pd.read_csv(f'bootstrap_samples{suffix}.csv')

    return corr_df, samples_df


def compute_medians(samples_df):
    """Compute median horizons for each model."""
    medians = {}
    for col in samples_df.columns:
        s = samples_df[col].dropna().values
        if len(s) > 0:
            medians[col] = np.median(s)
    return medians


def plot_correlation_matrix(corr_df, output_file='correlation_matrix.png'):
    """Generate heatmap of correlation matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))

    if HAS_SEABORN:
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlBu_r',
                    center=0, vmin=-0.2, vmax=1.0,
                    square=True, linewidths=0.5, ax=ax,
                    annot_kws={'size': 7})
    else:
        im = ax.imshow(corr_df.values, cmap='RdYlBu_r', vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_df.index)
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                ax.text(j, i, f'{corr_df.iloc[i, j]:.2f}',
                       ha='center', va='center', fontsize=6)

    ax.set_title('Bootstrap Error Correlation Matrix\n(Models sorted by median horizon)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")


def plot_correlation_by_capability(corr_df, medians, output_file='correlation_by_capability.png'):
    """Plot correlation vs capability difference."""
    # Build pairs with their correlation and horizon difference
    models = list(corr_df.index)
    pairs = []

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue

            # Get medians - handle name variations
            med1 = medians.get(m1) or medians.get(f"{m1} (Inspect)")
            med2 = medians.get(m2) or medians.get(f"{m2} (Inspect)")

            if med1 is None or med2 is None:
                continue

            log_diff = abs(np.log10(med1) - np.log10(med2))
            corr = corr_df.loc[m1, m2]
            pairs.append((m1, m2, corr, log_diff))

    if not pairs:
        print("No valid pairs found for capability plot")
        return

    correlations = [p[2] for p in pairs]
    log_diffs = [p[3] for p in pairs]

    # Fit linear trend
    slope, intercept = np.polyfit(log_diffs, correlations, 1)
    r_val = np.corrcoef(log_diffs, correlations)[0, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(log_diffs, correlations, alpha=0.6, s=50)

    # Trend line
    x_line = np.linspace(0, max(log_diffs), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2,
            label=f'Linear fit: r={r_val:.2f}')

    ax.set_xlabel('Log₁₀ Horizon Difference (capability gap)', fontsize=12)
    ax.set_ylabel('Bootstrap Error Correlation', fontsize=12)
    ax.set_title('Correlation Structure: Similar Capabilities → Correlated Errors', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")

    # Print summary stats
    print(f"\nCorrelation vs capability gap: r = {r_val:.3f}")
    print(f"Mean correlation: {np.mean(correlations):.3f}")
    print(f"Correlation range: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")


def main():
    parser = argparse.ArgumentParser(description='Plot correlation matrix visualizations')
    parser.add_argument('--success-percent', type=int, default=50,
                        help='Target success percentage (default: 50)')
    args = parser.parse_args()

    suffix = '' if args.success_percent == 50 else f'_p{args.success_percent}'

    print(f"Loading data for p{args.success_percent}...")
    corr_df, samples_df = load_data(args.success_percent)
    medians = compute_medians(samples_df)

    print(f"Found {len(corr_df)} models in correlation matrix")

    # Generate plots
    plot_correlation_matrix(corr_df, f'correlation_matrix{suffix}.png')
    plot_correlation_by_capability(corr_df, medians, f'correlation_by_capability{suffix}.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
