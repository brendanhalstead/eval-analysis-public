"""
Generate individual histogram plot showing p(success) at different task lengths.
Standalone version that doesn't require DVC.

Usage:
    python plot_individual_histogram.py --agent "GPT-5.2"
    python plot_individual_histogram.py --agent "Claude Opus 4.5 (Inspect)"
"""

import sys
sys.path.insert(0, 'src')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

from horizon.utils.logistic import get_x_for_quantile, logistic_regression

# Configuration
WEIGHTING = 'invsqrt_task_weight'
REGULARIZATION = 0.1


def get_logarithmic_bins(min_time, max_time, n_bins=15):
    """Create logarithmically spaced bins."""
    return np.geomspace(max(min_time, 0.1), max_time, n_bins + 1)


def format_time_label(minutes):
    """Format minutes as human-readable time."""
    if minutes < 1:
        return f"{minutes*60:.0f}s"
    elif minutes < 60:
        return f"{minutes:.0f}m"
    elif minutes < 1440:
        return f"{minutes/60:.1f}h"
    else:
        return f"{minutes/1440:.1f}d"


def compute_logistic_fit(agent_data, weighting, regularization):
    """Compute logistic regression fit for an agent."""
    x = np.log2(agent_data['human_minutes'].values).reshape(-1, 1)
    y = agent_data['score_binarized'].values
    weights = agent_data[weighting].values

    if len(np.unique(y)) < 2:
        return None

    model = logistic_regression(x, y, sample_weight=weights,
                                 regularization=regularization,
                                 ensure_weights_sum_to_1=False)

    return {
        'coefficient': model.coef_[0][0],
        'intercept': model.intercept_[0],
        'p50': np.exp2(get_x_for_quantile(model, 0.5))
    }


def plot_histogram(agent_data, agent_name, fit_params, output_file, all_runs=None):
    """Generate histogram plot for a single agent."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use all agents' data range if provided, else just this agent
    if all_runs is not None:
        min_time = all_runs['human_minutes'].min()
        max_time = all_runs['human_minutes'].max()
    else:
        min_time = agent_data['human_minutes'].min()
        max_time = agent_data['human_minutes'].max()

    bins = get_logarithmic_bins(min_time, max_time)

    times = agent_data['human_minutes']
    successes = agent_data['score_binarized']
    task_weights = agent_data[WEIGHTING]

    # Calculate weighted success rates for each bin
    weighted_counts_success, _ = np.histogram(
        times[successes == 1], bins=bins, weights=task_weights[successes == 1]
    )
    weighted_counts_total, _ = np.histogram(times, bins=bins, weights=task_weights)

    success_rates = np.zeros_like(weighted_counts_total, dtype=float)
    mask = weighted_counts_total > 0
    success_rates[mask] = weighted_counts_success[mask] / weighted_counts_total[mask]

    # Plot histogram bars
    width = np.diff(bins)
    centers = bins[:-1]
    ax.bar(centers, success_rates, width=width, alpha=0.7, color='steelblue',
           align='edge', edgecolor='white', label='Empirical success rate')

    # Calculate and plot standard errors
    standard_errors = np.zeros_like(success_rates)
    for i in range(len(bins) - 1):
        if mask[i]:
            bin_mask = (times >= bins[i]) & (times < bins[i + 1])
            weights_in_bin = task_weights[bin_mask]
            p = success_rates[i]
            n_eff = np.sum(weights_in_bin) ** 2 / np.sum(weights_in_bin ** 2)
            if n_eff > 0:
                variance = (p * (1 - p)) / n_eff
                if variance > 0:
                    standard_errors[i] = np.sqrt(variance)

    ax.errorbar(centers[mask] + width[mask] / 2, success_rates[mask],
                yerr=2 * standard_errors[mask], fmt='o', color='darkblue',
                alpha=0.9, markersize=5, capsize=3, label='±2 SE')

    # Plot logistic curve
    if fit_params:
        x_curve = np.logspace(np.log10(max(min_time, 0.1)), np.log10(max_time), 200)
        y_curve = expit(fit_params['coefficient'] * np.log2(x_curve) + fit_params['intercept'])
        ax.plot(x_curve, y_curve, 'r-', linewidth=2.5, label='Fitted logistic curve')

        # Mark p50
        p50 = fit_params['p50']
        if min_time <= p50 <= max_time:
            ax.axvline(p50, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(p50, 0.5, 'rx', markersize=12, markeredgewidth=3)
            ax.annotate(f'p50 = {format_time_label(p50)}',
                       xy=(p50, 0.52), fontsize=11, color='darkred',
                       ha='center', va='bottom')

    # 50% reference line
    ax.axhline(0.5, linestyle=':', alpha=0.5, color='gray')

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Task length (human time)', fontsize=12)
    ax.set_ylabel('Success probability', fontsize=12)
    ax.set_title(f'Success probability at different task lengths\n{agent_name}', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Custom x-tick labels
    tick_values = [1, 5, 15, 60, 240, 480, 960, 1920]
    tick_values = [t for t in tick_values if min_time <= t <= max_time * 1.5]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([format_time_label(t) for t in tick_values])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate individual histogram plot')
    parser.add_argument('--agent', type=str, required=True,
                        help='Agent name (e.g., "GPT-5.2" or "Claude Opus 4.5 (Inspect)")')
    parser.add_argument('--runs-file', type=str,
                        default='reports/time-horizon-1-1/data/raw/runs.jsonl',
                        help='Path to runs.jsonl file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: histogram_<agent>.png)')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.runs_file}...")
    all_runs = pd.read_json(args.runs_file, lines=True)
    all_runs.rename(columns={'alias': 'agent'}, inplace=True)

    # Check available agents
    available_agents = all_runs['agent'].unique()
    print(f"Available agents: {list(available_agents)}")

    if args.agent not in available_agents:
        print(f"Error: Agent '{args.agent}' not found in data.")
        print("Available agents:")
        for a in sorted(available_agents):
            print(f"  - {a}")
        return

    # Filter to agent
    agent_data = all_runs[all_runs['agent'] == args.agent]
    print(f"Found {len(agent_data)} runs for {args.agent}")

    # Compute logistic fit
    fit_params = compute_logistic_fit(agent_data, WEIGHTING, REGULARIZATION)
    if fit_params:
        print(f"Logistic fit: coef={fit_params['coefficient']:.3f}, "
              f"intercept={fit_params['intercept']:.3f}, p50={fit_params['p50']:.1f} min")

    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        safe_name = args.agent.replace(' ', '_').replace('(', '').replace(')', '')
        output_file = f'histogram_{safe_name}.png'

    # Plot
    plot_histogram(agent_data, args.agent, fit_params, output_file, all_runs)


if __name__ == '__main__':
    main()
