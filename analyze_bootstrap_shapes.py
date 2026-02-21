"""
Analyze the shape of bootstrap distributions for time horizons.
This script generates bootstrap samples and examines their distributional properties
to inform the choice of likelihood function in a Bayesian model.

Usage:
    python analyze_bootstrap_shapes.py [--success-percent 50] [--n-bootstrap 1000]
"""

import sys
sys.path.insert(0, 'src')

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

from horizon.utils.logistic import get_x_for_quantile, logistic_regression

# Default configuration
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_REGULARIZATION = 0.1
DEFAULT_WEIGHTING = 'invsqrt_task_weight'
DEFAULT_SUCCESS_PERCENT = 50


def bootstrap_sample(data, categories, rng):
    """Hierarchical bootstrap over task_family, task_id, run_id."""
    indices = np.arange(len(data))

    for category in categories:
        if category == 'run_id':
            # Bootstrap runs within each task-agent group
            task_agent = data['task_id'].astype(str) + '|||' + data['agent'].astype(str)
            task_agents = task_agent.iloc[indices].values
            unique_tas, inverse = np.unique(task_agents, return_inverse=True)

            new_indices = []
            for j, ta in enumerate(unique_tas):
                group_idx = indices[inverse == j]
                sampled = rng.choice(group_idx, size=len(group_idx), replace=True)
                new_indices.append(sampled)
            indices = np.concatenate(new_indices)
        else:
            category_vals = data[category].iloc[indices].values
            unique_vals, inverse = np.unique(category_vals, return_inverse=True)
            n_vals = len(unique_vals)
            sampled_vals = rng.choice(n_vals, size=n_vals, replace=True)

            new_indices = []
            for sampled_val in sampled_vals:
                group_idx = indices[inverse == sampled_val]
                new_indices.append(group_idx)
            indices = np.concatenate(new_indices)

    return data.iloc[indices].copy()


def compute_horizon(agent_data, weights_col, regularization, success_percent):
    """Compute p50/p80 horizon for a single agent's data."""
    x = np.log2(agent_data['human_minutes'].values).reshape(-1, 1)
    y = agent_data['score_binarized'].values
    weights = agent_data[weights_col].values

    if len(np.unique(y)) < 2:
        return np.nan

    model = logistic_regression(x, y, sample_weight=weights,
                                 regularization=regularization,
                                 ensure_weights_sum_to_1=False)

    horizon = np.exp2(get_x_for_quantile(model, success_percent / 100))
    return horizon


def run_bootstrap(data, agents, n_bootstrap, categories):
    """Run bootstrap and collect samples for each agent."""
    results = {agent: [] for agent in agents}

    print(f"Running {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i + 1}/{n_bootstrap}")

        rng = np.random.default_rng(42 + i)
        boot_data = bootstrap_sample(data, categories, rng)

        for agent in agents:
            agent_data = boot_data[boot_data['agent'] == agent]
            if len(agent_data) == 0:
                results[agent].append(np.nan)
                continue

            horizon = compute_horizon(agent_data, DEFAULT_WEIGHTING, DEFAULT_REGULARIZATION, SUCCESS_PERCENT)
            results[agent].append(horizon)

    return {agent: np.array(vals) for agent, vals in results.items()}


def analyze_distribution(samples, agent_name):
    """Analyze the distributional properties of bootstrap samples."""
    samples = samples[~np.isnan(samples)]
    if len(samples) < 50:
        return None

    # Basic statistics
    mean = np.mean(samples)
    median = np.median(samples)
    std = np.std(samples)
    q025 = np.percentile(samples, 2.5)
    q975 = np.percentile(samples, 97.5)

    # Asymmetry ratio (lower CI width / upper CI width)
    lower_width = median - q025
    upper_width = q975 - median
    asymmetry_ratio = lower_width / upper_width if upper_width > 0 else np.nan

    # Skewness and kurtosis
    skewness = skew(samples)
    kurt = kurtosis(samples)  # excess kurtosis (normal = 0)

    # Log-transform analysis
    log_samples = np.log(samples[samples > 0])
    log_skewness = skew(log_samples) if len(log_samples) > 10 else np.nan
    log_kurt = kurtosis(log_samples) if len(log_samples) > 10 else np.nan

    # Normality tests (on raw and log-transformed)
    try:
        _, shapiro_p = shapiro(samples[:5000] if len(samples) > 5000 else samples)
    except:
        shapiro_p = np.nan

    try:
        _, shapiro_log_p = shapiro(log_samples[:5000] if len(log_samples) > 5000 else log_samples)
    except:
        shapiro_log_p = np.nan

    # Fit candidate distributions
    fits = {}

    # Normal fit
    norm_params = stats.norm.fit(samples)
    norm_ks, norm_ks_p = stats.kstest(samples, 'norm', norm_params)
    fits['normal'] = {'params': norm_params, 'ks_stat': norm_ks, 'ks_p': norm_ks_p}

    # Log-normal fit
    try:
        lognorm_params = stats.lognorm.fit(samples, floc=0)
        lognorm_ks, lognorm_ks_p = stats.kstest(samples, 'lognorm', lognorm_params)
        fits['lognormal'] = {'params': lognorm_params, 'ks_stat': lognorm_ks, 'ks_p': lognorm_ks_p}
    except:
        fits['lognormal'] = None

    # Gamma fit
    try:
        gamma_params = stats.gamma.fit(samples, floc=0)
        gamma_ks, gamma_ks_p = stats.kstest(samples, 'gamma', gamma_params)
        fits['gamma'] = {'params': gamma_params, 'ks_stat': gamma_ks, 'ks_p': gamma_ks_p}
    except:
        fits['gamma'] = None

    return {
        'agent': agent_name,
        'n_samples': len(samples),
        'mean': mean,
        'median': median,
        'std': std,
        'q025': q025,
        'q975': q975,
        'lower_ci_width': lower_width,
        'upper_ci_width': upper_width,
        'asymmetry_ratio': asymmetry_ratio,
        'skewness': skewness,
        'kurtosis': kurt,
        'log_skewness': log_skewness,
        'log_kurtosis': log_kurt,
        'shapiro_p_raw': shapiro_p,
        'shapiro_p_log': shapiro_log_p,
        'fits': fits,
        'samples': samples  # Keep for plotting
    }


def compute_correlation_matrix(samples_df):
    """Compute correlation matrix of log-space residuals."""
    log_residuals = {}
    medians = {}

    for col in samples_df.columns:
        s = samples_df[col].dropna().values
        if len(s) < 50:
            continue
        log_s = np.log(s)
        log_residuals[col] = log_s - np.median(log_s)
        medians[col] = np.exp(np.median(log_s))

    models = list(log_residuals.keys())
    n = len(models)
    corr_matrix = np.zeros((n, n))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            min_len = min(len(log_residuals[m1]), len(log_residuals[m2]))
            corr_matrix[i, j] = np.corrcoef(
                log_residuals[m1][:min_len],
                log_residuals[m2][:min_len]
            )[0, 1]

    # Create DataFrame with short names
    short_names = [m.replace(' (Inspect)', '') for m in models]
    corr_df = pd.DataFrame(corr_matrix, index=short_names, columns=short_names)

    # Sort by median horizon
    model_medians = [medians[m] for m in models]
    sort_idx = np.argsort(model_medians)
    sorted_names = [short_names[i] for i in sort_idx]
    corr_df_sorted = corr_df.loc[sorted_names, sorted_names]

    return corr_df_sorted, medians


def main(success_percent=DEFAULT_SUCCESS_PERCENT, n_bootstrap=DEFAULT_N_BOOTSTRAP):
    """Main analysis function.

    Args:
        success_percent: Target success rate (50 or 80)
        n_bootstrap: Number of bootstrap iterations
    """
    global SUCCESS_PERCENT
    SUCCESS_PERCENT = success_percent

    suffix = '' if success_percent == 50 else f'_p{success_percent}'

    # Load data
    print("Loading data...")
    data = pd.read_json('reports/time-horizon-1-1/data/raw/runs.jsonl', lines=True)
    data.rename(columns={'alias': 'agent'}, inplace=True)
    print(f"Loaded {len(data)} runs")

    # Get unique agents
    agents = data['agent'].unique()
    print(f"Found {len(agents)} agents: {list(agents)}")

    # Run bootstrap
    categories = ['task_family', 'task_id', 'run_id']
    bootstrap_results = run_bootstrap(data, agents, n_bootstrap, categories)

    # Analyze distributions
    print("\n" + "="*80)
    print("BOOTSTRAP DISTRIBUTION ANALYSIS")
    print("="*80)

    analyses = []
    for agent in agents:
        analysis = analyze_distribution(bootstrap_results[agent], agent)
        if analysis:
            analyses.append(analysis)

    # Print summary table
    print("\n" + "-"*120)
    print(f"{'Agent':<40} {'Median':>10} {'Skew':>8} {'Kurt':>8} {'Asym':>8} {'LogSkew':>8} {'Shapiro(raw)':>12} {'Shapiro(log)':>12}")
    print("-"*120)

    for a in sorted(analyses, key=lambda x: x['median']):
        print(f"{a['agent']:<40} {a['median']:>10.1f} {a['skewness']:>8.2f} {a['kurtosis']:>8.2f} "
              f"{a['asymmetry_ratio']:>8.2f} {a['log_skewness']:>8.2f} "
              f"{a['shapiro_p_raw']:>12.4f} {a['shapiro_p_log']:>12.4f}")

    # Print distribution fit comparison
    print("\n" + "-"*100)
    print("DISTRIBUTION FIT COMPARISON (KS test p-values, higher = better fit)")
    print("-"*100)
    print(f"{'Agent':<40} {'Normal':>12} {'LogNormal':>12} {'Gamma':>12} {'Best Fit':<15}")
    print("-"*100)

    for a in sorted(analyses, key=lambda x: x['median']):
        fits = a['fits']
        norm_p = fits['normal']['ks_p'] if fits['normal'] else 0
        lognorm_p = fits['lognormal']['ks_p'] if fits['lognormal'] else 0
        gamma_p = fits['gamma']['ks_p'] if fits['gamma'] else 0

        best = max([('Normal', norm_p), ('LogNormal', lognorm_p), ('Gamma', gamma_p)], key=lambda x: x[1])

        print(f"{a['agent']:<40} {norm_p:>12.4f} {lognorm_p:>12.4f} {gamma_p:>12.4f} {best[0]:<15}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS ACROSS ALL MODELS")
    print("="*80)

    skewnesses = [a['skewness'] for a in analyses]
    asymmetries = [a['asymmetry_ratio'] for a in analyses]
    log_skewnesses = [a['log_skewness'] for a in analyses if not np.isnan(a['log_skewness'])]

    print(f"\nSkewness (raw): mean={np.mean(skewnesses):.3f}, range=[{np.min(skewnesses):.3f}, {np.max(skewnesses):.3f}]")
    print(f"Asymmetry ratio: mean={np.mean(asymmetries):.3f}, range=[{np.min(asymmetries):.3f}, {np.max(asymmetries):.3f}]")
    print(f"Skewness (log): mean={np.mean(log_skewnesses):.3f}, range=[{np.min(log_skewnesses):.3f}, {np.max(log_skewnesses):.3f}]")

    # Count which distribution fits best
    best_fits = []
    for a in analyses:
        fits = a['fits']
        norm_p = fits['normal']['ks_p'] if fits['normal'] else 0
        lognorm_p = fits['lognormal']['ks_p'] if fits['lognormal'] else 0
        gamma_p = fits['gamma']['ks_p'] if fits['gamma'] else 0
        best = max([('Normal', norm_p), ('LogNormal', lognorm_p), ('Gamma', gamma_p)], key=lambda x: x[1])
        best_fits.append(best[0])

    from collections import Counter
    fit_counts = Counter(best_fits)
    print(f"\nBest-fitting distribution counts: {dict(fit_counts)}")

    # Save bootstrap samples for further analysis
    print("\n" + "="*80)
    samples_df = pd.DataFrame(bootstrap_results)
    samples_file = f'bootstrap_samples{suffix}.csv'
    print(f"Saving bootstrap samples to {samples_file}...")
    samples_df.to_csv(samples_file, index=False)

    # Compute and save correlation matrix
    corr_df, medians = compute_correlation_matrix(samples_df)
    corr_file = f'correlation_matrix{suffix}.csv'
    print(f"Saving correlation matrix to {corr_file}...")
    corr_df.to_csv(corr_file)

    print("\nModels sorted by horizon:")
    for name in corr_df.index:
        # Find original name with (Inspect) suffix
        orig_name = name if name in medians else f"{name} (Inspect)"
        if orig_name in medians:
            print(f"  {name:<40} {medians[orig_name]:>10.1f} min")

    print("\nDone!")

    return analyses, bootstrap_results, corr_df


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze bootstrap distribution shapes')
    parser.add_argument('--success-percent', type=int, default=DEFAULT_SUCCESS_PERCENT,
                        help='Target success percentage (default: 50)')
    parser.add_argument('--n-bootstrap', type=int, default=DEFAULT_N_BOOTSTRAP,
                        help='Number of bootstrap iterations (default: 1000)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyses, bootstrap_results, corr_df = main(
        success_percent=args.success_percent,
        n_bootstrap=args.n_bootstrap
    )
