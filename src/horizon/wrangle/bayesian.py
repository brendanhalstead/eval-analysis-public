"""Bayesian logistic regression pipeline for time horizon estimation.

Drop-in alternative to wrangle/logistic.py.  Produces a DataFrame in the same
format (same column names, same semantics) so downstream plotting code works
unchanged.  The key difference is that credible intervals come from the exact
posterior rather than from bootstrap resampling.

Usage (standalone):
    python -m horizon.wrangle.bayesian \
        --fig-name headline \
        --runs-file data/raw/runs.jsonl \
        --output-logistic-fits-file data/wrangled/logistic_fits/bayesian.csv \
        --release-dates ../../data/external/release_dates.yaml

Usage (as library):
    from horizon.wrangle.bayesian import run_bayesian_regressions
    df = run_bayesian_regressions(runs, release_dates_file, wrangle_params)
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import yaml

from horizon.compute_task_weights import add_task_weight_columns
from horizon.utils.bayesian import (
    BayesianPrior,
    bayesian_logistic_regression,
    compare_priors,
)

logger = logging.getLogger(__name__)


class WrangleParams(TypedDict):
    runs_file: pathlib.Path
    weighting: str
    categories: list[str]
    regularization: float
    exclude: list[str]
    success_percents: list[int]
    confidence_level: float


# ---------------------------------------------------------------------------
# Empirical success rates (reused from the frequentist pipeline for display)
# ---------------------------------------------------------------------------


def _empirical_success_rates(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    time_buckets: list[int] | None = None,
) -> tuple[pd.Series[Any], float]:
    """Weighted empirical success rates in log-spaced time buckets."""
    if time_buckets is None:
        time_buckets = [1, 4, 16, 64, 256, 960, 2 * 24 * 60]

    rates = []
    for i in range(len(time_buckets) - 1):
        mask = (np.exp2(x).reshape(-1) >= time_buckets[i]) & (
            np.exp2(x).reshape(-1) < time_buckets[i + 1]
        )
        if np.sum(weights[mask]) > 0:
            rate = np.sum(y[mask] * weights[mask]) / np.sum(weights[mask])
        else:
            rate = float("nan")
        rates.append(rate)

    average = float(np.sum(y * weights) / np.sum(weights))
    indices = [
        f"{lo}-{hi} min"
        for lo, hi in zip(time_buckets[:-1], time_buckets[1:])
    ]
    return pd.Series(rates, index=indices), average


# ---------------------------------------------------------------------------
# Per-agent Bayesian regression
# ---------------------------------------------------------------------------


def agent_bayesian_regression(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any] | None,
    agent_name: str,
    prior: BayesianPrior,
    success_percents: list[int],
    confidence_level: float,
    include_empirical_rates: bool = True,
    n_grid: int = 150,
) -> pd.Series[Any]:
    """Bayesian logistic regression for a single agent.

    Returns a pd.Series with the same index structure as agent_regression()
    in wrangle/logistic.py, so downstream code works unchanged.

    If weights is None, uses the true likelihood (each run = one observation).
    If weights is provided, uses a pseudo-likelihood — caller is responsible
    for the weight scale.
    """
    x_log = np.log2(x)

    # For empirical rates display, use uniform weights if none provided
    display_weights = weights if weights is not None else np.ones(len(y)) / len(y)
    empirical_rates, average_emp = None, None
    if include_empirical_rates:
        empirical_rates, average_emp = _empirical_success_rates(x_log, y, display_weights)

    # Run the Bayesian model
    result = bayesian_logistic_regression(
        x,  # raw minutes — the function does log2 internally
        y,
        weights=weights,  # None → unweighted (true likelihood)
        prior=prior,
        success_percents=success_percents,
        ci_level=confidence_level,
        n_grid=n_grid,
    )

    if result["coefficient"] > 0:
        logger.warning(f"Warning: {agent_name} MAP slope is positive ({result['coefficient']:.3f})")

    # Build the output Series in the same format as the frequentist pipeline
    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q

    indices = ["coefficient", "intercept", "bce_loss", "average"]
    values: list[float] = [
        result["coefficient"],
        result["intercept"],
        result["bce_loss"],
        average_emp if average_emp is not None else result["average"],
    ]

    for p in success_percents:
        indices.extend([f"p{p}", f"p{p}q{low_q:.3f}", f"p{p}q{high_q:.3f}"])
        values.extend([
            result[f"p{p}"],
            result[f"p{p}q{low_q:.3f}"],
            result[f"p{p}q{high_q:.3f}"],
        ])

    # Bayesian-specific columns
    for key in [
        "log_evidence",
        "alpha_posterior_mean",
        "alpha_posterior_std",
        "beta_posterior_mean",
        "beta_posterior_std",
        "posterior_correlation",
    ]:
        indices.append(key)
        values.append(result[key])

    series = pd.Series(values, index=indices)
    if include_empirical_rates and empirical_rates is not None:
        series = pd.concat([series, empirical_rates])
    return series


# ---------------------------------------------------------------------------
# Full pipeline: all agents
# ---------------------------------------------------------------------------


def run_bayesian_regressions(
    runs: pd.DataFrame,
    release_dates_file: pathlib.Path,
    wrangle_params: WrangleParams,
    prior: BayesianPrior | None = None,
    use_weights: bool = False,
    include_empirical_rates: bool = True,
    n_grid: int = 150,
) -> pd.DataFrame:
    """Run Bayesian logistic regression for all agents.

    Drop-in replacement for run_logistic_regressions() in wrangle/logistic.py.

    Parameters
    ----------
    use_weights : if False (default), uses the true likelihood — each run is
        one observation, no weighting.  If True, uses the pseudo-likelihood
        with METR's invsqrt_task_weight (or whatever wrangle_params["weighting"]
        specifies).  Note: because these weights are normalized to sum to 1,
        the pseudo-likelihood compresses all the data into ~1 unit of evidence.
        This is equivalent to a very strong prior and yields extremely wide
        credible intervals.  You almost certainly want use_weights=False
        until the hierarchical model is implemented.
    """
    if prior is None:
        prior = BayesianPrior()

    release_dates = yaml.safe_load(release_dates_file.read_text())

    # Handle exclusions (same logic as the frequentist pipeline)
    if wrangle_params["exclude"] is not None and len(wrangle_params["exclude"]) > 0:
        excluding = set(wrangle_params["exclude"])
        logger.info(f"Excluding task sources: {excluding}")
        runs = runs[~runs["task_source"].isin(excluding)]
        runs = runs.drop(columns=["equal_task_weight", "invsqrt_task_weight"])
        runs = add_task_weight_columns(runs)

    score_col = wrangle_params.get("score_col", "score_binarized")
    logger.info(
        f"Running Bayesian regressions for {len(runs)} runs "
        f"(score_col={score_col}, n_grid={n_grid}, use_weights={use_weights})"
    )

    results = []
    runs = runs.rename(columns={"alias": "agent"})
    for agent, agent_runs in runs.groupby("agent", as_index=False):
        weights = (
            agent_runs[wrangle_params["weighting"]].values
            if use_weights
            else None
        )
        regression = agent_bayesian_regression(
            agent_runs["human_minutes"].values,
            agent_runs[score_col].values,
            weights=weights,
            agent_name=str(agent),
            prior=prior,
            success_percents=wrangle_params["success_percents"],
            confidence_level=wrangle_params["confidence_level"],
            include_empirical_rates=include_empirical_rates,
            n_grid=n_grid,
        )
        regression["agent"] = agent
        results.append(regression)

    regressions = pd.DataFrame([s.to_dict() for s in results])
    regressions["release_date"] = regressions["agent"].map(release_dates["date"])

    numeric_columns = regressions.select_dtypes(include=["float64", "float32"]).columns
    regressions[numeric_columns] = regressions[numeric_columns].round(6)
    return regressions


# ---------------------------------------------------------------------------
# Prior sensitivity report
# ---------------------------------------------------------------------------


def run_prior_sensitivity(
    runs: pd.DataFrame,
    wrangle_params: WrangleParams,
    agent_name: str,
    priors: dict[str, BayesianPrior] | None = None,
    n_grid: int = 150,
) -> dict[str, dict[str, Any]]:
    """Run prior sensitivity analysis for a single agent.

    Compares multiple prior specifications via log Bayes factors.  This is the
    principled way to assess how much the prior matters: if the log Bayes
    factors are all near zero, the data dominate and the prior is irrelevant.
    If they differ substantially, you should think harder about your prior.

    Parameters
    ----------
    priors : dict mapping label -> BayesianPrior.  If None, uses a default
        set that tests robustness along the key dimensions:
        - reference: the default prior
        - wide_beta: wider uncertainty on the slope
        - narrow_beta: narrower uncertainty on the slope
        - flat_alpha: very weak prior on the intercept
    """
    if priors is None:
        priors = {
            "reference (default)": BayesianPrior(),
            "wide_beta (sigma=3)": BayesianPrior(beta_std=3.0),
            "narrow_beta (sigma=0.5)": BayesianPrior(beta_std=0.5),
            "steep_beta (mu=-2)": BayesianPrior(beta_mean=-2.0),
            "flat_alpha (sigma=20)": BayesianPrior(alpha_std=20.0),
        }

    score_col = wrangle_params.get("score_col", "score_binarized")
    runs = runs.rename(columns={"alias": "agent"})
    agent_runs = runs[runs["agent"] == agent_name]

    if len(agent_runs) == 0:
        raise ValueError(f"No runs found for agent '{agent_name}'")

    x = np.log2(agent_runs["human_minutes"].values)
    y = agent_runs[score_col].values

    # Unweighted: each observation is one unit of evidence (true likelihood).
    w = np.ones(len(y))

    return compare_priors(
        x, y, w,
        priors=priors,
        success_percents=wrangle_params["success_percents"],
        ci_level=wrangle_params["confidence_level"],
        n_grid=n_grid,
    )


# ---------------------------------------------------------------------------
# Hierarchical Bayesian model (run-level human baselines)
# ---------------------------------------------------------------------------


def _compute_human_duration_minutes(row: dict[str, Any]) -> float:
    """Compute duration in minutes for a single human run.

    Handles the three different timing conventions:
    - HCAST: started_at=0, completed_at=duration in ms
    - SWAA: both are timestamps, duration = completed_at - started_at
    - RE-Bench: completed_at=duration in ms (but not meaningful as
      completion time — these are fixed experimental windows)
    """
    started = float(row.get("started_at", 0))
    completed = float(row.get("completed_at", 0))
    if started == 0.0:
        return completed / (60 * 1000)
    else:
        return (completed - started) / (60 * 1000)


def prepare_hierarchical_data(
    runs: pd.DataFrame,
    exclude_sources: list[str] | None = None,
    score_col: str = "score_binarized",
) -> "HierarchicalData":
    """Transform raw runs.jsonl into HierarchicalData for the PyMC model.

    Extracts:
    - Run-level human baseline durations (HCAST + SWAA successful runs)
    - Expert estimates for tasks with no baselines (and RE-Bench tasks)
    - Agent (task, score) observations

    Parameters
    ----------
    runs : raw DataFrame from runs.jsonl (with 'alias' column)
    exclude_sources : task sources to drop (e.g. ["SWAA"])
    score_col : column to use for agent scores

    Returns
    -------
    HierarchicalData ready for build_hierarchical_model()
    """
    from horizon.utils.bayesian_hierarchical import HierarchicalData

    runs = runs.copy()

    if exclude_sources:
        runs = runs[~runs["task_source"].isin(exclude_sources)]

    # Separate human vs agent runs
    human_runs = runs[runs["alias"] == "human"].copy()
    agent_runs = runs[runs["alias"] != "human"].copy()

    # --- Human baseline observations ---
    # Compute actual duration for each human run
    human_runs["duration_minutes"] = human_runs.apply(
        _compute_human_duration_minutes, axis=1
    )

    # For RE-Bench, the "duration" is a fixed experimental window, not
    # a completion time.  Treat human_minutes as an expert estimate instead.
    rebench_tasks = set(
        human_runs.loc[human_runs["task_source"] == "RE-Bench", "task_id"]
    )

    # Successful non-RE-Bench baselines with positive duration
    baseline_mask = (
        (human_runs["score_binarized"] == 1)
        & (human_runs["duration_minutes"] > 0)
        & (~human_runs["task_id"].isin(rebench_tasks))
    )
    baseline_runs = human_runs[baseline_mask].copy()
    baseline_runs["log2_duration"] = np.log2(baseline_runs["duration_minutes"])

    # --- Build task/family index maps ---
    all_task_ids = sorted(agent_runs["task_id"].unique())
    all_families = sorted(agent_runs["task_family"].unique())
    task_to_idx = {t: i for i, t in enumerate(all_task_ids)}
    family_to_idx = {f: i for i, f in enumerate(all_families)}

    # Task -> family mapping
    task_family_map = (
        agent_runs.groupby("task_id")["task_family"]
        .first()
        .to_dict()
    )
    task_to_family_idx = np.array(
        [family_to_idx[task_family_map[t]] for t in all_task_ids]
    )

    # --- Human baseline observations (indexed) ---
    # Keep only baselines for tasks that appear in agent_runs
    valid_baselines = baseline_runs[baseline_runs["task_id"].isin(task_to_idx)]
    human_task_idx = np.array(
        [task_to_idx[t] for t in valid_baselines["task_id"]]
    )
    human_log2_duration = valid_baselines["log2_duration"].values.astype(float)

    # --- Expert estimates ---
    # Tasks with no baselines + RE-Bench tasks → use human_minutes as estimate
    tasks_with_baselines = set(valid_baselines["task_id"])
    estimate_tasks = set(all_task_ids) - tasks_with_baselines

    # Get human_minutes for estimate-only tasks from any run
    task_human_minutes = agent_runs.groupby("task_id")["human_minutes"].first()
    estimate_task_idx_list = []
    estimate_log2_list = []
    for task_id in sorted(estimate_tasks):
        if task_id in task_human_minutes.index:
            hm = task_human_minutes[task_id]
            if hm > 0:
                estimate_task_idx_list.append(task_to_idx[task_id])
                estimate_log2_list.append(np.log2(hm))

    estimate_task_idx = np.array(estimate_task_idx_list, dtype=int)
    estimate_log2_minutes = np.array(estimate_log2_list, dtype=float)

    # --- Agent observations (indexed) ---
    agent_names = sorted(agent_runs["alias"].unique())
    agent_to_idx = {a: i for i, a in enumerate(agent_names)}

    agent_task_idx = np.array(
        [task_to_idx[t] for t in agent_runs["task_id"]]
    )
    agent_agent_idx = np.array(
        [agent_to_idx[a] for a in agent_runs["alias"]]
    )
    agent_scores = agent_runs[score_col].values.astype(float)

    data = HierarchicalData(
        task_ids=all_task_ids,
        family_ids=all_families,
        task_to_family_idx=task_to_family_idx,
        human_task_idx=human_task_idx,
        human_log2_duration=human_log2_duration,
        estimate_task_idx=estimate_task_idx,
        estimate_log2_minutes=estimate_log2_minutes,
        agent_names=agent_names,
        agent_task_idx=agent_task_idx,
        agent_agent_idx=agent_agent_idx,
        agent_scores=agent_scores,
    )

    logger.info(data.summary())
    return data


def run_hierarchical_bayesian(
    runs: pd.DataFrame,
    release_dates_file: pathlib.Path,
    wrangle_params: WrangleParams,
    n_samples: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
    include_empirical_rates: bool = True,
    output_task_difficulty_file: pathlib.Path | None = None,
    output_hyperparams_file: pathlib.Path | None = None,
    output_diagnostics_file: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Run the full hierarchical Bayesian pipeline.

    Drop-in replacement for run_bayesian_regressions().  Produces the same
    output format (DataFrame with one row per agent) so downstream plotting
    code works unchanged.

    Additionally, the hierarchical model jointly infers per-task difficulty
    from run-level human baselines, rather than using the pre-computed
    geometric mean.

    Parameters
    ----------
    runs : raw DataFrame from runs.jsonl
    release_dates_file : YAML mapping agent -> release date
    wrangle_params : pipeline parameters (weighting, exclude, etc.)
    n_samples : MCMC posterior draws per chain
    n_tune : MCMC warmup draws per chain
    n_chains : number of MCMC chains
    target_accept : NUTS target acceptance rate
    random_seed : for reproducibility
    include_empirical_rates : whether to add binned empirical rates
    output_task_difficulty_file : optional path to save per-task posteriors
    output_hyperparams_file : optional path to save hyperparameter summaries
    output_diagnostics_file : optional path to save MCMC diagnostics

    Returns
    -------
    DataFrame matching the format of run_bayesian_regressions()
    """
    from horizon.utils.bayesian_hierarchical import (
        HierarchicalPrior,
        build_hierarchical_model,
        compute_bce_from_posterior,
        extract_agent_results,
        extract_hyperparameter_summary,
        extract_task_difficulty_summary,
        sample_hierarchical_model,
    )

    release_dates = yaml.safe_load(release_dates_file.read_text())

    # Handle exclusions
    exclude = wrangle_params.get("exclude", [])
    exclude_sources = exclude if exclude else None

    score_col = wrangle_params.get("score_col", "score_binarized")

    # Step 1: Prepare data
    data = prepare_hierarchical_data(
        runs, exclude_sources=exclude_sources, score_col=score_col
    )

    # Step 2: Build model
    prior = HierarchicalPrior()
    model = build_hierarchical_model(data, prior)
    n_free = sum(v.size for v in model.initial_point().values())
    logger.info(f"Built hierarchical model with {n_free} free parameters")

    # Step 3: Run MCMC
    logger.info(
        f"Sampling: {n_samples} draws x {n_chains} chains, "
        f"{n_tune} tune, target_accept={target_accept}"
    )
    idata = sample_hierarchical_model(
        model,
        n_samples=n_samples,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept,
        random_seed=random_seed,
    )

    # Step 4: Extract results
    success_percents = wrangle_params.get("success_percents", [50, 80])
    confidence_level = wrangle_params.get("confidence_level", 0.95)

    regressions = extract_agent_results(
        idata,
        agent_names=data.agent_names,
        success_percents=success_percents,
        ci_level=confidence_level,
    )

    # Add BCE from posterior means
    bce_dict = compute_bce_from_posterior(idata, data)
    regressions["bce_loss"] = regressions["agent"].map(bce_dict)

    # Add empirical weighted average scores
    if include_empirical_rates:
        runs_copy = runs.copy()
        if exclude_sources:
            runs_copy = runs_copy[~runs_copy["task_source"].isin(exclude_sources)]
        runs_copy = runs_copy.rename(columns={"alias": "agent"})
        for idx, row in regressions.iterrows():
            agent = row["agent"]
            agent_mask = runs_copy["agent"] == agent
            agent_data = runs_copy[agent_mask]
            if len(agent_data) > 0:
                avg = float(agent_data[score_col].mean())
                regressions.at[idx, "average"] = avg

    # Add release dates
    regressions["release_date"] = regressions["agent"].map(release_dates["date"])

    # Round numeric columns
    numeric_columns = regressions.select_dtypes(include=["float64", "float32"]).columns
    regressions[numeric_columns] = regressions[numeric_columns].round(6)

    # Step 5: Save auxiliary outputs
    if output_task_difficulty_file is not None:
        task_df = extract_task_difficulty_summary(
            idata, data.task_ids, ci_level=confidence_level
        )
        output_task_difficulty_file.parent.mkdir(parents=True, exist_ok=True)
        task_df.to_csv(output_task_difficulty_file, index=False)
        logger.info(f"Saved task difficulties to {output_task_difficulty_file}")

    if output_hyperparams_file is not None:
        hyper = extract_hyperparameter_summary(idata)
        output_hyperparams_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_hyperparams_file, "w") as f:
            yaml.dump(hyper, f, sort_keys=False)
        logger.info(f"Saved hyperparameters to {output_hyperparams_file}")

    if output_diagnostics_file is not None:
        _save_diagnostics(idata, output_diagnostics_file)

    return regressions


def _save_diagnostics(idata: Any, output_file: pathlib.Path) -> None:
    """Save MCMC diagnostic summary to YAML."""
    import arviz as az

    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = az.summary(
        idata,
        var_names=["alpha", "beta", "mu_global", "sigma_global", "sigma_family", "sigma_epsilon"],
        round_to=4,
    )

    # Check convergence
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())

    diagnostics = {
        "convergence": {
            "r_hat_max": round(rhat_max, 4),
            "ess_bulk_min": round(ess_min, 1),
            "converged": bool(rhat_max < 1.05 and ess_min > 100),
        },
        "agent_params": {},
    }

    # Per-agent diagnostics for alpha and beta
    for col in summary.index:
        name = str(col)
        if name.startswith("alpha[") or name.startswith("beta["):
            diagnostics["agent_params"][name] = {
                "mean": round(float(summary.loc[col, "mean"]), 4),
                "sd": round(float(summary.loc[col, "sd"]), 4),
                "r_hat": round(float(summary.loc[col, "r_hat"]), 4),
                "ess_bulk": round(float(summary.loc[col, "ess_bulk"]), 1),
            }

    with open(output_file, "w") as f:
        yaml.dump(diagnostics, f, sort_keys=False)
    logger.info(f"Saved diagnostics to {output_file}")

    if rhat_max >= 1.05:
        logger.warning(f"CONVERGENCE WARNING: max R-hat = {rhat_max:.4f} >= 1.05")
    if ess_min < 100:
        logger.warning(f"CONVERGENCE WARNING: min ESS = {ess_min:.1f} < 100")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _write_metrics_file(
    regressions: pd.DataFrame, output_metrics_file: pathlib.Path
) -> None:
    """Write per-agent metrics YAML (same format as the frequentist pipeline)."""
    metrics: dict[str, dict[str, float | dict[str, float]]] = {}
    bin_columns = [
        "1-4 min", "4-16 min", "16-64 min",
        "64-256 min", "256-960 min", "960-2880 min",
    ]
    for agent, row in regressions.set_index("agent").iterrows():
        agent_metrics: dict[str, float | dict[str, float]] = {
            "slope": round(float(row["coefficient"]), 3),
            "intercept": round(float(row["intercept"]), 3),
            "time_horizon_p50": round(float(row["p50"]), 3),
            "empirical_success_rates": {
                col: round(float(row[col]), 3) for col in bin_columns if col in row
            },
        }
        if "log_evidence" in row.index:
            agent_metrics["log_evidence"] = round(float(row["log_evidence"]), 3)
        metrics[str(agent)] = agent_metrics

    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_file, "w") as f:
        yaml.dump(metrics, f, sort_keys=False)
    logger.info(f"Wrote metrics file to {output_metrics_file}")


def main(
    fig_name: str,
    runs_file: pathlib.Path,
    output_logistic_fits_file: pathlib.Path,
    release_dates: pathlib.Path,
    output_metrics_file: pathlib.Path | None = None,
) -> None:
    import dvc.api

    params = dvc.api.params_show(stages="wrangle_logistic_regression", deps=True)
    wrangle_params = params["figs"]["wrangle_logistic"][fig_name]

    runs = pd.read_json(runs_file, lines=True, orient="records", convert_dates=False)
    logger.info(f"Loaded {len(runs)} runs")

    regressions = run_bayesian_regressions(
        runs, release_dates, wrangle_params, include_empirical_rates=True
    )
    logger.info("\n" + str(regressions))
    logger.info(f"Mean BCE loss: {regressions.bce_loss.mean():.3f}")

    output_logistic_fits_file.parent.mkdir(parents=True, exist_ok=True)
    regressions.to_csv(output_logistic_fits_file)
    logger.info(f"Saved Bayesian fits to {output_logistic_fits_file}")

    if output_metrics_file is not None:
        _write_metrics_file(regressions, output_metrics_file)


def main_hierarchical(
    fig_name: str,
    runs_file: pathlib.Path,
    output_logistic_fits_file: pathlib.Path,
    release_dates: pathlib.Path,
    output_metrics_file: pathlib.Path | None = None,
    output_task_difficulty_file: pathlib.Path | None = None,
    output_hyperparams_file: pathlib.Path | None = None,
    output_diagnostics_file: pathlib.Path | None = None,
    n_samples: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> None:
    """CLI entry point for the hierarchical Bayesian model."""
    import dvc.api

    params = dvc.api.params_show(stages="wrangle_logistic_regression", deps=True)
    wrangle_params = params["figs"]["wrangle_logistic"][fig_name]

    runs = pd.read_json(runs_file, lines=True, orient="records", convert_dates=False)
    logger.info(f"Loaded {len(runs)} runs")

    regressions = run_hierarchical_bayesian(
        runs,
        release_dates,
        wrangle_params,
        n_samples=n_samples,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept,
        random_seed=random_seed,
        include_empirical_rates=True,
        output_task_difficulty_file=output_task_difficulty_file,
        output_hyperparams_file=output_hyperparams_file,
        output_diagnostics_file=output_diagnostics_file,
    )
    logger.info("\n" + str(regressions))
    logger.info(f"Mean BCE loss: {regressions.bce_loss.mean():.3f}")

    output_logistic_fits_file.parent.mkdir(parents=True, exist_ok=True)
    regressions.to_csv(output_logistic_fits_file)
    logger.info(f"Saved hierarchical Bayesian fits to {output_logistic_fits_file}")

    if output_metrics_file is not None:
        _write_metrics_file(regressions, output_metrics_file)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bayesian logistic regression for time horizon estimation"
    )
    sub = parser.add_subparsers(dest="mode", help="Model variant")

    # Original (grid-based) Bayesian model
    grid = sub.add_parser("grid", help="Grid-based 2D exact posterior (default)")
    grid.add_argument("--fig-name", type=str, required=True)
    grid.add_argument("--runs-file", type=pathlib.Path, required=True)
    grid.add_argument("--output-logistic-fits-file", type=pathlib.Path, required=True)
    grid.add_argument("--release-dates", type=pathlib.Path, required=True)
    grid.add_argument("--output-metrics-file", type=pathlib.Path, default=None)
    grid.add_argument("-v", "--verbose", action="store_true")

    # Hierarchical Bayesian model
    hier = sub.add_parser(
        "hierarchical", help="Hierarchical model with run-level baselines"
    )
    hier.add_argument("--fig-name", type=str, required=True)
    hier.add_argument("--runs-file", type=pathlib.Path, required=True)
    hier.add_argument("--output-logistic-fits-file", type=pathlib.Path, required=True)
    hier.add_argument("--release-dates", type=pathlib.Path, required=True)
    hier.add_argument("--output-metrics-file", type=pathlib.Path, default=None)
    hier.add_argument("--output-task-difficulty-file", type=pathlib.Path, default=None)
    hier.add_argument("--output-hyperparams-file", type=pathlib.Path, default=None)
    hier.add_argument("--output-diagnostics-file", type=pathlib.Path, default=None)
    hier.add_argument("--n-samples", type=int, default=1000)
    hier.add_argument("--n-tune", type=int, default=1000)
    hier.add_argument("--n-chains", type=int, default=2)
    hier.add_argument("--target-accept", type=float, default=0.9)
    hier.add_argument("--random-seed", type=int, default=42)
    hier.add_argument("-v", "--verbose", action="store_true")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")

    mode = args.pop("mode", None)
    verbose = args.pop("verbose", False)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    if mode == "hierarchical":
        main_hierarchical(**args)
    else:
        # Default to grid mode (backwards compatible)
        # Remove hierarchical-only args if present
        for key in [
            "output_task_difficulty_file", "output_hyperparams_file",
            "output_diagnostics_file", "n_samples", "n_tune",
            "n_chains", "target_accept", "random_seed",
        ]:
            args.pop(key, None)
        main(**args)
