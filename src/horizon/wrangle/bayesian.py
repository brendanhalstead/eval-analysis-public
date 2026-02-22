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
            "log_evidence": round(float(row["log_evidence"]), 3),
            "empirical_success_rates": {
                col: round(float(row[col]), 3) for col in bin_columns if col in row
            },
        }
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bayesian logistic regression for time horizon estimation"
    )
    parser.add_argument("--fig-name", type=str, required=True)
    parser.add_argument("--runs-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-logistic-fits-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-metrics-file", type=pathlib.Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    args = vars(get_parser().parse_args())
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO if args.pop("verbose") else logging.WARNING)
    main(**args)
