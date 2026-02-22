"""Hierarchical Bayesian model for time horizon estimation.

Uses run-level human baseline data as first-class observations.  Instead of
collapsing multiple human runs into a single geometric mean per task and then
simulating noise, this model jointly infers:

  1. Per-task latent difficulty  mu_task_i  (in log2-minutes space)
  2. Per-agent success curves  (alpha_a, beta_a)
  3. Hyperparameters: family-level pooling, human variability

The model is estimated via PyMC's NUTS sampler.

Generative model
----------------

Hierarchy on task difficulty:

    mu_global    ~ Normal(6, 3)           # ~64 min, very weak
    sigma_global ~ HalfNormal(3)
    mu_family_f  ~ Normal(mu_global, sigma_global)
    sigma_family ~ HalfNormal(2)
    mu_task_i    ~ Normal(mu_family[f(i)], sigma_family)

Human observation model (successful baselines only):

    sigma_human ~ HalfNormal(1.5)
    log2(duration_ij) ~ Normal(mu_task_i, sigma_human)

Expert-estimate model (for tasks with no baselines):

    sigma_estimate ~ HalfNormal(2.5)
    log2(estimate_i) ~ Normal(mu_task_i, sigma_estimate)

Agent success model:

    alpha_a ~ Normal(0, 5)
    beta_a  ~ Normal(-0.5, 1.5), beta <= 0
    P(success) = logistic(alpha_a + beta_a * mu_task_i)
    score_aij ~ Bernoulli(P(success))

Derived quantity:

    T_a(q) = 2^{(logit(q) - alpha_a) / beta_a}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
from numpy.typing import NDArray
from scipy import special

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HierarchicalPrior:
    """Prior specification for the hierarchical model."""

    # Task difficulty hierarchy
    mu_global_mean: float = 6.0
    mu_global_std: float = 3.0
    sigma_global_scale: float = 3.0
    sigma_family_scale: float = 2.0

    # Human observation noise
    sigma_human_scale: float = 1.5

    # Expert estimate noise
    sigma_estimate_scale: float = 2.5

    # Agent success curves
    alpha_mean: float = 0.0
    alpha_std: float = 5.0
    beta_mean: float = -0.5
    beta_std: float = 1.5
    beta_upper: float = 0.0


@dataclass
class HierarchicalData:
    """Pre-processed data for the hierarchical model.

    All arrays use integer indices for tasks/families/agents
    to simplify PyMC indexing.
    """

    # Task metadata
    task_ids: list[str]
    family_ids: list[str]
    task_to_family_idx: NDArray[Any]  # (n_tasks,) int

    # Human baseline observations (successful runs)
    human_task_idx: NDArray[Any]     # (n_human,) int -> task index
    human_log2_duration: NDArray[Any]  # (n_human,) float

    # Expert estimates (tasks with no baselines)
    estimate_task_idx: NDArray[Any]  # (n_estimate,) int -> task index
    estimate_log2_minutes: NDArray[Any]  # (n_estimate,) float

    # Agent observations
    agent_names: list[str]
    agent_task_idx: NDArray[Any]     # (n_agent_obs,) int -> task index
    agent_agent_idx: NDArray[Any]    # (n_agent_obs,) int -> agent index
    agent_scores: NDArray[Any]       # (n_agent_obs,) float in [0, 1]

    n_tasks: int = field(init=False)
    n_families: int = field(init=False)
    n_agents: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_tasks = len(self.task_ids)
        self.n_families = len(self.family_ids)
        self.n_agents = len(self.agent_names)

    def summary(self) -> str:
        return (
            f"HierarchicalData: {self.n_tasks} tasks, {self.n_families} families, "
            f"{self.n_agents} agents, {len(self.human_log2_duration)} human obs, "
            f"{len(self.estimate_log2_minutes)} estimate obs, "
            f"{len(self.agent_scores)} agent obs"
        )


def build_hierarchical_model(
    data: HierarchicalData,
    prior: HierarchicalPrior | None = None,
) -> pm.Model:
    """Build the PyMC hierarchical model.

    Parameters
    ----------
    data : preprocessed HierarchicalData
    prior : prior specification (uses defaults if None)

    Returns
    -------
    PyMC Model (not yet sampled)
    """
    if prior is None:
        prior = HierarchicalPrior()

    coords = {
        "family": data.family_ids,
        "task": data.task_ids,
        "agent": data.agent_names,
    }

    with pm.Model(coords=coords) as model:
        # ---------------------------------------------------------------
        # Task difficulty hierarchy
        # ---------------------------------------------------------------
        mu_global = pm.Normal(
            "mu_global", mu=prior.mu_global_mean, sigma=prior.mu_global_std
        )
        sigma_global = pm.HalfNormal("sigma_global", sigma=prior.sigma_global_scale)

        mu_family = pm.Normal(
            "mu_family", mu=mu_global, sigma=sigma_global, dims="family"
        )

        sigma_family = pm.HalfNormal("sigma_family", sigma=prior.sigma_family_scale)

        mu_task = pm.Normal(
            "mu_task",
            mu=mu_family[data.task_to_family_idx],
            sigma=sigma_family,
            dims="task",
        )

        # ---------------------------------------------------------------
        # Human observation model
        # ---------------------------------------------------------------
        if len(data.human_log2_duration) > 0:
            sigma_human = pm.HalfNormal(
                "sigma_human", sigma=prior.sigma_human_scale
            )
            pm.Normal(
                "human_obs",
                mu=mu_task[data.human_task_idx],
                sigma=sigma_human,
                observed=data.human_log2_duration,
            )

        # ---------------------------------------------------------------
        # Expert estimate model
        # ---------------------------------------------------------------
        if len(data.estimate_log2_minutes) > 0:
            sigma_estimate = pm.HalfNormal(
                "sigma_estimate", sigma=prior.sigma_estimate_scale
            )
            pm.Normal(
                "estimate_obs",
                mu=mu_task[data.estimate_task_idx],
                sigma=sigma_estimate,
                observed=data.estimate_log2_minutes,
            )

        # ---------------------------------------------------------------
        # Agent success model
        # ---------------------------------------------------------------
        alpha = pm.Normal(
            "alpha", mu=prior.alpha_mean, sigma=prior.alpha_std, dims="agent"
        )

        # Truncated normal: beta <= 0
        beta = pm.TruncatedNormal(
            "beta",
            mu=prior.beta_mean,
            sigma=prior.beta_std,
            upper=prior.beta_upper,
            dims="agent",
        )

        eta = alpha[data.agent_agent_idx] + beta[data.agent_agent_idx] * mu_task[data.agent_task_idx]
        p_success = pm.math.sigmoid(eta)

        pm.Bernoulli(
            "agent_obs",
            p=p_success,
            observed=data.agent_scores,
        )

    return model


def sample_hierarchical_model(
    model: pm.Model,
    n_samples: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> Any:  # arviz.InferenceData
    """Run NUTS sampling on the hierarchical model.

    Parameters
    ----------
    model : built PyMC model
    n_samples : posterior samples per chain
    n_tune : tuning (warmup) samples per chain
    n_chains : number of independent chains
    target_accept : NUTS target acceptance rate
    random_seed : for reproducibility

    Returns
    -------
    arviz.InferenceData with posterior samples
    """
    with model:
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )
    return idata


def extract_agent_results(
    idata: Any,  # arviz.InferenceData
    agent_names: list[str],
    success_percents: list[int],
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Extract per-agent regression results from posterior samples.

    Produces a DataFrame in the same format as run_logistic_regressions()
    and run_bayesian_regressions() so downstream code works unchanged.

    Parameters
    ----------
    idata : arviz InferenceData with posterior samples
    agent_names : list of agent names matching the model coords
    success_percents : e.g. [50, 80]
    ci_level : credible interval level

    Returns
    -------
    DataFrame with one row per agent, columns matching the existing pipeline.
    """
    posterior = idata.posterior

    # Stack chains for easier quantile computation
    alpha_samples = posterior["alpha"].values.reshape(-1, len(agent_names))
    beta_samples = posterior["beta"].values.reshape(-1, len(agent_names))

    low_q = (1 - ci_level) / 2
    high_q = 1 - low_q

    rows = []
    for i, agent in enumerate(agent_names):
        a_samples = alpha_samples[:, i]
        b_samples = beta_samples[:, i]

        row: dict[str, Any] = {
            "agent": agent,
            "coefficient": float(np.median(b_samples)),
            "intercept": float(np.median(a_samples)),
            "bce_loss": float("nan"),  # not directly available from posterior
            "average": float("nan"),   # filled later from data
            "alpha_posterior_mean": float(np.mean(a_samples)),
            "alpha_posterior_std": float(np.std(a_samples)),
            "beta_posterior_mean": float(np.mean(b_samples)),
            "beta_posterior_std": float(np.std(b_samples)),
            "posterior_correlation": float(np.corrcoef(a_samples, b_samples)[0, 1]),
        }

        # Time horizon posteriors
        for p in success_percents:
            logit_q = np.log(p / 100 / (1 - p / 100))
            with np.errstate(divide="ignore", invalid="ignore"):
                log2_T = (logit_q - a_samples) / b_samples
                T = np.exp2(log2_T)

            # Filter to valid (finite, positive) values
            valid = np.isfinite(T) & (T > 0)
            T_valid = T[valid]

            if len(T_valid) > 0:
                row[f"p{p}"] = float(np.median(T_valid))
                row[f"p{p}q{low_q:.3f}"] = float(np.quantile(T_valid, low_q))
                row[f"p{p}q{high_q:.3f}"] = float(np.quantile(T_valid, high_q))
            else:
                row[f"p{p}"] = 0.0
                row[f"p{p}q{low_q:.3f}"] = 0.0
                row[f"p{p}q{high_q:.3f}"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def extract_task_difficulty_summary(
    idata: Any,
    task_ids: list[str],
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Extract per-task latent difficulty posteriors.

    Returns a DataFrame with columns:
        task_id, mu_task_median, mu_task_mean, mu_task_std,
        mu_task_lower, mu_task_upper, difficulty_minutes_median
    """
    posterior = idata.posterior
    mu_task_samples = posterior["mu_task"].values.reshape(-1, len(task_ids))

    low_q = (1 - ci_level) / 2
    high_q = 1 - low_q

    rows = []
    for i, task_id in enumerate(task_ids):
        samples = mu_task_samples[:, i]
        rows.append({
            "task_id": task_id,
            "mu_task_median": float(np.median(samples)),
            "mu_task_mean": float(np.mean(samples)),
            "mu_task_std": float(np.std(samples)),
            "mu_task_lower": float(np.quantile(samples, low_q)),
            "mu_task_upper": float(np.quantile(samples, high_q)),
            "difficulty_minutes_median": float(np.exp2(np.median(samples))),
        })

    return pd.DataFrame(rows)


def extract_hyperparameter_summary(idata: Any) -> dict[str, dict[str, float]]:
    """Extract hyperparameter posterior summaries."""
    posterior = idata.posterior
    result = {}

    for name in [
        "mu_global", "sigma_global", "sigma_family",
        "sigma_human", "sigma_estimate",
    ]:
        if name in posterior:
            samples = posterior[name].values.ravel()
            result[name] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "median": float(np.median(samples)),
                "q025": float(np.quantile(samples, 0.025)),
                "q975": float(np.quantile(samples, 0.975)),
            }

    return result


def compute_bce_from_posterior(
    idata: Any,
    data: HierarchicalData,
) -> dict[str, float]:
    """Compute posterior-mean BCE loss per agent (for comparison with frequentist).

    Uses the posterior mean of (alpha, beta, mu_task) to compute predicted
    success probabilities, then evaluates BCE against observed scores.
    """
    posterior = idata.posterior
    alpha_mean = posterior["alpha"].values.reshape(-1, data.n_agents).mean(axis=0)
    beta_mean = posterior["beta"].values.reshape(-1, data.n_agents).mean(axis=0)
    mu_task_mean = posterior["mu_task"].values.reshape(-1, data.n_tasks).mean(axis=0)

    bce_by_agent: dict[str, float] = {}
    for a_idx, agent in enumerate(data.agent_names):
        mask = data.agent_agent_idx == a_idx
        if not np.any(mask):
            continue
        task_idx = data.agent_task_idx[mask]
        scores = data.agent_scores[mask]
        eta = alpha_mean[a_idx] + beta_mean[a_idx] * mu_task_mean[task_idx]
        p = special.expit(eta)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        bce = -np.mean(scores * np.log(p) + (1 - scores) * np.log(1 - p))
        bce_by_agent[agent] = float(bce)

    return bce_by_agent
