"""Bayesian logistic model for time horizon estimation.

Theoretical foundation (following Jaynes, 2003, "Probability Theory: The Logic
of Science"):

1. LIKELIHOOD — derived from Maximum Entropy.

   The logistic function sigma(eta) = 1/(1+exp(-eta)) is the maximum entropy
   distribution on {0, 1} subject to a constraint on E[Y * eta], where eta is
   a linear function of the features (Jaynes Ch. 11).  With eta = alpha + beta *
   log2(h), this encodes exactly two things:

     (a) Success probability depends on task duration h only through log2(h).
     (b) No other distributional structure is assumed.

   The log2 transformation respects scale invariance: the model is equivariant
   under rescaling of time units.  This is the transformation-group argument
   Jaynes uses for Jeffreys priors on scale parameters (Ch. 12).  If doubling
   from 10 to 20 minutes is the "same kind of step" as doubling from 100 to 200
   minutes, then log(h) — not h — is the natural coordinate.

   Unlike the sklearn splitting trick (duplicating fractional scores into weighted
   binary rows), the cross-entropy loss  y * eta - log(1 + exp(eta))  is valid
   for any y in [0, 1] directly.  The splitting trick is mathematically equivalent
   but adds unnecessary data manipulation.

2. PRIORS — encoding what we know and nothing more.

   alpha (intercept = log-odds of success at h = 1 minute):
       N(0, sigma_alpha^2).  MaxEnt distribution for a location parameter given
       only a mean and variance (Jaynes Theorem 11.3).  sigma_alpha = 5 covers
       log-odds from -15 to +15 at 3-sigma, i.e., success probabilities from
       ~3e-7 to ~1 for trivial tasks.  Deliberately weak.

   beta (slope = change in log-odds per doubling of task duration):
       N(mu_beta, sigma_beta^2) truncated to beta <= 0.  The truncation is
       mandatory — we KNOW longer tasks are harder.  Not encoding this violates
       the desideratum of consistency (Jaynes Ch. 1).  The Gaussian form is
       MaxEnt given a mean and variance; the truncation encodes the hard sign
       constraint from domain knowledge.

   Contrast with the existing approach: sklearn's L2 regularization with C = 10
   is mathematically a N(0, sqrt(10)) prior on BOTH parameters identically —
   it encodes nothing about the sign of beta and uses the same scale for
   quantities with different physical meanings.

3. POSTERIOR — computed exactly, not approximated.

   For a 2-parameter model, we compute the posterior on an adaptive grid:

     (a) Find the MAP (posterior mode) via L-BFGS-B with bound beta <= 0.
     (b) Compute the Hessian at the MAP analytically — this gives the Laplace
         approximation covariance, used ONLY to set the grid range (MAP +/- 6 sigma).
     (c) Evaluate the exact unnormalized log-posterior on a 150x150 grid.
     (d) Normalize via log-sum-exp.

   This replaces the bootstrap entirely.  No resampling, no refitting, no
   convergence concerns.  The result is the exact posterior (up to grid
   discretization, which is negligible at 150 points per dimension).

4. DERIVED QUANTITIES — via change of variables.

   The time horizon T(q) = 2^{(logit(q) - alpha) / beta} is a deterministic
   function of (alpha, beta).  The joint posterior induces a posterior over T(q)
   by the standard change-of-variables formula.  We evaluate T(q) on the grid
   and compute its weighted CDF to get the posterior median and credible interval.

   This gives a direct probability statement: P(T in [a, b] | data) = 0.95.
   The frequentist bootstrap CI means something different and less useful.

5. MODEL EVIDENCE — for prior sensitivity and model comparison.

   The log marginal likelihood log p(data | model) falls out of the grid
   normalization for free.  This enables principled prior sensitivity analysis
   via Bayes factors rather than ad hoc robustness checks.

6. HIERARCHICAL EXTENSION (future).

   The 1/sqrt(n_tasks_in_family) weighting in the existing pipeline is an ad hoc
   approximation to what a hierarchical model does automatically.  A family
   random effect gamma_f ~ N(0, tau^2) would let each family's effective
   contribution scale naturally with its information content (~sqrt(n) in the
   balanced case), without manual tuning.  This requires MCMC for the
   higher-dimensional posterior; the grid approach here is the foundation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special


# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianPrior:
    """Prior hyperparameters for Bayesian logistic regression.

    Every default is justified from first principles — see module docstring.
    """

    # alpha ~ N(alpha_mean, alpha_std^2)
    alpha_mean: float = 0.0
    alpha_std: float = 5.0

    # beta ~ N(beta_mean, beta_std^2) truncated to (-inf, beta_upper]
    beta_mean: float = -0.5
    beta_std: float = 1.5
    beta_upper: float = 0.0  # hard constraint: longer tasks are harder


# ---------------------------------------------------------------------------
# Log-likelihood, log-prior, log-posterior
# ---------------------------------------------------------------------------


def _log_likelihood_vector(
    alpha: NDArray[Any],
    beta: NDArray[Any],
    x: NDArray[Any],
    y: NDArray[Any],
    w: NDArray[Any],
) -> NDArray[Any]:
    """Weighted log-likelihood, vectorized over a beta grid.

    Parameters
    ----------
    alpha : scalar
        Intercept value.
    beta : (n_beta,) array
        Slope values to evaluate over.
    x : (n_data,) array
        log2(human_minutes) for each observation.
    y : (n_data,) array
        Scores in [0, 1].
    w : (n_data,) array
        Sample weights (need not sum to 1 here).

    Returns
    -------
    (n_beta,) array of log-likelihood values.
    """
    # eta shape: (n_beta, n_data)
    eta = alpha + np.outer(beta, x)
    # log-likelihood: sum_i w_i [y_i * eta_i - log(1 + exp(eta_i))]
    # np.logaddexp(0, eta) = log(1 + exp(eta)), numerically stable
    return np.sum(w * (y * eta - np.logaddexp(0, eta)), axis=1)


def _log_prior_alpha(alpha: float, prior: BayesianPrior) -> float:
    """Log prior density for alpha (Gaussian, up to normalizing constant)."""
    return -0.5 * ((alpha - prior.alpha_mean) / prior.alpha_std) ** 2


def _log_prior_beta_vector(beta: NDArray[Any], prior: BayesianPrior) -> NDArray[Any]:
    """Log prior density for beta (truncated Gaussian), vectorized.

    Returns -inf for beta > beta_upper (hard constraint).
    """
    lp = -0.5 * ((beta - prior.beta_mean) / prior.beta_std) ** 2
    lp[beta > prior.beta_upper] = -np.inf
    return lp


def _neg_log_posterior(
    params: NDArray[Any],
    x: NDArray[Any],
    y: NDArray[Any],
    w: NDArray[Any],
    prior: BayesianPrior,
) -> float:
    """Negative log-posterior for MAP optimization."""
    alpha, beta = params[0], params[1]
    eta = alpha + beta * x
    ll = np.sum(w * (y * eta - np.logaddexp(0, eta)))
    lp = _log_prior_alpha(alpha, prior) + (
        -0.5 * ((beta - prior.beta_mean) / prior.beta_std) ** 2
    )
    return -(ll + lp)


def _neg_log_posterior_grad(
    params: NDArray[Any],
    x: NDArray[Any],
    y: NDArray[Any],
    w: NDArray[Any],
    prior: BayesianPrior,
) -> NDArray[Any]:
    """Gradient of negative log-posterior."""
    alpha, beta = params[0], params[1]
    eta = alpha + beta * x
    sigma_eta = special.expit(eta)
    residual = w * (y - sigma_eta)
    d_alpha = np.sum(residual) - (alpha - prior.alpha_mean) / prior.alpha_std**2
    d_beta = np.sum(residual * x) - (beta - prior.beta_mean) / prior.beta_std**2
    return -np.array([d_alpha, d_beta])


# ---------------------------------------------------------------------------
# Hessian (analytic) for the Laplace approximation
# ---------------------------------------------------------------------------


def _hessian_log_posterior(
    params: NDArray[Any],
    x: NDArray[Any],
    w: NDArray[Any],
    prior: BayesianPrior,
) -> NDArray[Any]:
    """Analytic Hessian of the log-posterior at a given point.

    The log-posterior for logistic regression is globally concave (the
    log-likelihood is concave and the Gaussian log-prior is concave), so
    this Hessian is negative-definite everywhere.
    """
    alpha, beta = params[0], params[1]
    eta = alpha + beta * x
    sigma_eta = special.expit(eta)
    v = w * sigma_eta * (1 - sigma_eta)  # Fisher information weights
    return np.array(
        [
            [-np.sum(v) - 1.0 / prior.alpha_std**2, -np.sum(v * x)],
            [-np.sum(v * x), -np.sum(v * x**2) - 1.0 / prior.beta_std**2],
        ]
    )


# ---------------------------------------------------------------------------
# MAP estimation
# ---------------------------------------------------------------------------


def find_map(
    x: NDArray[Any],
    y: NDArray[Any],
    w: NDArray[Any],
    prior: BayesianPrior,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Find the MAP estimate and Laplace covariance.

    Returns
    -------
    map_params : (2,) array  [alpha_map, beta_map]
    laplace_cov : (2, 2) array  covariance from the inverse negative Hessian
    """
    x0 = np.array([prior.alpha_mean, min(prior.beta_mean, prior.beta_upper - 0.1)])

    result = optimize.minimize(
        _neg_log_posterior,
        x0,
        args=(x, y, w, prior),
        jac=_neg_log_posterior_grad,
        method="L-BFGS-B",
        bounds=[(None, None), (None, prior.beta_upper)],
    )

    map_params = result.x
    H = _hessian_log_posterior(map_params, x, w, prior)

    # Negative Hessian should be positive-definite; if numerics fail, fall back
    # to a diagonal approximation.
    try:
        laplace_cov = np.linalg.inv(-H)
        # Sanity: variances must be positive
        if laplace_cov[0, 0] <= 0 or laplace_cov[1, 1] <= 0:
            raise np.linalg.LinAlgError("Non-positive variance in Laplace covariance")
    except np.linalg.LinAlgError:
        laplace_cov = np.diag([prior.alpha_std**2, prior.beta_std**2])

    return map_params, laplace_cov


# ---------------------------------------------------------------------------
# Exact grid posterior
# ---------------------------------------------------------------------------


def compute_posterior_grid(
    x: NDArray[Any],
    y: NDArray[Any],
    w: NDArray[Any],
    prior: BayesianPrior,
    n_grid: int = 150,
    n_sigmas: float = 6.0,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], float]:
    """Evaluate the posterior on an adaptive grid and normalize.

    The grid is centered on the MAP with extent set by the Laplace
    approximation — this is just for choosing where to evaluate, not an
    approximation to the posterior itself.

    Parameters
    ----------
    x : (n,) array of log2(human_minutes).
    y : (n,) array of scores in [0, 1].
    w : (n,) array of sample weights.
    prior : BayesianPrior specification.
    n_grid : number of grid points per dimension.
    n_sigmas : how many Laplace sigmas to extend the grid.

    Returns
    -------
    alpha_grid : (n_grid,) array
    beta_grid : (n_grid,) array
    log_cell_prob : (n_grid, n_grid) array of log P(cell | data),
        where each cell is a small rectangle d_alpha * d_beta.
        Sums to 0 in log-space (i.e. exp sums to 1).
    log_evidence : scalar, log marginal likelihood log p(data | model).
    """
    map_params, laplace_cov = find_map(x, y, w, prior)
    alpha_map, beta_map = map_params[0], map_params[1]
    alpha_sigma = np.sqrt(laplace_cov[0, 0])
    beta_sigma = np.sqrt(laplace_cov[1, 1])

    alpha_lo = alpha_map - n_sigmas * alpha_sigma
    alpha_hi = alpha_map + n_sigmas * alpha_sigma
    beta_lo = beta_map - n_sigmas * beta_sigma
    beta_hi = min(prior.beta_upper, beta_map + n_sigmas * beta_sigma)

    # Ensure beta grid has room (if MAP is right at the boundary)
    if beta_hi - beta_lo < 1e-6:
        beta_lo = prior.beta_mean - n_sigmas * prior.beta_std
        beta_hi = prior.beta_upper

    alpha_grid = np.linspace(alpha_lo, alpha_hi, n_grid)
    beta_grid = np.linspace(beta_lo, beta_hi, n_grid)

    # Precompute beta prior on the grid
    log_prior_beta = _log_prior_beta_vector(beta_grid, prior)

    # Evaluate log-posterior on the grid.
    # Loop over alpha (outer), vectorize over beta (inner).
    # Memory per iteration: n_grid * n_data floats — very manageable.
    log_post = np.empty((n_grid, n_grid))
    for i in range(n_grid):
        lp_alpha = _log_prior_alpha(alpha_grid[i], prior)
        ll = _log_likelihood_vector(alpha_grid[i], beta_grid, x, y, w)
        log_post[i, :] = ll + lp_alpha + log_prior_beta

    # Normalize.  Each grid cell has area d_alpha * d_beta.
    # Cell probability: p_ij = posterior(a_i, b_j) * d_alpha * d_beta
    # log p_ij = log_post[i,j] + log(d_alpha) + log(d_beta)
    d_alpha = alpha_grid[1] - alpha_grid[0] if n_grid > 1 else 1.0
    d_beta = beta_grid[1] - beta_grid[0] if n_grid > 1 else 1.0

    log_cell_unnorm = log_post + np.log(d_alpha) + np.log(d_beta)
    log_evidence = float(special.logsumexp(log_cell_unnorm))
    log_cell_prob = log_cell_unnorm - log_evidence

    return alpha_grid, beta_grid, log_cell_prob, log_evidence


# ---------------------------------------------------------------------------
# Posterior marginals
# ---------------------------------------------------------------------------


def posterior_marginals(
    alpha_grid: NDArray[Any],
    beta_grid: NDArray[Any],
    log_cell_prob: NDArray[Any],
) -> dict[str, float]:
    """Compute marginal posterior means, standard deviations, and correlation."""
    prob = np.exp(log_cell_prob)

    # Marginal over beta (sum columns)
    marg_alpha = prob.sum(axis=1)
    marg_alpha /= marg_alpha.sum()
    alpha_mean = float(np.sum(alpha_grid * marg_alpha))
    alpha_std = float(np.sqrt(np.sum((alpha_grid - alpha_mean) ** 2 * marg_alpha)))

    # Marginal over alpha (sum rows)
    marg_beta = prob.sum(axis=0)
    marg_beta /= marg_beta.sum()
    beta_mean = float(np.sum(beta_grid * marg_beta))
    beta_std = float(np.sqrt(np.sum((beta_grid - beta_mean) ** 2 * marg_beta)))

    # Correlation
    A, B = np.meshgrid(alpha_grid, beta_grid, indexing="ij")
    cov_ab = float(np.sum(prob * (A - alpha_mean) * (B - beta_mean)))
    denom = alpha_std * beta_std
    corr = cov_ab / denom if denom > 0 else 0.0

    return {
        "alpha_posterior_mean": alpha_mean,
        "alpha_posterior_std": alpha_std,
        "beta_posterior_mean": beta_mean,
        "beta_posterior_std": beta_std,
        "posterior_correlation": corr,
    }


# ---------------------------------------------------------------------------
# Time horizon posterior
# ---------------------------------------------------------------------------


def horizon_posterior_quantiles(
    alpha_grid: NDArray[Any],
    beta_grid: NDArray[Any],
    log_cell_prob: NDArray[Any],
    success_fraction: float,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Posterior median and credible interval for the time horizon T(q).

    T(q) = 2^{(logit(q) - alpha) / beta}  [in minutes]

    Parameters
    ----------
    success_fraction : e.g. 0.50 for p50, 0.80 for p80.
    ci_level : e.g. 0.95 for a 95% credible interval.

    Returns
    -------
    (posterior_median, ci_lower, ci_upper)  all in minutes.
    """
    logit_q = np.log(success_fraction / (1 - success_fraction))
    A, B = np.meshgrid(alpha_grid, beta_grid, indexing="ij")

    with np.errstate(divide="ignore", invalid="ignore"):
        log2_T = (logit_q - A) / B
        T = np.exp2(log2_T)

    T_flat = T.ravel()
    w_flat = np.exp(log_cell_prob.ravel())

    # Keep only finite, positive horizons with nonzero posterior weight
    valid = np.isfinite(T_flat) & (T_flat > 0) & (w_flat > 1e-300)
    T_valid = T_flat[valid]
    w_valid = w_flat[valid]

    if len(T_valid) == 0:
        return 0.0, 0.0, 0.0

    w_valid = w_valid / w_valid.sum()

    # Sort and build CDF
    order = np.argsort(T_valid)
    T_sorted = T_valid[order]
    cdf = np.cumsum(w_valid[order])

    low_q = (1 - ci_level) / 2
    high_q = 1 - low_q

    def _quantile(p: float) -> float:
        idx = np.searchsorted(cdf, p, side="left")
        idx = min(idx, len(T_sorted) - 1)
        return float(T_sorted[idx])

    return _quantile(0.5), _quantile(low_q), _quantile(high_q)


# ---------------------------------------------------------------------------
# BCE at a point estimate (for comparison with the frequentist pipeline)
# ---------------------------------------------------------------------------


def bce_loss_at_params(
    alpha: float,
    beta: float,
    x: NDArray[Any],
    y: NDArray[Any],
    w: NDArray[Any],
) -> float:
    """Weighted binary cross-entropy at given parameter values.

    Matches the metric reported by the existing pipeline for comparability.
    """
    eta = alpha + beta * x
    y_pred = special.expit(eta)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    w_norm = w / w.mean()
    bce = -w_norm * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return float(np.mean(bce))


# ---------------------------------------------------------------------------
# Prior sensitivity analysis
# ---------------------------------------------------------------------------


def compare_priors(
    x: NDArray[Any],
    y: NDArray[Any],
    w: NDArray[Any],
    priors: dict[str, BayesianPrior],
    success_percents: list[int] | None = None,
    ci_level: float = 0.95,
    n_grid: int = 150,
) -> dict[str, dict[str, Any]]:
    """Run the model under multiple priors and compare via log-evidence.

    This is the principled way to assess prior sensitivity: the Bayes factor
    K = p(data | prior_1) / p(data | prior_2) quantifies the evidence in
    favour of one prior specification over another (Jaynes Ch. 20).

    A |log K| < 1 means the data barely distinguish the two priors.
    A |log K| > 3 means one prior is strongly favoured.

    Parameters
    ----------
    priors : dict mapping descriptive label -> BayesianPrior.
    success_percents : e.g. [50, 80].  If None, defaults to [50].

    Returns
    -------
    dict mapping label -> {
        "log_evidence": float,
        "map_alpha": float,
        "map_beta": float,
        "alpha_posterior_mean": float,
        "beta_posterior_mean": float,
        "horizons": {p: (median, lower, upper), ...},
    }
    """
    if success_percents is None:
        success_percents = [50]

    # Rescale weights so the likelihood reflects the actual information content
    w = _rescale_weights(w)

    results: dict[str, dict[str, Any]] = {}

    for label, prior in priors.items():
        alpha_grid, beta_grid, log_cell_prob, log_ev = compute_posterior_grid(
            x, y, w, prior, n_grid=n_grid
        )

        map_params, _ = find_map(x, y, w, prior)
        marginals = posterior_marginals(alpha_grid, beta_grid, log_cell_prob)

        horizons = {}
        for p in success_percents:
            med, lo, hi = horizon_posterior_quantiles(
                alpha_grid, beta_grid, log_cell_prob, p / 100, ci_level
            )
            horizons[p] = {"median": med, "ci_lower": lo, "ci_upper": hi}

        results[label] = {
            "log_evidence": log_ev,
            "map_alpha": float(map_params[0]),
            "map_beta": float(map_params[1]),
            **marginals,
            "horizons": horizons,
        }

    # Add pairwise log Bayes factors relative to the first prior
    labels = list(results.keys())
    if len(labels) > 1:
        ref = results[labels[0]]["log_evidence"]
        for label in labels:
            results[label]["log_bayes_factor_vs_first"] = (
                results[label]["log_evidence"] - ref
            )

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _rescale_weights(w: NDArray[Any]) -> NDArray[Any]:
    """Rescale probability-measure weights to reflect effective sample size.

    The METR pipeline normalizes weights to sum to 1.0 (a probability measure).
    But in a Bayesian likelihood, each observation should contribute approximately
    one unit of evidence.  With weights summing to 1.0, the entire dataset looks
    like a single observation to the prior — which is why credible intervals
    blow up.

    We rescale so that the weights sum to the Kish effective sample size:

        N_eff = (sum w_i)^2 / sum(w_i^2) = 1 / sum(w_i^2)

    This preserves relative weighting while ensuring the total evidence reflects
    the actual information content of the weighted sample.  (For uniform weights,
    N_eff = N; for highly skewed weights, N_eff << N.)
    """
    w_sum = np.sum(w)
    if w_sum == 0:
        return w
    w_normalized = w / w_sum
    n_eff = 1.0 / np.sum(w_normalized**2)
    return w_normalized * n_eff


def bayesian_logistic_regression(
    x_raw: NDArray[Any],
    y: NDArray[Any],
    weights: NDArray[Any],
    prior: BayesianPrior | None = None,
    success_percents: list[int] | None = None,
    ci_level: float = 0.95,
    n_grid: int = 150,
) -> dict[str, float]:
    """Bayesian logistic regression for time horizon estimation.

    Drop-in replacement for the information produced by agent_regression()
    in wrangle/logistic.py, plus additional Bayesian quantities.

    Parameters
    ----------
    x_raw : human_minutes (NOT log-transformed).
    y : scores in [0, 1]  (binary or continuous — no splitting needed).
    weights : sample weights (from compute_task_weights).  These are
        automatically rescaled so the total evidence reflects the Kish
        effective sample size rather than being compressed to 1.
    prior : prior specification.  Uses principled defaults if None.
    success_percents : quantiles to compute horizons for, e.g. [50, 80].
    ci_level : credible interval level, e.g. 0.95.
    n_grid : posterior grid resolution per dimension.

    Returns
    -------
    dict with keys compatible with the existing pipeline:
        coefficient, intercept, bce_loss, average,
        p50, p50q0.025, p50q0.975,  (etc. for each success_percent)
    plus Bayesian-specific keys:
        alpha_posterior_mean, alpha_posterior_std,
        beta_posterior_mean, beta_posterior_std,
        posterior_correlation, log_evidence
    """
    if prior is None:
        prior = BayesianPrior()
    if success_percents is None:
        success_percents = [50, 80]

    x = np.log2(x_raw)
    weights = _rescale_weights(weights)

    # Handle degenerate case: agent never succeeds
    if np.all(y == 0):
        result: dict[str, float] = {
            "coefficient": -np.inf,
            "intercept": 0.0,
            "bce_loss": 0.0,
            "average": 0.0,
            "log_evidence": -np.inf,
        }
        low_q = (1 - ci_level) / 2
        high_q = 1 - low_q
        for p in success_percents:
            result[f"p{p}"] = 0.0
            result[f"p{p}q{low_q:.3f}"] = 0.0
            result[f"p{p}q{high_q:.3f}"] = 0.0
        return result

    # Compute the exact posterior on an adaptive grid
    alpha_grid, beta_grid, log_cell_prob, log_evidence = compute_posterior_grid(
        x, y, weights, prior, n_grid=n_grid
    )

    # MAP for compatibility (coefficient, intercept are what the existing
    # pipeline reports — they correspond to the MAP under the Gaussian prior)
    map_params, _ = find_map(x, y, weights, prior)

    # Posterior marginal summaries
    marginals = posterior_marginals(alpha_grid, beta_grid, log_cell_prob)

    # BCE at MAP (for comparison with the frequentist fit)
    bce = bce_loss_at_params(map_params[0], map_params[1], x, y, weights)

    # Weighted average score
    average = float(np.sum(y * weights) / np.sum(weights))

    # Assemble result dict
    low_q = (1 - ci_level) / 2
    high_q = 1 - low_q

    result = {
        "coefficient": float(map_params[1]),  # beta (slope)
        "intercept": float(map_params[0]),  # alpha (intercept)
        "bce_loss": bce,
        "average": average,
        "log_evidence": log_evidence,
        **marginals,
    }

    # Time horizon posteriors
    for p in success_percents:
        median, lower, upper = horizon_posterior_quantiles(
            alpha_grid, beta_grid, log_cell_prob, p / 100, ci_level
        )
        result[f"p{p}"] = median
        result[f"p{p}q{low_q:.3f}"] = lower
        result[f"p{p}q{high_q:.3f}"] = upper

    return result
