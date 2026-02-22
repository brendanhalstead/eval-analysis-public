# Review: Hierarchical Bayesian Model (Jaynesian Perspective)

## Overview

The non-hierarchical model in `bayesian.py` is rigorous and self-aware about its
Jaynesian foundations.  The hierarchical extension in `bayesian_hierarchical.py`
introduces several ad-hockeries and one genuine conceptual flaw.  Organized from
most to least consequential.


## 1. Silent truncation of posterior mass at T → ∞ (conceptual flaw)

**Location**: `src/horizon/utils/bayesian_hierarchical.py:319-334`

In `extract_agent_results`, when computing time horizons from posterior samples:

```python
valid = np.isfinite(T) & (T > 0)
T_valid = T[valid]
```

This silently discards posterior samples where T is infinite or negative (which
occurs when `beta` samples are near zero and `logit(q) - alpha` has the wrong
sign).  The quantiles are then computed over only the "valid" subset.

**Why this is wrong**: Those divergent samples represent genuine posterior belief
that the agent *would never reach that success rate* at any finite time horizon.
Conditioning them away changes the meaning of the reported credible intervals —
you're reporting P(T ∈ [a,b] | T is finite), not P(T ∈ [a,b]).  The posterior
fraction `1 - len(T_valid)/len(T)` is the probability that the horizon is
undefined, and it should be reported rather than swept under the rug.

The non-hierarchical model (`bayesian.py:456`) has the same structural issue but
at least operates on weighted grid cells where the discarded mass is explicitly
computable.  Here, with MCMC samples, you should at minimum report
`P(T = ∞ | data)` alongside the median and CIs.


## 2. Binarization of continuous scores discards information (MaxEnt violation)

**Location**: `src/horizon/utils/bayesian_hierarchical.py:223-227`

```python
pm.Bernoulli("agent_obs", p=p_success, observed=data.agent_scores)
```

The `agent_scores` come from `score_binarized`, which collapses continuous
scores into {0, 1}.  The plan acknowledges this as Open Question 2, but from a
Jaynesian perspective it's not optional — it's a violation of the desideratum
that the model should use all available information.

The Bernoulli is MaxEnt for binary outcomes given E[Y·η], but if the actual
outcomes are continuous on [0, 1], the MaxEnt likelihood is Beta or at minimum
allows fractional y in the cross-entropy (which the non-hierarchical model
already supports — `bayesian.py:9-10` explicitly notes this).  The hierarchical
model regresses by requiring binarization that the non-hierarchical model was
designed to avoid.


## 3. Expert estimates assumed unbiased (unstated strong assumption)

**Location**: `src/horizon/utils/bayesian_hierarchical.py:29-32`

```python
log2(estimate_i) ~ Normal(mu_task_i, sigma_estimate)
```

This asserts that expert estimates are *unbiased* observations of true task
difficulty, differing only by having wider noise than actual run times.  But
expert time-estimates are notoriously biased — typically optimistic (planning
fallacy).  A more honest model:

```
log2(estimate_i) ~ Normal(mu_task_i + delta_estimate, sigma_estimate)
delta_estimate ~ Normal(0, 1)
```

With 17 estimate-only tasks and presumably some tasks that have both estimates
and baselines, `delta_estimate` is partially identifiable.  Even if you decide
to fix it at zero, that decision should be explicit and defended, not implicit.


## 4. HalfNormal scale priors are not MaxEnt-justified (inconsistency)

The non-hierarchical model makes an explicit appeal to MaxEnt for every prior
choice (Jaynes Theorems 11.3, Ch. 12).  The hierarchical model uses HalfNormal
for all five scale parameters without any MaxEnt argument.

For a scale parameter σ > 0 with no information beyond positivity and a finite
expected value, the MaxEnt distribution is Exponential, not HalfNormal.  For a
scale parameter where you also know the variance, it's a Gamma.  For a true
"no information" prior, Jaynes' transformation-group argument gives the Jeffreys
prior 1/σ (improper but usable in hierarchical models where the data identify
it).

HalfNormal is a reasonable default — it penalizes very large scales and has nice
computational properties in NUTS — but calling it Jaynesian requires stating what
constraints it maximizes entropy subject to.  The honest justification is
computational convenience + weak regularization, which is fine, but breaks the
principled-prior narrative established in `bayesian.py`.


## 5. Single sigma_human — exchangeability assumption on human variability

**Location**: `src/horizon/utils/bayesian_hierarchical.py:180-188`

A single `sigma_human` governs all tasks.  This assumes human completion-time
variability (in log2 space) is the same for every task.  A 5-minute debugging
task and a 12-hour reverse-engineering challenge probably don't have the same σ.

With 539 successful runs, a per-family or per-source `sigma_human` would be
identifiable:

```
sigma_human_source ~ HalfNormal(1.5)  # one per {HCAST, SWAA}
```

The plan notes this as Open Question 3 and dismisses it as "probably overkill,"
but from a Jaynesian standpoint: if you have 467 HCAST and 235 SWAA runs, you
have enough data to *let the model tell you* whether they differ, rather than
assuming they don't.  Jaynes' general rule: don't assume what you can estimate.


## 6. Human time as sufficient statistic for AI difficulty (structural assumption)

The core model equation:

```
P(success) = logistic(alpha_a + beta_a * mu_task_i)
```

asserts that the latent human-time difficulty `mu_task_i` is the *only* task
feature that matters for predicting agent success.  This is the same assumption
as the non-hierarchical model, but the hierarchical framing makes it more
visible: you're jointly estimating `mu_task_i` and then using it as the sole
predictor.

If some tasks are "hard for humans, easy for AI" (mass text search, parallel
exploration) or vice versa (tasks requiring physical intuition, social
reasoning), then `mu_task` is not sufficient and the model is systematically
misspecified.  The family-level hierarchy partially addresses this (families
might capture human-vs-AI difficulty mismatches), but the logistic link has no
task-level residual term — every task in a family with the same `mu_task` gets
the same predicted success probability.

This isn't fixable without a more fundamental model redesign, but it should be
stated as a known limitation.


## 7. Minor: plan vs. code prior discrepancy

The plan says `mu_global ~ Normal(5, 3)` ("~32 min") but the code has
`mu_global_mean: float = 6.0` and the docstring says `Normal(6, 3)` ("~64 min").
With σ=3 this is a negligible difference (the posteriors will be data-dominated),
but the inconsistency suggests the plan wasn't updated when the prior was.


## What's done well

The model gets the big things right:

- **Latent task difficulty propagating uncertainty** into agent predictions —
  this is exactly what the non-hierarchical model couldn't do.
- **Truncation of beta at 0** — correctly encoding domain knowledge.
- **Partial pooling via family hierarchy** — replacing the ad-hoc 1/√n weighting.
- **Separation of model specification from inference** — clean software design.
- **Using run-level data instead of collapsing to geomeans** — the core
  motivation is sound.

The model is a substantial improvement over both the frequentist pipeline and the
non-hierarchical Bayesian model.  The issues above are refinements, not reasons
to reject the approach — except for #1 (silent truncation), which should be fixed
before trusting the credible intervals.
