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


---

# Comparison with Moss's IRT Reanalysis

Reference: Jonas Moss, "METR's data can't distinguish between trajectories
(and 80% horizons are an order of magnitude off)"
https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T
Code: https://github.com/JonasMoss/metr-stats

Moss fits a Bayesian 2PL item response theory model to the same underlying
METR task data.  The two models answer overlapping but different questions,
and comparing them reveals structural gaps in ours.


## Side-by-side model comparison

| Feature | Our hierarchical model | Moss's 2PL IRT |
|---------|----------------------|-----------------|
| **Link function** | `logistic(α_a + β_a · μ_task_i)` | `logistic(a_j · (θ_i − b_j))` |
| **What varies by task** | Difficulty `μ_task_i` only | Difficulty `b_j` AND discrimination `a_j` |
| **What varies by agent** | Intercept `α_a` and slope `β_a` | Scalar ability `θ_i` only |
| **Time trend** | None — agents are independent | `θ_i ~ Normal(γ₀ + γ·date_i, σ_θ)` |
| **Task difficulty structure** | Family hierarchy: global → family → task | Regression on log(human_time) + random effect |
| **Human baseline treatment** | Latent variable (run-level durations observed) | Fixed covariate (human_minutes taken as known) |
| **Likelihood** | Bernoulli (one run = one observation) | Binomial (multiple attempts aggregated) |
| **Inference** | PyMC NUTS | Stan NUTS |
| **Trajectory comparison** | Not applicable | 4 models: linear, quadratic, power-law, saturating |
| **Identification** | Implicit via prior on (α, β) in physical units | Two anchor models with fixed θ |


## Key structural differences and their consequences


### 1. Missing per-task discrimination parameter (most consequential)

Our model: `η = α_a + β_a · μ_task_i`

Moss's model: `η = a_j · (θ_i − b_j)`

In our parameterization, the slope `β_a` is per-agent: each agent has its
own sensitivity to difficulty.  But ALL tasks are equally "discriminating" —
two tasks with the same `μ_task` produce identical predicted success
probabilities for any agent.

Moss adds a per-task discrimination `a_j ~ LogNormal(μ_log_a, σ_log_a)`.
This captures the fact that some tasks cleanly separate strong from weak
models (high `a_j`) while others are noisy indicators (low `a_j`).  His
posterior median for σ_log_a implies substantial task-to-task variation in
discrimination.

Our model can be rewritten as `η = β_a · (μ_task_i − (−α_a/β_a))`, which
looks like a 1PL (Rasch) model with per-agent discrimination `β_a` and
threshold `−α_a/β_a`.  In IRT terms, we're fitting a model where items all
have unit discrimination — agents differ in how steeply they respond to
difficulty, but every task at a given difficulty is equally informative.

This matters because it directly feeds into the typical-vs-marginal
distinction (see #2 below).  Without per-task discrimination, the model
can't properly capture how much tasks of the "same difficulty" vary in
their ability to predict agent success.


### 2. Typical vs. marginal horizons — the ~10x gap our model ignores

This is Moss's central empirical finding: at 80% success, there is roughly
an order of magnitude gap between:

- **Typical**: "pick a task of AVERAGE difficulty for its length — can the
  model solve it 80% of the time?"
- **Marginal**: "pick a RANDOM task of that length — what's the expected
  success rate?"

The gap arises because the logistic function is concave in the tails.  By
Jensen's inequality, E[logistic(x)] < logistic(E[x]) when x has nonzero
variance.  Hard tasks drag down the marginal more than easy tasks push it
up.

Moss finds this effect is large: one SD of unexplained difficulty (the
residual σ_b after regressing on log human_time) corresponds to a ~4.7×
multiplier in equivalent difficulty, and tasks of the same human time can
differ by 23× in AI difficulty.

Our model HAS the machinery for this: `mu_task_i` has residual variance
from `sigma_family` around the family mean, which is analogous to Moss's
`sigma_b`.  But `extract_agent_results` computes horizons using only
`T(q) = 2^{(logit(q) − α) / β}`, which is the TYPICAL horizon — it asks
"at what difficulty does this agent reach success rate q?" without
integrating over the distribution of task difficulties at that difficulty
level.

To compute the marginal horizon from our model's posterior, you would need
to integrate:

```
P_marginal(success | T, agent) = ∫ logistic(α_a + β_a · d) · p(d | T) dd
```

where `p(d | T)` is the posterior predictive distribution of `mu_task` for
a new task with human time T.  This integral will produce strictly lower
success probabilities than the typical calculation, and the gap grows with
the residual variance in task difficulty.

**Recommendation**: compute both.  The posterior already contains
`sigma_family` and `mu_family` samples; integrating over the predictive
distribution of mu_task for a hypothetical new task at duration T is
straightforward.  Report the difference as a measure of how much
task-difficulty heterogeneity matters.


### 3. No time-trend model — different question, but limits forecasting

Moss models ability as a function of release date:

```
θ_i ~ Normal(γ₀ + γ · date_i, σ_θ)
```

This is what enables his key finding: the data can't distinguish linear
from quadratic from power-law time trends (ELPD-LOO differs by <6 points),
even though they diverge dramatically in forecasts (2028–2034 for the
125-year crossing depending on trajectory shape).

Our model has no time-trend structure at all.  Each agent gets independent
(α_a, β_a) from a common prior.  This is appropriate for the stated goal
("what are the time horizons for these specific agents?") but it means the
model can't:

- Extrapolate to future models
- Estimate doubling times
- Compare trajectory shapes
- Quantify how much of the agent-to-agent variation is explained by
  release date vs. idiosyncratic capability

If forecasting is a goal, a time trend on the latent ability scale would
need to be added.  The natural extension: replace the independent prior on
(α_a, β_a) with a regression on release date, analogous to Moss's approach
but keeping our per-agent slope structure.


### 4. Human baseline treatment — our model is strictly better here

Moss uses `x_j = log(human_time_j) − mean(log human_time)` as a KNOWN
covariate.  Task difficulty is `b_j ~ Normal(α + κ · x_j, σ_b)`, so human
time feeds in as a fixed, exact number.

Our model treats human run durations as noisy observations of latent
difficulty:

```
log2(duration_ij) ~ Normal(mu_task_i, sigma_human)
```

This is more principled.  Tasks with 1 baseline run have wide posterior
uncertainty on `mu_task`; tasks with 57 runs have tight posteriors.  This
uncertainty propagates into the agent success predictions automatically.

Moss acknowledges this limitation in the post: "human baseline times are
treated as known rather than estimated via latent variables — acknowledged
as introducing artificially narrow credible intervals."

Alexander Barry (METR statistician) reinforces this in the comments: "only
5/30 of the longest tasks currently have baselined human times" and
"only 60% of estimates were within a factor of 3" of actual baseline times.
This is exactly the problem our latent-variable approach with separate
`sigma_estimate` was designed to handle.


### 5. Per-agent slopes vs. scalar ability — expressiveness tradeoff

Moss: each agent has a single scalar ability `θ_i`.  Given θ, the agent's
predicted success on every task is determined — there's no room for an
agent to be "better at short tasks but worse at long ones" beyond what the
per-task discrimination captures.

Our model: each agent has `(α_a, β_a)`, so an agent CAN have a different
intercept-to-slope ratio.  One agent might be excellent on easy tasks
(high α) with rapid degradation on hard ones (very negative β), while
another is mediocre on easy tasks but degrades slowly.

This is more flexible but comes at a cost: 2 parameters per agent instead
of 1, with no hierarchical structure to share information across agents.
Moss's scalar-ability model is more parsimonious and lets the per-task
parameters (a_j, b_j) absorb the task-level variation.

In practice, the per-agent slope is important for this application because
the "time horizon" IS the ratio −α/β, so agents with different slopes
have qualitatively different horizon structures even at the same average
success rate.  Moss's model computes horizons differently (via the
relationship between θ and the difficulty scale), so the approaches aren't
directly comparable here.


### 6. What METR's two-stage approach implicitly computes

Moss and the comments illuminate an important subtlety about what our
non-hierarchical model (and METR's original approach) actually estimates.

METR fits a separate logistic regression per agent: P(success) = logistic(α
+ β · log2(human_minutes)).  When you fit this to a mix of easy and hard
tasks at the same human_minutes, the fitted curve is SHALLOWER than the
true per-task curve, because it's averaging over the task-difficulty
residual.  This means the non-hierarchical fit implicitly estimates
something between "typical" and "marginal" — not cleanly either one.

Alexander Barry's comment captures this: "models without explicit random
effects accounting should naturally learn marginal behavior into the
logistic curve shape."

Our hierarchical model, by separating mu_task from (α, β), now estimates
the TRUE per-task logistic curve.  But the horizon calculation in
`extract_agent_results` then computes the typical horizon from those
steeper curves — which may be HIGHER than the non-hierarchical estimate
despite being computed from a better model.  This is not a bug; it means
the non-hierarchical model was conflating two effects (difficulty
sensitivity and difficulty heterogeneity) that the hierarchical model
correctly separates.  But the comparison only makes sense if you report
both typical and marginal from the hierarchical model.


## Summary of relative strengths

**Our model wins on:**
- Human baseline treatment (latent vs. fixed covariate)
- Separate handling of expert estimates vs. baselines
- Family-level partial pooling on task difficulty
- Per-agent slope structure (richer agent parameterization)

**Moss's model wins on:**
- Per-task discrimination (2PL vs. our implicit 1PL)
- Time-trend modeling (enables forecasting and trajectory comparison)
- Explicit typical-vs-marginal distinction
- Priors chosen more pragmatically (no pretense of MaxEnt justification)

**Both models share:**
- Logistic link function
- Task difficulty as a function of human time
- NUTS sampling for inference
- The fundamental assumption that human completion time predicts AI success

**The biggest gap in our model**, illuminated by the comparison, is the
missing marginal-horizon calculation.  We already have the posterior
ingredients (sigma_family, mu_family samples) — the computation is a
straightforward integral over the predictive distribution of mu_task.
Without it, our 80% horizons suffer from the same ~10× overstatement that
Moss identifies in METR's published numbers.
