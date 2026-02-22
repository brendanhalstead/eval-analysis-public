# Plan: Bayesian Time Horizon with Run-Level Human Baselines

## Motivation

The current pipeline collapses multiple human baseline runs into a single
`human_minutes` geometric mean per task, then treats that aggregate as the
"true" difficulty.  The multiverse analysis (`multiverse_boxplot.py`) then
adds noise *back in* to simulate uncertainty about that aggregate.  This is
throwing away information and then trying to reconstruct it.

We have **793 human runs** (567 successful) across **153 baselined tasks**,
with a median of 3 runs per task (range 1-57).  The Bayesian model should
treat these as first-class observations, jointly inferring each task's true
difficulty and the agent success curves.

## Data schema (what we have)

From `reports/time-horizon-1-0/data/raw/runs.jsonl` (41,629 rows):

**Human runs** (alias="human", 793 total):
- HCAST (467 runs, 108 tasks): `started_at=0`, `completed_at` = duration in ms
- SWAA (235 runs, ~40 tasks): `started_at` and `completed_at` are timestamps;
  duration = `completed_at - started_at`
- RE-Bench (91 runs, 7 tasks): `completed_at` = duration in ms, but
  `human_minutes` is a fixed estimate (~480 min), NOT derived from run times.
  RE-Bench uses continuous scores, not completion-time-based measurement.

**Agent runs** (~40,836 total, 33 agents across 170 tasks):
- `human_minutes`: pre-computed geometric mean of successful baselines (or estimate)
- `score_binarized`: 0/1 success
- `task_family`: grouping variable (e.g., "pico_ctf", "acdc_bug")
- `task_source`: HCAST / SWAA / RE-Bench

**17 tasks** have `human_source="estimate"` (no human baseline runs at all).

## Proposed model

### Latent variables

For each task *i*, a latent true difficulty `mu_task_i` in log2(minutes) space.

### Hierarchical structure

```
# Task family hierarchy
mu_family_f ~ Normal(mu_global, sigma_global)
mu_task_i   ~ Normal(mu_family[f(i)], sigma_family)

# Hyperpriors
mu_global    ~ Normal(5, 3)        # ~32 minutes, very weak
sigma_global ~ HalfNormal(3)
sigma_family ~ HalfNormal(2)
```

### Human observation model

Each successful human baseline run *j* on task *i* gives us a noisy
observation of the true difficulty:

```
# Human run durations (HCAST + SWAA)
sigma_human ~ HalfNormal(1.0)      # prior centered on empirical ~0.97 in log2
log2(duration_ij) ~ Normal(mu_task_i, sigma_human)

# Expert estimates (for the 17 tasks with no baselines)
sigma_estimate ~ HalfNormal(2.0)    # wider: estimates are noisier
log2(estimate_i) ~ Normal(mu_task_i, sigma_estimate)
```

`sigma_human` will be learned from the ~539 successful runs.  The empirical
pooled SD (computed in `multiverse_boxplot.py:find_baseline_time_se`) is
~0.97 in log2 space (factor of ~1.95x), so the prior `HalfNormal(1.0)` is
well-calibrated.

### Agent success model

For each agent *a* and task *i*:

```
alpha_a ~ Normal(0, 5)              # intercept (same prior as existing)
beta_a  ~ Normal(-0.5, 1.5), beta <= 0  # slope (truncated, same as existing)
P(success_aij) = logistic(alpha_a + beta_a * mu_task_i)
score_aij ~ Bernoulli(P(success_aij))
```

This uses `mu_task_i` (inferred latent difficulty) rather than the pre-computed
`human_minutes`, so uncertainty in task difficulty propagates into the agent
success predictions.

### Derived quantities

Time horizon T_a(q) = 2^{(logit(q) - alpha_a) / beta_a} — computed from the
posterior samples of (alpha_a, beta_a) for each agent.

## What this buys us

1. **No information loss**: 57 runs on a task gives much more certainty about
   `mu_task` than 1 run.  The current geomean flattens that distinction.

2. **Proper uncertainty propagation**: tasks with few baselines have wide
   posteriors on `mu_task`; that uncertainty flows through to agent success
   predictions automatically.

3. **sigma_human estimated, not assumed**: the model learns inter-human
   variability from the data rather than us plugging in 0.67.

4. **Estimate-only tasks get appropriate skepticism**: `sigma_estimate` is
   learned from the tasks that have BOTH baselines and estimates (we can add
   a calibration likelihood for those).

5. **Hierarchical family effects**: replaces the ad hoc 1/sqrt(n) weighting
   with a principled partial-pooling model.

6. **One model, one inference**: no multi-stage pipeline where each stage
   discards information the next stage needs.

## Complications to handle

### RE-Bench tasks
RE-Bench human runs measure something different — they have continuous scores,
and the run "durations" are fixed windows (2h or 8h), not completion times.
Options:
- **(a)** Treat `human_minutes` as an expert estimate for RE-Bench tasks
  (use `sigma_estimate`), ignore the individual run durations.
- **(b)** Model RE-Bench separately with its own observation model.

**Recommendation: (a)** — 7 tasks is too few to justify a separate submodel.

### Failed human runs
We have 226 failed human runs.  These are informative — they tell us the task
takes *at least* as long as the failed attempt duration (right-censored
observations).  Options:
- **(a)** Ignore failed runs (what the current pipeline does).
- **(b)** Model as right-censored: `log2(duration_failed) < mu_task_i + noise`.
- **(c)** Include as a lower bound: P(failure | duration < mu_task) is higher.

**Recommendation: (a) initially**, add (b) as a refinement if needed.  The
failed runs are informative but adding censoring to a PyMC model is
straightforward and can be done in a second pass.

### Computational cost
~170 latent `mu_task` variables + ~33 (alpha, beta) pairs + hyperparameters
= ~250 parameters.  This is well within PyMC/NumPyro capacity.  The exact
grid approach in `utils/bayesian.py` cannot scale to this — we need MCMC.

## Implementation plan

### Step 1: Data wrangling function
Create a function that extracts the run-level human data from `runs.jsonl`:
- Compute actual duration for each human run (handling HCAST/SWAA timestamp
  difference)
- Filter to successful runs with positive duration
- Tag RE-Bench tasks as estimate-only
- Return a clean DataFrame: `(task_id, task_family, duration_log2, source_type)`

**Location**: new function in `src/horizon/wrangle/bayesian.py`

### Step 2: PyMC model specification
Implement the hierarchical model described above.  Separate the model
specification from the inference so we can inspect the model graph before
running.

**Location**: new module `src/horizon/utils/bayesian_hierarchical.py`

### Step 3: Inference wrapper
Run MCMC (NUTS), extract posterior samples, compute derived quantities
(time horizons with credible intervals).  Output a DataFrame in the same
format as `wrangle/bayesian.py:run_bayesian_regressions()` so downstream
plotting code works unchanged.

**Location**: new function in `src/horizon/wrangle/bayesian.py`

### Step 4: Diagnostic checks
- Trace plots, R-hat, effective sample size
- Posterior predictive checks: can the model reproduce the observed human
  run-time distribution and agent success rates?
- Compare inferred `mu_task` to the existing `human_minutes` geomean —
  they should be similar for well-baselined tasks, wider for poorly-baselined
  ones.

### Step 5: Integration with existing pipeline
- Add a DVC stage for the hierarchical Bayesian fit
- Wire into the existing plotting code (same output format)
- Compare results to the grid-based Bayesian and frequentist pipelines

## Open questions

1. **Should we also model agent-level hierarchy?**  The 33 agents are not
   exchangeable (they span GPT-2 to GPT-5), but agents from the same family
   (e.g., Claude 3.5 Sonnet Old/New) might share partial pooling.  Probably
   not worth the complexity for a first pass.

2. **Continuous scores for RE-Bench**: the current pipeline binarizes them.
   A beta-likelihood or ordinal model could use the continuous scores directly.
   Defer to a later iteration.

3. **SWAA timing data**: The SWAA webapp enables "more accurate timing" per
   the METR paper.  Should SWAA runs get a different (smaller) `sigma_human`?
   Could add a `sigma_human_source` hierarchy but probably overkill.

4. **Prior on sigma_human**: `HalfNormal(1.0)` is centered on the empirical
   value.  An alternative is `HalfCauchy(1.0)` for heavier tails, or just
   let the data speak with a vaguer `HalfNormal(2.0)`.  The 539 successful
   runs should overwhelm any reasonable prior here.
