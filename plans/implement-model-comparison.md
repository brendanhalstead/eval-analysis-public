# Implementation Plan: Bayesian Model Comparison

## Goal

Implement the model comparison framework from `maxent-prior-derivations.md`:
fit competing models (M0–M8) to the same data, compute LOO-CV and/or
marginal likelihoods, and report Bayes factors.  Minimize code duplication
by factoring shared structure out of model-specific code.


## Design principles

1. **Adding or removing a model = two lines in one file.**  A model is a
   builder function + one registry entry, both in `bayesian_models.py`.
   The comparison runner, CLI, and DVC stage never hard-code model names.
   See "How to add a new model" below for the full recipe.

2. **Same data preparation for all models.**  `prepare_hierarchical_data()`
   in `wrangle/bayesian.py` already does this well.  Reuse it as-is.

3. **Model builders are composable.**  Each model is a function
   `(HierarchicalData) → pm.Model` that assembles PyMC components.
   Shared components (task hierarchy, observation models) are helper
   functions that multiple builders call.  But nothing forces a model
   to use the shared components — a structurally novel model can build
   its graph from scratch as long as it includes `agent_obs`.

4. **One new module for model definitions, one new module for comparison.**
   Don't bloat the existing files.

5. **The existing pipeline is untouched.**  The comparison is a new DVC stage
   that runs alongside (not instead of) the existing hierarchical stage.


## File plan

```
src/horizon/
  utils/
    bayesian_hierarchical.py        # UNCHANGED — M0 lives here
    bayesian_models.py              # NEW — model component library + M1–M8
    bayesian_comparison.py          # NEW — LOO-CV, comparison table
  wrangle/
    bayesian.py                     # MINOR CHANGE — add `compare` subcommand
```

### Why three files, not one?

- `bayesian_hierarchical.py` stays untouched because it's depended on by the
  existing DVC stage.  Breaking it risks the existing pipeline.
- `bayesian_models.py` contains the model component library and competitor
  definitions.  It imports from `bayesian_hierarchical.py` to reuse
  `HierarchicalData` and `HierarchicalPrior`.
- `bayesian_comparison.py` contains the comparison logic (fit N models,
  compute LOO, format results).  It's separate from model definitions
  because comparison logic doesn't change when you add a new model.


## Module 1: `utils/bayesian_models.py`

### Shared components (functions, not classes)

Each component is a function that takes a `pm.Model` context and adds
variables to it.  This is how PyMC models compose naturally — you build
up the model graph by calling functions inside `with model:`.

```python
def add_task_hierarchy(
    model: pm.Model,
    data: HierarchicalData,
) -> pt.TensorVariable:
    """Add global → family → task difficulty hierarchy.

    Returns mu_task (dims="task") for use by other components.
    Used by: M0, M1, M2, M3, M4, M5, M8.
    """
    with model:
        mu_global = pm.Normal("mu_global", mu=6, sigma=3)
        log_sigma_global = pm.Flat("log_sigma_global")
        sigma_global = pm.Deterministic("sigma_global", pt.exp(log_sigma_global))
        mu_family = pm.Normal("mu_family", mu=mu_global, sigma=sigma_global, dims="family")
        log_sigma_family = pm.Flat("log_sigma_family")
        sigma_family = pm.Deterministic("sigma_family", pt.exp(log_sigma_family))
        mu_task = pm.Normal("mu_task", mu=mu_family[data.task_to_family_idx],
                            sigma=sigma_family, dims="task")
    return mu_task


def add_flat_task_difficulty(
    model: pm.Model,
    data: HierarchicalData,
) -> pt.TensorVariable:
    """Add independent task difficulties (no hierarchy).

    Returns mu_task (dims="task").
    Used by: M6.
    """
    with model:
        mu_task = pm.Normal("mu_task", mu=6, sigma=3, dims="task")
    return mu_task


def add_human_obs(model: pm.Model, data: HierarchicalData,
                  mu_task: pt.TensorVariable) -> None:
    """Add human duration observation model.  Used by all models."""
    with model:
        log_sigma_human = pm.Flat("log_sigma_human")
        sigma_human = pm.Deterministic("sigma_human", pt.exp(log_sigma_human))
        if len(data.human_log2_duration) > 0:
            pm.Normal("human_obs", mu=mu_task[data.human_task_idx],
                       sigma=sigma_human, observed=data.human_log2_duration)


def add_estimate_obs(model: pm.Model, data: HierarchicalData,
                     mu_task: pt.TensorVariable,
                     bias: bool = False) -> None:
    """Add expert estimate model.  If bias=True, adds a δ parameter (M4)."""
    with model:
        log_sigma_estimate = pm.Flat("log_sigma_estimate")
        sigma_estimate = pm.Deterministic("sigma_estimate", pt.exp(log_sigma_estimate))
        mu = mu_task[data.estimate_task_idx]
        if bias:
            delta = pm.Flat("delta_estimate")
            mu = mu + delta
        if len(data.estimate_log2_minutes) > 0:
            pm.Normal("estimate_obs", mu=mu, sigma=sigma_estimate,
                       observed=data.estimate_log2_minutes)


def add_agent_slopes(model: pm.Model, data: HierarchicalData):
    """Add per-agent (α, β) with honest priors.  Used by M0–M6."""
    with model:
        alpha = pm.Normal("alpha", mu=0, sigma=5, dims="agent")
        beta = pm.Uniform("beta", lower=-50, upper=0, dims="agent")
    return alpha, beta
```

### Model builders

Each builder is a function `build_M{n}(data) → pm.Model` that composes
the shared components:

```python
def build_M0(data: HierarchicalData) -> pm.Model:
    """Hierarchical, additive residual (current model with honest priors)."""
    coords = {"family": data.family_ids, "task": data.task_ids, "agent": data.agent_names}
    with pm.Model(coords=coords) as model:
        mu_task = add_task_hierarchy(model, data)
        add_human_obs(model, data, mu_task)
        add_estimate_obs(model, data, mu_task)
        alpha, beta = add_agent_slopes(model, data)
        log_sigma_eps = pm.Flat("log_sigma_epsilon")
        sigma_eps = pm.Deterministic("sigma_epsilon", pt.exp(log_sigma_eps))
        epsilon = pm.Normal("epsilon", mu=0, sigma=sigma_eps, dims="task")
        eta = alpha[data.agent_agent_idx] + beta[data.agent_agent_idx] * mu_task[data.agent_task_idx] + epsilon[data.agent_task_idx]
        pm.Bernoulli("agent_obs", p=pm.math.sigmoid(eta), observed=data.agent_scores)
    return model


def build_M1(data: HierarchicalData) -> pm.Model:
    """Hierarchical, per-task discrimination (2PL)."""
    coords = {"family": data.family_ids, "task": data.task_ids, "agent": data.agent_names}
    with pm.Model(coords=coords) as model:
        mu_task = add_task_hierarchy(model, data)
        add_human_obs(model, data, mu_task)
        add_estimate_obs(model, data, mu_task)
        alpha, beta = add_agent_slopes(model, data)
        a = pm.Exponential("discrimination", lam=1, dims="task")
        eta = a[data.agent_task_idx] * (alpha[data.agent_agent_idx] + beta[data.agent_agent_idx] * mu_task[data.agent_task_idx])
        pm.Bernoulli("agent_obs", p=pm.math.sigmoid(eta), observed=data.agent_scores)
    return model


# M2: build_M2 — combines discrimination + residual
# M3: build_M3 — probit link (pm.math.erfc instead of sigmoid)
# M4: build_M4 — calls add_estimate_obs(bias=True)
# M5: build_M5 — splits sigma_human into HCAST/SWAA
# M6: build_M6 — calls add_flat_task_difficulty instead of hierarchy
# M8: build_M8 — scalar theta + discrimination, no alpha/beta
```

### Model registry

```python
MODEL_REGISTRY: dict[str, Callable[[HierarchicalData], pm.Model]] = {
    "M0_hierarchical_residual": build_M0,
    "M1_discrimination": build_M1,
    "M2_both": build_M2,
    "M3_probit": build_M3,
    "M4_estimate_bias": build_M4,
    "M5_per_source_sigma": build_M5,
    "M6_flat": build_M6,
    "M8_irt": build_M8,
}
```

This lets the comparison runner iterate over models by name without
hard-coding which models exist.

### How to add a new model

Adding a model requires exactly two changes in one file (`bayesian_models.py`):

1. **Write a builder function.**  It takes `HierarchicalData`, returns
   `pm.Model`.  Compose from the shared components above — or don't, if
   the model is structurally different.  The only contract is: the model
   must include a `pm.Bernoulli("agent_obs", ...)` observed variable (so
   LOO-CV has something to compare).

2. **Add one line to `MODEL_REGISTRY`.**

```python
# Example: adding a new M9 with per-family discrimination
def build_M9(data: HierarchicalData) -> pm.Model:
    """Per-family discrimination parameters."""
    coords = {"family": data.family_ids, "task": data.task_ids, "agent": data.agent_names}
    with pm.Model(coords=coords) as model:
        mu_task = add_task_hierarchy(model, data)
        add_human_obs(model, data, mu_task)
        add_estimate_obs(model, data, mu_task)
        alpha, beta = add_agent_slopes(model, data)
        a_family = pm.Exponential("discrimination_family", lam=1, dims="family")
        a = a_family[data.task_to_family_idx]
        eta = a[data.agent_task_idx] * (alpha[data.agent_agent_idx] + beta[data.agent_agent_idx] * mu_task[data.agent_task_idx])
        pm.Bernoulli("agent_obs", p=pm.math.sigmoid(eta), observed=data.agent_scores)
    return model

# Then add to registry:
MODEL_REGISTRY["M9_family_discrimination"] = build_M9
```

Nothing else changes — the comparison runner, CLI, and DVC stage all
discover models from the registry.  To *remove* a model, delete the
builder function and its registry entry.  To run a *subset*, pass
`--models M0_hierarchical_residual M6_flat` on the command line.


## Module 2: `utils/bayesian_comparison.py`

```python
def compare_models(
    data: HierarchicalData,
    model_names: list[str] | None = None,
    n_samples: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Fit multiple models and compare via LOO-CV.

    Parameters
    ----------
    data : prepared HierarchicalData (same for all models)
    model_names : which models to fit (default: all in registry)

    Returns
    -------
    DataFrame with columns:
        model, elpd_loo, se_loo, p_loo, n_params, n_warnings,
        delta_elpd (vs best), se_delta, weight (stacking or pseudo-BMA)
    """
```

Implementation:

1. For each model name, call `MODEL_REGISTRY[name](data)` to build it.
2. Sample with `pm.sample(...)`.
3. Compute `az.loo(idata)` (PSIS-LOO).
4. Collect results into a DataFrame.
5. Call `az.compare({"M0": idata_m0, "M1": idata_m1, ...})` for the
   comparison table with stacking weights.

```python
def compare_models(...) -> pd.DataFrame:
    import arviz as az

    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    idatas = {}
    for name in model_names:
        builder = MODEL_REGISTRY[name]
        model = builder(data)
        with model:
            idata = pm.sample(
                draws=n_samples, tune=n_tune, chains=n_chains,
                target_accept=target_accept, random_seed=random_seed,
                compute_convergence_checks=True,
            )
            # LOO needs log-likelihood; tell PyMC to compute it
            pm.compute_log_likelihood(idata)
        idatas[name] = idata

    comparison = az.compare(idatas, ic="loo", scale="log")
    return comparison
```

The `az.compare` output is already a well-formatted DataFrame with
exactly the columns we want (rank, elpd_loo, p_loo, d_loo, weight, se,
dse, warning, scale).

### Important: `compute_log_likelihood`

For PSIS-LOO to work, the InferenceData must contain pointwise
log-likelihood values.  PyMC can compute these automatically if the
observed variables are named consistently.  The `pm.compute_log_likelihood`
call (or `idata_kwargs={"log_likelihood": True}` in `pm.sample`) handles
this.  All our models use `pm.Bernoulli("agent_obs", ...)` for the agent
scores and `pm.Normal("human_obs", ...)` / `pm.Normal("estimate_obs", ...)`
for the observation models, so the pointwise log-likelihood decomposes
naturally.

**Which observations to LOO over?**  We want to compare how well models
predict *agent scores*, not human durations.  So the LOO should be
computed over the `agent_obs` variable only.  ArviZ's `az.loo` accepts
a `var_name` parameter for this:

```python
az.loo(idata, var_name="agent_obs")
```

This ensures we're comparing models on their ability to predict agent
success, holding the human observation model fixed.


## Module 3: changes to `wrangle/bayesian.py`

Add a third subcommand `compare` alongside `grid` and `hierarchical`:

```python
# In get_parser():
comp = sub.add_parser("compare", help="Compare competing model structures via LOO-CV")
comp.add_argument("--fig-name", type=str, required=True)
comp.add_argument("--runs-file", type=pathlib.Path, required=True)
comp.add_argument("--output-comparison-file", type=pathlib.Path, required=True)
comp.add_argument("--models", nargs="*", default=None,
                  help="Model names to compare (default: all)")
comp.add_argument("--n-samples", type=int, default=1000)
comp.add_argument("--n-tune", type=int, default=1000)
comp.add_argument("--n-chains", type=int, default=2)
comp.add_argument("--target-accept", type=float, default=0.9)
comp.add_argument("--random-seed", type=int, default=42)
comp.add_argument("-v", "--verbose", action="store_true")
```

The `main_compare` function:

```python
def main_compare(fig_name, runs_file, output_comparison_file, models, ...):
    # 1. Load data (same as hierarchical)
    # 2. prepare_hierarchical_data(runs, ...)
    # 3. compare_models(data, model_names=models, ...)
    # 4. Save comparison DataFrame to CSV
    # 5. Save summary to YAML (for DVC metrics)
```


## DVC stage

Add to `reports/time-horizon-1-1/dvc.yaml`:

```yaml
compare_bayesian_models:
  cmd: python -m horizon.wrangle.bayesian compare
    --fig-name headline
    --runs-file data/raw/runs.jsonl
    --output-comparison-file data/wrangled/model_comparison/headline.csv
    --output-summary-file metrics/model_comparison/headline.yaml
    --models M0_hierarchical_residual M1_discrimination M6_flat M8_irt
    --n-samples ${hierarchical_bayesian.n_samples}
    --n-tune ${hierarchical_bayesian.n_tune}
    --n-chains ${hierarchical_bayesian.n_chains}
    --target-accept ${hierarchical_bayesian.target_accept}
    --random-seed ${hierarchical_bayesian.random_seed}
    -v
  deps:
  - data/raw/runs.jsonl
  - ${code_dir}/horizon/wrangle/bayesian.py
  - ${code_dir}/horizon/utils/bayesian_models.py
  - ${code_dir}/horizon/utils/bayesian_comparison.py
  - ${code_dir}/horizon/utils/bayesian_hierarchical.py
  params:
  - hierarchical_bayesian
  - fig_params/figs.yaml:
    - figs.wrangle_logistic.headline
  outs:
  - data/wrangled/model_comparison/headline.csv
  metrics:
  - metrics/model_comparison/headline.yaml:
      cache: false
  desc: Compare competing Bayesian model structures via PSIS-LOO-CV.
```

This stage is independent of `wrangle_hierarchical_bayesian` — they
can run in parallel.  The comparison stage is expensive (~N× the cost
of a single model fit), so it's a separate stage you run deliberately,
not on every pipeline invocation.


## What NOT to build

- **Bridge sampling**: ArviZ's implementation is experimental and unreliable.
  Use LOO-CV only.  If the LOO comparison is ambiguous, report that —
  don't reach for a less-tested method.
- **M7 (feature regression)**: Requires defining task features, which is a
  data question, not a modeling question.  Defer until someone specifies
  what features to use.
- **Plotting of comparison results**: The `az.compare` DataFrame is
  self-explanatory.  A plot can be added later if anyone wants one.
- **Refactoring `bayesian_hierarchical.py`**: Leave it alone.  The new
  `bayesian_models.py` reimplements M0 with honest priors alongside the
  competitors.  The existing code continues to work for the existing
  pipeline.  If M0-with-honest-priors produces different results from
  the current M0-with-HalfNormal-priors, that's interesting and worth
  reporting, not hiding.


## Implementation order

1. **`utils/bayesian_models.py`** — shared components + model builders.
   Start with M0 and M6 (the simplest structural comparison: hierarchy
   vs. flat).  Add M1 and M8 next.  Leave M2/M3/M5 for later — they're
   less interesting.

2. **`utils/bayesian_comparison.py`** — the comparison runner.  Small
   module, mostly wrapping `az.compare`.

3. **`wrangle/bayesian.py`** — add the `compare` subcommand.

4. **DVC stage** — add to `dvc.yaml`.

5. **Smoke test** — run comparison on a subset of data (e.g., 3 agents,
   50 tasks) to verify the pipeline works end-to-end before committing
   to a full run.


## Estimated parameter counts (for runtime planning)

| Model | Parameters | Notes |
|-------|-----------|-------|
| M0 | ~250 | 170 μ_task + 170 ε + 33×2 α,β + ~6 hyper |
| M1 | ~250 | 170 μ_task + 170 a + 33×2 α,β + ~5 hyper |
| M6 | ~240 | 170 μ_task + 170 ε + 33×2 α,β + ~3 hyper (no family) |
| M8 | ~210 | 170 μ_task + 170 a + 33 θ + ~5 hyper |

All are within PyMC/NUTS capacity.  Expect ~5–15 minutes per model
with 1000 draws × 2 chains on a single CPU.  Running 4 models takes
~20–60 minutes total — not fast, but not prohibitive for a DVC stage
that runs once.
