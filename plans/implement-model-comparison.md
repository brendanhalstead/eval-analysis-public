# Implementation Plan: Bayesian Model Comparison

## Goal

Implement the model comparison framework from `maxent-prior-derivations.md`:
fit competing models to the same data, compute Bayes factors where
honest MaxEnt priors make them well-defined, and report posterior summaries
for all models.  Minimize code duplication by factoring shared structure
out of model-specific code.


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
    bayesian_comparison.py          # NEW — Bayes factors, posterior summaries
  wrangle/
    bayesian.py                     # MINOR CHANGE — add `compare` subcommand
```

### Why three files, not one?

- `bayesian_hierarchical.py` stays untouched because it's depended on by the
  existing DVC stage.  Breaking it risks the existing pipeline.
- `bayesian_models.py` contains the model component library and competitor
  definitions.  It imports from `bayesian_hierarchical.py` to reuse
  `HierarchicalData` and `HierarchicalPrior`.
- `bayesian_comparison.py` contains the comparison logic (fit models,
  compute Bayes factors where eligible, format results).  It's separate
  from model definitions because comparison logic doesn't change when
  you add a new model.


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
def build_M_base(data: HierarchicalData) -> pm.Model:
    """Bare hierarchical — no residual, no discrimination."""
    coords = {"family": data.family_ids, "task": data.task_ids, "agent": data.agent_names}
    with pm.Model(coords=coords) as model:
        mu_task = add_task_hierarchy(model, data)
        add_human_obs(model, data, mu_task)
        add_estimate_obs(model, data, mu_task)
        alpha, beta = add_agent_slopes(model, data)
        eta = alpha[data.agent_agent_idx] + beta[data.agent_agent_idx] * mu_task[data.agent_task_idx]
        pm.Bernoulli("agent_obs", p=pm.math.sigmoid(eta), observed=data.agent_scores)
    return model


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

Each entry records: the builder function, and whether all model-specific
parameters have proper MaxEnt priors (needed for Bayes factors).

```python
@dataclass
class ModelSpec:
    builder: Callable[[HierarchicalData], pm.Model]
    proper_model_priors: bool  # True → eligible for Bayes factor computation


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "M_base":                   ModelSpec(build_M_base, proper_model_priors=True),
    "M0_hierarchical_residual": ModelSpec(build_M0,     proper_model_priors=False),  # σ_ε: Jeffreys
    "M1_discrimination":        ModelSpec(build_M1,     proper_model_priors=True),   # a_i: Exp(1)
    "M2_both":                  ModelSpec(build_M2,     proper_model_priors=False),  # σ_ε: Jeffreys
    "M3_probit":                ModelSpec(build_M3,     proper_model_priors=False),  # σ_ε: Jeffreys + prior mismatch
    "M4_estimate_bias":         ModelSpec(build_M4,     proper_model_priors=False),  # δ: flat
    "M5_per_source_sigma":      ModelSpec(build_M5,     proper_model_priors=False),  # σ_HCAST, σ_SWAA: Jeffreys
    "M6_flat":                  ModelSpec(build_M6,     proper_model_priors=False),  # σ_ε: Jeffreys
    "M8_irt":                   ModelSpec(build_M8,     proper_model_priors=False),  # σ_θ: Jeffreys
}
```

**BF eligibility rule**: BF(Mi, Mj) is well-defined iff both have
`proper_model_priors=True`.  This is computed at runtime from the
registry — no hardcoded model pairs.

Note: an earlier draft had a `shared_params_group` flag for models
that differ only in link function (e.g., M0 logistic vs. M3 probit).
This was wrong — the link function changes the meaning of the linear
predictor parameters (logistic(η) ≈ Φ(0.55·η)), so "same prior on β"
encodes different beliefs about probability-scale effects.  The priors
don't actually cancel.  M3 is BF-ineligible for this reason too.

### How to add a new model

Adding a model requires exactly two changes in one file (`bayesian_models.py`):

1. **Write a builder function.**  It takes `HierarchicalData`, returns
   `pm.Model`.  Compose from the shared components above — or don't, if
   the model is structurally different.  The only contract is: the model
   must include a `pm.Bernoulli("agent_obs", ...)` observed variable.

2. **Add one entry to `MODEL_REGISTRY`**, setting
   `proper_model_priors=True` if every model-specific parameter has
   a proper MaxEnt prior derived from honest constraints.

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

# Exponential(1) is proper → eligible for BF
MODEL_REGISTRY["M9_family_discrimination"] = ModelSpec(build_M9, proper_model_priors=True)
```

Nothing else changes — the comparison runner, CLI, and DVC stage all
discover models from the registry.  To *remove* a model, delete the
builder function and its registry entry.  To run a *subset*, pass
`--models M0_hierarchical_residual M6_flat` on the command line.


## Module 2: `utils/bayesian_comparison.py`

### Core functions

```python
def bf_eligible_pairs(
    model_names: list[str],
) -> list[tuple[str, str]]:
    """Return all (Mi, Mj) pairs with well-defined Bayes factors.

    A pair is eligible iff both have proper_model_priors=True.
    """
    pairs = []
    for i, name_i in enumerate(model_names):
        spec_i = MODEL_REGISTRY[name_i]
        if not spec_i.proper_model_priors:
            continue
        for name_j in model_names[i + 1:]:
            spec_j = MODEL_REGISTRY[name_j]
            if spec_j.proper_model_priors:
                pairs.append((name_i, name_j))
    return pairs


def fit_models(
    data: HierarchicalData,
    model_names: list[str] | None = None,
    n_samples: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> dict[str, az.InferenceData]:
    """Fit multiple models and return their InferenceData objects."""
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    idatas = {}
    for name in model_names:
        spec = MODEL_REGISTRY[name]
        model = spec.builder(data)
        with model:
            idata = pm.sample(
                draws=n_samples, tune=n_tune, chains=n_chains,
                target_accept=target_accept, random_seed=random_seed,
                compute_convergence_checks=True,
            )
        idatas[name] = idata
    return idatas


def compute_bayes_factors(
    idatas: dict[str, az.InferenceData],
) -> pd.DataFrame:
    """Compute Bayes factors for all eligible model pairs.

    Returns a DataFrame with columns:
        model_i, model_j, log10_bf, method, eligible_reason
    """
    model_names = list(idatas.keys())
    eligible = bf_eligible_pairs(model_names)

    results = []
    for name_i, name_j in eligible:
        log10_bf = _bridge_sampling_bf(idatas[name_i], idatas[name_j])
        results.append({
            "model_i": name_i,
            "model_j": name_j,
            "log10_bf": log10_bf,
            "method": "bridge_sampling",
        })

    return pd.DataFrame(results)
```

### Bridge sampling

For BF(M_base, M1), we estimate each model's marginal likelihood
separately via bridge sampling (Gronau et al., 2017), then take the
ratio.  Bridge sampling estimates log p(data | M) from posterior samples
using an iterative scheme that constructs an optimal bridge distribution
between prior and posterior.  ArviZ has experimental support; if it's
not reliable enough, use the `bridgesampling` package or implement
stepping-stone sampling.

Note: M_base is technically nested in M1 (set a_i = 1 for all tasks),
so the Savage-Dickey density ratio could apply in principle.  But it
requires estimating the 170-dimensional joint posterior density at
(1, 1, ..., 1), which is infeasible from MCMC samples.  Bridge sampling
is the practical method.

### What the comparison runner does NOT do

- **No LOO-CV.**  LOO-CV answers "which model predicts best?" (a
  frequentist question about posterior predictive accuracy).  We want
  "which model does the data support?" (a Bayesian question about prior
  × likelihood).  Substituting one for the other would dodge the prior
  question rather than confronting it.

- **No Bayes factors for ineligible pairs.**  If a model has
  `proper_model_priors=False`, the BF is undefined.  The code reports
  this explicitly rather than computing a substitute.


## Module 3: changes to `wrangle/bayesian.py`

Add a third subcommand `compare` alongside `grid` and `hierarchical`:

```python
# In get_parser():
comp = sub.add_parser("compare", help="Fit competing models and compute Bayes factors")
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
    # 3. fit_models(data, model_names=models, ...)
    # 4. compute_bayes_factors(idatas) — only for eligible pairs
    # 5. Save BF results + posterior summaries to CSV
    # 6. Save summary to YAML (for DVC metrics)
    # 7. Log which pairs were eligible and which weren't
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
    --models M_base M0_hierarchical_residual M1_discrimination
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
  desc: >
    Fit competing Bayesian models and compute Bayes factors where
    honest MaxEnt priors make them well-defined.
```

With `M_base M0 M1`, the eligible BF pair is:
- M_base vs M1 (both `proper_model_priors=True`) — does discrimination help?

M0 is included for posterior comparison but has no eligible BF partner
(σ_ε is Jeffreys).

This stage is independent of `wrangle_hierarchical_bayesian` — they
can run in parallel.  The comparison stage is expensive (~N× the cost
of a single model fit), so it's a separate stage you run deliberately,
not on every pipeline invocation.


## What NOT to build

- **LOO-CV / WAIC**: These answer "which model predicts best?" — a
  frequentist predictive-accuracy question.  We want Bayes factors,
  which answer "which model does the data support?"  Including both
  would invite confusion about which to trust when they disagree.
- **Bayes factors for ineligible pairs**: Don't make priors proper just
  to get a BF.  If the honest MaxEnt prior is improper, the BF is
  undefined.  Report this, don't paper over it.
- **M7 (feature regression)**: Requires defining task features, which is a
  data question, not a modeling question.  Defer until someone specifies
  what features to use.
- **Plotting of comparison results**: The BF table is self-explanatory.
  A plot can be added later if anyone wants one.
- **Refactoring `bayesian_hierarchical.py`**: Leave it alone.  The new
  `bayesian_models.py` reimplements M0 with honest priors alongside the
  competitors.  The existing code continues to work for the existing
  pipeline.


## Implementation order

1. **`utils/bayesian_models.py`** — `ModelSpec` dataclass, shared
   components, model builders.  Start with M_base, M0, and M1
   (the set that yields the one defined BF comparison).  Add remaining
   models as stubs for posterior-only fitting.

2. **`utils/bayesian_comparison.py`** — `bf_eligible_pairs()`,
   `fit_models()`, `compute_bayes_factors()`.  Bridge sampling
   implementation (or wrapper around external library).

3. **`wrangle/bayesian.py`** — add the `compare` subcommand.

4. **DVC stage** — add to `dvc.yaml`.

5. **Smoke test** — run comparison on a subset of data (e.g., 3 agents,
   50 tasks) to verify the pipeline works end-to-end before committing
   to a full run.


## Estimated parameter counts (for runtime planning)

| Model | Parameters | BF-eligible? | Notes |
|-------|-----------|-------------|-------|
| M_base | ~80 | Yes (proper) | 170 μ_task + 33×2 α,β + ~5 hyper |
| M0 | ~250 | No (σ_ε: Jeffreys) | + 170 ε + 1 σ_ε |
| M1 | ~250 | Yes (proper) | + 170 a (Exp(1)) |

All are within PyMC/NUTS capacity.  Expect ~5–15 minutes per model
with 1000 draws × 2 chains on a single CPU.  Running 3 models takes
~15–45 minutes total.
