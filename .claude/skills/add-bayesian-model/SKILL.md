---
name: add-bayesian-model
description: Add a new competing Bayesian model to the model comparison framework. Use when adding a new model to compare against existing models (M_base, M0, M1, etc.) in the time-horizon analysis. Guides you through writing the builder function, setting honest MaxEnt priors, registering in MODEL_REGISTRY, and determining Bayes factor eligibility.
---

# Add a Bayesian Model to the Comparison Framework

Guide for adding a new competing model to the Bayesian model comparison
framework in `src/horizon/utils/bayesian_models.py`.

## Before you start

Read these files to understand the existing framework:

- `plans/maxent-prior-derivations.md` — Prior derivations, model definitions,
  and which Bayes factors are well-defined
- `plans/implement-model-comparison.md` — Implementation plan, ModelSpec
  pattern, shared components
- `src/horizon/utils/bayesian_models.py` — Model registry and builders
- `src/horizon/utils/bayesian_hierarchical.py` — HierarchicalData definition

## Step 1: Define the model mathematically

Before writing code, write out:

1. The full generative model (what does it assume about the data?)
2. Every parameter, its support, and its prior
3. What structural question does this model answer?

Add the model definition to `plans/maxent-prior-derivations.md` in the
"Competing models" section, following the format of existing entries.

## Step 2: Derive honest priors for model-specific parameters

For each parameter unique to your model (not in the shared block):

1. What is the support? (structural constraint)
2. Do you have an honest external constraint on any moments?
3. Derive the MaxEnt prior from ONLY defensible constraints
4. Tag each constraint with provenance: [structural], [order-of-magnitude],
   [external], or [this dataset]

**Do not fabricate constraints to make priors proper.** If you have no
honest moment constraint, use Jeffreys (improper). This means your model
won't be eligible for Bayes factors — that's fine. Report it honestly.

## Step 3: Write the builder function

The builder takes `HierarchicalData` and returns `pm.Model`. Use the
shared components where applicable:

```python
def build_M_new(data: HierarchicalData) -> pm.Model:
    """One-line description of what this model tests."""
    coords = {"family": data.family_ids, "task": data.task_ids, "agent": data.agent_names}
    with pm.Model(coords=coords) as model:
        # Task difficulty — use shared component or write your own
        mu_task = add_task_hierarchy(model, data)

        # Observation models — use shared components
        add_human_obs(model, data, mu_task)
        add_estimate_obs(model, data, mu_task)

        # Agent slopes — use shared component or replace entirely
        alpha, beta = add_agent_slopes(model, data)

        # --- Model-specific structure goes here ---
        # ...

        # REQUIRED: observed agent scores
        pm.Bernoulli("agent_obs", p=pm.math.sigmoid(eta), observed=data.agent_scores)
    return model
```

**Contract**: The model MUST include `pm.Bernoulli("agent_obs", ...)`
as the observed variable. Everything else is flexible.

**Shared components available** (call inside `with model:`):
- `add_task_hierarchy(model, data)` — hierarchical difficulty, returns mu_task
- `add_flat_task_difficulty(model, data)` — independent difficulty, returns mu_task
- `add_human_obs(model, data, mu_task)` — human duration observation model
- `add_estimate_obs(model, data, mu_task, bias=False)` — expert estimate model
- `add_agent_slopes(model, data)` — per-agent (alpha, beta) with honest priors

You don't have to use any of these. A structurally novel model can build
its graph from scratch.

### Implementing improper Jeffreys priors in PyMC

For scale parameters with Jeffreys prior p(sigma) ~ 1/sigma:

```python
log_sigma = pm.Flat("log_sigma_name")
sigma = pm.Deterministic("sigma_name", pt.exp(log_sigma))
```

For flat priors on unbounded parameters:

```python
param = pm.Flat("param_name")
```

For flat priors on half-lines (e.g., beta <= 0):

```python
beta = pm.Uniform("beta", lower=-50, upper=0, dims="agent")
```

## Step 4: Register in MODEL_REGISTRY

Add one entry to the registry dict:

```python
MODEL_REGISTRY["M_new_descriptive_name"] = ModelSpec(
    build_M_new,
    proper_model_priors=True,  # or False if any model-specific param is improper
)
```

**`proper_model_priors` rule**: Set to `True` ONLY if every parameter
unique to this model (not in the shared block) has a proper MaxEnt prior
derived from honest constraints. If ANY model-specific parameter has a
Jeffreys or flat (improper) prior, set to `False`.

## Step 5: Update documentation

In `plans/maxent-prior-derivations.md`:

1. Add model definition in Section 2
2. Add row to the model-specific parameters table in Section 3
3. If the model has `proper_model_priors=True`, note the new well-defined
   Bayes factor pair(s) in Section 3

## Step 6: Verify

1. The model builds without error: `spec.builder(data)` returns a pm.Model
2. The model samples: `pm.sample(draws=100, tune=100)` runs
3. `bf_eligible_pairs()` returns the expected pairs given your flag setting

## Checklist

- [ ] Model defined mathematically in maxent-prior-derivations.md
- [ ] Every prior has provenance tags
- [ ] No fabricated constraints to force properness
- [ ] Builder function written in bayesian_models.py
- [ ] Includes `pm.Bernoulli("agent_obs", ...)` observed variable
- [ ] Registry entry added with correct `proper_model_priors` flag
- [ ] Model-specific parameters table updated in Section 3
- [ ] Smoke test passes (build + short sample)
