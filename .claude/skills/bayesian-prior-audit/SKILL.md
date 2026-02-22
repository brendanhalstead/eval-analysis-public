---
name: bayesian-prior-audit
description: Audit Bayesian model priors for intellectual honesty. Use when reviewing or building Bayesian models to check whether priors are honestly derived from defensible constraints, identify data-peeking, tag provenance of each constraint, and determine which model comparisons (Bayes factors) are well-defined given the priors. Applicable to any Bayesian modeling project, not just this codebase.
---

# Bayesian Prior Audit

Audit the priors of a Bayesian model (or set of competing models) for
intellectual honesty, following Jaynes (2003).

## When to use

- Reviewing an existing Bayesian model's prior choices
- Building a new model and choosing priors
- Comparing multiple models via Bayes factors
- Checking whether a published analysis's priors are defensible

## Audit procedure

Work through each parameter in the model(s). For each one, answer the
questions below and record your findings in a structured table.

### Step 1: Identify every prior

For each parameter in the model, record:

| Parameter | Support | Current prior | Constraints claimed |
|-----------|---------|---------------|---------------------|

### Step 2: Tag provenance of each constraint

Every constraint that informs a prior must have a provenance tag:

- **[structural]**: Follows from the parameter's definition — sign, support,
  symmetry. Unchallengeable. Examples: "sigma > 0", "probability in [0,1]",
  "discrimination is positive by convention."
- **[order-of-magnitude]**: Rough physical/domain reasoning that doesn't
  require seeing this dataset. Defensible but imprecise. Example: "tasks
  were designed to take minutes to hours, so mean difficulty ~ 6 in log2."
- **[external]**: Claimed to come from outside this dataset. Cite the source
  so it can be verified. Example: "IRT discrimination typically has mean ~1
  (de Ayala, 2009)."
- **[this dataset]**: Computed from the data being modeled. Using it in the
  prior is double-counting. Must be replaced with a weaker constraint or
  Jeffreys prior. Example: "the empirical SD is 0.97" computed from the
  same runs the model will fit.

**Red flags to check for:**
- Parameters whose priors are "suspiciously close" to empirical summaries
- External citations that are actually about this dataset laundered through
  commentary (e.g., a blog comment analyzing this data, cited as "external")
- Moment constraints (E[X] = c) with no stated source for c
- Priors described as "weakly informative" that actually encode specific
  quantitative claims

### Step 3: Derive the honest MaxEnt prior

For each parameter, given ONLY the honestly-tagged constraints:

| Support | Known constraints | MaxEnt distribution |
|---------|-------------------|---------------------|
| R | E[X] = mu, Var[X] = v | Normal(mu, v) |
| [0, inf) | nothing beyond support | Jeffreys: p(sigma) ~ 1/sigma |
| [0, inf) | E[X] = s | Exponential(1/s) |
| [0, inf) | E[X], E[X^2] | Gamma |
| (-inf, 0] | nothing beyond sign | Flat on (-inf, 0] |
| (-inf, 0] | E[X] = mu | Reflected Exponential |
| {0, 1} | E[Y * eta] | Logistic (Jaynes Ch. 11) |

If the only honest constraint is support (e.g., sigma > 0), the MaxEnt
prior is the Jeffreys/transformation-group prior. This is improper but
usable when the data identify the parameter.

**Do NOT fabricate constraints to make a prior proper.** If the honest
prior is improper, say so. An improper prior is more honest than a proper
prior encoding information you don't have.

### Step 4: Compare current vs. honest priors

Build a divergence table:

| Parameter | Current prior | Honest MaxEnt | Issue |
|-----------|---------------|---------------|-------|
| ... | HalfNormal(1.5) | Jeffreys 1/sigma | E ~ 1.2 suspiciously close to empirical 0.97 |

Flag every parameter where the current prior encodes quantitative claims
that trace back to either this dataset or to nothing at all.

### Step 5: Assess Bayes factor eligibility (if comparing models)

A Bayes factor BF(Mi, Mj) is well-defined when:
1. Every parameter **unique to one model** has a **proper** prior
2. Shared parameters with identical (even improper) priors cancel in the ratio

**Critical subtlety — link functions**: If two models use different link
functions (e.g., logistic vs. probit), the "same prior on beta" encodes
DIFFERENT beliefs about probability-scale effects because
logistic(eta) ~ Phi(0.55 * eta). The priors do NOT cancel. Such pairs
are BF-ineligible.

For each model pair, determine:
- Which parameters are shared (identical prior + identical meaning)?
- Which parameters are unique to one model?
- Do all unique parameters have proper priors?
- Result: Well-defined / Undefined (and why)

### Step 6: Write the report

Summarize findings:

1. **Parameters with defensible priors** (keep as-is)
2. **Parameters with data-peeked or unjustified priors** (replace with MaxEnt)
3. **Well-defined Bayes factors** (list eligible pairs)
4. **Undefined Bayes factors** (list ineligible pairs and why)
5. **Recommendations** for each problematic prior

## Key principles

- The prior should encode exactly the constraints you can honestly state,
  and nothing else (Jaynes, 2003, Ch. 11-12)
- If you can't set honest proper priors on model-specific parameters,
  you can't compute Bayes factors for those models. This is an epistemic
  fact, not a failure.
- Do not substitute LOO-CV or WAIC for Bayes factors. They answer different
  questions: "which model predicts best?" vs. "which model does the data
  support?" Substituting one for the other dodges the prior question.
- For scale parameters with no known moments, use Jeffreys p(sigma) ~ 1/sigma
- For undefined comparisons, fit all models, inspect posteriors, and do
  posterior predictive checks. Report honestly that formal comparison is
  not possible.
