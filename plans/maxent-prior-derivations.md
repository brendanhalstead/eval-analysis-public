# Max-Ent Prior Derivations for Competing Models

Following Jaynes (2003), Chapters 11–12.  The principle: the prior should
encode exactly the constraints you can honestly state, and nothing else.
The maximum entropy distribution subject to those constraints is the unique
prior that is maximally noncommittal about everything you did NOT state.

All models below predict the same observed data:

- Human run durations (log₂ space)
- Expert time estimates (log₂ space)
- Agent binary scores

so their marginal likelihoods are directly comparable.  The models need not
share the same internal structure — they need not all be hierarchical, or use
the same parameterization, or even have the same number of latent variables.
The only requirement is that each model assigns a probability to the same
set of observations.  A flat model with independent task difficulties, a
hierarchical model with family pooling, and a regression model that predicts
difficulty from task features are all valid competitors as long as they
produce P(human durations, estimates, agent scores | θ) for their respective
θ.


## Jaynes's toolkit (reference)

| Support | Known constraints | MaxEnt distribution |
|---------|------------------|-------------------|
| ℝ | E[X] = μ, Var[X] = σ² | Normal(μ, σ²) |
| [0, ∞) | E[X] = s | Exponential(1/s) |
| [0, ∞) | E[X], E[X²] | Gamma |
| [0, ∞) | E[log X], Var[log X] | LogNormal |
| (-∞, 0] | E[X] = μ < 0 | Reflected Exponential: p(β) = λe^{λβ}, λ = -1/μ |
| (-∞, 0] | E[X] = μ, Var[X] = v | Truncated Normal |
| {0, 1} | E[Y · η] | Logistic (Jaynes Ch. 11) |

The transformation-group argument for scale parameters: if you know *nothing*
about a scale σ > 0 (not even its expected value), the unique prior invariant
under rescaling σ → cσ is the Jeffreys prior p(σ) ∝ 1/σ.  This is improper
but usable when the data identify σ (which they do in all our models, given
hundreds of human runs).


## 1. Parameters and their MaxEnt priors

When a parameter (or a parameter with the same *meaning*) appears in
multiple models, it should get the same prior in all of them.  This doesn't
require identical parameterizations — a hierarchical model's σ_human and a
flat model's σ_human encode the same physical quantity (human timing noise)
and should get the same prior, even though the two models structure task
difficulty completely differently.

Parameters that exist in only one model (e.g., per-task discrimination a_i
in the 2PL, or family-level variance σ_global in the hierarchical model)
get their own MaxEnt prior, derived from whatever you know about that
parameter.  The marginal likelihood will automatically penalize the model
for spreading prior mass over a parameter space that the data don't reward.


Each constraint below is tagged with its provenance:

- **[structural]**: Follows from the parameter's definition (sign, support,
  symmetry).  Unchallengeable.
- **[order-of-magnitude]**: Rough physical reasoning that doesn't require
  seeing this dataset.  Defensible but imprecise.
- **[external]**: Claimed to come from outside this dataset.  Source cited;
  you can check it.
- **[this dataset]**: Computed from the data we're modeling.  Using it in
  the prior is double-counting.  Must be replaced with a weaker constraint
  or Jeffreys prior.


### σ_human (human observation noise)

**What we know**: Scale parameter, σ > 0 **[structural]**.

The empirical pooled SD is ~0.97 in log₂ space (from
`multiverse_boxplot.py`).  But that's computed from this dataset —
**[this dataset]**.  We cannot use it to set E[σ_human] without
double-counting.

Can we get an external constraint?  We'd need published data on human
timing variability for comparable cognitive tasks (multi-hour software
engineering challenges timed in a controlled setting).  We don't have a
citation for this.  If such a study exists, its SD estimate could be used.
Until then, we can't honestly claim to know E[σ_human].

**Honest prior**: Jeffreys p(σ) ∝ 1/σ **[structural]**.  Improper, but
with 539 human runs the posterior is well-identified.

**If you're uncomfortable with improper priors**: Use a very weakly
informative proper prior like Exponential(1/5) (E[σ] = 5, covering
everything from near-zero to absurdly high noise).  This puts negligible
prior mass where the posterior will concentrate (~1) but keeps the prior
proper.  Call this what it is: a computational convenience, not a
knowledge claim.

**Current**: HalfNormal(1.5).  The effective E[σ] ≈ 1.20 is suspiciously
close to the empirical 0.97.  It's not clear whether the original author
peeked or got lucky.


### σ_estimate (expert estimate noise)

**What we know**: Scale parameter, σ > 0 **[structural]**.  Expert estimates
are noisier than actual run times **[order-of-magnitude]** — people are
bad at estimating how long things take, and these are estimates for novel
tasks, not familiar routines.

Alexander Barry (METR statistician) comments on the LessWrong post that
"only 60% of estimates were within a factor of 3" of baseline times
**[external — source: LessWrong comment on Moss's post]**.  A factor of 3
in minutes is log₂(3) ≈ 1.58 in log₂ space.  For a Normal, 60% within
±1.58 SD means σ ≈ 1.58/0.84 ≈ 1.88.

**Caveat**: This is a single comment about this specific dataset's
estimates.  It's *about* this data, derived from comparing estimates to
baselines *in this dataset*.  Strictly, it's **[this dataset]** laundered
through external commentary.  If you use it, be upfront about that.

**Honest prior with no data-peeking**: We know σ > 0 and that estimates
are noisier than run times.  That's it.  Jeffreys p(σ) ∝ 1/σ, or a weak
Exponential(1/5) as a proper alternative.

**If you accept the Barry quote as external**: Exponential(1/2) with
E[σ] = 2.  This is the one-constraint MaxEnt.

**Current**: HalfNormal(2.5), E[σ] ≈ 2.0.


### α_a (agent intercept: log-odds at log₂(h) = 0, i.e., h = 1 minute)

**What we know**: Location parameter **[structural]**.  At a 1-minute task,
most agents should succeed, so α is probably positive
**[order-of-magnitude]**.  But centering at 0 (50/50 on a trivial task)
is agnostic and defensible **[structural — symmetry argument]**.  We're
very uncertain about the scale — log-odds could range from -15 to +15.

**Constraints**: E[α] = 0 **[structural/symmetry]**, Var[α] = 25
**[order-of-magnitude — "3σ covers log-odds ±15"]**.

**MaxEnt**: Normal(0, 25).  This IS the current prior.  ✓


### β_a (agent slope: change in log-odds per doubling of difficulty)

**What we know**: β ≤ 0 — harder tasks have lower success
**[structural — this is domain knowledge, not a modeling choice]**.

Rough magnitude: E[β] ≈ -0.5?  Where does this number come from?  If it
came from fitting the frequentist model to this data and eyeballing
slopes, it's **[this dataset]** and can't be used.  If it came from "each
doubling of difficulty should roughly halve the odds" — that's a specific
quantitative claim that we have no external basis for.

**Honest assessment**: We know the sign.  We don't have a defensible
external estimate of the magnitude.

**Case 1 — we know only the sign** (most honest):

Constraint: β ≤ 0 **[structural]**.

MaxEnt on (-∞, 0] with no moment constraint: the least informative proper
prior is an Exponential with very small rate (i.e., very large mean
magnitude), approaching the improper p(β) ∝ 1 on (-∞, 0].  In practice,
use a reflected Exponential with a weak rate, e.g., λ = 0.2 (E[β] = -5),
or Jeffreys-like flat on (-∞, 0] if the posterior is well-identified.

With 33 agents × 170 tasks, β is well-identified for each agent.  A
flat prior on (-∞, 0] is usable.

**Case 2 — we claim to know E[β] ≈ -0.5** (requires justification):

If you can defend E[β] = -0.5 from outside this dataset, then:

MaxEnt on (-∞, 0] with E[β] = -0.5: p(β) = 2·exp(2β), the reflected
Exponential with rate λ = 2.  Properties: E[β] = -0.5, SD[β] = 0.5,
mode = 0.

**Case 3 — we claim sign, mean, AND variance** (current prior):

The current TruncatedNormal(mu=-0.5, sigma=1.5, upper=0) encodes three
constraints.  But the mu and sigma are *pre-truncation* parameters.
After truncation the effective moments shift:

    E[β | β ≤ 0] ≈ -1.4   (not -0.5)
    SD[β | β ≤ 0] ≈ 0.9    (not 1.5)

So the current prior encodes E[β] ≈ -1.4 — steeper slopes than stated.
This systematically shortens time horizons (T ∝ 1/|β|).

**Recommendation**: Use flat on (-∞, 0] (Case 1) unless you have a
genuine external source for E[β].  The data are informative enough.


### Hierarchy-specific scale parameters

These only appear in models that use hierarchical pooling.

**σ_global** (between-family SD): Scale, σ > 0 **[structural]**.
Tasks range from ~1 min to ~10,000 min (log₂ ≈ 0 to 13), so the
between-family spread is at most ~13 log₂ units **[order-of-magnitude]**.
That gives us a rough upper bound, not a mean.

Honest prior: Jeffreys p(σ) ∝ 1/σ or weak Exponential(1/5).
Current HalfNormal(3) has E[σ] ≈ 2.39 — it's unclear where "3" comes from
beyond seeming reasonable after seeing the data.

**σ_family** (within-family SD): Same situation.  Jeffreys or weak
Exponential.  Current HalfNormal(2) has E[σ] ≈ 1.60.

**μ_global** (global mean difficulty): The tasks in this dataset are
software engineering challenges designed to take between a few minutes
and many hours **[order-of-magnitude — this is knowable from the task
design, not from the timing data]**.  The geometric mean of such a range
is roughly 2^6 = 64 minutes.  We're quite uncertain: SD = 3 in log₂ space
covers a 500× range at ±3σ.

MaxEnt: Normal(6, 9).  Current prior matches.  ✓ — but note that E[μ] = 6
is defensible as "rough midpoint of the design range," not from computing
the actual mean of this dataset's task durations.

These priors should be identical across all models that have them, but
models without hierarchy (e.g., M6 below) simply don't have these
parameters — they're not "shared" across models that structurally differ.


## 2. Competing models

The models differ in structure — hierarchy, link function, parameterization
of how task difficulty affects agent success.  They need not share the same
internal architecture.  They need only predict the same observations.

Every model that uses a given parameter gives it the same prior.  To avoid
repetition, here is the shared prior block that models inherit from:

```
# --- Shared honest priors (all models that use these parameters) ---

# Human observation noise
σ_human:     p(σ) ∝ 1/σ             [structural — Jeffreys]
             PyMC: log_σ_human = pm.Flat(); σ_human = pt.exp(log_σ_human)

# Expert estimate noise
σ_estimate:  p(σ) ∝ 1/σ             [structural — Jeffreys]
             PyMC: log_σ_est = pm.Flat(); σ_estimate = pt.exp(log_σ_est)

# Agent intercept
α_a:         Normal(0, 25)           [structural/symmetry + order-of-magnitude]
             PyMC: pm.Normal("alpha", mu=0, sigma=5, dims="agent")

# Agent slope
β_a:         flat on (-∞, 0]         [structural — sign only]
             PyMC: pm.Uniform("beta", lower=-50, upper=0, dims="agent")

# Human observation model (for tasks with baselines)
log₂(duration_ij) ~ Normal(μ_task_i, σ_human)

# Expert estimate model (for tasks without baselines)
log₂(estimate_i) ~ Normal(μ_task_i, σ_estimate)
```

Models below show ONLY what they add or change relative to this block.


### M0: Hierarchical, additive residual (current model)

```
# Task difficulty hierarchy
μ_global    ~ Normal(6, 9)           [order-of-magnitude — task design range]
σ_global    ~ p(σ) ∝ 1/σ            [structural — Jeffreys]
μ_family_f  ~ Normal(μ_global, σ_global)
σ_family    ~ p(σ) ∝ 1/σ            [structural — Jeffreys]
μ_task_i    ~ Normal(μ_family[f(i)], σ_family)

# Agent success model
σ_ε ~ p(σ) ∝ 1/σ                    [structural — Jeffreys]
ε_i ~ Normal(0, σ_ε)                # AI-specific task residual

η_ai = α_a + β_a · μ_task_i + ε_i
score_ai ~ Bernoulli(logistic(η_ai))
```

The additive ε_i captures "AI-specific difficulty" — a task that's harder
(or easier) for AI than its human time would predict.  The same ε_i
applies to ALL agents.

**Parameter count (model-specific)**: 1 per task (ε_i) + 1 (σ_ε) = n_tasks + 1.

**Structural assumption**: Tasks differ in AI difficulty only by a constant
additive shift in log-odds, the same for all agents.


### M_base: Bare hierarchical (no residual, no discrimination)

```
# Task difficulty hierarchy (same as M0)
μ_global    ~ Normal(6, 9)
σ_global    ~ p(σ) ∝ 1/σ
μ_family_f  ~ Normal(μ_global, σ_global)
σ_family    ~ p(σ) ∝ 1/σ
μ_task_i    ~ Normal(μ_family[f(i)], σ_family)

# Agent success model — bare linear predictor, no extras
η_ai = α_a + β_a · μ_task_i
score_ai ~ Bernoulli(logistic(η_ai))
```

No model-specific parameters.  The simplest hierarchical model: task
difficulty drives agent success through a linear predictor, nothing else.

**Purpose**: Baseline for Bayes factor comparisons.  Since M_base adds
no model-specific parameters beyond the shared block, BF(M_base, Mx) is
well-defined whenever Mx's unique parameters all have proper priors.


### M1: Per-task discrimination (2PL)

```
η_ai = a_i · (α_a + β_a · μ_task_i)
P(success) = logistic(η_ai)
score_ai ~ Bernoulli(P)

a_i ~ Exponential(1)         # MaxEnt given E[a] = 1 (see derivation below)
```

No additive ε — instead, each task has a multiplicative discrimination
parameter a_i that scales how sharply the task separates strong from weak
agents.

**Derivation of prior for a_i**:

What we know: a_i > 0 (discrimination is positive by convention /
identification).  We adopt the convention E[a] = 1 (average discrimination
= 1, which sets the scale).

MaxEnt on (0, ∞) with E[a] = 1: **Exponential(rate = 1)**.

Properties: mode = 0, mean = 1, SD = 1.  This allows some tasks to have
very low discrimination (noisy indicators of ability) and some to have high
discrimination (clean separators).  The Exponential's mode at 0 might seem
odd — it says "most tasks are weak discriminators" a priori.  If that's
uncomfortable, you have two options:

*Option A*: Also claim E[log a] = 0 (geometric mean = 1) and Var[log a] = σ².
MaxEnt → **LogNormal(0, σ²)**.  But then you must choose σ², which requires
additional knowledge.  LogNormal(0, 1) has mean = √e ≈ 1.65, so you'd need
to adjust μ_log to get E[a] = 1: μ_log = -σ²/2.  With σ = 1:
LogNormal(-0.5, 1).  This is what Moss effectively uses.

*Option B*: Hierarchical — a_i ~ LogNormal(μ_a, σ_a) with hyperpriors on
μ_a and σ_a, letting the data determine the discrimination distribution.
Most principled but adds parameters.

**For model comparison vs. M0, use Exponential(1)** — it's the minimal-
information choice.

**Parameter count**: 1 per task (a_i), no σ_ε.  Same total as M0.

**Structural assumption**: Tasks differ in how *informative* they are about
agent ability, not just in how they shift the log-odds.


### M2: 2PL + additive residual (nesting M0 and M1)

```
η_ai = a_i · (α_a + β_a · μ_task_i) + ε_i
P(success) = logistic(η_ai)
score_ai ~ Bernoulli(P)

a_i ~ Exponential(1)
ε_i ~ Normal(0, σ_ε)
σ_ε ~ p(σ) ∝ 1/σ
```

This nests both M0 (when a_i = 1 for all tasks) and M1 (when ε_i = 0 for
all tasks).  If the marginal likelihood of M2 is no better than M0 or M1,
Occam's razor (via the automatic complexity penalty in the marginal
likelihood) will favor the simpler model.

**Parameter count**: 2 per task + 1 hyperparameter.  This is expensive —
identification may be poor with the data available.


### M3: Probit link

```
η_ai = α_a + β_a · μ_task_i + ε_i
P(success) = Φ(η_ai)         # standard normal CDF instead of logistic
score_ai ~ Bernoulli(P)
```

The parameter names and prior distributions are the same as M0.  But the
parameters have **different meaning** because the link functions have
different scales: logistic(η) ≈ Φ(η · √3/π) ≈ Φ(0.55·η).  A slope
β = -1 in the logistic model corresponds to roughly β ≈ -0.55 in the
probit model in terms of effect on probabilities.  So "same prior on β"
is not "same prior belief about how difficulty affects success."

This means the priors do NOT cancel in a Bayes factor — even though
both models use β ~ Uniform(-50, 0), this encodes different prior
beliefs about the probability-scale relationship.  **BF(M0, M3) is
not well-defined** without rescaling the probit priors.

**Jaynesian argument for logistic over probit**: The logistic function is
the maximum entropy distribution on {0, 1} given the linear predictor η
(Jaynes Ch. 11).  The probit assumes a latent Gaussian threshold model
(Y = 1 iff Z > 0, Z ~ N(η, 1)), which is an additional structural
assumption beyond what the data require.  On Jaynesian grounds, the
logistic is the default; probit requires justification.

**No new parameters, but priors are not equivalent.**  This is not a
pure structural comparison — the link function changes what the
parameters mean.


### M4: Estimate bias

Replaces the unbiased estimate model with:

```
log₂(estimate_i) ~ Normal(μ_task_i + δ, σ_estimate)
```

where δ captures systematic bias in expert time estimates.

**Derivation of prior for δ**:

What we know: δ is a location parameter (bias in log₂ space)
**[structural]**.  A positive δ means experts overestimate difficulty
(pessimistic); negative means they underestimate (optimistic / planning
fallacy).  We don't know the sign **[structural — symmetry]**.

We don't have an external estimate of the magnitude of expert bias for
this type of task.  The planning fallacy literature exists but covers
different settings (personal time estimates, not expert estimates of
novel software tasks).  So: **no defensible moment constraint**.

**Honest prior**: Flat (improper) on ℝ.  Since δ is identifiable from
tasks that have both baselines and estimates, the posterior will be proper.

**If you want a proper prior**: Normal(0, σ²) with σ large enough to be
negligible (e.g., σ = 5 covers biases up to a factor of 32× in either
direction).  Call this computational convenience, not a knowledge claim.

**Parameter count**: 1 (δ is shared across all estimate-only tasks).

**Structural assumption**: Expert estimates may be systematically biased in
one direction, but the bias is the same for all tasks.


### M5: Per-source σ_human

Replaces the single σ_human with source-specific noise:

```
σ_human_HCAST ~ p(σ) ∝ 1/σ    # Jeffreys
σ_human_SWAA  ~ p(σ) ∝ 1/σ    # Jeffreys

# HCAST runs:
log₂(duration_ij) ~ Normal(μ_task_i, σ_human_HCAST)

# SWAA runs:
log₂(duration_ij) ~ Normal(μ_task_i, σ_human_SWAA)
```

**Prior**: Both σ's get identical Jeffreys priors — same prior, different
data subsets.  No external knowledge distinguishes HCAST noise from SWAA
noise a priori.

With 467 HCAST and 235 SWAA runs, both σ's are well-identified.

**Parameter count**: 1 additional (2 σ's instead of 1).

**Structural assumption**: HCAST and SWAA timing methodology may have
different measurement precision.


### M6: Flat (non-hierarchical) joint model

No family hierarchy.  Each task gets an independent difficulty:

```
# Task difficulty — independent, no pooling
μ_task_i ~ Normal(6, 9)        # MaxEnt given E = 6, Var = 9

# Human observation model (same as hierarchical)
σ_human ~ p(σ) ∝ 1/σ          # Jeffreys
log₂(duration_ij) ~ Normal(μ_task_i, σ_human)

# Expert estimates (same)
σ_estimate ~ p(σ) ∝ 1/σ       # Jeffreys
log₂(estimate_i) ~ Normal(μ_task_i, σ_estimate)

# Agent success (same as M0)
α_a ~ Normal(0, 25)
β_a ~ flat on (-∞, 0]         # Jeffreys-like, sign constraint only
ε_i ~ Normal(0, σ_ε)
σ_ε ~ p(σ) ∝ 1/σ

η_ai = α_a + β_a · μ_task_i + ε_i
score_ai ~ Bernoulli(logistic(η_ai))
```

**What this tests**: Is family-level pooling actually helpful, or does it
just add parameters?  In the flat model, each task's difficulty is informed
only by its own human runs (and possibly its estimate).  Tasks with many
runs will be well-estimated; tasks with one run will have wide posteriors.
The hierarchy helps tasks with few runs by borrowing from family-mates —
but if families aren't meaningful groupings, or if most tasks have enough
runs on their own, the hierarchy isn't doing real work and the Bayes factor
will penalize it for the extra parameters (σ_global, σ_family, μ_family).

**Prior for μ_task_i**: Normal(6, 9) — the same prior that the hierarchical
model's μ_global has.  This is the honest "I know roughly where task
difficulties live but nothing about this specific task" prior.  The data
(human runs) will dominate for well-baselined tasks.


### M7: Feature regression on task difficulty

Instead of hierarchy OR independence, model task difficulty as a function
of observable features:

```
# Task difficulty as regression
μ_task_i = γ_0 + γ_source · x_source_i + γ_family · x_family_i + ν_i
ν_i ~ Normal(0, σ_ν)

# Priors
γ_0 ~ Normal(6, 9)            [order-of-magnitude — task design range, same as μ_global]
γ_source ~ Normal(0, σ_γ²)    [structural/symmetry — E = 0, no directional belief]
γ_family ~ Normal(0, σ_γ²)    [structural/symmetry — E = 0, no directional belief]
σ_γ ~ p(σ) ∝ 1/σ             [structural — Jeffreys; let data determine coeff scale]
σ_ν ~ p(σ) ∝ 1/σ             [structural — Jeffreys]
```

where x_source and x_family are encoded task features (one-hot or similar).
This tests whether the hierarchy's family structure is better captured by
explicit features than by random effects.

The regression coefficients γ get Normal(0, σ_γ²) with a shared
hyperprior on σ_γ.  E[γ] = 0 is defensible by symmetry
**[structural]** — we have no reason to expect any source or family to
be systematically harder before seeing data.  The scale σ_γ gets a
Jeffreys prior rather than a fixed value, because we have no external
basis for how large the effects are.


### M8: IRT-style scalar ability (Moss's approach, joint version)

Replace the per-agent (α, β) with a single scalar ability θ_a, and put
per-task discrimination on the task side.  This model does NOT inherit
the shared α_a / β_a priors — it replaces them entirely.

```
# Task difficulty hierarchy (same as M0, or flat as in M6)
μ_task_i ~ [hierarchical or flat — pick one and hold fixed for comparison]

# Agent ability — single scalar
θ_a ~ Normal(0, σ_θ)
σ_θ ~ p(σ) ∝ 1/σ            [structural — Jeffreys; let data set the ability scale]

# Per-task discrimination
a_i ~ Exponential(1)          [structural — E[a] = 1 is a scale convention]

# Observation models for human durations and estimates: same shared block

η_ai = a_i · (θ_a − μ_task_i)
score_ai ~ Bernoulli(logistic(η_ai))
```

This is a fundamentally different parameterization of agents.  Instead of
each agent having its own intercept and slope (2 parameters), each agent
has one scalar ability θ.  The tradeoff: the model is more parsimonious
per agent (1 parameter vs. 2) but can't express "good at easy tasks, bad
at hard ones" vs. "mediocre everywhere" — all agents with the same θ
behave identically.

**Prior for θ_a**: Normal(0, σ_θ) with σ_θ ~ Jeffreys.  E[θ] = 0 is a
location convention **[structural]** — the absolute scale is absorbed by
μ_task.  We have no external knowledge of how spread out agent abilities
are, so σ_θ gets Jeffreys rather than a fixed value.

**Prior for a_i**: Exponential(1).  E[a] = 1 is a scale convention
**[structural]** — it fixes the units of the θ–μ_task difference.
This is not data-derived; any positive mean would work, but 1 is the
natural choice.

**This model cannot be compared by sharing α/β priors with M0**, because
the parameters are structurally different.  The comparison is purely through
the marginal likelihood — how well each model predicts the data, weighted by
how much prior mass each wastes on regions of parameter space the data
don't support.


## 3. Which Bayes factors are well-defined?

A Bayes factor BF(Mi, Mj) is well-defined when every parameter unique
to one model has a proper prior.  Shared parameters with identical
(even improper) priors cancel in the ratio.  Model-specific parameters
with improper priors make the Bayes factor undefined — not "hard to
compute," literally not a number.

### Model-specific parameters and their priors

| Model | Parameters unique to this model | Prior | Proper? |
|-------|-------------------------------|-------|---------|
| M_base | (none beyond shared block) | — | n/a |
| M0 | σ_ε, ε_i | Jeffreys on σ_ε | **No** |
| M1 | a_i (discrimination) | Exponential(1) | **Yes** — E[a]=1 is scale identification |
| M2 | a_i + σ_ε + ε_i | Exp(1) + Jeffreys | **No** (σ_ε) |
| M3 | (same names as M0, but link rescales meaning) | Jeffreys on σ_ε + prior mismatch | **No** |
| M4 | δ (estimate bias) | Flat on ℝ | **No** |
| M5 | σ_HCAST, σ_SWAA (replace σ_human) | Jeffreys × 2 | **No** |
| M6 | (lacks σ_global, σ_family from hierarchical models) | — | see below |
| M7 | σ_γ, σ_ν + γ coefficients | Jeffreys × 2 | **No** |
| M8 | σ_θ, θ_a, a_i (replace α, β) | Jeffreys on σ_θ | **No** |

"Unique to this model" means relative to the shared block.  When
comparing two specific models, parameters shared between them cancel
regardless of properness.

### Well-defined Bayes factors

**BF(M_base, M1)** — does per-task discrimination help?
M_base has no unique parameters.  M1 adds a_i ~ Exponential(1) (proper).
All shared parameters (hierarchy, observation models, α, β) cancel.
Defined. ✓

This is the only well-defined Bayes factor with our honest priors.

### Undefined Bayes factors (and why)

**M0 vs. M1** (residual vs. discrimination): σ_ε in M0 is Jeffreys
(improper), unique to M0.  We have no honest constraint on E[σ_ε]
that would make it proper.  Undefined.

**M0 vs. M3** (logistic vs. probit): Although the parameter names are
identical, the link function changes what the parameters mean.
logistic(η) ≈ Φ(0.55·η), so the "same prior" on β encodes different
beliefs about probability-scale effects.  The priors don't actually
cancel.  Undefined without explicit prior rescaling.

**M0 vs. M6** (hierarchy vs. flat): σ_global and σ_family appear only
in M0 (and other hierarchical models), both Jeffreys.  Undefined.

**Anything vs. M8** (IRT scalar ability): σ_θ is Jeffreys.  We can
identify the scale via E[a]=1 OR σ_θ=1, but not both (one degree of
freedom).  Either way one improper prior remains.  Undefined.

**M0 vs. M4** (unbiased vs. biased estimates): δ is flat (improper).
No honest constraint on expert estimation bias magnitude.  Undefined.

**M0 vs. M5** (shared vs. per-source σ_human): M5 replaces one
Jeffreys parameter with two.  The priors don't cancel.  Undefined.

### What this means

The structurally interesting comparisons — hierarchy vs. flat,
residual vs. discrimination, slopes vs. scalar ability, logistic vs.
probit — are all formally undecidable with our honest priors.  This is
not a failure of the method.  It reflects a genuine epistemic fact: we
don't have enough prior information about model-specific parameters to
say how much Occam penalty each model deserves.

The one comparison that IS defined tests: does adding per-task
discrimination to a bare hierarchical model improve the marginal
likelihood?

For the undefined comparisons, we can still:
- Fit all models and inspect posteriors.
- Do posterior predictive checks (does the model reproduce observed
  patterns?).
- Report the posteriors honestly and let readers judge qualitatively.

But we should not substitute a different metric (LOO-CV, WAIC, etc.)
and pretend it answers the same question as Bayes factors.  It doesn't.
LOO-CV measures posterior predictive accuracy — a frequentist concept
that doesn't depend on priors.  Bayes factors measure which model the
data support given the priors.  They answer different questions.


## 4. Computing Bayes factors

### Why not LOO-CV?

LOO-CV (PSIS-LOO via ArviZ) estimates posterior predictive accuracy.
It's well-defined even with improper priors, which makes it tempting as
a substitute for Bayes factors.  But it answers a different question:
"which model predicts held-out data best?" rather than "which model does
the data support?"  The Bayesian question is the latter, and it requires
marginal likelihoods, which require proper priors on model-specific
parameters.  Substituting LOO-CV would dodge the prior question rather
than confronting it.

### How to compute Bayes factors for the eligible pairs

For models where all model-specific parameters have proper priors (see
§3), the Bayes factor BF(Mi, Mj) = p(data | Mi) / p(data | Mj).

Individual marginal likelihoods p(data | M) are not available in closed
form for our PyMC models.  Options for estimating them:

1. **Bridge sampling** (Gronau et al., 2017): Estimates log p(data | M)
   from posterior samples.  Good accuracy when the proposal distribution
   is well-chosen.  ArviZ has experimental support.

2. **Stepping-stone sampling** (Xie et al., 2011): More robust than
   bridge sampling, but expensive (requires sampling from a sequence of
   power posteriors).

3. **Savage-Dickey density ratio** (for nested models): When Mi is
   nested within Mj by restricting θ_extra = θ_0, the Bayes factor is
   BF = p(θ_extra = θ_0 | data, Mj) / p(θ_extra = θ_0 | Mj).  This
   only requires samples from the more complex model.  **Caveat**: this
   requires estimating the joint posterior density at a single point.
   For scalar or low-dimensional restrictions this works well.  For
   high-dimensional restrictions (e.g., 170 discrimination parameters
   all equal to 1) it's impractical — you can't estimate a
   170-dimensional density from MCMC samples.

For **BF(M_base, M1)** (our only eligible pair): M_base is nested in M1
by setting a_i = 1 for all 170 tasks.  In principle Savage-Dickey applies,
but estimating the joint posterior density at the 170-dimensional point
(1, 1, ..., 1) is not feasible.  Use bridge sampling instead (fit both
models, estimate each marginal likelihood separately, take the ratio).

### Interpretation

Report log₁₀ BF with the Kass & Raftery (1995) scale:

| |log₁₀ BF| | Evidence |
|------------|----------|
| 0 – 0.5 | Not worth more than a bare mention |
| 0.5 – 1 | Substantial |
| 1 – 2 | Strong |
| > 2 | Decisive |

If the Bayes factor is close to 1 (|log₁₀ BF| < 0.5), report that
the data don't distinguish the models.  This is a valid finding.


## 5. Where the current priors diverge from honest MaxEnt

| Parameter | Current | Honest MaxEnt | Issue |
|-----------|---------|---------------|-------|
| σ_human | HalfNormal(1.5) | Jeffreys 1/σ | E ≈ 1.20 is suspiciously close to empirical 0.97 |
| σ_estimate | HalfNormal(2.5) | Jeffreys 1/σ (or Exp(1/2) if Barry quote accepted) | Barry quote is about this dataset |
| σ_global | HalfNormal(3) | Jeffreys 1/σ | No external basis for E[σ] ≈ 3 |
| σ_family | HalfNormal(2) | Jeffreys 1/σ | No external basis for E[σ] ≈ 1.5 |
| σ_epsilon | HalfNormal(1.5) | Jeffreys 1/σ | No external basis for E[σ] ≈ 1.5 |
| β_a | TruncNorm(-0.5, 1.5, ≤0) | Flat on (-∞, 0] | **Effective E[β] = -1.4, no external source for any moment** |
| μ_global | Normal(6, 9) | Normal(6, 9) | ✓ E = 6 defensible from task design range |
| α_a | Normal(0, 25) | Normal(0, 25) | ✓ Symmetry + order-of-magnitude scale |

The pattern: every scale parameter prior and the slope prior encode
quantitative claims (specific means or variances) that trace back to
either this dataset or to nothing at all.  The location parameters
(μ_global, α_a) are defensible from structural arguments.

The most consequential divergence is **β_a**.  The TruncatedNormal has
effective E[β] ≈ -1.4 — but worse than the wrong number is the absence
of any external justification for picking a number at all.  The honest
prior is flat on (-∞, 0].

For scale parameters, the practical impact is small (data dominate with
hundreds of observations), but the intellectual dishonesty accumulates:
five parameters whose priors were "tuned to seem reasonable" is five
opportunities for the prior to push the posterior in a direction that
looks good but isn't warranted.  Jeffreys priors for all five scales
would be cleaner.


## 6. Summary: the honest Jaynesian recipe

1. **For each constraint, write down where you learned it.**  If the source
   is "I computed it from this dataset" or "it seemed reasonable" or "an LLM
   told me it was in the literature," drop it.

2. **Classify what remains.**  You'll mostly have: support constraints
   (σ > 0, β ≤ 0), symmetry arguments (E[α] = 0), and order-of-magnitude
   reasoning about the design of the experiment (tasks were designed to take
   minutes to hours → μ_global ≈ 6).

3. **Derive the MaxEnt distribution from those constraints.**  For most
   scale parameters in this problem, you'll end up with Jeffreys (1/σ),
   because the only honest constraint is σ > 0.  For location parameters
   with a defensible center and scale, you'll get Normals.

4. **Shared parameters get identical priors across models.**  Parameters
   with the same physical meaning get the same prior, regardless of what
   model they appear in.

5. **Model-specific parameters get MaxEnt priors from their own constraints.**
   For discrimination a_i: Exponential(1) (E[a] = 1 is a scale convention,
   not data-derived).  For bias δ: flat (we don't know the sign or magnitude).

6. **Compute Bayes factors where defined.**  Only model pairs where all
   model-specific parameters have proper MaxEnt priors yield well-defined
   Bayes factors.  For other pairs, report posterior fits and predictive
   checks, but do not substitute LOO-CV or other metrics that dodge the
   prior question.  If |log₁₀ BF| < 0.5, report that the data don't
   distinguish the models.

7. **Do a prior sensitivity check.**  For any constraint you weren't fully
   sure about, vary it and check that the Bayes factors are stable.  If
   they're not, the comparison is prior-dependent and you should say so
   rather than picking the prior that gives the answer you wanted.
