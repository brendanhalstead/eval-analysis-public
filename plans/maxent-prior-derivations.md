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


### σ_human (human observation noise)

**What we know**: Scale parameter, σ > 0.  The empirical pooled SD of
log₂(duration) across tasks is ~0.97 (computed in `multiverse_boxplot.py`).
So we have a fairly good point estimate.

**Constraint**: E[σ_human] ≈ 1.0.

**MaxEnt (one constraint)**: Exponential(1).
Mode = 0, mean = 1, heavy right tail.

**Current**: HalfNormal(1.5), with E[σ] ≈ 1.20.

With 539 successful human runs, the posterior on σ_human will be tightly
concentrated regardless of the prior.  But for intellectual honesty: we
know the mean of this scale, not its variance, so Exponential is the right
form.

**Alternative**: If you're comfortable claiming E[σ] ≈ 1 AND E[σ²] ≈ 1.5
(i.e., you also know the second moment), then Gamma(shape, rate) with
those moments: shape = E²/Var = 1/(1.5 - 1) = 2, rate = E/Var = 1/0.5 = 2
→ Gamma(2, 2).  This concentrates mass around 1 with less tail than
Exponential.  Only use if you can defend the second-moment claim.


### σ_estimate (expert estimate noise)

**What we know**: Scale parameter, σ > 0.  Expert estimates are noisier than
actual run times.  Alexander Barry (METR) reports "only 60% of estimates
were within a factor of 3" of baseline times.  A factor of 3 in minutes is
log₂(3) ≈ 1.58 in log₂ space.  For a Normal, 60% within 1.58 SDs means
σ ≈ 1.58/0.84 ≈ 1.88.  Call it E[σ_estimate] ≈ 2.

**MaxEnt**: Exponential(1/2).

**Current**: HalfNormal(2.5), with E[σ] ≈ 2.0.  The effective mean
coincidentally matches, but the shape is HalfNormal rather than Exponential.


### α_a (agent intercept: log-odds at log₂(h) = 0, i.e., h = 1 minute)

**What we know**: Location parameter.  At a 1-minute task, most agents should
succeed, so α is probably positive.  But how positive?  An agent with 99%
success at 1 min has α ≈ 4.6.  We're quite uncertain.

**Constraints**: E[α] ≈ 0 (agnostic center), Var[α] = 25 (very wide).

**MaxEnt**: Normal(0, 25).  This IS the current prior.  ✓

(One could argue E[α] should be positive — agents should succeed on trivial
tasks — but centering at 0 is the conservative "I don't know" choice, and
with 33 agents' worth of data, the posterior will be data-dominated.)


### β_a (agent slope: change in log-odds per doubling of difficulty)

**What we know**: β ≤ 0 (harder tasks have lower success — domain knowledge,
not a modeling choice).  Rough magnitude: E[β] ≈ -0.5 (each doubling of
task duration reduces log-odds by ~0.5).

**Case 1 — we know only the sign and mean**:

Constraint: β ≤ 0, E[β] = -0.5.

MaxEnt on (-∞, 0]: p(β) = λ exp(λβ) where λ = -1/E[β] = 2.

This is a **reflected Exponential**: mass concentrated near 0 (shallow
slopes), exponentially decaying toward very negative β.

Properties: E[β] = -0.5, SD[β] = 0.5, mode = 0.

**Case 2 — we know the sign, mean, AND variance**:

Constraints: β ≤ 0, E[β] = -0.5, Var[β] = 2.25 (SD = 1.5).

MaxEnt: Truncated Normal(-0.5, 1.5, upper=0).  This is the current prior.

**But there's a catch.**  The current code specifies
`TruncatedNormal(mu=-0.5, sigma=1.5, upper=0)`, where mu and sigma are
the *pre-truncation* parameters.  After truncating to β ≤ 0, the effective
moments shift:

    E[β | β ≤ 0] ≈ -1.4   (not -0.5)
    SD[β | β ≤ 0] ≈ 0.9    (not 1.5)

So the current prior actually encodes a belief that E[β] ≈ -1.4 — steeper
slopes than intended.  If you meant to encode E[β] = -0.5, the reflected
Exponential does it exactly.

**Recommendation for model comparison**: Use the reflected Exponential (Case 1).
It encodes the minimum information — sign constraint + rough mean — and is
the most conservative choice.  If the data are informative about β (they should
be, with 33 agents × 170 tasks), the posterior will concentrate regardless.


### α_a (agent intercept: log-odds at log₂(h) = 0, i.e., h = 1 minute)

**What we know**: Location parameter.  At a 1-minute task, most agents should
succeed, so α is probably positive.  But how positive?  An agent with 99%
success at 1 min has α ≈ 4.6.  We're quite uncertain.

**Constraints**: E[α] ≈ 0 (agnostic center), Var[α] = 25 (very wide).

**MaxEnt**: Normal(0, 25).  This IS the current prior.  ✓


### β_a (agent slope: change in log-odds per doubling of difficulty)

**What we know**: β ≤ 0 (harder tasks have lower success — domain knowledge,
not a modeling choice).  Rough magnitude: E[β] ≈ -0.5 (each doubling of
task duration reduces log-odds by ~0.5).

**Case 1 — we know only the sign and mean**:

Constraint: β ≤ 0, E[β] = -0.5.

MaxEnt on (-∞, 0]: p(β) = λ exp(λβ) where λ = -1/E[β] = 2.

This is a **reflected Exponential**: mass concentrated near 0 (shallow
slopes), exponentially decaying toward very negative β.

Properties: E[β] = -0.5, SD[β] = 0.5, mode = 0.

**Case 2 — we know the sign, mean, AND variance**:

Constraints: β ≤ 0, E[β] = -0.5, Var[β] = 2.25 (SD = 1.5).

MaxEnt: Truncated Normal(-0.5, 1.5, upper=0).  This is the current prior.

**But there's a catch.**  The current code specifies
`TruncatedNormal(mu=-0.5, sigma=1.5, upper=0)`, where mu and sigma are
the *pre-truncation* parameters.  After truncating to β ≤ 0, the effective
moments shift:

    E[β | β ≤ 0] ≈ -1.4   (not -0.5)
    SD[β | β ≤ 0] ≈ 0.9    (not 1.5)

So the current prior actually encodes a belief that E[β] ≈ -1.4 — steeper
slopes than intended.  If you meant to encode E[β] = -0.5, the reflected
Exponential does it exactly.

**Recommendation for model comparison**: Use the reflected Exponential (Case 1).
It encodes the minimum information — sign constraint + rough mean — and is
the most conservative choice.  If the data are informative about β (they should
be, with 33 agents × 170 tasks), the posterior will concentrate regardless.


### Hierarchy-specific scale parameters

These only appear in models that use hierarchical pooling.

**σ_global** (between-family SD): E[σ] ≈ 3 → Exponential(1/3).
Current HalfNormal(3) has E[σ] ≈ 2.39.

**σ_family** (within-family SD): E[σ] ≈ 1.5 → Exponential(1/1.5).
Current HalfNormal(2) has E[σ] ≈ 1.60.

**μ_global** (global mean difficulty): E[μ] = 6, Var = 9 → Normal(6, 9).
Current prior matches.  ✓

These priors should be identical across all models that have them, but
models without hierarchy (e.g., M6 below) simply don't have these
parameters — they're not "shared" across models that structurally differ.


## 2. Competing models

The models differ in structure — hierarchy, link function, parameterization
of how task difficulty affects agent success.  They need not share the same
internal architecture.  They need only predict the same observations.


### M0: Hierarchical, additive residual (current model)

```
η_ai = α_a + β_a · μ_task_i + ε_i
P(success) = logistic(η_ai)
score_ai ~ Bernoulli(P)

ε_i ~ Normal(0, σ_ε)        # AI-specific task residual
σ_ε ~ Exponential(1/1.5)    # MaxEnt given E[σ_ε] ≈ 1.5
```

This is the current hierarchical model with MaxEnt priors substituted for
the HalfNormals.  The additive ε_i captures "AI-specific difficulty" — a
task that's harder (or easier) for AI than its human time would predict.
The same ε_i applies to ALL agents.

**Parameter count (model-specific)**: 1 per task (ε_i) + 1 (σ_ε) = n_tasks + 1.

**Structural assumption**: Tasks differ in AI difficulty only by a constant
additive shift in log-odds, the same for all agents.


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
σ_ε ~ Exponential(1/1.5)
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

Everything else identical to M0.  The ONLY difference is the link function.

**Jaynesian argument for logistic over probit**: The logistic function is
the maximum entropy distribution on {0, 1} given the linear predictor η
(Jaynes Ch. 11).  The probit assumes a latent Gaussian threshold model
(Y = 1 iff Z > 0, Z ~ N(η, 1)), which is an additional structural
assumption beyond what the data require.

However: both are nearly indistinguishable for moderate η, differing mainly
in the tails (logistic has heavier tails).  If the data contain many
observations in the tails (tasks where success is near 0% or near 100%),
the comparison will be informative.  Otherwise, the Bayes factor will be
close to 1 — the data can't tell them apart.

**No new parameters.  No new priors.**  This is a pure structural comparison.


### M4: Estimate bias

Replaces the unbiased estimate model with:

```
log₂(estimate_i) ~ Normal(μ_task_i + δ, σ_estimate)
```

where δ captures systematic bias in expert time estimates.

**Derivation of prior for δ**:

What we know: δ is a location parameter (bias in log₂ space).  A positive δ
means experts overestimate difficulty (pessimistic); negative means they
underestimate (optimistic / planning fallacy).  We don't know the sign.

If we claim E[δ] = 0 and SD[δ] ≈ 1 (a factor-of-2 bias in either direction
is plausible):

MaxEnt on ℝ: **Normal(0, 1)**.

If we claim only E[δ] = 0 with no variance constraint: the MaxEnt
distribution is improper (flat).  Since δ is identifiable from the ~30+ tasks
that have both baselines and estimates, a flat prior is actually usable here
— the posterior will be proper.  But Normal(0, 1) is safer and still weakly
informative.

**Parameter count**: 1 (δ is shared across all estimate-only tasks).

**Structural assumption**: Expert estimates may be systematically biased in
one direction, but the bias is the same for all tasks.


### M5: Per-source σ_human

Replaces the single σ_human with source-specific noise:

```
σ_human_HCAST ~ Exponential(1)
σ_human_SWAA  ~ Exponential(1)

# HCAST runs:
log₂(duration_ij) ~ Normal(μ_task_i, σ_human_HCAST)

# SWAA runs:
log₂(duration_ij) ~ Normal(μ_task_i, σ_human_SWAA)
```

**Prior**: Both σ's get Exponential(1) — identical prior, different data
subsets.  Same MaxEnt argument as σ_human in §1.

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
σ_human ~ Exponential(1)
log₂(duration_ij) ~ Normal(μ_task_i, σ_human)

# Expert estimates (same)
σ_estimate ~ Exponential(1/2)
log₂(estimate_i) ~ Normal(μ_task_i, σ_estimate)

# Agent success (same as M0)
α_a ~ Normal(0, 25)
β_a ~ ReflectedExponential(2)   [i.e., p(β) = 2·exp(2β), β ≤ 0]
ε_i ~ Normal(0, σ_ε)
σ_ε ~ Exponential(1/1.5)

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
ν_i ~ Normal(0, σ_ν)          # residual

# Priors
γ_0 ~ Normal(6, 9)            # intercept (same as μ_global)
γ_source ~ Normal(0, 4)       # MaxEnt, E = 0, rough scale
γ_family ~ Normal(0, 4)       # MaxEnt, E = 0, rough scale
σ_ν ~ Exponential(1/2)
```

where x_source and x_family are encoded task features (one-hot or similar).
This tests whether the hierarchy's family structure is better captured by
explicit features than by random effects.

The regression coefficients γ get Normal(0, σ²) priors — MaxEnt given
mean zero (no prior directional belief) and a rough variance.


### M8: IRT-style scalar ability (Moss's approach, joint version)

Replace the per-agent (α, β) with a single scalar ability θ_a, and put
per-task discrimination on the task side:

```
# Task difficulty hierarchy (same as M0)
μ_task_i ~ [hierarchical or flat]

# Agent ability — single scalar
θ_a ~ Normal(0, σ_θ)
σ_θ ~ Exponential(1)

# Per-task discrimination
a_i ~ Exponential(1)

η_ai = a_i · (θ_a − μ_task_i)
score_ai ~ Bernoulli(logistic(η_ai))
```

This is a fundamentally different parameterization of agents.  Instead of
each agent having its own intercept and slope (2 parameters), each agent
has one scalar ability θ.  The tradeoff: the model is more parsimonious
per agent (1 parameter vs. 2) but can't express "good at easy tasks, bad
at hard ones" vs. "mediocre everywhere" — all agents with the same θ
behave identically.

**Prior for θ_a**: Normal(0, σ_θ) with σ_θ ~ Exponential(1).  The
hierarchical prior on θ lets the data determine the spread of abilities.
MaxEnt for the hyperprior: we know σ_θ > 0 and have a rough expected
value of ~1 (abilities spread over a few units on the logit scale).

**This model cannot be compared by sharing α/β priors with M0**, because
the parameters are structurally different.  The comparison is purely through
the marginal likelihood.  This is fine — two models with different
parameterizations but the same observables are compared through how well
each predicts the data, weighted by how much prior mass each wastes on
regions of parameter space the data don't support.


## 3. Which comparisons to actually run

Not all pairs are interesting.  Organize by what hypothesis each comparison
tests:

**Does hierarchy help?**
```
M0 (hierarchical) vs. M6 (flat) vs. M7 (feature regression)
```
Tests whether family-level partial pooling, independent task priors, or
explicit features best explain the task difficulty structure.

**How do tasks affect agents?**
```
M0 (additive residual) vs. M1 (multiplicative discrimination) vs. M2 (both)
```
Tests whether AI-specific task difficulty is additive (same log-odds shift
for all agents) or multiplicative (some tasks more diagnostic than others).

**How are agents parameterized?**
```
M0 (per-agent α, β) vs. M8 (scalar ability θ + per-task discrimination)
```
Tests whether agents differ in a single dimension (ability) or two
dimensions (intercept + slope).

**Link function:**
```
M0 vs. M3 (probit)
```
Probably uninteresting — logistic and probit are nearly indistinguishable
except in the tails.

**Observation model refinements (orthogonal, can combine freely):**
```
M4 (estimate bias) — are expert estimates systematically off?
M5 (per-source σ) — do HCAST and SWAA have different timing noise?
```

The most consequential comparisons are M0 vs. M6 (hierarchy worth it?),
M0 vs. M1 (additive vs. multiplicative), and M0 vs. M8 (per-agent slopes
vs. scalar ability).


## 4. Computing the marginal likelihood

For PyMC models, the marginal likelihood is not available in closed form.
Options:

**Bridge sampling** (recommended): The `bridgesampling` approach (Gronau
et al., 2017) estimates log p(data | model) from posterior samples with
good accuracy.  The ArviZ library has experimental support; alternatively,
use the `harmonic mean estimator` (unstable) or `stepping-stone sampling`
(accurate but expensive).

**LOO-CV** (practical alternative): Leave-one-out cross-validation via
Pareto-smoothed importance sampling (PSIS-LOO, Vehtari et al., 2017) is
implemented in ArviZ as `az.loo()`.  It approximates the expected log
pointwise predictive density (ELPD), which is closely related to the
marginal likelihood for well-specified models.  It's model-comparison
without explicit prior dependence — but this means it doesn't test the
prior specification, only the likelihood structure.

**Recommendation**: Use LOO-CV (PSIS-LOO via ArviZ) for practical model
comparison, and bridge sampling for the principled Bayesian comparison
where prior choice matters.  If the two methods agree, the result is robust.


## 5. Where the current priors diverge from MaxEnt

| Parameter | Current | MaxEnt | Difference |
|-----------|---------|--------|------------|
| σ_global | HalfNormal(3) | Exponential(1/3) | Shape: HN encodes E[σ²] = 9; Exp doesn't |
| σ_family | HalfNormal(2) | Exponential(1/2) | Same issue |
| σ_human | HalfNormal(1.5) | Exponential(1) | HN has E ≈ 1.20, Exp has E = 1.0 |
| σ_estimate | HalfNormal(2.5) | Exponential(1/2) | HN has E ≈ 2.0, Exp has E = 2.0 (coincidence) |
| σ_epsilon | HalfNormal(1.5) | Exponential(1/1.5) | Shape differs |
| β_a | TruncNorm(-0.5, 1.5, ≤0) | Reflected Exp(2) | **Effective E[β] = -1.4 vs. -0.5** |
| μ_global | Normal(6, 9) | Normal(6, 9) | ✓ Matches |
| α_a | Normal(0, 25) | Normal(0, 25) | ✓ Matches |

The most consequential divergence is **β_a**.  The TruncatedNormal(-0.5, 1.5)
has effective mean -1.4 after truncation — nearly 3× steeper than the stated
intent of E[β] = -0.5.  This pulls the prior toward steep slopes, which
systematically shortens estimated time horizons (since T ∝ 1/|β|).

The scale parameter divergences (HalfNormal vs. Exponential) are less
consequential because (a) both are proper and weakly informative, and (b)
with hundreds of observations, the posterior is data-dominated.  But for the
model comparison to be "fair" in Jaynes's sense, all competing models should
use the same (MaxEnt) priors for shared parameters.


## 6. Summary: the honest Jaynesian recipe

1. **State your constraints explicitly.**  For each parameter, write down what
   you actually know — sign, rough mean, rough variance, symmetry.  Nothing
   else.

2. **Derive the MaxEnt distribution from those constraints.**  Don't pick a
   distribution family and then set its hyperparameters — derive the family
   from the constraints.

3. **Shared parameters get identical priors across models.**

4. **Model-specific parameters get MaxEnt priors from their own constraints.**
   For discrimination a_i: Exponential(1).  For bias δ: Normal(0, 1) or flat.

5. **Compute marginal likelihoods (or LOO-CV) and compare.**  The Bayes factor
   is the right summary.  If |log K| < 1, the data don't distinguish the
   models.  Report this honestly — "the data can't tell" is a valid finding.

6. **Do a prior sensitivity check anyway.**  Vary the constraint values (e.g.,
   E[σ_human] = 0.5 vs. 1.0 vs. 2.0) and check that the Bayes factors are
   stable.  If they're not, the comparison is prior-dependent and you should
   say so.
