# Gray-Box Residual and Closure Learning

Gray-box modeling starts from the observation that we often know too much to
justify a fully black-box model, but too little to trust a purely first
principles description. The known model may capture geometry, balance laws,
kinematic structure, or dominant forcing mechanisms, while still missing a
constitutive relation, an unmodeled damping term, a closure stress, or a
source term induced by unresolved scales. In that regime, learning the entire
dynamics from scratch wastes reliable prior structure. Learning only the
missing part is usually the more defensible problem statement.

This chapter studies that intermediate setting. The goal is to combine known
physics with a learned correction in a way that improves prediction without
destroying the structure that made the original model valuable. That viewpoint
is central in hybrid physics-ML workflows: the question is not only how to fit
data, but also where a learned component should enter the equations and what
constraints it should respect.

## From Known Dynamics to Residual Models

Suppose a continuous-time model is available in partially known form:

```{math}
:label: eq:ch08-graybox-continuous

\dot{x}(t)=f_{\mathrm{known}}(x(t),u(t);\mu)+g_\theta(x(t),u(t)).
```

The term $f_{\mathrm{known}}$ encodes the trusted part of the mechanism, while
$g_\theta$ is a learned residual. In discrete time the same idea becomes

```{math}
:label: eq:ch08-graybox-discrete

x_{k+1}=F_{\mathrm{known}}(x_k,u_k;\mu)+G_\theta(x_k,u_k).
```

This is the basic gray-box decomposition. It is useful when the dominant
structure is known and only a mismatch remains between the model and the
observed dynamics. That mismatch may come from neglected forces, coarse
discretization, empirical constitutive behavior, or missing dependence on an
operating condition.

The separation between known and learned terms is not cosmetic. It changes the
data requirement. Since the learner is not responsible for reconstructing the
entire map or vector field, fewer trajectories may be needed to estimate the
residual than would be needed to fit a black-box model of the full system. The
known term also supplies an inductive bias that can improve extrapolation away
from the center of the training set, provided the model-form assumptions are
basically correct.

## What Counts as a Residual?

The most common case is additive correction as in
{eq}`eq:ch08-graybox-continuous` and {eq}`eq:ch08-graybox-discrete`, but that
is not the only option.

### Additive forcing, damping, and source terms

If the missing physics appears as an extra force, damping law, reaction source,
or closure stress, an additive residual is the natural form. For example, a
reduced model may have the correct transport terms but an inaccurate effective
dissipation. Then $g_\theta$ should be interpreted as a structured missing
contribution rather than an arbitrary function.

### Unknown constitutive relations

Sometimes the governing equations are known except for an embedded constitutive
map. Then the correction is better introduced inside that law than appended at
the end of the state equation. For instance, a flux function, drag law, or
stress-strain relation may be parameterized as

```{math}
:label: eq:ch08-constitutive

\sigma=\sigma_{\theta}(x,u,\nabla x,\mu),
```

with the state equation depending on $\sigma$ through a known operator. This is
still gray-box learning, but the learned object is a constitutive component,
not a free residual on $\dot{x}$.

### Multiplicative or operator-valued corrections

In some systems the model error changes the strength of an existing mechanism
rather than adding a new one. Then a multiplicative or operator-valued
correction may be more appropriate than an additive residual. The practical
lesson is that the insertion point matters. We should learn the missing physics
where it naturally lives in the equations, not automatically at the outermost
level.

## Residual Learning Versus Closure Learning

The term "closure" is often used when the known equations describe resolved
variables but depend on quantities that cannot be computed from the resolved
state alone. A reduced model for $x_a$ may take the form

```{math}
:label: eq:ch08-closure

\dot{x}_a=f_a(x_a,u)+c_\theta(x_a,u),
```

where $c_\theta$ approximates the aggregate effect of unresolved physics on the
resolved dynamics. In fluid and multiscale settings this learned term may stand
in for Reynolds stresses, subgrid fluxes, effective diffusion, or unresolved
chemical source terms.

Residual learning and closure learning overlap, but the emphasis differs:

- Residual learning starts from model mismatch between prediction and data.
- Closure learning starts from missing mechanisms needed to make a reduced
  description self-contained.

That distinction matters because a residual can sometimes be corrected by a
simple function of the current resolved state, while a true closure may encode
systematic dependence on filtered variables, coarse observables, or operating
regime. When that dependence cannot be expressed as an instantaneous function
of the available state, Chapter 09's non-Markovian viewpoint becomes necessary.

## Where Should the Learned Term Enter?

A central modeling decision is the placement of the learned component.

### Correct the evolution law directly

The most direct choice is to learn $g_\theta$ in the state equation itself.
This is appropriate when the known model already uses the right state,
coordinates, and constitutive variables, but its rate law is quantitatively
incomplete.

### Correct a parameterization inside the physics

If the structure of the equation is correct but one coefficient law is wrong,
then it is better to learn that coefficient law explicitly. This preserves
interpretability and often makes the learned term easier to constrain.

### Correct the observation-to-state interface

In practice, some "dynamics" error is really an error in how external inputs,
boundary conditions, or operating parameters enter the model. Before adding a
large residual to the dynamics, it is worth checking whether the mismatch comes
from a poor forcing model or an omitted parameter dependence.

These options are not equivalent. A highly expressive residual can absorb many
different mistakes, but that flexibility can blur physical interpretation and
hurt extrapolation. Good gray-box modeling asks first where the physics is
incomplete, then chooses the learned term accordingly.

## Learning Objectives and Loss Design

Suppose we observe trajectories through

```{math}
:label: eq:ch08-observation

y_k=h(x_k)+\eta_k.
```

If full-state data are available, one may fit the residual from approximate
derivative or one-step targets. With partial observations, the residual is
usually learned through rollout consistency, filtering or smoothing, or an
optimization over latent trajectories. In either case, the objective should
match the intended use of the model.

A generic discrete-time training objective is

```{math}
:label: eq:ch08-objective

\min_{\theta}\;
\sum_{r=1}^{N_{\mathrm{traj}}}
\sum_{k=0}^{N_r-1}
\ell\!\left(
\hat{x}^{(r)}_{k+1},
F_{\mathrm{known}}\!\left(\hat{x}^{(r)}_k,u^{(r)}_k;\mu^{(r)}\right)
+G_\theta\!\left(\hat{x}^{(r)}_k,u^{(r)}_k\right)
\right)
+\lambda \, \mathcal{R}(\theta),
```

where $\hat{x}^{(r)}_k$ denotes a state estimate or measured full state,
$\ell$ is a mismatch penalty, and $\mathcal{R}(\theta)$ encodes regularization
or structural constraints. Depending on the application, $\ell$ may measure
one-step prediction, multi-step rollout error, violation of a balance law, or
disagreement with observed outputs after simulating the gray-box model.

Because the known term already explains part of the evolution, the regularizer
should usually encourage the learned term to remain small unless the data
clearly support a correction. Otherwise the residual can overwhelm the
mechanistic prior and effectively collapse the model back into a black box.

## Choosing a Residual Model Class

The syllabus highlights several model classes for the correction term. The
choice should reflect the expected structure of the missing physics, not only
function-approximation power.

### Sparse and low-complexity residuals

If the correction is expected to be simple and interpretable, a sparse library
model is attractive. One may write

```{math}
:label: eq:ch08-sparse-residual

g_\theta(x,u)=\Theta(x,u)\xi,
```

where $\Theta$ is a candidate feature library and $\xi$ is encouraged to be
sparse. This is useful when the known model is nearly correct and the missing
term is expected to have a small analytic representation.

### Neural residuals

Neural networks are useful when the missing mechanism is strongly nonlinear,
high-dimensional, or hard to encode by hand. They can approximate complex
closure laws, but their flexibility increases the risk that the learned term
compensates for data issues, discretization errors, or weakly observed state
components rather than genuine missing physics.

### Gaussian-process corrections

Gaussian processes are appealing when uncertainty quantification is essential
and the correction is relatively smooth and low-dimensional. A GP residual can
represent model discrepancy together with posterior uncertainty, which is often
valuable when the gray-box model will be used for downstream decision making.

No class is uniformly best. Sparse corrections favor interpretability, neural
corrections favor expressive power, and GP corrections favor calibrated local
uncertainty. The modeling question is what kind of mismatch we believe is left
after the known physics has been accounted for.

## Constraint-Aware and Conservation-Aware Residuals

A learned correction should not casually violate the structural properties that
made the known model meaningful. If the physical system obeys conservation,
symmetry, positivity, passivity, or invariance constraints, then the residual
should respect them whenever possible.

There are several ways to enforce that requirement:

- Choose a parameterization that satisfies the constraint by construction.
- Add penalty terms that discourage violation of conservation or balance laws.
- Project the residual onto an admissible subspace.
- Learn only in coordinates where the constraint is easier to enforce.

For example, if total mass must be conserved, then an additive source term with
nonzero net production is unacceptable unless the apparent imbalance is itself
the unresolved mechanism being modeled. Likewise, if an energy-dissipation rate
should remain nonnegative, the residual should not create energy in regimes
where the underlying physics forbids it.

This is one of the main advantages of gray-box modeling over a purely black-box
approach. The known structure gives us a place to encode hard constraints, and
the learned correction can then focus on the remaining discrepancy.

## Model-Form Error and Discrepancy Modeling

Gray-box learning is often motivated by model-form error: the governing
equations are approximately right but not exact. In that setting, the learned
term should be interpreted as a discrepancy model, not as ground-truth physics
discovered directly from data.

That interpretation has two important consequences.

First, the residual depends on the baseline model. If
$f_{\mathrm{known}}$ changes, the learned discrepancy usually changes too.
Residuals are therefore not universally portable across model classes.

Second, a residual can hide multiple sources of error at once: missing
mechanisms, parameter misspecification, coarse numerics, unmodeled boundary
effects, or biased state estimates. The better the baseline physics and data
pipeline are understood, the more meaningful the learned discrepancy becomes.

## Why Partial Physics Can Help

When the decomposition is chosen well, partial physics improves learning in two
ways.

### Better sample efficiency

The known model explains broad qualitative behavior, so the learner only has to
estimate the correction. This reduces the burden on the data and often makes
the fitted model more stable in low-data regimes.

### Better extrapolation

A gray-box model inherits physically reasonable trends from the trusted
equations. Even if the correction is learned from a narrow operating window,
the baseline model can anchor behavior outside that window better than a purely
black-box fit. This benefit is real only if the baseline structure remains
valid in the extrapolation regime; otherwise the residual may be forced to
repair errors it cannot represent.

## Failure Modes

Gray-box models are not automatically safer than black-box models. Common
failure modes include:

- placing the residual in the wrong part of the model, so it learns an
  unnatural correction;
- giving the residual too much flexibility, so it overrides the known physics;
- using a residual to compensate for hidden-state or memory effects that
  require state augmentation instead;
- enforcing constraints too weakly, so short-term fit improves while long-term
  physical fidelity degrades;
- trusting extrapolation because the model is "physics informed" even when the
  baseline equations are wrong in the new regime.

The discipline of gray-box learning is to be explicit about what is known, what
is missing, and what evidence supports the chosen form of the correction.

## Main Takeaways

Gray-box residual and closure learning occupies the middle ground between
inverse problems in known models and fully black-box dynamics learning. The key
idea is to learn only the discrepancy that the known physics leaves behind.

Additive residuals are common, but the most important design question is where
the learned term should enter the equations. In some problems the right object
is a missing force or source term; in others it is a constitutive law or a
closure for unresolved scales.

Partial physics can improve sample efficiency, interpretability, and
extrapolation, but only if the learned correction respects the constraints and
mechanistic structure that justify the gray-box decomposition in the first
place.
