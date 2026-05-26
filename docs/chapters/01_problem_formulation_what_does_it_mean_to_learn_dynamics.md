# Problem Formulation: What Does It Mean to Learn Dynamics?

Before choosing an identification, estimation, or control method, we need to be
clear about the object being learned. In this course, "learning dynamics" does
not mean one single task. It can mean learning a discrete-time state-update
rule, a continuous-time vector field, unknown parameters inside a known model,
a closure term for missing physics, a latent state that makes the dynamics
Markovian, or a surrogate model designed mainly for downstream control.

That distinction matters because the same dataset can support different
questions. A model that is adequate for short-horizon prediction may be poor
for scientific explanation. A closure model that improves rollout accuracy may
still hide unobservable states. A control-oriented surrogate may only need to
be reliable near a planned operating regime. Problem formulation comes first
because it determines what data are informative, what assumptions are
reasonable, and what validation should count as success.

## A Common Starting Point

We will move between continuous-time mathematical models and sampled datasets.
The notation convention for this course uses italic symbols for mathematical
objects and bold symbols for stored numerical data. A general continuous-time
model takes the form

```{math}
:label: eq:ch01-continuous-dynamics

\dot{x}(t)=f(x(t),u(t);\mu,\theta),
\qquad
x(t)\in \mathbb{R}^n,
\quad
u(t)\in \mathbb{R}^m.
```

The corresponding sampled, discrete-time description is

```{math}
:label: eq:ch01-discrete-dynamics

x_{k+1}=F(x_k,u_k;\mu,\theta).
```

When the full state is not directly observed, the data are filtered through an
observation map:

```{math}
:label: eq:ch01-observation-model

y_k=h(x_k)+\eta_k,
\qquad
y_k\in \mathbb{R}^p.
```

Here, $\theta$ denotes unknown parameters to be learned, while $\mu$ denotes a
known operating parameter that indexes a family of systems. If we store sampled
trajectories numerically, a dataset may be written as

```{math}
:label: eq:ch01-dataset

\mathcal{D}
=
\left\{
\left(\mathbf{u}^{(r)}_k,\mathbf{y}^{(r)}_k\right)_{k=0}^{N_r}
\right\}_{r=1}^{N_{\mathrm{traj}}},
```

or, in the full-state case, by replacing $\mathbf{y}^{(r)}_k$ with
$\mathbf{x}^{(r)}_k$.

Equations {eq}`eq:ch01-continuous-dynamics` through
{eq}`eq:ch01-observation-model` already show why problem formulation is not
trivial: the unknown may live in the dynamics, the observation process, the
hidden state, or all three.

## What Exactly Is Being Learned?

The first classification question is the learned object itself.

### Learning a Discrete-Time Map

If the sampling interval is fixed and prediction at sample times is the main
goal, then learning the map $F$ in {eq}`eq:ch01-discrete-dynamics` is often the
most direct formulation. This viewpoint is natural when the available data are
already discrete, when numerical differentiation would be noisy, or when the
downstream task uses one-step or multi-step prediction on the sampled grid.

### Learning a Continuous-Time Vector Field

If the scientific object of interest is the underlying mechanism, then learning
the vector field $f$ in {eq}`eq:ch01-continuous-dynamics` is usually the better
statement of the problem. This formulation is more sensitive to sampling and
noise, but it separates the model from any particular numerical integrator and
supports analysis of equilibria, stability, and parameter dependence in
continuous time.

### Learning Parameters Inside a Known Structure

Sometimes the model class is known and only a parameter vector is missing. Then
the problem is not to discover the form of $f$ or $F$, but to estimate
$\theta$ in a structured model such as $f(x,u;\theta)$. This is an inverse
problem: the main question is whether the available data constrain the
parameters strongly enough to determine them in practice.

### Learning Residuals, Closures, or Latent States

In many systems, part of the physics is known but incomplete. A common
formulation is residual learning:

```{math}
:label: eq:ch01-residual-model

\dot{x}=f_{\mathrm{known}}(x,u;\mu)+g_\theta(x,u).
```

Here, the learned object is the correction term $g_\theta$, not the entire
dynamics. In other settings the missing ingredient is unresolved state rather
than a missing force law. Then the learned object may be a latent variable
$z_k$, a closure law, or a memory term that restores predictive sufficiency.

### Learning a Control-Relevant Model

A model used for control is judged by different standards than a model used for
explanation. The learned object might be a locally accurate surrogate,
a linearization around an operating point, or an input-output model that is
useful for decision making even if it is not globally faithful. This is still a
dynamics-learning problem, but the objective is shaped by the controller that
will consume the model.

## A Taxonomy of Dynamics-Learning Problems

Once the learned object is clear, the next step is to classify the task along a
few axes that recur throughout the course.

### Time Representation

The first axis is whether the system is formulated in discrete time or
continuous time. Map learning and vector-field learning are related, but they
are not interchangeable. A discrete-time model absorbs the effect of the sample
interval into $F$, while a continuous-time model keeps the underlying rate law
explicit through $f$.

### Observation Type

The second axis is whether the state is directly observed. Full-state data make
the learning task closer to regression on $x_k$ or $\dot{x}(t)$. Partial-state
data require extra structure because the unknown state must be inferred through
the observation model {eq}`eq:ch01-observation-model`. That is where
observability, filtering, smoothing, and latent-state modeling enter.

### Inputs and Operating Parameters

Some systems are autonomous, some are driven by control inputs, and some vary
with known operating conditions. These three cases should be separated early:

- Autonomous dynamics: $x_{k+1}=F(x_k)$ or $\dot{x}=f(x)$.
- Controlled dynamics: $x_{k+1}=F(x_k,u_k)$ or $\dot{x}=f(x,u)$.
- Parametric dynamics: $x_{k+1}=F(x_k,u_k;\mu)$ or $\dot{x}=f(x,u;\mu)$.

The data requirements change across these settings because the learner must
distinguish intrinsic evolution from forced response and from dependence on
$\mu$.

### Model Knowledge

The prior level of model knowledge also matters. At one end, the structure is
known and only parameters are unknown. In the middle, some physics is known but
closure terms or constitutive laws are missing. At the other end, the model is
treated as largely unknown and must be learned from data with only generic
regularity or architectural assumptions.

### Intended Use

Finally, the same dynamical system can be learned for very different purposes:

- Prediction: forecast future trajectories accurately.
- Explanation: reveal interpretable governing mechanisms.
- Estimation: infer hidden states or disturbances from observations.
- Control: support planning, regulation, or closed-loop decision making.

These goals overlap, but they are not identical. Good problem formulation makes
the intended use explicit rather than assuming that every accurate predictor is
also an explanatory or control-ready model.

## What Makes the Problem Well Posed?

Four ideas should be checked at the beginning of almost every dynamics-learning
project.

### Identifiability

Identifiability asks whether distinct model choices can produce the same
observable behavior. If multiple values of $\theta$ or multiple candidate
models explain the data equally well, then the learning problem is ambiguous
before any algorithm is run.

### Observability

Observability matters whenever the full state is not measured. Even a perfect
learning algorithm cannot recover hidden variables that leave no detectable
signature in the outputs. In practice, weak observability often shows up as
large uncertainty in state estimates or severe sensitivity to noise.

### Excitation

Excitation asks whether the data probe the system richly enough to distinguish
important behaviors. A controlled system observed only near one equilibrium
cannot support broad claims about off-nominal dynamics. Likewise, constant or
nearly constant inputs usually do not reveal how the system responds to forcing.

### Validation

Validation should reflect the intended use named earlier. One-step prediction
error, rollout fidelity, recovery of qualitative features, state-estimation
accuracy, and control performance are all legitimate metrics, but they answer
different questions. Validation is therefore part of the problem statement, not
only an after-the-fact scorecard.

## Summary

This chapter frames the rest of the course. Once we know whether we are
learning a map, vector field, parameter set, closure term, latent state, or
control-relevant surrogate, the appropriate tools become clearer. Full-state,
discrete-time, model-free problems lead toward regression and system
identification methods. Partial-observation problems lead toward state
estimation and latent-variable formulations. Known-physics models point toward
inverse problems and gray-box learning. Control-motivated tasks push us to ask
whether a learned model is useful in closed loop, not only whether it fits past
data.

The main lesson is simple: learning dynamics is not one problem but a family of
problems. The hardest mistakes often come from using the right algorithm on the
wrong formulation.
