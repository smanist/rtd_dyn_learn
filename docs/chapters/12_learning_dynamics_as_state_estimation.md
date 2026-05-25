# Learning Dynamics as State Estimation

Many dynamics-learning methods are first introduced as regression problems:
fit a map $F$ from $(x_k,u_k)$ to $x_{k+1}$, or fit a vector field $f$ from
state samples and estimated derivatives. That viewpoint works well when the
state is measured directly and noise is modest. The situation changes when the
observations are partial, noisy, or irregular enough that the hidden trajectory
itself is uncertain. Then learning the dynamics is no longer only about
choosing parameters. It is also about reconstructing the state history that the
parameters are meant to explain.

This chapter develops that shift in viewpoint. Rather than treating state
reconstruction as a separate preprocessing step, we treat it as part of the
learning problem. The unknown trajectory $x_{0:N}$ becomes a decision variable
alongside the unknown model parameter $\theta$. That reinterpretation connects
dynamics learning to filtering, smoothing, data assimilation, and latent-state
inference, and it explains why estimation-based formulations are often more
robust under realistic measurements.

## A State-Space View of Learning

The basic model is a discrete-time state-space system with process and
measurement noise:

```{math}
:label: eq:ch12-state-space-model

x_{k+1}=F(x_k,u_k;\theta)+w_k,
\qquad
y_k=h(x_k,u_k;\theta)+v_k.
```

Here $x_k$ is the latent state, $u_k$ is a known input, $y_k$ is the
observation, $w_k$ is process noise, and $v_k$ is measurement noise. Following
the course notation convention, $Q$ and $R$ denote the covariances of $w_k$ and
$v_k$, respectively. Equation {eq}`eq:ch12-state-space-model` is more than a
statistical wrapper around a dynamics model. It changes the learning problem
because the state sequence is no longer assumed known.

This matters in at least three common situations:

- Only a subset of the state is observed, so $y_k$ does not determine $x_k$.
- Measurements are noisy enough that direct differentiation or regression would
  amplify error.
- The model is only approximate, so part of the mismatch should be assigned to
  state uncertainty or model error rather than to a single deterministic fit.

Under this viewpoint, a dataset of measurements does not specify a unique
trajectory. It specifies a family of plausible hidden trajectories, and
learning means selecting both a model and a trajectory that are jointly
consistent with the data.

## Why the Hidden Trajectory Becomes a Learning Variable

If full-state data were available exactly, then learning $\theta$ could reduce
to fitting the evolution law directly. With partial observations, that is no
longer possible. The unknown dynamics generate $x_k$, but the learner only sees
$y_k=h(x_k,u_k;\theta)+v_k$. Any parameter update therefore depends on what
trajectory we believe produced the observations, and any trajectory estimate
depends on what dynamics we currently believe.

This coupling is the core reason state estimation enters dynamics learning. In
place of a regression only over $\theta$, we obtain a joint inference problem
over $(x_{0:N},\theta)$. A convenient deterministic form is to minimize a
penalty that balances observation fit against dynamical consistency:

```{math}
:label: eq:ch12-joint-objective

\min_{x_{0:N},\theta}
\mathcal{J}(x_{0:N},\theta)
=
\frac{1}{2}\sum_{k=0}^{N}\|y_k-h(x_k,u_k;\theta)\|_{R^{-1}}^2
+
\frac{1}{2}\sum_{k=0}^{N-1}\|x_{k+1}-F(x_k,u_k;\theta)\|_{Q^{-1}}^2.
```

The first term asks the reconstructed trajectory to explain the measurements.
The second asks consecutive states to agree with the learned dynamics. This is
already a derivative-free learning formulation: nowhere do we require direct
estimates of $\dot{x}(t)$ from noisy data. Instead, the temporal structure is
enforced through the model itself.

Equation {eq}`eq:ch12-joint-objective` also makes a conceptual point. When
observations are incomplete, parameter learning and trajectory reconstruction
should not be separated too early. Solving only for $\theta$ while pretending
that $x_k$ is known often leads to biased models, because the unexplained
observation error gets folded into the wrong part of the problem.

## Strong-Constraint and Weak-Constraint Formulations

One of the main modeling choices is whether the dynamics are enforced exactly or
allowed to have explicit model error.

### Strong Constraint

In a strong-constraint formulation, the dynamics are treated as exact:

```{math}
:label: eq:ch12-strong-constraint

x_{k+1}=F(x_k,u_k;\theta),
\qquad
y_k=h(x_k,u_k;\theta)+v_k.
```

The trajectory is then determined by $x_0$ and $\theta$, and the learning
problem reduces to estimating those unknowns from the observations. This view
is appropriate when model mismatch is believed small relative to measurement
noise, or when the main uncertainty lies in the initial condition.

### Weak Constraint

In a weak-constraint formulation, we allow the state transition itself to be
imperfect:

```{math}
:label: eq:ch12-weak-constraint-residual

q_k=x_{k+1}-F(x_k,u_k;\theta).
```

The variables $q_k$ represent model error, unresolved forcing, discretization
mismatch, or other departures from exact dynamics. The second term in
{eq}`eq:ch12-joint-objective` penalizes these residuals according to $Q$ rather
than forcing them to vanish.

The weak-constraint view is often the more realistic one for learning from data
because the model class is rarely perfect. A learned map may omit unresolved
state variables, neglected disturbances, or unmodeled physics. Process noise is
therefore not just random perturbation added for convenience. It is a formal
way to represent model inadequacy.

## Filtering Versus Smoothing

Once learning is posed as latent-state inference, the distinction between
filtering and smoothing becomes central.

Filtering estimates the current state from past and present data:

```{math}
:label: eq:ch12-filter

\widehat{x}_{k|k}
=
\mathbb{E}[x_k\mid y_{0:k}].
```

Smoothing uses the entire observation record:

```{math}
:label: eq:ch12-smoother

\widehat{x}_{k|N}
=
\mathbb{E}[x_k\mid y_{0:N}].
```

For online monitoring or control, filtering is the natural object because future
measurements are unavailable. For learning, smoothing is usually more valuable.
When estimating $\theta$, we want the most informative reconstruction of the
whole trajectory, and later observations often clarify earlier hidden states.
This is why many identification procedures under partial observation look more
like smoothing or batch estimation than like one-pass online prediction.

The same distinction appears in optimization language. A recursive estimator
updates the current belief as data arrive, while a batch estimator revisits the
entire trajectory after all measurements are known. Both come from the same
state-estimation viewpoint, but they serve different computational and
operational needs.

## Process Noise as Model Error

One of the most useful consequences of the estimation viewpoint is that process
noise can absorb structured imperfections in the model. Suppose the true system
contains unresolved dynamics, neglected inputs, or discretization effects. If
we force a deterministic model to explain every discrepancy, the parameter
estimate $\theta$ may drift toward values that compensate for missing physics.

Allowing nonzero $w_k$ changes that interpretation. The learned model is then
asked to capture the predictable part of the dynamics, while $w_k$ explains the
part that is transient, unresolved, or outside the model class. The covariance
$Q$ becomes a modeling choice: a small $Q$ expresses trust in the dynamics,
while a large $Q$ allows more freedom in the latent trajectory. Tuning $Q$ is
therefore not merely a numerical detail. It encodes how much of the mismatch we
attribute to the model versus to hidden disturbances.

This idea improves robustness in several ways:

- It reduces pressure to overfit noisy observations by distorting the dynamics.
- It allows partial-state models to remain useful even when unresolved effects
  cannot be captured deterministically.
- It provides a principled route to uncertainty quantification through state and
  parameter covariances or posterior distributions.

## Why This Viewpoint Helps Learning

Recasting learning as state estimation is especially useful when direct
full-state regression is fragile or impossible.

### Partial Observations

If only outputs are measured, then hidden-state reconstruction is unavoidable.
The state-estimation viewpoint makes that fact explicit instead of hiding it
inside ad hoc preprocessing steps. Observability and sensor design therefore
become part of the learning problem, not separate concerns.

### Noisy Data

If measurements are noisy, direct derivative estimation can be unstable.
Jointly reconstructing the trajectory and the dynamics avoids differentiating
noise-amplified signals. The dynamics impose temporal coherence, while the
observation model explains what is actually measured.

### Uncertainty Quantification

A regression fit can return a point estimate for $\theta$ without saying much
about hidden-state ambiguity. Estimation-based formulations naturally expose
where the data are informative and where they are not. Large posterior
covariances, flat objective directions, or strong dependence on priors all
signal that the data do not uniquely determine the latent trajectory or the
parameters.

### Robustness to Model Mismatch

Weak-constraint formulations acknowledge that the chosen model class may be
incomplete. That makes the learned model more honest: it can represent the
predictable dynamics without falsely claiming that every residual has been
explained mechanistically.

## Relation to Later Chapters

This chapter is conceptual rather than algorithmic. Its main purpose is to
replace a narrow regression mindset with a latent-state viewpoint that will
support later methods. The next chapters make that viewpoint concrete: linear
Gaussian filtering and smoothing provide closed-form estimators in special
cases; nonlinear, ensemble, and moving-horizon methods extend the same ideas to
harder systems; and joint state-parameter methods turn the coupling between
$x_{0:N}$ and $\theta$ into explicit iterative algorithms.

The key lesson is that under partial observation or significant noise, learning
dynamics is not only about fitting equations. It is about inferring the hidden
trajectory that makes those equations meaningful.
