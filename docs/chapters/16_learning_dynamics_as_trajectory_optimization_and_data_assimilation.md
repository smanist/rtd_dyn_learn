# Learning Dynamics as Trajectory Optimization and Data Assimilation

Chapter 12 framed learning under partial observations as a latent-state
inference problem. This chapter takes the next step and treats the hidden
trajectory itself as an optimization variable. Instead of updating a filter one
time step at a time, we optimize over the whole path $x_{0:N}$, optionally
along with unknown parameters $\theta$, while enforcing or penalizing the
dynamics. That shift brings dynamics learning into the language of data
assimilation, nonlinear least squares, and trajectory optimization.

This viewpoint is useful when observations are sparse, horizons are long, and
the dynamics couple all time steps together. It also makes the structure of the
problem visible. The objective decomposes into measurement-misfit terms,
dynamics-consistency terms, and prior information. The derivatives inherit
sparsity from the time-local state transition map. As a result, algorithms that
look expensive at first glance often reduce to structured linear algebra on
block-banded systems.

## From Smoothing to Trajectory Optimization

Consider a sampled nonlinear state-space model

```{math}
:label: eq:ch16-state-space

x_{k+1}=F_\theta(x_k,u_k),
\qquad
y_k=h(x_k)+\eta_k.
```

If the model is treated as exact, then the latent trajectory is constrained to
lie on the manifold induced by $F_\theta$. If model error is allowed, then the
trajectory may deviate from exact propagation but must pay a penalty for doing
so. Either way, the estimation problem can be written as an optimization over
the complete path rather than a recursion over filtered summaries.

For a Gaussian prior on the initial state and Gaussian observation noise, a
maximum a posteriori trajectory estimate solves a nonlinear least-squares
problem. This is the optimization form of smoothing: rather than viewing the
posterior only as a probability distribution, we look directly for the most
likely path under the model and the data.

## Strong-Constraint Formulations

In a strong-constraint formulation, the model in
{eq}`eq:ch16-state-space` is assumed exact. The path is determined by an
initial condition and parameters, but it is often still useful to write the
problem over the whole trajectory:

```{math}
:label: eq:ch16-strong-constraint

\begin{aligned}
\min_{x_{0:N},\theta} \quad &
\mathcal{J}_{\mathrm{sc}}(x_{0:N},\theta)
=
\frac{1}{2}\norm{x_0-\overline{x}_0}_{B_0^{-1}}^2
+
\frac{1}{2}\sum_{k=0}^{N}\norm{y_k-h(x_k)}_{R_k^{-1}}^2 \\
\text{subject to} \quad &
x_{k+1}-F_\theta(x_k,u_k)=0,
\qquad k=0,\dots,N-1.
\end{aligned}
```

Here $\overline{x}_0$ is a background or prior state and $B_0$ is its
covariance. This formulation is the discrete-time analog of deterministic
smoothing with exact dynamics. The optimization variables are globally coupled
through the constraints, so local changes in one state affect the whole
trajectory.

Strong-constraint problems are appropriate when model error is negligible
relative to measurement noise, or when the main unknowns are the initial
condition and parameters. They become brittle when the model class is
misspecified, because all data mismatch must then be explained through
observation noise or parameter changes.

## Weak-Constraint Formulations and Model Error

When the model is only approximate, the path should be allowed to violate the
state update rule in a controlled way. Following the notation convention for
this module, define the model-error variable

```{math}
:label: eq:ch16-model-error

q_k=x_{k+1}-F_\theta(x_k,u_k).
```

Then a weak-constraint objective takes the form

```{math}
:label: eq:ch16-weak-constraint

\mathcal{J}(x_{0:N},\theta)
=
\frac{1}{2}\norm{x_0-\overline{x}_0}_{B_0^{-1}}^2
+
\frac{1}{2}\sum_{k=0}^{N}\norm{y_k-h(x_k)}_{R_k^{-1}}^2
+
\frac{1}{2}\sum_{k=0}^{N-1}\norm{q_k}_{Q_k^{-1}}^2.
```

The matrices $Q_k$ weight how expensive it is to depart from exact dynamics.
If $Q_k$ is small, the optimizer is pushed toward model consistency; if $Q_k$
is large, the path can absorb unmodeled disturbances or structural error.

This viewpoint connects learning and data assimilation directly. Process noise
in a probabilistic state-space model becomes a quadratic regularizer in the
trajectory objective. Weak-constraint smoothing, regularized residual fitting,
and trajectory optimization are different descriptions of essentially the same
problem.

## 3D-Var, 4D-Var, and Weak-Constraint 4D-Var

Data assimilation language emphasizes how many time levels enter each solve.

- `3D-Var` estimates the state at one analysis time from a background prior and
  observations available at that same time.
- `4D-Var` fits one trajectory over a time window while enforcing the dynamics
  exactly.
- Weak-constraint `4D-Var` relaxes exact model propagation by including the
  penalty in {eq}`eq:ch16-weak-constraint`.

The conceptual difference is important for learning dynamics. `3D-Var` is a
single-time correction method and does not by itself identify dynamical
parameters. `4D-Var` uses temporal coherence to infer initial conditions,
latent trajectories, and sometimes parameters by requiring one path to explain
an entire observation window. The weak-constraint version is often the more
realistic learning model because it separates observation noise from model
error.

## Multiple Shooting and Collocation

Long-horizon trajectory optimization can be numerically fragile if everything
is parameterized only through the initial state. Small local linearization
errors or unstable dynamics then propagate through the whole rollout. Multiple
shooting reduces that sensitivity by treating intermediate states as decision
variables and enforcing continuity through constraints or penalties.

If the interval is split into subarcs with interface states
$x_{k_0},x_{k_1},\dots,x_{k_M}$, then each segment can be integrated locally
while matching endpoints across segments. This improves conditioning and makes
the Jacobian sparser than a naive single-shooting parameterization.

Collocation goes one step further and enforces the dynamics at selected points
inside each interval. In continuous-time learning, collocation is attractive
because it couples trajectory reconstruction and discretization in one
optimization problem. It also exposes discretization mismatch directly: the
model, solver, and transcription are no longer hidden behind one forward
integration call.

## Adjoint Gradients and Costate Variables

Direct differentiation of a long unrolled trajectory is possible, but the
trajectory-optimization view suggests a more structured route. Introduce
adjoint variables $p_k$ for the dynamics constraints and define the discrete
Lagrangian

```{math}
:label: eq:ch16-lagrangian

\mathcal{L}(x_{0:N},\theta,p_{0:N-1})
=
\mathcal{J}_{\mathrm{sc}}(x_{0:N},\theta)
+
\sum_{k=0}^{N-1}
p_k^\top \bigl(x_{k+1}-F_\theta(x_k,u_k)\bigr).
```

Stationarity with respect to the states yields a terminal condition and a
backward recursion for the adjoints. If

```{math}
:label: eq:ch16-stage-loss

\ell_k(x_k)=\frac{1}{2}\norm{y_k-h(x_k)}_{R_k^{-1}}^2,
```

then the terminal state $x_N$ gives

```{math}
:label: eq:ch16-adjoint-terminal

p_{N-1}=-\nabla_{x_N}\ell_N(x_N),
```

and the interior states satisfy

```{math}
:label: eq:ch16-adjoint

p_{k-1}
=
\bigl(\partial_x F_\theta(x_k,u_k)\bigr)^\top p_k
-
\nabla_{x_k}\ell_k(x_k),
\qquad k=N-1,\dots,1.
```

The parameter gradient is then accumulated without differentiating each state
independently:

```{math}
:label: eq:ch16-parameter-gradient

\nabla_\theta \mathcal{L}
=
\nabla_\theta \mathcal{J}_{\mathrm{sc}}
-
\sum_{k=0}^{N-1}
\bigl(\partial_\theta F_\theta(x_k,u_k)\bigr)^\top p_k.
```

This is the adjoint perspective on long-horizon learning. It is the same basic
idea used in optimal control, backpropagation through time, and variational
data assimilation: compute sensitivities by a forward simulation and a backward
costate sweep rather than by perturbing every variable separately.

## Gauss-Newton Smoothing and Structured Linear Algebra

For objectives like {eq}`eq:ch16-weak-constraint`, Newton-type methods build a
local quadratic model around the current trajectory estimate. When second-order
terms from residual Hessians are dropped, the step is a Gauss-Newton solve.
This is especially natural when the objective is a sum of squared residuals.

The key computational fact is structural. Each residual involves only nearby
time indices:

- observation residuals depend on one state $x_k$,
- dynamics residuals couple only $x_k$ and $x_{k+1}$,
- background terms involve only $x_0$.

As a result, the normal equations for the Gauss-Newton step have a
block-tridiagonal or closely related sparse structure. The step computation is
therefore analogous to a smoothing problem in a linearized Gaussian model. In
fact, linearized weak-constraint smoothing and Gauss-Newton trajectory
optimization can be interpreted as the same algebra written in two different
languages.

This is one of the chapter's main conceptual bridges: extended Kalman
smoothing, ensemble-based smoothing, and nonlinear least-squares trajectory
optimization are not unrelated tool families. They often differ more in how
they approximate curvature and propagate uncertainty than in the underlying
objective they are trying to optimize.

## Relationship to Ensemble Kalman Smoothers

The ensemble Kalman smoother (EKS) is often introduced as a derivative-free or
Monte Carlo state-estimation method, but it can also be read as an approximate
optimizer for a trajectory objective. The ensemble covariance plays the role of
a low-rank metric or preconditioner, and the update resembles a regularized
Gauss-Newton step on the path space.

That interpretation matters for dynamics learning because it explains why EKS
methods can work well on large inverse problems even when exact Jacobians are
too expensive. It also clarifies their limits: if the ensemble does not span
the relevant directions, the induced Gauss-Newton approximation is poor no
matter how well tuned the implementation is.

## Why This View Helps Learning

Trajectory optimization is more than a new derivation style. It changes what we
pay attention to in learning problems.

First, it separates observation noise from model error through the distinction
between strong and weak constraints. That is often essential when a learned
model class is only an approximation.

Second, it exposes numerical conditioning. Multiple shooting, collocation,
adjoints, and sparse linear algebra are not optional implementation details for
long horizons; they are what make the learning problem computationally
tractable.

Third, it unifies methods that are sometimes taught separately. Smoothing,
variational data assimilation, nonlinear least squares, and some ensemble
methods all solve versions of the same latent-trajectory problem. The main
differences lie in parameterization, approximation, and uncertainty handling.

## Summary

Learning dynamics from partial and noisy observations can be posed as a
trajectory optimization problem over states and parameters. Strong-constraint
formulations enforce exact dynamics, weak-constraint formulations penalize
model error through variables such as $q_k$, and `4D-Var` provides the standard
data-assimilation language for these windowed optimization problems. Multiple
shooting and collocation improve the numerical formulation, adjoint variables
$p_k$ provide efficient long-horizon gradients, and Gauss-Newton methods expose
the block-structured linear algebra that links smoothing to nonlinear least
squares. This optimization viewpoint is therefore not separate from state
estimation; it is one of the clearest ways to understand modern data
assimilation and learning algorithms as members of the same family.
