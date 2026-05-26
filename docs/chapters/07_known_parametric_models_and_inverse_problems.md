# Known Parametric Models and Inverse Problems

In many dynamics-learning settings, the unknown object is the model form
itself: a discrete-time map, a vector field, or a sparse library
representation inferred from data. In this chapter, the structure of the
dynamics is assumed known. The learning task is instead to estimate an unknown
parameter vector inside that known structure.

That shift matters. When the model class is fixed, the main questions are no
longer "Which nonlinear terms should be present?" or "What architecture should
we choose?" The central questions become whether the data contain enough
information to determine the parameter values, how to compute gradients
efficiently through the dynamical system, and how to quantify uncertainty in
the resulting estimate. This is the classical inverse-problem viewpoint on
dynamics learning.

## Forward Models and the Parameter-Estimation Objective

We begin with a known parametric model,

```{math}
:label: eq:ch07-parametric-model

\dot{x}(t)=f(x(t),u(t);\theta),
\qquad
y(t)=h(x(t),u(t);\theta),
```

where $x(t)\in \mathbb{R}^n$ is the state, $u(t)\in \mathbb{R}^m$ is a known
input, $y(t)\in \mathbb{R}^p$ is the model output, and $\theta\in \mathbb{R}^q$
is the unknown parameter vector. The structure of $f$ and $h$ is assumed to be
known from physics, phenomenology, or prior modeling decisions. Only the
parameter values are unknown.

In practice, the data are available at sample times $t_k$ as numerical
measurements $\mathbf{y}_k$. For a candidate parameter vector $\theta$, the
forward problem is to simulate {eq}`eq:ch07-parametric-model` and compare the
predicted outputs with the data. A standard prediction-error objective is

```{math}
:label: eq:ch07-prediction-error

\widehat{\theta}
=
\arg\min_{\theta} \mathcal{J}(\theta),
\qquad
\mathcal{J}(\theta)
=
\frac{1}{2}\sum_{k=0}^{N}
r_k(\theta)^\top R^{-1} r_k(\theta),
```

with residual

```{math}
:label: eq:ch07-residual

r_k(\theta)=\mathbf{y}_k-h(x(t_k;\theta),u(t_k);\theta).
```

The matrix $R$ weights output errors by noise level or importance. When the
measurements are scalar and comparably scaled, this reduces to ordinary
nonlinear least squares. When outputs have different units or different noise
levels, the weighting in {eq}`eq:ch07-prediction-error` becomes essential.

This formulation already shows why inverse problems are harder than static
curve fitting. The map from $\theta$ to the residual is indirect: changing
$\theta$ changes the entire state trajectory through the differential equation,
and the changed trajectory then changes the predicted observations.

## Prediction-Error and Maximum-Likelihood Viewpoints

Equation {eq}`eq:ch07-prediction-error` is often introduced as a prediction
error method, but it also has a probabilistic interpretation. Suppose the
measurement model is

```{math}
:label: eq:ch07-measurement-model

\mathbf{y}_k=h(x(t_k;\theta),u(t_k);\theta)+\eta_k,
\qquad
\eta_k \sim \mathcal{N}(0,R),
```

with independent Gaussian noise. Then minimizing
{eq}`eq:ch07-prediction-error` is equivalent, up to additive constants, to
maximizing the likelihood of the observations under the model. In that case,

```{math}
:label: eq:ch07-mle

\widehat{\theta}_{\mathrm{MLE}}
=
\arg\max_{\theta} p(\mathcal{D}\mid \theta)
=
\arg\min_{\theta} \mathcal{J}(\theta).
```

This equivalence is useful because it links optimization, statistics, and
uncertainty analysis. The least-squares objective is not merely a convenient
loss; under a noise model it is a negative log-likelihood. If the measurement
noise is non-Gaussian, correlated, or heteroscedastic, the form of the optimal
objective changes accordingly.

For deterministic dynamics, prediction-error methods often absorb all
mismatch into output noise. For more realistic models with process noise or
unmodeled disturbances, the likelihood is more subtle because the hidden state
trajectory itself becomes uncertain. That state-estimation extension appears
later in the course, but the present chapter already provides its parameter
learning backbone.

## Sensitivity Equations for Gradient Computation

Gradient-based parameter estimation requires derivatives of the loss with
respect to $\theta$. A direct finite-difference approximation is possible, but
it scales poorly with parameter dimension and can be inaccurate when the
dynamics are stiff or the loss is ill-conditioned. The standard alternative is
to differentiate the state equation itself.

Define the parameter sensitivity matrix

```{math}
:label: eq:ch07-sensitivity-definition

S_\theta(t)=\frac{\partial x(t)}{\partial \theta}
\in \mathbb{R}^{n\times q}.
```

Differentiating {eq}`eq:ch07-parametric-model` with respect to $\theta$ yields
the sensitivity equation

```{math}
:label: eq:ch07-sensitivity-equation

\dot{S}_\theta(t)
=
f_x(x(t),u(t);\theta)\,S_\theta(t)
+
f_\theta(x(t),u(t);\theta),
```

where $f_x$ and $f_\theta$ denote Jacobians of the vector field with respect to
the state and parameters. If the initial condition is known and independent of
$\theta$, then $S_\theta(0)=0$. If the initial condition is also estimated,
then its derivative must be included as well.

Once $S_\theta(t_k)$ is available, each residual Jacobian follows from the
chain rule:

```{math}
:label: eq:ch07-residual-jacobian

\frac{\partial r_k}{\partial \theta}
=
-h_x(x(t_k),u(t_k);\theta)\,S_\theta(t_k)
- h_\theta(x(t_k),u(t_k);\theta).
```

This turns parameter learning into a coupled forward solve: integrate the state
equation and the sensitivity equation together, then assemble the gradient of
$\mathcal{J}(\theta)$. Sensitivity methods are natural when the number of
parameters is moderate and the number of outputs or observation times is not
too large.

## Adjoint Methods for Large Inverse Problems

When the parameter dimension is large, forward sensitivities become expensive
because {eq}`eq:ch07-sensitivity-equation` evolves an $n\times q$ matrix. An
adjoint method reverses that scaling. Instead of propagating derivatives of the
state with respect to every parameter, it propagates a costate that measures
how the loss depends on perturbations of the state.

For a continuous-time objective of the form

```{math}
:label: eq:ch07-continuous-objective

\mathcal{J}(\theta)
=
\Phi(x(T),\theta)
+
\int_0^T \ell(x(t),u(t);\theta)\,\dd t,
```

the adjoint variable $p(t)$ satisfies a terminal-value problem of the form

```{math}
:label: eq:ch07-adjoint

\dot{p}(t)
=
-f_x(x(t),u(t);\theta)^\top p(t)
- \ell_x(x(t),u(t);\theta),
\qquad
p(T)=\Phi_x(x(T),\theta).
```

The gradient can then be written without forming the full sensitivity matrix:

```{math}
:label: eq:ch07-adjoint-gradient

\ddf{\mathcal{J}}{\theta}
=
\Phi_\theta
+
\int_0^T
\left(
f_\theta(x(t),u(t);\theta)^\top p(t)
+
\ell_\theta(x(t),u(t);\theta)
\right)\,\dd t.
```

The exact adjoint equations depend on whether the observations appear as
continuous penalties or discrete sampling terms, but the idea is the same: one
forward integration produces the trajectory, one backward integration produces
the costate, and their combination yields the gradient. Adjoint methods are
especially important in large-scale inverse problems, PDE-constrained
optimization, and long-horizon parameter estimation.

## Gauss-Newton and Levenberg-Marquardt Updates

Because the residuals depend nonlinearly on $\theta$, minimizing
{eq}`eq:ch07-prediction-error` usually requires an iterative optimizer. For
least-squares objectives, the most common local methods are Gauss-Newton and
Levenberg-Marquardt.

Let $\mathbf{r}(\theta)$ collect all residuals and let $\mathbf{J}_\theta$ be
the stacked Jacobian of those residuals with respect to $\theta$. A Gauss-Newton
step solves the linearized normal equations

```{math}
:label: eq:ch07-gauss-newton

\left(
\mathbf{J}_\theta^\top W \mathbf{J}_\theta
\right)\delta \theta
=
-\mathbf{J}_\theta^\top W \mathbf{r},
```

where $W$ is a residual-weighting matrix built from $R^{-1}$. This update
approximates the Hessian of $\mathcal{J}$ by dropping second-order derivatives
of the residual map. The approximation is often accurate near a good fit and
is much cheaper than a full Newton method.

Levenberg-Marquardt adds damping to improve robustness:

```{math}
:label: eq:ch07-levenberg-marquardt

\left(
\mathbf{J}_\theta^\top W \mathbf{J}_\theta
+
\lambda_{\mathrm{reg}} I
\right)\delta \theta
=
-\mathbf{J}_\theta^\top W \mathbf{r}.
```

When $\lambda_{\mathrm{reg}}$ is large, the step behaves more like a cautious
gradient method; when it is small, the step approaches Gauss-Newton. In
practice, these methods are effective because the dominant curvature of the
objective is often captured by the residual Jacobian alone.

These optimization methods are only as good as the model-data geometry. If the
problem is poorly initialized, weakly identifiable, or strongly multimodal, a
local optimizer may converge slowly or settle in an incorrect basin. That is
why uncertainty and identifiability analysis are part of the modeling workflow,
not an optional postscript.

## Structural and Practical Identifiability

Inverse problems can fail even when the optimizer is implemented perfectly.
Sometimes the parameters are not uniquely determined by the input-output map of
the model. Other times they are unique in principle but only weakly informed by
the available data.

Structural identifiability is a property of the model class itself. It asks
whether two distinct parameter vectors can produce exactly the same ideal,
noise-free observations under the same experiment design. If the answer is yes,
then no amount of noise reduction or algorithmic tuning will recover a unique
parameter estimate.

Practical identifiability is a property of a particular dataset and experiment.
Even a structurally identifiable model may be practically unidentifiable if the
measurements are too noisy, the observation window is too short, or the input
does not excite the relevant parameter directions. In practice, broad flat
valleys in the objective function often signal this weakness.

The distinction is crucial for model criticism. Structural failure suggests a
reparameterization or a new experiment design. Practical failure suggests
better data, richer excitation, or more informative priors.

## Fisher Information, Profile Likelihood, and Bayesian Inference

Local uncertainty around a well-fit parameter estimate is often summarized by
the Fisher information matrix. In the weighted least-squares setting, a common
approximation is

```{math}
:label: eq:ch07-fisher

\mathcal{I}(\widehat{\theta})
\approx
\mathbf{J}_{\widehat{\theta}}^\top
W
\mathbf{J}_{\widehat{\theta}}.
```

If $\mathcal{I}(\widehat{\theta})$ is ill-conditioned or nearly singular, then
small changes in the data can cause large changes in the estimated parameters.
This is a local signature of weak practical identifiability.

Profile likelihood pushes beyond that local picture. To study one component
$\theta_i$, we hold it fixed and re-optimize the remaining parameters:

```{math}
:label: eq:ch07-profile-likelihood

\mathcal{P}_i(\alpha)
=
\min_{\theta:\,\theta_i=\alpha} \mathcal{J}(\theta).
```

Flat or asymmetric profiles reveal non-Gaussian uncertainty, parameter
confounding, and one-sided confidence structure that a quadratic approximation
around $\widehat{\theta}$ may miss.

A Bayesian formulation goes further by treating $\theta$ as uncertain from the
start. With prior density $p(\theta)$ and likelihood $p(\mathcal{D}\mid
\theta)$, the posterior is

```{math}
:label: eq:ch07-bayes

p(\theta\mid \mathcal{D})
\propto
p(\mathcal{D}\mid \theta)\,p(\theta).
```

This viewpoint is useful when prior physical knowledge matters, when parameter
constraints should be encoded directly, or when uncertainty propagation is part
of the downstream task. In simple settings, a Laplace approximation around the
maximum a posteriori estimate may be adequate. In harder settings, sampling or
variational approximations may be needed.

## What Makes a Parameter-Estimation Problem Succeed?

Known parametric models are often attractive because they are interpretable and
sample-efficient, but that does not make them easy. Several conditions still
govern success.

- The experiment must be informative enough to excite the parameters that need
  to be estimated.
- The observation model must distinguish the effects of those parameters on the
  measured outputs.
- Numerical solvers, sensitivities, and optimization tolerances must be
  accurate enough that computational error does not dominate the inverse
  problem.
- Model-form mismatch must be considered carefully, since a poor model can
  produce confident but physically misleading parameter estimates.

These points explain why inverse problems sit at the intersection of modeling,
numerics, statistics, and experiment design. The challenge is not only to fit
parameters, but to know whether the fitted values mean what we hope they mean.

## Summary

When the model structure is known, learning dynamics becomes an inverse problem
for the parameter vector $\theta$. Prediction-error minimization, maximum
likelihood, and Bayesian inference are different lenses on the same basic
question: which parameter values make the observed data most plausible under
the model? Sensitivity and adjoint methods make these problems computationally
tractable, while Gauss-Newton and Levenberg-Marquardt provide effective local
optimization tools. But the central conceptual issue is identifiability: a
parameter estimate is only as meaningful as the experiment's ability to
distinguish that parameter from its competitors.
