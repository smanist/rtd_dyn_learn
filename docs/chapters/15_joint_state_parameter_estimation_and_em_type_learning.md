# Joint State-Parameter Estimation and EM-Type Learning

When the state trajectory is hidden, system identification is no longer only a
parameter-fitting problem. The unknown dynamics parameter $\theta$ and the
unknown trajectory $x_{0:N}$ must be inferred together from noisy, partial
observations. That joint viewpoint is the natural continuation of the previous
chapters on filtering and smoothing: once trajectories are treated as latent
variables, learning dynamics becomes a latent-variable inference problem rather
than a direct regression on observed data.

This shift matters for both statistics and algorithms. Statistically, the
relevant objective is no longer a simple prediction loss on observed states,
because the observations may not reveal the latent trajectory uniquely.
Algorithmically, many identification methods can be organized around repeated
state estimation and parameter updates. Expectation-maximization (EM),
augmented-state filtering, dual estimation, Laplace approximations, and
variational inference all fit into that pattern, even though they make
different approximations.

## State-Space Formulation with Unknown Parameters

We begin from a parameterized state-space model,

```{math}
:label: eq:ch15-state-space-model

x_{k+1}=F(x_k,u_k;\theta)+w_k,
\qquad
y_k=h(x_k)+v_k,
```

where $x_k\in \mathbb{R}^n$ is the latent state, $u_k\in \mathbb{R}^m$ is the
input, and $y_k\in \mathbb{R}^p$ is the observation. The process noise $w_k$
and measurement noise $v_k$ represent model error and sensor uncertainty,
respectively. In this chapter, the unknown quantity is not only $\theta$ but
the full latent trajectory

```{math}
:label: eq:ch15-joint-unknowns

\left(x_{0:N},\theta\right).
```

Under this model, the complete-data density factors as

```{math}
:label: eq:ch15-complete-data-density

p_\theta(x_{0:N},y_{0:N})
=
p_\theta(x_0)\prod_{k=0}^{N-1}p_\theta(x_{k+1}\mid x_k,u_k)
\prod_{k=0}^{N}p_\theta(y_k\mid x_k).
```

The quantity we would like to maximize for parameter learning is the marginal
log-likelihood

```{math}
:label: eq:ch15-log-likelihood

\ell(\theta)
=
\log p_\theta(y_{0:N})
=
\log \int p_\theta(x_{0:N},y_{0:N})\,\dd x_{0:N}.
```

Equation {eq}`eq:ch15-log-likelihood` is conceptually simple and
computationally hard. The integral over all hidden trajectories is usually
intractable except in special linear-Gaussian cases. The basic strategies in
this chapter differ mainly in how they approximate the posterior over
$x_{0:N}$ and how they update $\theta$ once that posterior information is
available.

## Joint Inference as a Latent-Variable Problem

The latent-variable interpretation is the central conceptual step. Instead of
treating the hidden state as a nuisance, we recognize that identification from
partial observations requires a posterior distribution over trajectories:

```{math}
:label: eq:ch15-joint-posterior

p_\theta(x_{0:N}\mid y_{0:N}).
```

If this smoothing distribution is concentrated, then parameter updates can rely
on a nearly known trajectory. If it is broad or multimodal, parameter learning
inherits that ambiguity. In other words, poor observability and weak
identifiability appear as posterior uncertainty over latent trajectories and
over $\theta$ itself.

This perspective also clarifies the relationship between chapters. Pure system
identification with fully observed state can often regress directly on
$\mathbf{x}_k$ or $\dot{\mathbf{x}}(t)$. Joint state-parameter estimation
cannot. It must alternate between explaining the observations with a hidden
trajectory and explaining that trajectory with a parameterized dynamics model.

## Augmented-State and Dual Estimation Viewpoints

One direct strategy is to treat the parameter as part of the state:

```{math}
:label: eq:ch15-augmented-state

\bar{x}_k
=
\begin{bmatrix}
x_k\\
\theta_k
\end{bmatrix},
\qquad
\theta_{k+1}=\theta_k+\xi_k.
```

The artificial evolution $\theta_{k+1}=\theta_k+\xi_k$ turns a static
parameter into a slowly varying state component. Standard filtering or
smoothing methods can then be applied to $\bar{x}_k$ rather than to $x_k$
alone. This augmented-state construction is attractive because it reuses state
estimation machinery, but it introduces modeling choices that matter:

- The covariance assigned to $\xi_k$ controls how aggressively the filter
  adapts parameters.
- If $\xi_k$ is too small, the parameter estimate can freeze before enough
  information accumulates.
- If $\xi_k$ is too large, the method may explain model mismatch by allowing
  implausible parameter drift.

An alternative is dual estimation: update the state estimate using a current
parameter guess, then update the parameter estimate using state or innovation
statistics produced by the filter or smoother. Dual methods separate the two
tasks conceptually, but the coupling remains strong because poor state
estimates bias parameter updates and poor parameter values bias state
estimation.

Augmented-state and dual schemes are often practical online approximations.
They are useful when data arrive sequentially and a fully batch method is too
expensive. Their main limitation is that they can understate posterior
coupling between $x_{0:N}$ and $\theta$.

## EM as Repeated Smoothing and Parameter Update

Expectation-maximization organizes the batch problem more cleanly. Starting
from a current iterate $\theta^{(i)}$, define the surrogate objective

```{math}
:label: eq:ch15-em-q-function

\mathcal{Q}(\theta,\theta^{(i)})
=
\mathbb{E}_{p_{\theta^{(i)}}(x_{0:N}\mid y_{0:N})}
\left[
\log p_\theta(x_{0:N},y_{0:N})
\right].
```

The EM iteration has two conceptual steps:

1. E-step: compute or approximate the smoothing distribution
   $p_{\theta^{(i)}}(x_{0:N}\mid y_{0:N})$.
2. M-step: update the parameter by maximizing
   $\mathcal{Q}(\theta,\theta^{(i)})$ with respect to $\theta$.

The power of EM is organizational. It converts direct optimization of the
marginal likelihood {eq}`eq:ch15-log-likelihood` into repeated inference on the
latent trajectory plus an easier parameter subproblem. In a linear-Gaussian
state-space model, the E-step is a Rauch-Tung-Striebel smoother and the
M-step often reduces to closed-form updates for system matrices or noise
covariances using smoothed moments such as $\mathbb{E}[x_kx_k^\top\mid y]$ and
$\mathbb{E}[x_{k+1}x_k^\top\mid y]$.

Outside the linear-Gaussian setting, EM remains useful but the E-step becomes
approximate. Then the phrase "EM-type learning" is more accurate than exact EM:
the algorithm preserves the same structure while replacing one or both steps by
approximations.

## Approximate E-Steps for Nonlinear Models

Nonlinear dynamics rarely admit an exact smoothing distribution, so practical
EM depends on approximate state estimators.

### Iterated EKF and EKS

Local linearization methods approximate the nonlinear model around a nominal
trajectory. An iterated extended Kalman smoother repeatedly refines the
trajectory and covariance by linearizing around the current estimate. In EM
language, that smoother provides a Gaussian approximation to the E-step.

This works best when the posterior is locally unimodal and the nonlinearities
are not too severe over the posterior support. When the observations are sparse
or the model is strongly nonlinear, the local Gaussian approximation can become
misleading.

### Ensemble Smoothers

Ensemble Kalman methods propagate an ensemble of trajectories and approximate
the posterior mean and covariance empirically. For joint estimation, ensemble
members can carry state and parameter components together, or the ensemble can
be used to construct approximate sufficient statistics for the M-step.

Ensemble smoothers are attractive in higher-dimensional settings because they
avoid explicit Jacobians and can exploit covariance localization or inflation.
Their weakness is the same Gaussian closure already seen in ensemble filtering:
the posterior is represented mainly through first and second moments.

### Particle and Sampling-Based Approximations

When the posterior is highly non-Gaussian, Monte Carlo approximations are more
faithful in principle. Particle EM and related Monte Carlo EM methods replace
the exact smoothing expectation in {eq}`eq:ch15-em-q-function` by an empirical
average over weighted trajectories. The price is computational cost and weight
degeneracy, especially over long horizons.

The right approximation therefore depends on the regime. Mildly nonlinear,
moderately observed systems may support EKF- or ensemble-based EM. Strongly
nonlinear, multimodal problems may require particle or optimization-based
approximations even if they are more expensive.

## MAP and Laplace Viewpoints

EM is not the only way to exploit latent-state structure. A closely related
route is maximum a posteriori (MAP) estimation of states and parameters:

```{math}
:label: eq:ch15-map-objective

\left(\widehat{x}_{0:N},\widehat{\theta}\right)
\in
\arg\max_{x_{0:N},\theta}
\log p(x_{0:N},\theta\mid y_{0:N}).
```

Equivalently, one may minimize a negative log-posterior composed of an
observation mismatch term, a dynamics-consistency term, and prior penalties.
This viewpoint connects joint estimation to trajectory optimization and weak-
constraint smoothing.

Once a MAP point is found, a Laplace approximation builds a local Gaussian
posterior by expanding the negative log-posterior to second order around
$\left(\widehat{x}_{0:N},\widehat{\theta}\right)$. The result is a Gaussian
approximation whose covariance is related to the inverse Hessian of the
objective. Laplace methods are attractive when one wants uncertainty estimates
without running a full sampling method, but they inherit the locality of the
quadratic expansion. A curved or multimodal posterior cannot be captured well
by a single Gaussian around one mode.

## Variational Inference and ELBO Formulations

Variational inference replaces exact posterior computation by optimization over
an approximating family. For the maximum-likelihood or MAP formulations above,
the parameter $\theta$ remains an optimization variable, while the latent
trajectory is approximated by a tractable distribution $q(x_{0:N})$. One then
maximizes the evidence lower bound

```{math}
:label: eq:ch15-elbo

\mathcal{L}_{\mathrm{ELBO}}(q,\theta)
=
\mathbb{E}_{q}
\left[
\log p_\theta(x_{0:N},y_{0:N})
-\log q(x_{0:N})
\right].
```

The ELBO trades fidelity to the joint model against complexity of the
approximate posterior. EM is a special case of this viewpoint: at iteration
$k$, the E-step sets
$q(x_{0:N}) = p_{\theta^{(k)}}(x_{0:N}\mid y_{0:N})$, and the M-step updates
$\theta$ by maximizing the resulting expected complete-data log-likelihood.
More general variational methods allow structured approximations, smoothing
families with restricted covariance, or amortized inference networks for
approximating $q(x_{0:N})$.

That flexibility is useful for nonlinear and large-scale models, but it comes
with approximation bias. The chosen family for $q$ may artificially suppress
dependence induced by the latent trajectory, which can make posterior or
profile-likelihood uncertainty appear more certain than it really is.

## Identifiability Under Partial Observations

Joint estimation fails most often because the data do not separate hidden-state
uncertainty from parameter uncertainty. Several pathologies are common:

- Different parameter values can produce nearly identical outputs after the
  hidden state is reoptimized.
- A poor initial state can mimic the effect of a parameter perturbation over
  short horizons.
- Weak excitation can leave some components of $\theta$ effectively
  unobserved.
- Process noise and parameter mismatch can compensate for each other, making
  model error hard to localize.

These are not merely numerical annoyances. They indicate that the likelihood or
posterior has flat directions, ridges, or multiple modes. In practice, one
should inspect sensitivity to priors, initialization, smoothing assumptions,
and input design rather than trusting a single fitted parameter vector.

## What This Chapter Adds to System Identification

The main lesson of joint state-parameter estimation is that learning dynamics
from partial observations is fundamentally an inference problem over latent
trajectories. EM makes that explicit by alternating smoothing and parameter
updates. Augmented-state and dual estimators provide online approximations.
MAP and Laplace methods recast the same problem as optimization plus local
uncertainty quantification. Variational inference generalizes the idea to
broader approximate posterior families.

All of these methods live or die by the same structural questions: are the
hidden trajectories sufficiently observable, are the parameters sufficiently
identifiable, and do the chosen approximations represent the posterior
coupling between states and parameters well enough for the intended use? Those
questions will reappear in the next chapter when the same learning problem is
viewed through trajectory optimization and data assimilation.
