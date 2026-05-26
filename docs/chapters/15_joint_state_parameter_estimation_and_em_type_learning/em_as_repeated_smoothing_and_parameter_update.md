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
