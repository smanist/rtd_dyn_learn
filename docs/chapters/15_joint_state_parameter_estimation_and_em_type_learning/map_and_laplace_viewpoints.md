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
