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

The reason this construction targets the marginal likelihood is visible from
the standard ELBO decomposition. Let
$q^{(i)}(x_{0:N})=p_{\theta^{(i)}}(x_{0:N}\mid y_{0:N})$. For any candidate
$\theta$,

```{math}
:label: eq:ch15-em-elbo-identity

\ell(\theta)
=
\log p_\theta(y_{0:N})
=
\mathbb{E}_{q^{(i)}}
\left[
\log p_\theta(x_{0:N},y_{0:N})
-\log q^{(i)}(x_{0:N})
\right]
+
\operatorname{KL}
\left(
q^{(i)}(x_{0:N})
\;\middle\|\;
p_\theta(x_{0:N}\mid y_{0:N})
\right).
```

The first term is a lower bound on $\ell(\theta)$ because the KL divergence is
nonnegative. At the current iterate $\theta^{(i)}$, the bound is tight because
$q^{(i)}$ is exactly the posterior under $\theta^{(i)}$. Since
$-\mathbb{E}_{q^{(i)}}[\log q^{(i)}]$ does not depend on the candidate
$\theta$, maximizing the lower bound is equivalent to maximizing
{eq}`eq:ch15-em-q-function`. This gives the monotonicity argument

```{math}
:label: eq:ch15-em-monotonicity

\ell(\theta)-\ell(\theta^{(i)})
=
\mathcal{Q}(\theta,\theta^{(i)})
-\mathcal{Q}(\theta^{(i)},\theta^{(i)})
+
\operatorname{KL}
\left(
q^{(i)}
\;\middle\|\;
p_\theta(x_{0:N}\mid y_{0:N})
\right)
\geq
\mathcal{Q}(\theta,\theta^{(i)})
-\mathcal{Q}(\theta^{(i)},\theta^{(i)}).
```

Thus an exact M-step that increases $\mathcal{Q}$ cannot decrease the
observed-data likelihood. This is the main distinction between exact EM and
more heuristic alternating schemes: exact EM optimizes a likelihood lower bound
that touches the likelihood at the current parameter.

Using the factorization in {eq}`eq:ch15-complete-data-density`, the expected
complete-data log-likelihood separates into prior, transition, and observation
terms:

```{math}
:label: eq:ch15-em-q-factorization

\begin{aligned}
\mathcal{Q}(\theta,\theta^{(i)})
&=
\mathbb{E}_{q^{(i)}}
\left[\log p_\theta(x_0)\right]
+
\sum_{k=0}^{N-1}
\mathbb{E}_{q^{(i)}}
\left[
\log p_\theta(x_{k+1}\mid x_k,u_k)
\right]\\
&\quad
+
\sum_{k=0}^{N}
\mathbb{E}_{q^{(i)}}
\left[
\log p_\theta(y_k\mid x_k)
\right].
\end{aligned}
```

This expression shows what the E-step must provide to the M-step. If the model
belongs to an exponential family, the parameter update depends on posterior
expectations of sufficient statistics rather than on individual hidden
trajectories.

For a linear-Gaussian example, suppose

```{math}
:label: eq:ch15-em-linear-gaussian-model

x_{k+1}=Ax_k+Bu_k+w_k,
\qquad
w_k\sim \mathcal{N}(0,Q),
```

and take $C$, $R$, and the initial distribution to be fixed for the moment.
Let

```{math}
:label: eq:ch15-em-augmented-regressor

z_k
=
\begin{bmatrix}
x_k\\
u_k
\end{bmatrix},
\qquad
G
=
\begin{bmatrix}
A & B
\end{bmatrix}.
```

Ignoring constants independent of $(G,Q)$, the transition part of
{eq}`eq:ch15-em-q-factorization` is

```{math}
:label: eq:ch15-em-transition-objective

\mathcal{Q}_{\mathrm{dyn}}(G,Q)
=
-\frac{N}{2}\log\det Q
-\frac{1}{2}
\sum_{k=0}^{N-1}
\mathbb{E}_{q^{(i)}}
\left[
\left(x_{k+1}-Gz_k\right)^\top
Q^{-1}
\left(x_{k+1}-Gz_k\right)
\right].
```

The E-step therefore needs smoothed first moments, second moments, and lag-one
moments:

```{math}
:label: eq:ch15-em-smoothed-moments

\begin{aligned}
\mathbb{E}[x_k\mid y]
&=\widehat{x}_{k|N},\\
\mathbb{E}[x_kx_k^\top\mid y]
&=P_{k|N}+\widehat{x}_{k|N}\widehat{x}_{k|N}^\top,\\
\mathbb{E}[x_{k+1}x_k^\top\mid y]
&=P_{k+1,k|N}
+\widehat{x}_{k+1|N}\widehat{x}_{k|N}^\top.
\end{aligned}
```

Here $P_{k+1,k|N}$ is the smoothed lag-one cross-covariance. Define the
accumulated sufficient statistics

```{math}
:label: eq:ch15-em-dynamics-sufficient-statistics

\begin{aligned}
S_{++}
&=
\sum_{k=0}^{N-1}
\mathbb{E}[x_{k+1}x_{k+1}^\top\mid y],\\
S_{+z}
&=
\sum_{k=0}^{N-1}
\mathbb{E}[x_{k+1}z_k^\top\mid y],\\
S_{zz}
&=
\sum_{k=0}^{N-1}
\mathbb{E}[z_kz_k^\top\mid y].
\end{aligned}
```

Differentiating {eq}`eq:ch15-em-transition-objective` with respect to $G$
gives the normal equations

```{math}
:label: eq:ch15-em-normal-equations

G S_{zz}=S_{+z}.
```

When $S_{zz}$ is nonsingular, the M-step update is the smoothed least-squares
fit

```{math}
:label: eq:ch15-em-dynamics-update

G^{(i+1)}
=
S_{+z}S_{zz}^{-1},
\qquad
\begin{bmatrix}
A^{(i+1)} & B^{(i+1)}
\end{bmatrix}
=
G^{(i+1)}.
```

With $G$ fixed at this optimizer, differentiating with respect to $Q$ gives the
residual covariance update

```{math}
:label: eq:ch15-em-process-covariance-update

Q^{(i+1)}
=
\frac{1}{N}
\sum_{k=0}^{N-1}
\mathbb{E}
\left[
\left(x_{k+1}-G^{(i+1)}z_k\right)
\left(x_{k+1}-G^{(i+1)}z_k\right)^\top
\;\middle|\; y
\right].
```

Equivalently,

```{math}
:label: eq:ch15-em-process-covariance-statistics

Q^{(i+1)}
=
\frac{1}{N}
\left(
S_{++}
-G^{(i+1)}S_{+z}^\top
-S_{+z}\left(G^{(i+1)}\right)^\top
+G^{(i+1)}S_{zz}\left(G^{(i+1)}\right)^\top
\right).
```

The observation model has the same structure if it is also linear and unknown.
For

```{math}
:label: eq:ch15-em-linear-observation-model

y_k=Cx_k+Du_k+v_k,
\qquad
v_k\sim \mathcal{N}(0,R),
```

define $\widetilde{z}_k=[x_k^\top,u_k^\top]^\top$ and
$H=[C\;D]$. Then the M-step uses

```{math}
:label: eq:ch15-em-observation-update

H^{(i+1)}
=
\left(
\sum_{k=0}^{N}
y_k\,\mathbb{E}[\widetilde{z}_k^\top\mid y]
\right)
\left(
\sum_{k=0}^{N}
\mathbb{E}[\widetilde{z}_k\widetilde{z}_k^\top\mid y]
\right)^{-1},
```

and

```{math}
:label: eq:ch15-em-observation-covariance-update

R^{(i+1)}
=
\frac{1}{N+1}
\sum_{k=0}^{N}
\mathbb{E}
\left[
\left(y_k-H^{(i+1)}\widetilde{z}_k\right)
\left(y_k-H^{(i+1)}\widetilde{z}_k\right)^\top
\;\middle|\; y
\right].
```

If the initial distribution is learned, its update follows from the same
moment-matching principle:
$m_0^{(i+1)}=\widehat{x}_{0|N}$ and
$P_0^{(i+1)}=P_{0|N}$.

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
