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

## Minimal EM+EKS Joint State-Parameter Example

To keep the Chapter 15 examples comparable, use the same partially observed
nonlinear oscillator as the other worked examples:

```{math}
:label: eq:ch15-em-shared-model

x_k
=
\begin{bmatrix}
q_k\\
v_k
\end{bmatrix},
\qquad
x_{k+1}
=
\begin{bmatrix}
q_k + \Delta t\, v_k\\
v_k + \Delta t\left(-\theta \sin q_k - c v_k\right)
\end{bmatrix}
+ w_k,
\qquad
y_k = q_k + \eta_k,
```

with known damping $c=0.15$, step size $\Delta t=0.15$, and one unknown scalar
parameter $\theta$. Only the position $q_k$ is measured, so the velocity is
latent. For this minimal benchmark, fix

- $\theta_{\mathrm{true}} = 1.35$,
- $Q=\operatorname{diag}(0.015^2,\,0.08^2)$,
- $R=0.12^2$,
- $x_0 \sim \mathcal{N}([0.9,\;0.0]^\top,\operatorname{diag}(0.25^2,\,0.25^2))$,
- $\theta \sim \mathcal{N}(0.9,\,0.35^2)$.

The goal is to estimate both the hidden trajectory and $\theta$ from one noisy
position record. In this simplified EM-type example, the two steps are:

1. E-step: with $\theta^{(i)}$ fixed, run an extended Kalman filter followed by
   a Rauch-Tung-Striebel backward pass to get approximate smoothed means
   $\widehat{x}_{k|N}^{(i)}$ and covariances $P_{k|N}^{(i)}$.
2. M-step: freeze the nonlinear factor $\sin q_k$ at the smoothed mean
   $\widehat{q}_{k|N}^{(i)}$ and update $\theta$ by a scalar weighted least-
   squares solve.

That second step is why this is an EM-type method rather than exact EM: the
E-step is already approximate because it uses local linearization, and the
M-step replaces $\mathbb{E}[\sin q_k\mid y]$ by
$\sin(\widehat{q}_{k|N}^{(i)})$ to keep the update readable.

## Minimal Implementation

The code below simulates one trajectory from
{eq}`eq:ch15-em-shared-model`, runs eight EM iterations, and records both the
parameter trace and the EKF innovation log-likelihood:

```python
import numpy as np

rng = np.random.default_rng(7)

dt, c, N = 0.15, 0.15, 40
theta_true = 1.35
Q = np.diag([0.015**2, 0.08**2])
R = 0.12**2
H = np.array([[1.0, 0.0]])

m0 = np.array([0.9, 0.0])
P0 = np.diag([0.25**2, 0.25**2])
theta_prior_mean, theta_prior_std = 0.9, 0.35


def F(x, theta):
    q, v = x
    return np.array([
        q + dt * v,
        v + dt * (-theta * np.sin(q) - c * v),
    ])


def dF_dx(x, theta):
    q, _ = x
    return np.array([
        [1.0, dt],
        [-dt * theta * np.cos(q), 1.0 - dt * c],
    ])


x_true = np.zeros((N + 1, 2))
x_true[0] = np.array([1.1, -0.15])
for k in range(N):
    x_true[k + 1] = F(x_true[k], theta_true) + rng.multivariate_normal(
        np.zeros(2), Q
    )
y = x_true[:, 0] + np.sqrt(R) * rng.normal(size=N + 1)


def ekf_smoother(y, theta):
    x_filter = np.zeros((N + 1, 2))
    P_filter = np.zeros((N + 1, 2, 2))
    x_pred = np.zeros((N + 1, 2))
    P_pred = np.zeros((N + 1, 2, 2))
    innovation_loglik = 0.0

    x_pred[0] = m0
    P_pred[0] = P0
    innovation = y[0] - x_pred[0, 0]
    S = H @ P_pred[0] @ H.T + np.array([[R]])
    K = P_pred[0] @ H.T / S[0, 0]
    x_filter[0] = x_pred[0] + K[:, 0] * innovation
    P_filter[0] = (np.eye(2) - K @ H) @ P_pred[0]
    innovation_loglik += -0.5 * (
        np.log(2.0 * np.pi * S[0, 0]) + innovation**2 / S[0, 0]
    )

    A_history = []
    for k in range(N):
        A_k = dF_dx(x_filter[k], theta)
        A_history.append(A_k)
        x_pred[k + 1] = F(x_filter[k], theta)
        P_pred[k + 1] = A_k @ P_filter[k] @ A_k.T + Q

        innovation = y[k + 1] - x_pred[k + 1, 0]
        S = H @ P_pred[k + 1] @ H.T + np.array([[R]])
        K = P_pred[k + 1] @ H.T / S[0, 0]
        x_filter[k + 1] = x_pred[k + 1] + K[:, 0] * innovation
        P_filter[k + 1] = (np.eye(2) - K @ H) @ P_pred[k + 1]
        innovation_loglik += -0.5 * (
            np.log(2.0 * np.pi * S[0, 0]) + innovation**2 / S[0, 0]
        )

    x_smooth = x_filter.copy()
    P_smooth = P_filter.copy()
    for k in range(N - 1, -1, -1):
        C_k = P_filter[k] @ A_history[k].T @ np.linalg.inv(P_pred[k + 1])
        x_smooth[k] = x_filter[k] + C_k @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = (
            P_filter[k]
            + C_k @ (P_smooth[k + 1] - P_pred[k + 1]) @ C_k.T
        )

    return x_smooth, P_smooth, innovation_loglik


def m_step_theta(x_smooth):
    sigma_v2 = Q[1, 1]
    numerator = theta_prior_mean / theta_prior_std**2
    denominator = 1.0 / theta_prior_std**2

    for k in range(N):
        a_k = x_smooth[k + 1, 1] - (1.0 - dt * c) * x_smooth[k, 1]
        b_k = -dt * np.sin(x_smooth[k, 0])
        numerator += b_k * a_k / sigma_v2
        denominator += b_k * b_k / sigma_v2

    return numerator / denominator


theta = theta_prior_mean
theta_history = []
loglik_history = []

for _ in range(8):
    x_smooth, P_smooth, innovation_loglik = ekf_smoother(y, theta)
    theta_history.append(theta)
    loglik_history.append(innovation_loglik)
    theta = m_step_theta(x_smooth)

x_smooth, P_smooth, _ = ekf_smoother(y, theta)
```

On this benchmark, the parameter estimate moves from the prior mean $0.9$ to
$\widehat{\theta}\approx 1.290$ while the innovation log-likelihood rises from
about $15.28$ to $21.07$. The final smoothed position has root-mean-square
error about $0.024$, compared with about $0.097$ for the raw observations.

```{figure} em_eks_convergence.svg
:alt: EM plus extended Kalman smoothing convergence history for the Chapter 15 benchmark.
:width: 98%

Left: the scalar M-step pushes $\theta$ quickly toward a stable value. Right:
the EKF innovation log-likelihood from the E-step improves sharply in the first
few EM iterations and then plateaus.
```

```{figure} em_eks_reconstruction.svg
:alt: Reconstructed position and latent velocity for the Chapter 15 EM plus extended Kalman smoothing example.
:width: 98%

Top: noisy position observations, the hidden true position, and the smoothed
position reconstruction. Bottom: the hidden true velocity and the smoothed
velocity estimate recovered from position-only data.
```

The reconstruction is stronger than the parameter estimate. That is typical in
partially observed nonlinear problems: many nearby $\theta$ values can support
similar smoothed trajectories over a short window, so the state estimate looks
good before the parameter estimate becomes fully sharp.

:::{foldbox} Simplified M-step for the scalar parameter

Only the velocity equation depends on $\theta$, so with
$\widehat{x}_{k|N}^{(i)}=[\widehat{q}_{k|N}^{(i)},\widehat{v}_{k|N}^{(i)}]^\top$
the approximate transition residual is

```{math}
:label: eq:ch15-em-scalar-residual

r_k(\theta)
=
\widehat{v}_{k+1|N}^{(i)}
- \left(1-\Delta t\,c\right)\widehat{v}_{k|N}^{(i)}
+ \Delta t\,\theta \sin\!\left(\widehat{q}_{k|N}^{(i)}\right).
```

Ignoring constants independent of $\theta$, the approximate M-step reduces to
the weighted least-squares cost

```{math}
:label: eq:ch15-em-scalar-objective

\widetilde{\mathcal{J}}(\theta)
=
\frac{1}{2}\frac{(\theta-\mu_\theta)^2}{\sigma_\theta^2}
+
\frac{1}{2}
\sum_{k=0}^{N-1}
\frac{r_k(\theta)^2}{Q_{vv}},
```

where $Q_{vv}=0.08^2$ is the process-noise variance for the velocity
component. Write

```{math}
:label: eq:ch15-em-scalar-regression

a_k
=
\widehat{v}_{k+1|N}^{(i)}
- \left(1-\Delta t\,c\right)\widehat{v}_{k|N}^{(i)},
\qquad
b_k = -\Delta t \sin\!\left(\widehat{q}_{k|N}^{(i)}\right),
```

so that $r_k(\theta)=a_k-b_k\theta$. Minimizing
{eq}`eq:ch15-em-scalar-objective` by setting its derivative to zero gives

```{math}
:label: eq:ch15-em-scalar-update

\theta^{(i+1)}
=
\frac{
\mu_\theta \sigma_\theta^{-2}
+
\sum_{k=0}^{N-1} b_k a_k / Q_{vv}
}{
\sigma_\theta^{-2}
+
\sum_{k=0}^{N-1} b_k^2 / Q_{vv}
}.
```

This is just one-dimensional weighted least squares with a Gaussian prior. The
full EM M-step still maximizes the expected complete-data log posterior; in
this simplified derivation that is equivalent, up to $\theta$-independent
constants and an overall minus sign, to minimizing
$\widetilde{\mathcal{J}}(\theta)$. A more exact nonlinear EM update would replace
$\sin(\widehat{q}_{k|N}^{(i)})$ by posterior expectations such as
$\mathbb{E}[\sin q_k \mid y]$ and
$\mathbb{E}[\sin^2 q_k \mid y]$, but the simplified formula is much easier to
teach and already shows the E-step/M-step division clearly.

:::

## Behavior, Strengths, and Limitations

- Behavior: most of the improvement happens in the first two or three EM
  iterations, then both the parameter trace and the innovation log-likelihood
  flatten.
- Strength: the smoother uses temporal coupling to recover a plausible hidden
  velocity trajectory even though only position is observed.
- Limitation: the final parameter estimate stays below the true
  $\theta_{\mathrm{true}}=1.35$ because the method relies on local
  linearization and on the mean-field-style replacement
  $\mathbb{E}[\sin q_k\mid y]\approx \sin(\widehat{q}_{k|N})$.

This is exactly the teaching point of EM+EKS in Chapter 15: the workflow is
simple and modular, but the quality of the result depends on how faithfully the
approximate smoother captures the latent posterior.
