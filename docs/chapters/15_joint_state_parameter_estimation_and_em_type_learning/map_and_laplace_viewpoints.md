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
an approximating family $q(x_{0:N},\theta)$. One then maximizes the evidence
lower bound

```{math}
:label: eq:ch15-elbo

\mathcal{L}_{\mathrm{ELBO}}(q)
=
\mathbb{E}_{q}
\left[
\log p(x_{0:N},\theta,y_{0:N})
-\log q(x_{0:N},\theta)
\right].
```

The ELBO trades fidelity to the joint model against complexity of the
approximate posterior. EM is a special case of this viewpoint: at iteration
$k$, the E-step sets
$q(x_{0:N},\theta)=p_{\theta^{(k)}}(x_{0:N}\mid y_{0:N})
\delta(\theta-\theta^{(k)})$, and the M-step updates $\theta$ by maximizing
the resulting expected complete-data log-likelihood. More general variational
methods allow restricted Gaussian families, structured mean-field
factorizations, or amortized inference networks.

That flexibility is useful for nonlinear and large-scale models, but it comes
with approximation bias. The chosen family for $q$ may artificially suppress
dependence induced by the latent trajectory, which can make posterior or
profile-likelihood uncertainty appear more certain than it really is.

## Minimal Variational Joint State-Parameter Example

For a concrete benchmark, use the partially observed nonlinear oscillator:

```{math}
:label: eq:ch15-variational-shared-model

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

with the fixed benchmark choices

- $\Delta t = 0.15$ and $c = 0.15$,
- $N = 40$ transitions and $\theta_{\mathrm{true}} = 1.35$,
- $Q=\operatorname{diag}(0.015^2,\,0.08^2)$ and $R=0.12^2$,
- $x_0 \sim \mathcal{N}([0.9,\;0.0]^\top,\operatorname{diag}(0.25^2,\,0.25^2))$,
- $\theta \sim \mathcal{N}(0.9,\,0.35^2)$.

Only the position $q_k$ is observed, so the latent velocity and the restoring
force parameter must be inferred jointly.

## Chosen Variational Family

For a small instructional example, a useful compromise is

```{math}
:label: eq:ch15-variational-family

q(x_{0:N},\theta)
=
\delta(x_{0:N}-m_{0:N})\;
\mathcal{N}(\theta;\mu_\theta,\sigma_\theta^2).
```

This approximation is deliberately simple:

- the state posterior is collapsed to one representative trajectory
  $m_{0:N}$,
- the parameter posterior keeps one Gaussian factor
  $q(\theta)=\mathcal{N}(\mu_\theta,\sigma_\theta^2)$.

The approximation is strong, but it is easy to teach and it makes the key idea
explicit: unlike MAP plus Laplace, the parameter uncertainty is part of the
optimization variables during the fit rather than a curvature calculation added
afterward.

Because the state factor is a Dirac mass, this family is not a proper density
on $x_{0:N}$, so the entropy term in the ELBO is singular. The finite objective
used below should therefore not be read as a literal ELBO for
{eq}`eq:ch15-variational-family`. Instead, it is a restricted free-energy
approximation obtained by taking a vanishing-state-covariance limit of a proper
Gaussian trajectory factor and then optimizing only the remaining finite terms.
The expected dynamics term keeps the usual squared residual at the current mean
trajectory, plus an additional penalty proportional to
$\sigma_\theta^2 \sin^2 q_k$. That extra term is the price of carrying
uncertainty in $\theta$ while the trajectory is held to a point estimate.

## Minimal Coordinate-Ascent Implementation

Because this oscillator is affine in $\theta$ once $q_k$ is fixed, the
Gaussian parameter factor has a closed-form update. The trajectory update is
then a small Gauss-Newton solve for $m_{0:N}$. The code below alternates those
two steps:

```python
import numpy as np

rng = np.random.default_rng(7)

dt, c, N = 0.15, 0.15, 40
theta_true = 1.35
Q_std = np.array([0.015, 0.08])
sigma_y = 0.12


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


def dF_dtheta(x):
    q, _ = x
    return np.array([0.0, -dt * np.sin(q)])


def ddF_dtheta_dx(x):
    q, _ = x
    return np.array([
        [0.0, 0.0],
        [-dt * np.cos(q), 0.0],
    ])


x_true = np.zeros((N + 1, 2))
x_true[0] = np.array([1.1, -0.15])
for k in range(N):
    x_true[k + 1] = F(x_true[k], theta_true) + Q_std * rng.normal(size=2)
y = x_true[:, 0] + sigma_y * rng.normal(size=N + 1)

m0 = np.array([0.9, 0.0])
P0_sqrt_inv = np.diag([1.0 / 0.25, 1.0 / 0.25])
mu0, sigma0 = 0.9, 0.35
prior_precision = 1.0 / sigma0**2
Q_inv = np.diag(1.0 / Q_std**2)
Q_sqrt_inv = np.diag(1.0 / Q_std)


def update_theta(x):
    precision = prior_precision
    linear_term = mu0 * prior_precision
    for k in range(N):
        qk, vk = x[k]
        a_k = x[k + 1] - np.array([
            qk + dt * vk,
            vk - dt * c * vk,
        ])
        d_k = np.array([0.0, -dt * np.sin(qk)])
        precision += d_k @ Q_inv @ d_k
        linear_term += d_k @ Q_inv @ a_k

    sigma2 = 1.0 / precision
    mu = sigma2 * linear_term
    return mu, sigma2


def residual_and_jacobian(z, mu_theta, sigma2_theta):
    x = z.reshape(N + 1, 2)
    rows = 2 + 2 * N + 2 * N + (N + 1)
    r = np.zeros(rows)
    J = np.zeros((rows, z.size))
    row = 0

    r[row : row + 2] = P0_sqrt_inv @ (x[0] - m0)
    J[row : row + 2, 0:2] = P0_sqrt_inv
    row += 2

    root_sigma = np.sqrt(max(sigma2_theta, 1e-12))
    for k in range(N):
        dyn = Q_sqrt_inv @ (x[k + 1] - F(x[k], mu_theta))
        J[row : row + 2, 2 * k : 2 * k + 2] = -Q_sqrt_inv @ dF_dx(x[k], mu_theta)
        J[row : row + 2, 2 * (k + 1) : 2 * (k + 1) + 2] = Q_sqrt_inv
        r[row : row + 2] = dyn
        row += 2

    for k in range(N):
        # Expected dynamics cost adds a variance residual from q(theta).
        extra = root_sigma * (Q_sqrt_inv @ dF_dtheta(x[k]))
        J[row : row + 2, 2 * k : 2 * k + 2] = root_sigma * (
            Q_sqrt_inv @ ddF_dtheta_dx(x[k])
        )
        r[row : row + 2] = extra
        row += 2

    for k in range(N + 1):
        r[row] = (y[k] - x[k, 0]) / sigma_y
        J[row, 2 * k] = -1.0 / sigma_y
        row += 1

    return r, J


def free_energy(x, mu_theta, sigma2_theta):
    value = 0.5 * np.sum((P0_sqrt_inv @ (x[0] - m0)) ** 2)
    value += 0.5 * (((mu_theta - mu0) ** 2 + sigma2_theta) / sigma0**2 - np.log(sigma2_theta))
    for k in range(N):
        diff = x[k + 1] - F(x[k], mu_theta)
        value += 0.5 * diff @ Q_inv @ diff
        value += 0.5 * sigma2_theta * (dF_dtheta(x[k]) @ Q_inv @ dF_dtheta(x[k]))
    value += 0.5 * np.sum((y - x[:, 0]) ** 2 / sigma_y**2)
    return value


def gauss_newton_states(x0, mu_theta, sigma2_theta, damping=1e-2, max_iter=5):
    z = x0.reshape(-1).copy()
    for _ in range(max_iter):
        r, J = residual_and_jacobian(z, mu_theta, sigma2_theta)
        objective = 0.5 * (r @ r)
        H = J.T @ J
        g = J.T @ r
        local_damping = damping

        while True:
            step = np.linalg.solve(H + local_damping * np.eye(z.size), -g)
            candidate = z + step
            r_candidate, _ = residual_and_jacobian(candidate, mu_theta, sigma2_theta)
            candidate_objective = 0.5 * (r_candidate @ r_candidate)
            if candidate_objective < objective:
                z = candidate
                damping = max(local_damping / 3.0, 1e-6)
                break
            local_damping *= 10.0

        if np.linalg.norm(step) < 1e-6 * (1.0 + np.linalg.norm(z)):
            break

    return z.reshape(N + 1, 2)


x = np.zeros((N + 1, 2))
x[:, 0] = y
x[:, 1] = 0.0
free_energy_history = []

for _ in range(8):
    mu_theta, sigma2_theta = update_theta(x)
    x = gauss_newton_states(x, mu_theta, sigma2_theta)
    free_energy_history.append(free_energy(x, mu_theta, sigma2_theta))

theta_std = np.sqrt(sigma2_theta)
```

On this benchmark, the restricted free energy falls from about $66.3$ to
$17.7$ over eight outer iterations. The final parameter factor is

```{math}
:label: eq:ch15-variational-final-parameter

q(\theta)
=
\mathcal{N}(1.366,\;0.106^2),
```

and the reconstructed position trajectory has root-mean-square error about
$0.027$ relative to the hidden truth, compared with about $0.097$ for the raw
observations.

```{figure} variational_free_energy_history.svg
:alt: Restricted free-energy history for the Chapter 15 variational example.
:width: 95%

Restricted free-energy history for the point-mass-trajectory plus Gaussian-
parameter variational approximation. Most of the improvement happens in the
first two coordinate-ascent passes.
```

```{figure} variational_reconstruction.svg
:alt: Reconstructed trajectory and final Gaussian parameter factor for the Chapter 15 variational example.
:width: 98%

Top: noisy observations, hidden true position, and the variational mean
trajectory. Bottom: the final Gaussian factor $q(\theta)$ after the coordinate-
ascent updates converge.
```

The reconstruction is competitive because the trajectory update still enforces
the nonlinear dynamics strongly. The main difference from MAP plus Laplace is
conceptual: the lower panel is not a post hoc Hessian approximation around a
single optimum. It is the optimized Gaussian parameter factor from the
restricted objective itself.

:::{foldbox} Restricted free-energy limit and closed-form Gaussian parameter update

A convenient justification for the finite objective is to start from a proper
family

```{math}
q_\varepsilon(x_{0:N},\theta)
=
\mathcal{N}(x_{0:N};m_{0:N},\varepsilon I)\,
\mathcal{N}(\theta;\mu_\theta,\sigma_\theta^2),
\qquad \varepsilon > 0,
```

write its ELBO, and then examine the limit as $\varepsilon \to 0$. Terms that
depend only on $\varepsilon$ diverge because the state entropy collapses, but
they are independent of $(m_{0:N},\mu_\theta,\sigma_\theta^2)$. After dropping
those additive constants, the remaining finite part is the restricted free
energy

```{math}
:label: eq:ch15-variational-free-energy

\begin{aligned}
\mathcal{F}(m_{0:N},\mu_\theta,\sigma_\theta^2)
&=
\frac{1}{2}(m_0-\bar{x}_0)^\top P_0^{-1}(m_0-\bar{x}_0)
+ \frac{1}{2}\frac{(\mu_\theta-\mu_0)^2+\sigma_\theta^2}{\sigma_0^2}
- \frac{1}{2}\log \sigma_\theta^2\\
&\quad
+ \frac{1}{2}\sum_{k=0}^{N-1}
\left(m_{k+1}-F(m_k;\mu_\theta)\right)^\top
Q^{-1}
\left(m_{k+1}-F(m_k;\mu_\theta)\right)\\
&\quad
+ \frac{1}{2}\sigma_\theta^2
\sum_{k=0}^{N-1}
\left(\ppf{F}{\theta}(m_k;\mu_\theta)\right)^\top
Q^{-1}
\left(\ppf{F}{\theta}(m_k;\mu_\theta)\right)\\
&\quad
+ \frac{1}{2}\sum_{k=0}^{N}
\frac{(y_k-m_{k,q})^2}{R},
\end{aligned}
```

up to constants independent of $(m_{0:N},\mu_\theta,\sigma_\theta^2)$. This is
the objective minimized by the code above; it is the small-variance limit of a
proper ELBO, not the exact ELBO of the singular family in
{eq}`eq:ch15-variational-family`. Here
$m_{k,q}$ is the position component of $m_k$ and

```{math}
:label: eq:ch15-variational-theta-derivative

\ppf{F}{\theta}(m_k;\mu_\theta)
=
\begin{bmatrix}
0\\
-\Delta t \sin(m_{k,q})
\end{bmatrix}.
```

The extra term with $\sigma_\theta^2$ is exactly the expected dynamics
variance induced by the Gaussian factor on $\theta$.

To update $q(\theta)$ with the trajectory fixed, write

```{math}
:label: eq:ch15-variational-affine-theta

F(m_k;\theta)
=
\begin{bmatrix}
m_{k,q} + \Delta t\,m_{k,v}\\
m_{k,v} - \Delta t\,c\,m_{k,v}
\end{bmatrix}
+
d_k \theta,
\qquad
d_k
=
\begin{bmatrix}
0\\
-\Delta t \sin(m_{k,q})
\end{bmatrix}.
```

If
$a_k = m_{k+1} - [m_{k,q} + \Delta t\,m_{k,v},\;
m_{k,v} - \Delta t\,c\,m_{k,v}]^\top$,
then the quadratic terms in $\theta$ imply

```{math}
:label: eq:ch15-variational-theta-update

\sigma_\theta^2
=
\left(
\sigma_0^{-2}
+
\sum_{k=0}^{N-1} d_k^\top Q^{-1} d_k
\right)^{-1},
\qquad
\mu_\theta
=
\sigma_\theta^2
\left(
\mu_0 \sigma_0^{-2}
+
\sum_{k=0}^{N-1} d_k^\top Q^{-1} a_k
\right).
```

The trajectory update then minimizes
{eq}`eq:ch15-variational-free-energy` with respect to $m_{0:N}$. In code, that
is implemented by stacking the usual prior, dynamics, and observation
residuals, then adding one extra residual block
$\sqrt{\sigma_\theta^2}\,Q^{-1/2}\ppf{F}{\theta}(m_k;\mu_\theta)$ per time
step before taking a Gauss-Newton step.

:::

## What This Approximation Keeps and Drops

This example should be read as a deliberately restricted variational baseline.

- Compared with EM plus EKS, it does not build an approximate smoothing
  distribution with cross-covariances. It only keeps one trajectory mean and
  one scalar Gaussian factor.
- Compared with MAP plus Laplace, it does not postpone uncertainty to the end.
  The variance of $q(\theta)$ influences the trajectory update through the
  extra dynamics penalty during every outer iteration.
- The main approximation bias is clear in the family itself: all trajectory
  uncertainty and all state-parameter posterior correlation are collapsed away.

That bias is the right teaching tradeoff here. Students can see an ELBO-
motivated restricted approximation on a nonlinear identification problem,
understand exactly what was simplified, and compare its reconstruction and
convergence behavior conceptually against the EM-type and MAP-type approaches.
