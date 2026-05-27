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

## Shared Nonlinear Benchmark for Examples

The worked examples in the EM, MAP/Laplace, and variational sections use the
same partially observed nonlinear oscillator so that their behavior can be
compared directly:

```{math}
:label: eq:ch15-shared-benchmark-model

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
y_k = q_k + \eta_k.
```

Here $c=0.15$, $\Delta t=0.15$, and the unknown scalar parameter
$\theta$ controls the restoring force. Only position $q_k$ is observed, so
velocity is latent. The benchmark fixes
$\theta_{\mathrm{true}}=1.35$,
$Q=\operatorname{diag}(0.015^2,0.08^2)$, $R=0.12^2$,
$x_0\sim\mathcal{N}([0.9,0]^\top,\operatorname{diag}(0.25^2,0.25^2))$,
and $\theta\sim\mathcal{N}(0.9,0.35^2)$.

The common simulation utilities and Gauss-Newton helper are:

```python
import numpy as np

rng = np.random.default_rng(7)

dt, c, N = 0.15, 0.15, 40
theta_true = 1.35
Q_std = np.array([0.015, 0.08])
Q = np.diag(Q_std**2)
Q_inv = np.diag(1.0 / Q_std**2)
Q_sqrt_inv = np.diag(1.0 / Q_std)
sigma_y = 0.12
R = sigma_y**2
H = np.array([[1.0, 0.0]])

m0 = np.array([0.9, 0.0])
P0 = np.diag([0.25**2, 0.25**2])
P0_sqrt_inv = np.diag([1.0 / 0.25, 1.0 / 0.25])
theta_mean, sigma_theta = 0.9, 0.35
prior_precision = 1.0 / sigma_theta**2


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


def solve_damped_least_squares(z0, residual_and_jacobian, max_iter=8, damping=1e-2):
    z = z0.copy()
    history = []
    for _ in range(max_iter):
        r, J = residual_and_jacobian(z)
        objective = 0.5 * (r @ r)
        history.append(objective)
        H_gn = J.T @ J
        g = J.T @ r
        local_damping = damping

        while True:
            step = np.linalg.solve(H_gn + local_damping * np.eye(z.size), -g)
            candidate = z + step
            r_candidate, _ = residual_and_jacobian(candidate)
            candidate_objective = 0.5 * (r_candidate @ r_candidate)
            if candidate_objective < objective:
                z = candidate
                damping = max(local_damping / 3.0, 1e-6)
                break
            local_damping *= 10.0

        if np.linalg.norm(step) < 1e-6 * (1.0 + np.linalg.norm(z)):
            break

    return z, history


x_true = np.zeros((N + 1, 2))
x_true[0] = np.array([1.1, -0.15])
for k in range(N):
    x_true[k + 1] = F(x_true[k], theta_true) + Q_std * rng.normal(size=2)
y = x_true[:, 0] + sigma_y * rng.normal(size=N + 1)
```

The later code blocks assume this setup has already been run and show only the
method-specific estimation logic.
