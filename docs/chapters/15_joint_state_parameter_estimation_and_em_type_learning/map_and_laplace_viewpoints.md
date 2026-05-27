## MAP and Laplace Viewpoints

EM updates parameters through repeated smoothing. A closely related batch view
is to optimize over the whole latent trajectory and the unknown parameter at
once:

```{math}
:label: eq:ch15-map-objective

\left(\widehat{x}_{0:N},\widehat{\theta}\right)
\in
\arg\max_{x_{0:N},\theta}
\log p(x_{0:N},\theta\mid y_{0:N}).
```

This is a trajectory-optimization problem with priors and noise models built
in. It gives a single best-fitting state trajectory and parameter value. A
Laplace approximation then adds a local Gaussian uncertainty model around that
point estimate.

For the shared benchmark in {eq}`eq:ch15-shared-benchmark-model`, MAP
estimation is equivalent to minimizing the negative log posterior

```{math}
:label: eq:ch15-map-batch-objective

\begin{aligned}
\mathcal{J}(x_{0:N},\theta)
&=
\frac{1}{2}(x_0-m_0)^\top P_0^{-1}(x_0-m_0)
+
\frac{1}{2}\frac{(\theta-\mu_\theta)^2}{\sigma_\theta^2}\\
&\quad
+
\frac{1}{2}\sum_{k=0}^{N-1}
\left(x_{k+1}-F(x_k;\theta)\right)^\top
Q^{-1}
\left(x_{k+1}-F(x_k;\theta)\right)\\
&\quad
+
\frac{1}{2}\sum_{k=0}^{N}
\frac{\left(y_k-q_k\right)^2}{R}.
\end{aligned}
```

The terms have clear meanings: keep the initial condition near its prior, keep
the parameter near its prior, stay close to the dynamics, and match the noisy
position observations.

## Minimal Optimization-Based MAP Solve

Write the unknowns as one stacked vector

```{math}
:label: eq:ch15-map-stacked-variable

z
=
\begin{bmatrix}
x_0^\top & x_1^\top & \cdots & x_N^\top & \theta
\end{bmatrix}^\top.
```

A minimal Gauss-Newton or Levenberg-Marquardt solve only needs residuals and
Jacobians for the prior, dynamics, and observation blocks. The code below
assumes the shared benchmark setup and `solve_damped_least_squares` helper
have already been run:

```python
def unpack_map(z):
    x = z[:-1].reshape(N + 1, 2)
    theta = z[-1]
    return x, theta


def map_residual_and_jacobian(z):
    x, theta = unpack_map(z)
    rows = 2 + 1 + 2 * N + (N + 1)
    J = np.zeros((rows, z.size))
    r = np.zeros(rows)
    row = 0

    r[row : row + 2] = P0_sqrt_inv @ (x[0] - m0)
    J[row : row + 2, 0:2] = P0_sqrt_inv
    row += 2

    r[row] = (theta - theta_mean) / sigma_theta
    J[row, -1] = 1.0 / sigma_theta
    row += 1

    for k in range(N):
        dyn = Q_sqrt_inv @ (x[k + 1] - F(x[k], theta))
        r[row : row + 2] = dyn
        J[row : row + 2, 2 * k : 2 * k + 2] = -Q_sqrt_inv @ dF_dx(x[k], theta)
        J[row : row + 2, 2 * (k + 1) : 2 * (k + 1) + 2] = Q_sqrt_inv
        J[row : row + 2, -1] = -Q_sqrt_inv @ dF_dtheta(x[k])
        row += 2

    for k in range(N + 1):
        r[row] = (y[k] - x[k, 0]) / sigma_y
        J[row, 2 * k] = -1.0 / sigma_y
        row += 1

    return r, J


z0 = np.zeros(2 * (N + 1) + 1)
z0[:-1:2] = y
z0[1:-1:2] = 0.0
z0[-1] = theta_mean

z_map, objective_history = solve_damped_least_squares(
    z0,
    map_residual_and_jacobian,
    max_iter=8,
)
x_map, theta_map = unpack_map(z_map)
r_map, J_map = map_residual_and_jacobian(z_map)
H_gn = J_map.T @ J_map
laplace_cov = np.linalg.pinv(H_gn)
theta_std = np.sqrt(laplace_cov[-1, -1])
```

On this benchmark, the objective drops from about $3974.5$ to $14.9$ in five
iterations, and the recovered parameter is
$\theta^{\mathrm{MAP}} \approx 1.37$ for
$\theta_{\mathrm{true}} = 1.35$.

```{figure} map_laplace_objective_history.svg
:alt: Objective history for the Chapter 15 MAP plus Laplace example.
:width: 95%

Objective history for the minimal damped-pendulum MAP solve. Most of the
decrease happens in the first Gauss-Newton step because the initialization uses
the observations directly for position but starts with zero velocity and only a
prior guess for $\theta$.
```

```{figure} map_laplace_reconstruction.svg
:alt: Reconstructed trajectory and Laplace parameter marginal for the Chapter 15 MAP plus Laplace example.
:width: 98%

Top: noisy observations, hidden true position, and the MAP reconstruction.
Bottom: the one-dimensional Laplace marginal for $\theta$, obtained from the
local Gaussian approximation at the MAP point.
```

The top panel is the point estimate: one optimized trajectory paired with one
optimized parameter. The lower panel is different. It does not reoptimize the
trajectory for many new values of $\theta$; it only reads off local curvature
near the optimum. That is the key distinction students should keep in mind.

:::{foldbox} Gauss-Newton linearization and the Laplace covariance

Stack the residual blocks from the code into one vector

```{math}
:label: eq:ch15-map-residual-stack

r(z)
=
\begin{bmatrix}
r_{\mathrm{prior}}(z)\\
r_{\theta}(z)\\
r_{\mathrm{dyn}}(z)\\
r_{\mathrm{obs}}(z)
\end{bmatrix},
\qquad
\mathcal{J}(z)=\frac{1}{2}r(z)^\top r(z).
```

Linearizing around the current iterate $z^{(i)}$ gives

```{math}
:label: eq:ch15-map-residual-linearization

r(z^{(i)}+\delta)
\approx
r(z^{(i)}) + J(z^{(i)})\,\delta,
```

where $J=\ppf{r}{z}$ is the residual Jacobian. Minimizing the quadratic model
produces the Gauss-Newton step

```{math}
:label: eq:ch15-map-gauss-newton-step

\left(J^\top J + \lambda I\right)\delta
=
-J^\top r,
```

with $\lambda \ge 0$ as a simple Levenberg-Marquardt damping term. The exact
Hessian of $\mathcal{J}$ contains additional second-derivative terms from the
nonlinear residuals, but near a well-fit solution the approximation
$\nabla^2 \mathcal{J}(z) \approx J(z)^\top J(z)$ is often accurate enough for a
small instructional example.

At the MAP point $z^{\mathrm{MAP}}$, the negative log posterior therefore has
the second-order approximation

```{math}
:label: eq:ch15-map-laplace-expansion

\mathcal{J}(z)
\approx
\mathcal{J}(z^{\mathrm{MAP}})
+
\frac{1}{2}
\left(z-z^{\mathrm{MAP}}\right)^\top
H_{\mathrm{GN}}
\left(z-z^{\mathrm{MAP}}\right),
\qquad
H_{\mathrm{GN}} = J(z^{\mathrm{MAP}})^\top J(z^{\mathrm{MAP}}).
```

Exponentiating the negative quadratic gives the Laplace approximation

```{math}
:label: eq:ch15-map-laplace-gaussian

p(z\mid y_{0:N})
\approx
\mathcal{N}\!\left(
z^{\mathrm{MAP}},
H_{\mathrm{GN}}^{-1}
\right).
```

The scalar parameter uncertainty in the figure is just the $\theta\theta$
entry of this covariance matrix.

:::

## What the Laplace Approximation Adds

The MAP point answers the optimization question: which trajectory and parameter
best balance the priors, the dynamics, and the observations? The Laplace step
answers a nearby uncertainty question: how sharply curved is that optimum?

For this benchmark, the local Gaussian parameter marginal is

```{math}
:label: eq:ch15-map-theta-marginal

\theta \mid y_{0:N}
\approx
\mathcal{N}\!\left(1.367,\;0.127^2\right).
```

That spread is useful because it distinguishes "best fit" from "well
identified." A flat posterior mode would have produced a much larger local
variance even if the MAP point itself looked reasonable. The limitation is just
as important: if the joint posterior is strongly skewed, multimodal, or has a
curved ridge, one local Gaussian can misrepresent the true uncertainty.
