# Notation Convention for *Learning Dynamics from Data*

This convention is designed for course notes that move across discrete-time and continuous-time dynamics, full-state and partial-state observations, system identification, state estimation, trajectory optimization, optimal control, stochastic dynamics, structure-preserving learning, and reduced-order modeling.

The main convention follows the same principle as the reference notation guide: **font/style indicates the mathematical category**, while subscripts and superscripts indicate components, time, trajectories, modes, labels, and estimates.

---

## 1. Guiding Principle

Use **font/style** to identify what kind of object is being discussed.

| Style | Meaning | Examples |
|---|---|---|
| italic lowercase | mathematical variables, scalar functions, vector-valued states when unambiguous | $x(t)$, $u(t)$, $y(t)$, $z(t)$, $f(x)$ |
| italic uppercase | finite-dimensional matrices or matrix-valued functions | $A$, $B$, $C$, $D$, $Q$, $R$, $P$, $K$ |
| bold lowercase | numerical data vectors, sampled snapshots, realized vectors | $\mathbf{x}_k$, $\mathbf{u}_k$, $\mathbf{y}_k$, $\mathbf{w}_k$ |
| bold uppercase | data matrices, sampled arrays, assembled design matrices | $\mathbf{X}$, $\mathbf{U}$, $\mathbf{Y}$, $\mathbf{\Theta}$ |
| Greek lowercase | parameters, operating conditions, eigenvalues, noise symbols, test functions | $\theta$, $\mu$, $\lambda$, $\eta$, $\xi$, $\varphi$ |
| Greek uppercase | covariance-like matrices, diagonal spectra, feature matrices when conventional | $\Sigma$, $\Lambda$, $\Phi$, $\Psi$ |
| $\mathbb{\cdot}$ | standard number spaces and expectation | $\mathbb{R}$, $\mathbb{C}$, $\mathbb{N}$, $\mathbb{E}$ |
| $\mathcal{\cdot}$ | operators, function spaces, manifolds, feasible sets, distributions | $\mathcal{F}$, $\mathcal{K}$, $\mathcal{L}$, $\mathcal{H}$, $\mathcal{M}$ |
| $\mathrm{\cdot}$ | non-variable labels in subscripts/superscripts | $A_{\mathrm{r}}$, $e_{\mathrm{roll}}$, $\theta^{\mathrm{MAP}}$ |

Core rule:

> Use plain italic symbols for mathematical model variables, but bold symbols for sampled numerical data and assembled arrays.

Thus $x(t)$ is the abstract state trajectory, while $\mathbf{x}_k$ is a stored snapshot in a dataset.

---

## 2. States, Inputs, Outputs, and Time

Use

\[
x(t)\in \mathbb{R}^n
\]

for the continuous-time state.

Use

\[
u(t)\in \mathbb{R}^m,
\qquad
y(t)\in \mathbb{R}^p
\]

for input and output.

For discrete time, use

\[
x_k = x(t_k)
\]

for the mathematical state at time index $k$, and use

\[
\mathbf{x}_k
\]

for the stored numerical snapshot.

Recommended distinction:

| Quantity | Notation |
|---|---|
| continuous-time state | $x(t)$ |
| discrete-time mathematical state | $x_k$ |
| numerical snapshot | $\mathbf{x}_k$ |
| input | $u(t)$ or $u_k$ |
| stored input sample | $\mathbf{u}_k$ |
| output/measurement | $y(t)$ or $y_k$ |
| stored output sample | $\mathbf{y}_k$ |
| sample time | $t_k$ |
| time step | $\Delta t$ |

Components are indexed by subscripts:

\[
x_i(t)
\]

means the $i$-th component of $x(t)$.

If both component and time are needed, use

\[
x_i(t_k)
\qquad \text{or} \qquad
(\mathbf{x}_k)_i.
\]

Avoid using $x_i$ for the $i$-th sample, since $x_i$ already denotes a component.

---

## 3. Discrete-Time and Continuous-Time Dynamics

Use $F$ for discrete-time maps and $f$ for continuous-time vector fields.

### Discrete-time dynamics

\[
x_{k+1}=F(x_k,u_k;\mu,\theta).
\]

Special cases:

\[
x_{k+1}=F(x_k)
\]

for autonomous dynamics,

\[
x_{k+1}=F(x_k,u_k)
\]

for controlled dynamics, and

\[
x_{k+1}=F(x_k,u_k;\mu)
\]

for parametric dynamics with operating parameter $\mu$.

### Continuous-time dynamics

\[
\dot{x}(t)=f(x(t),u(t);\mu,\theta).
\]

Special cases:

\[
\dot{x}=f(x),
\qquad
\dot{x}=f(x,u),
\qquad
\dot{x}=f(x,u;\mu).
\]

Use $\theta$ for unknown parameters to be learned. Use $\mu$ for known or externally specified operating parameters, such as Reynolds number, Mach number, mass ratio, graph parameter, or environment parameter.

| Symbol | Meaning |
|---|---|
| $F$ | discrete-time map |
| $f$ | continuous-time vector field |
| $\theta$ | unknown learned parameter vector |
| $\mu$ | operating parameter or system parameter used to index a family |
| $n$ | state dimension |
| $m$ | input dimension |
| $p$ | output dimension |

---

## 4. Data Matrices and Trajectory Collections

Use bold uppercase matrices for sampled datasets.

For full-state snapshot data:

\[
\mathbf{X}_-
=
\begin{bmatrix}
\mathbf{x}_0 & \mathbf{x}_1 & \cdots & \mathbf{x}_{N-1}
\end{bmatrix},
\]

\[
\mathbf{X}_+
=
\begin{bmatrix}
\mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_{N}
\end{bmatrix}.
\]

For input data:

\[
\mathbf{U}_-
=
\begin{bmatrix}
\mathbf{u}_0 & \mathbf{u}_1 & \cdots & \mathbf{u}_{N-1}
\end{bmatrix}.
\]

For output data:

\[
\mathbf{Y}
=
\begin{bmatrix}
\mathbf{y}_0 & \mathbf{y}_1 & \cdots & \mathbf{y}_{N}
\end{bmatrix}.
\]

Use a parenthesized superscript for trajectory or realization index:

\[
\mathbf{x}^{(r)}_k
\]

means the $k$-th snapshot from trajectory or realization $r$.

For a collection of trajectories, use

\[
\mathcal{D}
=
\left\{
\left(\mathbf{x}^{(r)}_k,\mathbf{u}^{(r)}_k,\mathbf{y}^{(r)}_k\right)_{k=0}^{N_r}
\right\}_{r=1}^{N_{\mathrm{traj}}}.
\]

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$ | dataset |
| $N$ | number of time steps in one trajectory |
| $N_{\mathrm{traj}}$ | number of trajectories |
| $r$ | trajectory/realization index |
| $\mathbf{X}_-$ | current-state snapshot matrix |
| $\mathbf{X}_+$ | shifted/future-state snapshot matrix |
| $\mathbf{U}_-$ | input snapshot matrix |
| $\mathbf{Y}$ | output/measurement data matrix |

---

## 5. Linear State-Space Models

Use italic uppercase matrices for finite-dimensional state-space models.

Discrete-time:

\[
x_{k+1}=Ax_k+Bu_k,
\qquad
y_k=Cx_k+Du_k.
\]

Continuous-time:

\[
\dot{x}=Ax+Bu,
\qquad
y=Cx+Du.
\]

| Symbol | Meaning |
|---|---|
| $A$ | state matrix or linear generator |
| $B$ | input matrix |
| $C$ | output matrix |
| $D$ | feedthrough matrix |
| $G(s)$ | transfer function |
| $Q$ | process-noise covariance |
| $R$ | measurement-noise covariance |
| $P$ | state-estimation error covariance or Riccati matrix |
| $K$ | feedback gain or Kalman gain, depending on context |

Because $K$ can denote feedback gain, Kalman gain, stiffness matrix, or a finite Koopman approximation, always label when needed:

\[
K_{\mathrm{fb}},
\qquad
K_{\mathrm{KF}},
\qquad
K_{\mathrm{stiff}},
\qquad
K_{\mathrm{EDMD}}.
\]

Use $\mathcal{K}$, not $K$, for the Koopman operator.

---

## 6. Linear Identification and Subspace Identification

For full-state linear identification, write

\[
\mathbf{X}_+ \approx A\mathbf{X}_-.
\]

The least-squares estimate is

\[
\widehat{A}
=
\mathbf{X}_+\mathbf{X}_-^\dagger.
\]

For controlled full-state identification:

\[
\mathbf{X}_+
\approx
A\mathbf{X}_-+B\mathbf{U}_-
=
\begin{bmatrix}
A & B
\end{bmatrix}
\begin{bmatrix}
\mathbf{X}_-\\
\mathbf{U}_-
\end{bmatrix}.
\]

Use hats for estimated quantities:

\[
\widehat{A},\qquad \widehat{B},\qquad \widehat{\theta}.
\]

For input-output and subspace identification, use Markov parameters

\[
H_j = CA^{j-1}B
\]

and block Hankel matrices

\[
\mathbf{H}_{p,q}.
\]

Use $\mathbf{H}$ for empirical Hankel matrices, and reserve $\mathcal{H}$ for Hilbert spaces.

Recommended terms:

| Symbol | Meaning |
|---|---|
| $\widehat{A}$ | estimated state matrix |
| $\mathbf{H}_{p,q}$ | block Hankel matrix |
| $H_j$ | Markov parameter |
| $n_{\mathrm{sys}}$ | identified system order, if distinct from full state dimension |
| $r$ | truncation rank or reduced dimension |

If DMD-like methods are mentioned, frame them as special cases of full-state linear or subspace-style identification:

\[
A_{\mathrm{DMD}}
=
\mathbf{X}_+\mathbf{X}_-^\dagger.
\]

---

## 7. Feature Libraries and Linear-in-Parameter Models

Use $\Theta$ for the feature/library map and $\mathbf{\Theta}$ for its sampled data matrix.

Continuous-time model:

\[
\dot{x}
=
\Theta(x,u;\mu)\xi.
\]

Discrete-time model:

\[
x_{k+1}
=
\Theta(x_k,u_k;\mu)\xi.
\]

Here $\xi$ denotes coefficients in a linear-in-parameter expansion. Use $\theta$ for general nonlinear parameters.

Sampled library matrix:

\[
\mathbf{\Theta}
=
\begin{bmatrix}
\Theta(\mathbf{x}_0,\mathbf{u}_0)^\top\\
\Theta(\mathbf{x}_1,\mathbf{u}_1)^\top\\
\vdots\\
\Theta(\mathbf{x}_{N-1},\mathbf{u}_{N-1})^\top
\end{bmatrix}.
\]

A typical sparse regression problem is

\[
\widehat{\xi}
=
\arg\min_{\xi}
\|\mathbf{\dot{X}}-\mathbf{\Theta}\xi\|_2^2
+
\lambda\|\xi\|_1.
\]

| Symbol | Meaning |
|---|---|
| $\Theta(x,u)$ | feature/library map |
| $\mathbf{\Theta}$ | sampled feature matrix |
| $\xi$ | linear-in-parameter coefficient vector |
| $\lambda_{\mathrm{reg}}$ | regularization parameter |
| $s$ | sparsity level, if needed |

Avoid using $\lambda$ alone for regularization when eigenvalues also appear. Prefer $\lambda_{\mathrm{reg}}$.

---

## 8. Known Parametric Models and Inverse Problems

For known model structure with unknown parameters, write

\[
\dot{x}=f(x,u;\theta),
\qquad
y=h(x,u;\theta).
\]

Use $\theta$ for unknown parameters and $\widehat{\theta}$ for their estimates.

A typical parameter-estimation problem is

\[
\widehat{\theta}
=
\arg\min_{\theta}
\sum_{k=0}^{N}
\left\|
\mathbf{y}_k-h(x_k;\theta)
\right\|_{R^{-1}}^2
\]

subject to

\[
x_{k+1}=F(x_k,u_k;\theta).
\]

Sensitivities:

\[
S_\theta(t)
=
\frac{\partial x(t)}{\partial \theta}.
\]

Sensitivity equation:

\[
\dot{S}_\theta
=
f_x S_\theta + f_\theta.
\]

Adjoint variable:

\[
\lambda(t)
\]

or, if $\lambda$ is already used for eigenvalues, use

\[
p(t)
\]

for costate/adjoint variables in optimal-control derivations.

| Symbol | Meaning |
|---|---|
| $\theta$ | unknown parameter vector |
| $\widehat{\theta}$ | estimated parameter vector |
| $S_\theta$ | parameter sensitivity matrix |
| $\lambda(t)$ or $p(t)$ | adjoint/costate |
| $\mathcal{J}(\theta)$ | objective/loss functional |

---

## 9. Gray-Box, Residual, and Closure Models

Use $f_{\mathrm{known}}$ for known dynamics and $g_\theta$ for learned missing terms.

A standard gray-box model is

\[
\dot{x}
=
f_{\mathrm{known}}(x,u;\mu)
+
g_\theta(x,u;\mu).
\]

A residual target is

\[
r(t)
=
\dot{x}(t)-f_{\mathrm{known}}(x(t),u(t);\mu).
\]

Use $\mathbf{r}_k$ for sampled residuals.

For unknown constitutive or closure components, write

\[
\dot{x}
=
f\bigl(x,u,c_\theta(x,u)\bigr),
\]

where $c_\theta$ is a learned constitutive relation, closure term, damping law, source term, or forcing model.

| Symbol | Meaning |
|---|---|
| $f_{\mathrm{known}}$ | known physics/dynamics |
| $g_\theta$ | learned residual dynamics |
| $c_\theta$ | learned closure or constitutive term |
| $r(t)$ | continuous residual |
| $\mathbf{r}_k$ | sampled residual |

---

## 10. Incomplete State Models, Latent States, and Memory

When separating resolved and unresolved states, use

\[
x
=
\begin{bmatrix}
x_a\\
x_b
\end{bmatrix},
\]

where $x_a$ is the resolved or modeled state and $x_b$ is the unresolved state.

If a learned latent state is introduced, use $z$:

\[
\dot{x}_a
=
f_a(x_a,z,u),
\qquad
\dot{z}
=
g_\theta(x_a,z,u).
\]

Use $z$ for latent coordinates inferred or learned from data. Use $a$ for reduced coordinates from projection-based model reduction.

For non-Markovian closure, use history notation:

\[
\dot{x}_a(t)
=
f_a(x_a(t),u(t))
+
\int_0^t
\mathcal{M}_\theta(t-\tau,x_a(\tau),u(\tau))\,d\tau.
\]

| Symbol | Meaning |
|---|---|
| $x_a$ | resolved, modeled, or observed physical state |
| $x_b$ | unresolved or hidden physical state |
| $z$ | learned latent state |
| $\mathcal{M}_\theta$ | learned memory kernel |
| $\tau$ | integration or delay variable |
| $m$ | number of delays, when using delay coordinates |

Delay-coordinate state:

\[
z_k
=
\begin{bmatrix}
y_k\\
y_{k-1}\\
\vdots\\
y_{k-m}
\end{bmatrix}.
\]

---

## 11. Black-Box Nonlinear Dynamics Models

For a learned discrete-time nonlinear model, use

\[
x_{k+1}
=
F_\theta(x_k,u_k;\mu).
\]

For a learned continuous-time vector field, use

\[
\dot{x}
=
f_\theta(x,u;\mu).
\]

For neural ODEs:

\[
\dot{x}(t)
=
f_\theta(x(t),t),
\qquad
x(t_1)
=
x(t_0)
+
\int_{t_0}^{t_1}
f_\theta(x(t),t)\,dt.
\]

For encoder-decoder latent dynamics:

\[
z_k=E_\theta(y_{0:k}),
\qquad
z_{k+1}=G_\theta(z_k,u_k),
\qquad
\widehat{y}_k=D_\theta(z_k).
\]

| Symbol | Meaning |
|---|---|
| $F_\theta$ | learned discrete-time map |
| $f_\theta$ | learned continuous-time vector field |
| $E_\theta$ | encoder |
| $D_\theta$ | decoder |
| $G_\theta$ | latent discrete-time dynamics |
| $z_k$ | latent state |
| $\widehat{y}_k$ | predicted output |

Use $\widehat{x}_k$ for predicted or estimated states, depending on context. If both are present, distinguish explicitly:

\[
\widehat{x}_{k|k}
\quad \text{filtered estimate},
\qquad
\widehat{x}_{k|N}
\quad \text{smoothed estimate},
\qquad
\widetilde{x}_k
\quad \text{model rollout prediction}.
\]

---

## 12. Observation Models, Noise, and Stochastic Dynamics

Use

\[
y_k=h(x_k,u_k;\theta)+v_k
\]

for observations.

Use $w_k$ for process noise and $v_k$ for measurement noise:

\[
x_{k+1}=F(x_k,u_k;\theta)+w_k,
\qquad
w_k\sim \mathcal{N}(0,Q),
\]

\[
y_k=h(x_k,u_k;\theta)+v_k,
\qquad
v_k\sim \mathcal{N}(0,R).
\]

For continuous-time stochastic dynamics:

\[
dx=f(x,u;\theta)\,dt+G(x,u;\theta)\,dW_t.
\]

| Symbol | Meaning |
|---|---|
| $w_k$ | process noise |
| $v_k$ | measurement noise |
| $Q$ | process-noise covariance |
| $R$ | measurement-noise covariance |
| $W_t$ | Wiener process |
| $G$ | diffusion/noise input matrix |
| $p(x)$ | probability density |
| $\mu$ | probability measure or invariant measure, when context is probabilistic |
| $\pi(x)$ | stationary distribution, if $\mu$ is already used for parameters |

Use $\eta_k$ only for generic noise when the distinction between process and measurement noise is not important.

---

## 13. Filtering, Smoothing, and State Estimation

Use conditional-estimate notation when discussing filtering and smoothing.

Filtering estimate:

\[
\widehat{x}_{k|k}
=
\mathbb{E}[x_k\mid y_{0:k}].
\]

Prediction estimate:

\[
\widehat{x}_{k+1|k}
=
\mathbb{E}[x_{k+1}\mid y_{0:k}].
\]

Smoothing estimate:

\[
\widehat{x}_{k|N}
=
\mathbb{E}[x_k\mid y_{0:N}].
\]

Covariances:

\[
P_{k|k},
\qquad
P_{k+1|k},
\qquad
P_{k|N}.
\]

Innovation:

\[
\nu_k
=
y_k-h(\widehat{x}_{k|k-1}).
\]

Kalman gain:

\[
K_k^{\mathrm{KF}}.
\]

| Symbol | Meaning |
|---|---|
| $\widehat{x}_{k|k}$ | filtered estimate |
| $\widehat{x}_{k+1|k}$ | predicted estimate |
| $\widehat{x}_{k|N}$ | smoothed estimate |
| $P_{k|k}$ | filtered covariance |
| $P_{k|N}$ | smoothed covariance |
| $\nu_k$ | innovation |
| $K_k^{\mathrm{KF}}$ | Kalman gain |
| $\mathcal{Y}_{0:k}$ | sigma-algebra or information set generated by observations |

For ensemble methods, use superscript $(e)$ for ensemble members:

\[
x_k^{(e)},\qquad e=1,\ldots,N_{\mathrm{ens}}.
\]

Use $N_{\mathrm{ens}}$ for ensemble size.

---

## 14. Joint State-Parameter Estimation, EM, and Variational Inference

Use augmented states when parameters are estimated as states:

\[
\bar{x}_k
=
\begin{bmatrix}
x_k\\
\theta_k
\end{bmatrix}.
\]

Use $q$ for approximate posterior distributions:

\[
q(x_{0:N},\theta)
\approx
p(x_{0:N},\theta\mid y_{0:N}).
\]

Use $\mathcal{L}_{\mathrm{ELBO}}$ for the evidence lower bound:

\[
\mathcal{L}_{\mathrm{ELBO}}(q,\theta)
=
\mathbb{E}_{q}
\left[
\log p_\theta(x_{0:N},y_{0:N})
-
\log q(x_{0:N})
\right].
\]

Use $\ell(\theta)$ for log-likelihood:

\[
\ell(\theta)
=
\log p_\theta(y_{0:N}).
\]

| Symbol | Meaning |
|---|---|
| $\bar{x}_k$ | augmented state |
| $q$ | approximate posterior |
| $p_\theta$ | probabilistic model parameterized by $\theta$ |
| $\ell(\theta)$ | log-likelihood |
| $\mathcal{L}_{\mathrm{ELBO}}$ | evidence lower bound |
| $\theta^{(i)}$ | parameter estimate at iteration $i$ |
| $i$ | optimization/EM iteration index |

Use superscript $(i)$ for algorithm iteration, not for time.

---

## 15. Trajectory Optimization and Data Assimilation

Use $J$ or $\mathcal{J}$ for cost/objective functionals.

A weak-constraint trajectory optimization problem:

\[
\min_{x_{0:N},\theta}
\mathcal{J}(x_{0:N},\theta)
=
\frac{1}{2}
\sum_{k=0}^{N}
\|y_k-h(x_k)\|_{R^{-1}}^2
+
\frac{1}{2}
\sum_{k=0}^{N-1}
\|x_{k+1}-F_\theta(x_k,u_k)\|_{Q^{-1}}^2.
\]

Use

\[
q_k
=
x_{k+1}-F_\theta(x_k,u_k)
\]

for model-error/control variables in weak-constraint formulations, if needed.

Use $p_k$ for adjoint/costate variables in discrete-time trajectory optimization:

\[
p_k
\]

instead of $\lambda_k$ when eigenvalues are also in use.

| Symbol | Meaning |
|---|---|
| $\mathcal{J}$ | trajectory optimization objective |
| $q_k$ | model-error variable or weak-constraint control |
| $p_k$ | adjoint/costate |
| $\nabla_x \mathcal{J}$ | state gradient |
| $\nabla_\theta \mathcal{J}$ | parameter gradient |
| $H$ | Hamiltonian in optimal-control derivations |

To avoid conflict with block Hankel matrices, write the optimal-control Hamiltonian as

\[
\mathscr{H}(x,u,p)
\]

when both appear in the same module.

---

## 16. Optimal Control and Identification for Control

Use $\ell$ for stage cost and $\ell_f$ for terminal cost:

\[
\min_{u_{0:N-1}}
\sum_{k=0}^{N-1}
\ell(x_k,u_k)
+
\ell_f(x_N)
\]

subject to

\[
x_{k+1}=F_\theta(x_k,u_k).
\]

Use $\pi$ for policies:

\[
u_k=\pi_\theta(x_k).
\]

Use $V$ and $Q_{\mathrm{RL}}$ for value and action-value functions:

\[
V^\pi(x),
\qquad
Q_{\mathrm{RL}}^\pi(x,u).
\]

Avoid using $Q$ alone for the action-value function because $Q$ also denotes process covariance. Use $Q_{\mathrm{RL}}$ if both appear.

| Symbol | Meaning |
|---|---|
| $\ell(x,u)$ | stage cost |
| $\ell_f(x)$ | terminal cost |
| $\pi_\theta$ | policy |
| $V^\pi$ | value function |
| $Q_{\mathrm{RL}}^\pi$ | action-value function |
| $K_{\mathrm{fb}}$ | feedback gain |
| $\mathcal{U}$ | admissible input set |
| $\mathcal{X}$ | admissible state set |

For MPC, use horizon $N_{\mathrm{MPC}}$ if confusion with dataset length $N$ is possible.

---

## 17. Losses, Errors, and Validation Metrics

Use $e$ for scalar error metrics and $\mathcal{J}$ for training objectives.

One-step prediction error:

\[
e_{\mathrm{1step}}
=
\sum_{k=0}^{N-1}
\left\|
\mathbf{x}_{k+1}
-
\widehat{F}(\mathbf{x}_k,\mathbf{u}_k)
\right\|^2.
\]

Rollout error:

\[
e_{\mathrm{roll}}
=
\sum_{k=0}^{N}
\left\|
\mathbf{x}_k-\widetilde{x}_k
\right\|^2.
\]

Vector-field error:

\[
e_{\mathrm{vf}}
=
\sum_{k=0}^{N}
\left\|
\dot{\mathbf{x}}_k-\widehat{f}(\mathbf{x}_k,\mathbf{u}_k)
\right\|^2.
\]

Control-performance error or regret:

\[
e_{\mathrm{ctrl}}
\quad \text{or} \quad
\mathcal{R}_{\mathrm{ctrl}}.
\]

| Symbol | Meaning |
|---|---|
| $e_{\mathrm{1step}}$ | one-step prediction error |
| $e_{\mathrm{roll}}$ | rollout error |
| $e_{\mathrm{vf}}$ | vector-field error |
| $e_{\mathrm{stat}}$ | statistical/invariant-measure error |
| $e_{\mathrm{ctrl}}$ | control-performance error |
| $\mathcal{J}_{\mathrm{train}}$ | training objective |
| $\mathcal{J}_{\mathrm{val}}$ | validation objective |

Use roman labels in subscripts for error names.

---

## 18. Geometry, Constraints, and Invariance

Use $\mathcal{M}$ for manifolds and $T_x\mathcal{M}$ for tangent spaces.

If the state is constrained by

\[
c(x)=0,
\]

then admissible vector fields satisfy

\[
\nabla c(x)^\top f(x)=0.
\]

Use $C(x)$ for conserved quantities or invariants, but avoid conflict with output matrix $C$ by writing $I_j(x)$ for invariants when both appear.

Invariant condition:

\[
\nabla I_j(x)^\top f(x)=0.
\]

For energy, use

\[
E(x)
\]

or Hamiltonian

\[
H(q,p).
\]

Hamiltonian dynamics:

\[
\dot{x}=J\nabla H(x),
\qquad
J^\top=-J.
\]

Port-Hamiltonian dynamics:

\[
\dot{x}
=
[J(x)-R(x)]\nabla H(x)+G(x)u,
\qquad
R(x)\succeq 0.
\]

For symmetry groups, use $\mathcal{G}$ for a group and $\rho(g)$ for its representation:

\[
f(\rho(g)x)=\rho(g)f(x).
\]

| Symbol | Meaning |
|---|---|
| $\mathcal{M}$ | manifold/state space |
| $T_x\mathcal{M}$ | tangent space |
| $c(x)$ | constraint function |
| $I_j(x)$ | invariant/conserved quantity |
| $E(x)$ | energy/Lyapunov-like function |
| $H(q,p)$ | Hamiltonian |
| $J$ | skew-symmetric structure matrix |
| $R$ | dissipation matrix, when not used for measurement covariance in same derivation |
| $\mathcal{G}$ | symmetry group |
| $\rho(g)$ | group representation |

If $R$ is already measurement covariance in the same discussion, use $D_{\mathrm{diss}}$ for dissipation.

---

## 19. Reduced-Order Modeling and Latent Coordinates

Use $V_r$ for trial/reduction basis and $W_r$ for test basis.

Projection-based reduced representation:

\[
x(t)\approx V_r a(t).
\]

Use $a(t)$ for projection-based reduced coordinates.

Galerkin projection:

\[
\dot{a}
=
V_r^\top f(V_r a,u).
\]

Petrov-Galerkin projection:

\[
\dot{a}
=
(W_r^\top V_r)^{-1}W_r^\top f(V_r a,u).
\]

Autoencoder latent representation:

\[
z=E_\theta(x),
\qquad
\widehat{x}=D_\theta(z).
\]

Use $a$ for projection coordinates and $z$ for learned latent coordinates.

| Symbol | Meaning |
|---|---|
| $V_r$ | trial/reduction basis |
| $W_r$ | test basis |
| $a(t)$ | projection-based reduced coordinates |
| $z(t)$ | learned latent coordinates |
| $r$ | reduced dimension |
| $A_r,B_r,C_r$ | reduced state-space matrices |
| $W_c,W_o$ | controllability and observability Gramians |
| $\sigma_j^{\mathrm{H}}$ | Hankel singular value |

Use $\sigma_j$ for ordinary singular values and $\sigma_j^{\mathrm{H}}$ for Hankel singular values.

---

## 20. Operators and Function Spaces

Although the course is not organized around operator theory as a standalone module, operator notation is still useful for function spaces, generators, transfer operators, weak forms, and reduced models.

Use calligraphic letters for abstract operators:

| Symbol | Meaning |
|---|---|
| $\mathcal{F}$ | abstract dynamics map, when $F$ is reserved for finite-dimensional map |
| $\mathcal{L}$ | differential operator, generator, or loss functional when context is clear |
| $\mathcal{K}$ | Koopman operator, if mentioned |
| $\mathcal{P}$ | Perron-Frobenius/transfer operator |
| $\mathcal{R}(z;\mathcal{A})$ | resolvent operator |
| $\mathcal{H}$ | Hilbert space |
| $\mathcal{V}$ | trial space |
| $\mathcal{W}$ | test space |
| $\mathcal{D}(\mathcal{A})$ | domain of operator $\mathcal{A}$ |

For weak forms, use test functions

\[
\varphi_j(t)
\]

or

\[
\psi_j(t)
\]

but avoid conflict with Koopman eigenfunctions by reserving $\varphi_j$ for test functions in weak-form modules and explicitly labeling Koopman eigenfunctions if needed.

Weak-form residual:

\[
\int_{t_0}^{t_1}
\varphi(t)
\left[
\dot{x}(t)-f_\theta(x(t),u(t))
\right]dt.
\]

After integration by parts:

\[
-\int_{t_0}^{t_1}
\dot{\varphi}(t)x(t)\,dt
-
\int_{t_0}^{t_1}
\varphi(t)f_\theta(x(t),u(t))\,dt.
\]

---

## 21. Subscripts and Superscripts

### Subscripts

Use subscripts for components, time indices, modes, reduced dimensions, and labels.

| Notation | Meaning |
|---|---|
| $x_i$ | component $i$ of $x$ |
| $x_k$ | mathematical state at time step $k$ |
| $\mathbf{x}_k$ | stored snapshot at time step $k$ |
| $\phi_j$ | mode or basis function $j$ |
| $A_r$ | reduced matrix of dimension $r$ |
| $t_k$ | $k$-th time sample |
| $\theta_j$ | parameter component $j$ |
| $e_{\mathrm{roll}}$ | rollout error |

### Superscripts

Use superscripts for realizations, iterations, method labels, transposes, inverses, and adjoints.

| Notation | Meaning |
|---|---|
| $x^{(r)}$ | trajectory or realization $r$ |
| $\theta^{(i)}$ | optimization/EM iteration $i$ |
| $A^T$ | transpose |
| $A^*$ | conjugate transpose / adjoint |
| $A^{-1}$ | inverse |
| $A^\dagger$ | Moore-Penrose pseudoinverse |
| $A^{\mathrm{LS}}$ | least-squares estimate label |
| $\theta^{\mathrm{MAP}}$ | MAP estimate |

Use parenthesized superscripts for indices or realizations:

\[
x^{(r)}
\]

not $x^r$, unless it is a power.

Use roman superscripts for method labels:

\[
\widehat{A}^{\mathrm{LS}},
\qquad
\widehat{\theta}^{\mathrm{MAP}},
\qquad
z^{\mathrm{enc}}.
\]

---

## 22. Brackets, Norms, and Inner Products

Use parentheses for function application:

\[
f(x),
\qquad
h(x,u).
\]

Use square brackets for expectation and probability operations:

\[
\mathbb{E}[X],
\qquad
\operatorname{Var}[X].
\]

Use angle brackets for inner products:

\[
\langle f,g\rangle_{\mathcal{H}}.
\]

Weighted finite-dimensional inner product:

\[
\langle x,y\rangle_M
=
x^\top M y.
\]

Weighted norm:

\[
\|x\|_M^2
=
x^\top M x.
\]

For covariance-weighted residuals:

\[
\|y-h(x)\|_{R^{-1}}^2
=
(y-h(x))^\top R^{-1}(y-h(x)).
\]

Use $\|\cdot\|_2$ for Euclidean vector norms and induced matrix norms when context is clear.

---

## 23. Recommended Compact Table

| Quantity | Recommended notation |
|---|---|
| continuous-time state | $x(t)$ |
| discrete-time state | $x_k$ |
| numerical snapshot | $\mathbf{x}_k$ |
| component of state | $x_i$ |
| input/output | $u(t), y(t)$ |
| data matrices | $\mathbf{X}_-,\mathbf{X}_+,\mathbf{U},\mathbf{Y}$ |
| dataset | $\mathcal{D}$ |
| discrete-time map | $F$ or $F_\theta$ |
| continuous-time vector field | $f$ or $f_\theta$ |
| unknown parameter | $\theta$ |
| operating parameter | $\mu$ |
| learned linear coefficient | $\xi$ |
| feature library | $\Theta$ |
| sampled feature matrix | $\mathbf{\Theta}$ |
| known dynamics | $f_{\mathrm{known}}$ |
| learned residual | $g_\theta$ |
| closure/constitutive law | $c_\theta$ |
| resolved state | $x_a$ |
| unresolved state | $x_b$ |
| learned latent state | $z$ |
| reduced coordinate | $a$ |
| observation map | $h$ |
| process noise | $w_k$ |
| measurement noise | $v_k$ |
| process covariance | $Q$ |
| measurement covariance | $R$ |
| filtered estimate | $\widehat{x}_{k|k}$ |
| smoothed estimate | $\widehat{x}_{k|N}$ |
| covariance estimate | $P_{k|k}$, $P_{k|N}$ |
| objective/loss | $\mathcal{J}$ |
| stage cost | $\ell$ |
| policy | $\pi_\theta$ |
| value function | $V^\pi$ |
| action-value function | $Q_{\mathrm{RL}}^\pi$ |
| model-error variable | $q_k$ |
| adjoint/costate | $p_k$ |
| manifold | $\mathcal{M}$ |
| tangent space | $T_x\mathcal{M}$ |
| invariant | $I_j(x)$ |
| energy | $E(x)$ |
| Hamiltonian | $H(q,p)$ |
| reduction basis | $V_r$ |
| test basis | $W_r$ |
| Hilbert space | $\mathcal{H}$ |
| trial/test spaces | $\mathcal{V},\mathcal{W}$ |

---

## 24. Short Rule of Thumb for Students

> Plain italic symbols describe mathematical variables in the model.  
> Bold symbols describe sampled numerical data or assembled arrays.  
> Calligraphic symbols describe operators, spaces, manifolds, datasets, or feasible sets.  
> Blackboard-bold symbols describe standard number spaces and expectations.  
> Subscripts index components, time steps, modes, or labels.  
> Parenthesized superscripts index trajectories, realizations, or algorithm iterations.  
> Hats denote estimates or learned quantities.  
> Roman labels in subscripts/superscripts denote methods, errors, or roles.

The most important distinctions are:

\[
x(t) \quad \text{mathematical state},
\qquad
x_k \quad \text{state at time step }k,
\qquad
\mathbf{x}_k \quad \text{stored snapshot},
\qquad
\mathbf{X} \quad \text{assembled data matrix}.
\]

and

\[
\theta \quad \text{unknown learned parameter},
\qquad
\mu \quad \text{operating/system parameter},
\qquad
z \quad \text{learned latent state},
\qquad
a \quad \text{projection-based reduced coordinate}.
\]
