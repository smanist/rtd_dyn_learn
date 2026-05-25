# Nonlinear and Ensemble State Estimation

When the state is only partially observed, we combine a dynamical model with
noisy measurements to infer the hidden trajectory. In the linear-Gaussian case,
that inference is analytically tractable. This chapter keeps the same
estimation goal but drops those linear-Gaussian assumptions. The result is not
one replacement for the Kalman filter, but a family of approximations that
trade off local accuracy, distributional fidelity, computational cost, and
robustness in high dimension.

The common starting point is the nonlinear discrete-time state-space model

```{math}
:label: eq:ch14-state-space

x_{k+1}=F(x_k,u_k)+w_k,
\qquad
y_k=h(x_k)+v_k,
```

with process noise $w_k$ and measurement noise $v_k$. As in the notation guide,
the filtered estimate is $\widehat{x}_{k|k}$, the predicted estimate is
$\widehat{x}_{k+1|k}$, and the smoothed estimate is $\widehat{x}_{k|N}$. The
main difficulty is that once either $F$ or $h$ is nonlinear, the conditional
distribution $p(x_k \mid y_{0:k})$ is usually no longer Gaussian and often
cannot be propagated in closed form.

## Why Nonlinearity Changes the Estimation Problem

In the linear-Gaussian case, the Kalman filter is exact because means and
covariances are sufficient statistics for the posterior. For
{eq}`eq:ch14-state-space`, nonlinear maps distort Gaussian densities, create
state-dependent uncertainty, and can make the posterior multi-modal or skewed.
That means every practical method must decide what information to preserve and
what information to discard.

Three approximation strategies dominate practice:

- Linearize the dynamics or observation map near the current estimate.
- Represent the uncertainty with deterministic sigma points or random samples.
- Replace the full posterior by an optimization problem posed over a finite
  estimation window.

Those choices lead to the extended Kalman filter, unscented Kalman filter,
ensemble methods, particle methods, and moving-horizon estimation.

## Extended Kalman Filtering and Smoothing

The extended Kalman filter (EKF) keeps the Kalman filter algebra but replaces
the nonlinear model by a first-order Taylor approximation around the current
estimate. Define the Jacobians

```{math}
:label: eq:ch14-jacobians

A_k
=
\left.\frac{\partial F}{\partial x}\right|_{(\widehat{x}_{k|k},u_k)},
\qquad
H_k
=
\left.\frac{\partial h}{\partial x}\right|_{\widehat{x}_{k+1|k}}.
```

The prediction step uses the nonlinear map itself for the mean and the
linearized map for the covariance:

```{math}
:label: eq:ch14-ekf-prediction

\widehat{x}_{k+1|k}=F(\widehat{x}_{k|k},u_k),
\qquad
P_{k+1|k}=A_k P_{k|k} A_k^\top + Q.
```

After observing $y_{k+1}$, the EKF forms the innovation

```{math}
:label: eq:ch14-ekf-innovation

\nu_{k+1}
=
y_{k+1}-h(\widehat{x}_{k+1|k}),
```

then updates the estimate with the Kalman-style gain

```{math}
:label: eq:ch14-ekf-update

K_{k+1}^{\mathrm{KF}}
=
P_{k+1|k} H_k^\top
\left(H_k P_{k+1|k} H_k^\top + R\right)^{-1},
```

```{math}
:label: eq:ch14-ekf-correction

\widehat{x}_{k+1|k+1}
=
\widehat{x}_{k+1|k}+K_{k+1}^{\mathrm{KF}} \nu_{k+1}.
```

The EKF is attractive because it is inexpensive and closely connected to local
stability analysis and sensitivity calculations. Its limitation is equally
clear: first-order linearization is only trustworthy when the uncertainty is
small enough that higher-order curvature does not matter over the support of
the posterior. Strong nonlinearities, poor initial conditions, or intermittent
observations can make the EKF biased or unstable.

The extended Kalman smoother uses the same local-linear viewpoint but revisits
past states after future data arrive. A backward pass improves
$\widehat{x}_{k|k}$ to $\widehat{x}_{k|N}$, which is useful when the goal is
trajectory reconstruction rather than online estimation.

## Unscented Kalman Filtering

The unscented Kalman filter (UKF) avoids explicit Jacobians. Instead, it
approximates a Gaussian posterior by a set of carefully chosen sigma points
whose sample mean and covariance match the current estimate. Those points are
pushed through the nonlinear maps, and the transformed cloud is used to
reconstruct predicted means and covariances.

If $\chi_k^{(i)}$ denotes the sigma points associated with
$(\widehat{x}_{k|k}, P_{k|k})$, then the prediction step applies
$F(\chi_k^{(i)},u_k)$ to every point and forms weighted moments. The same idea
is applied to the observation map $h$. Relative to the EKF, the UKF typically
captures second-order effects more accurately for comparable state dimension and
without symbolic differentiation. The tradeoff is that the number of sigma
points still scales with dimension, so the method becomes expensive in large
systems.

The UKF remains a Gaussian approximation. It can improve local nonlinear
propagation, but it still cannot represent a genuinely multi-modal posterior,
and it can still fail when nonlinearity is too strong over the uncertainty
region.

## Ensemble Kalman Methods

When the state dimension is large, explicitly propagating a dense covariance
matrix can be the dominant cost. Ensemble Kalman methods replace that matrix by
an empirical covariance built from ensemble members
$x_k^{(e)}$, $e=1,\ldots,N_{\mathrm{ens}}$.

The forecast step propagates each member through the nonlinear model:

```{math}
:label: eq:ch14-ensemble-forecast

x_{k+1|k}^{(e)}
=
F(x_{k|k}^{(e)},u_k)+w_k^{(e)}.
```

The empirical forecast mean is

```{math}
:label: eq:ch14-ensemble-mean

\overline{x}_{k+1|k}
=
\frac{1}{N_{\mathrm{ens}}}
\sum_{e=1}^{N_{\mathrm{ens}}} x_{k+1|k}^{(e)},
```

and the sample covariance is formed from the ensemble anomalies. The ensemble
Kalman filter (EnKF) then uses those empirical covariances in a Kalman-style
update. This keeps the computation tractable in many high-dimensional settings
because $N_{\mathrm{ens}}$ can be much smaller than the state dimension.

That efficiency comes with characteristic approximations:

- The posterior update is still Gaussian in spirit even though the forecast
  ensemble was propagated nonlinearly.
- Sampling error contaminates the empirical covariance when
  $N_{\mathrm{ens}}$ is too small.
- Spurious long-range correlations appear in high dimension when the ensemble
  does not adequately span the uncertainty directions.

Two standard remedies are localization and inflation. Localization damps
unreliable long-range covariance couplings, usually by tapering empirical
covariances according to physical distance or graph structure. Inflation
broadens the ensemble spread so that repeated assimilation does not collapse the
filter into an unrealistically confident estimate. Ensemble smoothers apply the
same empirical-covariance idea to reconstruct whole trajectories or to combine
all observations in a time window.

## Particle Filters

Particle filters are designed for the case where the posterior is not well
described by a single Gaussian family. They represent the filtering density by
weighted particles:

```{math}
:label: eq:ch14-particle-approximation

p(x_k \mid y_{0:k})
\approx
\sum_{i=1}^{N_{\mathrm{p}}} \omega_k^{(i)} \delta(x_k-x_k^{(i)}),
\qquad
\sum_{i=1}^{N_{\mathrm{p}}} \omega_k^{(i)} = 1.
```

Each particle is propagated through the nonlinear model, and the weights are
updated according to how well the predicted particle explains the new
observation. This gives particle filters the flexibility to represent skewed,
heavy-tailed, or multi-modal posteriors that Gaussian filters necessarily miss.

The main failure mode is particle degeneracy: after repeated updates, a small
number of particles can carry almost all of the weight. Resampling reduces that
collapse, but it also lowers particle diversity. In moderate to high
dimension, the number of particles required for stable performance can become
prohibitively large, which is why particle filters are usually preferred when
posterior non-Gaussianity is essential and the effective dimension is still
manageable.

## Moving-Horizon Estimation

Moving-horizon estimation (MHE) takes a different view. Rather than propagating
an approximate posterior recursively, it solves a finite-window optimization
problem for the recent state trajectory:

```{math}
:label: eq:ch14-mhe

\min_{x_{k-L:k}}
\;
\Gamma_{k-L}(x_{k-L})
+
\sum_{j=k-L}^{k-1}
\norm{x_{j+1}-F(x_j,u_j)}_{Q^{-1}}^2
+
\sum_{j=k-L}^{k}
\norm{y_j-h(x_j)}_{R^{-1}}^2.
```

Here $\Gamma_{k-L}$ is an arrival cost that summarizes information entering the
window from the past. MHE is attractive when the model is nonlinear, the system
must respect state or input constraints, or batch optimization tools are more
robust than recursive covariance propagation. Its cost is computational: each
assimilation step solves a nonlinear program, so real-time use depends on
window length, solver quality, and warm-starting.

MHE also makes the connection between state estimation and trajectory
optimization explicit. In a Gaussian setting, the objective in
{eq}`eq:ch14-mhe` is a maximum a posteriori estimator over the moving window.
Outside the Gaussian setting, the same formulation still provides a practical
way to combine dynamics, observations, and constraints.

## Choosing Among Nonlinear Estimators

Each method reflects a different approximation philosophy:

- EKF and extended smoothers are local and derivative-based.
- UKF methods are local but derivative-free, using sigma points to capture
  nonlinear propagation more accurately.
- EnKF and ensemble smoothers scale better to large systems by replacing dense
  covariances with empirical ones.
- Particle filters retain richer posterior structure at the price of sampling
  burden and degeneracy.
- MHE turns estimation into constrained optimization over a sliding window.

That comparison is not only algorithmic. It tells us what kinds of estimation
errors to expect. Gaussian filters can be overconfident when the true posterior
is skewed or multi-modal. Linearization methods can fail because the mean is a
poor expansion point. Ensemble methods can look accurate while hiding sampling
artifacts caused by too few members. Particle methods can appear flexible in
principle but collapse in dimension. MHE can be robust to constraints and
strong nonlinearities but may inherit local minima or solver sensitivity.

## Approximation Error and Failure Modes

The central lesson of nonlinear state estimation is that approximation error is
structural, not incidental. An estimator can be internally consistent with its
own approximation class and still be wrong about the true posterior. In
practice, three questions should always be asked.

First, is the posterior close enough to Gaussian that mean-covariance methods
are credible? If not, a beautifully tuned EKF, UKF, or EnKF may still miss the
important uncertainty structure.

Second, is the effective dimension low enough for the chosen representation?
Particle methods struggle with weight collapse in high dimension, while
ensemble methods need localization and inflation because finite ensembles do
not estimate full covariances reliably.

Third, are the dominant errors coming from estimation or from the dynamical
model itself? No filter can repair a systematically biased model merely by
reweighting observations. In that sense, nonlinear estimation remains tightly
coupled to the model-learning questions of earlier chapters.

For learning dynamics from data, these methods are valuable not only as
standalone estimators but also as subroutines inside broader identification and
assimilation pipelines. They let us infer hidden trajectories, compare model
classes under partial observations, and quantify how much of the remaining
error comes from state uncertainty versus model misspecification.
