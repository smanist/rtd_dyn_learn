# Stochastic Dynamics and Probabilistic Learning

Real systems rarely evolve as perfectly deterministic maps or vector fields.
Even when the governing equations are known in principle, unresolved forcing,
thermal fluctuations, unmodeled disturbances, random inputs, and limited data
make uncertainty part of the dynamics themselves. A probabilistic treatment is
therefore not an optional add-on to dynamics learning. It changes what the
model is, what the data can identify, and what counts as a good prediction.

This chapter studies dynamics models whose outputs are distributions over future
states and observations rather than single trajectories. The main goals are to
separate different sources of uncertainty, estimate both mean behavior and
stochastic variability, and produce predictive distributions that are
calibrated rather than merely sharp. That perspective matters for forecasting,
state estimation, robustness analysis, and downstream decision making under
uncertainty.

## Stochastic State-Space Models

The discrete-time stochastic state-space model extends deterministic dynamics by
including process noise:

```{math}
:label: eq:ch20-discrete-stochastic-dynamics

x_{k+1}=F_\theta(x_k,u_k)+w_k,
\qquad
y_k=h(x_k)+v_k.
```

Here $w_k$ represents random forcing or model discrepancy in the state-update
law, while $v_k$ represents measurement noise in the observation process. In a
probabilistic model, the learned object is not only the map $F_\theta$ but also
the distributional assumptions attached to $w_k$ and $v_k$. Common choices are
Gaussian models such as

```{math}
:label: eq:ch20-gaussian-noise

w_k \sim \mathcal{N}(0,Q_\theta),
\qquad
v_k \sim \mathcal{N}(0,R_\theta),
```

but the same framework can accommodate state-dependent, heavy-tailed, or
non-Gaussian noise models.

In continuous time, stochastic forcing is modeled through a stochastic
differential equation (SDE):

```{math}
:label: eq:ch20-sde

dx(t)=f_\theta(x(t),u(t))\,dt+G_\theta(x(t),u(t))\,dW_t.
```

The drift term $f_\theta$ describes the mean local tendency, while the
diffusion matrix $G_\theta$ scales the increments of a Wiener process $W_t$.
The deterministic model is recovered when the diffusion vanishes. In this
sense, stochastic dynamics learning generalizes vector-field learning by
requiring us to infer both systematic motion and random fluctuation.

## What Kind of Uncertainty Is Being Modeled?

Probabilistic learning is only meaningful if we are explicit about what the
randomness stands for. Several distinct uncertainties are often mixed together.

### Process Noise

Process noise describes randomness in the latent evolution law itself. In
{eq}`eq:ch20-discrete-stochastic-dynamics`, it is the term $w_k$. It can encode
genuine physical randomness, unresolved fast variables, aggregate disturbance
inputs, or deliberate model-error slack in a weak-constraint formulation. When
process noise is present, repeated runs from the same state and input need not
follow identical trajectories.

### Measurement Noise

Measurement noise is uncertainty in the observation device rather than the
dynamics. In {eq}`eq:ch20-discrete-stochastic-dynamics`, it is represented by
$v_k$. This distinction matters because state estimation can often average down
measurement noise over time, whereas process noise changes the distribution of
the latent trajectory itself.

### Model Error

Model error arises when the chosen drift or map class is structurally
inadequate. In practice, process-noise terms are often used to absorb this
misspecification, but the interpretation is different. A large fitted $Q_\theta$
may indicate truly stochastic physics, poor state resolution, or a deterministic
model class that is missing important mechanisms.

### Epistemic Uncertainty

Epistemic uncertainty comes from limited information about the parameters or the
model class. Even if the underlying system were deterministic conditional on
$\theta$, we may still be uncertain about the correct value of $\theta$ after
seeing finite data. This motivates Bayesian dynamics models and posterior
predictive distributions. Aleatoric randomness concerns variability in future
realizations given the model; epistemic uncertainty concerns what model is
plausible given the data. Good probabilistic learning keeps those roles
conceptually separate even when they interact numerically.

## Learning Drift and Diffusion

For deterministic systems, learning often means estimating a single map
$F_\theta$ or vector field $f_\theta$. In stochastic systems, we additionally
need to model how uncertainty enters the dynamics.

### Discrete-Time View

If the sample interval is fixed, one may parameterize the conditional
distribution directly:

```{math}
:label: eq:ch20-conditional-transition

p_\theta(x_{k+1}\mid x_k,u_k).
```

Under Gaussian assumptions, this transition density is determined by a mean
function $F_\theta(x_k,u_k)$ and a covariance $Q_\theta(x_k,u_k)$. Then
learning can be posed as conditional density estimation, not only regression of
$x_{k+1}$ on $(x_k,u_k)$. Heteroscedastic models, where the covariance depends
on the state and input, are especially useful when uncertainty grows in some
regions of phase space more than others.

### Continuous-Time View

For SDEs of the form {eq}`eq:ch20-sde`, drift estimation and diffusion
estimation are separate but coupled tasks. The drift controls the mean
increment:

```{math}
:label: eq:ch20-drift-mean

\mathbb{E}[x(t+\Delta t)-x(t)\mid x(t)]
\approx
f_\theta(x(t),u(t))\,\Delta t,
```

while the diffusion controls the local covariance:

```{math}
:label: eq:ch20-diffusion-covariance

\operatorname{Cov}(x(t+\Delta t)-x(t)\mid x(t))
\approx
G_\theta(x(t),u(t))G_\theta(x(t),u(t))^\top \Delta t.
```

These approximations show why estimating diffusion is harder than fitting a
mean field alone: the data must resolve second-order fluctuation structure on
the sampling scale, and coarse sampling can confound drift, diffusion, and
numerical discretization error.

## Likelihoods for Stochastic Dynamics

A central advantage of probabilistic models is that they provide a likelihood.
For a fully observed Markov model with transition density
$p_\theta(x_{k+1}\mid x_k,u_k)$, the trajectory likelihood factors as

```{math}
:label: eq:ch20-markov-likelihood

p_\theta(x_{0:N}\mid u_{0:N-1})
=
p(x_0)\prod_{k=0}^{N-1} p_\theta(x_{k+1}\mid x_k,u_k).
```

With partial observations, we instead use the latent-state state-space
likelihood

```{math}
:label: eq:ch20-state-space-likelihood

p_\theta(x_{0:N},y_{0:N}\mid u_{0:N-1})
=
p(x_0)\prod_{k=0}^{N-1} p_\theta(x_{k+1}\mid x_k,u_k)
\prod_{k=0}^{N} p_\theta(y_k\mid x_k).
```

This factorization is the basis for maximum likelihood, expectation-maximization,
Bayesian inference, filtering, and smoothing. It also clarifies why stochastic
dynamics learning and state estimation are tightly connected: under partial
observation, likelihood evaluation already requires integrating over hidden
trajectories.

For SDEs, exact continuous-time likelihoods are rarely available except in
special cases. In practice, one often works with discretized approximations
such as Euler-Maruyama transition densities, approximate innovation
likelihoods, or simulation-based objectives. The key point is not the specific
formula but the modeling shift: the loss is now a log-density score for random
outcomes, not only a squared error on a mean prediction.

## Density Evolution and the Fokker-Planck View

A stochastic dynamical system induces evolution not only of individual
trajectories but also of probability densities. If $\rho(x,t)$ denotes the
density of the state under the SDE {eq}`eq:ch20-sde`, then its time evolution
is governed by the Fokker-Planck equation:

```{math}
:label: eq:ch20-fokker-planck

\partial_t \rho
=
-\nabla \cdot \left(f_\theta \rho\right)
+ \frac{1}{2}
\sum_{i,j}
\partial_{x_i}\partial_{x_j}
\left(
\left[G_\theta G_\theta^\top\right]_{ij}\rho
\right).
```

This equation expresses conservation of probability mass under deterministic
transport and diffusive spreading. It is useful conceptually even when we do
not solve it directly. Deterministic learning focuses on tracking trajectories;
stochastic learning must also ask whether the model reproduces stationary
densities, transition densities, escape times, and other distributional
quantities. For long-horizon behavior, these may matter more than one-step mean
error.

## Gaussian-Process and Bayesian Dynamics Models

Probabilistic learning can place uncertainty either in the state transition
noise, in the model parameters, or in the unknown function itself.

### Gaussian-Process State-Space Models

A Gaussian-process (GP) dynamics model treats the unknown map or drift as a
random function. In discrete time, one may write

```{math}
:label: eq:ch20-gp-model

x_{k+1}=F(x_k,u_k)+w_k,
\qquad
F \sim \mathcal{GP}(m,K),
```

where $m$ is a prior mean function and $K$ is a covariance kernel. This view
is attractive when uncertainty should increase away from training data and when
sample efficiency is more important than scaling to extremely large datasets.
GP state-space models also highlight the difference between uncertainty about
the underlying dynamics function and process noise acting within the dynamics.

### Bayesian Parameter Learning

If the model form is fixed but the parameter vector is unknown, a Bayesian
approach places a prior on $\theta$ and updates it using the data likelihood:

```{math}
:label: eq:ch20-bayes-posterior

p(\theta \mid \mathcal{D})
\propto
p(\mathcal{D}\mid \theta)\,p(\theta).
```

Predictions then average over parameter uncertainty rather than plugging in one
point estimate:

```{math}
:label: eq:ch20-posterior-predictive

p(x_{k+1}\mid x_k,u_k,\mathcal{D})
=
\int p(x_{k+1}\mid x_k,u_k,\theta)\,p(\theta\mid\mathcal{D})\,d\theta.
```

This posterior predictive distribution is the natural object for decision
making under uncertainty. It widens when the data are sparse or ambiguous and
collapses when the parameters become well identified.

## Uncertainty Propagation and Calibration

Probabilistic models are useful only if their predictive distributions behave
sensibly under rollout.

### Uncertainty Propagation

Even if the one-step uncertainty is modest, repeated propagation can create
wide predictive spreads, multimodal distributions, or concentration onto
attractors. In nonlinear systems, the future distribution is not determined by
the mean state alone. Monte Carlo rollout, sigma-point approximations, moment
closure, ensemble methods, and density solvers are all ways to propagate
uncertainty through nonlinear dynamics.

### Calibration

Calibration asks whether predicted probabilities match empirical frequencies. A
calibrated model should assign approximately $90\%$ coverage intervals that
contain the truth about $90\%$ of the time. A model can have low mean squared
error and still be poorly calibrated if it systematically underestimates or
overestimates its uncertainty. Proper scoring rules such as log likelihood and
continuous ranked probability score are therefore often more informative than
pointwise regression losses when evaluating probabilistic forecasts.

### Stochastic Stability

Stability in stochastic systems is richer than in deterministic ones. One may
care about bounded moments, recurrence, invariant measures, mean-square
stability, or almost-sure behavior. For learned models, this means that a
useful validation set should probe not only short-horizon prediction but also
whether the model produces realistic long-run random behavior, including
stationary variance levels and rare-event tendencies when those are relevant.

## Practical Modeling Choices

Several recurring decisions shape a stochastic dynamics-learning pipeline.

- Decide whether randomness is intrinsic to the system, a surrogate for model
  error, or mainly a statement about parameter uncertainty.
- Match the loss to the probabilistic target: pointwise least squares is not
  enough if the goal is a calibrated predictive density.
- Check whether the sampling interval is fine enough to distinguish drift from
  diffusion.
- Use distributional validation, not only one-step mean prediction error.
- Treat uncertainty estimates skeptically when the model class is strongly
  misspecified or the data explore only a narrow regime.

These points are easy to understate. A poorly interpreted noise model can make
the learner look probabilistic while hiding deterministic bias, and an
uncalibrated Bayesian or GP model can look sophisticated while still giving
misleading confidence statements.

## Summary

Stochastic dynamics and probabilistic learning extend classical dynamics
identification from predicting trajectories to predicting distributions. The
basic discrete-time model {eq}`eq:ch20-discrete-stochastic-dynamics` and SDE
model {eq}`eq:ch20-sde` force us to distinguish drift from diffusion, process
noise from measurement noise, and aleatoric randomness from epistemic
uncertainty. Likelihood-based learning, Fokker-Planck density evolution,
Gaussian-process state-space models, Bayesian posterior prediction, and
calibration-oriented validation all follow from that shift.

The main lesson is that uncertainty is part of the dynamical object being
learned, not only a confidence interval added afterward. A strong stochastic
model should explain how randomness enters, propagate it coherently through
time, and remain calibrated when used for forecasting, estimation, or control.
