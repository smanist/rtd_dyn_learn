# Black-Box Nonlinear Dynamics Models

When little model structure is known, a flexible nonlinear approximator can be
the most direct way to learn dynamics from data. In this setting the learner is
not asked to identify a known parameter vector or a small residual correction.
Instead, the map or vector field itself is represented by a broad function
class such as a neural network, a recurrent model, or a kernel method.

That flexibility is useful, but it changes the failure modes. A black-box
model can fit one-step transitions very well while producing poor long-horizon
rollouts. It can interpolate within the training regime yet extrapolate badly
outside it. It can also inherit numerical artifacts from the solver used during
training. The central question in this chapter is therefore not only how to fit
flexible nonlinear models, but how to do so in a way that respects the
dynamical task.

## What "Black-Box" Means in Dynamics Learning

In these notes, a black-box model is one in which most of the structure is
learned from data rather than prescribed by physics. That does not mean the
model is assumption free. The assumptions simply move from known equations into
the choice of architecture, state representation, regularization, and training
loss.

The standard discrete-time formulation is

```{math}
:label: eq:ch10-discrete-black-box

x_{k+1}=F_\theta(x_k,u_k;\mu),
```

while the continuous-time formulation is

```{math}
:label: eq:ch10-continuous-black-box

\dot{x}(t)=f_\theta(x(t),u(t);\mu).
```

Here $\theta$ denotes learned parameters and $\mu$ denotes a known operating
parameter when the same model family must cover multiple regimes. The choice
between {eq}`eq:ch10-discrete-black-box` and
{eq}`eq:ch10-continuous-black-box` is already a modeling decision: a map
absorbs the sample interval into the learned transition, while a vector field
separates the dynamics from the numerical integrator.

## Discrete-Time Neural Maps

If the data are sampled on a fixed grid and the main task is prediction at that
grid, then learning a discrete-time map is often the most direct route.
Feedforward neural networks can be used to represent $F_\theta$ in
{eq}`eq:ch10-discrete-black-box` with the current state, input, and operating
parameter as features.

One common parameterization uses a residual update,

```{math}
:label: eq:ch10-resnet-map

x_{k+1}=x_k+N_\theta(x_k,u_k;\mu),
```

where $N_\theta$ learns the increment over one sample. This is attractive when
the sampling interval is small, because the identity term in
{eq}`eq:ch10-resnet-map` builds in the prior belief that successive states are
nearby. It also resembles a forward Euler step, which can make optimization
easier than learning an unconstrained map from scratch.

The tradeoff is that a map learned only on one-step transitions is not
automatically reliable under repeated composition. Even a small bias in
$F_\theta$ compounds over time, so a model that looks accurate on pairs
$(\mathbf{x}_k,\mathbf{x}_{k+1})$ may still drift to unrealistic trajectories.

## Continuous-Time Neural Vector Fields

If the scientific object of interest is the underlying rate law, then a neural
vector field may be the better black-box choice:

```{math}
:label: eq:ch10-neural-ode

x(t_{k+1})
=
x(t_k)
+
\int_{t_k}^{t_{k+1}}
f_\theta(x(t),u(t),t;\mu)\,\dd t.
```

This is the neural-ODE viewpoint: the model is the vector field $f_\theta$, and
predictions are obtained by numerical integration. The main benefit is that the
same learned dynamics can be evaluated at different step sizes or queried for
continuous-time properties such as equilibria and local stability.

The main cost is that training now depends on the solver. If the vector field
is stiff, highly nonlinear, or poorly scaled, the integrator may dominate
runtime and may even bias what parameter values are reachable by gradient-based
optimization. In practice, solver tolerances, event handling, and state
scaling become part of the model class rather than a harmless implementation
detail.

## Recurrent and Latent Black-Box Models

When the measured variables are not themselves Markovian, black-box learning
often introduces a hidden state. One option is a recurrent architecture that
updates an internal memory state:

```{math}
:label: eq:ch10-recurrent-model

z_{k+1}=G_\theta(z_k,\mathbf{u}_k,\mathbf{y}_k),
\qquad
\widehat{y}_k=D_\theta(z_k).
```

Here the latent coordinate $z_k$ summarizes the relevant history. This lets the
model absorb unmeasured state, delays, and other non-Markovian effects without
writing an explicit closure law.

A related idea is to separate encoding, latent dynamics, and decoding:

```{math}
:label: eq:ch10-latent-model

z_k=E_\theta(\mathbf{y}_{0:k}),
\qquad
z_{k+1}=G_\theta(z_k,\mathbf{u}_k),
\qquad
\widehat{y}_k=D_\theta(z_k).
```

In a latent neural ODE, the middle equation is replaced by a continuous-time
latent vector field. These models are useful when the physical state is
high-dimensional or only partially observed, but they give up direct
interpretability because the latent coordinate need not correspond to a
physical variable.

## Gaussian-Process and Kernel Dynamics Models

Not every black-box model is neural. Gaussian-process and kernel methods define
a flexible nonlinear model together with an uncertainty measure. A typical
discrete-time statement is

```{math}
:label: eq:ch10-gp-model

x_{k+1}\mid x_k,u_k,\mathcal{D}
\sim
\mathcal{N}\bigl(\bar{F}(x_k,u_k),\Sigma_F(x_k,u_k)\bigr),
```

where the predictive mean $\bar{F}$ and covariance $\Sigma_F$ are determined by
the data and the chosen kernel. Compared with a neural network, this approach
usually provides better-calibrated local uncertainty and can work well in
small-data regimes.

The limitation is scalability and long-horizon behavior. Kernel methods become
expensive as the dataset grows, and predictive uncertainty does not by itself
prevent physically implausible rollouts. As with neural models, the dynamical
question is still whether repeated forecasts remain trustworthy.

## One-Step Loss Versus Trajectory Loss

The objective used to train a black-box dynamics model strongly influences what
the model learns. A one-step objective fits local transitions:

```{math}
:label: eq:ch10-one-step-loss

\mathcal{J}_{\mathrm{train}}^{\mathrm{1step}}(\theta)
=
\sum_{k=0}^{N-1}
\norm{
\mathbf{x}_{k+1}
-F_\theta(\mathbf{x}_k,\mathbf{u}_k;\mu)
}^2.
```

This is statistically convenient and often easy to optimize, but it only asks
for local accuracy.

A rollout objective instead measures the error after repeated simulation:

```{math}
:label: eq:ch10-rollout-loss

\widetilde{x}_{k+1}=F_\theta(\widetilde{x}_k,\mathbf{u}_k;\mu),
\qquad
\widetilde{x}_0=\mathbf{x}_0,
\qquad
\mathcal{J}_{\mathrm{train}}^{\mathrm{roll}}(\theta)
=
\sum_{k=0}^{N}
\norm{\mathbf{x}_k-\widetilde{x}_k}^2.
```

The rollout objective is closer to the true forecasting task, but it is harder
to optimize because errors compound through time. In practice many workflows
combine the two: one-step training for initialization, followed by short- or
long-horizon rollout refinement.

## Stability, Solver Effects, and Generalization

Black-box models are judged not only by interpolation error, but by their
behavior under repeated use. Three issues matter especially.

### Stability and Regularization

A flexible function approximator can create unstable dynamics even when the
training trajectories are bounded. For that reason, many black-box methods add
regularization or architectural constraints such as spectral normalization,
contractive penalties, or energy-based parameterizations. A generic regularized
objective can be written as

```{math}
:label: eq:ch10-regularized-loss

\mathcal{J}_{\mathrm{train}}(\theta)
=
\mathcal{J}_{\mathrm{data}}(\theta)
+
\lambda_{\mathrm{reg}}\mathcal{J}_{\mathrm{reg}}(\theta).
```

The role of $\mathcal{J}_{\mathrm{reg}}$ is not merely to prevent overfitting
in the static regression sense. It is often used to discourage exploding
Jacobians, unbounded energy growth, or trajectories that violate basic
qualitative constraints.

### Solver Dependence and Stiffness

For continuous-time models, the learned behavior depends on how the vector
field is integrated. Two parameter values may induce nearly identical one-step
fits under a coarse solver while producing different trajectories under a finer
solver. Stiff systems add another layer of difficulty because training may push
the model toward vector fields that are accurate but numerically expensive.

This means that solver choice, step size, and tolerance should be treated as
part of model validation. A neural ODE that trains only under one specific
integrator may have learned the solver as much as the dynamics.

### Extrapolation and Rollout Error

A black-box model is often weakest when asked to move outside the region where
the training data are dense. Short-term interpolation accuracy does not imply
robust extrapolation to new initial conditions, new inputs, or longer horizons.
For dynamical systems, this gap usually appears as large rollout error
$e_{\mathrm{roll}}$, incorrect attractors, or spurious instability.

That is why validation should include multi-step simulation, regime changes,
and qualitative checks on the geometry of the trajectories rather than only
pointwise prediction error.

## Choosing Among Black-Box Model Classes

There is no universally best black-box representation. Discrete-time neural
maps are simple and effective when the sampling grid is fixed and the task is
short- to medium-horizon prediction. Continuous-time neural vector fields are
better when continuous-time interpretation or irregular sampling matters.
Recurrent and latent models help when the observed variables are not Markovian.
Gaussian-process and kernel models are appealing when uncertainty quantification
and small-data efficiency matter more than raw scalability.

The important comparison is therefore not "which architecture is most
expressive?" but "which architecture matches the data, the observation process,
and the downstream task with the fewest hidden failure modes?"

## Summary

Black-box nonlinear dynamics models replace known structure with flexible
function classes, but they do not eliminate modeling assumptions. The main
choices are whether to learn a discrete-time map or a continuous-time vector
field, whether hidden state should be represented recurrently or through an
explicit latent model, and whether uncertainty should be modeled directly as in
Gaussian-process approaches.

The main caution is that local fit is not enough. In dynamics learning, the
real test is rollout behavior, solver robustness, and generalization beyond the
training trajectories.
