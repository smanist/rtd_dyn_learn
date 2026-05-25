# Continuous-Time Identification and Derivative-Free Formulations

Continuous-time modeling asks for the vector field itself rather than only a
sample-to-sample update rule. That is attractive when we care about
interpretable rates, equilibria, stability, or how the model behaves under a
different time step or solver. It is also harder than discrete-time map
learning because the data arrive as sampled trajectories, not as direct
measurements of $\dot{x}(t)$.

This chapter studies that tension. We want to identify a continuous-time model
from sampled data, but naive differentiation often amplifies noise so strongly
that the regression target is less reliable than the state measurements
themselves. The main alternatives are to smooth before differentiating, or to
avoid explicit derivatives by fitting integral, weak-form, collocation, or
multiple-shooting constraints instead.

## Continuous-Time Models and Their Sampled Counterparts

The continuous-time identification problem starts from a vector field

```{math}
:label: eq:ch05-vector-field

\dot{x}(t)=f(x(t),u(t);\theta),
\qquad
x(t)\in \mathbb{R}^n,
\quad
u(t)\in \mathbb{R}^m.
```

If the same system is observed only at sample times $t_k$, then the data do
not directly reveal the right-hand side of
{eq}`eq:ch05-vector-field`. They reveal points on trajectories:
$x_k=x(t_k)$ mathematically, or stored samples $\mathbf{x}_k$ numerically.
Over one sampling interval, the vector field induces a flow map

```{math}
:label: eq:ch05-flow-map

x_{k+1}=F_{\Delta t_k}(x_k,u_{[t_k,t_{k+1}]};\theta),
\qquad
\Delta t_k=t_{k+1}-t_k.
```

Equation {eq}`eq:ch05-flow-map` is exact even when we do not know the closed
form of $F_{\Delta t_k}$. This already explains the distinction between map
learning and vector-field learning. A discrete-time learner fits the induced
map at the observed sample rate. A continuous-time learner instead tries to
recover the generator that produces those transitions.

That distinction matters when the sampling interval changes, when data are
irregularly sampled, or when the main goal is mechanistic interpretation. A
discrete-time map can predict well on the training grid and still obscure the
underlying rate law. A continuous-time model separates the system from the
chosen integrator, but it pays for that flexibility by making identification
more sensitive to noise and discretization choices.

## The Direct Route: Differentiate, Then Regress

The most literal strategy is to estimate derivatives from sampled state data
and regress those estimates against candidate features in $f$. With uniform
sampling, the simplest approximation is the forward difference

```{math}
:label: eq:ch05-forward-difference

\widehat{\dot{\mathbf{x}}}_k
=
\frac{\mathbf{x}_{k+1}-\mathbf{x}_k}{\Delta t}.
```

Central differences or higher-order formulas reduce truncation error for smooth
signals, but they do not remove the basic statistical problem. If the measured
sample is $\mathbf{x}_k=x_k+\eta_k$, then differencing turns the noise into

```{math}
:label: eq:ch05-difference-noise

\widehat{\dot{\mathbf{x}}}_k
=
\frac{x_{k+1}-x_k}{\Delta t}
+
\frac{\eta_{k+1}-\eta_k}{\Delta t}.
```

The second term scales like $1/\Delta t$, so taking derivatives from dense,
noisy samples can be worse than working with a coarser but smoother signal.
Derivative regression is therefore attractive mainly when the state is measured
cleanly, the sampling rate is appropriate, and the vector field is simple
enough that approximation error dominates noise amplification.

Once a derivative estimate is available, continuous-time identification often
looks like ordinary regression:

```{math}
:label: eq:ch05-derivative-loss

\min_{\theta}
\sum_{k=0}^{N-1}
\left\|
\widehat{\dot{\mathbf{x}}}_k
-f(\mathbf{x}_k,\mathbf{u}_k;\theta)
\right\|^2.
```

Equation {eq}`eq:ch05-derivative-loss` is conceptually simple, but its quality
depends more on the derivative estimator than on the optimizer. In practice,
many failures blamed on the model class are actually failures of numerical
differentiation.

## Smoothing Before Differentiation

When direct finite differences are too noisy, the next idea is to fit a smooth
surrogate trajectory and differentiate that surrogate instead of the raw
samples. Common choices include local polynomial fits, smoothing splines, and
low-pass filtering when the frequency content is well separated.

This is not a purely technical preprocessing step. Smoothing imposes an
implicit prior about the bandwidth and regularity of the true state trajectory.
Too little smoothing leaves derivative estimates dominated by noise; too much
smoothing erases transient features and biases the inferred vector field. The
right amount depends on the learning goal:

- For parameter estimation in a known model, moderate bias may be acceptable if
  the smoother removes variance that would otherwise overwhelm the inverse
  problem.
- For equation discovery, oversmoothing can suppress small but real nonlinear
  terms and change the apparent sparsity pattern.
- For stiff or multi-scale systems, a single smoothing scale may distort the
  fast modes that the model is supposed to capture.

Smoothing before differentiation is often the fastest workable baseline, but it
should be viewed as a compromise rather than a full solution. The more hostile
the measurement noise or sampling irregularity, the more useful derivative-free
formulations become.

## Integral Matching

Integral formulations use the fact that the vector field generates increments in
state over a time window:

```{math}
:label: eq:ch05-integral-matching

x(t_{k+1})-x(t_k)
=
\int_{t_k}^{t_{k+1}} f(x(t),u(t);\theta)\,dt.
```

This replaces a derivative target with a state increment, which is usually much
less sensitive to high-frequency measurement noise. The data enter through the
left-hand side as observed differences, while the model enters through an
integral constraint on the right-hand side. If the interval is short and the
trajectory within the interval can be approximated, one can fit $\theta$
without ever forming $\widehat{\dot{\mathbf{x}}}_k$ explicitly.

The gain in robustness comes with a tradeoff. Integral matching blurs local
errors over time windows, so it can be less sensitive to rapidly varying
dynamics unless the quadrature and interpolation inside each interval are
carefully chosen. Still, for noisy data the bias-variance tradeoff is often
much better than in derivative regression because integration averages noise
instead of amplifying it.

Integral matching is especially natural when the input $u(t)$ is known between
samples, when the trajectories are smooth enough to interpolate, or when the
model class is linear in unknown parameters and the integral can be expressed
as a regression in accumulated features.

## Weak-Form Identification

Weak-form methods push the same idea further by testing the differential
equation against smooth functions rather than matching it pointwise. Let
$\varphi(t)$ be a test function that is smooth on $[t_0,t_1]$. Then

```{math}
:label: eq:ch05-weak-residual

\int_{t_0}^{t_1}
\varphi(t)\left[\dot{x}(t)-f(x(t),u(t);\theta)\right]dt = 0.
```

After integration by parts, the derivative moves from the noisy state to the
chosen test function:

```{math}
:label: eq:ch05-weak-form

\varphi(t_1)x(t_1)-\varphi(t_0)x(t_0)
-
\int_{t_0}^{t_1}\dot{\varphi}(t)x(t)\,dt
-
\int_{t_0}^{t_1}\varphi(t)f(x(t),u(t);\theta)\,dt
=0.
```

Equation {eq}`eq:ch05-weak-form` is the key robustness mechanism. Instead of
differentiating noisy data, we differentiate the known test function
$\varphi(t)$. By choosing families of compactly supported, oscillatory, or
polynomial test functions, weak-form methods can localize information in time,
filter noise, and produce regression constraints that are often much better
conditioned than pointwise derivative targets.

Weak-form identification is especially useful when derivatives are unusable,
when data are irregularly sampled, or when the model is linear in unknown
coefficients and the weak constraints reduce to linear algebra. The method is
not free of choices, however: the test space, quadrature rule, interpolation
scheme, and window length all affect what information is emphasized.

## Collocation and Trajectory-Based Formulations

A different derivative-free viewpoint is to treat the state trajectory itself
as an optimization variable. Instead of insisting that the observed samples lie
exactly on a differentiable curve whose derivative is directly estimated, we
fit a smooth approximate trajectory $\tilde{x}(t)$ together with model
parameters and penalize the defect in the differential equation at selected
collocation points $\tau_{k,j}$:

```{math}
:label: eq:ch05-collocation-defect

\delta_{k,j}
=
\ddf{\tilde{x}}{t}(\tau_{k,j})
-f(\tilde{x}(\tau_{k,j}),u(\tau_{k,j});\theta).
```

The optimization then balances trajectory fit to the measurements against small
defects $\delta_{k,j}$. This can be seen as a bridge between system
identification and numerical optimal control: the learned dynamics must explain
the data while also supporting a smooth trajectory that is dynamically
consistent.

Collocation is attractive because it works directly with sampled observations,
handles irregular sampling naturally, and avoids forming derivatives of the raw
data. It also exposes an important source of error: discretization mismatch. If
the collocation grid, interpolation basis, or quadrature rule is too coarse for
the true dynamics, then part of the residual reflects the chosen numerical
scheme rather than a modeling error in $f$.

## Multiple Shooting

Multiple shooting introduces separate initial values on many short intervals and
propagates the model across each interval with a numerical solver. If $z_k$
denotes the shooting state at time $t_k$, then continuity between intervals is
enforced through constraints of the form

```{math}
:label: eq:ch05-multiple-shooting

z_{k+1}
-\Phi_{t_k,t_{k+1}}(z_k,u_{[t_k,t_{k+1}]};\theta)
=0,
```

where $\Phi_{t_k,t_{k+1}}$ is the numerical flow produced by the current model.
Compared with single shooting, this reduces long-horizon sensitivity because
errors do not compound over the entire trajectory before the optimizer can
correct them.

For identification, multiple shooting is useful when the model is stiff, when
the initial state is uncertain, or when long rollouts make direct trajectory
matching unstable. It also makes clear that continuous-time learning is never
completely separated from discretization: the solver and time mesh become part
of the estimation procedure.

## Choosing Between Map Learning and Vector-Field Learning

At this point the central tradeoff is clear. Vector-field learning seeks the
continuous-time law in {eq}`eq:ch05-vector-field`, while map learning fits the
induced sampled evolution in {eq}`eq:ch05-flow-map`. Neither dominates in every
setting.

Vector-field learning is preferable when:

- the scientific goal is to interpret mechanisms in continuous time,
- the sampling interval may change across datasets or deployment settings,
- the downstream task depends on equilibria, stability, or sensitivity of the
  rate law itself.

Map learning is often preferable when:

- the data are noisy enough that differentiation or interpolation is fragile,
- prediction on the observed sample grid matters more than continuous-time
  interpretation,
- solver choice and discretization are part of the operational definition of
  the model anyway.

Derivative-free continuous-time formulations occupy the middle ground. They keep
the vector-field viewpoint while avoiding the most unstable part of the direct
approach, namely differentiation of noisy measurements.

## Main Takeaways

Continuous-time identification is not just discrete-time regression with dots
added to the variables. The central difficulty is that sampled data do not
directly provide the vector-field values we want to learn. Finite differences
create that target artificially, but they also amplify measurement noise and
sampling artifacts.

That is why integral matching, weak forms, collocation, and multiple shooting
are core tools rather than technical variations. They replace fragile
pointwise derivative information with time-averaged, test-function-weighted, or
trajectory-constrained formulations that are usually more robust on real data.
The practical lesson is simple: if derivative estimates look unreliable, change
the formulation before changing the optimizer.
