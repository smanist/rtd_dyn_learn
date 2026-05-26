# Classical Linear Gaussian Filtering and Smoothing

Chapter 12 reframed dynamics learning under noisy or partial observations as a
state-estimation problem. In the linear Gaussian setting, that viewpoint leads
to closed-form estimators. This chapter studies the classical case because it
is the clearest place to see the main ideas: recursive prediction, measurement
update, covariance propagation, smoothing with future data, and likelihood
evaluation from innovations.

The chapter is not only about computing $\widehat{x}_{k|k}$ and
$\widehat{x}_{k|N}$. It also explains why Kalman methods sit naturally beside
least squares and system identification. The filter is a sequential weighted
least-squares estimator, the smoother is its batch counterpart over an entire
trajectory, and the innovation likelihood turns state estimation into a
parameter-learning objective.

## The Linear Gaussian State-Space Model

We work with the discrete-time linear state-space model

```{math}
:label: eq:ch13-linear-gaussian-model

x_{k+1}=Ax_k+Bu_k+w_k,
\qquad
y_k=Cx_k+Du_k+v_k,
```

where $x_k\in \mathbb{R}^n$ is the hidden state, $u_k\in \mathbb{R}^m$ is a
known input, and $y_k\in \mathbb{R}^p$ is the measured output. The process
noise and measurement noise are modeled as independent Gaussian sequences,

```{math}
:label: eq:ch13-noise-model

w_k\sim \mathcal{N}(0,Q),
\qquad
v_k\sim \mathcal{N}(0,R),
```

with $Q\succeq 0$ and $R\succ 0$. We also assume a Gaussian prior on the
initial state,

```{math}
:label: eq:ch13-initial-prior

x_0 \sim \mathcal{N}(\widehat{x}_{0|-1},P_{0|-1}).
```

Because the dynamics, observation model, and prior are all Gaussian, every
conditional distribution of interest remains Gaussian. The filtering and
smoothing problem is therefore reduced to propagating means and covariances.

## The Kalman Filter

The Kalman filter computes the filtered estimate
$\widehat{x}_{k|k}=\mathbb{E}[x_k\mid y_{0:k}]$ and the covariance
$P_{k|k}$. Each step has a prediction stage, which uses the dynamics, and an
update stage, which incorporates the new measurement.

### Prediction

Given the filtered estimate at time $k-1$, the one-step prediction is

```{math}
:label: eq:ch13-prediction

\widehat{x}_{k|k-1}
=
A\widehat{x}_{k-1|k-1}+Bu_{k-1},
\qquad
P_{k|k-1}
=
AP_{k-1|k-1}A^\top+Q.
```

The mean follows the deterministic state transition, while the covariance is
pushed forward by $A$ and enlarged by the process uncertainty $Q$.

### Innovation and Update

The measurement does not act directly on the state. It acts through the
innovation, the mismatch between the observed output and the predicted output:

```{math}
:label: eq:ch13-innovation

\nu_k
=
y_k-C\widehat{x}_{k|k-1}-Du_k.
```

Its covariance is

```{math}
:label: eq:ch13-innovation-covariance

S_k
=
CP_{k|k-1}C^\top+R.
```

The Kalman gain is then

```{math}
:label: eq:ch13-kalman-gain

K_k^{\mathrm{KF}}
=
P_{k|k-1}C^\top S_k^{-1}.
```

With that gain, the filter update is

```{math}
:label: eq:ch13-update

\widehat{x}_{k|k}
=
\widehat{x}_{k|k-1}+K_k^{\mathrm{KF}}\nu_k,
```

and a numerically robust covariance update is

```{math}
:label: eq:ch13-covariance-update

P_{k|k}
=
\left(I-K_k^{\mathrm{KF}}C\right)P_{k|k-1}\left(I-K_k^{\mathrm{KF}}C\right)^\top
+
K_k^{\mathrm{KF}}R\left(K_k^{\mathrm{KF}}\right)^\top.
```

Equations {eq}`eq:ch13-prediction` through
{eq}`eq:ch13-covariance-update` define the classical discrete-time Kalman
filter. The gain $K_k^{\mathrm{KF}}$ determines how strongly the estimate moves
toward the measurement:

- If $R$ is large relative to $CP_{k|k-1}C^\top$, the filter trusts the model
  more than the measurement, and the gain is small.
- If $P_{k|k-1}$ is large or the measurement is very accurate, the gain is
  larger, and the update responds more strongly to $\nu_k$.

The innovation sequence is also diagnostically important. For a correctly
specified linear Gaussian model, $\nu_k$ is a zero-mean white sequence with
covariance $S_k$. Structured residual correlation usually signals model
mismatch, poor noise covariances, or unmodeled dynamics.

## Why the Filter Is a Least-Squares Estimator

The Kalman update can be derived as a weighted least-squares problem. Before
seeing $y_k$, the predicted state has Gaussian prior
$x_k\sim \mathcal{N}(\widehat{x}_{k|k-1},P_{k|k-1})$. The measurement model
adds the linear observation constraint $y_k\approx Cx_k+Du_k$ with covariance
$R$. The posterior mean is therefore the minimizer of

```{math}
:label: eq:ch13-map-update

\widehat{x}_{k|k}
=
\arg\min_{x}
\frac{1}{2}\norm{x-\widehat{x}_{k|k-1}}_{P_{k|k-1}^{-1}}^2
+
\frac{1}{2}\norm{y_k-Cx-Du_k}_{R^{-1}}^2.
```

This interpretation matters for two reasons. First, it shows that the filter
balances model trust against sensor trust through the inverse covariances.
Second, it connects Kalman filtering to deterministic regularized least squares
and to the joint state-estimation objectives introduced in Chapter 12. The
Kalman filter is simply the special linear Gaussian case where the recursive
solution is available in closed form.

## Covariance Propagation and Uncertainty

The state estimate alone is not enough. The covariance matrices
$P_{k|k-1}$ and $P_{k|k}$ tell us which directions in state space are well
constrained and which remain uncertain. That information is essential for
sensor fusion, trajectory reconstruction, and later parameter learning.

Several qualitative facts are worth keeping in view:

- Process noise $Q$ widens the predicted uncertainty because unresolved
  disturbances and model error accumulate through time.
- Measurement noise $R$ limits how sharply each observation can reduce the
  covariance.
- Poor observability appears as state directions whose covariance stays large
  because the measurements do not reveal them reliably.

In this sense, covariance propagation is the algebraic expression of what can
and cannot be learned from the measurement stream.

## The Rauch-Tung-Striebel Smoother

Filtering uses data up to time $k$. Smoothing uses the full record
$y_{0:N}$. In offline dynamics learning, smoothing is often the more relevant
operation because later measurements can clarify earlier hidden states.

The Rauch-Tung-Striebel (RTS) smoother runs a backward recursion after the
forward Kalman filter has been completed. Define the smoothing gain

```{math}
:label: eq:ch13-rts-gain

J_k
=
P_{k|k}A^\top P_{k+1|k}^{-1}.
```

Then the smoothed state estimate satisfies

```{math}
:label: eq:ch13-rts-state

\widehat{x}_{k|N}
=
\widehat{x}_{k|k}
+
J_k\left(\widehat{x}_{k+1|N}-\widehat{x}_{k+1|k}\right),
```

and the smoothed covariance is

```{math}
:label: eq:ch13-rts-covariance

P_{k|N}
=
P_{k|k}
+
J_k\left(P_{k+1|N}-P_{k+1|k}\right)J_k^\top.
```

The correction term compares the future-informed estimate
$\widehat{x}_{k+1|N}$ with the one-step prediction $\widehat{x}_{k+1|k}$. If
future data show that the forward prediction was inaccurate, that discrepancy
is pulled backward through the dynamics using $J_k$.

The smoother is the batch counterpart of the filter. In linear Gaussian
problems, the smoothed trajectory is both the conditional mean
$\mathbb{E}[x_{0:N}\mid y_{0:N}]$ and the maximum a posteriori trajectory. It
is therefore the natural reconstructed state history for offline identification
and latent-trajectory analysis.

## Innovation Likelihood and Maximum Likelihood

The same innovation quantities used for filtering also provide the observation
likelihood. Because $y_k\mid y_{0:k-1}$ is Gaussian with mean
$C\widehat{x}_{k|k-1}+Du_k$ and covariance $S_k$, the log-likelihood of the
measurement record is

```{math}
:label: eq:ch13-innovation-likelihood

\log p(y_{0:N})
=
-\frac{1}{2}\sum_{k=0}^{N}
\left(
\nu_k^\top S_k^{-1}\nu_k
+
\log\det S_k
+
p\log(2\pi)
\right).
```

This formula is a direct bridge from state estimation to system
identification. If $A$, $B$, $C$, $D$, $Q$, $R$, or the initial covariance are
unknown, then the Kalman filter turns each candidate parameter choice into a
likelihood value. Maximizing {eq}`eq:ch13-innovation-likelihood` is therefore a
prediction-error method: the best parameters are those whose innovations are
small after being normalized by their covariance and whose uncertainty model is
also plausible.

When $Q$ and $R$ are fixed, maximizing the likelihood is closely related to
weighted least squares. When they are estimated too, the $\log\det S_k$ term
matters because it prevents the optimizer from explaining the data with
artificially inflated uncertainty.

## Observability, Detectability, and Steady-State Filtering

The Kalman filter only performs well if the state is inferable from the output
history. In linear systems, observability is the strongest version of that
requirement: every state direction must eventually leave a measurable signature
through repeated application of $C$ and $A$.

For asymptotic filtering, the weaker notion of detectability is often the right
condition. Detectability allows unobservable modes provided those modes are
already stable. Intuitively, unstable hidden directions cannot be ignored
because their uncertainty grows and contaminates prediction, while stable
hidden directions may decay harmlessly.

Under standard assumptions, especially detectability of $(A,C)$ and
stabilizability of $(A,Q^{1/2})$, the covariance recursion converges to a
steady-state solution $P_\infty$ satisfying the discrete algebraic Riccati
equation

```{math}
:label: eq:ch13-dare

P_\infty
=
AP_\infty A^\top + Q
-
AP_\infty C^\top
\left(CP_\infty C^\top+R\right)^{-1}
CP_\infty A^\top.
```

The corresponding steady-state Kalman gain is

```{math}
:label: eq:ch13-steady-state-gain

K_\infty^{\mathrm{KF}}
=
P_\infty C^\top\left(CP_\infty C^\top+R\right)^{-1}.
```

Steady-state filtering matters computationally and conceptually. Computationally,
it replaces a time-varying recursion by a fixed gain after transients die out.
Conceptually, it shows that filtering is tied to the same structural questions
as identification and control: which modes are observable, which unstable
directions must be corrected by measurement, and how noise enters the closed
loop formed by model prediction plus estimator feedback.

## Connections to System Identification

Kalman filtering is often introduced as a state-estimation algorithm, but it is
equally important as part of identification workflows.

First, the innovation likelihood in
{eq}`eq:ch13-innovation-likelihood` provides an objective for fitting linear
state-space models from output data. This is the classical maximum-likelihood
route to estimating $A$, $B$, $C$, $D$, $Q$, and $R$.

Second, the innovation sequence offers a diagnostic criterion. A useful model
should leave residuals that are close to white and consistent with the assumed
covariance. If the innovations retain temporal structure, the issue is often
not numerical optimization but model structure: missing states, incorrect noise
modeling, or unmodeled inputs.

Third, subspace identification and prediction-error methods can be interpreted
through the same lens. Subspace methods estimate a realization from input-output
data, while prediction-error methods choose parameters that make the filter's
innovations statistically small. Both are trying to explain the same measured
trajectories with a hidden linear dynamical model.

## Summary

The main contribution of the classical linear Gaussian theory is clarity. It
shows, in a setting where every formula is explicit, how state estimation,
uncertainty propagation, trajectory reconstruction, and likelihood-based
learning fit together. Later nonlinear and ensemble methods keep the same
conceptual structure even when the Gaussian closure and exact recursions are no
longer available.
