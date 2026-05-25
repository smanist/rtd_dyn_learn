# Data, Noise, Preprocessing, and Validation Metrics

Chapter 1 framed the question of what object is being learned. The next step is
to ask whether the available data support that question at all. In dynamics
learning, disappointing models are often caused less by the regression method
than by a mismatch between the data-generation process and the quantity we hope
to infer. Sampling may be too coarse, trajectories may cover only a narrow
regime, derivatives may be estimated from noisy signals, or the validation
metric may reward the wrong behavior.

This chapter treats data preparation and model validation as part of the
modeling problem itself. The main goal is to turn sampled trajectories into a
dataset that preserves the dynamical information we care about, while being
explicit about what has been corrupted by noise, interpolation, smoothing, and
finite sampling.

## Sampled Trajectories as Learning Data

For a controlled system, a sampled dataset can be written as

```{math}
:label: eq:ch02-dataset

\mathcal{D}
=
\left\{
\left(\mathbf{u}^{(r)}_k,\mathbf{y}^{(r)}_k,t^{(r)}_k\right)_{k=0}^{N_r}
\right\}_{r=1}^{N_{\mathrm{traj}}},
```

where $r$ indexes trajectories, $t^{(r)}_k$ are sample times, and the bold
symbols denote stored numerical data. If the full state is observed, then
$\mathbf{y}^{(r)}_k$ may be replaced by $\mathbf{x}^{(r)}_k$. If only partial
observations are available, then the learning method must account for the gap
between measured outputs and hidden states.

Before fitting any model, it helps to ask four practical questions:

- What variables were actually measured?
- At what times were they measured?
- Under what inputs, regimes, and operating conditions were they measured?
- What transformations were applied before the data were stored?

Those questions determine whether the dataset supports map learning,
vector-field learning, state estimation, or only a narrower prediction task.

## Sampling Rate, Aliasing, and Time Irregularity

Sampling is not a neutral bookkeeping choice. It changes which behaviors are
visible in the data and which are effectively lost.

### Sampling Rate

If the sample interval $\Delta t$ is too large relative to the fastest relevant
timescale, then distinct trajectories can look artificially similar after
sampling. Rapid oscillations, transients, or unstable growth may be hidden
between samples. A discrete-time model fit to such data can still predict the
sampled sequence, but it may not represent the underlying continuous-time
mechanism faithfully.

This is the basic aliasing problem: high-frequency behavior is folded into a
lower-frequency representation. For dynamics learning, aliasing is especially
dangerous because it can make a wrong model look smooth and plausible.

### Irregular Sampling

Many real datasets are not sampled on a uniform grid. Missing packets, sensor
dropout, asynchronous measurements, and event-triggered logging all produce
nonuniform sample times $t_k$. That matters because methods that assume a fixed
$\Delta t$ may silently misinterpret the data.

Irregular sampling can be handled in several ways:

- Learn a continuous-time model that uses the actual time stamps.
- Resample onto a common grid when the interpolation error is acceptable.
- Use methods written directly for variable-step observations.

Resampling is convenient, but it is not free. Interpolation inserts a modeling
assumption before learning begins, so it should be treated as part of the
pipeline rather than as an invisible preprocessing detail.

### Missing Data

Missing samples are not merely shorter trajectories. They can break derivative
estimates, distort lagged embeddings, and bias train/test splits if the missing
pattern is regime dependent. Common responses include masking, imputation,
dropping corrupted windows, or reformulating the method so that it only uses
available timestamps. The right choice depends on whether the missingness is
rare and random or persistent and structured.

## Measurement Noise and Process Noise

The course notation distinguishes mathematical variables from stored data. That
distinction becomes essential once noise enters:

```{math}
:label: eq:ch02-noise-model

x_{k+1}=F(x_k,u_k;\mu,\theta)+w_k,
\qquad
y_k=h(x_k)+\eta_k.
```

Here $w_k$ represents process noise or unmodeled state evolution, while
$\eta_k$ represents measurement noise at the sensor level.

These two noise sources play different roles:

- Measurement noise corrupts what we observe, even if the underlying dynamics
  are deterministic.
- Process noise perturbs the state evolution itself, even if sensors are
  perfect.

Confusing them leads to the wrong preprocessing and the wrong validation. For
example, aggressive smoothing may reduce measurement noise in $\mathbf{y}_k$,
but it does not remove genuine stochasticity in the underlying state
transition. Conversely, a model that explains noisy outputs by inflating
process uncertainty can hide a poor observation model.

## Preprocessing for Dynamics Learning

Preprocessing is often presented as data cleaning. In dynamics learning it is
better understood as controlled distortion: every preprocessing step may remove
noise, but it may also remove dynamics.

### Basic Signal Conditioning

Typical first steps include unit checks, synchronization across sensors,
detrending, centering, and scaling. These are not glamorous, but they matter.
A learned model should not waste capacity on offsets caused by sensor
calibration or on scale disparities between variables that are artifacts of
measurement units.

Normalization also affects the meaning of error metrics. If one state component
has values around $10^3$ and another around $10^{-2}$, an unweighted Euclidean
loss will mostly report the large-scale component unless the data are scaled or
the metric is weighted deliberately.

### Smoothing, Filtering, and Denoising

Smoothing and filtering are useful when raw measurements are visibly corrupted,
but they should be chosen to match the downstream task.

- For discrete-time prediction, modest denoising may improve one-step fits
  without requiring derivative estimates.
- For continuous-time identification, smoothing is often used before numerical
  differentiation, so the bias introduced by the smoother directly affects the
  estimated vector field.
- For state-estimation problems, filtering should respect the temporal
  structure of the data rather than treating samples as independent points.

The central tradeoff is bias versus variance. Heavy smoothing lowers variance
but suppresses peaks, delays phase information, and can flatten genuinely
unstable or stiff behavior. Light smoothing preserves fast structure but leaves
more noise for the learner to absorb.

### Numerical Differentiation

Continuous-time learning often requires estimates of $\dot{x}(t)$ from sampled
data. A naive finite-difference formula,

```{math}
:label: eq:ch02-finite-difference

\dot{x}(t_k)\approx \frac{x_{k+1}-x_k}{\Delta t},
```

is highly sensitive to noise because differentiation amplifies high-frequency
errors. This is why derivative-free formulations are attractive in later
chapters: they avoid placing the noisiest operation at the front of the
pipeline.

If derivatives must be estimated, the key question is not only whether the
curve looks smooth, but whether the derivative estimate preserves the features
the model is meant to learn: equilibria, decay rates, oscillation frequency,
switching structure, or control response.

## Validation Metrics Must Match the Task

A model can score well on one validation metric and fail badly on another.
There is no universally correct metric; the metric has to reflect the intended
use of the learned dynamics.

### One-Step Prediction Error

For sampled models, the most common metric is one-step prediction error:

```{math}
:label: eq:ch02-one-step-loss

\mathcal{L}_{\mathrm{1step}}
=
\frac{1}{N}
\sum_{k=0}^{N-1}
\norm{\mathbf{y}_{k+1}-\hat{\mathbf{y}}_{k+1\mid k}}^2.
```

This metric is convenient and usually stable to optimize. It measures local
predictive consistency, but it does not guarantee good long-horizon behavior.
A model can make small one-step corrections while drifting badly in rollout.

### Rollout Error

Rollout validation checks whether repeated application of the learned model
tracks a full trajectory over time. This is the right test when the model will
be simulated recursively, as in forecasting, planning, or closed-loop control.
Rollout error is harder because small local biases accumulate. That is exactly
why it is often more informative than one-step loss.

### Vector-Field Error

For continuous-time identification, we may compare the learned vector field
$f_\theta(x,u)$ against trusted derivative targets or against known benchmark
dynamics. This is useful when the scientific object of interest is the rate
law itself. The limitation is obvious: if derivative targets were obtained from
aggressive smoothing or poor differencing, then vector-field error may mostly
measure preprocessing quality rather than model quality.

### Beyond Pointwise Prediction

For dynamics, pointwise prediction is often not enough. Additional validation
questions include:

- Stability validation: does the learned model preserve stable and unstable
  behavior in the right regions?
- Spectral validation: are dominant frequencies, growth rates, or modal
  structures reproduced?
- Invariant-measure validation: for long-run stochastic or chaotic systems,
  does the model reproduce stationary statistics rather than only short
  trajectories?
- Control-performance validation: if the model is used inside a controller,
  does that controller achieve the desired closed-loop behavior?

These criteria are not optional extras. They are often the real objective.

## Train/Test Splits by Trajectory, Regime, and Operating Condition

A random sample-wise split is usually too optimistic for time-series dynamics.
Nearby samples from the same trajectory are strongly dependent, so random
shuffling leaks future information into the training set.

More defensible splits include:

- By trajectory: train and test on distinct rollouts.
- By regime: hold out specific initial conditions, forcing patterns, or
  transients.
- By operating condition: train at some values of $\mu$ and test at held-out
  values to evaluate interpolation or extrapolation across the parameter
  family.

The split should mirror the claim we want the model to support. If we claim a
model generalizes across operating conditions, then a split that mixes all
conditions into both train and test sets does not test that claim.

## A Practical Chapter 02 Checklist

Before moving on to identification methods, it is worth checking the following:

- The sampling rate resolves the dynamics that matter for the task.
- Irregular sampling and missing data have been handled explicitly.
- Measurement noise and process noise are not being conflated.
- Smoothing or denoising choices are documented and justified.
- Numerical differentiation, if used, is treated as a major modeling decision.
- Validation metrics reflect the intended use of the model.
- Train/test splits separate trajectories or regimes in a way that tests the
  desired generalization claim.

The main lesson is that data preparation is not separate from dynamics
learning. Sampling choices determine what is visible. Noise models determine
what can be trusted. Preprocessing changes the target the learner sees. And
validation metrics decide which failures are visible at all.
