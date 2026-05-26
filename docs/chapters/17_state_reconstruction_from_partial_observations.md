# State Reconstruction from Partial Observations

Many dynamical systems are only observed through a few sensors. We may measure
positions but not velocities, surface pressures but not the full flow field, or
aggregate activity but not the internal variables that actually close the
evolution law. In that setting, learning dynamics is not only a problem of
fitting a predictor. It is also a problem of constructing a state
representation that is rich enough to make future evolution depend primarily on
the present representation rather than on the entire past.

This chapter studies that reconstruction problem. The central question is:
given partial observations, how can we build coordinates that recover an
approximately Markovian description of the system? Classical answers come from
observability theory, delay embeddings, and subspace identification. Modern
answers include recurrent estimators and latent-state neural models. All of
them confront the same structural issue: the sensor output is usually not a
state by itself.

## Partial Observations and Loss of the Markov Property

A standard partially observed discrete-time model is

```{math}
:label: eq:ch17-partial-observation-model

x_{k+1}=F(x_k),
\qquad
y_k=h(x_k)+v_k.
```

The hidden state $x_k\in \mathbb{R}^n$ may evolve Markovianly, but the
observation $y_k\in \mathbb{R}^p$ generally does not. If the map $h$ is not
injective, then different states can produce the same measurement. Knowing
$y_k$ alone may therefore be insufficient to determine the distribution of
$y_{k+1}$.

This is the basic reason partial observation creates memory. Even when the
underlying physics are first order in $x_k$, the observed process may behave as
though it were higher order. The missing information can sometimes be supplied
by a window of past measurements, by known inputs, or by a learned latent
state that summarizes the relevant history.

For driven systems, the same principle holds with inputs included:
$x_{k+1}=F(x_k,u_k)$ and $y_k=h(x_k,u_k)+v_k$. In that case, reconstructing a
state from outputs alone is usually impossible; the representation must also
account for the forcing history.

## Delay Coordinates as Reconstructed State

The simplest reconstruction idea is to stack delayed observations into one
augmented vector:

```{math}
:label: eq:ch17-delay-coordinates

z_k
=
\begin{bmatrix}
y_k\\
y_{k-1}\\
\vdots\\
y_{k-m}
\end{bmatrix}.
```

The new variable $z_k$ is not the original physical state, but it may still be
a useful state representation. If the delay window is long enough and the
measurement function is informative enough, then the current delay vector can
disambiguate states that a single observation cannot.

When this succeeds, the dynamics can be rewritten approximately as a Markovian
map on reconstructed coordinates:

```{math}
:label: eq:ch17-delay-dynamics

z_{k+1}=G(z_k).
```

Equation {eq}`eq:ch17-delay-dynamics` is important conceptually. It says that
state reconstruction does not require recovering the original variables
exactly. It is enough to find coordinates in which the future is determined, to
use a good approximation, by the present reconstructed state.

### Why Delays Can Work

Delay coordinates encode short-term history, and short-term history often
contains the hidden variables implicitly. For example, position alone is not a
Markovian state for a second-order oscillator, but a short history of position
reveals velocity information through how the measurements change in time. In
that sense, delay coordinates convert temporal information into state
information.

Takens' embedding theorem formalizes this idea for generic smooth observations
of deterministic dynamics on a compact attractor. Roughly stated, it says that
for a sufficiently informative scalar measurement and a sufficiently large
number of delays, the delay map is an embedding of the attractor. The theorem
does not say that every sensor works, nor that finite noisy data are easy to
use in practice. What it provides is a structural explanation for why
delay-based reconstruction can recover a valid state description even when the
physical state is hidden.

### Practical Limits of Delay Embedding

Delay reconstructions are sensitive to several design choices:

- If $m$ is too small, then distinct latent states may still collapse to the
  same delay vector.
- If $m$ is too large, the representation becomes high dimensional and more
  data hungry.
- If the sampling interval is too small, consecutive delays carry nearly
  redundant information.
- If the sampling interval is too large, the window may skip over important
  fast dynamics.
- Measurement noise can make naive delay stacks poorly conditioned.

These tradeoffs matter because Takens-type results are asymptotic and generic,
while learning from data is finite, noisy, and model-class dependent.

## Observability and Finite-Horizon Reconstruction

Classical observability theory gives a more algebraic view of the same problem.
Consider the linear autonomous model

```{math}
:label: eq:ch17-linear-state-space

x_{k+1}=Ax_k,
\qquad
y_k=Cx_k.
```

If we stack $m$ consecutive outputs, we obtain

```{math}
:label: eq:ch17-observability-stack

\begin{bmatrix}
y_k\\
y_{k+1}\\
\vdots\\
y_{k+m-1}
\end{bmatrix}
=
\mathcal{O}_m x_k,
\qquad
\mathcal{O}_m
=
\begin{bmatrix}
C\\
CA\\
\vdots\\
CA^{m-1}
\end{bmatrix}.
```

When the observability matrix $\mathcal{O}_m$ has rank $n$ for some finite
$m$, the state can in principle be recovered from a finite output window in the
noise-free linear setting. This is the linear analogue of delay-based
reconstruction: the output history carries enough information to determine the
hidden state.

Observability is a structural property of the model and sensor map. It answers
whether exact recovery is possible in principle. It does not answer whether a
chosen learning architecture can recover that state reliably from finite noisy
data, or whether the reconstructed coordinates will generalize well outside the
observed regime.

## Subspace Identification as State Reconstruction

Subspace identification methods operationalize reconstruction directly from
data. Rather than first postulating a hidden state and then estimating it, they
assemble block Hankel matrices of past and future measurements and extract a
low-dimensional state sequence whose linear evolution explains the observed
trajectories well.

In this view, the state is defined by predictability. A good reconstructed state
is one that summarizes the part of the past needed to predict the future. For
linear systems, this links naturally to the observability matrix in
{eq}`eq:ch17-observability-stack`. The recovered coordinates are not unique;
they are determined only up to an invertible change of basis. That ambiguity is
usually harmless, because prediction and control depend on the state-space
realization rather than on a preferred coordinate naming convention.

For controlled systems, subspace methods reconstruct state from joint histories
of inputs and outputs. That detail matters: under forcing, the same output
trace can arise from different states if the input history is ignored.

## Recurrent and Latent-State Reconstruction

Modern learning methods replace hand-designed delay coordinates with learned
representations. A common latent-state formulation is

```{math}
:label: eq:ch17-latent-state-model

z_k=E_\theta(y_{0:k}),
\qquad
z_{k+1}=G_\theta(z_k),
\qquad
\widehat{y}_k=D_\theta(z_k).
```

Here $z_k$ is a learned latent state, $E_\theta$ is an encoder or recurrent
state estimator, $G_\theta$ is latent dynamics, and $D_\theta$ is a decoder
back to the observation space. In controlled settings, $G_\theta$ would also
depend on $u_k$.

This perspective includes several model families:

- Recurrent state estimators update a hidden representation online from the
  current measurement and past hidden state.
- Autoencoder-style models constrain the latent state to decode back to the
  observations while evolving simply in latent space.
- Neural state-space models combine a latent dynamics law with a learned
  observation map, making hidden-state inference part of training.

The main advantage is flexibility. The representation can adapt to nonlinear
observation maps and to systems for which a fixed delay stack is inefficient.
The main risk is that the learned latent state may optimize short-term
prediction without capturing the physically relevant hidden variables.

## Observability Versus Learnability

State reconstruction sits at the boundary between structural system theory and
statistical learning. Two systems may both be observable in principle, yet one
may be much harder to learn from finite data than the other.

Observability asks whether the hidden state leaves enough signature in the
measurements. Learnability asks whether the available data, noise level, sample
rate, excitation, and model class let us recover a useful representation in
practice. Weakly excited trajectories, poor sensor placement, heavy measurement
noise, or an overly restrictive latent model can all defeat learning even when
observability is not the fundamental obstacle.

This distinction is especially important when evaluating neural latent-state
models. A small prediction loss does not by itself prove that the learned
representation has recovered a faithful state. It may only show that the model
has built a compressed memory useful on the training distribution. To argue
that the representation behaves like state, we usually want evidence such as
multi-step rollout stability, transfer across trajectories, recovery of hidden
variables when ground truth exists, or success in downstream estimation and
control tasks.

## Summary

A useful reconstructed state should satisfy three practical tests.

First, it should make prediction approximately Markovian: conditioning on the
current representation should remove most of the value of longer histories.
Second, it should be low dimensional enough to estimate robustly from data.
Third, it should support the downstream task, whether that task is forecasting,
filtering, interpretation, or control.

These goals do not always point to the same representation. Delay coordinates
are simple and model agnostic, but they can be redundant. Linear subspace
states are efficient when the system is well approximated by a linear
realization. Learned latent states are expressive, but they require stronger
regularization and more careful validation.

The main lesson of this chapter is that partial observation does not remove the
possibility of state-space learning. It changes the problem from direct
identification of $x_k$ to construction of coordinates that recover the
information needed for evolution. Delay embeddings, observability theory,
subspace reconstruction, and learned latent states are all methods for
answering the same question: what should count as the state when we cannot
measure it directly?
