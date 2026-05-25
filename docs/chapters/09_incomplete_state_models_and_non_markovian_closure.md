# Incomplete State Models and Non-Markovian Closure

Chapter 08 studied gray-box residual learning when the state is already a
predictively sufficient description of the system and the main gap is missing
physics inside the evolution law. This chapter addresses a different failure
mode: the available state is itself incomplete. In that setting, a model built
only on the resolved variables may fail even when its instantaneous residual is
fit very accurately, because the missing information is not a force term but an
unobserved part of the state.

That distinction is central in reduced-order modeling, coarse-graining, and
partially observed dynamics. If two trajectories share the same resolved
coordinate $x_a$ at time $t$ but differ in unresolved coordinates $x_b$, then
their future resolved evolution need not agree. The reduced model is therefore
not Markovian in $x_a$ alone. To recover predictive sufficiency we must add
something beyond a static residual: a closure variable, a memory term, a delay
state, or another latent representation of the missing information.

## From Full State to Reduced State

Suppose the full system state is partitioned as

```{math}
:label: eq:ch09-state-partition

x(t)=\bigl(x_a(t),x_b(t)\bigr),
\qquad
x_a(t)\in \mathbb{R}^{n_a},
\quad
x_b(t)\in \mathbb{R}^{n_b}.
```

The resolved coordinates $x_a$ are the variables we retain in the reduced
model, while $x_b$ collects unresolved, latent, or discarded variables. A
continuous-time full model can then be written as

```{math}
:label: eq:ch09-partitioned-dynamics

\dot{x}_a(t)=f_a\bigl(x_a(t),x_b(t),u(t)\bigr),
\qquad
\dot{x}_b(t)=f_b\bigl(x_a(t),x_b(t),u(t)\bigr).
```

If $x_b$ were measured and modeled, then
{eq}`eq:ch09-partitioned-dynamics` would define an ordinary Markovian state
model. The difficulty appears when we try to evolve only $x_a$. In general,
there is no closed autonomous rule of the form

```{math}
:label: eq:ch09-naive-markov-closure

\dot{x}_a(t)=\widehat{f}\bigl(x_a(t),u(t)\bigr)
```

that reproduces the resolved dynamics for all trajectories, because
$x_b(t)$ carries information about unresolved history, hidden modes, or
unmeasured environmental variables.

The same issue appears in discrete time:

```{math}
:label: eq:ch09-discrete-partitioned-dynamics

x_{a,k+1}=F_a(x_{a,k},x_{b,k},u_k),
\qquad
x_{b,k+1}=F_b(x_{a,k},x_{b,k},u_k).
```

If we discard $x_{b,k}$, then the map from $x_{a,k}$ to $x_{a,k+1}$ is
generally not single-valued without extra state information.

## Why the Reduced Model Becomes Non-Markovian

A Markov model is predictively sufficient in the sense that the present state
contains all the information needed for future evolution. When the retained
state is incomplete, future increments depend on more than the present resolved
coordinate. Eliminating $x_b$ typically produces a history-dependent evolution
law rather than a closed Markovian one.

At a schematic level, the resolved dynamics may be written as

```{math}
:label: eq:ch09-memory-closure

\dot{x}_a(t)
=
\widehat{f}\bigl(x_a(t),u(t)\bigr)
+
\int_0^t
K_\theta\bigl(t,s;x_a(s),u(s)\bigr)\,\dd s
+
r_\theta(t),
```

where $\widehat{f}$ is an instantaneous Markov term, the memory kernel
$K_\theta$ summarizes how unresolved history feeds back into the present, and
$r_\theta$ represents dependence on unresolved initial conditions or other
leftover effects. Equation {eq}`eq:ch09-memory-closure` is intentionally
generic. The main point is structural: after projection onto $x_a$, the
effective closure depends on path history, not only on the current value
$x_a(t)$.

This is why a static residual correction can fail. A model such as

```{math}
:label: eq:ch09-static-residual

\dot{x}_a(t)=f_{\mathrm{known}}\bigl(x_a(t),u(t)\bigr)+g_\theta\bigl(x_a(t),u(t)\bigr)
```

assumes that the missing effect is an instantaneous function of the current
resolved state and input. That assumption is appropriate when the main issue is
missing physics at a complete state description. It is too restrictive when the
missing ingredient is unresolved state information.

## Missing Physics Versus Missing State Information

The distinction between residual learning and incomplete-state closure is
conceptual, not only algorithmic.

### Missing Physics

If the relevant state is already included, but the evolution law omits a force,
constitutive relation, source term, or damping mechanism, then a residual model
like {eq}`eq:ch09-static-residual` may be sufficient. The defect is in the map
from state to derivative, not in the state itself.

### Missing State Information

If the future of the resolved variables depends on unresolved variables, then
even the correct instantaneous force law on $x_a$ is not enough. The problem is
that $x_a$ is not a sufficient state. Two trajectories can share the same
$x_a(t)$ and input $u(t)$ while producing different $\dot{x}_a(t)$ because
their latent coordinates $x_b(t)$ differ.

A useful diagnostic is to ask whether identical resolved snapshots can lead to
systematically different short-horizon futures. If they can, the model deficit
is likely due to hidden state or memory, not merely a missing residual term.

## Closure Through Latent-State Augmentation

One way to restore a Markovian description is to augment the resolved state
with a learned latent variable $z(t)$. The reduced model becomes

```{math}
:label: eq:ch09-latent-closure

\dot{x}_a(t)=f_a^{\mathrm{red}}\bigl(x_a(t),z(t),u(t)\bigr),
\qquad
\dot{z}(t)=g_\theta\bigl(x_a(t),z(t),u(t)\bigr).
```

Here, $z$ is not necessarily meant to recover the original unresolved state
$x_b$. Instead, it is an auxiliary variable chosen so that $(x_a,z)$ is close
to Markovian and therefore predictive. The same idea in discrete time is

```{math}
:label: eq:ch09-latent-closure-discrete

x_{a,k+1}=F_\theta(x_{a,k},z_k,u_k),
\qquad
z_{k+1}=G_\theta(x_{a,k},z_k,u_k).
```

This augmentation viewpoint underlies latent ordinary differential equations,
recurrent state-space models, and many reduced-order closure architectures. The
learned variable $z$ acts as a compressed memory of unresolved influences.

Latent augmentation is attractive because it converts a non-Markovian problem
back into a Markovian one in an expanded state space. That makes simulation,
stability analysis, filtering, and control more straightforward than working
directly with a full history integral.

## Delay and Recurrent Closures

A second approach is to encode memory explicitly through past resolved states
and inputs. In discrete time, one may posit a delay model of the form

```{math}
:label: eq:ch09-delay-model

x_{a,k+1}
=
F_\theta\bigl(
x_{a,k},x_{a,k-1},\dots,x_{a,k-d},
u_k,u_{k-1},\dots,u_{k-d}
\bigr).
```

The delayed coordinates act as an implicit latent state. Delay models are often
effective when unresolved effects have a finite memory horizon or when the
hidden dynamics can be reconstructed from a sufficiently rich observation
history.

Recurrent models package the same idea into an evolving hidden summary. A
generic recurrent closure has the form

```{math}
:label: eq:ch09-recurrent-closure

z_{k+1}=G_\theta(z_k,x_{a,k},u_k),
\qquad
x_{a,k+1}=F_\theta(z_k,x_{a,k},u_k),
```

where $z_k$ is a learned memory state. Delay-coordinate models store a fixed
window of past observations explicitly, while recurrent models learn a compact
state update for that history.

## Mori-Zwanzig and Memory as a Modeling Principle

The Mori-Zwanzig viewpoint gives a principled explanation for why memory terms
appear after reduction. When the full dynamics are projected onto resolved
coordinates, the exact reduced equation generally decomposes into three pieces:
an instantaneous Markov term, a history-dependent memory term, and a remaining
noise-like contribution associated with unresolved initial conditions. This
structure is consistent with the schematic form
{eq}`eq:ch09-memory-closure`.

For this chapter, the important lesson is not the formal derivation but the
implication: non-Markovian closure is not an ad hoc patch. It is the expected
mathematical consequence of eliminating degrees of freedom from a larger
Markovian system.

That perspective clarifies why some reduced models benefit from convolutional
memory kernels, generalized Langevin equations, autoregressive forcing, or
learned recurrent states. These are different approximations to the same
structural fact that unresolved variables leave a time-delayed footprint on the
resolved dynamics.

## When Simple Residual Learning Is Insufficient

Residual learning becomes inadequate when the discrepancy is not a single
instantaneous function of the present resolved state. Common warning signs
include:

- Hysteresis or path dependence, where the same $x_a$ leads to different
  futures depending on how the system arrived there.
- Strong scale separation, where slow resolved variables depend on fast
  unresolved fluctuations only through accumulated memory effects.
- Partial observation, where key modes, phases, or internal variables are never
  measured directly.
- Regime switching driven by hidden variables, so that local residual fits look
  inconsistent across trajectories.

In such settings, adding a more expressive instantaneous regressor may improve
one-step errors without fixing the underlying state deficiency. The model can
still drift badly in rollout because it has no mechanism to carry unresolved
information forward in time.

## Implications for Identification and Validation

Incomplete-state modeling changes both the learning problem and the validation
strategy.

First, the learner must infer more than a right-hand side. It must decide how
to represent unresolved influence: through explicit latent variables, delay
coordinates, memory kernels, or recurrent hidden states. This is a structural
choice about what counts as state.

Second, training losses based only on instantaneous residuals or one-step
prediction can be misleading. A closure that ignores memory may fit local data
well but fail on long rollouts, transient recovery, or regime transfer. For
non-Markovian models, validation should therefore emphasize trajectory-level
behavior, recovery of lagged responses, and robustness under new initial
conditions or forcing histories.

Third, excitation matters in a stronger sense. To learn closure effects, the
data must expose how unresolved dynamics feed back into resolved variables over
time. Short, weakly forced, or nearly steady trajectories may be insufficient
to distinguish a static residual from a true memory mechanism.

## Main Takeaways

Incomplete-state dynamics and residual-model errors are not the same problem.
If the retained state is not Markovian, then a reduced model needs some way to
carry forward unresolved information. That extra information may appear as a
latent augmented state, a delay embedding, a recurrent hidden variable, or an
explicit memory term. The right question is not only "what forcing is missing?"
but also "what state information is missing?" Chapter 09 therefore marks the
transition from closure as a correction term to closure as a reconstruction of
predictive state.
