# Reduced-Order Dynamics and Closure

High-dimensional dynamical systems often evolve on state spaces that are too
large for direct simulation, inference, uncertainty propagation, or feedback
design to be practical. A reduced-order model (ROM) replaces the full state
$x(t)\in \mathbb{R}^n$ with a lower-dimensional coordinate system of dimension
$r\ll n$ while trying to preserve the behavior that matters for the task at
hand. Chapter 22 studies that compression problem and the central modeling
question it creates: what dynamics can be written purely in reduced
coordinates, and what information is lost when unresolved directions are
truncated away?

Two themes organize the chapter. First, reduced models are not only numerical
accelerations. They are learned dynamical models whose state representation is
part of the identification problem. Second, truncation creates closure error.
Once neglected modes influence the retained ones, a naive Markovian ROM is no
longer exact, so the reduced dynamics must account for unresolved physics
through residual, latent, or memory terms.

## Projection-Based Reduced Coordinates

The classical reduced-order ansatz assumes that the full state is well
approximated by a low-dimensional basis:

```{math}
:label: eq:ch22-linear-rom

x(t)\approx V_r a(t),
```

where $V_r\in \mathbb{R}^{n\times r}$ contains reduced basis vectors and
$a(t)\in \mathbb{R}^r$ is the reduced state. Equation
{eq}`eq:ch22-linear-rom` is the projection-based view of model reduction: the
high-dimensional trajectory is compressed into generalized coordinates
$a(t)$.

The basis $V_r$ may come from physics, symmetry arguments, balanced
coordinates, or data. In data-driven ROMs, a common starting point is a
snapshot matrix

```{math}
:label: eq:ch22-snapshot-matrix

\mathbf{X}
=
\begin{bmatrix}
\mathbf{x}_0 & \mathbf{x}_1 & \cdots & \mathbf{x}_N
\end{bmatrix},
```

followed by a truncated singular value decomposition

```{math}
:label: eq:ch22-pod-svd

\mathbf{X}\approx U_r \Sigma_r W_r^\top.
```

The first $r$ columns of $U_r$ then define $V_r$. This is the proper
orthogonal decomposition (POD) viewpoint: choose the $r$-dimensional linear
subspace that captures as much snapshot energy as possible in an average
$L^2$ sense.

Projection alone, however, is not yet a reduced dynamical model. We still need
an evolution law for $a(t)$.

## POD-Galerkin Models and Truncation Error

Suppose the full system obeys

```{math}
:label: eq:ch22-full-dynamics

\dot{x}=f(x,u).
```

If $V_r$ has orthonormal columns and we substitute $x\approx V_r a$, then a
Galerkin projection gives the reduced dynamics

```{math}
:label: eq:ch22-galerkin

\dot{a}=V_r^\top f(V_r a,u).
```

Equation {eq}`eq:ch22-galerkin` is attractive because it produces an explicit
Markovian system in $\mathbb{R}^r$. It is often the first ROM one writes for
fluid, structural, and PDE-based models. But it is only exact if the resolved
subspace is dynamically closed, which is rare in nonlinear systems.

To see the missing piece, extend the basis to $[V_r\;V_\perp]$ and decompose
the exact state as

```{math}
:label: eq:ch22-state-splitting

x=V_r a + V_\perp b,
```

where $b(t)$ collects unresolved coordinates. The exact projected dynamics are

```{math}
:label: eq:ch22-exact-projected

\dot{a}
=
V_r^\top f(V_r a + V_\perp b,u)
=
g_{\mathrm{r}}(a,u)+c(a,b,u),
```

with the nominal Markovian part

```{math}
:label: eq:ch22-nominal-closure

g_{\mathrm{r}}(a,u)=V_r^\top f(V_r a,u)
```

and the closure term

```{math}
:label: eq:ch22-closure-term

c(a,b,u)
=
V_r^\top
\left[
f(V_r a + V_\perp b,u)-f(V_r a,u)
\right].
```

This decomposition makes truncation error concrete. The reduced coordinates
$a$ do not evolve independently; they are forced by the neglected coordinates
$b$. A ROM that drops $c$ assumes that unresolved scales neither backscatter
energy nor change the drift seen by the retained variables. That assumption is
often the main source of reduced-model instability or long-horizon drift.

## Balanced Truncation and Control-Relevant Reduction

Projection based on snapshot energy is not always the right criterion. In
input-output settings, we may care more about which directions are controllable
and observable than which ones have the largest variance. For the stable
linear system

```{math}
:label: eq:ch22-linear-system

\dot{x}=Ax+Bu,
\qquad
y=Cx,
```

balanced truncation searches for coordinates in which the state components are
ordered by joint controllability and observability. Modes that are hard to
excite and hard to measure are truncated first.

This perspective matters because a good ROM for control is not merely one that
matches state snapshots. It should also preserve the input-output map,
stability properties, and feedback-relevant transients. Balanced truncation is
therefore a useful complement to POD-Galerkin: POD emphasizes energetic
content, while balanced truncation emphasizes control authority and sensor
sensitivity.

For linear systems, this distinction leads to explicit validation questions:
does the reduced model preserve gain, dominant poles, transient amplification,
and closed-loop behavior? Those are often more relevant than pointwise state
reconstruction error.

## Nonlinear Latent Coordinates

Linear subspaces are not always expressive enough. If the dynamics evolve near
a curved low-dimensional manifold, a nonlinear encoder-decoder pair can define
reduced coordinates:

```{math}
:label: eq:ch22-autoencoder

z=E(x),
\qquad
x\approx D(z),
```

where $z(t)\in \mathbb{R}^r$ is a latent state. A learned latent ROM then
posits

```{math}
:label: eq:ch22-latent-dynamics

\dot{z}=g_\theta(z,u),
\qquad
x(t)\approx D(z(t)).
```

This is the nonlinear analogue of {eq}`eq:ch22-linear-rom`. The encoder
$E$ learns a compact coordinate system, the decoder $D$ reconstructs the full
state approximately, and $g_\theta$ supplies the reduced dynamics. Autoencoder
ROMs are attractive when the dominant state manifold is curved, when a linear
basis needs too many modes, or when the full coordinates are not themselves
the most natural reduced variables.

The main caution is that compression and dynamics learning are coupled. A
latent coordinate system that reconstructs snapshots well may still be poor
for forecasting if nearby latent states do not evolve smoothly or uniquely.
Reduced representation quality must therefore be judged dynamically, not only
geometrically.

## Closure, Memory, and Non-Markovian Effects

The closure term in {eq}`eq:ch22-closure-term` usually cannot be written as a
function of $a(t)$ alone. Once unresolved variables are removed, the exact
reduced dynamics often become non-Markovian. A more faithful model may take
the form

```{math}
:label: eq:ch22-memory-rom

\dot{a}(t)
=
g_{\mathrm{r}}(a(t),u(t))
+
\int_0^t K\bigl(t-s,a(s),u(s)\bigr)\,\mathrm{d}s
+
\eta(t),
```

where the kernel $K$ summarizes memory from unresolved modes and $\eta(t)$
collects remaining approximation error. This is the key conceptual difference
between a reduced-order model and a merely truncated model. Truncation removes
variables; closure tries to represent what those removed variables still do to
the retained ones.

That viewpoint connects ROMs to several ideas from earlier chapters:

- A learned residual term can play the role of a closure model.
- Delay coordinates or recurrent states can approximate memory effects without
  explicitly storing the full unresolved dynamics.
- Stochastic forcing can be used when unresolved scales act more like random
  fluctuations than deterministic hidden variables.

In practice, Markovian ROMs are often preferred because they are easier to
simulate and control, but one should understand what is being sacrificed. If
history dependence is strong, enforcing a purely Markovian reduced model can
produce biased long-term statistics even when one-step prediction looks good.

## Parametric Reduced-Order Models

Many systems vary across operating conditions, geometries, or material
parameters. A useful ROM should therefore handle parameter dependence rather
than being tied to a single regime. One abstract formulation is

```{math}
:label: eq:ch22-parametric-rom

\dot{a}=g(a,u;\mu),
\qquad
x\approx V_r(\mu)a,
```

where $\mu$ is a known operating parameter. The basis itself may depend on
$\mu$, or the model may use a shared latent space with parameter-dependent
dynamics. Either way, the challenge is no longer only dimension reduction; it
is interpolation across a family of reduced systems without losing stability
or physical consistency.

Parametric ROMs are especially important for design studies, robust control,
and uncertainty propagation. A reduced model that is accurate at one operating
point but unstable or biased away from that point is of limited practical use.

## Summary

Reduced models should be validated against their intended use, not only by
reconstruction error. Several questions matter:

- For prediction, does the ROM preserve transient growth, long-horizon phase,
  and invariant-set structure?
- For uncertainty propagation, does the ROM preserve the right sensitivity to
  initial-condition, parameter, or forcing perturbations?
- For control, does the ROM support the same stabilizing or tracking decisions
  as the full model over the operating range of interest?

These validation goals explain why stabilization is often part of ROM design.
A low-dimensional model that is slightly cheaper but numerically unstable is
not a useful surrogate. Closure terms, dissipativity constraints, regularized
latent dynamics, or control-oriented reduction criteria may all be justified
if they improve long-horizon robustness.

The larger lesson of this chapter is that reduced-order dynamics learning is
not finished when a low-dimensional coordinate system has been found. The real
task is to construct reduced dynamics that remain faithful after unresolved
degrees of freedom have been removed. That is why projection, latent
coordinates, truncation, closure, memory, and validation belong in one
conversation rather than in separate numerical subroutines.
