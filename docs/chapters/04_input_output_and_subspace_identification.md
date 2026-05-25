# Input-Output and Subspace Identification

Chapter 03 treated linear identification in the easiest setting: the full
state was measured directly, so fitting $A$ and $B$ reduced to regression on
snapshots. Many real experiments do not give us that luxury. We measure
outputs such as lift, voltage, displacement, or a few sensor channels, while
the internal state that actually evolves remains hidden. Chapter 04 studies
how to learn linear state-space models from input-output data in that partial
observation setting.

The central shift is conceptual. When the state is hidden, identification is
no longer only about estimating coefficients. It is also about constructing a
state realization whose dimension, controllability, and observability explain
the observed input-output behavior. That is why realization theory and
subspace methods belong in the same chapter as prediction-error models such as
ARX and ARMAX. They are different ways to use the same data when the unknown
state is part of the modeling problem.

## The Linear Input-Output Model

We take the discrete-time linear state-space model

```{math}
:label: eq:ch04-state-space

x_{k+1}=Ax_k+Bu_k,
\qquad
y_k=Cx_k+Du_k,
```

where $x_k\in \mathbb{R}^n$ is hidden, $u_k\in \mathbb{R}^m$ is known, and
$y_k\in \mathbb{R}^p$ is measured. The order $n$ is now part of the
identification task, not an input supplied by full-state data.

Iterating {eq}`eq:ch04-state-space` shows how outputs depend on past inputs and
the initial condition:

```{math}
:label: eq:ch04-io-expansion

y_k
=
CA^k x_0
+
\sum_{j=0}^{k-1} CA^{k-1-j}Bu_j
+
Du_k.
```

Equation {eq}`eq:ch04-io-expansion` separates three effects:

- The hidden initial state contributes the transient term $CA^k x_0$.
- The forced response is encoded by products of $C$, $A$, and $B$.
- The direct feedthrough term $D$ acts instantaneously on the input.

This expansion already explains why partial-observation identification is
harder than full-state regression. Different triples $(A,B,C)$ can produce the
same input-output behavior if they are related by a change of state basis, and
poorly observable or poorly controllable modes can be invisible in finite
data.

## Markov Parameters and Impulse Responses

The input-output behavior of {eq}`eq:ch04-state-space` is summarized by the
Markov parameters

```{math}
:label: eq:ch04-markov-parameters

G_0=D,
\qquad
G_\ell = CA^{\ell-1}B \quad \text{for } \ell \geq 1.
```

If the initial condition is zero, then {eq}`eq:ch04-io-expansion` becomes the
convolution law

```{math}
:label: eq:ch04-convolution

y_k
=
\sum_{j=0}^{k} G_j u_{k-j}.
```

The sequence $\{G_j\}$ is the impulse response of the system. It is an
input-output object: it does not depend on a particular choice of state
coordinates. Realization theory asks the inverse question. Given Markov
parameters estimated from data, can we construct a finite-dimensional state
model that generates them?

That question matters operationally because a finite-dimensional realization
compresses a long convolution history into a low-dimensional hidden state. The
quality of that compression is governed by the effective order $n$ of the
system.

## Hankel Matrices and Realization

To recover a state-space model from impulse-response data, we collect Markov
parameters into block Hankel matrices. For block sizes $s$ and $r$, define

```{math}
:label: eq:ch04-hankel

H_0
=
\begin{bmatrix}
G_1 & G_2 & \cdots & G_r \\
G_2 & G_3 & \cdots & G_{r+1} \\
\vdots & \vdots & \ddots & \vdots \\
G_s & G_{s+1} & \cdots & G_{s+r-1}
\end{bmatrix},
\qquad
H_1
=
\begin{bmatrix}
G_2 & G_3 & \cdots & G_{r+1} \\
G_3 & G_4 & \cdots & G_{r+2} \\
\vdots & \vdots & \ddots & \vdots \\
G_{s+1} & G_{s+2} & \cdots & G_{s+r}
\end{bmatrix}.
```

For an exact order-$n$ linear system, these matrices factor as

```{math}
:label: eq:ch04-hankel-factorization

H_0 = \mathcal{O}_s \mathcal{C}_r,
\qquad
H_1 = \mathcal{O}_s A \mathcal{C}_r,
```

where

```{math}
:label: eq:ch04-obs-ctrb

\mathcal{O}_s
=
\begin{bmatrix}
C \\
CA \\
\vdots \\
CA^{s-1}
\end{bmatrix},
\qquad
\mathcal{C}_r
=
\begin{bmatrix}
B & AB & \cdots & A^{r-1}B
\end{bmatrix}
```

are the block observability and controllability matrices.

Equation {eq}`eq:ch04-hankel-factorization` is the core structural fact behind
subspace identification. The rank of $H_0$ is at most $n$, and in the minimal
case it is exactly $n$. That rank reveals the hidden state dimension provided
the data are rich enough and noise does not mask the singular-value gap.

Controllability and observability therefore enter in a precise way:

- If $(A,B)$ is not controllable, some internal modes cannot be excited by the
  input, so they do not appear reliably in input-output data.
- If $(A,C)$ is not observable, some internal modes do not leave a measurable
  output signature.
- A minimal realization is both controllable and observable, and only that
  minimal part can be identified from external behavior alone.

## Ho-Kalman and ERA

The Ho-Kalman algorithm uses the Hankel factorization directly. In the
noise-free setting, we compute a rank-$n$ singular value decomposition

```{math}
:label: eq:ch04-svd

H_0 \approx U_n \Sigma_n V_n^\top,
```

then define a balanced factorization

```{math}
:label: eq:ch04-balanced-factors

\widehat{\mathcal{O}}_s = U_n \Sigma_n^{1/2},
\qquad
\widehat{\mathcal{C}}_r = \Sigma_n^{1/2} V_n^\top.
```

One realization is then

```{math}
:label: eq:ch04-ho-kalman

\widehat{A}
=
\widehat{\mathcal{O}}_s^\dagger
H_1
\widehat{\mathcal{C}}_r^\dagger,
\qquad
\widehat{B}
=
\text{first } m \text{ columns of } \widehat{\mathcal{C}}_r,
\qquad
\widehat{C}
=
\text{first } p \text{ rows of } \widehat{\mathcal{O}}_s.
```

Because state coordinates are not unique, Ho-Kalman does not recover the
original $(A,B,C)$ exactly unless the coordinate basis is fixed in advance.
What it recovers is an equivalent realization with the same Markov parameters.

The eigensystem realization algorithm (ERA) is the same realization idea
applied to impulse-response data, often from experimental modal analysis. In
practice, ERA is valuable when the measured response to known forcing already
gives a clean estimate of the Markov parameters. Ho-Kalman and ERA therefore
share the same mathematical backbone: low-rank factorization of a Hankel matrix
built from impulse-response information.

## Subspace Identification from General Input-Output Data

Impulse-response estimation is not always the most convenient first step.
Subspace identification methods such as N4SID work more directly with measured
input-output trajectories. The data are arranged into block Hankel matrices of
past and future inputs and outputs. For a window length $s$, one typical choice
is

```{math}
:label: eq:ch04-data-hankel

\mathbf{U}_{\mathrm{p}}
=
\begin{bmatrix}
\mathbf{u}_0 & \mathbf{u}_1 & \cdots & \mathbf{u}_{N-2s+1} \\
\mathbf{u}_1 & \mathbf{u}_2 & \cdots & \mathbf{u}_{N-2s+2} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{u}_{s-1} & \mathbf{u}_{s} & \cdots & \mathbf{u}_{N-s}
\end{bmatrix},
\qquad
\mathbf{Y}_{\mathrm{f}}
=
\begin{bmatrix}
\mathbf{y}_{s} & \mathbf{y}_{s+1} & \cdots & \mathbf{y}_{N-s+1} \\
\mathbf{y}_{s+1} & \mathbf{y}_{s+2} & \cdots & \mathbf{y}_{N-s+2} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{y}_{2s-1} & \mathbf{y}_{2s} & \cdots & \mathbf{y}_{N}
\end{bmatrix},
```

with analogous definitions for $\mathbf{Y}_{\mathrm{p}}$ and
$\mathbf{U}_{\mathrm{f}}$. Each column stacks a time window of past or future
samples from the measured trajectories. The key idea is that the future outputs
live in a subspace generated by future inputs and the current state. By
projecting future outputs onto the orthogonal complement of future inputs, one
isolates the part of $\mathbf{Y}_{\mathrm{f}}$ explained by the hidden state.

In exact arithmetic, that state-dependent component has rank $n$. In noisy
data, one estimates its dominant singular subspace and uses it as a proxy for
the state sequence. A common workflow is:

1. Build block Hankel matrices from input-output records.
2. Remove the direct contribution of future inputs by projection or oblique
   projection.
3. Estimate the dominant rank-$n$ state subspace from an SVD.
4. Regress the shifted state sequence and outputs onto that estimated subspace
   to obtain $\widehat{A}$, $\widehat{B}$, $\widehat{C}$, and $\widehat{D}$.

This is why the method family is called subspace identification: it estimates
the state sequence only up to a linear coordinate transform, but that is enough
to recover an equivalent input-output model.

N4SID-style algorithms are attractive because they use linear algebra rather
than nonconvex optimization for the main realization step. They are often more
robust than directly fitting a high-order transfer function and then reducing
it, especially for multi-input multi-output data.

## Choosing the Hidden State Dimension

Partial-observation identification always includes an order-selection question.
Too small an order leaves systematic dynamics in the residuals; too large an
order fits noise and spurious lightly observable modes.

The first diagnostic is usually the singular-value decay of a Hankel or
projected data matrix. A sharp drop suggests an effective realization order,
but this should be checked against prediction and simulation behavior. Useful
questions include:

- Does the identified model reproduce the measured impulse or step response?
- Are the innovation or output residuals structured in time?
- Do additional states improve validation substantially, or only on the
  training trajectories?
- Are the added modes controllable and observable enough to matter for the
  intended use?

The correct order is therefore not purely algebraic. It depends on noise,
experiment design, and whether the model will be used for prediction,
estimation, or control.

## ARX, ARMAX, and Output-Error Models

State-space realizations are not the only way to model input-output data.
Prediction-error models describe the measured output directly in terms of past
outputs, past inputs, and sometimes filtered disturbances.

An ARX model takes the form

```{math}
:label: eq:ch04-arx

y_k
=
\sum_{i=1}^{n_a} A_i y_{k-i}
+
\sum_{j=0}^{n_b} B_j u_{k-j}
+
e_k,
```

where $e_k$ is an unexplained residual or innovation. ARX is attractive because
the coefficients enter linearly, so fitting can be reduced to least squares
once the model orders are fixed.

An ARMAX model adds a moving-average disturbance model:

```{math}
:label: eq:ch04-armax

y_k
=
\sum_{i=1}^{n_a} A_i y_{k-i}
+
\sum_{j=0}^{n_b} B_j u_{k-j}
+
\sum_{\ell=1}^{n_c} C_\ell e_{k-\ell}
+
e_k.
```

This better captures colored disturbances, but the coefficients no longer enter
as a single linear regression because the noise model depends on unobserved
errors.

Output-error models instead emphasize simulation fidelity:

```{math}
:label: eq:ch04-output-error

y_k = \widehat{y}_k(\theta) + e_k,
```

where $\widehat{y}_k(\theta)$ is generated by simulating a parametric
input-output model. Output-error fitting can perform well for low-noise
actuation data, but it is more sensitive to initialization and nonconvexity.

These model classes are not competitors to realization theory so much as
different abstractions:

- ARX provides a finite-memory predictor that is easy to estimate.
- ARMAX augments ARX with a disturbance model.
- Output-error models target simulation accuracy directly.
- State-space and subspace methods build an internal realization that supports
  modal analysis, estimation, and control design.

In practice, ARX-type fits are often useful baselines or initialization tools
even when the final model is a state-space realization.

## Special Cases and Links to Other Methods

Chapter 03 already introduced DMD as the autonomous full-state estimator
$\mathbf{X}_+\mathbf{X}_-^\dagger$. With control inputs and full-state
measurements, the analogous regression

```{math}
:label: eq:ch04-dmdc

\mathbf{X}_+
\approx
\begin{bmatrix}
A & B
\end{bmatrix}
\begin{bmatrix}
\mathbf{X}_- \\
\mathbf{U}_-
\end{bmatrix}
```

is the DMDc setting. It is still a full-state identification problem, not a
general partial-observation subspace method. Its role here is mainly to clarify
the boundary: once the state is measured, we are back in Chapter 03.

Hankel DMD sits on the other boundary. By stacking delayed measurements into a
Hankel matrix, it constructs a lifted coordinate system from output history.
That makes it closely related to delay embeddings and subspace ideas. The
method can be interpreted as using past outputs to approximate a latent state,
then applying DMD in that lifted space. This is why Hankel DMD belongs in the
conceptual neighborhood of subspace identification even though it is often
introduced through Koopman or delay-coordinate language.

## What This Chapter Adds

The main lesson of this chapter is that hidden state is not an obstacle to be
patched after the fact. It changes the identification problem itself.
Input-output data determine a realization only through its externally visible
behavior, so state dimension, controllability, observability, and disturbance
modeling all become part of the learning problem.

That perspective sets up later chapters in two ways. First, it motivates state
estimation methods, since hidden trajectories often need to be inferred rather
than directly measured. Second, it shows why linear algebraic constructions
such as Hankel factorizations can recover low-dimensional structure from long
histories. Subspace identification is therefore both a practical algorithmic
tool and a bridge between prediction-error models, realization theory, and
latent-state learning.
