# Linear-in-Parameter Nonlinear Dynamics and Sparse Discovery

Many nonlinear systems are not linear in the state, but they are close to
linear in a carefully chosen set of candidate functions. That is the main idea
of linear-in-parameter dynamics learning: instead of learning an arbitrary
nonlinear map directly, we choose a library of nonlinear features and fit the
dynamics as a linear combination of those features. This keeps the regression
problem relatively simple while still allowing rich nonlinear behavior.

This viewpoint is especially attractive when interpretability matters. A sparse
combination of candidate terms can often be read as a governing equation rather
than only as a predictor. The promise is scientific structure with modest data
requirements. The risk is that the discovered equation is only as good as the
library, the data quality, and the assumptions that justify sparsity.

## The Library View of Nonlinear Dynamics

The continuous-time version of the model writes the vector field as

```{math}
:label: eq:ch06-library-continuous

\dot{x}
=
\Theta(x,u;\mu)\xi.
```

The discrete-time analogue is

```{math}
:label: eq:ch06-library-discrete

x_{k+1}
=
\Theta(x_k,u_k;\mu)\xi.
```

Here $\Theta(x,u;\mu)$ is a feature or library map, and $\xi$ collects the
unknown coefficients. In the notation convention for this course, $\xi$ is used
for coefficients in a linear-in-parameter expansion, while $\theta$ is reserved
for more general nonlinear parameterizations. For vector-valued systems,
equations {eq}`eq:ch06-library-continuous` and
{eq}`eq:ch06-library-discrete` are shorthand for fitting one coefficient vector
per state component or, equivalently, a block of such vectors at once.

The important distinction is between nonlinear dependence on the state and
linear dependence on the coefficients. A library may contain terms such as
$1$, $x_i$, $x_i x_j$, $\sin(x_i)$, or $u_j x_i$, yet once those features are
chosen, estimating $\xi$ is still a regression problem.

Given sampled snapshots, we evaluate the library row by row to form the sampled
design matrix

```{math}
:label: eq:ch06-library-matrix

\mathbf{\Theta}
=
\begin{bmatrix}
\Theta(\mathbf{x}_0,\mathbf{u}_0)^\top\\
\Theta(\mathbf{x}_1,\mathbf{u}_1)^\top\\
\vdots\\
\Theta(\mathbf{x}_{N-1},\mathbf{u}_{N-1})^\top
\end{bmatrix}.
```

If derivatives are available or can be estimated, then each component of the
continuous-time dynamics can be fit from a regression of the form

```{math}
:label: eq:ch06-sparse-regression

\widehat{\xi}
=
\arg\min_{\xi}
\norm{\mathbf{\dot{x}}-\mathbf{\Theta}\xi}_2^2
+
\lambda_{\mathrm{reg}} \norm{\xi}_1.
```

Equation {eq}`eq:ch06-sparse-regression` captures the standard sparse-discovery
tradeoff: fit the observed dynamics well while preferring a model with few
active terms.

## Why Sparsity Is a Modeling Assumption

Without extra structure, a large library simply turns nonlinear modeling into
an ill-conditioned linear regression. Sparsity adds a modeling hypothesis: only
a small subset of candidate terms is truly active. In many scientific systems
this is plausible because governing equations are often short relative to the
space of all possible monomials, trigonometric terms, and cross-couplings that
could have been included.

That assumption is what makes sparse discovery different from generic basis
expansion. The goal is not only low prediction error, but recovery of a compact
equation with a useful mechanistic interpretation. If the true system is not
sparse in the chosen coordinates, then forcing sparsity can hide important
terms and produce a deceptively clean but wrong model.

## SINDy and Sparse Equation Discovery

Sparse Identification of Nonlinear Dynamics, or SINDy, is the canonical example
of this approach. Its workflow is conceptually simple:

1. Build a candidate library $\Theta$ from the measured states and inputs.
2. Estimate derivatives or use a derivative-free surrogate formulation.
3. Solve a sparse regression problem for each state equation.
4. Prune small coefficients and refit on the retained support.

The original attraction of SINDy is that it separates nonlinear feature design
from coefficient estimation. That makes it easy to inject prior knowledge into
the library and easy to inspect the discovered terms afterward.

SINDy with control extends the same idea to forced systems by including input
terms such as $u_j$, $x_i u_j$, or other control-relevant combinations in the
library. The conceptual caution is that the method must distinguish autonomous
dynamics from forced response. If the inputs are weakly exciting or strongly
correlated with state evolution, the coefficients attached to control terms can
be difficult to interpret.

## Structured Sparse Regression

Plain elementwise sparsity is only one option. Many systems have additional
structure that should enter the regression directly.

Group sparsity is useful when terms should appear or disappear together across
multiple state equations, operating conditions, or trajectories. Constrained
sparse regression is useful when the recovered model should satisfy known sign
patterns, symmetry relations, conservation laws, or zero-pattern restrictions.
These constraints reduce the search space and can prevent the algorithm from
using mathematically convenient but physically implausible terms.

The deeper lesson is that sparse discovery is not only about optimization. It
is about matching the regression problem to the structural assumptions of the
system. A good library with the wrong constraints can still fail, and a good
constraint set cannot rescue an incoherent library.

## Weak-Form and Integral SINDy

Derivative estimation is often the weakest part of continuous-time sparse
discovery. Finite differences amplify measurement noise, and smoothing before
differentiation introduces its own bias. For that reason, module 6 connects
directly to the derivative-free formulations from the previous chapter.

The integral viewpoint replaces pointwise derivative matching with a balance
over a time interval:

```{math}
:label: eq:ch06-integral-library

x(t_{k+1})-x(t_k)
=
\int_{t_k}^{t_{k+1}} \Theta(x(t),u(t);\mu)\xi \,\dd t.
```

Weak-form sparse discovery goes one step further by multiplying the dynamics by
test functions and integrating by parts. This shifts derivatives away from the
measured state and onto smooth test functions, which is often much more robust
in noisy data. The resulting regression is still linear in $\xi$, but the rows
of the sampled design matrix now come from integrated feature statistics rather
than pointwise snapshots.

Weak-form and integral SINDy therefore preserve the main sparse-discovery idea
while changing the data representation to improve robustness.

## Choosing the Library

Library design is the central modeling decision in this chapter. Common
choices include:

- Polynomial libraries for low-order smooth nonlinearities near a reference regime.
- Trigonometric libraries for oscillatory or angle-valued dynamics.
- Rational libraries for systems with saturating or quotient-type behavior.
- Physics-informed libraries built from dimensional analysis, symmetries, conservation structure, or known constitutive terms.

There is no universal best library. A very small library cannot represent the
true mechanism. A very large one creates collinearity, weak identifiability,
and unstable coefficient estimates. A physically mismatched library may fit the
training data yet fail badly outside the sampled regime because the active terms
are only compensating for missing structure elsewhere.

The practical goal is not maximum expressiveness in the abstract. It is a
library that is expressive enough to contain a plausible model and disciplined
enough that sparse regression can identify it from finite, noisy data.

## Failure Modes and Diagnostics

Three failure modes appear repeatedly in sparse discovery.

### Noise and Derivative Error

When $\mathbf{\dot{x}}$ is estimated poorly, the regression target is already
corrupted before sparsity enters. The result can be support instability,
spurious high-order terms, or coefficient magnitudes that change dramatically
under mild resampling.

### Collinearity and Weak Identifiability

Different library columns can look nearly identical on the sampled trajectories.
Then several coefficient patterns explain the data almost equally well. A sparse
solver may still return one answer, but that answer should not be mistaken for
unique scientific truth. This is an identifiability problem expressed through
the sampled library matrix $\mathbf{\Theta}$.

### Poor Library Design

If the true mechanism is absent from the library, the optimizer can only choose
the least-wrong combination of available terms. Sparse models are especially
vulnerable here because omitted physics is often absorbed into a few misleading
surrogates that look interpretable but are not transportable.

These failure modes are why model selection and coefficient uncertainty matter.
A discovered sparse model should be checked under perturbations of the data,
changes in the library, threshold sensitivity, and out-of-sample rollouts. If a
term appears only for one preprocessing choice or disappears under minor
resampling, it should be treated as weak evidence rather than a recovered law.

## What Sparse Discovery Can and Cannot Promise

Linear-in-parameter nonlinear modeling occupies an important middle ground. It
is more expressive than linear identification and often more interpretable than
fully black-box nonlinear models. It is especially valuable when the system is
believed to admit a compact representation in a meaningful coordinate system.

But sparse discovery is not automatic equation truth extraction. It depends on
state quality, excitation, coordinate choice, library design, sparsity
assumptions, and validation strategy. The right question is not "Did the solver
return a sparse model?" but "Why should this sparse model be trusted?"

## Summary

- Nonlinear dynamics can be written as regression over a candidate feature library that is nonlinear in state but linear in coefficients.
- SINDy is the archetypal sparse-discovery method, and its control, group-sparse, constrained, weak-form, and integral variants all modify the same core regression idea.
- Interpretability comes from structural assumptions, not from sparsity alone.
- Noise, collinearity, and poor library design are the main reasons sparse discovery fails in practice.
