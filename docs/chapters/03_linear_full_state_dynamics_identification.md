# Linear Full-State Dynamics Identification

Linear full-state identification is the simplest setting in system
identification: the state is measured directly, and the model class is
restricted to linear dynamics. That simplicity makes it foundational. It is
the first place where we can see how regression, excitation, conditioning,
noise, stability, and spectral interpretation all interact without the extra
complication of hidden states.

The scope of this chapter is deliberately narrow. We assume that the full state
is available in the data and ask how to estimate linear state-space dynamics
from those measurements. When the state is hidden and only outputs are
available, the problem changes substantially and becomes the subject of
input-output and subspace identification.

## Full-State Linear Models

For sampled dynamics with direct state measurements, the standard controlled
discrete-time model is

```{math}
:label: eq:ch03-discrete-linear-model

x_{k+1}=Ax_k+Bu_k.
```

Here, $A\in \mathbb{R}^{n\times n}$ is the state-transition matrix and
$B\in \mathbb{R}^{n\times m}$ maps inputs into the state update. The
autonomous case is the special case $B=0$.

Given stored trajectory data, assemble the snapshot matrices

```{math}
:label: eq:ch03-snapshot-matrices

\mathbf{X}_-
=
\begin{bmatrix}
\mathbf{x}_0 & \mathbf{x}_1 & \cdots & \mathbf{x}_{N-1}
\end{bmatrix},
\qquad
\mathbf{X}_+
=
\begin{bmatrix}
\mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_{N}
\end{bmatrix},
\qquad
\mathbf{U}_-
=
\begin{bmatrix}
\mathbf{u}_0 & \mathbf{u}_1 & \cdots & \mathbf{u}_{N-1}
\end{bmatrix}.
```

Under the model {eq}`eq:ch03-discrete-linear-model`, these data should satisfy

```{math}
:label: eq:ch03-snapshot-regression

\mathbf{X}_+
\approx
A\mathbf{X}_-+B\mathbf{U}_-
=
\begin{bmatrix}
A & B
\end{bmatrix}
\begin{bmatrix}
\mathbf{X}_- \\
\mathbf{U}_-
\end{bmatrix}.
```

This is a regression problem in matrix form. The unknown is not a trajectory or
a nonlinear feature map, but the matrices $A$ and $B$ themselves.

## Least Squares and Its Rank Conditions

If we define

```{math}
:label: eq:ch03-compact-regression

\mathbf{Z}_-
=
\begin{bmatrix}
\mathbf{X}_- \\
\mathbf{U}_-
\end{bmatrix},
\qquad
M=
\begin{bmatrix}
A & B
\end{bmatrix},
```

then {eq}`eq:ch03-snapshot-regression` becomes $\mathbf{X}_+\approx M\mathbf{Z}_-$.
The ordinary least-squares estimate solves

```{math}
:label: eq:ch03-least-squares

(\widehat{A},\widehat{B})
\in
\arg\min_{A,B}
\norm{\mathbf{X}_+-A\mathbf{X}_--B\mathbf{U}_-}_F^2.
```

When the minimizer is unique, the compact solution is

```{math}
:label: eq:ch03-pseudoinverse-solution

\widehat{M}
=
\mathbf{X}_+\mathbf{Z}_-^\dagger,
\qquad
\widehat{M}
=
\begin{bmatrix}
\widehat{A} & \widehat{B}
\end{bmatrix}.
```

In the autonomous case, this reduces to

```{math}
:label: eq:ch03-autonomous-least-squares

\widehat{A}
=
\mathbf{X}_+\mathbf{X}_-^\dagger.
```

The algebra is straightforward, but the data requirement is not. To identify
$A$ and $B$ from full-state data, the stacked regressor matrix $\mathbf{Z}_-$
must contain enough independent variation to distinguish every column of $M$.
The practical rank condition is

```{math}
:label: eq:ch03-rank-condition

\operatorname{rank}(\mathbf{Z}_-)=n+m
```

for the controlled case, or $\operatorname{rank}(\mathbf{X}_-)=n$ in the
autonomous case. This is the full-state version of persistence of excitation:
the state and input histories must span the directions we are trying to learn.
If the rank condition fails, many different linear models interpolate the same
data equally well.

Even when the rank is formally sufficient, poor conditioning can make
$\widehat{A}$ and $\widehat{B}$ highly sensitive to noise. A trajectory that
explores only a narrow subspace may produce a nearly singular $\mathbf{Z}_-$,
which means the numerical problem is ill posed even before we discuss model
misspecification.

## Noise Bias, Total Least Squares, and Regularization

Ordinary least squares treats the regressor matrix as if it were known exactly
and places all mismatch in the residual on the left-hand side. That assumption
is reasonable when inputs are prescribed accurately and state measurements are
high quality. It becomes questionable when the same noisy state data appear in
both $\mathbf{X}_-$ and $\mathbf{X}_+$.

This is an errors-in-variables problem. If the snapshots are noisy, then
ordinary least squares typically biases the estimate toward overly damped
dynamics or otherwise distorted spectra. Total least squares addresses that
issue by allowing perturbations on both sides of
{eq}`eq:ch03-snapshot-regression`. In schematic form, it solves

```{math}
:label: eq:ch03-total-least-squares

\min_{M,\Delta \mathbf{X}_+,\Delta \mathbf{Z}_-}
\left\|
\begin{bmatrix}
\Delta \mathbf{X}_+ & \Delta \mathbf{Z}_-
\end{bmatrix}
\right\|_F^2
\quad
\text{subject to}
\quad
\mathbf{X}_+ + \Delta \mathbf{X}_+
=
M(\mathbf{Z}_- + \Delta \mathbf{Z}_-).
```

The point is not that total least squares is always preferable, but that full
state does not mean noise-free. Once state measurements are contaminated,
estimator choice affects bias as much as variance.

Regularization is the second major correction. When $\mathbf{Z}_-$ is noisy or
poorly conditioned, ridge-style fitting shrinks the coefficients and reduces
sensitivity:

```{math}
:label: eq:ch03-ridge

(\widehat{A},\widehat{B})
\in
\arg\min_{A,B}
\norm{\mathbf{X}_+-A\mathbf{X}_--B\mathbf{U}_-}_F^2
+
\lambda
\norm{\begin{bmatrix} A & B \end{bmatrix}}_F^2,
\qquad
\lambda > 0.
```

Regularization trades some bias for robustness. In practice, that trade is
often favorable when the dataset is short, the state components have very
different scales, or the experiment excites only a small range of the linear
dynamics.

## Stable Linear Model Fitting

A good least-squares fit is not automatically a good dynamical model. With
finite or noisy data, the unconstrained optimum can yield an unstable
$\widehat{A}$ even when the underlying system is stable. If the learned model
is meant to represent dissipative dynamics, stability may need to be enforced
explicitly during fitting.

For autonomous discrete-time models, asymptotic stability requires the
eigenvalues of $A$ to lie inside the unit disk:

```{math}
:label: eq:ch03-discrete-stability

\rho(A)<1,
```

where $\rho(A)$ is the spectral radius. One constrained formulation is

```{math}
:label: eq:ch03-stability-constrained-fit

\widehat{A}
\in
\arg\min_A
\norm{\mathbf{X}_+-A\mathbf{X}_-}_F^2
\quad
\text{subject to}
\quad
\rho(A)\leq 1-\varepsilon
```

for some margin $\varepsilon>0$. Related continuous-time constraints require
all eigenvalues of $A$ to satisfy $\mathrm{Re}(\lambda)<0$.

These constraints are not merely cosmetic. An unstable estimate can produce
acceptable one-step error while exploding in rollout, which makes it unusable
for long-horizon prediction, model reduction, or downstream control design.

## Continuous-Time Identification

The continuous-time linear model is

```{math}
:label: eq:ch03-continuous-linear-model

\dot{x}(t)=Ax(t)+Bu(t).
```

One route is direct regression on estimated derivatives:

```{math}
:label: eq:ch03-derivative-regression

\dot{\mathbf{X}}
\approx
A\mathbf{X}+B\mathbf{U}.
```

This keeps the continuous-time parameters explicit, but it inherits the
difficulty of estimating derivatives from sampled and noisy trajectories. The
advantage of the linear setting is that there is also a second route: identify
a discrete-time model first and then interpret it as the sampled image of a
continuous-time generator.

Under uniform sampling with step size $\Delta t$ and zero-order-hold inputs,
the exact discrete-time matrices satisfy

```{math}
:label: eq:ch03-discrete-from-continuous

x_{k+1}=A_{\mathrm{d}}x_k+B_{\mathrm{d}}u_k,
\qquad
A_{\mathrm{d}}=e^{A\Delta t},
\qquad
B_{\mathrm{d}}=\int_0^{\Delta t} e^{A\tau}B\,\dd \tau.
```

In the autonomous case, if $\widehat{A}_{\mathrm{d}}$ is a reliable estimate
of $A_{\mathrm{d}}$, then a continuous-time generator can be recovered through
the matrix logarithm:

```{math}
:label: eq:ch03-matrix-logarithm

\widehat{A}
=
\frac{1}{\Delta t}\log(\widehat{A}_{\mathrm{d}}).
```

This viewpoint is powerful because it separates fitting at the sample level
from interpretation in continuous time. It also requires care: the matrix
logarithm is sensitive to noise, branch choices matter, and not every estimated
discrete-time matrix corresponds cleanly to the stable generator one expects.

## Spectral and Dynamical Interpretation

A learned linear model is useful not only for prediction but also for analysis.
Its eigenvalues and eigenvectors summarize growth, decay, oscillation, and
modal structure.

In discrete time, eigenvalues $\lambda_i(A)$ near the unit circle indicate
slowly decaying or weakly unstable modes. Complex-conjugate pairs encode
oscillations, with magnitude controlling decay or growth per sample. In
continuous time, the real part of an eigenvalue determines exponential
growth or decay, and the imaginary part determines oscillation frequency.

This spectral viewpoint is essential, but it is not the whole story. Non-normal
matrices can exhibit large transient amplification even when every eigenvalue
suggests asymptotic stability. That is why linear identification should be
interpreted dynamically as well as spectrally: inspect rollouts, modal content,
and sensitivity, not only the spectrum.

## Where DMD Fits

Dynamic Mode Decomposition is best viewed as a special case of the full-state
linear identification problem, not as a separate modeling philosophy. In its
basic autonomous form,

```{math}
:label: eq:ch03-dmd

A_{\mathrm{DMD}}
=
\mathbf{X}_+\mathbf{X}_-^\dagger,
```

which is exactly the unregularized least-squares estimator
{eq}`eq:ch03-autonomous-least-squares`. What DMD adds is a computational
workflow and a modal interpretation, not a fundamentally different estimation
problem.

That perspective prevents two common confusions. First, DMD does not solve the
partial-observation problem; it still assumes access to full-state snapshots or
a chosen proxy representation. Second, DMD does not remove the need for
excitation, conditioning, noise handling, or stability checks. It inherits all
of those issues because it is one instance of linear full-state identification.

## Summary

Linear full-state identification is the right baseline whenever direct state
measurements make the hidden-state problem unnecessary and a linear model is a
reasonable approximation. In that setting, the main questions are not whether
we can write down a regression, but whether the data are informative enough,
whether noise biases the estimate, whether regularization or stability
constraints are needed, and whether the learned spectrum supports the intended
interpretation.

Those questions reappear throughout the rest of the course. The setting will
become more complicated, but the core lesson stays the same: identification is
never just curve fitting. It is the combination of model class, data richness,
estimator choice, and dynamical interpretation.
