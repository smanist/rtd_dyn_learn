# Controlled and Parametric Nonlinear Systems

Many of the identification problems in this course start from autonomous
nonlinear dynamics, but control and operating variation change the learning
task in essential ways. Once a system is driven by an input $u$ or indexed by a
known operating parameter $\mu$, the learner must separate three effects that
can be confounded in data: the system's intrinsic state evolution, the forced
response induced by actuation, and the variation of the dynamics across
regimes. A model that ignores any one of these can fit a narrow dataset while
failing to generalize to new commands or new operating conditions.

This chapter focuses on model classes and data requirements for those settings.
The goal is not only to write down controlled nonlinear dynamics, but to
understand what kinds of experiments identify them reliably and what kinds of
datasets leave important mechanisms ambiguous.

## Controlled and Parametric State-Space Models

The basic discrete-time and continuous-time forms are

```{math}
:label: eq:ch11-controlled-discrete

x_{k+1}=F(x_k,u_k;\mu,\theta),
```

and

```{math}
:label: eq:ch11-controlled-continuous

\dot{x}(t)=f(x(t),u(t);\mu,\theta).
```

Here, $u_k$ or $u(t)$ denotes the applied input, $\mu$ denotes a known
operating parameter that indexes a family of systems, and $\theta$ denotes
unknown parameters or learned weights. The notation matters: $\mu$ is not
something inferred from the same role as $\theta$. Instead, it tells the model
which member of a family is being observed, such as a Reynolds number, a mass
ratio, or a geometric setting.

The first modeling question is whether the state is available. If full-state
data are measured, then learning resembles regression on
{eq}`eq:ch11-controlled-discrete` or
{eq}`eq:ch11-controlled-continuous`. If only outputs are observed, we need an
observation relation such as

```{math}
:label: eq:ch11-controlled-observation

y_k=h(x_k,u_k)+v_k,
```

and the identification problem becomes nonlinear input-output learning or
joint state-model estimation.

## Separating Autonomous, Forced, and Parametric Effects

In a controlled nonlinear system, good prediction depends on disentangling
state dependence from input dependence. The learner should be able to answer
questions of the following form:

- What would the system do with zero input?
- How does the response change when the same state is driven by a different
  command?
- How does that response change again when $\mu$ varies?

Those distinctions are easy to blur in observational data. For example, if one
operating regime always uses one style of forcing and another regime always
uses a different style, then the data do not reveal whether the output shift is
caused by $\mu$, by the input distribution, or by both. In controlled learning,
data coverage has to span states, inputs, and operating conditions jointly.

One useful training objective is one-step prediction across multiple
trajectories and regimes:

```{math}
:label: eq:ch11-one-step-objective

\mathcal{J}_{\mathrm{train}}(\theta)
=
\sum_{r=1}^{N_{\mathrm{traj}}}
\sum_{k=0}^{N_r-1}
\left\|
\mathbf{x}_{k+1}^{(r)}
-
F_\theta\!\left(
\mathbf{x}_k^{(r)},
\mathbf{u}_k^{(r)};
\mu^{(r)}
\right)
\right\|^2.
```

Equation {eq}`eq:ch11-one-step-objective` makes the bookkeeping explicit:
training samples come from different trajectories $r$, each trajectory may have
its own operating parameter $\mu^{(r)}$, and the learned map must explain all
of them with one parameter vector $\theta$.

## Control-Affine and Structured Nonlinear Models

One common compromise between black-box flexibility and mechanistic structure is
the control-affine model

```{math}
:label: eq:ch11-control-affine

\dot{x}(t)=f(x(t);\mu)+g(x(t);\mu)u(t).
```

This form separates the drift term $f$ from the input channel $g$. It is
useful when physics suggests that the actuator enters linearly even though the
state dependence is nonlinear. The separation also helps interpretability:
equilibria of the unforced dynamics come from $f$, while control authority is
described by $g$.

A related structured class is the bilinear model,

```{math}
:label: eq:ch11-bilinear

x_{k+1}=A x_k+\sum_{j=1}^{m} u_{k,j} B_j x_k + B u_k,
```

or its continuous-time analogue. Bilinear models are not fully general
nonlinear systems, but they capture state-input coupling beyond a purely linear
state-space model. They often appear when a nonlinear system is approximated
locally or when actuation modulates part of the dynamics multiplicatively.

These structured classes are valuable when they are justified by the domain.
They reduce variance and improve interpretation, but they can also fail badly
if the true forcing mechanism lies outside the assumed structure.

## Nonlinear Input-Output Identification

When the state is hidden or not worth reconstructing explicitly, the learning
problem can be written directly in input-output form. A standard discrete-time
choice is a nonlinear autoregressive model with exogenous inputs (NARX):

```{math}
:label: eq:ch11-narx

y_{k+1}
=
\mathcal{N}_\theta\!\left(
y_k,\ldots,y_{k-n_y+1},
u_k,\ldots,u_{k-n_u+1};
\mu
\right).
```

The regressors are delayed outputs and delayed inputs. The model uses finite
memory to represent the effect of unobserved state. This is often effective in
practice when the observed signal already carries enough information about the
relevant history.

NARMAX models extend this idea by including moving-average noise terms. In
conceptual form, one writes

```{math}
:label: eq:ch11-narmax

y_{k+1}
=
\mathcal{N}_\theta\!\left(
y_{k:k-n_y+1},
u_{k:k-n_u+1},
\eta_{k:k-n_\eta+1};
\mu
\right),
```

where $\eta_k$ denotes a disturbance or innovation sequence. The advantage is that
serially correlated disturbances and unmodeled dynamics can be represented more
faithfully than in a pure NARX model. The price is a more delicate estimation
problem because the regressors now depend on unobserved noise terms.

Input-output models are attractive when a control engineer only measures
outputs, but their finite-memory structure should not be confused with a true
state realization. If the required memory length is large, that is evidence
that a state-space formulation may be the cleaner representation.

## LPV and Multi-Regime Parametric Models

When dynamics vary with a known scheduling or operating variable, an LPV
viewpoint can be more informative than fitting separate models for each regime.
In a discrete-time form,

```{math}
:label: eq:ch11-lpv

x_{k+1}=A(\mu_k)x_k+B(\mu_k)u_k,
```

the matrices depend on a known parameter $\mu_k$. The LPV model is linear in
state and input for fixed $\mu_k$, but nonlinear as a family across operating
conditions. This is useful when the nonlinear effect of regime variation is
stronger than the nonlinear effect of the state itself, or when local
linearizations are available across a parameter grid.

More generally, a nonlinear parametric family may be learned as

```{math}
:label: eq:ch11-parametric-family

x_{k+1}=F_\theta(x_k,u_k;\mu_k),
```

with $\mu_k$ or trajectory-level values $\mu^{(r)}$ supplied as inputs to a
shared model. That formulation supports multi-regime and multi-task learning:
the model pools information across related operating conditions while still
adapting predictions to a particular regime. The central question is whether
shared structure is real. If the regimes are genuinely related, pooling reduces
sample complexity. If they are too different, forced parameter sharing creates
bias.

## Data Requirements: Excitation, Regime Coverage, and Bias

Controlled nonlinear identification is only as good as the experiments used to
collect data. Three issues dominate practice.

### Persistent Excitation

Inputs must be rich enough to reveal how the system responds. If the forcing is
nearly constant, always tiny, or restricted to a narrow frequency band, then
many candidate models will agree on the observed trajectories. Persistent
excitation is the condition that the input sequence probes enough directions and
time scales to distinguish competing explanations.

For nonlinear systems, excitation is not just a property of $u_k$ alone. It is
a property of the state-input pairs actually visited. Strong actuation in one
corner of state space does not identify the dynamics elsewhere.

### Multi-Regime Coverage

Parametric learning requires coverage over $\mu$ as well as over $x$ and $u$.
Sparse sampling in the operating parameter may let a model interpolate between a
few regimes, but it will not justify claims about extrapolation to unseen
conditions. A chapter on parametric systems is therefore also a chapter on
experimental design: choose trajectories so that regime variation is informative
rather than incidental.

### Closed-Loop Data Bias

When data are collected under feedback, the input $u_k$ is correlated with the
state and with disturbances through the controller. That is exactly what makes
closed-loop operation useful, but it breaks the naive assumption that the input
is an external probing signal independent of model error. In identification,
that correlation can bias parameter estimates or overstate confidence in a
model's apparent accuracy. A dataset generated by a stabilizing controller may
also cover only a narrow neighborhood of the desired operating point.

These are not reasons to avoid closed-loop data. They are reasons to model the
data-generation process honestly and to validate on tasks that reflect the
closed-loop use case.

## Active Input Design and Control-Oriented Learning

If the learner can choose experiments, then identification and control start to
interact. Active input design asks which signals should be applied to maximize
information about the dynamics while respecting safety or operating
constraints. The best excitation signal for parameter learning may not be the
best signal for tracking or regulation, so data collection often alternates
between informative probing and nominal control objectives.

This tradeoff is especially important for nonlinear models intended for control.
A controller rarely needs globally accurate dynamics everywhere. It needs a
model that is reliable on the states, inputs, and operating parameters that the
closed-loop system will actually visit. That observation explains why local
surrogates, LPV models, and multi-regime models remain important even when more
expressive black-box architectures are available.

## Summary

The main lesson is that inputs and operating parameters are not minor
decorations on an autonomous model. They change the identifiability question,
the admissible model classes, and the experimental design problem. Controlled
and parametric nonlinear systems therefore sit between two themes in the
course:

- They extend state-space and nonlinear-identification ideas to forced and
  regime-dependent settings.
- They prepare the ground for later chapters on state estimation, optimal
  control, and identification for control, where the data-collection policy and
  the learned model can no longer be treated as independent.

When reading later chapters, return to the distinctions introduced here:
autonomous versus forced response, known operating variation versus unknown
model parameters, and open-loop excitation versus closed-loop data collection.
Those distinctions determine what information is present in the data before any
learning algorithm begins.
