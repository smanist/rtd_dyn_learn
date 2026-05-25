# Identification for Control and Closed-Loop Learning

The identification problem changes once the learned model is meant to sit
inside a feedback loop. A model that predicts open-loop trajectories well is
not automatically a good control model. Feedback concentrates the data in
visited regions of state space, changes the input distribution, and magnifies
errors that matter for stability, constraint satisfaction, and long-horizon
decision making. For control, the real question is not only "does the model fit
the data?" but also "does the model support safe and effective closed-loop
behavior?"

This chapter studies identification from that control-oriented viewpoint. The
main themes are closed-loop data, feedback-induced distribution shift,
persistent excitation under constraints, adaptive and dual control, model
predictive control with learned models, robust and uncertainty-aware MPC,
model-based reinforcement learning, and validation through closed-loop
performance.

## Why Control-Oriented Identification Is Different

In a standard prediction task, we often learn a model by minimizing one-step or
rollout error over a dataset. In control, the learned model is a component in a
decision rule. Consider a controlled discrete-time system

```{math}
:label: eq:ch19-controlled-dynamics

x_{k+1}=F_\theta(x_k,u_k),
\qquad
y_k=h(x_k).
```

When a controller is already in the loop, the applied input is not an
independent probe but a feedback law,

```{math}
:label: eq:ch19-feedback-law

u_k=\pi_k(y_{0:k},r_k),
```

where $r_k$ denotes a reference, task parameter, or planning signal. The data
therefore come from the joint behavior of plant and controller rather than from
an externally designed experiment. Identification is now entangled with the
policy that generated the observations.

This matters for at least three reasons:

- Feedback changes which states and inputs are visited.
- Stabilizing controllers often suppress informative transients.
- Small model errors can produce large control errors when optimization or
  long-horizon rollout is involved.

As a result, a model intended for regulation or planning should be judged by
how it supports closed-loop decisions, not only by how well it interpolates
historical samples.

## Closed-Loop Data and Feedback-Induced Distribution Shift

Suppose data are gathered while a nominal controller keeps the system near an
operating condition. Then the dataset is concentrated near the trajectories
that controller allows. A learned model may be highly accurate in that local
region yet unreliable elsewhere. This is a form of distribution shift induced
by feedback: the controller chooses future data based on the current model,
state estimate, and objectives.

In open-loop identification, one often thinks of a dataset as

```{math}
:label: eq:ch19-dataset

\mathcal{D}
=
\left\{
\left(\mathbf{u}^{(r)}_k,\mathbf{y}^{(r)}_k\right)_{k=0}^{N_r}
\right\}_{r=1}^{N_{\mathrm{traj}}}.
```

Under feedback, however, the stored inputs $\mathbf{u}^{(r)}_k$ are themselves
policy outputs. The dataset therefore reflects both plant dynamics and the
closed-loop sampling strategy. If the controller changes after each model
update, then the data distribution changes as well. That feedback-learning
coupling is central in adaptive control and model-based reinforcement learning.

The practical consequence is that identification for control should ask where
accuracy is needed. A model used only for local tracking may not need to be
globally correct. A model used inside a planner that explores wide regions of
state space needs much broader reliability.

## Learning Models That Are Useful for Feedback

Control rarely needs a model that is uniformly best in every metric. It needs a
model that is useful for decision making. A common closed-loop objective is

```{math}
:label: eq:ch19-closed-loop-objective

\mathcal{J}_{\mathrm{cl}}(\pi,F)
=
\mathbb{E}
\left[
\sum_{k=0}^{N-1}\ell(x_k,u_k)+\ell_f(x_N)
\right],
\qquad
u_k=\pi_k(y_{0:k},r_k).
```

This expression makes the main point explicit: the value of a learned model is
mediated by the controller it supports. Different controllers may tolerate
different modeling errors. For example, a robust controller may need certified
uncertainty bounds more than pointwise accuracy, while an economic MPC scheme
may need accurate gradients of the value-relevant dynamics.

This suggests several control-oriented modeling targets:

- A local linear or affine surrogate accurate near one operating regime.
- A reduced model that preserves the dominant input-output behavior.
- A nonlinear predictor with uncertainty quantification for planning.
- A residual model that improves a known nominal dynamics model where control
  performance is most sensitive.

The right target depends on the control architecture downstream.

## Persistent Excitation Under Safety and Actuation Constraints

Identification requires informative data, but feedback control often tries to
avoid the very transients that reveal the dynamics. This creates tension
between regulation and excitation. If the controller keeps the state too close
to equilibrium, important parameters may remain weakly identifiable.

The classical remedy is persistent excitation: design inputs so that the system
response contains enough variation to distinguish competing models. In a
control setting, that requirement must be balanced against state constraints,
input limits, and safety margins:

```{math}
:label: eq:ch19-constrained-control

x_k \in \mathcal{X},
\qquad
u_k \in \mathcal{U}.
```

Control-oriented identification therefore uses structured probing signals,
reference changes, exploratory maneuvers, or experiment schedules that enrich
the data without violating admissible sets. The key design question is not
simply "how do we excite the plant?" but "how do we excite the plant safely and
in the directions that matter for the controller?"

This point becomes sharper in high-stakes systems. Safe identification is often
about selecting experiments that reduce uncertainty in control-relevant modes
while keeping the system inside a verified operating envelope.

## Identification Under Feedback

Closed-loop data complicate classical system identification because regressors
and disturbances can become statistically dependent under feedback. Even when
the plant is linear, naive least-squares estimates may be biased if the inputs
correlate with unmodeled disturbances through the controller.

Conceptually, the problem is that the controller reacts to past outputs, and
those outputs contain both state information and noise. Then the input sequence
is no longer exogenous. Reliable identification under feedback therefore often
requires one or more of the following ideas:

- instrumental-variable or prediction-error methods that account for feedback
  correlations,
- experiment designs that inject independent excitation on top of the nominal
  control action,
- explicit disturbance, noise, or observer models,
- validation protocols based on simulated or real closed-loop performance
  rather than only residual whiteness or one-step error.

The core lesson is that closed-loop data are not unusable; they simply require
an identification viewpoint that respects how the controller generated them.

## Adaptive Control and the Dual-Control Tension

Adaptive control treats learning and control as simultaneous tasks. The model
parameters are updated online while the controller keeps operating. A simple
picture is

```{math}
:label: eq:ch19-adaptive-loop

\widehat{\theta}_{k+1}=\mathcal{A}(\widehat{\theta}_k,y_k,u_k),
\qquad
u_k=\pi_{\widehat{\theta}_k}(y_{0:k},r_k),
```

where $\mathcal{A}$ denotes the adaptation law or estimator. The controller
uses the current parameter estimate, and the future data depend on that
control. Learning and action are therefore coupled in both directions.

This leads to the dual-control problem. Good control actions exploit the
current model to reduce cost, but good learning actions may deliberately
explore uncertain regimes to improve the model. Pure exploitation can freeze
the information pattern and trap the learner in a poor local understanding of
the dynamics. Pure exploration can violate performance or safety requirements.

Dual control is the principled attempt to optimize both objectives at once.
Exact solutions are typically intractable because the controller must reason
about how current actions affect future knowledge. Still, the concept is
important because it explains why certainty-equivalent control can perform
poorly when uncertainty is large or the data are weakly informative.

## Model Predictive Control with Learned Models

Model predictive control is a natural place to use learned dynamics because MPC
repeatedly solves a finite-horizon planning problem with state and input
constraints. With a learned model $\widehat{F}$, a typical planning problem is

```{math}
:label: eq:ch19-mpc

\begin{aligned}
\min_{u_{0:N_{\mathrm{MPC}}-1}} \quad &
\sum_{k=0}^{N_{\mathrm{MPC}}-1}\ell(x_k,u_k)+\ell_f(x_{N_{\mathrm{MPC}}}) \\
\text{subject to} \quad &
x_{k+1}=\widehat{F}(x_k,u_k), \\
& x_k \in \mathcal{X}, \\
& u_k \in \mathcal{U}.
\end{aligned}
```

This architecture separates two questions:

1. Is the learned model accurate enough over the prediction horizon that the
   optimizer chooses sensible actions?
2. Is the uncertainty small enough, or bounded tightly enough, that constraint
   satisfaction remains credible?

For regulation problems, learned models are often combined with nominal
stability ingredients such as terminal penalties, terminal sets, local backup
controllers, or trusted nominal physics. For economic or trajectory-tracking
problems, the emphasis may shift toward accurate predictions in the parts of
state space that the optimizer will deliberately visit.

MPC also provides a concrete definition of control-oriented validation: the
model is good if the resulting receding-horizon controller behaves well.

## Robust and Uncertainty-Aware MPC

Once learned models are used inside optimization, uncertainty must be treated
as a first-class object rather than an afterthought. The relevant uncertainty
may come from limited data, approximation error, unmodeled disturbances, or
distribution shift between training and deployment.

Robust and uncertainty-aware MPC incorporate this uncertainty explicitly. The
main strategies include:

- robust constraints enforced for all models in an uncertainty set,
- tube-based designs that track a nominal prediction with a feedback-corrected
  error tube,
- stochastic or chance-constrained formulations that trade violation risk
  against performance,
- scenario-based planning using sampled model realizations.

These methods differ in conservatism, computational cost, and assumptions about
the uncertainty description. What they share is the recognition that a point
estimate $\widehat{F}$ is often not enough. For control, one usually needs some
description of what the model may get wrong and how that error propagates under
feedback.

## Model-Based Reinforcement Learning as Closed-Loop Identification

Model-based reinforcement learning can be viewed as identification for control
in a sequential-decision setting. The learner fits a dynamics model from
experience, uses it for planning or policy improvement, gathers new data under
the updated policy, and repeats. In that sense, model-based RL makes the
feedback-learning loop fully explicit.

The control-oriented questions remain the same:

- Is the learned model accurate in decision-relevant regions?
- Does the data collection strategy reduce epistemic uncertainty efficiently?
- Can exploration be kept safe while still improving the model?

What changes is the scale of the exploration problem and the typical lack of a
strong nominal model. Planning may be performed by rollout with $\widehat{F}$,
by shooting methods, by MPC-like receding-horizon optimization, or by policy
optimization through a differentiable simulator. In all cases, errors in the
model matter because they shape the policy that will collect the next batch of
experience.

Seen this way, model-based RL is not separate from identification for control;
it is one modern realization of it.

## Control-Oriented Validation

Prediction error alone is not an adequate final metric for control-oriented
learning. A model can achieve low one-step loss and still yield poor decisions
if it misses long-horizon drift, sensitivity to inputs, active constraints, or
stability-relevant modes.

Control-oriented validation asks whether the learned model supports the
intended closed-loop behavior. Useful evaluation criteria include:

- tracking and regulation performance under the deployed controller,
- cumulative closed-loop cost or regret,
- constraint violations and safety margins,
- robustness to disturbances, noise, and initial-condition mismatch,
- performance degradation under model mismatch or shifted operating regimes.

These metrics should be measured on the actual closed-loop task whenever
possible. If real experiments are expensive, then validation in a trusted
high-fidelity simulator can still reveal whether the learned model is adequate
for planning and regulation.

## Safe Exploration and Practical Design Principles

The need to learn while controlling motivates safe exploration. In practice,
this usually means restricting identification updates and probing actions to a
verified region where the controller can recover from errors. Typical design
principles are:

- start from a conservative nominal controller or model,
- add excitation gradually and monitor uncertainty or constraint margins,
- separate trusted operating regions from exploratory regions,
- use fallback policies when the learned model leaves its domain of validity,
- validate the model on the closed-loop objective before expanding authority.

These principles do not remove the exploration-exploitation tradeoff, but they
make it manageable in systems where unsafe experiments are unacceptable.

## Summary

Identification for control is about learning models that enable good feedback
decisions, not merely good open-loop prediction. Closed-loop data are shaped by
the policy that generated them, which creates feedback-induced distribution
shift and weakens naive identification assumptions. Persistent excitation must
be balanced against constraints and safety. Adaptive and dual control make the
learning-control coupling explicit. MPC, robust MPC, and model-based
reinforcement learning provide concrete architectures in which learned models
drive decisions. The final test is closed-loop performance: a model is useful
when it supports stable, safe, and effective behavior on the control task it
was learned to serve.
