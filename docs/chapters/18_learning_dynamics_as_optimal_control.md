# Learning Dynamics as Optimal Control

Chapter 16 framed learning dynamics as trajectory optimization. This chapter
takes the next step and asks what changes when we interpret the learning task
through the language of optimal control. That shift is useful whenever the
unknowns affect an entire trajectory over a long horizon, because the data-fit
objective is then constrained by the dynamics in the same way that a control
problem is constrained by state evolution. The result is not that every
identification problem becomes a controller-synthesis problem, but that
adjoints, shooting methods, Hamiltonians, and experiment-design ideas become
natural tools for learning.

The control viewpoint is especially helpful for long-horizon identification.
Small parameter changes can produce large downstream trajectory changes, and
those changes accumulate through the dynamics rather than appearing as
independent regression residuals. Optimal-control structure makes that
dependence explicit.

## Parameter Estimation as a Dynamically Constrained Optimization Problem

Suppose the model class is known up to parameters $\theta$, and the measured
data are outputs $y_k$ generated from hidden states $x_k$. A basic
identification problem is

```{math}
:label: eq:ch18-strong-constraint

\begin{aligned}
\min_{x_0,\theta}\;
\mathcal{J}_{\mathrm{id}}(x_0,\theta)
&=
\frac{1}{2}
\sum_{k=0}^{N}
\norm{y_k-h(x_k)}_{R^{-1}}^2
+
\frac{\gamma_\theta}{2}\norm{\theta-\theta_{\mathrm{ref}}}^2 \\
\text{subject to}\qquad
x_{k+1} &= F_\theta(x_k,u_k),
\qquad k=0,\dots,N-1.
\end{aligned}
```

The decision variables are not only parameters but also the initial condition
$x_0$, because long-horizon mismatch can come from either incorrect dynamics or
incorrect trajectory initialization. Equation
{eq}`eq:ch18-strong-constraint` already looks like an optimal-control problem:
there is an objective, a state recursion, and a set of variables whose effect
propagates through that recursion.

This perspective differs from ordinary one-step regression in an important way.
Regression treats the targets as independent once the inputs are given, while
dynamically constrained learning respects the fact that the entire simulated
trajectory is coupled. That coupling is what makes long-horizon identification
hard, but it is also what lets optimal-control tools organize the computation.

## Model-Error Controls and Weak-Constraint Formulations

In practice, data mismatch is not explained only by the wrong choice of
$x_0$ or $\theta$. The model class itself may be imperfect. A weak-constraint
formulation introduces model-error controls $q_k$:

```{math}
:label: eq:ch18-weak-constraint

\begin{aligned}
\min_{x_{0:N},\theta,q_{0:N-1}}\;
\mathcal{J}(x_{0:N},\theta,q_{0:N-1})
&=
\frac{1}{2}
\sum_{k=0}^{N}
\norm{y_k-h(x_k)}_{R^{-1}}^2 \\
&\quad +
\frac{1}{2}
\sum_{k=0}^{N-1}
\norm{q_k}_{Q^{-1}}^2
+
\frac{\gamma_\theta}{2}\norm{\theta-\theta_{\mathrm{ref}}}^2 \\
\text{subject to}\qquad
x_{k+1} &= F_\theta(x_k,u_k)+q_k .
\end{aligned}
```

The variable

```{math}
:label: eq:ch18-model-error-control

q_k=x_{k+1}-F_\theta(x_k,u_k)
```

plays the role of a control input that compensates for model inadequacy. When
$Q$ is small, the optimization is pushed toward trajectories that closely obey
the model. When $Q$ is large, the problem tolerates more dynamical correction
in order to explain the observations. This is the same strong- versus
weak-constraint distinction that appears in data assimilation, but here it also
clarifies what kind of learning problem we are solving:

- If $q_k\equiv 0$, we are learning only parameters and initial conditions.
- If $q_k$ is penalized but nonzero, we are learning parameters while admitting
  model discrepancy.
- If $q_k$ is large or structured, the optimization starts to resemble
  residual-model or closure learning.

Seen this way, weak-constraint learning is optimal control over hidden
trajectory variables with model correction as an auxiliary control channel.

## Adjoint and Pontryagin Viewpoints

The next advantage of the control formulation is algorithmic. Instead of
differentiating the full simulation by brute force, we can introduce adjoint
variables $p_k$ and derive backward sensitivity recursions.

For the strong-constraint problem, define a stage loss

```{math}
:label: eq:ch18-stage-loss

\ell_k(x_k,\theta)
=
\frac{1}{2}\norm{y_k-h(x_k)}_{R^{-1}}^2,
\qquad
\ell_N(x_N,\theta)
=
\frac{1}{2}\norm{y_N-h(x_N)}_{R^{-1}}^2.
```

Then a discrete-time Hamiltonian can be written as

```{math}
:label: eq:ch18-hamiltonian

\mathscr{H}_k(x_k,u_k,p_{k+1};\theta)
=
\ell_k(x_k,\theta)
+
p_{k+1}^\top F_\theta(x_k,u_k).
```

The corresponding adjoint recursion is

```{math}
:label: eq:ch18-adjoint

\begin{aligned}
p_N &= \nabla_{x_N}\ell_N(x_N,\theta), \\
p_k &= \nabla_{x_k}\ell_k(x_k,\theta)
+
\left(D_x F_\theta(x_k,u_k)\right)^\top p_{k+1},
\qquad k=N-1,\dots,0.
\end{aligned}
```

Once the forward trajectory is known, {eq}`eq:ch18-adjoint` propagates
gradient information backward in time. The parameter gradient follows from the
same sweep:

```{math}
:label: eq:ch18-parameter-gradient

\nabla_\theta \mathcal{J}_{\mathrm{id}}
=
\gamma_\theta(\theta-\theta_{\mathrm{ref}})
+
\sum_{k=0}^{N-1}
\left(D_\theta F_\theta(x_k,u_k)\right)^\top p_{k+1}
+
\sum_{k=0}^{N}
\nabla_\theta \ell_k(x_k,\theta).
```

These equations are the learning analogue of Pontryagin's maximum principle.
In a classical control problem, the adjoint tells us how the objective changes
with respect to control actions. In dynamics learning, it tells us how the
objective changes with respect to parameters, initial conditions, and sometimes
model-error controls. The same mathematical object explains both tasks.

## Shooting and Control-Variable Perspectives

The simplest control-style solver is single shooting: optimize over a compact
set of variables such as $(x_0,\theta)$, simulate forward through the entire
horizon, and use adjoint gradients to update the optimizer. This is attractive
because it keeps the decision space small.

Single shooting can also be fragile. If the dynamics are unstable, chaotic, or
merely sensitive over long horizons, then a small perturbation in $\theta$ or
$x_0$ may produce a very large change in the terminal state. The resulting
optimization landscape becomes ill-conditioned.

Multiple shooting addresses that issue by introducing intermediate states
$z_j$ at segment boundaries and enforcing continuity as constraints. The
trajectory is then reconstructed from shorter windows rather than one long
rollout. The control viewpoint helps here too: the segment boundary states act
like additional control variables that stabilize the optimization.

Weak-constraint formulations go one step further. Once the $q_k$ variables from
{eq}`eq:ch18-model-error-control` are included, the optimizer can trade off
observation fit against dynamical fidelity at each time step. That is useful
when the model class is only approximately correct or when discretization error
should not be forced into the parameter vector.

## Dynamic Programming and HJB Connections

Adjoint methods give first-order optimality conditions, but optimal-control
theory also offers a value-function viewpoint. If we summarize the information
available at time $k$ by a belief state $b_k$ over hidden states and
parameters, then sequential learning design can be posed through a Bellman
recursion:

```{math}
:label: eq:ch18-bellman

V_k(b_k)
=
\min_{u_k\in\mathcal{U}}
\left[
\ell_{\mathrm{id}}(b_k,u_k)
+
\mathbb{E}\bigl[V_{k+1}(b_{k+1}) \mid b_k,u_k\bigr]
\right].
```

This is not how routine offline identification is usually solved, because the
belief state is high dimensional and the Bellman equation is expensive. Still,
the connection matters conceptually. It explains why long-horizon identification
is not only a curve-fitting problem but also a sequential decision problem: the
choice of present input affects future information.

In continuous time, the same idea leads to Hamilton-Jacobi-Bellman equations.
Those equations are rarely solved directly for model learning, but they justify
the language of value, information gain, and exploration-exploitation tradeoff
that appears in active system identification and reinforcement-learning
settings.

## Differentiable Simulation

Modern differentiable simulators package the adjoint idea into software.
Instead of hand-deriving every sensitivity equation, we define a simulator for
$F_\theta$ or $\dot{x}=f_\theta(x,u)$ and differentiate the resulting rollout
with respect to $\theta$ using automatic differentiation.

That convenience does not remove the optimal-control structure. It only hides
some of the algebra. The same questions remain:

- Is the discretization fine enough that the simulation gradient reflects the
  underlying model?
- Are unstable or stiff trajectories causing exploding or vanishing adjoints?
- Should gradients be computed through one long rollout, multiple shooting, or
  a weak-constraint formulation with explicit $q_k$ variables?

Differentiable simulation is therefore best understood as an implementation of
adjoint-based optimal-control ideas, not as a separate theory.

## Optimal Experiment Design and Active Data Acquisition

Once learning is posed as a sequential decision problem, it is natural to ask
how the inputs should be chosen to make the parameters more identifiable.
Optimal experiment design treats the excitation signal as a design variable.

For a parameterized observation model, one common proxy is the Fisher
information matrix

```{math}
:label: eq:ch18-fisher

\mathcal{I}(\theta)
=
\sum_{k=0}^{N}
\left(D_\theta \widehat{y}_k(\theta)\right)^\top
R^{-1}
\left(D_\theta \widehat{y}_k(\theta)\right),
\qquad
\widehat{y}_k(\theta)=h(x_k(\theta)).
```

Inputs can then be selected to maximize a scalar summary of
$\mathcal{I}(\theta)$, such as its determinant, trace, or smallest eigenvalue.
The point is not that Fisher information solves every identification problem,
but that control choices and learning quality are coupled. Rich excitation is
not an afterthought; it is part of the optimization problem.

This is where active data acquisition enters. Rather than passively fitting the
model to whatever data happen to be available, we design experiments that
reduce uncertainty in the directions that matter for prediction, explanation,
or downstream control.

## Why the Optimal-Control View Clarifies Long-Horizon Learning

Long-horizon identification often fails for reasons that are hard to see in a
pure regression formulation:

- Errors compound through the dynamics, so local mismatch can become large
  terminal deviation.
- Unknown initial conditions, model discrepancy, and parameter error can mimic
  one another unless the optimization separates their roles.
- Informative data require deliberate excitation, not only more samples.

The optimal-control viewpoint organizes all three issues. It separates state,
parameter, and model-error variables; it provides adjoint gradients for
long-horizon objectives; and it connects learning quality to the control inputs
that generate the data.

The main lesson of this chapter is that learning dynamics over trajectories is
not merely estimation with a long loss function. It is a dynamically
constrained decision problem. Optimal-control theory gives the right language
for that structure, and with it comes a principled way to reason about
gradients, horizon effects, model discrepancy, and informative experimentation.
