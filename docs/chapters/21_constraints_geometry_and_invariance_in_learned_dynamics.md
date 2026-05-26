# Constraints, Geometry, and Invariance in Learned Dynamics

By Chapter 20, the course has already seen that a dynamics model can be judged
by prediction accuracy, likelihood, state-estimation quality, or control
performance. Chapter 21 adds another criterion: many systems are not free to
evolve arbitrarily in $\mathbb{R}^n$. They obey algebraic constraints, conserve
quantities, dissipate energy, respect symmetry groups, live on manifolds, or
interact only through local couplings. When a learned model ignores that
structure, it may fit short trajectories and still fail in the ways that matter
most: it can drift off a constraint surface, invent or destroy conserved
quantities, violate passivity, or break equivariance under coordinate changes
or relabelings.

This changes the learning problem itself. We are no longer asking only for a
map or vector field that matches data. We are asking for one that remains
admissible under the geometric and physical rules of the system. Structure can
be enforced by choosing a model class that satisfies it automatically, by
imposing hard constraints during training, or by regularizing violations when
exact enforcement is impractical. The central lesson is that structure
preservation is not cosmetic prior knowledge. It often determines whether a
model extrapolates sensibly, composes with downstream algorithms, and remains
credible under long rollouts.

## Why Structure Changes the Learning Problem

An unconstrained model treats the state as if every nearby direction were
equally feasible. Many real systems do not behave that way. Mechanical systems
may be restricted to configuration manifolds, incompressible flows may obey
divergence constraints, network dynamics may couple only along graph edges, and
chemical or population models may preserve positivity or total mass.

As a result, two models with similar one-step error can behave very
differently. One may preserve the system's feasible set and qualitative
features under repeated simulation, while the other slowly accumulates
structure-breaking errors. That distinction matters especially when the learned
model will be rolled out, optimized, or embedded inside estimation and control
loops.

For a continuous-time model

```{math}
:label: eq:ch21-vector-field

\dot{x}=f_\theta(x,u),
```

the relevant question is no longer only whether $f_\theta$ matches observed
state changes. We also ask whether it lies in the correct tangent directions,
preserves the right invariants, and respects the intended symmetry and locality
properties of the original system.

## Algebraic Constraints and Tangent Dynamics

Suppose admissible states satisfy an algebraic constraint

```{math}
:label: eq:ch21-constraint-manifold

c(x)=0,
```

where $c:\mathbb{R}^n\to \mathbb{R}^q$. The feasible states lie on the set of
points where the constraint is satisfied. For the trajectory to remain on that
set, the vector field cannot point in an arbitrary direction. Differentiating
$c(x(t))=0$ along solutions gives the tangency condition

```{math}
:label: eq:ch21-tangency

\ddf{}{t} c(x(t))
=
J_c(x(t)) f_\theta(x(t),u(t))
=
0.
```

Equation {eq}`eq:ch21-tangency` is the local compatibility condition between
the learned dynamics and the constraint surface. If it fails, then the model
predicts motion that leaves the feasible set even when the training data all
lie on it.

Three common enforcement strategies are:

- By construction: parameterize $f_\theta$ so that it always lies in the
  tangent space of the constraint surface.
- By projection: first predict an unconstrained update, then project the state
  or vector field back onto the feasible set.
- By penalty or constrained optimization: add a loss term or explicit
  constraint that drives $\|J_c(x) f_\theta(x,u)\|_2^2$ toward zero on the
  training data.

The tradeoff is familiar. By-construction models preserve structure exactly but
may be harder to design. Penalties are flexible but only preserve the
constraint to the degree seen in training. Projection methods can repair
violations after the fact, but they separate the learned dynamics from the
evolution law actually simulated.

## Invariants and Conserved Quantities

Many systems preserve a scalar quantity $C(x)$ along trajectories. Examples
include mass, momentum, circulation, probability normalization, or a first
integral specific to the model class. Conservation means

```{math}
:label: eq:ch21-invariant

\ddf{}{t} C(x(t))
=
\nabla C(x(t))^\top f_\theta(x(t),u(t))
=
0.
```

This looks similar to the tangency condition, but the interpretation is
different. A constraint $c(x)=0$ defines where the state is allowed to live,
whereas an invariant $C(x)$ defines a quantity that must remain constant along
the motion.

For learned models, conservation laws are valuable for two reasons. First, they
encode real domain knowledge and reduce the hypothesis space. Second, they
offer powerful diagnostics during validation: if a model claims to represent a
conservative system yet shows systematic drift in $C(x)$ over long rollouts,
then short-horizon accuracy is not enough to trust it.

In practice, invariants can be built in through specialized architectures,
enforced with equality constraints, or encouraged through losses such as

```{math}
:label: eq:ch21-invariant-penalty

\mathcal{L}_{\mathrm{inv}}
=
\frac{1}{N}
\sum_{k=0}^{N-1}
\left|
\nabla C(x_k)^\top f_\theta(x_k,u_k)
\right|^2.
```

This does not guarantee exact conservation away from the sampled states, but it
does align training with the geometric property we care about.

## Dissipation, Lyapunov Structure, and Stability

Not every important quantity is preserved. Some systems are organized by
dissipation: energy decreases, entropy increases, or a Lyapunov function
certifies asymptotic stability. If $E(x)$ is an energy-like quantity, a
dissipative model should satisfy

```{math}
:label: eq:ch21-dissipation

\ddf{}{t} E(x(t))
=
\nabla E(x(t))^\top f_\theta(x(t),u(t))
\le 0.
```

This inequality is stronger than "the rollout looks stable on the training
set." It encodes a one-step directional rule that shapes all trajectories.

The modeling choices again follow the same pattern:

- By construction: represent the dynamics in a form that makes dissipation
  automatic.
- By certificate: learn $f_\theta$ together with a candidate Lyapunov function
  and constrain the sign of its derivative.
- By regularization: penalize positive values of $\nabla E(x)^\top f_\theta(x,u)$
  on sampled states.

The main caution is that a naive stability penalty can conflict with the real
system if the system is only locally stable, has multiple attractors, or is
externally forced. The structure must match the physics, not an overly simple
preference for monotone decay.

## Hamiltonian, Lagrangian, Port-Hamiltonian, and Dissipative Forms

Some geometric structure is richer than a single invariant. Classical
Hamiltonian systems, for example, encode dynamics through an energy function
$H(x)$ and a skew-symmetric interconnection operator. A canonical form is

```{math}
:label: eq:ch21-hamiltonian

\dot{x}=J(x)\nabla H(x),
\qquad
J(x)^\top=-J(x),
```

which immediately implies

```{math}
\ddf{}{t} H(x(t))
=
\nabla H(x(t))^\top J(x(t)) \nabla H(x(t))
=
0.
```

The point is not that every system is Hamiltonian. The point is that once a
system is known to belong to a structured class, learning should target the
structured ingredients rather than a generic vector field. Instead of learning
$f_\theta$ directly, we may learn an energy $H_\theta$, an interconnection
matrix, a dissipation operator, or a Lagrangian from which the evolution law is
derived.

Port-Hamiltonian and related dissipative formulations extend this idea to open
systems with inputs, outputs, and energy exchange. They are attractive because
passivity and energy bookkeeping become part of the model definition rather
than an afterthought. In gray-box settings, this often yields better
extrapolation than unconstrained residual learning because the correction term
must still respect the system's energetic structure.

## Symmetry and Equivariance

Many systems look the same after a transformation of coordinates, labels, or
reference frames. If a group action $g$ acts on the state, symmetry in the
dynamics is expressed through equivariance:

```{math}
:label: eq:ch21-equivariance

f_\theta(gx,u)=g\,f_\theta(x,u).
```

Equation {eq}`eq:ch21-equivariance` says that transforming the state and then
evaluating the vector field is equivalent to evaluating the vector field first
and transforming the result afterward. Rotation symmetry, translation
invariance, permutation symmetry, and frame indifference are all instances of
this general idea.

Ignoring symmetry wastes data and can produce inconsistent predictions. If two
states differ only by a symmetry transformation, then a non-equivariant model
may learn unrelated responses to situations that should be equivalent. By
contrast, an equivariant model shares information across symmetry-related
examples and reduces the effective size of the search space.

Common enforcement mechanisms include:

- Symmetry-aware coordinates or reduced variables.
- Equivariant architectures whose layers commute with the group action.
- Data augmentation over symmetry transformations when exact equivariance is
  not built into the model class.

Augmentation can help, but it is not identical to structural equivariance.
Built-in equivariance constrains the model everywhere, not only on transformed
copies of the observed data.

## Dynamics on Manifolds

Sometimes the state does not merely satisfy one or two constraints; it lives on
a manifold $\mathcal{M}$ such as a sphere, a rotation group, or a shape space.
Then the admissible vector field must satisfy

```{math}
:label: eq:ch21-manifold

f_\theta(x,u)\in T_x\mathcal{M},
```

where $T_x\mathcal{M}$ is the tangent space at the current state.

This viewpoint clarifies why ambient-coordinate learning can go wrong. If a
state on the unit sphere is evolved by an unconstrained Euclidean model, even
small local errors may push it off the sphere, after which subsequent
predictions no longer have the intended geometric meaning. Similar issues arise
for attitude dynamics, rigid-body motion, and latent representations designed
to encode reduced coordinates.

Two broad strategies are common:

- Intrinsic modeling: learn the dynamics directly in local or global manifold
  coordinates.
- Extrinsic modeling with correction: work in an ambient space but enforce
  tangency or retract predictions back onto the manifold.

The right choice depends on the geometry. Intrinsic coordinates avoid some
constraint-handling issues but may introduce singularities or chart changes.
Extrinsic methods are often easier to implement, but their correction step has
to be treated as part of the actual learned simulator.

## Locality, Graph Structure, and PDE-Inspired Inductive Bias

Geometry also appears through locality. In graph-based or spatially extended
systems, the evolution at one site often depends only on nearby states and
inputs. This is not a symmetry in the same sense as
{eq}`eq:ch21-equivariance`, but it is a structural restriction on admissible
couplings.

If the state is organized over nodes of a graph, permutation equivariance under
node relabeling is often required, while locality says that information should
propagate along edges or within finite neighborhoods. For PDE-like systems,
local differential operators and conservation fluxes play the same role:
updates should depend on nearby values and respect the spatial structure of the
domain.

These assumptions matter because a fully connected black-box model can fit
nonlocal spurious dependencies that never survive mesh refinement, topology
changes, or transfer to larger systems. Local architectures, message-passing
schemes, convolutional operators, and flux-based parameterizations restrict the
model class to something closer to the underlying mechanism.

## How to Enforce Structure in Practice

The chapter's themes can be organized into three broad implementation choices.

### Enforce by Construction

This is the strongest option. We design the model so that constraints,
conservation laws, symmetry, or dissipative structure hold identically. Examples
include tangent-space parameterizations, equivariant neural layers, Hamiltonian
neural networks, and conservative flux forms.

The benefit is exact structure preservation inside the model class. The cost is
reduced modeling flexibility and more design work up front.

### Enforce by Constrained Optimization

Here the model class is flexible, but training solves an optimization problem
with explicit equality or inequality constraints. This can be appealing when
the structure is known precisely but hard to encode architecturally.

The cost is algorithmic: constrained training is more complex, and feasibility
may only be checked on sampled states or collocation points rather than on the
entire state space.

### Enforce by Regularization

Regularization adds soft penalties for violating invariants, symmetry,
dissipation, or locality. This is usually the easiest approach and often a good
first step when the exact structure is approximate or only partly trusted.

The limitation is equally clear. A regularized model may preserve structure on
the training distribution yet drift under long extrapolations or under inputs
that were weakly represented in the data.

## Structure-Preserving Validation

Module 21 ends with a shift in validation philosophy. For structured systems,
prediction error alone is incomplete. Validation should ask whether the model
preserves the qualitative and geometric features it was meant to respect.

Typical diagnostics include:

- Constraint violation over long rollouts.
- Drift in conserved quantities or balance laws.
- Sign violations of dissipation or Lyapunov decay conditions.
- Equivariance error under known transformations.
- Distance from the target manifold after repeated simulation.
- Locality checks, such as whether perturbations propagate only through
  admissible couplings.

These checks are not secondary embellishments. They are often the actual test
of whether the learned dynamics remain faithful outside the interpolation regime
of the training set.

## Summary

Constraints, geometry, and invariance do not define one specialized corner of
dynamics learning. They expose a general principle: the most useful inductive
bias is often a structural statement about which motions are allowed and which
are forbidden. A learned model that respects those statements can generalize
with less data because it is prevented from using many implausible degrees of
freedom. A model that ignores them may fit quickly but fail exactly when the
course later asks it to extrapolate, estimate hidden states, or support
control.

That is why Chapter 21 belongs near the end of the sequence. By this point, the
course has many ways to fit dynamical models. The harder question is which of
those fits remain meaningful once physical admissibility, geometry, and
invariance are treated as part of the learning objective rather than as
optional postprocessing.
