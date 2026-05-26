# Joint State-Parameter Estimation and EM-Type Learning

When the state trajectory is hidden, system identification is no longer only a
parameter-fitting problem. The unknown dynamics parameter $\theta$ and the
unknown trajectory $x_{0:N}$ must be inferred together from noisy, partial
observations. That joint viewpoint is the natural continuation of the previous
chapters on filtering and smoothing: once trajectories are treated as latent
variables, learning dynamics becomes a latent-variable inference problem rather
than a direct regression on observed data.

This shift matters for both statistics and algorithms. Statistically, the
relevant objective is no longer a simple prediction loss on observed states,
because the observations may not reveal the latent trajectory uniquely.
Algorithmically, many identification methods can be organized around repeated
state estimation and parameter updates. Expectation-maximization (EM),
augmented-state filtering, dual estimation, Laplace approximations, and
variational inference all fit into that pattern, even though they make
different approximations.

**Sections**

- [State-Space Formulation with Unknown Parameters - Augmented-State and Dual Estimation Viewpoints](state_space_formulation_with_unknown_parameters.md)
- [EM as Repeated Smoothing and Parameter Update - Approximate E-Steps for Nonlinear Models](em_as_repeated_smoothing_and_parameter_update.md)
- [MAP and Laplace Viewpoints - Variational Inference and ELBO Formulations](map_and_laplace_viewpoints.md)
- [Identifiability Under Partial Observations](identifiability_under_partial_observations.md)

```{toctree}
:hidden:
:maxdepth: 1

state_space_formulation_with_unknown_parameters
em_as_repeated_smoothing_and_parameter_update
map_and_laplace_viewpoints
identifiability_under_partial_observations
```

**Summary**

The main lesson of joint state-parameter estimation is that learning dynamics
from partial observations is fundamentally an inference problem over latent
trajectories. EM makes that explicit by alternating smoothing and parameter
updates. Augmented-state and dual estimators provide online approximations.
MAP and Laplace methods recast the same problem as optimization plus local
uncertainty quantification. Variational inference generalizes the idea to
broader approximate posterior families.

All of these methods live or die by the same structural questions: are the
hidden trajectories sufficiently observable, are the parameters sufficiently
identifiable, and do the chosen approximations represent the posterior
coupling between states and parameters well enough for the intended use? Those
questions will reappear in the next chapter when the same learning problem is
viewed through trajectory optimization and data assimilation.
