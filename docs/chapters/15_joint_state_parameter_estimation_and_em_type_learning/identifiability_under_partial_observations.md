## Identifiability Under Partial Observations

Joint estimation fails most often because the data do not separate hidden-state
uncertainty from parameter uncertainty. Several pathologies are common:

- Different parameter values can produce nearly identical outputs after the
  hidden state is reoptimized.
- A poor initial state can mimic the effect of a parameter perturbation over
  short horizons.
- Weak excitation can leave some components of $\theta$ effectively
  unobserved.
- Process noise and parameter mismatch can compensate for each other, making
  model error hard to localize.

These are not merely numerical annoyances. They indicate that the likelihood or
posterior has flat directions, ridges, or multiple modes. In practice, one
should inspect sensitivity to priors, initialization, smoothing assumptions,
and input design rather than trusting a single fitted parameter vector.
