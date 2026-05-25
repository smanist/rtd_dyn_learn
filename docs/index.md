# Course Notes

These notes are built with Sphinx and MyST Markdown. Chapters can use ordinary
Markdown, Sphinx cross-references, figures, and LaTeX equations.

```{toctree}
:maxdepth: 1
:caption: Contents

chapters/01_problem_formulation_what_does_it_mean_to_learn_dynamics
chapters/02_data_noise_preprocessing_and_validation_metrics
chapters/03_linear_full_state_dynamics_identification
chapters/04_input_output_and_subspace_identification
chapters/05_continuous_time_identification_and_derivative_free_formulations
chapters/06_linear_in_parameter_nonlinear_dynamics_and_sparse_discovery
chapters/getting-started
chapters/interactive-example
```

## Site Conventions

- Write chapters as Markdown files in `docs/chapters/`.
- Put reusable browser code in `docs/_static/js/`.
- Put small embedded example placeholders in the Markdown where each
  interactive example should appear.
- Keep heavyweight libraries pinned by exact CDN version until the site needs
  vendored or offline assets.
