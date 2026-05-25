# Getting Started

This placeholder chapter shows the basic structure for course content.

## Math

Use MyST math fences for display equations:

```{math}
:label: eq:sample-ode

\ddf{x}{t} = f(t, x).
```

You can reference the equation with {eq}`eq:sample-ode`.

## Figures

Keep reusable figures under `docs/pics/` or another documented static asset
folder, then reference them with MyST image or figure directives.

## Sections

Use one page-level heading per chapter. Use second-level headings for sections
and third-level headings for subsections so the generated sidebars remain tidy.
