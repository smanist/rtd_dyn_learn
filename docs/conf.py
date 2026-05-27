# a-rtd:begin managed name="sphinx-myst-course" version="0.1.0"
from a_rtd.sphinx_ext import apply_defaults

project = "Course Notes"
author = "Course Staff"
a_rtd_theme_description = "Static Sphinx/MyST notes with browser-side examples"
a_rtd_example_js_files = ["js/examples/demo-plot.js", "js/examples/python-demo.js"]
extensions = ["a_rtd.sphinx_ext"]
templates_path = ["_templates"]

apply_defaults(globals())
# a-rtd:end managed

mathjax3_config = {
    "tex": {
        "macros": {
            "dd": r"\mathrm{d}",
            "ppf": [r"\frac{\partial #1}{\partial #2}", 2],
            "pppf": [r"\frac{\partial^2 #1}{\partial #2^2}", 2],
            "ddf": [r"\frac{\mathrm{d} #1}{\mathrm{d} #2}", 2],
            "norm": [r"\left\lVert #1 \right\rVert", 1],
            "mbf": r"\mathbf",
            "mcl": r"\mathcal",
            "mbb": r"\mathbb",
            "Re": r"\mathrm{Re}",
            "Im": r"\mathrm{Im}",
        },
    },
}

apply_defaults(globals())
