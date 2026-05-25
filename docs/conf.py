from html import escape

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx import addnodes


project = "Course Notes"
author = "Course Staff"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
]

source_suffix = {
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "README.md"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]

myst_heading_anchors = 3
numfig = True

html_theme = "alabaster"
html_title = project
html_static_path = ["_static"]
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "navigation.html",
    ],
}

html_css_files = [
    "css/course.css",
]

html_js_files = [
    "js/course-interactives.js",
    "js/course-page-toc.js",
    "js/examples/demo-plot.js",
    "js/examples/python-demo.js",
]

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

html_theme_options = {
    "description": "Static Sphinx/MyST notes with browser-side examples",
    "fixed_sidebar": True,
}


def _doc_title(env, docname, explicit_title=None):
    if explicit_title:
        return explicit_title
    title_node = env.titles.get(docname)
    if title_node is not None:
        return title_node.astext()
    return docname.rsplit("/", 1)[-1].replace("_", " ").title()


def _toctree_entries(env, docname):
    doctree = env.get_doctree(docname)
    for node in doctree.findall(addnodes.toctree):
        for explicit_title, ref in node.get("entries", []):
            if ref in env.found_docs:
                yield {
                    "docname": ref,
                    "title": _doc_title(env, ref, explicit_title),
                }


class CourseInteractiveDirective(Directive):
    has_content = True
    option_spec = {
        "data-example": directives.unchanged_required,
        "name": directives.unchanged,
    }

    def run(self):
        self.assert_has_content()
        content = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, content)
        data_example = escape(self.options["data-example"], quote=True)

        return [
            nodes.raw(
                "",
                f'<div class="course-interactive" data-example="{data_example}">',
                format="html",
            ),
            *content.children,
            nodes.raw("", "</div>", format="html"),
        ]


def add_course_sidebar_context(app, pagename, templatename, context, doctree):
    parts = pagename.split("/")
    group_index = None
    for length in range(len(parts) - 1, 0, -1):
        candidate = "/".join([*parts[:length], "index"])
        if candidate != pagename and candidate in app.env.found_docs:
            group_index = candidate
            break

    sidebar_items = []
    group_children = list(_toctree_entries(app.env, group_index)) if group_index else []
    for number, item in enumerate(_toctree_entries(app.env, app.config.master_doc), 1):
        item = dict(item)
        item["number"] = number
        item["current"] = pagename == item["docname"] or group_index == item["docname"]
        item["children"] = group_children if group_index == item["docname"] else []
        sidebar_items.append(item)

    context["course_group_child"] = group_index is not None
    context["course_group_index"] = group_index
    context["course_sidebar_items"] = sidebar_items


def setup(app):
    app.add_directive("course-interactive", CourseInteractiveDirective)
    app.connect("html-page-context", add_course_sidebar_context)
