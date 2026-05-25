from helpers import CHAPTERS_DIR, DOCS_DIR, EXAMPLES_DIR, INDEX_PATH
from helpers import PY_EXAMPLES_DIR, load_conf, read_text
from helpers import index_toctree_entries


def test_sphinx_config_defines_reusable_infrastructure() -> None:
    conf = load_conf()
    macros = conf["mathjax3_config"]["tex"]["macros"]

    assert conf["project"] == "Course Notes"
    assert "myst_parser" in conf["extensions"]
    assert conf["numfig"] is True
    assert "js/course-interactives.js" in conf["html_js_files"]
    assert "js/course-page-toc.js" in conf["html_js_files"]
    assert "js/examples/demo-plot.js" in conf["html_js_files"]
    assert "js/examples/python-demo.js" in conf["html_js_files"]
    assert {"dd", "ddf", "norm", "ppf", "pppf"} <= set(macros)


def test_no_login_gate_assets_or_placeholders() -> None:
    conf = load_conf()
    index = read_text(INDEX_PATH)
    css = read_text(DOCS_DIR / "_static" / "css" / "course.css")

    assert "landing-gate" not in "\n".join(conf["html_js_files"])
    assert "course-landing-auth" not in index
    assert "course-landing" not in css


def test_index_toctree_entries_exist() -> None:
    missing = [
        entry
        for entry in index_toctree_entries()
        if not (DOCS_DIR / f"{entry}.md").is_file()
    ]

    assert missing == []


def test_configured_example_scripts_exist_and_register() -> None:
    conf = load_conf()

    for script in conf["html_js_files"]:
        if not script.startswith("js/examples/"):
            continue
        path = EXAMPLES_DIR / script.removeprefix("js/examples/")
        assert path.is_file()
        assert "registerExample(" in read_text(path)


def test_interactive_examples_use_data_example_only() -> None:
    chapter_text = "\n".join(read_text(path) for path in CHAPTERS_DIR.glob("*.md"))
    example_text = "\n".join(read_text(path) for path in EXAMPLES_DIR.glob("*.js"))
    loader_text = read_text(DOCS_DIR / "_static" / "js" / "course-interactives.js")

    assert "course-interactive-" not in chapter_text
    assert "selectors:" not in example_text
    assert "selectors" not in loader_text


def test_python_backed_example_files_exist() -> None:
    assert (PY_EXAMPLES_DIR / "python_demo.py").is_file()
    assert "py/examples/python_demo.py" in read_text(EXAMPLES_DIR / "python-demo.js")
