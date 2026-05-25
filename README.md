# Course Notes Skeleton

A reusable static Sphinx/MyST starter for course notes with:

- Markdown chapters in `docs/chapters/`
- MathJax macros configured in `docs/conf.py`
- A left navigation sidebar from Sphinx
- A generated right sidebar for the current page
- Browser-side interactive examples with lazy-loaded plotting/runtime libraries
- No login, account gate, or backend dependency

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build

```bash
make html
```

Run the full local check before review or handoff:

```bash
make check
```

## Serve Locally

```bash
make serve-html
```

Open `http://127.0.0.1:8765/docs/_build/html/`.

The local server workflow is recommended for interactive examples, especially
when an example fetches static assets such as Python source files.

When finished, stop the server with:

```bash
scripts/kill-local-http-server <pid>
```

To build, serve, fetch representative pages, and run tests in one command:

```bash
make check-local-html
```

For smoother Codex browser verification, allow `http://127.0.0.1:8765` in
`.codex/browser/config.toml`:
```toml
[origins]
allowed = ["http://127.0.0.1:8765"]
```

For smoother local command execution, allow these
prefix rules in `.codex/rules/default.rules`:

```python
prefix_rule(pattern=["make", "serve-html"], decision="allow")
prefix_rule(pattern=["make", "check-local-html"], decision="allow")
prefix_rule(pattern=["scripts/kill-local-http-server"], decision="allow")
prefix_rule(pattern=["scripts/prepare-template-sync"], decision="allow")
prefix_rule(pattern=["rm", "docs/_static/py/examples/__pycache__"], decision="allow")
```

## Sync From This Template

Course repositories created from this skeleton can use
`scripts/prepare-template-sync` to review updates from a fresh `rtd_skeleton`
commit without applying them immediately. Run it from a clean target course
worktree:

```bash
scripts/prepare-template-sync --template-repo ../rtd_skeleton --ref main
```

The script fetches the template ref, writes review artifacts under
`.git/template-sync/<timestamp>/`, and prints commands for reviewing the full
candidate diff and selecting individual hunks with `git restore -p`. The
generated artifact directory includes `stat.txt`, `files.txt`, `candidate.patch`,
and `select-hunks.sh`.

Useful options:

- `--select` starts interactive hunk selection after writing the artifacts.
- `--scaffold-only` excludes `docs/chapters/` so course content is left alone.
- `--include-syllabus` includes `syllabus/`, which is excluded by default.
- `--template-repo` and `--ref` choose a different template source or branch.

After selecting hunks, inspect the working tree, run the relevant checks, and
commit the accepted template updates.

## Add an Interactive Example

1. Add a semantic placeholder to a chapter:

   ```md
   :::{course-interactive}
   :data-example: my-example

   Interactive example loading...
   :::
   ```

2. Add `docs/_static/js/examples/my-example.js`.
3. Register the initializer with `CourseInteractives.registerExample("my-example", ...)`.
4. Add the script to `html_js_files` in `docs/conf.py`.

For Python-backed examples, keep reusable Python source under
`docs/_static/py/examples/`, load it through `CourseInteractives.staticAssetUrl`
when served over HTTP, and keep an embedded source fallback in the JavaScript
for `file://` previews.
