from __future__ import annotations

from pathlib import Path
from typing import Any
import runpy


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
CHAPTERS_DIR = DOCS_DIR / "chapters"
EXAMPLES_DIR = DOCS_DIR / "_static" / "js" / "examples"
PY_EXAMPLES_DIR = DOCS_DIR / "_static" / "py" / "examples"
INDEX_PATH = DOCS_DIR / "index.md"
CONF_PATH = DOCS_DIR / "conf.py"


def read_text(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = REPO_ROOT / file_path
    return file_path.read_text(encoding="utf-8")


def load_conf() -> dict[str, Any]:
    return runpy.run_path(str(CONF_PATH))


def toctree_entries(path: str | Path = INDEX_PATH) -> list[str]:
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path

    entries: list[str] = []
    in_toctree = False

    for raw_line in read_text(source_path).splitlines():
        line = raw_line.strip()
        if line == "```{toctree}":
            in_toctree = True
            continue
        if in_toctree and line == "```":
            in_toctree = False
            continue
        if in_toctree and line and not line.startswith(":"):
            entries.append(line)

    return entries


def index_toctree_entries() -> list[str]:
    return toctree_entries(INDEX_PATH)
