import importlib.resources as pkg_resources
import os
from typing import List
from jinja2 import Environment
from .core import Run
from . import templates


# TODO: read from configuration
SPRINT_NUMBER = 3


def get_index_html(runs: List[Run]) -> str:
    template = pkg_resources.read_text(templates, "index.html")
    return Environment().from_string(template).render(sprint=SPRINT_NUMBER, runs=runs)


def save_reports(runs: List[Run], path: str) -> None:
    html = get_index_html(runs)
    with open(os.path.join(path, "index.html"), "w") as f:
        f.write(html)
