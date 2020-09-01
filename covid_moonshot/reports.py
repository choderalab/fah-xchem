import importlib.resources as pkg_resources
from math import floor, log10
import os
from typing import List
from jinja2 import Environment
from .analysis.constants import KT_KCALMOL
from .core import Binding, Run
from . import templates


# TODO: read from configuration
SPRINT_NUMBER = 3


def format_uncertainty(estimate: float, stderr: float) -> str:
    precision = -floor(log10(stderr))
    return f"{round(estimate, precision)} ± {round(stderr, precision)}"


def format_binding(binding: Binding) -> str:
    return format_uncertainty(
        binding.delta_f * KT_KCALMOL, binding.ddelta_f * KT_KCALMOL
    )


def get_index_html(runs: List[Run]) -> str:
    template = pkg_resources.read_text(templates, "index.html")
    environment = Environment()
    environment.filters["format_binding"] = format_binding
    return environment.from_string(template).render(sprint=SPRINT_NUMBER, runs=runs)


def save_reports(runs: List[Run], path: str) -> None:
    html = get_index_html(runs)
    with open(os.path.join(path, "index.html"), "w") as f:
        f.write(html)
