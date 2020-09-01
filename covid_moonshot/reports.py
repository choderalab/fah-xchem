from dataclasses import dataclass
import importlib.resources as pkg_resources
from math import floor, log10
import os
from typing import List, Tuple
from jinja2 import Environment
from .analysis.constants import KT_KCALMOL
from .core import Binding, Run
from . import templates


# TODO: read from configuration
SPRINT_NUMBER = 3


@dataclass
class Estimate:
    point: float
    stderr: float


def canonicalize(estimate: Estimate) -> Estimate:
    precision = -floor(log10(estimate.stderr))
    return Estimate(
        point=round(estimate.point, precision), stderr=round(estimate.stderr, precision)
    )


def binding_estimate_kcal(binding: Binding) -> Estimate:
    return Estimate(
        point=binding.delta_f * KT_KCALMOL, stderr=binding.ddelta_f * KT_KCALMOL
    )


def format_negative(number: float) -> str:
    return (
        f"{number}" if number > 0 else f'<span class="negative">−{abs(number)}</span>'
    )


def get_index_html(runs: List[Run]) -> str:
    template = pkg_resources.read_text(templates, "index.html")
    environment = Environment()
    environment.filters["canonicalize"] = canonicalize
    environment.filters["format_negative"] = format_negative
    environment.filters["binding_estimate_kcal"] = binding_estimate_kcal
    return environment.from_string(template).render(sprint=SPRINT_NUMBER, runs=runs)


def save_reports(runs: List[Run], path: str) -> None:
    html = get_index_html(runs)
    with open(os.path.join(path, "index.html"), "w") as f:
        f.write(html)
