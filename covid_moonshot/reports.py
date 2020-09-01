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

    def precision(self) -> int:
        return -floor(log10(self.stderr))


def binding_estimate_kcal(binding: Binding) -> Estimate:
    return Estimate(
        point=binding.delta_f * KT_KCALMOL, stderr=binding.ddelta_f * KT_KCALMOL
    )


def format_estimate_point(est: Estimate) -> str:
    rounded = round(est.point, est.precision())
    return (
        f"{rounded:.{est.precision()}f}"
        if est.point > 0
        else f'<span class="negative">−{abs(rounded):.{est.precision()}f}</span>'
    )


def format_estimate_stderr(est: Estimate) -> str:
    return f"{round(est.stderr, est.precision()):.{est.precision()}f}"


def get_index_html(runs: List[Run]) -> str:
    template = pkg_resources.read_text(templates, "index.html")
    environment = Environment()
    environment.filters["binding_estimate_kcal"] = binding_estimate_kcal
    environment.filters["format_estimate_point"] = format_estimate_point
    environment.filters["format_estimate_stderr"] = format_estimate_stderr
    return environment.from_string(template).render(sprint=SPRINT_NUMBER, runs=runs)


def save_reports(runs: List[Run], path: str) -> None:
    html = get_index_html(runs)
    with open(os.path.join(path, "index.html"), "w") as f:
        f.write(html)
