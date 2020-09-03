from dataclasses import dataclass
import importlib.resources as pkg_resources
from math import floor, isfinite, log10
import os
from typing import List, Optional
from jinja2 import Environment
from ..analysis.constants import KT_KCALMOL
from ..core import Analysis, Binding
from . import templates


# TODO: read from configuration
SPRINT_NUMBER = 3


@dataclass
class Estimate:
    point: float
    stderr: float

    def precision(self) -> Optional[int]:
        return -floor(log10(self.stderr)) if isfinite(self.stderr) else None


def binding_estimate_kcal(binding: Binding) -> Estimate:
    return Estimate(
        point=binding.delta_f * KT_KCALMOL, stderr=binding.ddelta_f * KT_KCALMOL
    )


def format_estimate_point(est: Estimate) -> str:
    prec = est.precision()
    if prec is None or not isfinite(est.point):
        return ""
    rounded = round(est.point, prec)
    return (
        f"{rounded:.{prec}f}"
        if est.point > 0
        else f'<span class="negative">−{abs(rounded):.{prec}f}</span>'
    )


def format_estimate_stderr(est: Estimate) -> str:
    prec = est.precision()
    if prec is None or not isfinite(est.point):
        return ""
    return f"{round(est.stderr, prec):.{prec}f}"


def get_index_html(analysis: Analysis) -> str:
    template = pkg_resources.read_text(templates, "index.html")
    environment = Environment()
    environment.filters["binding_estimate_kcal"] = binding_estimate_kcal
    environment.filters["format_estimate_point"] = format_estimate_point
    environment.filters["format_estimate_stderr"] = format_estimate_stderr
    return environment.from_string(template).render(
        sprint=SPRINT_NUMBER, analysis=analysis
    )
