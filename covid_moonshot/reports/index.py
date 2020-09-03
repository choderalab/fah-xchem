import importlib.resources as pkg_resources
from math import floor, isfinite, log10
import os
import requests
from simplejson.errors import JSONDecodeError
from typing import NamedTuple, Optional
from jinja2 import Environment
from ..analysis.constants import KT_KCALMOL
from ..core import Analysis, Binding
from . import templates


# TODO: remove hardcoded values
SPRINT_NUMBER = 3
NUM_GENS = 2687 * 50 * 6  # sprint 3
PROJECT = 13424


class Estimate(NamedTuple):
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


class Progress(NamedTuple):
    completed: int
    total: int

    def percent_complete(self) -> float:
        return 100 * self.completed / self.total


def _get_progress(
    project: int, api_url: str = "http://aws3.foldingathome.org/api/"
) -> Progress:

    url = f"{api_url}/projects/{project}"

    try:
        response = requests.get(url=url)
        json = response.json()
    except (ConnectionError, JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to get progress data from FAH server ({api_url})"
        ) from exc

    return Progress(completed=json["gens_completed"], total=NUM_GENS)


def get_index_html(analysis: Analysis) -> str:
    template = pkg_resources.read_text(templates, "index.html")
    environment = Environment()
    environment.filters["binding_estimate_kcal"] = binding_estimate_kcal
    environment.filters["format_estimate_point"] = format_estimate_point
    environment.filters["format_estimate_stderr"] = format_estimate_stderr
    return environment.from_string(template).render(
        sprint=SPRINT_NUMBER, analysis=analysis, progress=_get_progress(PROJECT)
    )
