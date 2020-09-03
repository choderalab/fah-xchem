import importlib.resources as pkg_resources
from math import floor, isfinite, log10
import os
from typing import List, NamedTuple, Optional
from jinja2 import Environment
from ..analysis.constants import KT_KCALMOL
from ..core import Analysis, Binding
from . import templates


# TODO: read from configuration
SPRINT_NUMBER = 3


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


# TODO: remove hardcoded values
def _get_progress(
    project: int = 13424, api_url: str = "http://aws3.foldingathome.org/api/"
) -> Progress:
    import requests
    from simplejson.errors import JSONDecodeError

    url = f"{api_url}/projects/{project}"

    try:
        response = requests.get(url=url)
        json = response.json()
    except (ConnectionError, JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to get progress data from FAH server ({api_url})"
        ) from exc

    # TODO: remove hardcoded values
    # override while we are still shifting WUs
    n_gens = 2687 * 50 * 6  # sprint 3
    # wus_total = json["gens"]

    return Progress(completed=json["gens_completed"], total=n_gens,)


def get_index_html(analysis: Analysis) -> str:
    template = pkg_resources.read_text(templates, "index.html")
    environment = Environment()
    environment.filters["binding_estimate_kcal"] = binding_estimate_kcal
    environment.filters["format_estimate_point"] = format_estimate_point
    environment.filters["format_estimate_stderr"] = format_estimate_stderr
    return environment.from_string(template).render(
        sprint=SPRINT_NUMBER, analysis=analysis, progress=_get_progress(),
    )
