import importlib.resources as pkg_resources
from math import floor, isfinite, log10
import os
import requests
from simplejson.errors import JSONDecodeError
from typing import NamedTuple, Optional
from urllib.parse import urljoin
from jinja2 import Environment
from ..analysis.constants import KT_KCALMOL
from ..core import Analysis, Binding
from . import templates


# TODO: remove hardcoded values
SPRINT_NUMBER = 3
NUM_GENS = 2687 * 50 * 6  # sprint 3
PROJECT = 13424


class Estimate(NamedTuple):
    """
    Representation of a quantity with uncertainty

    Parameters
    ----------
    point : float
        Point estimate
    stderr : float
        Standard error
    """

    point: float
    stderr: float

    def precision(self) -> Optional[int]:
        """
        Return precision of the estimate in decimal places. Positive
        numbers indicate digits to the right of the decimal point;
        negative number represent digits to the left.

        Returns
        -------
        int or None
            If `point` and `stderr` are both finite, the precision in
            decimal places. Otherwise, `None`.
        """
        return -floor(log10(self.stderr)) if isfinite(self.stderr) else None


def binding_estimate_kcal(binding: Binding) -> Estimate:
    return Estimate(
        point=binding.delta_f * KT_KCALMOL, stderr=binding.ddelta_f * KT_KCALMOL
    )


def format_estimate_point(est: Estimate) -> str:
    """
    Format a point estimate with appropriate precision given the
    associated uncertainty. If the point estimate is negative, wrap
    the result in a span tag with class `negative` for styling.
    """
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
    """
    Format an uncertainty with appropriate precision (one significant
    digit, by convention)
    """
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
    """
    Query a FAH work server for project status and return progress

    Parameters
    ----------
    project : int
        Project
    api_url : str, optional
        URL of the FAH work server API

    Returns
    -------
    Progress
        Number of completed and total work units
    """
    url = urljoin(api_url, f"projects/{project}")
    response = requests.get(url=url)
    json = response.json()
    return Progress(completed=json["gens_completed"], total=NUM_GENS)


def get_index_html(analysis: Analysis) -> str:
    """
    Return index page of html report summarizing analysis results

    Parameters
    ----------
    analysis : Analysis
        Analysis results

    Returns
    -------
    str
        Report html
    """
    template = pkg_resources.read_text(templates, "index.html")
    environment = Environment()
    environment.filters["binding_estimate_kcal"] = binding_estimate_kcal
    environment.filters["format_estimate_point"] = format_estimate_point
    environment.filters["format_estimate_stderr"] = format_estimate_stderr
    return environment.from_string(template).render(
        sprint=SPRINT_NUMBER, analysis=analysis, progress=_get_progress(PROJECT)
    )
