import datetime as dt
from math import floor, isfinite, log10
import os
import requests
from urllib.parse import urljoin

from jinja2 import Environment
from simplejson.errors import JSONDecodeError
from typing import NamedTuple, Optional

from ...schema import CompoundMicrostate, CompoundSeriesAnalysis, PointEstimate
from ..constants import KT_KCALMOL


# TODO: remove hardcoded values
SPRINT_NUMBER = 3
NUM_GENS = 2687 * 50 * 6  # sprint 3
PROJECT = 13424


def format_point(est: PointEstimate) -> str:
    """
    Format a point estimate with appropriate precision given the
    associated uncertainty. If the point estimate is negative, wrap
    the result in a span tag with class `negative` for styling.
    """
    prec = est.precision_decimals()
    if prec is None or not isfinite(est.point):
        return ""
    rounded = round(est.point, prec)
    return (
        f"{rounded:.{prec}f}"
        if est.point > 0
        else f'<span class="negative">−{abs(rounded):.{prec}f}</span>'
    )


def format_stderr(est: PointEstimate) -> str:
    """
    Format an uncertainty with appropriate precision (one significant
    digit, by convention)
    """
    prec = est.precision_decimals()
    if prec is None or not isfinite(est.point):
        return ""
    return f"{round(est.stderr, prec):.{prec}f}"


class Progress(NamedTuple):
    completed: int
    total: int

    def percent_complete(self) -> float:
        return min(100.0, 100.0 * self.completed / self.total)


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


def get_index_html(series: CompoundSeriesAnalysis, timestamp: dt.datetime) -> str:
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

    template_filename = os.path.join(
        os.path.dirname(__file__), "templates", "index.html"
    )

    with open(template_filename, "r") as template_file:
        template = template_file.read()

    environment = Environment()
    environment.filters["format_point"] = format_point
    environment.filters["format_stderr"] = format_stderr
    return environment.from_string(template).render(
        sprint=SPRINT_NUMBER,
        series=series,
        microstate_detail={
            CompoundMicrostate(
                compound_id=compound.metadata.compound_id,
                microstate_id=microstate.microstate.microstate_id,
            ): (compound, microstate)
            for compound in series.compounds
            for microstate in compound.microstates
        },
        timestamp=timestamp,
        # progress=_get_progress(PROJECT),
        progress=Progress(completed=75, total=100),
        KT_KCALMOL=KT_KCALMOL,
    )