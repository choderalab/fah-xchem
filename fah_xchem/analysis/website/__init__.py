import datetime as dt
from functools import partial
import logging
from math import isfinite
import os
import re
from typing import Any, NamedTuple, Optional

import jinja2
import requests
from urllib.parse import urljoin

from ..._version import get_versions
from ...schema import CompoundMicrostate, CompoundSeriesAnalysis, PointEstimate
from ..constants import KT_KCALMOL
from .molecules import generate_molecule_images, get_image_filename


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


def maybe_postera_link(compound_or_microstate_id: str) -> str:
    """
    If `compound_id` matches regex, link to Postera compound details
    """
    import re

    match = re.match(
        "^(?P<compound_id>[A-Z]{3}-[A-Z]{3}-[0-9a-f]{8}-[0-9])(_(?P<microstate_index>[0-9]+))?$",
        compound_or_microstate_id,
    )

    if not match:
        return compound_or_microstate_id

    return f'<a href="https://postera.ai/covid/submissions/{match["compound_id"]}">{compound_or_microstate_id}</a>'


class Progress(NamedTuple):
    completed: int
    total: int

    def percent_complete(self) -> float:
        return min(100.0, 100.0 * self.completed / self.total)


def _get_progress(
    project: int, api_url: str = "http://aws3.foldingathome.org/api/"
) -> Optional[Progress]:
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
    try:
        response = requests.get(url=url).json()
    except Exception as exc:
        logging.warning("Failed to get progress from FAH work server: %s", exc)
        return None

    return Progress(completed=response["gens_completed"], total=response["gens"])


def get_sprint_number(description: str) -> Optional[int]:
    match = re.search(r"Sprint (\d+)", description)
    return int(match[1]) if match else None


def get_index_html(series: CompoundSeriesAnalysis, timestamp: dt.datetime) -> str:
    """
    Return index page of html report summarizing analysis results

    Parameters
    ----------
    series : Analysis
        Compound series analysis results

    Returns
    -------
    str
        Website html
    """


def _paginate(items, items_per_page):
    return (
        (
            (start + 1, min(len(items), start + items_per_page)),
            items[start : start + items_per_page],
        )
        for start in range(0, len(items), items_per_page)
    )


def generate_website(
    series: CompoundSeriesAnalysis,
    path: str,
    timestamp: dt.datetime,
    base_url: str,
    transformations_per_page: int = 1000,
) -> None:

    generate_molecule_images(
        microstates=[
            microstate.microstate
            for compound in series.compounds
            for microstate in compound.microstates
        ],
        path=os.path.join(path, "molecule_images"),
    )

    template_path = os.path.join(os.path.dirname(__file__), "templates")
    template_loader = jinja2.FileSystemLoader(searchpath=template_path)
    environment = jinja2.Environment(loader=template_loader)
    environment.filters["format_point"] = format_point
    environment.filters["format_stderr"] = format_stderr
    environment.filters["maybe_postera_link"] = maybe_postera_link
    environment.filters["smiles_to_filename"] = get_image_filename

    for subdir in ["compounds", "microstates", "transformations"]:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)

    def write_html(
        template_file: str, output_file: Optional[str] = None, **kwargs: Any
    ):

        if output_file is None:
            output_file = template_file

        environment.get_template(template_file).stream(
            base_url=base_url if base_url.endswith("/") else f"{base_url}/",
            series=series,
            sprint_number=get_sprint_number(series.metadata.description),
            timestamp=timestamp,
            fah_xchem_version=get_versions()["version"],
            KT_KCALMOL=KT_KCALMOL,
            **kwargs,
        ).dump(os.path.join(path, output_file))

    write_html(
        "index.html",
        progress=_get_progress(series.metadata.fah_projects.complex_phase)
        or Progress(0, 1),
    )

    write_html("compounds/index.html")
    for compound in series.compounds:
        write_html(
            "compounds/compound.html",
            output_file=f"compounds/{compound.metadata.compound_id}.html",
            compound=compound,
            transformations=[
                transformation
                for transformation in series.transformations
                if transformation.transformation.initial_microstate.compound_id
                == compound.metadata.compound_id
                if transformation.transformation.final_microstate.compound_id
                == compound.metadata.compound_id
            ],
        )

    write_html(
        "microstates/index.html",
        microstates=[
            microstate
            for compound in series.compounds
            for microstate in compound.microstates
        ],
    )

    # Generate transformations index pages

    def get_transformations_page(start_index, end_index):
        return (
            "transformations/index.html"
            if start_index == 1
            else f"transformations/index-{start_index}-{end_index}.html"
        )

    pages = list(_paginate(series.transformations, transformations_per_page))

    for (prev_page, ((start_index, end_index), transformations), next_page) in zip(
        [None] + pages,
        pages,
        pages[1:] + [None],
    ):
        write_html(
            template_file="transformations/index.html",
            output_file=get_transformations_page(start_index, end_index),
            start_index=start_index,
            end_index=end_index,
            prev_page=get_transformations_page(*prev_page[0]) if prev_page else None,
            next_page=get_transformations_page(*next_page[0]) if next_page else None,
            transformations=transformations,
            microstate_detail={
                CompoundMicrostate(
                    compound_id=compound.metadata.compound_id,
                    microstate_id=microstate.microstate.microstate_id,
                ): (compound.metadata, microstate.microstate)
                for compound in series.compounds
                for microstate in compound.microstates
            },
        )
