import datetime as dt
from functools import partial
import logging
from math import isfinite
import os
import re
import pathlib
from typing import Any, NamedTuple, Optional
import numpy as np
from pydantic import BaseModel, Field

import jinja2
import requests
from rich.progress import track
from urllib.parse import urljoin

from ..._version import get_versions
from ...schema import (
    CompoundMicrostate,
    CompoundSeriesAnalysis,
    PointEstimate,
    CompoundAnalysis,
)
from ..constants import KT_KCALMOL, KT_PIC50
from ..filters import Racemic
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
    if est.point >= 0:
        return f"{rounded:.{prec}f}"
    else:
        return f'<span class="negative">−{abs(rounded):.{prec}f}</span>'


def format_stderr(est: PointEstimate) -> str:
    """
    Format an uncertainty with appropriate precision (one significant
    digit, by convention)
    """
    prec = est.precision_decimals()
    if prec is None or not isfinite(est.point):
        return ""
    return f"{round(est.stderr, prec):.{prec}f}"


def format_pIC50(compound: CompoundAnalysis) -> str:
    """
    Format the compound's experimental pIC50 if present, or TBD if not
    """
    experimental_data = compound.metadata.experimental_data
    if "pIC50" in experimental_data:
        return experimental_data["pIC50"]
    else:
        return "TBD"


def postera_url(compound_or_microstate_id: str) -> Optional[str]:
    """
    If `compound_id` matches regex, link to Postera compound details
    """
    import re

    match = re.match(
        "^(?P<compound_id>[A-Z_]{3}-[A-Z_]{3}-[0-9a-f]{8}-[0-9]+)(_(?P<microstate_index>[0-9]+))?([_0-9]*)$",
        compound_or_microstate_id,
    )

    return (
        f"https://postera.ai/covid/submissions/{match['compound_id']}"
        if match
        else None
    )


def experimental_data_url(compound: CompoundAnalysis) -> Optional[str]:
    """
    If `compound_id` contains experimental data, return the URL to Postera compound details
    """
    experimental_data = compound.metadata.experimental_data
    if "pIC50" in experimental_data:
        return postera_url(compound.metadata.compound_id)
    else:
        return None


def format_compound_id(compound_id: str) -> str:
    """
    Format a compound ID as a link if it is in PostEra format
    """
    url = postera_url(compound_id)
    if url is None:
        return compound_id
    else:
        return f'<a href="{url}">{compound_id}</a>'


class Progress(NamedTuple):
    completed: int
    total: int

    def percent_complete(self) -> float:
        return min(100.0, 100.0 * self.completed / self.total)


def _get_progress(project: int, api_url: str) -> Optional[Progress]:
    """
    Query a FAH work server for project status and return progress

    Parameters
    ----------
    project : int
        Project
    api_url : str
        URL of the FAH work server API
        If `None`, this function will return `None`.

    Returns
    -------
    Progress
        Number of completed and total work units
    """
    if api_url is None:
        return None

    url = urljoin(api_url, f"projects/{project}")
    try:
        response = requests.get(url=url).json()
    except Exception as exc:
        logging.warning("Failed to get progress from FAH work server: %s", exc)
        return None

    completed_work_units = response["gens_completed"]
    total_work_units = response["runs"] * response["clones"] * response["gens"]

    return Progress(completed=completed_work_units, total=total_work_units)


def get_sprint_number(description: str) -> Optional[int]:
    match = re.search(r"Sprint (\d+)", description)
    return int(match[1]) if match else None


class WebsiteArtifactory(BaseModel):

    base_url: str
    path: pathlib.Path
    series: CompoundSeriesAnalysis
    timestamp: dt.datetime = None
    fah_ws_api_url: str = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.environment = self._initialize_environment()

    @staticmethod
    def _paginate(items, items_per_page):
        return (
            (
                (start + 1, min(len(items), start + items_per_page)),
                items[start : start + items_per_page],
            )
            for start in range(0, len(items), items_per_page)
        )

    def _generate_paginated_index(
        self, write_html, url_prefix, items, items_per_page, description
    ):
        pages = list(self._paginate(items, items_per_page))

        def get_page_name(start_index, end_index):
            return (
                f"{url_prefix}/index.html"
                if start_index == 1
                else f"{url_prefix}/index-{start_index}-{end_index}.html"
            )

        for (prev_page, ((start_index, end_index), page_items), next_page) in track(
            zip(
                [None] + pages,
                pages,
                pages[1:] + [None],
            ),
            description=description,
            total=len(pages),
        ):
            write_html(
                page_items,
                template_file=f"{url_prefix}/index.html",
                output_file=get_page_name(start_index, end_index),
                start_index=start_index,
                end_index=end_index,
                prev_page=get_page_name(*prev_page[0]) if prev_page else None,
                next_page=get_page_name(*next_page[0]) if next_page else None,
            )

    def _write_html(
        self,
        template_file: pathlib.Path,
        output_file: Optional[pathlib.Path] = None,
        **kwargs: Any,
    ):
        """From template, generate actual page with result outputs.

        Parameters
        ----------
        template_file : pathlib.Path
            Name of template file relative to `self.environment`.
        output_file : Optional[pathlib.Path] = None,
            Path to deposit generated page to.
        **kwargs : Any
            Passed to `jinja2.Environment.


        """

        if output_file is None:
            output_file = template_file

        self.environment.get_template(template_file).stream(
            base_url=self.base_url
            if self.base_url.endswith("/")
            else f"{self.base_url}/",
            series=self.series,
            sprint_number=get_sprint_number(self.series.metadata.description),
            timestamp=self.timestamp,
            fah_xchem_version=get_versions()["version"],
            KT_KCALMOL=KT_KCALMOL,
            KT_PIC50=KT_PIC50,
            microstate_detail={
                CompoundMicrostate(
                    compound_id=compound.metadata.compound_id,
                    microstate_id=microstate.microstate.microstate_id,
                ): (compound.metadata, microstate.microstate)
                for compound in self.series.compounds
                for microstate in compound.microstates
            },
            **kwargs,
        ).dump(os.path.join(self.path, output_file))

    def _initialize_environment(self):
        template_path = os.path.join(os.path.dirname(__file__), "templates")
        template_loader = jinja2.FileSystemLoader(searchpath=template_path)
        environment = jinja2.Environment(loader=template_loader)
        environment.filters["format_point"] = format_point
        environment.filters["format_stderr"] = format_stderr
        environment.filters["format_compound_id"] = format_compound_id
        environment.filters["format_pIC50"] = format_pIC50
        environment.filters["postera_url"] = postera_url
        environment.filters["experimental_data_url"] = experimental_data_url
        environment.filters["smiles_to_filename"] = get_image_filename

        return environment

    def generate_website(
        self,
        items_per_page: int = 100,
        num_top_compounds: int = 100,
    ) -> None:

        generate_molecule_images(
            compounds=self.series.compounds,
            path=os.path.join(self.path, "molecule_images"),
        )

        self.generate_summary(num_top_compounds)
        self.generate_compounds(items_per_page, num_top_compounds)
        self.generate_microstates(items_per_page)
        self.generate_transformations(items_per_page)
        self.generate_retrospective_transformations(items_per_page)

    def generate_summary(self, num_top_compounds):
        self.write_html(
            "index.html",
            progress=_get_progress(
                self.series.metadata.fah_projects.complex_phase, self.fah_ws_api_url
            )
            or Progress(0, 1),
            num_top_compounds=num_top_compounds,
        )

    def generate_compounds(self, items_per_page, num_top_compounds):
        subdir = "compounds"
        os.makedirs(os.path.join(self.path, subdir), exist_ok=True)

        compounds_sorted = sorted(
            [compound for compound in self.series.compounds if compound.free_energy],
            key=lambda m: m.free_energy.point,
        )

        for compound in track(
            compounds_sorted[:num_top_compounds],
            description="Generating html for individual compound views",
        ):
            self._write_html(
                "compounds/compound.html",
                output_file=f"compounds/{compound.metadata.compound_id}.html",
                compound=compound,
                transformations=[
                    transformation
                    for transformation in self.series.transformations
                    if transformation.transformation.initial_microstate.compound_id
                    == compound.metadata.compound_id
                    or transformation.transformation.final_microstate.compound_id
                    == compound.metadata.compound_id
                ],
            )

        self._generate_paginated_index(
            write_html=lambda items, **kwargs: self._write_html(
                compounds=items, num_top_compounds=num_top_compounds, **kwargs
            ),
            url_prefix="compounds",
            items=compounds_sorted,
            items_per_page=items_per_page,
            description="Generating html for compounds index",
        )

    def generate_microstates(self, items_per_page):
        subdir = "microstates"
        os.makedirs(os.path.join(self.path, subdir), exist_ok=True)

        microstates_sorted = sorted(
            [
                microstate
                for compound in self.series.compounds
                for microstate in compound.microstates
                if microstate.free_energy
            ],
            key=lambda m: m.free_energy.point,
        )

        self._generate_paginated_index(
            write_html=lambda items, **kwargs: self._write_html(
                microstates=items, total_microstates=len(microstates_sorted), **kwargs
            ),
            url_prefix="microstates",
            items=microstates_sorted,
            items_per_page=items_per_page,
            description="Generating html for microstates index",
        )

    def generate_transformations(self, items_per_page):
        subdir = "transformations"
        os.makedirs(os.path.join(self.path, subdir), exist_ok=True)

        self._generate_paginated_index(
            write_html=lambda items, **kwargs: self._write_html(
                transformations=items, **kwargs
            ),
            url_prefix="transformations",
            items=self.series.transformations,
            items_per_page=items_per_page,
            description="Generating html for transformations index",
        )

    def generate_reliable_transformations(self, items_per_page):
        subdir = "reliable_transformations"
        os.makedirs(os.path.join(self.path, subdir), exist_ok=True)

        self._generate_paginated_index(
            write_html=lambda items, **kwargs: self._write_html(
                transformations=items, **kwargs
            ),
            url_prefix="reliable_transformations",
            items=self.series.transformations,
            items_per_page=items_per_page,
            description="Generating html for reliable transformations index",
        )

    def generate_retrospective_transformations(self, items_per_page):

        subdir = "retrospective_transformations"
        os.makedirs(os.path.join(self.path, subdir), exist_ok=True)

        racemic_filter = Racemic(self.series)
        self._generate_paginated_index(
            write_html=lambda items, **kwargs: self._write_html(
                transformations=items, **kwargs
            ),
            url_prefix="retrospective_transformations",
            items=sorted(
                [
                    transformation
                    for transformation in self.series.transformations
                    if (
                        not racemic_filter.compound_microstate(
                            transformation.transformation.initial_microstate
                        )
                        and not racemic_filter.compound_microstate(
                            transformation.transformation.final_microstate
                        )
                    )
                ],
                key=lambda transformation: -transformation.absolute_error.point,
            ),
            items_per_page=items_per_page,
            description="Generating html for retrospective transformations index",
        )
