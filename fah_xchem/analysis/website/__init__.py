import datetime as dt
import os

from ...schema import CompoundSeriesAnalysis
from .index import get_index_html
from .molecules import generate_molecule_images


def generate_website(
    series_analysis: CompoundSeriesAnalysis, path: str, timestamp: dt.datetime
) -> None:
    generate_molecule_images(
        microstates=[
            microstate.microstate
            for compound in series_analysis.compounds
            for microstate in compound.microstates
        ],
        path=os.path.join(path, "molecule_images"),
    )
    html = get_index_html(series=series_analysis, timestamp=timestamp)
    with open(os.path.join(path, "index.html"), "w") as f:
        f.write(html)
