import os
from typing import List
from ..models import Analysis
from .index import get_index_html
from .molecules import save_molecule_images


def save_reports(analysis: Analysis, path: str) -> None:
    save_molecule_images(
        runs=[run.details for run in analysis.runs], path=os.path.join(path, "images")
    )
    html = get_index_html(analysis)
    with open(os.path.join(path, "index.html"), "w") as f:
        f.write(html)
