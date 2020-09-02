import os
from typing import List
from ..core import Run
from .index import get_index_html
from .molecules import save_molecule_images


def save_reports(runs: List[Run], path: str) -> None:
    save_molecule_images(
        runs=[run.details for run in runs], path=os.path.join(path, "images")
    )
    html = get_index_html(runs)
    with open(os.path.join(path, "index.html"), "w") as f:
        f.write(html)
