from glob import glob
import logging
import os
import re
from typing import List, Optional

from .schema import DataPath, FahConfig


def _get_data_path(project_data_path: str, run: str, clone: str, gen: str) -> str:
    return os.path.join(
        project_data_path, f"RUN{run}", f"CLONE{clone}", f"results{gen}", "globals.csv"
    )


def list_results(config: FahConfig, project: int, run: int) -> List[DataPath]:

    project_data_dir = os.path.join(config.data_dir, f"PROJ{project}")
    glob_pattern = _get_data_path(project_data_dir, str(run), clone="*", gen="*")

    logging.info("Searching for files matching '%s'", glob_pattern)
    paths = glob(glob_pattern)
    logging.info("Found %d matches", len(paths))

    regex = _get_data_path(
        project_data_dir, str(run), clone=r"(?P<clone>\d+)", gen=r"(?P<gen>\d+)"
    )

    def result_path(path: str) -> Optional[DataPath]:
        match = re.match(regex, path)

        if match is None:
            logging.info(
                "Path '%s' matched glob '%s' but not regex '%s'",
                path,
                glob_pattern,
                regex,
            )
            return None

        return DataPath(path=path, clone=int(match["clone"]), gen=int(match["gen"]))

    results = [result_path(path) for path in paths]
    return [r for r in results if r is not None]
