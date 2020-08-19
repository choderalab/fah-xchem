import functools
from glob import glob
import json
import logging
import multiprocessing
import os
import re
from typing import List, Optional
import joblib
from tqdm.auto import tqdm
from .analysis import get_run_analysis
from .core import (
    ResultPath,
    Run,
    RunAnalysis,
    RunDetails,
    Work,
)
from .extract_work import extract_work


def get_result_path(project_path, run, clone, gen):
    return os.path.join(
        project_path, f"RUN{run}", f"CLONE{clone}", f"results{gen}", "globals.csv"
    )


def extract_works(project_path: str, run: int, cache_dir: Optional[str]):

    paths = list_results(project_path, run)

    if not paths:
        raise ValueError(
            f"Empty result set for project path '{project_path}', run {run}"
        )

    _extract_work = (
        extract_work
        if cache_dir is None
        else joblib.Memory(cachedir=cache_dir).cache(extract_work)
    )

    def try_extract_work(path: str) -> Optional[Work]:
        try:
            return _extract_work(path)
        except ValueError as e:
            logging.warning("Failed to extract works from '%s': %s", path, e)
            return None

    results = [try_extract_work(p.path) for p in paths]
    return [r for r in results if r is not None]


def list_results(project_path: str, run: int) -> List[ResultPath]:
    glob_pattern = get_result_path(project_path, run, clone="*", gen="*")
    paths = glob(glob_pattern)

    regex = get_result_path(
        project_path, run, clone=r"(?P<clone>\d+)", gen=r"(?P<gen>\d+)"
    )

    def result_path(path: str) -> Optional[ResultPath]:
        match = re.match(regex, path)

        if match is None:
            logging.info(
                "Path '%s' matched glob '%s' but not regex '%s'",
                path,
                glob_pattern,
                regex,
            )
            return None

        return ResultPath(path, clone=int(match["clone"]), gen=int(match["gen"]))

    results = [result_path(path) for path in paths]
    return [r for r in results if r is not None]


def read_run_details(run_details_json_file: str):
    with open(run_details_json_file, "r") as f:
        runs = json.load(f)
    return [RunDetails.from_dict(r) for r in runs.values()]


def analyze_run(
    run: int,
    complex_project_path: str,
    solvent_project_path: str,
    cache_dir: Optional[str],
) -> RunAnalysis:

    try:
        complex_works = extract_works(complex_project_path, run, cache_dir=cache_dir)
    except ValueError as e:
        raise ValueError(f"Failed to extract work values for complex: {e}")

    try:
        solvent_works = extract_works(complex_project_path, run, cache_dir=cache_dir)
    except ValueError as e:
        raise ValueError(f"Failed to extract work values for solvent: {e}")

    return get_run_analysis(complex_works, solvent_works)


def _try_process_run(details: RunDetails, **kwargs) -> Optional[Run]:
    try:
        return Run(details=details, analysis=analyze_run(details.run_id(), **kwargs))
    except ValueError as e:
        logging.warning("Failed to process run %d: %s", details.run_id(), e)
        return None


def analyze_runs(
    run_details_json_file: str,
    complex_project_path: str,
    solvent_project_path: str,
    cache_dir: Optional[str] = None,
) -> List[Run]:
    """
    Run free energy analysis and return input augmented with analysis
    results for all runs.


    Parameters
    ----------
    run_details_json_file : str
        json file containing run metadata. The file should contain a
        json object with values deserializable to `RunDetails`
    complex_project_path: str
        root path of the FAH project containing simulations of the
        complex, e.g. "PROJ13420"
    solvent_project_path: str
        root path of the FAH project containing simulations of the solvent
    cache_dir: str, optional
        if given, cache work values extracted from simulation data in
        local directory of this name

    Returns
    -------
    list of Run
        run metadata with analysis results for each run

    """

    runs = read_run_details(run_details_json_file)

    try_process_run = functools.partial(
        _try_process_run,
        complex_project_path=complex_project_path,
        solvent_project_path=solvent_project_path,
        cache_dir=cache_dir,
    )

    with multiprocessing.Pool() as pool:
        results_iter = pool.imap_unordered(try_process_run, runs)
        results = list(tqdm(results_iter, total=len(runs)))

    valid = [r for r in results if r is not None]
    num_failed = len(results) - len(valid)

    if num_failed > 0:
        logging.warning("Failed to process %d runs out of %d", num_failed, len(results))

    return valid
