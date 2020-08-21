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
from .analysis.structures import save_representative_snapshots
from .core import (
    ResultPath,
    Run,
    RunAnalysis,
    RunDetails,
    Work,
)
from .extract_work import extract_work


def get_result_path(project_data_path, run, clone, gen) -> str:
    return os.path.join(
        project_data_path, f"RUN{run}", f"CLONE{clone}", f"results{gen}", "globals.csv"
    )


def extract_works(project_data_path: str, run: int) -> List[Work]:

    paths = list_results(project_data_path, run)

    if not paths:
        raise ValueError(
            f"Empty result set for project path '{project_data_path}', run {run}"
        )

    def try_extract_work(path: ResultPath) -> Optional[Work]:
        try:
            return extract_work(path)
        except ValueError as e:
            logging.warning("Failed to extract works from '%s': %s", path.path, e)
            return None

    results = [try_extract_work(path) for path in paths]
    return [r for r in results if r is not None]


def list_results(project_data_path: str, run: int) -> List[ResultPath]:
    glob_pattern = get_result_path(project_data_path, run, clone="*", gen="*")
    paths = glob(glob_pattern)

    regex = get_result_path(
        project_data_path, run, clone=r"(?P<clone>\d+)", gen=r"(?P<gen>\d+)"
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


def read_run_details(run_details_json_file: str) -> List[RunDetails]:
    with open(run_details_json_file, "r") as f:
        runs = json.load(f)
    return [RunDetails(**r) for r in runs.values()]


def analyze_run(
    run: int,
    complex_project_path: str,
    complex_project_data_path: str,
    solvent_project_data_path: str,
    snapshot_output_path: str,
    max_binding_delta_f: Optional[float],
    cache_dir: Optional[str],
) -> RunAnalysis:

    try:
        complex_works = extract_works(complex_project_data_path, run)
    except ValueError as e:
        raise ValueError(f"Failed to extract work values for complex: {e}")

    try:
        solvent_works = extract_works(solvent_project_data_path, run)
    except ValueError as e:
        raise ValueError(f"Failed to extract work values for solvent: {e}")

    analysis = get_run_analysis(complex_works, solvent_works)

    if (
        max_binding_delta_f is not None
        and analysis.binding.delta_f >= max_binding_delta_f
    ):
        logging.warning(
            f"Skipping snapshot for run {run}. "
            f"Binding free energy estimate {analysis.binding.delta_f} "
            f"exceeds threshold {max_binding_delta_f}."
        )
    else:
        try:
            save_representative_snapshots(
                project_path=complex_project_path,
                project_data_path=complex_project_data_path,
                run=run,
                works=complex_works,
                fragment_id="x10789",
                snapshot_output_path=snapshot_output_path,
                cache_dir=cache_dir,
            )
        except ValueError as e:
            raise ValueError(f"Failed to save structures for complex: {e}")

    return analysis


def _try_process_run(details: RunDetails, **kwargs) -> Optional[Run]:
    try:
        return Run(details=details, analysis=analyze_run(details.run_id(), **kwargs))
    except ValueError as e:
        logging.warning("Failed to process run %d: %s", details.run_id(), e)
        return None


def analyze_runs(
    run_details_json_file: str,
    complex_project_path: str,
    complex_project_data_path: str,
    solvent_project_data_path: str,
    snapshot_output_path: str,
    max_binding_delta_f: Optional[float] = None,
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = 8,
) -> List[Run]:
    """
    Run free energy analysis and return input augmented with analysis
    results for all runs.


    Parameters
    ----------
    run_details_json_file : str
        json file containing run metadata. The file should contain a
        json object with values deserializable to `RunDetails`
    complex_project_path : str
        path to the FAH project directory containing configuration for
        simulations of the complex,
        e.g. '/home/server/server2/projects/13422'
    complex_project_data_path : str
        path to the FAH project data directory containing output data
        from simulations of the complex,
        e.g. "/home/server/server2/data/SVR314342810/PROJ13422"
    solvent_project_data_path : str
        path to the FAH project data directory containing output data
        from simulations of the solvent,
        e.g. "/home/server/server2/data/SVR314342810/PROJ13423"
    snapshot_output_path : str
        path where snapshots will be written
    max_binding_delta_f : float, optional
        if given, skip storing snapshot if dimensionless binding free
        energy estimate exceeds this value
    cache_dir : str, optional
        if given, cache intermediate analysis results in local
        directory of this name
    num_procs : int, optional
        number of parallel processes to run

    Returns
    -------
    list of Run
        run metadata with analysis results for each run

    """

    runs = read_run_details(run_details_json_file)

    try_process_run = functools.partial(
        _try_process_run,
        complex_project_path=complex_project_path,
        complex_project_data_path=complex_project_data_path,
        solvent_project_data_path=solvent_project_data_path,
        snapshot_output_path=snapshot_output_path,
        max_binding_delta_f=max_binding_delta_f,
        cache_dir=cache_dir,
    )

    with multiprocessing.Pool(num_procs) as pool:
        results_iter = pool.imap_unordered(try_process_run, runs)
        results = list(tqdm(results_iter, total=len(runs)))

    valid = [r for r in results if r is not None]
    num_failed = len(results) - len(valid)

    if num_failed > 0:
        logging.warning("Failed to process %d runs out of %d", num_failed, len(results))

    return valid
