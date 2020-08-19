import functools
from glob import glob
import json
import logging
import multiprocessing
import os
import re
from typing import List, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import joblib
import numpy as np
import pandas as pd
import pymbar
from tqdm.auto import tqdm


def get_result_path(project_path, run, clone, gen):
    return os.path.join(
        project_path, f"RUN{run}", f"CLONE{clone}", f"results{gen}", "globals.csv"
    )


def _is_header_line(line):
    return "kT" in line


def _get_last_header_line(path: str) -> int:
    with open(path, "r") as f:
        lines = f.readlines()
    header_lines = [i for i, line in enumerate(lines) if _is_header_line(line)]
    if not header_lines:
        raise ValueError(f"Missing header in {path}")
    return header_lines[-1]


def _get_num_steps(df: pd.DataFrame) -> int:
    if df.empty:
        raise ValueError("Empty dataframe")
    step = df["Step"]
    return step.iloc[-1] - step.iloc[0]


@dataclass_json
@dataclass
class Work:
    forward_work: float
    reverse_work: float
    forward_final_potential: float
    reverse_final_potential: float


def extract_work(path: str) -> Work:

    NUM_WORKS_EXPECTED = 41
    NUM_STEPS_EXPECTED = 1000000

    header_line_number = _get_last_header_line(path)
    df = pd.read_csv(path, header=header_line_number)

    # TODO: explanation for duplicates?
    df.drop_duplicates(inplace=True)

    kT = df["kT"].astype(float)[0]

    protocol_work = df["protocol_work"].astype(float).values
    protocol_work_nodims = protocol_work / kT

    Enew = df["Enew"].astype(float).values
    Enew_nodims = Enew / kT

    if len(protocol_work_nodims) != NUM_WORKS_EXPECTED:
        raise ValueError(
            f"Expected {NUM_WORKS_EXPECTED} work values, "
            f"but found {len(protocol_work_nodims)}"
        )

    num_steps = _get_num_steps(df)
    if num_steps != NUM_STEPS_EXPECTED:
        raise ValueError(f"Expected {NUM_STEPS_EXPECTED} steps, but found {num_steps}")

    # TODO: magic numbers
    try:
        return Work(
            forward_work=protocol_work_nodims[20] - protocol_work_nodims[10],
            reverse_work=protocol_work_nodims[40] - protocol_work_nodims[30],
            forward_final_potential=Enew_nodims[20],
            reverse_final_potential=Enew_nodims[40],
        )
    except KeyError as e:
        raise ValueError(
            f"Tried to index into dataframe at row {e}, "
            f"but dataframe has {len(df)} rows"
        )


@dataclass_json
@dataclass
class ResultPath:
    path: str
    clone: int
    gen: int


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


@dataclass_json
@dataclass
class PhaseAnalysis:
    delta_f: float
    ddelta_f: float
    num_work_values: int


def extract_works(paths: List[ResultPath], cache_dir: Optional[str]):
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


def _mask_outliers(a: np.array, max_value: float, n_devs: float):
    """Returns a boolean array masking values that are more than
    `n_devs` standard deviations from the mean or larger in magnitude
    than `max_value`.

    Parameters
    ----------
    a : array
    max_value : float
        Values larger (in magnitude) than this will be discarded
    n_devs : float
        Values farther than this number of standard deviations from
        the mean will be discarded

    Returns
    -------
    array of bool
        Boolean array of same shape as `a`, with False elements
        marking outliers in `a` and all other elements True.
    """
    return (np.abs(a) < max_value) & (np.abs(a - np.mean(a)) < n_devs * np.std(a))


def analyze_phase(
    project_path: str,
    run: int,
    cache_dir: Optional[str],
    max_work_value=1e4,
    max_n_devs=5,
    min_num_work_values=10,
) -> PhaseAnalysis:

    paths = list_results(project_path, run)

    if not paths:
        raise ValueError(f"Empty result set for project path {project_path}, run {run}")

    works = extract_works(paths, cache_dir=cache_dir)

    f_works = np.array([w.forward_work for w in works])
    r_works = np.array([w.reverse_work for w in works])

    # remove pairs of works when either is determined to be an outlier
    f_good = _mask_outliers(f_works, max_value=max_work_value, n_devs=max_n_devs)
    r_good = _mask_outliers(r_works, max_value=max_work_value, n_devs=max_n_devs)
    both_good = f_good & r_good
    f_works_good = f_works[both_good]
    r_works_good = r_works[both_good]
    num_work_values = both_good.astype(int).sum().item()

    if num_work_values < min_num_work_values:
        raise ValueError(
            f"Need at least {min_num_work_values} good work values for analysis, "
            f"but got {num_work_values}"
        )

    delta_f, ddelta_f = pymbar.BAR(f_works_good, r_works_good)

    return PhaseAnalysis(
        delta_f=delta_f, ddelta_f=ddelta_f, num_work_values=num_work_values
    )


@dataclass_json
@dataclass
class Binding:
    delta_f: float
    ddelta_f: float


@dataclass_json
@dataclass
class RunAnalysis:
    complex_phase: PhaseAnalysis
    solvent_phase: PhaseAnalysis
    binding: Binding


def analyze_run(
    run: int,
    complex_project_path: str,
    solvent_project_path: str,
    cache_dir: Optional[str],
) -> RunAnalysis:

    _analyze_phase = functools.partial(analyze_phase, run=run, cache_dir=cache_dir)

    try:
        complex_phase = _analyze_phase(complex_project_path)
    except ValueError as e:
        raise ValueError(f"Failed to process complex phase: {e}")

    try:
        solvent_phase = _analyze_phase(solvent_project_path)
    except ValueError as e:
        raise ValueError(f"Failed to process solvent phase: {e}")

    binding = Binding(
        delta_f=complex_phase.delta_f - solvent_phase.delta_f,
        ddelta_f=(complex_phase.ddelta_f ** 2 + solvent_phase.ddelta_f ** 2) ** 0.5,
    )

    return RunAnalysis(
        complex_phase=complex_phase, solvent_phase=solvent_phase, binding=binding
    )


@dataclass_json
@dataclass
class RunDetails:
    JOBID: int
    directory: str
    end: int
    end_pIC50: float
    end_smiles: str
    end_title: str
    ff: str
    ligand: str
    protein: str
    start: int
    start_smiles: str
    start_title: str
    target: str

    def run_id(self):
        return self.JOBID - 1


def read_run_details(run_details_json_file: str):
    with open(run_details_json_file, "r") as f:
        runs = json.load(f)
    return [RunDetails.from_dict(r) for r in runs.values()]


@dataclass_json
@dataclass
class Run:
    details: RunDetails
    analysis: RunAnalysis


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
