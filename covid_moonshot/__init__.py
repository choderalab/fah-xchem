from dataclasses import dataclass
from dataclasses_json import dataclass_json
import functools
from glob import glob
import joblib
import json
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import pymbar
import re
from tqdm.auto import tqdm
from typing import List, Optional


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
    if header_lines:
        return header_lines[-1]
    else:
        raise ValueError(f"Missing header in {path}")


def _get_num_steps(df: pd.DataFrame) -> int:
    if df.empty:
        raise ValueError("empty dataframe")
    step = df["Step"]
    return step.iloc[-1] - step.iloc[0]


@dataclass_json
@dataclass
class Work:
    forward_work: float
    reverse_work: float
    forward_final_potential: float
    reverse_final_potential: float


def extract_work(path: str, num_works_expected: int, num_steps_expected: int,) -> Work:

    header_line_number = _get_last_header_line(path)
    df = pd.read_csv(path, header=header_line_number)

    # TODO: explanation for duplicates?
    df.drop_duplicates(inplace=True)

    kT = df["kT"].astype(float)[0]
    protocol_work_nodims = df["protocol_work"] / kT
    Enew_nodims = df["Enew"] / kT

    if len(protocol_work_nodims) != num_works_expected:
        raise ValueError(
            f"expected {num_works_expected} work values, "
            f"but found {len(protocol_work_nodims)}"
        )

    num_steps = _get_num_steps(df)
    if num_steps != num_steps_expected:
        raise ValueError(f"expected {num_steps_expected} steps, but found {num_steps}")
    # TODO: magic numbers
    return Work(
        forward_work=protocol_work_nodims[20] - protocol_work_nodims[10],
        reverse_work=protocol_work_nodims[40] - protocol_work_nodims[30],
        forward_final_potential=Enew_nodims[20],
        reverse_final_potential=Enew_nodims[40],
    )


@dataclass_json
@dataclass
class ResultPath:
    path: str
    clone: int
    gen: int


def list_results(project_path: str, run: int) -> List[ResultPath]:
    glob_pattern = get_result_path(project_path, run, "*", "*")
    paths = glob(glob_pattern)

    regex = get_result_path(
        project_path, run, clone="(?P<clone>\d+)", gen="(?P<gen>\d+)"
    )

    def result_path(path: str) -> ResultPath:
        match = re.match(regex, path)
        return ResultPath(path, clone=int(match["clone"]), gen=int(match["gen"]))

    return [result_path(path) for path in paths]


@dataclass_json
@dataclass
class PhaseAnalysis:
    delta_f: float
    ddelta_f: float
    num_work_values: int


def analyze_phase(
    project_path: str,
    run: int,
    num_works_expected: int,
    num_steps_expected: int,
    cache_dir: Optional[str],
) -> PhaseAnalysis:

    paths = list_results(project_path, run)

    if not paths:
        raise ValueError(f"empty result set for project path {project_path}, run {run}")

    _extract_work = (
        extract_work
        if cache_dir is None
        else joblib.Memory(cachedir=cache_dir).cache(extract_work)
    )

    works = [
        _extract_work(
            p.path,
            num_works_expected=num_works_expected,
            num_steps_expected=num_steps_expected,
        )
        for p in paths
    ]

    f_works = np.array([w.forward_work for w in works])
    r_works = np.array([w.reverse_work for w in works])

    # TODO remove outliers

    # TODO remove gens with fewer than the minimum number of samples

    delta_f, ddelta_f = pymbar.BAR(f_works, r_works)

    return PhaseAnalysis(
        delta_f=delta_f, ddelta_f=ddelta_f, num_work_values=len(f_works)  # XXX
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
    num_works_expected: int,
    num_steps_expected: int,
    cache_dir: Optional[str],
) -> RunAnalysis:

    _analyze_phase = functools.partial(
        analyze_phase,
        run=run,
        num_works_expected=num_works_expected,
        num_steps_expected=num_steps_expected,
        cache_dir=cache_dir,
    )

    complex_phase = _analyze_phase(complex_project_path)
    solvent_phase = _analyze_phase(solvent_project_path)

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


def _process_run(details: RunDetails, **kwargs) -> Optional[RunAnalysis]:
    try:
        return Run(details=details, analysis=analyze_run(details.run_id(), **kwargs),)
    except ValueError as e:
        # logging.warning(f"Failed to process run {run}: {e}")
        pass


def analyze_runs(
    run_details_json_file: str,
    complex_project_path: str,
    solvent_project_path: str,
    num_works_expected: int,
    num_steps_expected: int,
    cache_dir: Optional[str] = None,
) -> List[Run]:

    runs = read_run_details(run_details_json_file)

    process_run = functools.partial(
        _process_run,
        complex_project_path=complex_project_path,
        solvent_project_path=solvent_project_path,
        num_works_expected=num_works_expected,
        num_steps_expected=num_steps_expected,
        cache_dir=cache_dir,
    )

    with multiprocessing.Pool() as pool:
        results_iter = pool.imap(process_run, runs)
        results = list(tqdm(results_iter, total=len(runs)))

    valid = [r for r in results if r is not None]
    num_failed = len(results) - len(valid)

    if num_failed > 0:
        logging.warning("Failed to process %d runs out of %d", num_failed, len(results))

    return valid
