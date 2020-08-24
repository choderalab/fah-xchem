from dataclasses import asdict
from typing import Optional
import simplejson as json
import logging
import fire
from .lib import analyze_runs


def analyze_runs_cli(
    run_details_json_file: str,
    complex_project_path: str,
    complex_project_data_path: str,
    solvent_project_data_path: str,
    snapshot_output_path: str,
    output: str = "analysis.json",
    max_binding_delta_f: Optional[float] = None,
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = 8,
) -> str:
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
    output : str
        write json output to this path
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

    results = analyze_runs(
        run_details_json_file=run_details_json_file,
        complex_project_path=complex_project_path,
        complex_project_data_path=complex_project_data_path,
        solvent_project_data_path=solvent_project_data_path,
        snapshot_output_path=snapshot_output_path,
        max_binding_delta_f=max_binding_delta_f,
        cache_dir=cache_dir,
        num_procs=num_procs,
    )

    # NOTE: ignore_nan=True encodes NaN as null, ensuring we produce
    # valid json even if there are NaNs in the output
    with open(output, "w") as output_file:
        json.dump([asdict(r) for r in results], output_file, ignore_nan=True)


def main():
    logging.basicConfig(level=logging.WARNING)
    fire.Fire({"analyze_runs": analyze_runs_cli})
