import os
from typing import Optional
import logging
import fire
from .lib import analyze_runs


def run(
    run_details_json_file: str,
    complex_project_path: str,
    complex_project_data_path: str,
    solvent_project_data_path: str,
    output_dir: str,
    max_binding_delta_f: Optional[float] = None,
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = 8,
):
    """
    Run free energy analysis and return input augmented with analysis
    results for all runs.


    Parameters
    ----------
    run_details_json_file : str
        JSON file containing run metadata. The file should contain a
        JSON object with values deserializable to `RunDetails`
    complex_project_path : str
        Path to the FAH project directory containing configuration for
        simulations of the complex,
        e.g. '/home/server/server2/projects/13422'
    complex_project_data_path : str
        Path to the FAH project data directory containing output data
        from simulations of the complex,
        e.g. "/home/server/server2/data/SVR314342810/PROJ13422"
    solvent_project_data_path : str
        Path to the FAH project data directory containing output data
        from simulations of the solvent,
        e.g. "/home/server/server2/data/SVR314342810/PROJ13423"
    output_dir : str
        Path to output directory. Output will be written in the
        following locations:
        - ``{output_dir}/analysis.json``: analysis results
        - ``{output_dir}/structures``: structure snapshots
        - ``{output_dir}/plots``: plots
    max_binding_delta_f : float, optional
        If given, skip storing snapshot if dimensionless binding free
        energy estimate exceeds this value
    cache_dir : str, optional
        If given, cache intermediate analysis results in local
        directory of this name
    num_procs : int, optional
        Number of parallel processes to run
    """

    analysis = analyze_runs(
        run_details_json_file=run_details_json_file,
        complex_project_path=complex_project_path,
        complex_project_data_path=complex_project_data_path,
        solvent_project_data_path=solvent_project_data_path,
        output_dir=output_dir,
        max_binding_delta_f=max_binding_delta_f,
        cache_dir=cache_dir,
        num_procs=num_procs,
    )

    with open(os.path.join(output_dir, "analysis.json"), "w") as output_file:
        output_file.write(analysis.json())


def main():
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(run)
