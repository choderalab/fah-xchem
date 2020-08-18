import fire
import functools
import json
from . import analyze_runs


@functools.wraps(analyze_runs)
def analyze_runs_cli(
    run_details_json_file,
    complex_project_path,
    solvent_project_path,
    num_works_expected,
    num_steps_expected,
):
    results = analyze_runs(
        run_details_json_file=run_details_json_file,
        complex_project_path=complex_project_path,
        solvent_project_path=solvent_project_path,
        num_works_expected=num_works_expected,
        num_steps_expected=num_steps_expected,
    )

    return json.dumps([r.to_dict() for r in results])


def main():
    fire.Fire({"analyze_runs": analyze_runs_cli})
