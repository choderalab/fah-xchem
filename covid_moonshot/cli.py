import fire
import functools
import json
from . import analyze_runs


@functools.wraps(analyze_runs)
def analyze_runs_cli(*args, **kwargs):
    results = analyze_runs(*args, **kwargs)
    return json.dumps([r.to_dict() for r in results])


def main():
    import logging

    logging.basicConfig(level=logging.INFO)
    fire.Fire({"analyze_runs": analyze_runs_cli})
