import functools
import json
import logging
import fire
from .core import analyze_runs


@functools.wraps(analyze_runs)
def analyze_runs_cli(*args, **kwargs):
    results = analyze_runs(*args, **kwargs)
    return json.dumps([r.to_dict() for r in results])


def main():
    logging.basicConfig(level=logging.WARNING)
    fire.Fire({"analyze_runs": analyze_runs_cli})
