import functools
import simplejson as json
import logging
import fire
from .lib import analyze_runs


@functools.wraps(analyze_runs)
def analyze_runs_cli(*args, **kwargs) -> str:
    results = analyze_runs(*args, **kwargs)

    # NOTE: ignore_nan=True encodes NaN as null, ensuring we produce
    # valid json even if there are NaNs in the output
    return json.dumps([r.to_dict() for r in results], ignore_nan=True)


def main():
    logging.basicConfig(level=logging.WARNING)
    fire.Fire({"analyze_runs": analyze_runs_cli})
