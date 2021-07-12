
from os.path import dirname, join
import json
import bz2

from ..schema import TimestampedAnalysis

def get_compound_series_analysis_results():
    module_path = dirname(__file__)

    compound_series_analysis_file = join(module_path, "results/minimal-neq-sprint/minimal-neq-sprint-stereofilter-P0033-dimer-neutral-restrained/analysis.json.bz2")

    with bz2.open(compound_series_analysis_file, 'r') as f:
        tsa = TimestampedAnalysis.parse_obj(json.load(f))

    return tsa.series
