import datetime as dt
import os
from typing import Optional
import json
import logging

import fire
from typing import Callable

import fah_xchem
from .analysis import analyze_compound_series
from .schema import (
    AnalysisConfig,
    FahConfig,
    CompoundSeries,
    CompoundSeriesAnalysis,
    Model,
)


class TimestampedAnalysis(Model):
    as_of: dt.datetime
    analysis: CompoundSeriesAnalysis


def _get_config(
    cls,
    config_file: Optional[str],
    description: str,
    decoder: Callable[[str], object] = json.loads,
):
    if config_file is None:
        return cls()
    else:
        logging.info("Reading %s from '%s'", description, config_file)
        with open(config_file, "r") as infile:
            config = cls.parse_obj(decoder(infile.read()))

    logging.info("Using %s: %s", description, config)
    return config


def run_analysis(
    compound_series_file: str,
    generate_artifacts: bool = True,
    config_file: Optional[str] = None,
    fah_projects_dir: str = "projects",
    fah_data_dir: str = "data",
    output_dir: str = "results",
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = 8,
    log: str = "WARN",
):
    """
    Run free energy analysis and return input augmented with analysis
    results for all runs.


    Parameters
    ----------
    compound_series_file : str
        JSON file containing configuration for the compound series in
        the form of a serialized `CompoundSeries` object
    generate_artifacts : bool
        If `True`, generate additional artifacts (including plots, pdf
        report, and html webpage) in the output directory
    config_file : str, optional
        JSON file containing configuration for the analysis in the
        form of a serialized `AnalysisConfig` object
    fah_projects_dir : str, optional
        Path to Folding@home projects directory
    fah_data_dir : str, optional
        Path to Folding@home data directory
    cache_dir : str, optional
        If given, cache intermediate analysis results in local
        directory of this name
    num_procs : int, optional
        Number of parallel processes to run
    """

    logging.basicConfig(level=getattr(logging, log.upper()))

    compound_series = _get_config(
        CompoundSeries, compound_series_file, "compound series"
    )

    config = _get_config(AnalysisConfig, config_file, "analysis configuration")

    analysis = analyze_compound_series(
        series=compound_series,
        config=config,
        server=FahConfig(projects_dir=fah_projects_dir, data_dir=fah_data_dir),
        cache_dir=cache_dir,
        num_procs=num_procs,
    )

    timestamp = dt.datetime.now(dt.timezone.utc)
    output = TimestampedAnalysis(as_of=timestamp, analysis=analysis)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "analysis.json"), "w") as output_file:
        output_file.write(output.json())

    if generate_artifacts:
        fah_xchem.analysis.generate_artifacts(
            analysis=analysis, timestamp=timestamp, output_dir=output_dir
        )


def generate_artifacts(
    compound_series_analysis_file: str, output_dir: str = "results"
) -> None:
    with open(compound_series_analysis_file, "r") as infile:
        tsa = TimestampedAnalysis.parse_obj(json.load(infile))

    return fah_xchem.analysis.generate_artifacts(
        analysis=tsa.analysis, timestamp=tsa.as_of, output_dir=output_dir
    )


def main():
    fire.Fire({"run_analysis": run_analysis, "generate_artifacts": generate_artifacts})
