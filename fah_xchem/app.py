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
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
    CompoundSeriesAnalysis,
    TimestampedAnalysis,
    FragalysisConfig,
)


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


def normalize_experimental_data(
    experimental_data: dict,
) -> None:
    """
    Standardize the experimental_data dict to ensure all necessary quantities are available.

    Currently, this assumes the following are available:
    * pIC50
    * pIC50_lower
    * pIC50_upper

    and generates
    * g_exp : binding free energy (in kT)
    * g_dexp : standard error (in kT)
    """
    if "pIC50" not in experimental_data:
        return

    # TODO: Specify these in the experimental record or some higher-level metadata?
    s_conc = 375e-9  # substrate concentration (molar)
    Km = 40e-6  # Km (molar)
    kT = 0.596  # kcal/mol # TODO: Use temperature instead
    DEFAULT_pIC50_STDERR = (
        0.2  # from n=9 replicate measurements of CVD-0002707 : 7.23 +/- 1.57
    )

    # Compute dimensionless free energy and standard error

    if "pIC50_stderr" not in experimental_data:
        if ("pIC50_lower" in experimental_data) and (
            "pIC50_upper" in experimental_data
        ):
            experimental_data["pIC50_stderr"] = (
                abs(experimental_data["pIC50_upper"] - experimental_data["pIC50_lower"])
                / 4.0
            )
        else:
            experimental_data["pIC50_stderr"] = DEFAULT_pIC50_STDERR

    # Compute dimensionless free energy and uncertainty
    import numpy as np

    experimental_data["g_exp"] = -np.log(10.0) * experimental_data["pIC50"]
    experimental_data["g_dexp"] = np.log(10.0) * experimental_data["pIC50_stderr"]
    # TODO: Delete other records to avoid conflics?


def update_experimental_data(
    compound_series: CompoundSeries,
    experimental_data_file: str,
    update_key: Optional[str] = "smiles",
) -> None:
    """
    Update the experimental data records in the CompoundSeries with new data provided by an external file.

    Parameters
    ----------
    compound_series : CompoundSeries
        The compound series to update
    experimental_data_file : str
        The JSON experimental data file containing a serialized form of ExperimentalCompoundData
    update_key : str, optional, default='smiles'
        Select whether experimental data should be assigned based on suspected 'smiles' or 'compound_id'
        'compound_id': Assume measured compound identity is correct (often wrong with stereoisomers)
        'smiles': Use the suspected_SMILES CDD field to update based on SMILES matches (often a better choice)
        Note that designs are submitted using absolute stereochemistry while experimental measurements are assigned
        using relative stereochemistry, so 'smiles' should be more reliable.
    """
    ALLOWED_KEYS = ["smiles", "compound_id"]
    if not update_key in ALLOWED_KEYS:
        raise ValueError(f"update_key must be one of {ALLOWED_KEYS}")

    import os

    if not os.path.exists(experimental_data_file):
        raise ValueError(
            f"Experimental data file {experimental_data_file} does not exist."
        )

    # Read experimental data file containing compound ids and presumed SMILES for experimental measurements
    with open(experimental_data_file, "r") as infile:
        import json

        experimental_compound_data = ExperimentalCompoundDataUpdate.parse_obj(
            json.loads(infile.read())
        )
        logging.info(
            f"Data for {len(experimental_compound_data.compounds)} compounds read from {experimental_data_file}"
        )

    # Add information about composition (racemic or enantiopure) to experimental_data
    # TODO: Update object model instead of using the experimental_data dict?
    for compound in experimental_compound_data.compounds:
        if compound.is_racemic:
            compound.experimental_data["racemate"] = 1.0
        else:
            compound.experimental_data["enantiopure"] = 1.0

    # Build a lookup table for experimental data by suspected SMILES
    logging.info(f"Matching experimental data with compound designs via {update_key}")
    experimental_data = {
        getattr(compound, update_key): compound.experimental_data
        for compound in experimental_compound_data.compounds
    }

    number_of_compounds_updated = 0
    for compound in compound_series.compounds:
        metadata = compound.metadata
        key = getattr(metadata, update_key)
        if key in experimental_data:
            number_of_compounds_updated += 1
            metadata.experimental_data.update(experimental_data[key])
            # TODO: Standardize which experimental data records are available here: IC50, pIC50, DeltaG, delta_g, and uncertainties
            logging.info(
                f"Updating experiental data for {metadata.compound_id} : {metadata.experimental_data}"
            )

    logging.info(
        f"Updated experimental data for {number_of_compounds_updated} compounds"
    )


def run_analysis(
    compound_series_file: str,
    config_file: Optional[str] = None,
    fah_projects_dir: str = "projects",
    fah_data_dir: str = "data",
    output_dir: str = "results",
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = 8,
    max_transformations: Optional[int] = None,
    experimental_data_file: Optional[str] = None,
    log: str = "WARN",
) -> None:
    """
    Run free energy analysis and write JSON-serialized analysis
    consisting of input augmented with analysis results


    Parameters
    ----------
    compound_series_file : str
        File containing compound series as JSON-encoded :class:`~fah_xchem.schema.CompoundSeries`
    config_file : str, optional
        File containing analysis configuration as JSON-encoded :class:`~fah_xchem.schema.AnalysisConfig`
    fah_projects_dir : str, optional
        Path to Folding@home projects directory
    fah_data_dir : str, optional
        Path to Folding@home data directory
    cache_dir : str, optional
        If given, cache intermediate analysis results in local
        directory of this name
    num_procs : int, optional
        Number of parallel processes to run
    max_transformations : int, optional
        If not None, limit to this number of transformations
    experimental_data_file : str, optional, default=None
        If not None, load experimental compound data and update compound experimental_data dictionaries
    """

    logging.basicConfig(level=getattr(logging, log.upper()))

    compound_series = _get_config(
        CompoundSeries, compound_series_file, "compound series"
    )

    # Update available experimental data if an experimental data file is specified
    # TODO: Allow CLI specification of whether 'compound_id' or 'smiles' is used for update_key optional argument
    if experimental_data_file is not None:
        update_experimental_data(compound_series, experimental_data_file)

    # Normalize experimental data
    for compound in compound_series.compounds:
        normalize_experimental_data(compound.metadata.experimental_data)

    # Limit number of transformations considered (for debugging purposes) if requested
    if max_transformations is not None:
        logging.warning(
            f"Limiting maximum number of transformations to {max_transformations}"
        )
        compound_series = CompoundSeries(
            metadata=compound_series.metadata,
            compounds=compound_series.compounds,
            transformations=compound_series.transformations[:max_transformations],
        )

    config = _get_config(AnalysisConfig, config_file, "analysis configuration")

    series_analysis = analyze_compound_series(
        series=compound_series,
        config=config,
        server=FahConfig(projects_dir=fah_projects_dir, data_dir=fah_data_dir),
        num_procs=num_procs,
    )

    timestamp = dt.datetime.now(dt.timezone.utc)
    output = TimestampedAnalysis(as_of=timestamp, series=series_analysis)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "analysis.json"), "w") as output_file:
        output_file.write(output.json(indent=3))


def generate_artifacts(
    compound_series_analysis_file: str,
    fah_projects_dir: str,
    fah_data_dir: str,
    fah_api_url: str,
    output_dir: str = "results",
    base_url: str = "/",
    config_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = None,
    snapshots: bool = True,
    plots: bool = True,
    report: bool = True,
    website: bool = True,
    log: str = "WARN",
    fragalysis_config: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Given results of free energy analysis as JSON, generate analysis
    artifacts in `output_dir`

    By default the following are generated:

    - representative snapshots
    - plots
    - PDF report
    - static HTML for website

    Parameters
    ----------
    compound_series_analysis_file : str
        File containing analysis results as JSON-encoded :class:`~fah_xchem.schema.CompoundSeriesAnalysis`
    fah_projects_dir : str
        Path to directory containing Folding@home project definitions
    fah_data_dir : str
        Path to directory containing Folding@home project result data
    fah_api_url : str
        URL for work server API
    output_dir : str, optional
        Write output here
    base_url : str, optional
        Base URL to use for links in the static site. E.g. if using S3, something like
        https://fah-ws3.s3.amazonaws.com/covid-moonshot/sprints/sprint-4/2020-09-06-ugi-tBu-x3110-3v3m-2020-04-Jacobs/
    config_file : str, optional
        File containing analysis configuration as JSON-encoded :class:`~fah_xchem.schema.AnalysisConfig`
    cache_dir : str or None, optional
        If given, cache intermediate results in a local directory with this name
    num_procs : int or None, optional
        Maximum number of concurrent processes to run for analysis
    snapshots : bool, optional
        Whether to generate representative snapshots
    plots : bool, optional
        Whether to generate plots
    report : bool, optional
        Whether to generate PDF report
    website : bool, optional
        Whether to generate HTML for static site
    log : str, optional
        Logging level
    fragalysis_config : str, optional
        File containing information for Fragalysis upload as JSON-encoded :class: ~`fah_xchem.schema.FragalysisConfig`
    overwrite : bool
        If `True`, write over existing output files if present.
        Otherwise, skip writing output files for a given transformation when already present.
        Assumes that for a given `run_id` the output files do not ever change;
        does *no* checking that files wouldn't be different if inputs for a given `run_id` have changed.
    """

    logging.basicConfig(level=getattr(logging, log.upper()))

    config = _get_config(AnalysisConfig, config_file, "analysis configuration")

    fragalysis_config = _get_config(
        FragalysisConfig, fragalysis_config, "fragalysis configuration"
    )

    with open(compound_series_analysis_file, "r") as infile:
        tsa = TimestampedAnalysis.parse_obj(json.load(infile))

    server = FahConfig(
        projects_dir=fah_projects_dir, data_dir=fah_data_dir, api_url=fah_api_url
    )

    return fah_xchem.analysis.generate_artifacts(
        series=tsa.series,
        timestamp=tsa.as_of,
        projects_dir=fah_projects_dir,
        data_dir=fah_data_dir,
        output_dir=output_dir,
        base_url=base_url,
        config=config,
        server=server,
        cache_dir=cache_dir,
        num_procs=num_procs,
        snapshots=snapshots,
        plots=plots,
        report=report,
        website=website,
        fragalysis_config=fragalysis_config,
        overwrite=overwrite,
    )


def main():
    fire.Fire({"run_analysis": run_analysis, "generate_artifacts": generate_artifacts})
