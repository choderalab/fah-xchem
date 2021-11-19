import datetime as dt
import os
from typing import Optional
import json
import logging
from pathlib import Path

import click
from typing import Callable


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


@click.group()
def cli():
    ...


@cli.group()
def fragalysis():
    ...


@fragalysis.command()
@click.argument('structures-url', type=Path, help="Fragalysis URL for structures")
@click.argument('output-directory', type=Path, help="Directory to deposit output files")
def download_structures(structures_url, output_directory):
    from .prepare.fragalysis import FragalysisHarness

    fgh = FragalysisHarness(structures_url=structures_url)

    if not output_directory.exists() or not any(output_directory.iterdir()):
        logging.info(f"Downloading and extracting structure files files to {output_directory.absolute()}")
        fgh.download_target_structures(output_directory.absolute())


@cli.group()
def prepare():
    """Simulation preparation actions, pre-FAH compute.

    """
    ...


@prepare.command()
@click.argument('input-directory', type=Path, help="Path to receptors directory")
@click.argument('output-directory', type=Path, help="Directory to deposit output files")
@click.option('-f', '--structures-filter', type=str, help="Glob filter to find PDB files in STRUCTURES_DIRECTORY")
@click.option('-d', '--dry-run', is_flag=True, help="Dry run; output file paths will be printed STDOUT")
def receptors(
        input_directory,
        output_directory,
        structures_filter,
        dry_run
        ):
    """Prepare receptors for MD starting from Fragalysis receptor structures.

    """
    import itertools

    from openeye import oechem
    from .prepare.receptors import get_structures, ReceptorArtifactory

    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)


    input_directory = input_directory.absolute()

    source_pdb_files = list(input_directory.glob(structures_filter))
    if len(source_pdb_files) == 0:
        raise RuntimeError(f'Glob path {input_directory.joinpath(structures_filter)} '
                           f'has matched 0 files.')

    output_paths = [output_directory.absolute().joinpath(subdir) for subdir in ['monomer', 'dimer']]

    products = list(itertools.product(source_pdb_files, output_paths))
    factories = [ReceptorArtifactory(input=x, output=y, create_dimer=y.stem == 'dimer') for x, y in
               products]

    for factory in factories:
        if factory.output.exists():
            pass
        else:
            factory.output.mkdir(parents=True, exist_ok=True)

    for factory in factories:
        if dry_run:
            print(factory)
        else:
            factory.prepare_receptor()


@prepare.command()
def openmm():
    ...


@cli.group()
def analyze():
    """Analysis actions, using FAH data as input.
    
    """
    ...


@analyze.command()
@click.argument('compound-series-file', type=Path, help="File containing compound series as JSON-encoded CompoundSeries")
@click.argument('output-directory', type=Path, help="Directory to deposit output files")
@click.option('--config-file', type=Path, help="File containing analysis configuration as JSON-encoded AnalysisConfig")
@click.option('--fah-projects-dir', required=True, type=Path, help="Path to Folding@home project definitions directory")
@click.option('--fah-data-dir', required=True, type=Path, help="Path to Folding@home data directory")
@click.option('-n', '--nprocs', type=int, default=8, help="Number of parallel processes to run")
@click.option('--max-transformations', type=int, help="If not `None`, limit to this number of transformations")
@click.option('-l', '--loglevel', type=str, default='WARN', help="Logging level to use for execution")
def compound_series(
    compound_series_file,
    output_directory,
    config_file,
    fah_projects_dir,
    fah_data_dir,
    nprocs,
    max_transformations,
    loglevel,
) -> None:
    """
    Run free energy analysis and write JSON-serialized results consisting of
    input augmented with analysis results.

    """
    import fah_xchem
    from .schema import (
        AnalysisConfig,
        FahConfig,
        CompoundSeries,
        CompoundSeriesAnalysis,
        TimestampedAnalysis,
        FragalysisConfig,
    )

    logging.basicConfig(level=getattr(logging, loglevel.upper()))

    compound_series = _get_config(
        CompoundSeries, compound_series_file, "compound series"
    )

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

    series_analysis = fah_xchem.analysis.analyze_compound_series(
        series=compound_series,
        config=config,
        server=FahConfig(projects_dir=fah_projects_dir, data_dir=fah_data_dir),
        num_procs=nprocs,
    )

    timestamp = dt.datetime.now(dt.timezone.utc)
    output = TimestampedAnalysis(as_of=timestamp, series=series_analysis)

    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, "analysis.json"), "w") as output_file:
        output_file.write(output.json(indent=3))


@cli.group()
def generate():
    """Artifact generation actions, using analysis outputs as input.

    """

@click.argument('compound-series-analysis-file', type=Path, help="File containing analysis results as JSON-encoded CompoundSeriesAnalysis")
@click.argument('output-directory', type=Path, help="Directory to deposit output files")
@click.option('--config-file', type=Path, help="File containing analysis configuration as JSON-encoded AnalysisConfig")
@click.option('--fragalysis-config-file', type=Path, help="File containing information for Fragalysis upload as JSON-encoded FragalysisConfig")
@click.option('--fah-projects-dir', required=True, type=Path, help="Path to Folding@home project definitions directory")
@click.option('--fah-data-dir', required=True, type=Path, help="Path to Folding@home data directory")
@click.option('--fah-api-url', type=Path, help="URL to Folding@home work server API")
@click.option('-n', '--nprocs', type=int, default=8, help="Number of parallel processes to use for analysis components")
@click.option('--website-base-url', type=str, default='/', 
              help=("Base URL to use for links in the static site. E.g. if using S3, something like "
                   "https://fah-ws3.s3.amazonaws.com/covid-moonshot/sprints/sprint-4/2020-09-06-ugi-tBu-x3110-3v3m-2020-04-Jacobs/"))
@click.option('--website-base-url', type=str, default='/', help="")
@click.option('--cache-dir', type=Path, help="If given, cache intermediate results in a local directory with this name")
@click.option('--snapshots', is_flag=True, help="Whether to generate representative snapshots")
@click.option('--plots', is_flag=True, help="Whether to generate plots")
@click.option('--report', is_flag=True, help="Whether to generate PDF report")
@click.option('--website', is_flag=True, help="Whether to generate HTML for static site")
@click.option('-l', '--loglevel', type=str, default='WARN', help="Logging level to use for execution")
@click.option('--overwrite', is_flag=True, 
              help=("If `True`, write over existing output files if present."
                    "Otherwise, skip writing output files for a given transformation when already present."
                    "Assumes that for a given `run_id` the output files do not ever change;"
                    "does *no* checking that files wouldn't be different if inputs for a given `run_id` have changed."))
@generate.command()
def artifacts(
    compound_series_analysis_file: str,
    output_directory,
    config_file,
    fah_projects_dir,
    fah_data_dir,
    fah_api_url,
    nprocs,
    website_base_url,
    cache_dir: Optional[str] = None,
    snapshots: bool = True,
    plots: bool = True,
    report: bool = True,
    website: bool = True,
    loglevel: str = "WARN",
    fragalysis_config: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Given results of free energy analysis as JSON, generate analysis
    artifacts in `output_directory`

    By default the following are generated:

    - representative snapshots
    - plots
    - PDF report
    - static HTML for website

    """
    import fah_xchem
    from .schema import (
        AnalysisConfig,
        FahConfig,
        TimestampedAnalysis,
        FragalysisConfig,
    )

    logging.basicConfig(level=getattr(logging, loglevel.upper()))

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
        output_dir=output_directory,
        base_url=website_base_url,
        config=config,
        server=server,
        cache_dir=cache_dir,
        num_procs=nprocs,
        snapshots=snapshots,
        plots=plots,
        report=report,
        website=website,
        fragalysis_config=fragalysis_config,
        overwrite=overwrite,
    )
