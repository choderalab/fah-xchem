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


def _parse_config(args, config):

    if config:
        config_values = {key.replace('-', '_'): value for key, value in json.load(config).items()}
    else:
        config_values = {}

    for arg, value in args.items():
        if value and arg != 'config':
            config_values.update({arg: value})

    return config_values


@click.group()
def cli():
    ...

@cli.group()
def fragalysis():
    ...


@fragalysis.command()
@click.option('--structures-url', type=str)
@click.option('--data-dir', required=True, type=Path)
@click.option('--config', type=click.File('r'), help="Input via JSON; command-line arguments take precedence over JSON fields if both provided")
def retrieve_target_structures(structures_url, data_dir, config):
    """Get target structures from Fragalysis STRUCTURES-URL and place in DATA-DIR.

    """
    from .external.fragalysis import FragalysisData

    args = locals()
    config_values = _parse_config(args, config)

    fgh = FragalysisData(**config_values)

    if not data_dir.exists() or not any(data_dir.iterdir()):
        logging.info(f"Downloading and extracting structure files files to {data_dir.absolute()}")
        fgh.retrieve_target_structures()


@cli.group()
@click.option('--base-url', default="https://app.collaborativedrug.com/api/v1/vaults", type=str)
@click.option('--vault-token', envvar='CDD_VAULT_TOKEN', type=str)
@click.option('--vault-num', envvar='CDD_VAULT_NUM', default='5549', type=str)
@click.option('--data-dir', required=True, type=Path)
@click.option('--config', type=click.File('r'), help="Input via JSON; command-line arguments take precedence over JSON fields if both provided")
@click.pass_context
def cdd(ctx, base_url, vault_token, vault_num, data_dir, config):
    ctx.ensure_object(dict)
    
    args = locals()
    config_values = _parse_config(args, config)

    ctx.obj['CONFIG_VALUES'] = config_values

@cdd.command()
@click.pass_context
def retrieve_molecule_data(ctx):
    """Get protocol data from CDD and place in DATA-DIR.

    VAULT-TOKEN and VAULT-NUM can be set with the environment variables CDD_VAULT_TOKEN and CDD_VAULT_NUM, respectively.

    """
    from .external.cdd import CDDData

    config_values = ctx.obj['CONFIG_VALUES']

    cddd = CDDData(**config_values)
    cddd.retrieve_molecule_data()


@cdd.command()
@click.option('-i', '--protocol-id', type=str, multiple=True)
@click.option('-m', '--molecules', is_flag=True)
@click.pass_context
def retrieve_protocol_data(ctx, protocol_id, molecules):
    """Get protocol data from CDD and place in DATA-DIR.

    VAULT-TOKEN and VAULT-NUM can be set with the environment variables CDD_VAULT_TOKEN and CDD_VAULT_NUM, respectively.

    """
    from .external.cdd import CDDData

    config_values = ctx.obj['CONFIG_VALUES']

    cddd = CDDData(**config_values)
    cddd.retrieve_protocol_data(protocol_id=protocol_id, molecules=molecules)

@cdd.command()
def generate_experimental_compound_data(protocol_id, molecules):
    ...


@cli.group()
def receptors():
    ...

@receptors.command('generate')
@click.argument('input-structures', type=Path, nargs=-1)
@click.argument('output-directory', type=Path)
@click.option('-d', '--dry-run', is_flag=True, help="Dry run; output file paths will be printed STDOUT")
def receptors_generate(
        input_structures,
        output_directory,
        dry_run
        ):
    """Prepare receptors for MD starting from Fragalysis receptor structures in INPUT-DIRECTORY.

    Receptors are processed using the OpenEye toolkit to ensure consistency from raw PDB structures.


    """
    import itertools

    from openeye import oechem
    from .prepare.receptors import ReceptorArtifactory

    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)

    # TODO: generalize beyond monomer, dimer hardcoding here
    output_paths = [output_directory.absolute().joinpath(subdir) for subdir in ['monomer', 'dimer']]

    products = list(itertools.product(input_structures, output_paths))
    factories = [ReceptorArtifactory(input=x, output=y, create_dimer=y.stem == 'dimer') for x, y in
               products]

    for factory in factories:
        if factory.output.exists():
            pass
        else:
            factory.output.mkdir(parents=True, exist_ok=True)

    # TODO: easily parallelizable
    for factory in factories:
        if dry_run:
            print(factory)
        else:
            factory.prepare_receptor()


# not used at this time; pulled almost exactly from fah_prep
#@click.argument('receptors-directory', type=Path)
#@click.argument('project-directory', type=Path)
#@click.option('-m', '--metadata', type=Path, help="Metadata CSV file from Fragalysis")
#@click.option('-p', '--project', type=str, required=True, help="Folding@Home project code")
#@click.option('-r', '--run', type=str, required=True, help="RUN index to prepare (zero-indexed selection of first column of METADATA)")
#@prepare.command()
def fah_project_run(
        receptors_directory,
        project_directory,
        metadata,
        project,
        run,
        ):
    """Prepare the specified RUN for FAH by preparing all X-ray structure variants of a specific fragment.

    RECEPTORS-DIRECTORY corresponds to the output of `prepare receptors`.
    Prepared FAH RUNs are placed in PROJECT-DIRECTORY.

    """
    import sys
    import csv
    from collections import OrderedDict
    import tempfile
    import traceback
    import yaml

    import oechem

    from .prepare.dynamics import FAHProject
    # Read DiamondMX/XChem structure medatadata

    # TODO: replace with pandas
    metadata = OrderedDict()
    with open(metadata, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            run = row['RUN']
            metadata[run] = row

    # Extract relevant metadata
    run = f'RUN{run}'
    if run not in metadata:
        raise Exception(f'{run} not found in metadata.csv')
    print(f'Preparing {run}')
    metadata = metadata[run]

    # Extract crystal_name
    crystal_name = metadata['crystal_name']

    fp = FAHProject(project=project,
                    project_dir=project_directory)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, 'cache.json')

        fp.generate_run(run, )

        # TODO remove project hardcodes
        # prepare_variant('13430', args.run, crystal_name, 'monomer', 'His41(0) Cys145(0)', None)
        prepare_variant('13431', run, crystal_name, 'monomer', 'His41(+) Cys145(-)', None)
        if oemol is not None:
            # prepare_variant('13432', args.run, crystal_name, 'monomer', 'His41(0) Cys145(0)', oemol)
            prepare_variant('13433', run, crystal_name, 'monomer', 'His41(+) Cys145(-)', oemol)
        # prepare_variant('13434', args.run, crystal_name, 'dimer',   'His41(0) Cys145(0)', None)
        prepare_variant('13435', run, crystal_name, 'dimer', 'His41(+) Cys145(-)', None)
        if oemol is not None:
            # prepare_variant('13436', args.run, crystal_name, 'dimer',   'His41(0) Cys145(0)', oemol)
            prepare_variant('13437', run, crystal_name, 'dimer', 'His41(+) Cys145(-)', oemol)


@cli.command()
def poses():
    ...

@cli.command()
def transformations():
    ...


@cli.group()
def compound_series():
    """Modification actions for a compound series.

    """
    ...


@compound_series.command('generate')
def compound_series_generate():
    ...


@compound_series.group('update')
def compound_series_update():
    ...

# TODO want to support STDIN for any of these
@compound_series_update.command('experimental-data')
@click.argument('compound-series-file', type=click.File('r'))
@click.argument('compound-series-update-file', type=click.File('r'))
@click.argument('new-compound-series-file', type=click.File('w'))
def compound_series_update(compound_series_file, compound_series_analysis_file, new_compound_series_file):
    """

    """
    from .compute import CompoundSeries, CompoundSeriesUpdate

    cs = CompoundSeries.parse_obj(json.load(compound_series_file))
    csu = CompoundSeriesUpdate.parse_obj(json.load(compound_series_file))

    metadata = [c.metadata for c in csu.compounds]
    cs.update_experimental_data(metadata=metadata)

    new_compound_series_file.write(cs.json())


@compound_series.command('analyze')
@click.argument('compound-series-file', type=Path)
@click.argument('compound-series-analysis-file', type=Path)
@click.option('--config-file', type=Path, help="File containing analysis configuration as JSON-encoded AnalysisConfig")
@click.option('--fah-projects-dir', required=True, type=Path, help="Path to Folding@home project definitions directory")
@click.option('--fah-data-dir', required=True, type=Path, help="Path to Folding@home data directory")
@click.option('-n', '--nprocs', type=int, default=8, help="Number of parallel processes to run")
@click.option('--max-transformations', type=int, help="If not `None`, limit to this number of transformations")
@click.option('-l', '--loglevel', type=str, default='WARN', help="Logging level to use for execution")
def compound_series_analyze(
    compound_series_file,
    compound_series_analysis_file,
    config_file,
    fah_projects_dir,
    fah_data_dir,
    nprocs,
    max_transformations,
    loglevel,
):
    """
    Run free energy analysis on COMPOUND-SERIES-FILE with `CompoundSeries` data
    and write JSON-serialized results consisting of input augmented with
    analysis results to COMPOUND-SERIES-ANALYSIS-FILE.

    """
    import fah_xchem
    from .compute import CompoundSeries
    from .schema import (
        AnalysisConfig,
        FahConfig,
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

    os.makedirs(compound_series , exist_ok=True)
    with open(os.path.join(compound_series_analysis_file, "analysis.json"), "w") as output_file:
        output_file.write(output.json(indent=3))


@cli.group()
def artifacts():
    """Artifact generation actions, using analysis outputs as input.

    """
    ...

@artifacts.command('generate')
@click.argument('compound-series-analysis-file', type=Path)
@click.argument('output-directory', type=Path)
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
def artifacts_generate(
    compound_series_analysis_file,
    output_directory,
    config_file,
    fah_projects_dir,
    fah_data_dir,
    fah_api_url,
    nprocs,
    website_base_url,
    cache_dir,
    snapshots,
    plots, 
    report,
    website,
    loglevel,
    fragalysis_config,
    overwrite,
):
    """
    Given results of free energy analysis in COMPOUND-SERIES-ANALYSIS-FILE with
    `CompoundSeriesAnalysis` data, generate analysis artifacts.

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
