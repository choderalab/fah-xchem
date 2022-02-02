import datetime as dt
import os
from typing import Optional
import json
import logging
from pathlib import Path

import click
from typing import Callable

from .schema import (
    AnalysisConfig,
    FahConfig,
    CompoundSeries,
    CompoundSeriesAnalysis,
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
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


def _parse_config(args, config):

    if config:
        config_values = {
            key.replace("-", "_"): value for key, value in json.load(config).items()
        }
    else:
        config_values = {}

    for arg, value in args.items():
        if value and arg != "config":
            config_values.update({arg: value})

    return config_values


# TODO: this should go in a separate module, not here
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


# TODO: this should go in a separate module, not here
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


@click.group()
def cli():
    ...


@cli.group()
def fragalysis():
    """Commands for interaction with Fragalysis."""
    ...


@fragalysis.command()
@click.option("--structures-url", type=str)
@click.option("--data-dir", required=True, type=Path)
@click.option(
    "--config",
    type=click.File("r"),
    help="Input via JSON; command-line arguments take precedence over JSON fields if both provided",
)
def retrieve_target_structures(structures_url, data_dir, config):
    """Get target structures from Fragalysis STRUCTURES-URL and place in DATA-DIR."""
    from .external.fragalysis import FragalysisData

    args = locals()
    config_values = _parse_config(args, config)

    fgh = FragalysisData(**config_values)

    if not data_dir.exists() or not any(data_dir.iterdir()):
        logging.info(
            f"Downloading and extracting structure files files to {data_dir.absolute()}"
        )
        fgh.retrieve_target_structures()


@cli.group()
@click.option(
    "--base-url", default="https://app.collaborativedrug.com/api/v1/vaults", type=str
)
@click.option("--vault-token", envvar="CDD_VAULT_TOKEN", type=str)
@click.option("--vault-num", envvar="CDD_VAULT_NUM", default="5549", type=str)
@click.option("--data-dir", required=True, type=Path)
@click.option(
    "--config",
    type=click.File("r"),
    help="Input via JSON; command-line arguments take precedence over JSON fields if both provided",
)
@click.pass_context
def cdd(ctx, base_url, vault_token, vault_num, data_dir, config):
    """Commands for interaction with CDD.

    All subcommands require DATA_DIR be specified.
    VAULT-TOKEN and VAULT-NUM can be set with the environment variables CDD_VAULT_TOKEN and CDD_VAULT_NUM, respectively.
    """
    ctx.ensure_object(dict)

    args = locals()
    config_values = _parse_config(args, config)

    ctx.obj["CONFIG_VALUES"] = config_values


@cdd.command()
@click.pass_context
def retrieve_molecule_data(ctx):
    """Get molecule data from CDD and place in DATA-DIR."""
    from .external.cdd import CDDData

    config_values = ctx.obj["CONFIG_VALUES"]

    cddd = CDDData(**config_values)
    cddd.retrieve_molecule_data()


@cdd.command()
@click.option("-i", "--protocol-id", type=str, multiple=True)
@click.option("-m", "--molecules", is_flag=True)
@click.pass_context
def retrieve_protocol_data(ctx, protocol_id, molecules):
    """Get protocol data from CDD and place in DATA-DIR.

    Multiple PROTOCOL_IDs can be given with multiple uses of `-i`.
    """
    from .external.cdd import CDDData

    config_values = ctx.obj["CONFIG_VALUES"]

    cddd = CDDData(**config_values)
    cddd.retrieve_protocol_data(protocol_ids=protocol_id, molecules=molecules)


@cdd.command()
@click.option("-i", "--protocol-id", type=str, multiple=True)
@click.argument("experimental_compound_data_file", type=Path)
@click.pass_context
def generate_experimental_compound_data(
    ctx, protocol_id, experimental_compound_data_file
):
    """Generate experimental compound data file including selected protocols.

    Multiple PROTOCOL_IDs can be given with multiple uses of `-i`.
    """
    from .external.cdd import CDDData

    config_values = ctx.obj["CONFIG_VALUES"]

    cddd = CDDData(**config_values)
    experimental_compound_data = cddd.generate_experimental_compound_data(
        protocol_ids=protocol_id
    )

    if not experimental_compound_data_file.parent.exists():
        experimental_compound_data_file.mkdir(parents=True)

    with open(experimental_compound_data_file, "w") as f:
        f.write(experimental_compound_data.json())


@cli.group()
def receptors():
    ...


@receptors.command("generate")
@click.argument("input-structures", type=Path, nargs=-1)
@click.argument("output-directory", type=Path)
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="Dry run; output file paths will be printed STDOUT",
)
def receptors_generate(input_structures, output_directory, dry_run):
    """Prepare receptors for MD starting from Fragalysis receptor structures in INPUT-DIRECTORY.

    Receptors are processed using the OpenEye toolkit to ensure consistency from raw PDB structures.


    """
    import itertools

    from openeye import oechem
    from .prepare.receptors import Receptors

    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)

    # TODO: generalize beyond monomer, dimer hardcoding here
    output_paths = [
        output_directory.absolute().joinpath(subdir) for subdir in ["monomer", "dimer"]
    ]

    products = list(itertools.product(input_structures, output_paths))
    factories = [
        Receptors(input=x, output=y, create_dimer=y.stem == "dimer")
        for x, y in products
    ]

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
# @click.argument('receptors-directory', type=Path)
# @click.argument('project-directory', type=Path)
# @click.option('-m', '--metadata', type=Path, help="Metadata CSV file from Fragalysis")
# @click.option('-p', '--project', type=str, required=True, help="Folding@Home project code")
# @click.option('-r', '--run', type=str, required=True, help="RUN index to prepare (zero-indexed selection of first column of METADATA)")
# @prepare.command()
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
    with open(metadata, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            run = row["RUN"]
            metadata[run] = row

    # Extract relevant metadata
    run = f"RUN{run}"
    if run not in metadata:
        raise Exception(f"{run} not found in metadata.csv")
    print(f"Preparing {run}")
    metadata = metadata[run]

    # Extract crystal_name
    crystal_name = metadata["crystal_name"]

    fp = FAHProject(project=project, project_dir=project_directory)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "cache.json")

        fp.generate_run(run)

        # TODO remove project hardcodes
        # prepare_variant('13430', args.run, crystal_name, 'monomer', 'His41(0) Cys145(0)', None)
    # prepare_variant('13431', run, crystal_name, 'monomer', 'His41(+) Cys145(-)', None)
    # if oemol is not None:
    #     # prepare_variant('13432', args.run, crystal_name, 'monomer', 'His41(0) Cys145(0)', oemol)
    #     prepare_variant('13433', run, crystal_name, 'monomer', 'His41(+) Cys145(-)', oemol)
    # # prepare_variant('13434', args.run, crystal_name, 'dimer',   'His41(0) Cys145(0)', None)
    # prepare_variant('13435', run, crystal_name, 'dimer', 'His41(+) Cys145(-)', None)
    # if oemol is not None:
    #     # prepare_variant('13436', args.run, crystal_name, 'dimer',   'His41(0) Cys145(0)', oemol)
    #     prepare_variant('13437', run, crystal_name, 'dimer', 'His41(+) Cys145(-)', oemol)


@cli.command()
def poses():
    ...


@cli.command()
def transformations():
    ...


@cli.group()
def compound_series():
    """Modification actions for a compound series."""
    ...


@compound_series.command("generate")
def compound_series_generate():
    ...


# @compound_series.group('update')
# def compound_series_update():
#    ...

# TODO want to support STDIN for any of these
# @compound_series_update.command('experimental-data')
# @click.argument('compound-series-file', type=click.File('r'))
# @click.argument('compound-series-update-file', type=click.File('r'))
# @click.argument('new-compound-series-file', type=click.File('w'))
# def compound_series_update(compound_series_file, compound_series_analysis_file, new_compound_series_file):
#    """
#
#    """
#    from .compute import CompoundSeries, CompoundSeriesUpdate
#
#    cs = CompoundSeries.parse_obj(json.load(compound_series_file))
#    csu = CompoundSeriesUpdate.parse_obj(json.load(compound_series_file))
#
#    metadata = [c.metadata for c in csu.compounds]
#    cs.update_experimental_data(metadata=metadata)
#
#    new_compound_series_file.write(cs.json())


@compound_series.command("analyze")
@click.argument("compound-series-file", type=Path)
@click.argument("compound-series-analysis-file", type=Path)
@click.option(
    "--config-file",
    type=Path,
    help="File containing analysis configuration as JSON-encoded AnalysisConfig",
)
@click.option(
    "--fah-projects-dir",
    required=True,
    type=Path,
    help="Path to Folding@home project definitions directory",
)
@click.option(
    "--fah-data-dir",
    required=True,
    type=Path,
    help="Path to Folding@home data directory",
)
@click.option(
    "-n", "--nprocs", type=int, default=8, help="Number of parallel processes to run"
)
@click.option(
    "--max-transformations",
    type=int,
    help="If not `None`, limit to this number of transformations",
)
@click.option(
    "--experimental-data-file",
    type=Path,
    help="If given, load experimental compound data and update compound `experimental_data` dictionaries",
)
@click.option(
    "-l",
    "--loglevel",
    type=str,
    default="WARN",
    help="Logging level to use for execution",
)
def compound_series_analyze(
    compound_series_file,
    compound_series_analysis_file,
    config_file,
    fah_projects_dir,
    fah_data_dir,
    nprocs,
    max_transformations,
    experimental_data_file,
    loglevel,
):
    """
    Run free energy analysis on COMPOUND-SERIES-FILE with `CompoundSeries` data
    and write JSON-serialized results consisting of input augmented with
    analysis results to COMPOUND-SERIES-ANALYSIS-FILE.

    """
    from .analysis import analyze_compound_series

    logging.basicConfig(level=getattr(logging, loglevel.upper()))

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
        num_procs=nprocs,
    )

    timestamp = dt.datetime.now(dt.timezone.utc)
    output = TimestampedAnalysis(as_of=timestamp, series=series_analysis)

    os.makedirs(compound_series_analysis_file.parent, exist_ok=True)
    with open(compound_series_analysis_file, "w") as output_file:
        output_file.write(output.json(indent=3))


@cli.group()
def artifacts():
    """Artifact generation actions, using analysis outputs as input."""
    ...


@artifacts.command("generate")
@click.argument("compound-series-analysis-file", type=Path)
@click.argument("output-directory", type=Path)
@click.option(
    "--config-file",
    type=Path,
    help="File containing analysis configuration as JSON-encoded AnalysisConfig",
)
@click.option(
    "--fragalysis-config-file",
    type=Path,
    help="File containing information for Fragalysis upload as JSON-encoded FragalysisConfig",
)
@click.option(
    "--fah-projects-dir",
    required=True,
    type=Path,
    help="Path to Folding@home project definitions directory",
)
@click.option(
    "--fah-data-dir",
    required=True,
    type=Path,
    help="Path to Folding@home data directory",
)
@click.option("--fah-api-url", type=str, help="URL to Folding@home work server API")
@click.option(
    "-n",
    "--nprocs",
    type=int,
    default=8,
    help="Number of parallel processes to use for analysis components",
)
@click.option(
    "--website-base-url",
    type=str,
    default="/",
    help=(
        "Base URL to use for links in the static site. E.g. if using S3, something like "
        "https://fah-ws3.s3.amazonaws.com/covid-moonshot/sprints/sprint-4/2020-09-06-ugi-tBu-x3110-3v3m-2020-04-Jacobs/"
    ),
)
@click.option("--website-base-url", type=str, default="/", help="")
@click.option(
    "--cache-dir",
    type=Path,
    help="If given, cache intermediate results in a local directory with this name",
)
@click.option(
    "--snapshots/--no-snapshots", default=True, help="Whether to generate representative snapshots"
)
@click.option("--plots/--no-plots", default=True, help="Whether to generate plots")
@click.option("--report/--no-report", default=True, help="Whether to generate PDF report")
@click.option(
    "--website/--no-website", default=True, help="Whether to generate HTML for static site"
)
@click.option(
    "-l",
    "--loglevel",
    type=str,
    default="WARN",
    help="Logging level to use for execution",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help=(
        "If `True`, write over existing output files if present."
        "Otherwise, skip writing output files for a given transformation when already present."
        "Assumes that for a given `run_id` the output files do not ever change;"
        "does *no* checking that files wouldn't be different if inputs for a given `run_id` have changed."
    ),
)
def artifacts_generate(
    compound_series_analysis_file,
    output_directory,
    config_file,
    fragalysis_config_file,
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
    from .analysis import generate_artifacts

    logging.basicConfig(level=getattr(logging, loglevel.upper()))

    config = _get_config(AnalysisConfig, config_file, "analysis configuration")

    fragalysis_config = _get_config(
        FragalysisConfig, fragalysis_config_file, "fragalysis configuration"
    )

    with open(compound_series_analysis_file, "r") as infile:
        tsa = TimestampedAnalysis.parse_obj(json.load(infile))

    server = FahConfig(
        projects_dir=fah_projects_dir, data_dir=fah_data_dir, api_url=fah_api_url
    )

    return generate_artifacts(
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
