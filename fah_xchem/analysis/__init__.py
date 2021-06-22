from collections import defaultdict
import datetime as dt
from functools import partial
import logging
import multiprocessing
import os
from typing import List, Optional
import networkx as nx
import numpy as np

from ..fah_utils import list_results
from ..schema import (
    AnalysisConfig,
    CompoundSeries,
    CompoundSeriesAnalysis,
    CompoundMicrostate,
    FahConfig,
    GenAnalysis,
    PhaseAnalysis,
    PointEstimate,
    ProjectPair,
    Transformation,
    TransformationAnalysis,
    WorkPair,
    FragalysisConfig,
    RunStatus,
)
from .constants import KT_KCALMOL
from .diffnet import combine_free_energies, pIC50_to_DG
from .exceptions import AnalysisError, DataValidationError
from .extract_work import extract_work_pair
from .free_energy import compute_relative_free_energy, InsufficientDataError
from .plots import generate_plots
from .report import generate_report, gens_are_consistent
from .structures import SnapshotArtifactory
from .website import generate_website


EXP_DDG_IJ_ERR = 0.2  # TODO check this is correct


def analyze_phase(server: FahConfig, run: int, project: int, config: AnalysisConfig):

    paths = list_results(config=server, run=run, project=project)

    if not paths:
        raise AnalysisError(f"No data found for project {project}, RUN {run}")

    works_by_gen = defaultdict(list)
    for path in paths:
        try:
            work_pair = extract_work_pair(path)
            works_by_gen[path.gen].append(work_pair)
        except (DataValidationError, ValueError) as exc:
            logging.warning("Failed to extract work values from %s: %s", path, exc)

    # flattened list of work pairs for all GENs
    all_works = [work_pair for works in works_by_gen.values() for work_pair in works]

    # configure free energy computation
    compute_relative_free_energy_partial = partial(
        compute_relative_free_energy, min_num_work_values=config.min_num_work_values
    )

    free_energy, _ = compute_relative_free_energy_partial(work_pairs=all_works)

    def get_gen_analysis(gen: int, works: List[WorkPair]) -> GenAnalysis:
        free_energy, filtered_works = compute_relative_free_energy_partial(
            work_pairs=works
        )
        # TODO: round raw work output?
        return GenAnalysis(gen=gen, works=filtered_works, free_energy=free_energy)

    # Analyze gens, omitting incomplete gens
    gens = list()
    for gen, works in works_by_gen.items():
        try:
            gens.append(get_gen_analysis(gen, works))
        except InsufficientDataError as e:
            # It's OK if we don't have sufficient data here
            pass

    return PhaseAnalysis(
        free_energy=free_energy,
        gens=gens,
    )


def analyze_transformation(
    transformation: Transformation,
    compounds: CompoundSeries,
    projects: ProjectPair,
    server: FahConfig,
    config: AnalysisConfig,
    filter_gen_consistency: Optional[bool] = True,
) -> TransformationAnalysis:

    analyze_phase_partial = partial(
        analyze_phase, server=server, run=transformation.run_id, config=config
    )

    complex_phase = analyze_phase_partial(project=projects.complex_phase)
    solvent_phase = analyze_phase_partial(project=projects.solvent_phase)
    binding_free_energy = (
        complex_phase.free_energy.delta_f - solvent_phase.free_energy.delta_f
    )

    # get associated DDGs between compounds, if experimentally known
    exp_ddg = calc_exp_ddg(transformation=transformation, compounds=compounds)
    absolute_error = (
        abs(binding_free_energy - exp_ddg) if (exp_ddg.point is not None) else None
    )

    # Check for consistency across GENS, if requested
    consistent_bool = None
    if filter_gen_consistency:
        consistent_bool = gens_are_consistent(
            complex_phase=complex_phase, solvent_phase=solvent_phase, nsigma=1
        )

        return TransformationAnalysis(
            transformation=transformation,
            reliable_transformation=consistent_bool,
            binding_free_energy=binding_free_energy,
            complex_phase=complex_phase,
            solvent_phase=solvent_phase,
            exp_ddg=exp_ddg,
            absolute_error=absolute_error,
        )

    else:

        return TransformationAnalysis(
            transformation=transformation,
            binding_free_energy=binding_free_energy,
            complex_phase=complex_phase,
            solvent_phase=solvent_phase,
            exp_ddg=exp_ddg,
        )

    return TransformationAnalysis(
        transformation=transformation,
        reliable_transformation=consistent_bool,
        binding_free_energy=binding_free_energy,
        complex_phase=complex_phase,
        solvent_phase=solvent_phase,
    )


def calc_exp_ddg(transformation: TransformationAnalysis, compounds: CompoundSeries):
    """
    Compute experimental free energy difference between two compounds, if available.

    NOTE: This method makes the approximation that each microstate has the same affinity as the parent compound.

    TODO: Instead, solve DiffNet without experimental data and use derived DDGs between compounds (not transformations).

    Parameters
    ----------
    transformation : TransformationAnalysis
        The transformation of interest
    compounds : CompoundSeries
       Data for the compound series.

    Returns
    -------
    ddg : PointEstimate
        Point estimate of free energy difference for this transformation,
        or PointEstimate(None, None) if not available.

    """
    compounds_by_microstate = {
        microstate.microstate_id: compound
        for compound in compounds
        for microstate in compound.microstates
    }

    initial_experimental_data = compounds_by_microstate[
        transformation.initial_microstate.microstate_id
    ].metadata.experimental_data
    final_experimental_data = compounds_by_microstate[
        transformation.final_microstate.microstate_id
    ].metadata.experimental_data

    if ("pIC50" in initial_experimental_data) and ("pIC50" in final_experimental_data):
        initial_dg = PointEstimate(
            point=pIC50_to_DG(initial_experimental_data["pIC50"]), stderr=EXP_DDG_IJ_ERR
        )
        final_dg = PointEstimate(
            point=pIC50_to_DG(final_experimental_data["pIC50"]), stderr=EXP_DDG_IJ_ERR
        )
        error = final_dg - initial_dg
        return error
    else:
        return PointEstimate(point=None, stderr=None)


def analyze_transformation_or_warn(
    transformation: Transformation, **kwargs
) -> Optional[TransformationAnalysis]:

    try:
        return analyze_transformation(transformation, **kwargs)
    except AnalysisError as exc:
        logging.warning("Failed to analyze RUN%d: %s", transformation.run_id, exc)
        return None


def analyze_compound_series(
    series: CompoundSeries,
    config: AnalysisConfig,
    server: FahConfig,
    num_procs: Optional[int] = None,
) -> CompoundSeriesAnalysis:
    """
    Analyze a compound series to generate JSON.

    """
    from rich.progress import track

    # TODO: Cache results and only update RUNs for which we have received new data

    # Pre-filter based on which transformations have any work data
    logging.info(
        f"Pre-filtering {len(series.transformations)} transformations to identify those with work data..."
    )
    available_transformations = [
        transformation
        for transformation in series.transformations
        if len(
            list_results(
                config=server,
                run=transformation.run_id,
                project=series.metadata.fah_projects.complex_phase,
            )
        )
        > 0
        and len(
            list_results(
                config=server,
                run=transformation.run_id,
                project=series.metadata.fah_projects.solvent_phase,
            )
        )
        > 0
    ]
    # available_transformations = series.transformations[:50]

    # Process compound series in parallel
    logging.info(
        f"Processing {len(available_transformations)} / {len(series.transformations)} available transformations in parallel..."
    )
    with multiprocessing.Pool(num_procs) as pool:
        results_iter = pool.imap_unordered(
            partial(
                analyze_transformation_or_warn,
                projects=series.metadata.fah_projects,
                server=server,
                config=config,
                compounds=series.compounds,
            ),
            available_transformations,
        )
        transformations = [
            result
            for result in track(
                results_iter,
                total=len(available_transformations),
                description="Computing transformation free energies",
            )
            if result is not None
        ]

    # Reprocess transformation experimental errors to only include most favorable transformation
    # NOTE: This is a hack, and should be replaced by a more robust method for accounting for racemic mixtures
    # Compile list of all microstate transformations for each compound
    compound_ddgs = dict()
    for transformation in transformations:
        compound_id = transformation.transformation.final_microstate.compound_id
        if compound_id in compound_ddgs:
            compound_ddgs[compound_id].append(transformation.binding_free_energy.point)
        else:
            compound_ddgs[compound_id] = [transformation.binding_free_energy.point]
    # Collapse to a single estimate
    from scipy.special import logsumexp

    for compound_id, ddgs in compound_ddgs.items():
        compound_ddgs[compound_id] = -logsumexp(-np.array(ddgs)) + np.log(len(ddgs))
    # Regenerate list of transformations
    for index, t in enumerate(transformations):
        if (t.exp_ddg is None) or (t.exp_ddg.point is None):
            continue
        compound_id = t.transformation.final_microstate.compound_id
        absolute_error_point = abs(t.exp_ddg.point - compound_ddgs[compound_id])
        transformations[index] = TransformationAnalysis(
            transformation=t.transformation,
            reliable_transformation=t.reliable_transformation,
            binding_free_energy=t.binding_free_energy,
            complex_phase=t.complex_phase,
            solvent_phase=t.solvent_phase,
            exp_ddg=t.exp_ddg,
            absolute_error=PointEstimate(
                point=absolute_error_point, stderr=t.absolute_error.stderr
            ),
        )

    # Sort transformations by RUN
    # transformations.sort(key=lambda transformation_analysis : transformation_analysis.transformation.run_id)
    # Sort transformations by free energy difference
    transformations.sort(
        key=lambda transformation_analysis: transformation_analysis.binding_free_energy.point
    )

    # Warn about failures
    num_failed = len(series.transformations) - len(transformations)
    if num_failed > 0:
        logging.warning(
            "Failed to process %d transformations out of %d",
            num_failed,
            len(series.transformations),
        )

    logging.info("Running DiffNet compound free energy analysis")
    compounds = combine_free_energies(series.compounds, transformations)

    return CompoundSeriesAnalysis(
        metadata=series.metadata, compounds=compounds, transformations=transformations
    )


def generate_artifacts(
    series: CompoundSeriesAnalysis,
    fragalysis_config: FragalysisConfig,
    timestamp: dt.datetime,
    projects_dir: str,
    data_dir: str,
    output_dir: str,
    base_url: str,
    config: AnalysisConfig,
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = None,
    snapshots: bool = True,
    plots: bool = True,
    report: bool = True,
    website: bool = True,
    overwrite: bool = False,
) -> None:

    complex_project_dir = os.path.join(
        projects_dir, str(series.metadata.fah_projects.complex_phase)
    )

    complex_data_dir = os.path.join(
        data_dir, f"PROJ{series.metadata.fah_projects.complex_phase}"
    )

    # Pre-filter based on which transformations have any data
    available_transformations = [
        transformation
        for transformation in series.transformations
        if transformation.binding_free_energy is not None
        and transformation.binding_free_energy.point is not None
    ]

    if snapshots:
        logging.info("Generating representative snapshots")
        saf = SnapshotArtifactory(
            config=config,
            project_dir=complex_project_dir,
            project_data_dir=complex_data_dir,
            cache_dir=cache_dir,
        )

        saf.generate_representative_snapshots(
            transformations=available_transformations,
            output_dir=os.path.join(output_dir, "transformations"),
            num_procs=num_procs,
            overwrite=overwrite,
        )

    if plots:
        logging.info("Generating analysis plots")
        generate_plots(
            series=series,
            timestamp=timestamp,
            output_dir=output_dir,
            num_procs=num_procs,
            overwrite=overwrite,
        )

    if snapshots and report:
        logging.info("Generating pdf report")
        generate_report(
            series=series,
            results_path=output_dir,
            max_binding_free_energy=config.max_binding_free_energy,
            fragalysis_config=fragalysis_config,
        )

    if website:
        logging.info("Generating website")
        generate_website(
            series=series,
            path=output_dir,
            timestamp=timestamp,
            base_url=base_url,
        )
