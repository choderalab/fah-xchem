from collections import defaultdict
import datetime as dt
from functools import partial
import logging
import multiprocessing
import os
from typing import List, Optional

from ..fah_utils import list_results
from ..schema import (
    AnalysisConfig,
    CompoundSeries,
    CompoundSeriesAnalysis,
    FahConfig,
    GenAnalysis,
    PhaseAnalysis,
    ProjectPair,
    Transformation,
    TransformationAnalysis,
    WorkPair,
    FragalysisConfig,
)
from .diffnet import combine_free_energies
from .exceptions import AnalysisError, DataValidationError
from .extract_work import extract_work_pair
from .free_energy import compute_relative_free_energy
from .plots import generate_plots
from .report import generate_report, gens_are_consistent
from .structures import generate_representative_snapshots
from .website import generate_website


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

    return PhaseAnalysis(
        free_energy=free_energy,
        gens=[get_gen_analysis(gen, works) for gen, works in works_by_gen.items()],
    )


def analyze_transformation(
    transformation: Transformation,
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

    # Check for consistency across GENS, if requested
    consistent_bool = None
    if filter_gen_consistency:
        consistent_bool = gens_are_consistent(
            complex_phase=complex_phase, solvent_phase=solvent_phase, nsigma=3
        )

    return TransformationAnalysis(
        transformation=transformation,
        reliable_transformation=consistent_bool,
        binding_free_energy=binding_free_energy,
        complex_phase=complex_phase,
        solvent_phase=solvent_phase,
    )

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

    from rich.progress import track

    with multiprocessing.Pool(num_procs) as pool:
        results_iter = pool.imap_unordered(
            partial(
                analyze_transformation_or_warn,
                projects=series.metadata.fah_projects,
                server=server,
                config=config,
            ),
            series.transformations,
        )
        transformations = [
            result
            for result in track(
                results_iter,
                total=len(series.transformations),
                description="Computing transformation free energies",
            )
            if result is not None
        ]

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

    if snapshots:
        logging.info("Generating representative snapshots")
        generate_representative_snapshots(
            transformations=series.transformations,
            project_dir=complex_project_dir,
            project_data_dir=complex_data_dir,
            output_dir=os.path.join(output_dir, "transformations"),
            max_binding_free_energy=config.max_binding_free_energy,
            cache_dir=cache_dir,
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
