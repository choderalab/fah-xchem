from collections import defaultdict
import datetime as dt
from functools import partial
import logging
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
)
from .diffnet import combine_free_energies
from .exceptions import AnalysisError, DataValidationError
from .extract_work import extract_work_pair
from .free_energy import compute_relative_free_energy
from .plots import generate_plots
from .report import generate_report
from .website import generate_website


def analyze_phase(server: FahConfig, run: int, project: int, config: AnalysisConfig):

    paths = list_results(config=server, run=run, project=project)

    if not paths:
        raise AnalysisError(f"No data found for project {project}, RUN {run}")

    works_by_gen = defaultdict(list)
    for path in paths:
        try:
            work_pair = extract_work_pair(path.path)
            works_by_gen[path.gen].append(work_pair)
        except DataValidationError as exc:
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
) -> TransformationAnalysis:

    analyze_phase_partial = partial(
        analyze_phase, server=server, run=transformation.run_id, config=config
    )

    complex_phase = analyze_phase_partial(project=projects.complex_phase)
    solvent_phase = analyze_phase_partial(project=projects.solvent_phase)

    return TransformationAnalysis(
        transformation=transformation,
        binding_free_energy=solvent_phase.free_energy.delta_f
        - complex_phase.free_energy.delta_f,
        complex_phase=complex_phase,
        solvent_phase=solvent_phase,
    )


def analyze_compound_series(
    series: CompoundSeries,
    config: AnalysisConfig,
    server: FahConfig,
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = None,
) -> CompoundSeriesAnalysis:

    # TODO: multiprocessing, progress bar
    transformations = [
        analyze_transformation(
            transformation=transformation,
            projects=series.metadata.fah_projects,
            server=server,
            config=config,
        )
        for transformation in series.transformations
    ]

    compounds = combine_free_energies(series.compounds, transformations)

    return CompoundSeriesAnalysis(
        metadata=series.metadata, compounds=compounds, transformations=transformations
    )


def generate_artifacts(
    analysis: CompoundSeriesAnalysis, timestamp: dt.datetime, output_dir: str
) -> None:
    generate_plots(
        analysis=analysis,
        timestamp=timestamp,
        output_dir=os.path.join(output_dir, "plots"),
    )
    generate_report(analysis, output_dir)
    generate_website(series_analysis=analysis, path=output_dir, timestamp=timestamp)
