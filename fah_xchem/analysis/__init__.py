from collections import defaultdict
from functools import partial
import logging
from math import sqrt
from typing import List, Optional

from ..fah_utils import list_results
from ..schema import (
    AnalysisConfig,
    Compound,
    CompoundAnalysis,
    CompoundSeries,
    CompoundSeriesAnalysis,
    FahConfig,
    GenAnalysis,
    PhaseAnalysis,
    PointEstimate,
    ProjectPair,
    Transformation,
    TransformationAnalysis,
)
from .exceptions import AnalysisError, DataValidationError
from .extract_work import extract_work_pair
from .free_energy import compute_relative_free_energy


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

    # TODO: rounding for raw work data?
    return PhaseAnalysis(
        free_energy=compute_relative_free_energy_partial(work_pairs=all_works),
        gens=[
            GenAnalysis(
                gen=gen,
                works=works,
                free_energy=compute_relative_free_energy_partial(work_pairs=works),
            )
            for gen, works in works_by_gen.items()
        ],
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
        binding_free_energy=PointEstimate(
            point=solvent_phase.free_energy.delta_f.point
            - complex_phase.free_energy.delta_f.point,
            stderr=sqrt(
                solvent_phase.free_energy.delta_f.stderr ** 2
                + complex_phase.free_energy.delta_f.stderr ** 2
            ),
        ),
        complex_phase=complex_phase,
        solvent_phase=solvent_phase,
    )


def analyze_compounds(
    compounds: List[Compound], transformations: List[TransformationAnalysis]
) -> List[CompoundAnalysis]:
    """
    Run DiffNet analysis
    # TODO
    """
    return []


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

    compounds = analyze_compounds(series.compounds, transformations)

    return CompoundSeriesAnalysis(
        metadata=series.metadata, compounds=compounds, transformations=transformations
    )
