import logging
from math import log, sqrt
from typing import Dict, List


from ..schema import (
    Compound,
    CompoundAnalysis,
    CompoundMicrostate,
    MicrostateAnalysis,
    PointEstimate,
    TransformationAnalysis,
)
from .exceptions import AnalysisError


def pIC50_to_dG(pIC50: float, s_conc: float = 375e-9, Km: float = 40e-6) -> float:
    """
    Converts IC50 (in M units) to DG
    Parameters
    ----------
    pIC50 : float
        pIC50
    s_conc : float, default=375E-9
        Substrate concentration in M
    Km : float, default=40E-6
        Substrate concentration for half-maximal enzyme activity
    Returns
    -------
    type
        Description of returned object.
    """
    ic50 = 10 ** -pIC50

    if ic50 > 1e-5:
        logging.warning("Expecting IC50 in M units. Please check.")

    Ki = ic50 / (1 + s_conc / Km)
    return log(Ki)


def estimate_expected_value(
    estimates: List[PointEstimate], weights: List[float]
) -> PointEstimate:
    pairs = zip(weights, estimates)
    return PointEstimate(
        point=sum(weight * estimate.point for weight, estimate in pairs),
        stderr=sqrt(sum((weight * estimate.stderr) ** 2 for weight, estimate in pairs)),
    )


# TODO
def get_ensemble_average(free_energies: List[PointEstimate]) -> PointEstimate:
    return free_energies[0]


def get_compound_analysis(
    compound: Compound,
    absolute_free_energy_by_microstate: Dict[CompoundMicrostate, PointEstimate],
) -> CompoundAnalysis:

    microstates = [
        MicrostateAnalysis(
            microstate=microstate,
            absolute_free_energy=absolute_free_energy_by_microstate[
                CompoundMicrostate(
                    compound_id=compound.metadata.compound_id,
                    microstate_id=microstate.microstate_id,
                )
            ],
        )
        for microstate in compound.microstates
    ]

    return CompoundAnalysis(
        metadata=compound.metadata,
        microstates=microstates,
        absolute_free_energy=get_ensemble_average(
            [ms.absolute_free_energy for ms in microstates]
        ),
    )


def combine_free_energies(
    compounds: List[Compound],
    transformations: List[TransformationAnalysis],
) -> List[CompoundAnalysis]:

    from arsenic import stats
    import networkx as nx
    import numpy as np

    graph = nx.DiGraph()

    for analysis in transformations:
        transformation = analysis.transformation

        if analysis.binding_free_energy is None:
            continue

        graph.add_edge(
            transformation.initial_microstate,
            transformation.final_microstate,
            f_ij=analysis.binding_free_energy.point,
            f_dij=analysis.binding_free_energy.stderr,
        )

    for compound in compounds:

        pIC50 = compound.metadata.experimental_data.get("pIC50")

        if pIC50 is None:
            continue

        for microstate in compound.microstates:

            node = CompoundMicrostate(
                compound_id=compound.metadata.compound_id,
                microstate_id=microstate.microstate_id,
            )

            if node not in graph:
                raise AnalysisError(
                    f"Compound microstate '{node.microstate_id}' has "
                    f"experimental data, but does not appear in any "
                    f"transformation"
                )

            graph.nodes[node]["exp_DG"] = pIC50_to_dG(pIC50)

    if not nx.is_weakly_connected(graph):
        sizes = [len(g) for g in nx.weakly_connected_components(graph)]
        raise AnalysisError(
            f"Only weakly-connected graphs are supported, but found weakly-connected subgraphs of sizes {sizes}"
        )

    f_i, C = stats.mle(graph, factor="f_ij", node_factor="exp_DG")
    errs = np.diag(C)

    absolute_free_energy_by_microstate = {
        microstate: PointEstimate(point=delta_f, stderr=ddelta_f)
        for microstate, delta_f, ddelta_f in zip(graph.nodes(), f_i, errs)
    }

    return [
        get_compound_analysis(compound, absolute_free_energy_by_microstate)
        for compound in compounds
    ]
