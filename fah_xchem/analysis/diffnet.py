import logging
from math import log
from typing import List

import numpy as np

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


def get_compound_free_energy(microstates: List[MicrostateAnalysis]) -> PointEstimate:
    r"""
    Compute compound free energy from microstate free energies and
    microstate free energy penalties.

    The sum over microstates is given by

    .. math:: g_c = - \log \sum_{i} \exp \left[-(s_{ci} + g_{ci}) \right]

    Parameters
    ----------
    microstates : list of MicrostateAnalysis
        Microstate free energies
    """
    penalized_free_energies = [
        microstate.microstate.free_energy_penalty + microstate.free_energy
        for microstate in microstates
    ]

    gp = np.array([x.point for x in penalized_free_energies])
    stderr = np.array([x.stderr for x in penalized_free_energies])

    x = np.exp(-gp)
    z = np.sum(x)
    y = np.log(z)
    dy = -x / z

    return PointEstimate(point=y, stderr=np.sqrt(np.sum((dy * stderr) ** 2)))


def combine_free_energies(
    compounds: List[Compound],
    transformations: List[TransformationAnalysis],
) -> List[CompoundAnalysis]:
    """
    Perform DiffNet MLE analysis to compute free energies for all
    microstates given experimental free energies for a subset, and
    relative free energies of transformations.

    Parameters
    ----------
    compounds : list of Compound
    transformations : list of Transformation

    Returns
    -------
    List of CompoundAnalysis
        Result of DiffNet MLE analysis
    """

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

            if node in graph:
                graph.nodes[node]["exp_DG"] = pIC50_to_dG(pIC50)
            else:
                logging.warning(
                    "Compound microstate '%s' has experimental data, "
                    "but does not appear in any transformation",
                    node.microstate_id,
                )

    # TODO: support disconnected graphs
    if not nx.is_weakly_connected(graph):
        sizes = [len(g) for g in nx.weakly_connected_components(graph)]
        raise AnalysisError(
            f"Only weakly-connected graphs are supported, but found weakly-connected subgraphs of sizes {sizes}"
        )

    f_i, C = stats.mle(graph, factor="f_ij", node_factor="exp_DG")
    errs = np.diag(C)

    free_energy_by_microstate = {
        microstate: PointEstimate(point=delta_f, stderr=ddelta_f)
        for microstate, delta_f, ddelta_f in zip(graph.nodes(), f_i, errs)
    }

    microstates = [
        MicrostateAnalysis(
            microstate=microstate,
            free_energy=free_energy_by_microstate[
                CompoundMicrostate(
                    compound_id=compound.metadata.compound_id,
                    microstate_id=microstate.microstate_id,
                )
            ],
        )
        for microstate in compound.microstates
    ]

    return [
        CompoundAnalysis(
            metadata=compound.metadata,
            microstates=microstates,
            free_energy=get_compound_free_energy(microstates),
        )
        for compound in compounds
    ]
