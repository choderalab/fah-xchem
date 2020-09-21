import logging
from math import log
from typing import List, Optional

import networkx as nx
import numpy as np
from scipy.special import logsumexp

from ..schema import (
    Compound,
    CompoundAnalysis,
    CompoundMicrostate,
    Microstate,
    MicrostateAnalysis,
    PointEstimate,
    TransformationAnalysis,
)
from .exceptions import AnalysisError, ConfigurationError, InsufficientDataError


def pIC50_to_DG(pIC50: float, s_conc: float = 375e-9, Km: float = 40e-6) -> float:
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
    PointEstimate
        Dimensionless free energy (in kT)
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

    Returns
    -------
    PointEstimate
        Dimensionless compound free energy estimate (in kT)
    """
    penalized_free_energies = [
        microstate.free_energy + microstate.microstate.free_energy_penalty
        for microstate in microstates
        if microstate.free_energy is not None
    ]

    if not penalized_free_energies:
        raise InsufficientDataError("no microstate free energy estimates")

    g = np.array([x.point for x in penalized_free_energies])
    stderr = np.array([x.stderr for x in penalized_free_energies])

    x = np.exp(-g)
    z = np.sum(x)
    gc = logsumexp(x)
    dgc = -x / z

    return PointEstimate(point=gc, stderr=np.sqrt(np.sum((dgc * stderr) ** 2)))


def _validate_inputs(
    compounds: List[Compound], transformations: List[TransformationAnalysis]
):
    # Microstates appearing as initial or final in transformations
    transformation_microstates = set(
        microstate
        for transformation in transformations
        for microstate in [
            transformation.transformation.initial_microstate,
            transformation.transformation.final_microstate,
        ]
    )

    # Microstates appearing in compounds
    compound_microstates = set(
        CompoundMicrostate(
            compound_id=compound.metadata.compound_id,
            microstate_id=microstate.microstate_id,
        )
        for compound in compounds
        for microstate in compound.microstates
    )

    for microstate in compound_microstates - transformation_microstates:
        logging.warning("No transformation data for microstate '%s'", microstate)

    missing_microstates = transformation_microstates - compound_microstates
    if missing_microstates:
        raise ConfigurationError(
            f"The following undefined microstates are referenced in "
            f"transformations: {missing_microstates}"
        )


def build_transformation_graph(
    compounds: List[Compound], transformations: List[TransformationAnalysis]
) -> nx.DiGraph:

    _validate_inputs(compounds, transformations)
    graph = nx.DiGraph()

    for analysis in transformations:
        transformation = analysis.transformation

        if analysis.binding_free_energy is None:
            continue

        graph.add_edge(
            transformation.initial_microstate,
            transformation.final_microstate,
            g_ij=analysis.binding_free_energy.point,
            g_dij=analysis.binding_free_energy.stderr,
        )

    for compound in compounds:
        for microstate in compound.microstates:
            node = CompoundMicrostate(
                compound_id=compound.metadata.compound_id,
                microstate_id=microstate.microstate_id,
            )
            if node in graph:
                graph.nodes[node]["compound"] = compound
                graph.nodes[node]["microstate"] = microstate

    return graph


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

    # Type assertions (useful for type checking with mypy)
    node: CompoundMicrostate
    microstate: Microstate

    # TODO: can we bypass graph construction and just derive the adjacency matrix?
    supergraph = build_transformation_graph(compounds, transformations)

    # Split supergraph into weakly-connected subgraphs
    # NOTE: the subgraphs are "views" into the supergraph, meaning
    # updates made to the subgraphs are reflected in the supergraph
    # (we exploit this below)
    connected_subgraphs = [
        supergraph.subgraph(nodes)
        for nodes in nx.weakly_connected_components(supergraph)
    ]

    # Filter to connected subgraphs containing at least one
    # experimental measurement
    valid_subgraphs = [
        graph
        for graph in connected_subgraphs
        if any(
            "pIC50" in graph.nodes[node]["compound"].metadata.experimental_data
            for node in graph
        )
    ]

    if len(valid_subgraphs) < len(connected_subgraphs):
        logging.warning(
            "Found %d out of %d connected subgraphs without experimental data",
            len(connected_subgraphs) - len(valid_subgraphs),
            len(connected_subgraphs),
        )

    # Inital MLE pass: compute relative free energies without using
    # experimental reference values
    for graph in valid_subgraphs:
        # NOTE: no node_factor argument in the following
        # (because we do not use experimental data for the first pass)
        g1s, _ = stats.mle(graph, factor="g_ij")
        for node, g1 in zip(graph.nodes, g1s):
            graph.nodes[node]["g1"] = g1

    # Use microstate-level relative free energies g_1[c,i] to
    # distribute compound-level experimental data g_exp_compound[c]
    # over microstates, using the formula:
    #
    #    g_exp[c,i] = g_exp_compound[c]
    #               - ln( exp(-(s[c,i] + g_1[c,i]))
    #                   / sum(exp(-(s[c,:] + g_1[c,:])))
    #                   )
    #
    for compound in compounds:
        pIC50 = compound.metadata.experimental_data.get("pIC50")

        # Skip compounds with no experimental data
        if pIC50 is None:
            continue

        g_exp_compound = pIC50_to_DG(pIC50)

        nodes = [
            CompoundMicrostate(
                compound_id=compound.metadata.compound_id,
                microstate_id=microstate.microstate_id,
            )
            for microstate in compound.microstates
        ]

        # Filter to nodes that are part of a connected subgraph with
        # at least one experimental measurement
        # TODO: check for case where microstates are in different subgraphs
        valid_nodes = [
            (node, microstate)
            for node, microstate in zip(nodes, compound.microstates)
            if node in supergraph and "g1" in supergraph.nodes[node]
        ]

        # Skip compound if none of its microstates are in a subgraph
        # with experimental data
        if not valid_nodes:
            continue

        # gs = s[c,i] + g_1[c,i]
        gs = np.array(
            [
                microstate.free_energy_penalty.point + supergraph.nodes[node]["g1"]
                for node, microstate in valid_nodes
            ]
        )

        dgs = gs - logsumexp(-gs)  # TODO: check math
        assert (dgs >= 0).all()

        for (node, _), dg in zip(valid_nodes, dgs):
            if node in supergraph:
                supergraph.nodes[node]["g_exp"] = g_exp_compound - dg
            else:
                logging.warning(
                    "Compound microstate '%s' has experimental data, "
                    "but does not appear in any transformation",
                    node.microstate_id,
                )

    # Second pass: use microstate relative free energies and compound
    # experimental data to compute microstate absolute free energies.
    # Process each subgraph, collecting microstate free energy results
    microstate_free_energy = {}
    for g in valid_subgraphs:
        gs, C = stats.mle(g, factor="g_ij", node_factor="g_exp")
        errs = np.sqrt(np.diag(C))
        microstate_free_energy.update(
            {
                microstate: PointEstimate(point=point, stderr=stderr)
                for microstate, point, stderr in zip(g.nodes(), gs, errs)
            }
        )

    def get_compound_analysis(compound: Compound) -> Optional[CompoundAnalysis]:

        microstates = [
            MicrostateAnalysis(
                microstate=microstate,
                free_energy=microstate_free_energy.get(
                    CompoundMicrostate(
                        compound_id=compound.metadata.compound_id,
                        microstate_id=microstate.microstate_id,
                    )
                ),
            )
            for microstate in compound.microstates
        ]

        try:
            return CompoundAnalysis(
                metadata=compound.metadata,
                microstates=microstates,
                free_energy=get_compound_free_energy(microstates),
            )
        except AnalysisError as exc:
            logging.warning(
                "Failed to estimate free energy for compound '%s': %s",
                compound.metadata.compound_id,
                exc,
            )
            return None

    results = [get_compound_analysis(compound) for compound in compounds]
    return [r for r in results if r is not None]
