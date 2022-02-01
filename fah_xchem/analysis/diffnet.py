from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.special import logsumexp
from .constants import KT_KCALMOL, KCALMOL_KT

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


def experimental_data_to_point_estimate(experimental_data: dict) -> PointEstimate:
    """
    Convert experimental data dict to a point estimate of absolute binding free energy

    Parameters
    ----------
    experimental_data : dict
        Experimental data record which should contain 'g_exp' and 'g_dexp'

    Returns
    -------
    point : PointEstimate
        The experimental free energy point estimate (in kT)

    """
    return PointEstimate(
        point=experimental_data["g_exp"], stderr=experimental_data["g_dexp"]
    )


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

    microstate_free_energies = [
        (ms.free_energy, ms.microstate.free_energy_penalty)
        for ms in microstates
        if ms.free_energy is not None
    ]

    if not microstate_free_energies:
        raise InsufficientDataError("no microstate free energy estimates")

    g = np.array([p[0].point for p in microstate_free_energies])
    s = np.array([p[1].point for p in microstate_free_energies])

    g_err = np.array([p[0].stderr for p in microstate_free_energies])
    s_err = np.array([p[1].stderr for p in microstate_free_energies])

    gs = g - (-s - logsumexp(-s))

    # TODO: check the error propagation below. It was written in a hurry!
    # Error propagation for gs
    Kas = np.exp(-s)
    Zs = np.sum(Kas)
    ds = 1 - Kas / Zs
    gs_err = np.sqrt(g_err**2 + (ds * s_err) ** 2)

    # Error propagation for g
    g = -logsumexp(-gs)
    Ka = np.exp(-gs)
    Z = np.sum(Ka)
    dgs = Ka / Z
    g_err = np.sqrt(np.sum((dgs * gs_err) ** 2))

    return PointEstimate(point=g, stderr=g_err)


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
        logging.info("No transformation data for microstate '%s'", microstate)

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

        # Omit null transformations, which cause problems for diffnet
        if transformation.initial_microstate == transformation.final_microstate:
            continue

        graph.add_edge(
            transformation.initial_microstate,
            transformation.final_microstate,
            g_ij=analysis.binding_free_energy.point,
            # NOTE: name `g_dij` is derived by Arsenic from `g_ij`
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

    from openff.arsenic import stats

    # Type assertions (useful for type checking with mypy)
    node: CompoundMicrostate
    microstate: Microstate

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
            "g_exp" in graph.nodes[node]["compound"].metadata.experimental_data
            for node in graph
        )
    ]

    if len(valid_subgraphs) < len(connected_subgraphs):
        logging.warning(
            "Found %d out of %d connected subgraphs without experimental data",
            len(connected_subgraphs) - len(valid_subgraphs),
            len(connected_subgraphs),
        )

    if len(valid_subgraphs) > 0:
        experimental_data_available = True
    else:
        # No experimental data is available, so analyze the largest connected subgraph only
        # since we don't know how to connect different subgraphs with absolute free energies
        experimental_data_available = False
        logging.warning(
            "* * * No connected subgraphs with experimental data found; selecting largest connected subgraph to analyze"
        )
        largest_connected_subgraph_index = np.argmax(
            [graph.number_of_nodes() for graph in connected_subgraphs]
        )
        largest_connected_subgraph = connected_subgraphs[
            largest_connected_subgraph_index
        ]
        valid_subgraphs = [largest_connected_subgraph]
        # Select a compound to set experimental g_exp arbitrarily to zero
        compound = [
            compound
            for (node, compound) in largest_connected_subgraph.nodes(data="compound")
        ][0]
        logging.warning("* * * Setting g_exp = 0 for one compound arbitrarily")
        compound.metadata.experimental_data["g_exp"] = 0.0
        compound.metadata.experimental_data["g_dexp"] = 0.1  # arbitrary

    # Inital MLE pass: compute microstate free energies without using
    # experimental reference values
    for idx, graph in enumerate(valid_subgraphs):
        # NOTE: no node_factor argument in the following
        # (because we do not use experimental data for the first pass)
        g1s, C1 = stats.mle(graph, factor="g_ij")
        errs = np.sqrt(np.diag(C1))
        for node, g1, g1_err in zip(graph.nodes, g1s, errs):
            graph.nodes[node]["g1"] = g1
            graph.nodes[node]["g1_err"] = g1_err
            graph.nodes[node]["subgraph_index"] = idx

    # Use first-pass microstate free energies g_1[c,i] to distribute
    # compound-level experimental data g_exp_compound[c] over
    # microstates, using the formula:
    #
    #    g_exp[c,i] = g_exp_compound[c]
    #               - ln( exp(-(s[c,i] + g_1[c,i]))
    #                   / sum(exp(-(s[c,:] + g_1[c,:])))
    #                   )
    #
    for compound in compounds:
        # Skip compounds with no experimental data
        if "g_exp" not in compound.metadata.experimental_data:
            continue

        # DEBUG: Use only one reference compound
        if compound.metadata.compound_id != "VLA-UCB-50c39ae8-2":
            continue

        # Retrieve experimental dimensionless free energy and uncertainty
        g_exp_compound = compound.metadata.experimental_data["g_exp"]
        g_dexp_compound = compound.metadata.experimental_data["g_dexp"]

        nodes = [
            CompoundMicrostate(
                compound_id=compound.metadata.compound_id,
                microstate_id=microstate.microstate_id,
            )
            for microstate in compound.microstates
        ]

        # Filter to nodes that are part of a connected subgraph with
        # at least one experimental measurement

        subgraph_valid_nodes: Dict[int, List[Tuple[CompoundMicrostate, Microstate]]]
        subgraph_valid_nodes = defaultdict(list)
        for node, microstate in zip(nodes, compound.microstates):
            if node in supergraph and "subgraph_index" in supergraph.nodes[node]:
                idx = supergraph.nodes[node]["subgraph_index"]
                subgraph_valid_nodes[idx].append((node, microstate))

        # Skip compound if none of its microstates are in a subgraph
        # with experimental data
        if not subgraph_valid_nodes:
            continue

        # Pick the subgraph containing the largest number of microstates
        valid_nodes = max(subgraph_valid_nodes.values(), key=lambda ns: len(ns))

        g_is = np.array(
            [
                microstate.free_energy_penalty.point + supergraph.nodes[node]["g1"]
                for node, microstate in valid_nodes
            ]
        )

        # TODO: Fix this later when we have a better way of propagating uncertainty in allocation
        # For now, we skip anything that has multiple microstates
        if len(g_is) > 1:
            logging.info(
                f"Skipping use of experimental data for {compound.metadata.compound_id} because there are {len(g_is)} microstates"
            )
            continue

        # Compute normalized microstate probabilities
        p_is = np.exp(-g_is - logsumexp(-g_is))

        # Apportion compound K_a according to microstate probability
        Ka_is = p_is * np.exp(-g_exp_compound)

        logging.info(f"Computing allocations for {compound.metadata.compound_id}")
        for (node, _), Ka, p_i in zip(valid_nodes, Ka_is, p_is):
            if node in supergraph:
                supergraph.nodes[node]["g_exp"] = -np.log(Ka)
                # NOTE: naming of uncertainty fixed by Arsenic convention
                # TODO: Determine better experimental error scheme here
                supergraph.nodes[node]["g_dexp"] = g_dexp_compound
                logging.info(
                    f" Allocating {p_i} of experimental free energy {g_exp_compound} to microstate {node}: {-np.log(Ka)}"
                )
            else:
                logging.warning(
                    "Compound microstate '%s' has experimental data, "
                    "but does not appear in any transformation",
                    node.microstate_id,
                )

    # Second pass: use first-pass microstate free energies and
    # compound experimental data to compute microstate absolute free
    # energies.

    for graph in valid_subgraphs:
        gs, C = stats.mle(graph, factor="g_ij", node_factor="g_exp")
        print(gs)
        print(gs.min(), gs.max())

        errs = np.sqrt(np.diag(C))
        for node, g, g_err in zip(graph.nodes, gs, errs):
            graph.nodes[node]["g"] = g
            graph.nodes[node]["g_err"] = g_err
            if "g_exp" in graph.nodes[node]:
                print(
                    f"{node} : exp {graph.nodes[node]['g_exp']} +- {graph.nodes[node]['g_dexp']} : calc {g} +- {g_err}"
                )

    def get_compound_analysis(compound: Compound) -> CompoundAnalysis:
        def get_microstate_analysis(microstate: Microstate) -> MicrostateAnalysis:

            node = CompoundMicrostate(
                compound_id=compound.metadata.compound_id,
                microstate_id=microstate.microstate_id,
            )

            data = supergraph.nodes.get(node)

            return MicrostateAnalysis(
                microstate=microstate,
                free_energy=PointEstimate(point=data["g"], stderr=data["g_err"])
                if data and "g" in data and "g_err" in data
                else None,
                first_pass_free_energy=PointEstimate(
                    point=data["g1"], stderr=data["g1_err"]
                )
                if data and "g1" in data and "g1_err" in data
                else None,
            )

        microstates = [
            get_microstate_analysis(microstate) for microstate in compound.microstates
        ]

        free_energy: Optional[PointEstimate]
        try:
            free_energy = get_compound_free_energy(microstates)
        except AnalysisError as exc:
            logging.info(
                "Failed to estimate free energy for compound '%s': %s",
                compound.metadata.compound_id,
                exc,
            )
            free_energy = None

        experimental_free_energy = None
        if "g_exp" in compound.metadata.experimental_data:
            experimental_free_energy = PointEstimate(
                point=compound.metadata.experimental_data["g_exp"],
                stderr=compound.metadata.experimental_data["g_dexp"],
            )

        absolute_free_energy_error = None
        if free_energy and experimental_free_energy:
            absolute_free_energy_error = abs(free_energy - experimental_free_energy)

        return CompoundAnalysis(
            metadata=compound.metadata,
            microstates=microstates,
            free_energy=free_energy,
            experimental_free_energy=experimental_free_energy,
            absolute_free_energy_error=absolute_free_energy_error,
        )

    return [get_compound_analysis(compound) for compound in compounds]
