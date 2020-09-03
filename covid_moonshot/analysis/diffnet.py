from arsenic import stats
import networkx as nx
import numpy as np
import logging

_logger = logging.getLogger("DiffNet")


def _ic50_to_dG(ic50, s_conc=375E-9, Km=40E-6):
    """Converts IC50 (in M units) to DG

    Parameters
    ----------
    ic50 : float
        IC50 in M
    s_conc : float, default=375E-9
        Substrate concentration in M
    Km : float, default=40E-6
        Substrate concentration for half-maximal enzyme activity

    Returns
    -------
    type
        Description of returned object.

    """
    if ic50 > 1E-5:
        _logger.warning('Expecting IC50 in M units. Please check.')
    Ki = ic50 / (1 + (s_conc/Km))
    return np.log(Ki)


def separate_graphs(graphs):
    import copy
    graphs_to_replace = []
    new_graphs = []
    # now need to check that all of the graphs are weakly connected
    # and if not, split them up
    for name, g in graphs.items():
        if not nx.is_weakly_connected(g):
            graphs_to_replace.append(name)
            for j, subgraph in enumerate(nx.weakly_connected_components(g)):
                # this should perseve all necessary nodes/edges
                new_g = copy.deepcopy(g)
                new_g.remove_nodes_from([i for i in new_g.nodes()
                                         if i not in subgraph])
                # could name the new graphs something better
                new_graphs[name+f'_{j}'] = new_g
    # now get rid of non-connected graphs
    for name in graphs_to_replace:
        del graphs[name]

    # and add the new graphs
    graphs.update(new_graphs)

    return graphs


def combine_relative_free_energies(results):
    graphs = {}
    for result in results:
        protein = result['details']['protein'].split('/')[-1]
        if protein not in graphs:
            graphs[protein] = nx.DiGraph()
        graph = graphs[protein]
        # check for empty results
        if result['analysis']['binding']['delta_f'] is not None and \
           result['analysis']['binding']['ddelta_f'] is not None:
            graph.add_edge(result['details']['start_title'],
                           result['details']['end_title'],
                           f_ij=result['analysis']['binding']['delta_f'],
                           f_dij=result['analysis']['binding']['ddelta_f'])

        # see if either ligand has an experimental affinity associated node
        for state in ['start', 'end']:
            if f'{state}_pIC50' in result['details'].keys():
                for graph in graphs.values():
                    pIC50 = float(result['details'][f'{state}_pIC50'])
                    title = result['details'][f'{state}_title']
                    if graph.has_node(result['details'][f'{state}_title']):
                        # see if it already has an pIC50
                        if 'pIC50' in graph.nodes[title]:
                            assert(pIC50 == graph.nodes[title]['pIC50']), \
                                   f"Trying to add an pIC50, {float(result['details'][f'{state}_pIC50'])}, \
                                   that disagrees with a previous value for \
                                   node {graph.nodes[title]}"
                        else:
                            # or add the pIC50 if it's not available
                            graph.nodes[title]['pIC50'] = pIC50
                            ic50 = 10**(-pIC50)
                            dg = _ic50_to_dG(ic50)
                            graph.nodes[title]['exp_DG'] = dg

    # TODO this can be wrapped in a function
    _logger.info(f'There are {len(graphs)} unique protein files used')
    graphs = separate_graphs(graphs)
    _logger.info(f'There are {len(graphs)} graphs \
    after splitting into weakly connected sets')

    # now combine the relative simulations to get a per-ligand estimate
    for name, graph in graphs.items():
        f_i, C = stats.mle(graph)
        f_i = - f_i
        errs = np.diag(C)
        mean_calc = np.mean(f_i)
        exp_values = [n[1]['exp_DG'] for n in graph.nodes(data=True)
                      if 'exp_DG' in n[1]]
        if len(exp_values) == 0:
            _logger.warning(f'No experimental value in graph {name}. \
            Results are only useful for relative comparisions \
            within this set, NOT with other sets and should not \
            be considered as absolute predictions')
            mean_expt = 0.
            for dg, dg_err, node in zip(f_i, errs, graph.nodes(data=True)):
                node[1]['est_f_i'] = dg + mean_calc
                node[1]['df_i'] = dg_err
        else:
            mean_expt = np.mean(exp_values)
            for dg, dg_err, node in zip(f_i, errs, graph.nodes(data=True)):
                # shift the calc RBFE to the known experimental values
                node[1]['f_i'] = dg + mean_calc - mean_expt
                node[1]['df_i'] = dg_err

    # not sure how to best return from this script
    return graphs
