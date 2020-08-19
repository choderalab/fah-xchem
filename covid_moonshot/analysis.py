import functools
from typing import List
import numpy as np
from pymbar import BAR
from pymbar.mbar import MBAR
from .core import Binding, PhaseAnalysis, RunAnalysis, Work


def mask_outliers(a: np.array, max_value: float, n_devs: float):
    """Returns a boolean array masking values that are more than
    `n_devs` standard deviations from the mean or larger in magnitude
    than `max_value`.

    Parameters
    ----------
    a : array
    max_value : float
        Values larger (in magnitude) than this will be discarded
    n_devs : float
        Values farther than this number of standard deviations from
        the mean will be discarded

    Returns
    -------
    array of bool
        Boolean array of same shape as `a`, with False elements
        marking outliers in `a` and all other elements True.
    """
    return (np.abs(a) < max_value) & (np.abs(a - np.mean(a)) < n_devs * np.std(a))


def filter_work_values(
    works: np.array, max_work_value: float = 1e4, max_n_devs: float = 5,
) -> np.array:
    """Remove pairs of works when either is determined to be an outlier.

    Parameters
    ----------
    works: array
        1-d structured array with names "forward" and "reverse"

    Returns
    -------
    array
        filtered works
    """

    mask_work_outliers = functools.partial(
        mask_outliers, max_value=max_work_value, n_devs=max_n_devs
    )

    f_good = mask_work_outliers(works["forward"])
    r_good = mask_work_outliers(works["reverse"])

    both_good = f_good & r_good

    return works[both_good]


def get_bar_overlap(works: np.array) -> float:
    """
    Compute the overlap (should be in [0, 1] where close to 1 is good, close to 0 bad).

    Parameters
    ----------
    works: array
        1-d structured array with names "forward" and "reverse"

    Returns
    -------
    float
        overlap
    """

    n = len(works)
    u_kn = np.block([[works["forward"], np.zeros(n)], [np.zeros(n), works["reverse"]]])
    N_k = np.array([n, n])
    mbar = MBAR(u_kn, N_k)
    return mbar.computeOverlap()["scalar"]


def get_phase_analysis(works: List[Work], min_num_work_values=10):

    ws_all = np.array(
        [(w.forward_work, w.reverse_work) for w in works],
        dtype=[("forward", float), ("reverse", float)],
    )

    ws = filter_work_values(ws_all)

    if len(ws) < min_num_work_values:
        raise ValueError(
            f"Need at least {min_num_work_values} good work values for analysis, "
            f"but got {len(ws)}"
        )

    delta_f, ddelta_f = BAR(ws["forward"], ws["reverse"])
    bar_overlap = get_bar_overlap(ws)

    return PhaseAnalysis(
        delta_f=delta_f,
        ddelta_f=ddelta_f,
        bar_overlap=bar_overlap,
        num_work_values=len(ws),
    )


def get_run_analysis(
    complex_works: List[Work], solvent_works: List[Work],
) -> RunAnalysis:

    try:
        complex_phase = get_phase_analysis(complex_works)
    except ValueError as e:
        raise ValueError(f"Failed to analyze complex: {e}")

    try:
        solvent_phase = get_phase_analysis(solvent_works)
    except ValueError as e:
        raise ValueError(f"Failed to analyze solvent: {e}")

    binding = Binding(
        delta_f=solvent_phase.delta_f - complex_phase.delta_f,
        ddelta_f=(complex_phase.ddelta_f ** 2 + solvent_phase.ddelta_f ** 2) ** 0.5,
    )

    return RunAnalysis(
        complex_phase=complex_phase, solvent_phase=solvent_phase, binding=binding
    )
