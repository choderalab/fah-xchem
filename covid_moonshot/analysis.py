from typing import List, Tuple
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
    f_works: np.array,
    r_works: np.array,
    max_work_value: float = 1e4,
    max_n_devs: float = 5,
) -> Tuple[np.array, np.array, int]:
    """Remove pairs of works when either is determined to be an outlier.

    Parameters
    ----------
    f_works: array
        forward works
    r_works: array
        reverse works

    Returns
    -------
    tuple
        tuple of forward works, reverse_works, and number of remaining work values
    """
    f_good = mask_outliers(f_works, max_value=max_work_value, n_devs=max_n_devs)
    r_good = mask_outliers(r_works, max_value=max_work_value, n_devs=max_n_devs)
    both_good = f_good & r_good
    num_work_values = both_good.astype(int).sum().item()
    return f_works[both_good], r_works[both_good], num_work_values


def get_bar_overlap(f_works: np.array, r_works: np.array) -> float:
    """
    Compute the overlap (should be in [0, 1] where close to 1 is good, close to 0 bad).

    Parameters
    ----------
    f_works: array
        forward works
    r_works: array
        reverse works

    Returns
    -------
    float
        overlap
    """

    u_kn = np.block(
        [f_works, np.zeros_like(r_works)], [np.zeros_like(f_works), r_works]
    )

    N_k = np.array([len(f_works), len(r_works)])

    mbar = MBAR(u_kn, N_k)

    return mbar.computeOverlap()["scalar"]


def get_phase_analysis(works: List[Work], min_num_work_values=10):

    f_works_all = np.array([w.forward_work for w in works])
    r_works_all = np.array([w.reverse_work for w in works])

    f_works, r_works, num_work_values = filter_work_values(f_works_all, r_works_all)

    if num_work_values < min_num_work_values:
        raise ValueError(
            f"Need at least {min_num_work_values} good work values for analysis, "
            f"but got {num_work_values}"
        )

    delta_f, ddelta_f = BAR(f_works, r_works)
    bar_overlap = get_bar_overlap(f_works, r_works)

    return PhaseAnalysis(
        delta_f=delta_f,
        ddelta_f=ddelta_f,
        bar_overlap=bar_overlap,
        num_work_values=num_work_values,
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
