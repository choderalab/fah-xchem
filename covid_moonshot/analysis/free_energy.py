from collections import defaultdict
import functools
from typing import List, Optional
import numpy as np
from pymbar import BAR
from pymbar.mbar import MBAR
from covid_moonshot.core import FreeEnergy, GenAnalysis, PhaseAnalysis, Work


def mask_outliers(a: np.ndarray, max_value: float, n_devs: float) -> np.ndarray:
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
    works: np.ndarray, max_work_value: float = 1e4, max_n_devs: float = 5,
) -> np.ndarray:
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


def get_bar_overlap(works: np.ndarray) -> float:
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


def get_free_energy(
    works: List[Work],
    min_num_work_values: Optional[int] = 10,
    work_precision_decimals: Optional[int] = 3,
) -> FreeEnergy:

    ws_all = np.array(
        [(w.forward_work, w.reverse_work) for w in works],
        dtype=[("forward", float), ("reverse", float)],
    )

    ws = filter_work_values(ws_all)

    if min_num_work_values is not None and len(ws) < min_num_work_values:
        raise ValueError(
            f"Need at least {min_num_work_values} good work values for analysis, "
            f"but got {len(ws)}"
        )

    delta_f, ddelta_f = BAR(ws["forward"], ws["reverse"])
    bar_overlap = get_bar_overlap(ws)

    def maybe_round(works: np.ndarray) -> np.ndarray:
        return (
            works
            if work_precision_decimals is None
            else works.round(work_precision_decimals)
        )

    return FreeEnergy(
        delta_f=delta_f,
        ddelta_f=ddelta_f,
        bar_overlap=bar_overlap,
        num_work_values=len(ws),
    )


def get_phase_analysis(
    works: List[Work],
    min_num_work_values: Optional[int] = 10,
    work_precision_decimals: Optional[int] = 3,
) -> PhaseAnalysis:
    """
    Parameters
    ----------
    works : list of Work
        Work values for all clones and gens in a run/phase
    min_num_work_values : int or None, optional
        Minimum number of valid work values required for
        analysis. Raises ValueError if not satisfied.
    work_precision_decimals : int or None, optional
        If given, round returned `forward_works` and `reverse_works`
        to this number of decimal places

    Returns
    -------
    PhaseAnalysis
        Object containing analysis results for a run/phase

    Raises
    ------
    ValueError
        If `min_num_work_values` is given and the number of valid work
        values after filtering by `filter_work_values` is less than
        `min_num_work_values`
    """

    # gen-level results
    works_by_gen = defaultdict(list)
    for work in works:
        works_by_gen[work.path.gen].append(work)

    gens = [
        GenAnalysis(
            free_energy=get_free_energy(
                gen_works,
                min_num_work_values=min_num_work_values,
                work_precision_decimals=work_precision_decimals,
            ),
            forward_works=[w.forward_work for w in gen_works],
            reverse_works=[w.reverse_work for w in gen_works],
        )
        for _, gen_works in sorted(works_by_gen.items())
    ]

    return PhaseAnalysis(
        free_energy=get_free_energy(
            works,
            min_num_work_values=min_num_work_values,
            work_precision_decimals=work_precision_decimals,
        ),
        gens=gens,
    )
