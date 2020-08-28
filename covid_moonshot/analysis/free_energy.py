import functools
from typing import List, Optional, Tuple
import numpy as np
from pymbar import BAR
from pymbar.mbar import MBAR
from covid_moonshot.core import FreeEnergy, GenAnalysis, PhaseAnalysis, Work


def mask_outliers(a: np.ndarray, max_value: float, max_n_devs: float) -> np.ndarray:
    """Returns a boolean array masking values that are more than
    `n_devs` standard deviations from the mean or larger in magnitude
    than `max_value`.

    Parameters
    ----------
    a : array_like
        Input array
    max_value : float
        Remove values with magnitudes greater than this
    max_n_devs : float
        Remove values farther than this number of standard deviations
        from the mean

    Returns
    -------
    out : ndarray of bool
        Boolean array of same shape as the input array `a`, with
        `False` elements marking outliers in `a` and all other
        elements `True`.
        ``out.shape == a.shape``
    """
    return (np.abs(a) < max_value) & (np.abs(a - np.mean(a)) < max_n_devs * np.std(a))


def filter_work_values(
    works: np.ndarray, max_value: float = 1e4, max_n_devs: float = 5,
) -> np.ndarray:
    """Remove pairs of works when either is determined to be an outlier.

    Parameters
    ----------
    works : ndarray
        Array of records containing fields "forward" and "reverse"
    max_value : float
        Remove work values with magnitudes greater than this
    max_n_devs : float
        Remove work values farther than this number of standard
        deviations from the mean

    Returns
    -------
    out : ndarray
        1-D array of filtered works.
        ``out.shape == (works.size, 1)``
    """

    mask_work_outliers = functools.partial(
        mask_outliers, max_value=max_value, max_n_devs=max_n_devs
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
    works : (N,) ndarray
        1-D array of records containing fields "forward" and "reverse"

    Returns
    -------
    float
        overlap
    """

    n = len(works)
    u_kn = np.block([[works["forward"], np.zeros(n)], [np.zeros(n), works["reverse"]]])
    N_k = np.array([n, n])

    mbar = MBAR(u_kn, N_k)

    return float(mbar.computeOverlap()["scalar"])


def get_free_energy(
    works: np.ndarray, min_num_work_values: Optional[int] = 10
) -> FreeEnergy:
    """
    Parameters
    ----------
    works : (N,) ndarray
        1-D array of records containing fields "forward" and "reverse"
        representing forward and reverse works, respectively
    min_num_work_values : int or None, optional
        Minimum number of valid work values required for
        analysis. Raises ValueError if not satisfied.

    Returns
    -------
    FreeEnergy
        Results of free energy compuation

    Raises
    ------
    ValueError
        If `min_num_work_values` is given and the number of work
        values is less than this.
    """

    if min_num_work_values is not None and len(works) < min_num_work_values:
        raise ValueError(
            f"Need at least {min_num_work_values} good work values for analysis, "
            f"but got {len(works)}"
        )

    delta_f, ddelta_f = BAR(works["forward"], works["reverse"])
    bar_overlap = get_bar_overlap(works)

    return FreeEnergy(
        delta_f=delta_f,
        ddelta_f=ddelta_f,
        bar_overlap=bar_overlap,
        num_work_values=len(works),
    )


def get_gen_analysis(
    gen: int,
    works: np.ndarray,
    min_num_work_values: Optional[int],
    work_precision_decimals: Optional[int],
) -> GenAnalysis:
    """
    Parameters
    ----------
    works : list of Work
        Work values for all clones and a run/phase/gen
    min_num_work_values : int or None, optional
        Minimum number of valid work values required for
        analysis. Raises ValueError if not satisfied.
    work_precision_decimals : int or None, optional
        If given, round returned `forward_works` and `reverse_works`
        to this number of decimal places

    Returns
    -------
    GenAnalysis
        Results of free energy computations for a run/phase/gen

    Raises
    ------
    ValueError
        If `min_num_work_values` is given and the number of work
        values is less than this.
    """

    def maybe_round(works: np.ndarray) -> np.ndarray:
        return (
            works
            if work_precision_decimals is None
            else works.round(work_precision_decimals)
        )

    free_energy = get_free_energy(works, min_num_work_values=min_num_work_values)

    return GenAnalysis(
        gen=gen,
        free_energy=free_energy,
        forward_works=maybe_round(works["forward"]).tolist(),
        reverse_works=maybe_round(works["reverse"]).tolist(),
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
        Results of free energy computations for a run/phase

    Raises
    ------
    ValueError
        If `min_num_work_values` is given and the number of valid work
        values after filtering is less than this.
    """

    ws_all = np.array(
        [(w.path.gen, w.forward_work, w.reverse_work) for w in works],
        dtype=[("gen", int), ("forward", float), ("reverse", float)],
    )

    ws = filter_work_values(ws_all)

    gens = [
        get_gen_analysis(
            gen=int(gen),
            works=ws[ws["gen"] == gen],
            min_num_work_values=min_num_work_values,
            work_precision_decimals=work_precision_decimals,
        )
        for gen in sorted(set(ws["gen"]))
    ]

    free_energy = get_free_energy(ws, min_num_work_values=min_num_work_values)

    return PhaseAnalysis(free_energy=free_energy, gens=gens)
