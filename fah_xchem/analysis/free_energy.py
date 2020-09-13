import functools
from typing import List, Optional

import numpy as np

from ..schema import (
    RelativeFreeEnergy,
    PointEstimate,
    WorkPair,
)
from .exceptions import InvalidResultError, InsufficientDataError


def _mask_outliers(
    a: np.ndarray, max_value: float, max_n_devs: float, min_sample_size: int
) -> np.ndarray:
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
    min_sample_size : int
        Only apply the `max_n_devs` criterion when sample size is
        larger than this

    Returns
    -------
    out : ndarray of bool
        Boolean array of same shape as the input array `a`, with
        `False` elements marking outliers in `a` and all other
        elements `True`.
        ``out.shape == a.shape``
    """
    mask = np.abs(a) < max_value
    if len(a) >= min_sample_size:
        mask &= np.abs(a - np.mean(a)) < max_n_devs * np.std(a)
    return mask


def _filter_work_values(
    works: np.ndarray,
    max_value: float = 1e4,
    max_n_devs: float = 5,
    min_sample_size: int = 10,
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
    min_sample_size : int
        Only apply the `max_n_devs` criterion when sample size is
        larger than this

    Returns
    -------
    out : ndarray
        1-D array of filtered works.
        ``out.shape == (works.size, 1)``
    """

    mask_work_outliers = functools.partial(
        _mask_outliers,
        max_value=max_value,
        max_n_devs=max_n_devs,
        min_sample_size=min_sample_size,
    )

    f_good = mask_work_outliers(works["forward"])
    r_good = mask_work_outliers(works["reverse"])

    both_good = f_good & r_good

    return works[both_good]


def _get_bar_free_energy(works: np.ndarray) -> PointEstimate:
    """
    Compute the BAR free energy

    Parameters
    ----------
    works : (N,) ndarray
        1-D array of records containing fields "forward" and "reverse"

    Returns
    -------
    PointEstimate
        BAR free energy point estimate and standard error
    """
    from pymbar import BAR

    delta_f, ddelta_f = BAR(works["forward"], works["reverse"])
    return PointEstimate(point=delta_f, stderr=ddelta_f)


def _get_bar_overlap(works: np.ndarray) -> float:
    """
    Compute the overlap (should be in [0, 1] where close to 1 is good, close to 0 bad)

    Parameters
    ----------
    works : (N,) ndarray
        1-D array of records containing fields "forward" and "reverse"

    Returns
    -------
    float
        overlap
    """

    from pymbar.mbar import MBAR

    n = len(works)
    u_kn = np.block([[works["forward"], np.zeros(n)], [np.zeros(n), works["reverse"]]])
    N_k = np.array([n, n])

    mbar = MBAR(u_kn, N_k)

    return float(mbar.computeOverlap()["scalar"])


def compute_relative_free_energy(
    work_pairs: List[WorkPair], min_num_work_values: Optional[int] = None
):
    """
    Parameters
    ----------
    works : list of SimulatedWork
        Simulated forward and reverse work values
    min_num_work_values : int or None, optional
        Minimum number of valid work value pairs required for
        analysis. Raises InsufficientDataError if not satisfied.

    Returns
    -------
    FreeEnergy
        Results of free energy compuation

    Raises
    ------
    InsufficientDataError
        If `min_num_work_values` is given and the number of work
        values is less than this.
    """

    all_works = np.array(
        [(work.forward, work.reverse) for work in work_pairs],
        dtype=[("forward", float), ("reverse", float)],
    )

    works = _filter_work_values(all_works)

    if min_num_work_values is not None and len(works) < min_num_work_values:
        raise InsufficientDataError(
            f"Need at least {min_num_work_values} good work values for analysis, "
            f"but got {len(works)}"
        )

    delta_f = _get_bar_free_energy(works)

    if not np.isfinite(delta_f.point) or not np.isfinite(delta_f.stderr):
        raise InvalidResultError(
            f"BAR free energy computation returned "
            f"{delta_f.point} ± {delta_f.stderr}"
        )

    bar_overlap = _get_bar_overlap(works)

    if not np.isfinite(bar_overlap):
        raise InvalidResultError(f"BAR overlap computation returned {bar_overlap}")

    return RelativeFreeEnergy(
        delta_f=delta_f, bar_overlap=bar_overlap, num_work_pairs=len(works),
    )
