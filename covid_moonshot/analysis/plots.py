from contextlib import contextmanager
from functools import partial
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from simtk.openmm import unit
from openmmtools.constants import kB
from typing import List, Optional, Tuple
from ..core import Binding, PhaseAnalysis, Work, Run

_kT_kcal = kB * 300 * unit.kelvin / unit.kilocalories_per_mole


def plot_work_distribution(
    ax: plt.Axes,
    forward_works: List[float],
    reverse_works: List[float],
    delta_f: float,
) -> None:
    """
    Plot a single work distribution

    Parameters
    ----------
    ax : AxesSubplot
       Axes on which to draw the plot
    forward_works : list of float
       Forward work values (in kT)
    reverse_works : list of float
       Reverse work values (in kT)
    delta_f : float
       Free energy estimate (in kT)
    """

    distplot = partial(
        sns.distplot,
        hist=False,
        kde=True,
        rug=True,
        ax=ax,
        kde_kws=dict(shade=True),
        rug_kws=dict(alpha=0.5),
    )

    distplot(
        forward_works,
        color="cornflowerblue",
        label=f"forward : N={len(forward_works)}",
    )

    distplot(
        -np.array(reverse_works),
        color="hotpink",
        label=f"reverse : N={len(reverse_works)}",
    )

    ax.axvline(delta_f, color="k", ls=":")
    ax.set_xlabel(f"work / $k_B T$")


def plot_work_distributions(
    complex_forward_works: List[float],
    complex_reverse_works: List[float],
    complex_delta_f: float,
    solvent_forward_works: List[float],
    solvent_reverse_works: List[float],
    solvent_delta_f: float,
    figsize: Tuple[float, float] = (7.5, 3.25),
) -> plt.Figure:
    """
    Plot work distributions complex and solvent side by side

    Parameters
    ----------
    complex_forward_works, complex_reverse_works : list of float
       Work values for the complex (in kT)
    solvent_forward_works, solvent_reverse_works : list of float
       Work values for the solvent (in kT)
    complex_delta_f, solvent_delta_f : float
       Free energies computed for the complex and solvent (in kT)

    Returns
    -------
    Figure
        Figure containing the plot
    """

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=figsize)

    plot_work_distribution(
        ax1, complex_forward_works, complex_reverse_works, complex_delta_f
    )
    ax1.set_title("complex")

    plot_work_distribution(
        ax2, solvent_forward_works, solvent_reverse_works, solvent_delta_f
    )
    ax2.set_title("solvent")

    fig.subplots_adjust(top=0.9, wspace=0.15)
    ax1.legend()
    ax2.legend()

    return fig


def _filter_inclusive(
    x: np.ndarray, min_value: Optional[float] = None, max_value: Optional[float] = None
) -> np.ndarray:
    if min_value is not None:
        x = x[x >= min_value]
    if max_value is not None:
        x = x[x <= max_value]
    return x


def plot_relative_distribution(
    relative_delta_fs: List[float], min_delta_f: float = -30, max_delta_f: float = 30
) -> None:
    """
    Plot the distribution of relative free energies

    Parameters
    ----------
    relative_delta_fs : list of float
        Relative free energies (in kT)
    min_delta_f, max_delta_f : float
        Omit values less than `min_bound` or greater than `max_bound`
        (in kT)
    """

    valid_relative_delta_fs = _filter_inclusive(
        np.array(relative_delta_fs), min_delta_f, max_delta_f
    )
    valid_relative_delta_fs_kcal = valid_relative_delta_fs * _kT_kcal

    sns.distplot(
        valid_relative_delta_fs_kcal,
        hist=False,
        kde=True,
        rug=True,
        color="hotpink",
        kde_kws=dict(shade=True),
        rug_kws=dict(alpha=0.5),
        label=f"N={len(relative_delta_fs)}",
    )
    plt.xlabel(r"Relative free energy to ligand 0 / kcal mol$^{-1}$")


def plot_convergence(
    complex_gens: List[int],
    solvent_gens: List[int],
    complex_delta_fs: List[float],
    complex_delta_f_errs: List[float],
    solvent_delta_fs: List[float],
    solvent_delta_f_errs: List[float],
    binding_delta_f: float,
    binding_delta_f_err: float,
    n_devs_bounds: float = 1.65,  # 95th percentile
) -> plt.Figure:
    """
    Plot the convergence of free energy estimates with GEN

    Parameters
    ----------
    complex_gens, solvent_gens : list of int
        List of gens to plot
    complex_delta_fs, complex_delta_f_errs : list of float
        Free energies and errors for the complex; one of each per gen (in kT)
    solvent_delta_fs, solvent_delta_f_errs : list of float
        Free energies and errors for the solvent; one of each per gen (in kT)
    binding_delta_f, binding_delta_f_err : float
        Binding free energy and error, estimated using data for all gens (in kT)
    n_devs_bounds : float
        Number of standard deviations for drawing bounds

    Returns
    -------
    Figure
        Figure containing the plot
    """

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    complex_delta_fs_kcal = pd.Series(
        np.array(complex_delta_fs) * _kT_kcal, index=complex_gens,
    )
    complex_delta_f_errs_kcal = pd.Series(
        np.array(complex_delta_f_errs) * _kT_kcal, index=complex_gens,
    )
    solvent_delta_fs_kcal = pd.Series(
        np.array(solvent_delta_fs) * _kT_kcal, index=solvent_gens,
    )
    solvent_delta_f_errs_kcal = pd.Series(
        np.array(solvent_delta_f_errs) * _kT_kcal, index=solvent_gens,
    )

    DDG_kcal = solvent_delta_fs_kcal - complex_delta_fs_kcal
    DDG_err_kcal = np.sqrt(
        solvent_delta_f_errs_kcal ** 2 + complex_delta_f_errs_kcal ** 2
    )

    ax1.scatter(DDG_kcal.index, DDG_kcal, color="green", label="binding")
    ax1.vlines(
        DDG_kcal.index,
        DDG_kcal - DDG_err_kcal * n_devs_bounds,
        DDG_kcal + DDG_err_kcal * n_devs_bounds,
        color="green",
    )

    for label, (delta_fs_kcal, delta_f_errs_kcal), color in [
        ("solvent", (solvent_delta_fs_kcal, solvent_delta_f_errs_kcal), "blue"),
        ("complex", (complex_delta_fs_kcal, complex_delta_f_errs_kcal), "red"),
    ]:
        ax2.scatter(delta_fs_kcal.index, delta_fs_kcal, color=color, label=label)
        ax2.vlines(
            delta_fs_kcal.index,
            delta_fs_kcal - delta_f_errs_kcal * n_devs_bounds,
            delta_fs_kcal + delta_f_errs_kcal * n_devs_bounds,
            color=color,
        )

    gens = complex_delta_fs_kcal.index.union(solvent_delta_fs_kcal.index).values

    ax1.hlines(
        binding_delta_f * _kT_kcal,
        0,
        gens.max(),
        color="green",
        linestyle=":",
        label="binding (all GENS)",
    )
    ax1.fill_between(
        [0, gens.max()],
        (binding_delta_f - binding_delta_f_err * n_devs_bounds) * _kT_kcal,
        (binding_delta_f + binding_delta_f_err * n_devs_bounds) * _kT_kcal,
        alpha=0.2,
        color="green",
    )

    ax1.set_xticks([gen for gen in range(gens.max() + 1)])
    ax2.set_xlabel("GEN")
    ax1.legend()
    ax2.legend()

    for ax in [ax1, ax2]:
        ax.set_ylabel(r"Rel. $\Delta G$ / kcal mol$^{-1}$")

    return fig


def plot_poor_convergence_fe_table(
    runs: List[Run], energy_cutoff_kcal: float = 1.0,
) -> plt.Figure:
    """
    Plot table of poorly converging free energy estimates with GEN

    Parameters
    ----------
    runs : list of Run
        Relative free energies (in kT)
    energy_cutoff_kcal : float
        Cutoff to consider a result as poorly converged (in kcal/mol)

    Returns
    -------
    Figure
        Figure containing the plot

    """

    complex_phases = [run.analysis.complex_phase for run in runs]
    job_ids = [run.details.JOBID for run in runs]

    std_dev_store = []
    jobid_store = []

    for run in runs:

        try:

            complex_gens = run.analysis.complex_phase.gens
            std_dev = np.std([gen.free_energy.delta_f for gen in complex_gens])

            if std_dev * _kT_kcal >= energy_cutoff_kcal:

                jobid_store.append(run.details.JOBID)
                std_dev_store.append(np.round(std_dev * _kT_kcal, 3))
        
        except AttributeError: # skip if no delta_f recorded
            continue

    # Create 2D list for table input
    data = [[i, j] for i, j in zip(jobid_store, std_dev_store)]

    column_titles = ["JOBID", "Complex phase standard deviation / kcal mol$^{-1}$"]

    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=data, colLabels=column_titles, loc="center")

    return fig


def plot_cumulative_distribution(
    relative_delta_fs: List[float],
    min_delta_f_kcal: Optional[float] = None,
    max_delta_f_kcal: float = 5,
    cmap: str = "PiYG",
    n_bins: int = 100,
    markers_kcal: List[float] = [-2, -1, 0, 1, 2],
) -> None:
    """
    Plot cumulative distribution of ligand affinities

    Parameters
    ----------
    relative_delta_fs : list of float
        Relative free energies (in kT)
    min_delta_f_kcal : float
        Minimum free energy to plot (in kcal/mol)
    max_delta_f_kcal : float
        Maximum free energy to plot (in kcal/mol)
    cmap : str
        String name of colormap to use
    n_bins : int
        Number of bins to use
    markers_kcal : list of float
        Free energy values at which to label (in kcal/mol)

    """

    relative_delta_fs_kcal = np.array(relative_delta_fs) * _kT_kcal

    relative_delta_fs_kcal = _filter_inclusive(
        relative_delta_fs_kcal, min_delta_f_kcal, max_delta_f_kcal
    )

    cm = plt.cm.get_cmap(cmap)

    # Get the histogram
    Y, X = np.histogram(relative_delta_fs_kcal, n_bins)
    Y = np.cumsum(Y)
    x_span = X.max() - X.min()
    C = [cm(((X.max() - x) / x_span)) for x in X]

    plt.bar(X[:-1], Y, color=C, width=X[1] - X[0], edgecolor="k")

    for marker_kcal in markers_kcal:
        n_below = (relative_delta_fs_kcal < marker_kcal).astype(int).sum()
        plt.vlines(-marker_kcal, 0, Y.max(), "grey", linestyles="dashed")
        plt.text(
            marker_kcal - 0.5,
            0.8 * Y.max(),
            rf"$N$ = {n_below}",
            rotation=90,
            verticalalignment="center",
            color="green",
        )
    plt.xlabel(r"Relative free energy to ligand 0 / kcal mol$^{-1}$")
    plt.ylabel("Cumulative $N$ ligands")


@contextmanager
def save_table_pdf(path: str, name: str):
    """
    Context manager that creates a new figure on entry and saves the
    figure using a specified name and path on exit.

    Paramaters
    ----------
    path : str
        Path prefix to use in constructing the result path
    name : str
        Basename to use in constructing the result path

    """

    # Make sure the directory exists
    import os

    os.makedirs(path, exist_ok=True)
    file_name = os.path.join(path, os.extsep.join([name, "pdf"]))

    with PdfPages(file_name) as pdf_plt:
        yield
        pdf_plt.savefig()


@contextmanager
def save_plot(path: str, name: str, file_format: str):
    """
    Context manager that creates a new figure on entry and saves the
    figure using the specified name, format, and path on exit.

    Parameters
    ----------
    path : str
        Path prefix to use in constructing the result path
    name : str
        Basename to use in constructing the result path
    file_format : str
        File extension of the result. Must be accepted by ``plt.savefig``

    Examples
    --------
    >>> with save_plot('example/plots', 'test_plot', 'png'):
    >>>     plt.plot(np.cos(np.linspace(-np.pi, np.pi)))
    >>>     plt.title("My cool plot")
    """
    # Make sure the directory exists
    import os

    os.makedirs(path, exist_ok=True)

    plt.figure()
    yield
    plt.tight_layout()
    plt.savefig(os.path.join(path, os.extsep.join([name, file_format])))


def save_run_level_plots(
    run: int,
    complex_phase: PhaseAnalysis,
    solvent_phase: PhaseAnalysis,
    binding: Binding,
    path: str = os.curdir,
    file_format: str = "pdf",
) -> None:
    """
    Save plots specific to a run.

    The following plots are generated. See the docstrings for the
    individual plotting functions for more information.

    - Work distributions for the solvent and complex.
      Saved to ``{path}/RUN{run}.{file_format}``

    - Convergence of relative free energies by GEN.
      Useful for examining the convergence of the results.
      Saved to ``{path}/RUN{run}-convergence.{file_format}``

    Parameters
    ----------
    run : int
        Run
    complex_phase : PhaseAnalysis
        Results for complex
    solvent_phase : PhaseAnalysis
        Results for solvent
    binding : Binding
        Results for binding free energy
    path : str
        Where to write plot files
    file_format : str
        File format for plot output
    """

    with save_plot(path, f"RUN{run}", file_format):
        fig = plot_work_distributions(
            complex_forward_works=[
                w for gen in complex_phase.gens for w in gen.forward_works
            ],
            complex_reverse_works=[
                w for gen in complex_phase.gens for w in gen.reverse_works
            ],
            complex_delta_f=complex_phase.free_energy.delta_f,
            solvent_forward_works=[
                w for gen in solvent_phase.gens for w in gen.forward_works
            ],
            solvent_reverse_works=[
                w for gen in solvent_phase.gens for w in gen.reverse_works
            ],
            solvent_delta_f=solvent_phase.free_energy.delta_f,
        )
        fig.suptitle(f"RUN{run}")

    with save_plot(path, f"RUN{run}-convergence", file_format):

        # Filter to GENs for which free energy calculation is available
        complex_gens = [
            (gen.gen, gen.free_energy)
            for gen in complex_phase.gens
            if gen.free_energy is not None
        ]
        solvent_gens = [
            (gen.gen, gen.free_energy)
            for gen in solvent_phase.gens
            if gen.free_energy is not None
        ]

        fig = plot_convergence(
            complex_gens=[gen for gen, _ in complex_gens],
            solvent_gens=[gen for gen, _ in solvent_gens],
            complex_delta_fs=[fe.delta_f for _, fe in complex_gens],
            complex_delta_f_errs=[fe.ddelta_f for _, fe in complex_gens],
            solvent_delta_fs=[fe.delta_f for _, fe in solvent_gens],
            solvent_delta_f_errs=[fe.ddelta_f for _, fe in solvent_gens],
            binding_delta_f=binding.delta_f,
            binding_delta_f_err=binding.ddelta_f,
        )
        fig.suptitle(f"RUN{run}")


def save_summary_plots(
    runs: List[Run], path: str = os.curdir, file_format: str = "pdf",
) -> None:
    """
    Save plots summarizing all runs.

    The following plots are generated. See the docstrings for the
    individual plotting functions for more information.

    - Histogram of relative binding free energies (solvent - complex).
      Saved to ``{path}/relative_fe_dist.{file_format}``

    - Cumulative distribution of relative binding free energies.
      This is useful for seeing at a glance how many compounds are
      below a threshold binding free energy.
      Saved to ``{path}/cumulative_fe_dist.{file_format}``

    Parameters
    ----------
    runs : list of Run
        Details and results for all runs
    path : str
        Where to write plot files
    file_format : str
        File format for plot output
    """
    binding_delta_fs = [run.analysis.binding.delta_f for run in runs]

    with save_plot(path, "relative_fe_dist", file_format):
        plot_relative_distribution(binding_delta_fs)
        plt.title("Relative free energy")

    with save_plot(path, "cumulative_fe_dist", file_format):
        plot_cumulative_distribution(binding_delta_fs)
        plt.title("Cumulative distribution")

    with save_table_pdf(path, "poor_complex_convergence_fe_table"):
        plot_poor_convergence_fe_table(runs)
        plt.title("Poor complex convergence table")
