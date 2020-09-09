from contextlib import contextmanager
import datetime as dt
from functools import partial
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from pymbar import BAR
from typing import Generator, Iterable, List, Optional, Tuple
from ..core import Analysis, Binding, PhaseAnalysis, Run, Work
from .constants import KT_KCALMOL
from .free_energy import bootstrap


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

    phases = [
        ("complex", complex_delta_f, complex_forward_works, complex_reverse_works),
        ("solvent", solvent_delta_f, solvent_forward_works, solvent_reverse_works),
    ]

    df = pd.DataFrame.from_records(
        [
            {
                "phase": phase,
                "direction": direction,
                "work_kcal": work * KT_KCALMOL
                if direction == "forward"
                else -work * KT_KCALMOL,
            }
            for phase, _, forward_works, reverse_works in phases
            for direction, works in [
                ("forward", forward_works),
                ("reverse", reverse_works),
            ]
            for work in works
        ]
    )

    g = sns.displot(
        data=df,
        col="phase",
        hue="direction",
        x="work_kcal",
        kind="kde",
        rug=True,
        rug_kws=dict(alpha=0.5),
        fill=True,
        palette=["cornflowerblue", "hotpink"],
        facet_kws=dict(sharex=False),
    ).set_xlabels(r"work / kcal mol$^{-1}$")

    for phase, delta_f, forward_works, reverse_works in phases:
        ax = g.axes_dict[phase]
        ax.axvline(delta_f * KT_KCALMOL, color="k", ls=":")
        ax.text(
            0.05,
            0.95,
            f"$N_F={len(forward_works)}$",
            transform=ax.transAxes,
        )
        ax.text(
            0.05,
            0.88,
            f"$N_R={len(reverse_works)}$",
            transform=ax.transAxes,
        )

    return g.fig


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
    valid_relative_delta_fs_kcal = valid_relative_delta_fs * KT_KCALMOL

    sns.displot(
        valid_relative_delta_fs_kcal,
        kind="kde",
        rug=True,
        color="hotpink",
        fill=True,
        rug_kws=dict(alpha=0.5),
        label=f"$N={len(relative_delta_fs)}$",
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
        np.array(complex_delta_fs) * KT_KCALMOL,
        index=complex_gens,
    )
    complex_delta_f_errs_kcal = pd.Series(
        np.array(complex_delta_f_errs) * KT_KCALMOL,
        index=complex_gens,
    )
    solvent_delta_fs_kcal = pd.Series(
        np.array(solvent_delta_fs) * KT_KCALMOL,
        index=solvent_gens,
    )
    solvent_delta_f_errs_kcal = pd.Series(
        np.array(solvent_delta_f_errs) * KT_KCALMOL,
        index=solvent_gens,
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
        binding_delta_f * KT_KCALMOL,
        0,
        gens.max(),
        color="green",
        linestyle=":",
        label="binding (all GENS)",
    )
    ax1.fill_between(
        [0, gens.max()],
        (binding_delta_f - binding_delta_f_err * n_devs_bounds) * KT_KCALMOL,
        (binding_delta_f + binding_delta_f_err * n_devs_bounds) * KT_KCALMOL,
        alpha=0.2,
        color="green",
    )

    ax1.set_xticks([gen for gen in range(gens.max() + 1)])

    ax2.set_xlabel("GEN")
    ax1.set_ylabel(r"$\Delta\Delta G$ / kcal mol$^{-1}$")
    ax2.set_ylabel(r"$\Delta G$ / kcal mol$^{-1}$")

    ax1.legend()
    ax2.legend()

    return fig


def plot_poor_convergence_fe_table(
    runs: List[Run],
    energy_cutoff_kcal: float = 1.0,
) -> Optional[plt.Figure]:
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
    Figure : Figure or None, optional
        Figure containing the plot

    """

    std_dev_store = []
    jobid_store = []

    for run in runs:

        complex_gens = run.analysis.complex_phase.gens
        std_dev = np.std(
            [
                gen.free_energy.delta_f
                for gen in complex_gens
                if gen.free_energy is not None
            ]
        )

        if std_dev * KT_KCALMOL >= energy_cutoff_kcal:

            jobid_store.append(run.details.JOBID)  # JOBID X should = RUN X
            std_dev_store.append(np.round(std_dev * KT_KCALMOL, 3))

    # Create sorted 2D list for table input, from highest to lowest std_dev
    data = [
        [i, j]
        for i, j in sorted(
            zip(jobid_store, std_dev_store), key=lambda pair: pair[1], reverse=True
        )
    ]

    if not data:
        return None

    else:
        column_titles = ["RUN", "Complex phase standard deviation / kcal mol$^{-1}$"]
        fig, ax = plt.subplots()
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=data,
            colLabels=column_titles,
            loc="center",
            cellLoc="center",
        )

        # Make column headers bold
        for (row, col), cell in table.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight="bold"))

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

    relative_delta_fs_kcal = np.array(relative_delta_fs) * KT_KCALMOL

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
            rf"$N={n_below}$",
            rotation=90,
            verticalalignment="center",
            color="green",
        )
    plt.xlabel(r"Relative free energy to ligand 0 / kcal mol$^{-1}$")
    plt.ylabel("Cumulative $N$ ligands")


def plot_bootstrapped_clones(
    complex_phase: PhaseAnalysis,
    solvent_phase: PhaseAnalysis,
    clones_per_gen: int,
    n_gens: range,
    n_bootstrap: int = 100,
):
    """
    Plot free energy convergence with number of CLONEs

    Parameters
    ----------
    complex_phase : PhaseAnalysis
        results for complex
    solvent_phase : PhaseAnalysis
        results for solvent
    clones_per_gen : int
        Number of CLONEs per GEN
    n_gens : range
        Range of GENs for a RUN
    n_bootstrap : int
        Number of bootstrap samples

    """

    fig, ax = plt.subplots()

    for n in n_gens:
        # bootstrap

        complex_fes = bootstrap(
            free_energies=complex_phase.gens,
            n_bootstrap=n_bootstrap,
            clones_per_gen=clones_per_gen,
            gen_number=n,
        )

        plt.scatter(n, np.mean(complex_fes), color="red", label="complex")
        plt.errorbar(n, np.mean(complex_fes), yerr=np.std(complex_fes), c="red")

        solvent_fes = bootstrap(
            free_energies=solvent_phase.gens,
            n_bootstrap=n_bootstrap,
            clones_per_gen=clones_per_gen,
            gen_number=n,
        )

        plt.scatter(n, np.mean(solvent_fes), color="blue", label="solvent")
        plt.errorbar(n, np.mean(solvent_fes), yerr=np.std(solvent_fes), c="blue")

    plt.xlim(0, clones_per_gen + 10)
    plt.xlabel("Number of CLONEs")
    plt.ylabel(r"$\Delta$G / kcal mol$^{-1}$")
    plt.legend(["complex", "solvent"], loc="best")

    return fig


def _plot_updated_timestamp(timestamp: dt.datetime) -> None:
    fig = plt.gcf()
    fig.text(
        0.5,
        0.03,
        f"Updated {timestamp.isoformat()}",
        color="gray",
        horizontalalignment="center",
    )


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
        pdf_plt.savefig(bbox_inches="tight")


@contextmanager
def _save_plot(
    path: str,
    name: str,
    file_formats: Iterable[str],
    timestamp: Optional[dt.datetime] = None,
) -> Generator:
    """
    Context manager that creates a new figure on entry and saves the
    figure using the specified name, format, and path on exit.

    Parameters
    ----------
    path : str
        Path prefix to use in constructing the result path
    name : str
        Basename to use in constructing the result path
    file_formats : iterable of str
        File extensions with which to save the result. Elements must
        be accepted by ``plt.savefig``
    timestamp : datetime or None, optional
        If given, draw a watermark with the timestamp at the bottom of
        the figure

    Examples
    --------
    >>> with save_plot('example/plots', 'test_plot', 'png'):
    >>>     plt.plot(np.cos(np.linspace(-np.pi, np.pi)))
    >>>     plt.title("My cool plot")
    """
    # Make sure the directory exists
    import os

    os.makedirs(path, exist_ok=True)

    yield

    if timestamp is not None:
        plt.tight_layout(rect=(0, 0.05, 1, 1))  # leave space for timestamp on bottom
        _plot_updated_timestamp(timestamp)
    else:
        plt.tight_layout()

    for file_format in file_formats:
        plt.savefig(
            os.path.join(path, os.extsep.join([name, file_format])), transparent=True
        )

    fig.close()


def save_run_level_plots(
    run: int,
    complex_phase: PhaseAnalysis,
    solvent_phase: PhaseAnalysis,
    binding: Binding,
    path: str = os.curdir,
    file_formats: Iterable[str] = ("pdf", "png"),
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

    save_plot = partial(
        _save_plot,
        path=path,
        file_formats=file_formats,
        timestamp=dt.datetime.now(dt.timezone.utc),
    )

    with save_plot(name=f"RUN{run}"):
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

    with save_plot(name=f"RUN{run}-convergence"):
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

    with save_plot(name=f"RUN{run}-bootstrapped-CLONEs"):

        # Gather CLONES per GEN for run
        clones_per_gen = min(
            [
                len(works)
                for phase in [solvent_phase, complex_phase]
                for gen in phase.gens
                for works in [gen.forward_works, gen.reverse_works]
            ]
        )

        n_gens = range(10, clones_per_gen, 10)

        fig = plot_bootstrapped_clones(
            complex_phase=complex_phase,
            solvent_phase=solvent_phase,
            clones_per_gen=clones_per_gen,
            n_gens=n_gens,
        )
        fig.suptitle(f"RUN{run}")


def save_summary_plots(
    analysis: Analysis,
    path: str = os.curdir,
    file_formats: Iterable[str] = ("pdf", "png"),
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

    binding_delta_fs = [run.analysis.binding.delta_f for run in analysis.runs]

    save_plot = partial(
        _save_plot, path=path, file_formats=file_formats, timestamp=analysis.updated_at
    )

    with save_plot(
        name="relative_fe_dist",
    ):
        plot_relative_distribution(binding_delta_fs)
        plt.title("Relative free energy")

    with save_plot(
        name="cumulative_fe_dist",
    ):
        plot_cumulative_distribution(binding_delta_fs)
        plt.title("Cumulative distribution")

    with save_table_pdf(path, "poor_complex_convergence_fe_table"):
        plot_poor_convergence_fe_table(analysis.runs)
