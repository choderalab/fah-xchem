import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns
import numpy as np
from simtk.openmm import unit
from openmmtools.constants import kB
from typing import List
from ..core import Binding, PhaseAnalysis, RunAnalysis, Work

TEMPERATURE = 300.0 * unit.kelvin
KT = kB * TEMPERATURE


def plot_single_work_distribution(f, r, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(f, shade=True, color="cornflowerblue", ax=ax)
    sns.rugplot(
        f, ax=ax, color="cornflowerblue", alpha=0.5, label=f"forward : N={len(f)}",
    )

    sns.kdeplot([-x for x in r], shade=True, color="hotpink", ax=ax)
    sns.rugplot(
        [-x for x in r],
        ax=ax,
        color="hotpink",
        alpha=0.5,
        label=f"reverse : N={len(r)}",
    )
    if title is not None:
        ax.set_title(title)


def plot_two_work_distribution(f1, r1, f2, r2, phases=(None, None), title=None):
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 3.25))
    plot_single_work_distribution(f1, r1, ax1, phases[0])
    plot_single_work_distribution(f2, r2, ax2, phases[1])

    if title is not None:
        fig.suptitle(
            title, fontsize=16,
        )
    fig.subplots_adjust(top=0.9, wspace=0.15)
    ax1.legend()
    ax2.legend()


def plot_relative_distribution(relative_fes, bins=100, title="Relative affinity"):
    """ Plots the distribution of relative free energies

    Parameters
    ----------
    relative_fes : list
        Relative free energies in kcal/mol
    bins : int, default=100
        Number of bins for histogramming


    """
    # first convert kT to kcal/mol

    relative_fes = [
        (x * KT).value_in_unit(unit.kilocalories_per_mole) for x in relative_fes
    ]
    sns.kdeplot(relative_fes, shade=True, color="hotpink")
    sns.rugplot(
        relative_fes, color="hotpink", alpha=0.5, label=f"N={len(relative_fes)}",
    )
    plt.xlabel("Relative free energy to ligand 0 / kcal/mol")
    plt.title(title)


def plot_convergence(
    gens: List[int],
    complex_delta_fs: List[float],
    complex_delta_f_errs: List[float],
    solvent_delta_fs: List[float],
    solvent_delta_f_errs: List[float],
    binding_delta_f: float,
    binding_delta_f_err: float,
    bounds_zscore=1.65,  # 95th percentile
    title=None,
):

    for gen in gens:

        DDG = solvent_delta_fs[gen] - complex_delta_fs[gen]
        DDG_kC = (DDG * KT).value_in_unit(unit.kilocalories_per_mole)

        DDG_err = np.sqrt(
            solvent_delta_f_errs[gen] ** 2 + complex_delta_f_errs[gen] ** 2
        )
        DDG_err_kC = (DDG_err * KT).value_in_unit(unit.kilocalories_per_mole)

        plt.scatter(gen, DDG_kC, color="green")
        plt.vlines(
            gen,
            DDG_kC - DDG_err_kC * bounds_zscore,
            DDG_kC + DDG_err_kC * bounds_zscore,
            color="green",
        )

    for label, (delta_fs, delta_f_errs), color in [
        ("solvent", (solvent_delta_fs, solvent_delta_f_errs), "blue"),
        ("complex", (complex_delta_fs, complex_delta_f_errs), "red"),
    ]:

        delta_fs_kC = [
            (delta_f * KT).value_in_unit(unit.kilocalories_per_mole)
            for delta_f in delta_fs
        ]
        delta_f_errs_kC = [
            (delta_f_err * KT).value_in_unit(unit.kilocalories_per_mole)
            for delta_f_err in delta_f_errs
        ]

        shift = np.mean(delta_fs_kC)
        y = delta_fs_kC - shift
        plt.scatter(gens, y, color=color, label=label)
        for gen in gens:
            plt.vlines(
                gen,
                y[gen] - delta_f_errs_kC[gen] * bounds_zscore,
                y[gen] + delta_f_errs_kC[gen] * bounds_zscore,
                color=color,
            )

    plt.xlabel("GEN")
    plt.ylabel("Relative free energy /" + r" kcal mol${^-1}$")
    plt.hlines(
        (binding_delta_f * KT).value_in_unit(unit.kilocalories_per_mole),
        0,
        max(gens),
        color="green",
        linestyle=":",
        label="free energy (all GENS)",
    )
    plt.fill_between(
        [0, max(gens)],
        ((binding_delta_f - binding_delta_f_err * bounds_zscore) * KT).value_in_unit(
            unit.kilocalories_per_mole
        ),
        ((binding_delta_f + binding_delta_f_err * bounds_zscore) * KT).value_in_unit(
            unit.kilocalories_per_mole
        ),
        alpha=0.2,
        color="green",
    )
    plt.xticks([gen for gen in range(0, max(gens) + 1)])
    plt.legend()


def plot_cumulative_distributions(
    results,
    minimum=None,
    maximum=5,
    cmap="PiYG",
    n_bins=100,
    markers=[-2, -1, 0, 1, 2],
    title="Cumulative distribution",
):
    """Plots cumulative distribution of ligand affinities

    Parameters
    ----------
    results : list(float)
        List of affinities to plot
    maximum : int, default=5
        Maximum affinity to plot, saves plotting boring plateaus
    cmap : str, default='PiYG'
        string name of colormap to use
    n_bins : int, default=100
        Number of bins to use
    markers : list(float), default=range(-2,3)
        Affinity values at which to label
    title : str, default='Cumulative distribution'
        Title to label plot

    """
    results = [(x * KT).value_in_unit(unit.kilocalories_per_mole) for x in results]
    if minimum is None:
        results = [x for x in results if x < maximum]
    else:
        results = [x for x in results if minimum < x < maximum]

    # the colormap could be a kwarg
    cm = plt.cm.get_cmap(cmap)

    # Get the histogramp
    Y, X = np.histogram(list(results), n_bins)
    Y = np.cumsum(Y)
    x_span = X.max() - X.min()
    C = [cm(((X.max() - x) / x_span)) for x in X]

    plt.bar(X[:-1], Y, color=C, width=X[1] - X[0], edgecolor="k")

    for v in markers:
        plt.vlines(-v, 0, Y.max(), "grey", linestyles="dashed")
        plt.text(
            v - 0.5,
            0.8 * Y.max(),
            f"$N$ = {len([x for x in results if x < v])}",
            rotation=90,
            verticalalignment="center",
            color="green",
        )
    plt.xlabel("Affinity relative to ligand 0 / " + r"kcal mol$^{-1}$")
    plt.ylabel("Cumulative $N$ ligands")
    plt.title(title)


def get_plot_filename(path: str, name: str, file_format: str) -> str:
    return os.path.join(path, os.extsep.join([name, file_format]))


def save_run_level_plots(
    run: int,
    complex_phase: PhaseAnalysis,
    solvent_phase: PhaseAnalysis,
    binding: Binding,
    path: str = os.curdir,
    file_format: str = "pdf",
) -> None:

    complex_forward_works = [w for gen in complex_phase.gens for w in gen.forward_works]
    complex_reverse_works = [w for gen in complex_phase.gens for w in gen.reverse_works]
    solvent_forward_works = [w for gen in solvent_phase.gens for w in gen.forward_works]
    solvent_reverse_works = [w for gen in solvent_phase.gens for w in gen.reverse_works]

    plt.figure()
    plot_two_work_distribution(
        complex_forward_works,
        complex_reverse_works,
        solvent_forward_works,
        solvent_reverse_works,
        phases=("complex", "solvent"),
    )
    plt.savefig(get_plot_filename(path, f"run{run}", file_format))

    complex_gens = set([gen.gen for gen in complex_phase.gens])
    solvent_gens = set([gen.gen for gen in solvent_phase.gens])

    plt.figure()
    plot_convergence(
        gens=list(complex_gens.intersection(solvent_gens)),
        complex_delta_fs=[gen.free_energy.delta_f for gen in complex_phase.gens],
        complex_delta_f_errs=[gen.free_energy.ddelta_f for gen in complex_phase.gens],
        solvent_delta_fs=[gen.free_energy.delta_f for gen in solvent_phase.gens],
        solvent_delta_f_errs=[gen.free_energy.ddelta_f for gen in solvent_phase.gens],
        binding_delta_f=binding.delta_f,
        binding_delta_f_err=binding.ddelta_f,
    )
    plt.savefig(get_plot_filename(path, f"run{run}-convergence", file_format))


def save_summary_plots(
    runs: List[RunAnalysis], path: str = os.curdir, file_format: str = "pdf"
) -> None:
    binding_delta_fs = [run.binding.delta_f for run in runs]

    plt.figure()
    plot_relative_distribution(binding_delta_fs)
    plt.savefig(get_plot_filename(path, "rel_fe_hist", file_format))

    plt.figure()
    plot_cumulative_distributions(binding_delta_fs)
    plt.savefig(get_plot_filename(path, "cumulative_fe_hist", file_format))
