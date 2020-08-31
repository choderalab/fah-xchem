from typing import List
import numpy as np
from covid_moonshot.core import Binding, RunAnalysis, Work
from .free_energy import get_phase_analysis


def get_run_analysis(
    run: int, complex_works: List[Work], solvent_works: List[Work],
) -> RunAnalysis:

    try:
        complex_phase = get_phase_analysis(run, "complex", complex_works)
    except ValueError as e:
        raise ValueError(f"Failed to analyze complex: {e}")

    try:
        solvent_phase = get_phase_analysis(run, "solvent", solvent_works)
    except ValueError as e:
        raise ValueError(f"Failed to analyze solvent: {e}")

    binding = Binding(
        delta_f=solvent_phase.free_energy.delta_f - complex_phase.free_energy.delta_f,
        ddelta_f=np.sqrt(
            complex_phase.free_energy.ddelta_f ** 2
            + solvent_phase.free_energy.ddelta_f ** 2
        ),
    )

    return RunAnalysis(
        complex_phase=complex_phase, solvent_phase=solvent_phase, binding=binding
    )
