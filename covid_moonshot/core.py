from dataclasses import dataclass
import datetime as dt
from typing import Dict, List, Optional
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ResultPath:
    path: str
    clone: int
    gen: int


@dataclass_json
@dataclass
class Work:
    path: ResultPath
    forward_work: float
    reverse_work: float
    forward_final_potential: float
    reverse_final_potential: float


@dataclass_json
@dataclass
class FreeEnergy:
    delta_f: float
    ddelta_f: float
    bar_overlap: float
    num_work_values: int


@dataclass_json
@dataclass
class GenAnalysis:
    gen: int
    free_energy: Optional[FreeEnergy]
    forward_works: List[float]
    reverse_works: List[float]


@dataclass_json
@dataclass
class PhaseAnalysis:
    free_energy: FreeEnergy
    gens: List[GenAnalysis]


@dataclass_json
@dataclass
class Binding:
    delta_f: float
    ddelta_f: float


@dataclass_json
@dataclass
class RunAnalysis:
    complex_phase: PhaseAnalysis
    solvent_phase: PhaseAnalysis
    binding: Binding


@dataclass_json
@dataclass
class CompoundSeriesMetadata:
    name: str
    description: str
    creator: str
    creation_date: dt.date
    xchem_project: str
    biological_assembly: str
    protein_variant: str
    temperature_kelvin: float
    ionic_strength_millimolar: float
    pH: float


@dataclass_json
@dataclass
class Compound:
    compound_id: str
    smiles: str
    is_racemic_mixture: bool
    has_multiple_protonation_states: bool
    has_multiple_tautomers: bool
    experimental_data: Dict[str, str]


@dataclass_json
@dataclass
class Molecule:
    molecule_id: str
    cid: str
    smiles: str


@dataclass_json
@dataclass
class Transformation:
    run: str
    initial_molecule: str
    final_molecule: str
    xchem_fragment_id: str


@dataclass_json
@dataclass
class CompoundSeries:
    metadata: CompoundSeriesMetadata
    compounds: List[Compound]
    molecules: List[Molecule]
    transformations: List[Transformation]


@dataclass_json
@dataclass
class RunDetails:
    JOBID: int
    directory: str
    end: int
    end_pIC50: float
    end_smiles: str
    end_title: str
    ff: str
    ligand: str
    protein: str
    start: int
    start_smiles: str
    start_title: str
    target: str

    def run_id(self) -> int:
        return self.JOBID


@dataclass_json
@dataclass
class Run:
    """
    Results of free energy analysis for a single run.

    Parameters
    ----------
    details : RunDetails
        Compound details extracted from input JSON
    analysis : RunAnalysis
        Results of free energy analysis

    Examples
    --------
    >>> import json
    >>> from covid_moonshot import Run

    >>> # Extract results for RUN0, complex phase
    >>> run = results[0]
    >>> phase = run.analysis.complex_phase

    >>> # Extract all works (ignoring GENs)
    >>> all_forward_works = [forward_work for gen in phase.gens for forward_work in gen.forward_works]
    >>> all_reverse_works = [reverse_work for gen in phase.gens for reverse_work in gen.reverse_works]

    >>> # Extract works for specific gen
    >>> forward_works = phase.gens[0].forward_works
    """

    details: RunDetails
    analysis: RunAnalysis


@dataclass_json
@dataclass
class Analysis:
    updated_at: dt.datetime
    runs: List[Run]
