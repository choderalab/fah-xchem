from dataclasses import dataclass
from typing import Dict, List
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
    free_energy: FreeEnergy
    forward_works: List[float]
    reverse_works: List[float]


@dataclass_json
@dataclass
class PhaseAnalysis:
    free_energy: FreeEnergy
    gens: Dict[int, GenAnalysis]


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
        return self.JOBID - 1


@dataclass_json
@dataclass
class Run:
    details: RunDetails
    analysis: RunAnalysis
