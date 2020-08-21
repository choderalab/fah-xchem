from dataclasses import dataclass
from typing import List


@dataclass
class ResultPath:
    path: str
    clone: int
    gen: int


@dataclass
class Work:
    path: ResultPath
    forward_work: float
    reverse_work: float
    forward_final_potential: float
    reverse_final_potential: float


@dataclass
class PhaseAnalysis:
    delta_f: float
    ddelta_f: float
    bar_overlap: float
    forward_works: List[float]
    reverse_works: List[float]
    num_work_values: int


@dataclass
class Binding:
    delta_f: float
    ddelta_f: float


@dataclass
class RunAnalysis:
    complex_phase: PhaseAnalysis
    solvent_phase: PhaseAnalysis
    binding: Binding


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


@dataclass
class Run:
    details: RunDetails
    analysis: RunAnalysis
