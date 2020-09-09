import datetime as dt
from typing import Dict, List, Optional
from pydantic import BaseModel


class ResultPath(BaseModel):
    path: str
    clone: int
    gen: int


class Work(BaseModel):
    path: ResultPath
    forward_work: float
    reverse_work: float
    forward_final_potential: float
    reverse_final_potential: float


class FreeEnergy(BaseModel):
    delta_f: float
    ddelta_f: float
    bar_overlap: float
    num_work_values: int


class GenAnalysis(BaseModel):
    gen: int
    free_energy: Optional[FreeEnergy]
    forward_works: List[float]
    reverse_works: List[float]


class PhaseAnalysis(BaseModel):
    free_energy: FreeEnergy
    gens: List[GenAnalysis]


class Binding(BaseModel):
    delta_f: float
    ddelta_f: float


class RunAnalysis(BaseModel):
    complex_phase: PhaseAnalysis
    solvent_phase: PhaseAnalysis
    binding: Binding


class CompoundSeriesMetadata(BaseModel):
    name: str
    description: str
    creator: str
    creation_date: dt.date
    xchem_project: str
    receptor_variant: Dict[str, str]
    temperature_kelvin: float
    ionic_strength_millimolar: float
    pH: float


class Molecule(BaseModel):
    molecule_id: str
    smiles: str


class Compound(BaseModel):
    compound_id: str
    smiles: str
    experimental_data: Dict[str, float]
    molecules: List[Molecule]


class Transformation(BaseModel):
    run: str
    initial_molecule: str
    final_molecule: str
    xchem_fragment_id: str


class CompoundSeries(BaseModel):
    metadata: CompoundSeriesMetadata
    compounds: List[Compound]
    transformations: List[Transformation]


class RunDetails(BaseModel):
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


class Run(BaseModel):
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
    >>> from fah_xchem import Run

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


class Analysis(BaseModel):
    updated_at: dt.datetime
    runs: List[Run]
