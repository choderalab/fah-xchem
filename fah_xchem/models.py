import datetime as dt
from typing import Dict, List, Optional
from pydantic import BaseModel


class ModelWithConfig(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class ResultPath(ModelWithConfig):
    path: str
    clone: int
    gen: int


class Work(ModelWithConfig):
    path: ResultPath
    forward_work: float
    reverse_work: float
    forward_final_potential: float
    reverse_final_potential: float


class FreeEnergy(ModelWithConfig):
    delta_f: float
    ddelta_f: float
    bar_overlap: float
    num_work_values: int


class GenAnalysis(ModelWithConfig):
    gen: int
    free_energy: Optional[FreeEnergy]
    forward_works: List[float]
    reverse_works: List[float]


class PhaseAnalysis(ModelWithConfig):
    free_energy: FreeEnergy
    gens: List[GenAnalysis]


class Binding(ModelWithConfig):
    delta_f: float
    ddelta_f: float


class RunAnalysis(ModelWithConfig):
    complex_phase: PhaseAnalysis
    solvent_phase: PhaseAnalysis
    binding: Binding


class CompoundSeriesMetadata(ModelWithConfig):
    name: str
    description: str
    creator: str
    creation_date: dt.date
    xchem_project: str
    receptor_variant: Dict[str, str]
    temperature_kelvin: float
    ionic_strength_millimolar: float
    pH: float


class Molecule(ModelWithConfig):
    molecule_id: str
    smiles: str


class Compound(ModelWithConfig):
    compound_id: str
    smiles: str
    experimental_data: Dict[str, float]
    molecules: List[Molecule]


class Transformation(BaseModel):
    run: str
    compound_id: str
    initial_molecule_id: str
    final_molecule_id: str
    xchem_fragment_id: str


class CompoundSeries(ModelWithConfig):
    metadata: CompoundSeriesMetadata
    compounds: List[Compound]
    transformations: List[Transformation]


class RunDetails(ModelWithConfig):
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


class Run(ModelWithConfig):
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


class Analysis(ModelWithConfig):
    updated_at: dt.datetime
    runs: List[Run]
