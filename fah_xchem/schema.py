import datetime as dt
from typing import Dict, List, Optional

from pydantic import BaseModel


class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class ProjectPair(Model):
    complex_phase: int
    solvent_phase: int


class CompoundSeriesMetadata(Model):
    name: str
    description: str
    creator: str
    created_at: dt.date
    xchem_project: str
    receptor_variant: Dict[str, str]
    temperature_kelvin: float
    ionic_strength_millimolar: float
    pH: float
    fah_projects: ProjectPair


class Microstate(Model):
    microstate_id: str
    smiles: str


class CompoundMetadata(Model):
    compound_id: str
    smiles: str
    experimental_data: Dict[str, float]


class Compound(Model):
    metadata: CompoundMetadata
    microstates: List[Microstate]


class CompoundMicrostate(Model):
    compound_id: str
    microstate_id: str


class Transformation(Model):
    run_id: int
    xchem_fragment_id: str
    initial_microstate: CompoundMicrostate
    final_microstate: CompoundMicrostate


class CompoundSeries(Model):
    metadata: CompoundSeriesMetadata
    compounds: List[Compound]
    transformations: List[Transformation]


class WorkPair(Model):
    forward: float
    reverse: float


class PointEstimate(Model):
    point: float
    stderr: float


class RelativeFreeEnergy(Model):
    delta_f: PointEstimate
    bar_overlap: float
    num_work_pairs: int


class GenAnalysis(Model):
    gen: int
    works: List[WorkPair]
    free_energy: Optional[RelativeFreeEnergy]


class PhaseAnalysis(Model):
    free_energy: RelativeFreeEnergy
    gens: List[GenAnalysis]


class TransformationAnalysis(Model):
    transformation: Transformation
    binding_free_energy: PointEstimate
    complex_phase: PhaseAnalysis
    solvent_phase: PhaseAnalysis


class MicrostateAnalysis(Model):
    microstate: Microstate
    absolute_free_energy: PointEstimate


class CompoundAnalysis(Model):
    metadata: CompoundMetadata
    microstates: List[MicrostateAnalysis]


class CompoundSeriesAnalysis(Model):
    metadata: CompoundSeriesMetadata
    compounds: List[CompoundAnalysis]
    transformations: List[TransformationAnalysis]


class AnalysisConfig(Model):
    pass


class FahConfig(Model):
    projects_dir: str = "projects"
    data_dir: str = "data"
