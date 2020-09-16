import datetime as dt
from typing import Dict, List, Optional

from pydantic import BaseModel


class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class PointEstimate(Model):
    point: float
    stderr: float

    def __add__(self, other: "PointEstimate") -> "PointEstimate":
        from math import sqrt

        return PointEstimate(
            point=self.point + other.point,
            stderr=sqrt(self.stderr ** 2 + other.stderr ** 2),
        )

    def __neg__(self) -> "PointEstimate":
        return PointEstimate(point=-self.point, stderr=self.stderr)

    def __sub__(self, other: "PointEstimate"):
        return self + -other

    def __mul__(self, c: float) -> "PointEstimate":
        return PointEstimate(point=c * self.point, stderr=c * self.stderr)

    def precision_decimals(self) -> Optional[int]:
        from math import floor, isfinite, log10

        return -floor(log10(self.stderr)) if isfinite(self.stderr) else None


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
    free_energy_penalty: Optional[PointEstimate]
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

    def __hash__(self):
        return hash((self.compound_id, self.microstate_id))


class Transformation(Model):
    run_id: int
    xchem_fragment_id: str
    initial_microstate: CompoundMicrostate
    final_microstate: CompoundMicrostate


class CompoundSeries(Model):
    metadata: CompoundSeriesMetadata
    compounds: List[Compound]
    transformations: List[Transformation]


class DataPath(Model):
    path: str
    clone: int
    gen: int


class WorkPair(Model):
    clone: int
    forward: float
    reverse: float


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
    free_energy: Optional[PointEstimate]


class CompoundAnalysis(Model):
    metadata: CompoundMetadata
    microstates: List[MicrostateAnalysis]
    free_energy: PointEstimate


class CompoundSeriesAnalysis(Model):
    metadata: CompoundSeriesMetadata
    compounds: List[CompoundAnalysis]
    transformations: List[TransformationAnalysis]


class AnalysisConfig(Model):
    min_num_work_values: Optional[int] = None
    max_binding_free_energy: Optional[float] = 0


class FahConfig(Model):
    projects_dir: str = "projects"
    data_dir: str = "data"
