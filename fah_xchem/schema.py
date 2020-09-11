import datetime as dt
from typing import Dict, List, Optional

from pydantic import BaseModel


class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class ProjectIds(Model):
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
    fah_project_ids: ProjectIds


class Microstate(Model):
    microstate_id: str
    smiles: str


class Compound(Model):
    compound_id: str
    smiles: str
    experimental_data: Dict[str, float]
    microstates: List[Microstate]


class CompoundMicrostate(Model):
    compound_id: str
    microstate_id: str


class Transformation(Model):
    run_id: int
    initial_microstate: CompoundMicrostate
    final_microstate: CompoundMicrostate
    xchem_fragment_id: str


class CompoundSeries(Model):
    metadata: CompoundSeriesMetadata
    compounds: List[Compound]
    transformations: List[Transformation]
