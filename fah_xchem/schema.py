import datetime as dt
from typing import Dict, List, Optional

from pydantic import BaseModel


class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class CompoundSeriesMetadata(Model):
    name: str
    description: str
    creator: str
    creation_date: dt.date
    xchem_project: str
    receptor_variant: Dict[str, str]
    temperature_kelvin: float
    ionic_strength_millimolar: float
    pH: float


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
    run: int
    initial_microstate: CompoundMicrostate
    final_microstate: CompoundMicrostate
    xchem_fragment_id: str


class CompoundSeries(Model):
    metadata: CompoundSeriesMetadata
    compounds: List[Compound]
    transformations: List[Transformation]


class ServerConfig(Model):
    projects_path: str
    data_path: str


class AnalysisConfig(Model):
    max_binding_delta_f: Optional[float] = None
    min_num_work_values: Optional[int] = 40
    work_precision_decimals: Optional[int] = 3


class Config(Model):
    server: ServerConfig
    complex_project: int
    solvent_project: int
    analysis: AnalysisConfig
    output_dir: str
