import pathlib
import datetime as dt
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class PointEstimate(Model):
    point: Union[None, float]
    stderr: Union[None, float]

    def __add__(self, other: "PointEstimate") -> "PointEstimate":
        from math import sqrt

        return PointEstimate(
            point=self.point + other.point,
            stderr=sqrt(self.stderr ** 2 + other.stderr ** 2),
        )

    def __abs__(self) -> "PointEstimate":
        return PointEstimate(point=abs(self.point), stderr=self.stderr)
    
    def __neg__(self) -> "PointEstimate":
        return PointEstimate(point=-self.point, stderr=self.stderr)

    def __sub__(self, other: "PointEstimate"):
        return self + -other

    def __mul__(self, c: float) -> "PointEstimate":
        return PointEstimate(point=c * self.point, stderr=c * self.stderr)

    def precision_decimals(self) -> Optional[int]:
        from math import floor, isfinite, log10

        if self.point is None:
            return None
        else:
            return -floor(log10(self.stderr)) if isfinite(self.stderr) else None


class ProjectPair(Model):
    complex_phase: int = Field(
        None, description="The Folding@Home project code for the complex phase"
    )
    solvent_phase: int = Field(
        None, description="The Folding@Home project code for the solvent phase"
    )


class CompoundSeriesMetadata(Model):
    name: str
    description: str = Field(
        None, description="A description of the current sprint and compound series"
    )
    creator: str = Field(
        None,
        description="The full name of the creator. Optional addition of email address",
    )
    created_at: dt.date = Field(dt.date, description="Date of creation")
    xchem_project: str = Field(None, description="The name of the project")
    receptor_variant: Dict[str, str] = Field(
        dict(), description="A brief description of the receptor variant."
    )
    temperature_kelvin: float = Field(
        300,
        description="The temperature (in Kelvin) that the simulations are performed at",
    )
    ionic_strength_millimolar: float = Field(
        70,
        description="The ionic strength (in millimolar) that the simulations are performed at",
    )
    pH: float = Field(
        7.3, description="The pH at which the simulations are performed at"
    )
    fah_projects: ProjectPair = Field(
        None, description="The complex and solvent phase Folding@Home project codes"
    )


class Microstate(Model):
    microstate_id: str = Field(
        None,
        description="The unique microstate identifier (based on the PostEra or enumerated ID)",
    )
    free_energy_penalty: PointEstimate = PointEstimate(point=0.0, stderr=0.0)
    smiles: str = Field(
        None, description="The SMILES string of the compound in a unique microstate"
    )


class CompoundMetadata(Model):
    compound_id: str = Field(
        None, description="The unique compound identifier (PostEra or enumerated ID)"
    )
    smiles: str = Field(
        None,
        description="The SMILES string defining the compound in a canonical protonation state. Stereochemistry will be ambiguous for racemates",
    )
    experimental_data: Dict[str, float] = Field(
        dict(), description='Optional experimental data fields, such as "pIC50"'
    )


class Compound(Model):
    metadata: CompoundMetadata = Field(
        None,
        description="The compound metdata including compound ID, SMILES, and any associated experimental data",
    )
    microstates: List[Microstate] = Field(
        None,
        description="The associated microstates of the compound including microstate ID, free energy penalty, and SMILES",
    )


class CompoundMicrostate(Model):
    compound_id: str
    microstate_id: str

    def __hash__(self):
        return hash((self.compound_id, self.microstate_id))


class Transformation(Model):
    run_id: int = Field(
        None,
        description="The RUN number corresponding to the Folding@Home directory structure",
    )
    xchem_fragment_id: str = Field(None, description="The XChem fragment screening ID")
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
    reliable_transformation: bool = Field(None, description="Specify if the transformation is reliable or not") # JSON boolean
    binding_free_energy: PointEstimate
    exp_ddg: PointEstimate # TODO: Make optional, with None as default?
    absolute_error: Optional[PointEstimate] = None
    complex_phase: PhaseAnalysis
    solvent_phase: PhaseAnalysis


class MicrostateAnalysis(Model):
    microstate: Microstate
    free_energy: Optional[PointEstimate]
    first_pass_free_energy: Optional[PointEstimate]


class CompoundAnalysis(Model):
    metadata: CompoundMetadata
    microstates: List[MicrostateAnalysis]
    free_energy: Optional[PointEstimate]


class CompoundSeriesAnalysis(Model):
    metadata: CompoundSeriesMetadata
    compounds: List[CompoundAnalysis]
    transformations: List[TransformationAnalysis]


class AnalysisConfig(Model):
    """Configuration for fah-xchem analysis components.

    """
    min_num_work_values: Optional[int] = Field(
            None,
            description=("Minimum number of valid work value pairs required for "
                         "analysis. Raises InsufficientDataError if not satisfied.")
            )
    max_binding_free_energy: Optional[float] = Field(
            None,
            description="Don't report compounds with free energies greater than this (in kT)"
            )
    structure_path: pathlib.Path = Field(
            None,
            description="Path to reference structure directory."
            )
    target_name: str = Field(
            'Mpro',
            description="Name of target (e.g. 'Mpro')."
            )
    annotations: str = Field(
            "",
            description="Additional characters in the reference file name (e.g. '_0A_bound')."
            )
    component: str = Field(
            'protein',
            description="Component of the system the reference corresponds to (e.g. 'protein')"
            )


class FahConfig(Model):
    projects_dir: str = "projects"
    data_dir: str = "data"


class FragalysisConfig(Model):
    run: bool = Field(False)
    ligands_filename: str = None
    fragalysis_sdf_filename: str = None
    ref_url: str = None
    ref_mols: str = None
    ref_pdb: str = None
    target_name: str = None
    submitter_name: str = None
    submitter_email: str = None
    submitter_institution: str = None
    method: str = None
    upload_key: str = None
    new_upload: bool = Field(False)


class RunStatus(Model):
    run_id: int = Field(
        None,
        description="The RUN number corresponding to the Folding@Home directory structure",
    )
    complex_phase_work_units: int = Field(
        0,
        description="The number of completed complex phase work units",
    )
    solvent_phase_work_units: int = Field(
        0,
        description="The number of completed solvent phase work units",
    )
    has_changed: bool = True
