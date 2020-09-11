"""
Unit and regression test for the fah_xchem package.
"""

# Import package, test suite, and other packages as needed
import pytest
import sys
from fah_xchem.schema import *


@pytest.fixture
def compound_series():
    return CompoundSeries(
        metadata=CompoundSeriesMetadata(
            name="2020-08-20-benzotriazoles",
            description="Sprint 3: Prioritization of benzotriazole derivatives",
            creator="John D. Chodera",
            created_at=dt.datetime(2020, 9, 8, 10, 14, 48, 607238),
            xchem_project="Mpro",
            receptor_variant=dict(
                biological_assembly="monomer", protein_variant="thiolate"
            ),
            temperature_kelvin=300,
            ionic_strength_millimolar=70,
            pH=7.4,
            fah_project_ids=ProjectIds(complex_phase=12345, solvent_phase=12346),
        ),
        compounds=[
            Compound(
                compound_id="MAT-POS-f42f3716-1",
                smiles="Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccc(C3CC3(F)F)cc2)c1",
                experimental_data={"pIC50": 4.324},
                microstates=[
                    Microstate(
                        microstate_id="MAT-POS-f42f3716-1-1",
                        smiles="Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccc(C3CC3(F)F)cc2)c1",
                    ),
                    Microstate(
                        microstate_id="MAT-POS-f42f3716-1-2",
                        smiles="Cc1ccn[H+]cc1NC(=O)Cc1cc(Cl)cc(-c2ccc(C3CC3(F)F)cc2)c1",
                    ),
                ],
            ),
            Compound(
                compound_id="MAT-POS-f42f3716-2",
                smiles="Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccc(S(C)(=O)=O)cc2Cl)c1",
                experimental_data={"pIC50": 4.324},
                microstates=[
                    Microstate(
                        microstate_id="MAT-POS-f42f3716-2-1",
                        smiles="Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccc(C3CC3(F)F)cc2)c1",
                    ),
                    Microstate(
                        microstate_id="MAT-POS-f42f3716-2-2",
                        smiles="Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccc(C3CC3(F)F)cc2)c1",
                    ),
                ],
            ),
        ],
        transformations=[
            Transformation(
                run_id=0,
                initial_microstate=CompoundMicrostate(
                    compound_id="MAT-POS-f42f3716-1",
                    microstate_id="MAT-POS-f42f3716-1-1",
                ),
                final_microstate=CompoundMicrostate(
                    compound_id="MAT-POS-f42f3716-1",
                    microstate_id="MAT-POS-f42f3716-1-2",
                ),
                xchem_fragment_id="x10789",
            ),
            Transformation(
                run_id=1,
                initial_microstate=CompoundMicrostate(
                    compound_id="MAT-POS-f42f3716-2",
                    microstate_id="MAT-POS-f42f3716-2-1",
                ),
                final_microstate=CompoundMicrostate(
                    compound_id="MAT-POS-f42f3716-2",
                    microstate_id="MAT-POS-f42f3716-2-2",
                ),
                xchem_fragment_id="x10789",
            ),
        ],
    )


def test_fah_xchem_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "fah_xchem" in sys.modules


def test_compound_series_json_serialization(compound_series):
    """Test json serialize/deserialize roundtrip for CompoundSeries"""
    json = compound_series.json()
    deserialized = CompoundSeries.parse_raw(json)
    assert deserialized == compound_series
