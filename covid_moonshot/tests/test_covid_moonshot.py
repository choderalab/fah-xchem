import datetime as dt
import sys
from covid_moonshot.core import (
    CompoundSeries,
    CompoundSeriesMetadata,
    Compound,
    Molecule,
    Transformation,
)
import pytest

compound_series = CompoundSeries(
    metadata=CompoundSeriesMetadata(
        name="2020-08-20-benzotriazoles",
        description="Sprint 3: Prioritization of benzotriazole derivatives",
        creator="John D. Chodera",
        creation_date=dt.datetime(2020, 9, 8, 10, 14, 48, 607238),
        xchem_project="Mpro",
        biological_assembly="monomer",
        protein_variant="thiolate",
        temperature_kelvin=300,
        ionic_strength_millimolar=70,
        pH=7.4,
    ),
    compounds=[
        Compound(
            compound_id="MAT-POS-f42f3716-1",
            smiles="Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccc(C3CC3(F)F)cc2)c1",
            is_racemic_mixture=False,
            has_multiple_protonation_states=True,
            has_multiple_tautomers=False,
            experimental_data={"pIC50": 4.324},
        ),
        Compound(
            compound_id="MAT-POS-f42f3716-2",
            smiles="Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccc(S(C)(=O)=O)cc2Cl)c1",
            is_racemic_mixture=False,
            has_multiple_protonation_states=False,
            has_multiple_tautomers=False,
            experimental_data={"pIC50": 4.324},
        ),
    ],
    molecules=[
        Molecule(
            molecule_id="MAT-POS-f42f3716-1-1",
            cid="MAT-POS-f42f3716-1",
            smiles="Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccc(C3CC3(F)F)cc2)c1",
        ),
        Molecule(
            molecule_id="MAT-POS-f42f3716-1-2",
            cid="MAT-POS-f42f3716-1",
            smiles="Cc1ccn[H+]cc1NC(=O)Cc1cc(Cl)cc(-c2ccc(C3CC3(F)F)cc2)c1",
        ),
    ],
    transformations=[
        Transformation(
            run="RUN0",
            initial_molecule="MAT-POS-f42f3716-1-1",
            final_molecule="MAT-POS-f42f3716-1-2",
            xchem_fragment_id="x10789",
        ),
        Transformation(
            run="RUN1",
            initial_molecule="MAT-POS-f42f3716-1-1",
            final_molecule="MAT-POS-f42f3716-1-3",
            xchem_fragment_id="x10789",
        ),
    ],
)


def test_compound_series_json_serialization():
    """Test json serialize/deserialize roundtrip for CompoundSeries"""
    json = compound_series.to_json()
    deserialized = CompoundSeries.from_json(json)
    assert deserialized == compound_series
