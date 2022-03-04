"""Tests for CDD data retrieval.

"""
import os
import json
from pathlib import Path

import pytest

from fah_xchem.external.cdd import CDDData


@pytest.mark.slow
@pytest.mark.skipif(
    (("CDD_VAULT_TOKEN" not in os.environ) and ("CDD_VAULT_NUM" not in os.environ)),
    reason="require both CDD_VAULT_TOKEN and CDD_VAULT_NUM to run CDD tests",
)
class TestCDDData:
    @pytest.fixture(scope="class")
    def cdddata(self, tmpdir_factory):

        data_dir = tmpdir_factory.mktemp("cdd-data")

        vault_token = os.environ["CDD_VAULT_TOKEN"]
        vault_num = os.environ["CDD_VAULT_NUM"]

        cdd = CDDData(
            data_dir=os.path.abspath(data_dir),
            vault_num=vault_num,
            vault_token=vault_token,
        )

        protocol_ids = [cdd.fluorescence_IC50_protocol_id]
        cdd.retrieve_protocol_data(protocol_ids, molecules=True)

        return cdd

    @pytest.mark.skip(
        "not testing for now as this adds substantial wait time to test suite"
    )
    def test_retrieve_molecule_data(self, cdddata):
        cdddata.retrieve_molecule_data()

        assert (cdddata.data_dir / "molecules.json").exists()

        with open(cdddata.data_dir / "molecules.json", "r") as f:
            molecules = json.load(f)

        assert all([mol["class"] == "molecule" for mol in molecules["objects"]])

    @pytest.mark.skip(
        "not testing for now as this adds substantial wait time to test suite"
    )
    def test_retrieve_protocol_data(self, cdddata):
        protocol_ids = [cdddata.fluorescence_IC50_protocol_id]
        cdddata.retrieve_protocol_data(protocol_ids)

        assert (cdddata.data_dir / "protocols") == cdddata.protocols_dir
        assert (cdddata.data_dir / "protocols").exists()
        assert (
            cdddata.data_dir / "protocols" / cdddata.fluorescence_IC50_protocol_id
        ).exists()

        assert {"protocol-data.json", "protocol-defs.json"} == {
            p.name
            for p in (
                cdddata.protocols_dir / cdddata.fluorescence_IC50_protocol_id
            ).iterdir()
        }

        with open(
            cdddata.protocols_dir
            / cdddata.fluorescence_IC50_protocol_id
            / "protocol-data.json",
            "r",
        ) as f:
            protocol_data = json.load(f)

        with open(
            cdddata.protocols_dir
            / cdddata.fluorescence_IC50_protocol_id
            / "protocol-defs.json",
            "r",
        ) as f:
            protocol_defs = json.load(f)

        assert all(["readouts" in p for p in protocol_data["objects"]])
        assert all(
            [
                p["class"] == "readout definition"
                for p in protocol_defs["readout_definitions"]
            ]
        )

    def test_retrieve_protocol_data_with_molecules(self, cdddata):
        # already performed in fixture
        # protocol_ids = [cdddata.fluorescence_IC50_protocol_id]
        # cdddata.retrieve_protocol_data(protocol_ids, molecules=True)

        assert (cdddata.data_dir / "molecules.json").exists()

        with open(cdddata.data_dir / "molecules.json", "r") as f:
            molecules = json.load(f)

        assert all([mol["class"] == "molecule" for mol in molecules["objects"]])

        assert (cdddata.data_dir / "protocols") == cdddata.protocols_dir
        assert (cdddata.data_dir / "protocols").exists()
        assert (
            cdddata.data_dir / "protocols" / cdddata.fluorescence_IC50_protocol_id
        ).exists()

        assert {"protocol-records.json", "protocol-defs.json"} == {
            p.name
            for p in (
                cdddata.protocols_dir / cdddata.fluorescence_IC50_protocol_id
            ).iterdir()
        }

        with open(
            cdddata.protocols_dir
            / cdddata.fluorescence_IC50_protocol_id
            / "protocol-records.json",
            "r",
        ) as f:
            protocol_data = json.load(f)

        with open(
            cdddata.protocols_dir
            / cdddata.fluorescence_IC50_protocol_id
            / "protocol-defs.json",
            "r",
        ) as f:
            protocol_defs = json.load(f)

        assert all(["readouts" in p for p in protocol_data["objects"]])
        assert all(
            [
                p["class"] == "readout definition"
                for p in protocol_defs["readout_definitions"]
            ]
        )

    def test_generate_experimental_compound_data(self, cdddata):
        protocol_ids = [cdddata.fluorescence_IC50_protocol_id]

        # already performed in fixture
        # cdddata.retrieve_protocol_data(protocol_ids, molecules=True)

        ec = cdddata.generate_experimental_compound_data(protocol_ids)

        # test that the fields indicating variety of compound are mutually exclusive in the data
        for compound in ec.compounds:
            compound_varieties = [
                compound.racemic,
                compound.achiral,
                compound.absolute_stereochemistry_enantiomerically_pure,
                compound.relative_stereochemistry_enantiomerically_pure,
            ]

            if compound.smiles is not None:
                assert sum(compound_varieties) == 1
            else:
                assert sum(compound_varieties) == 0

            # here we check that pIC50s are within a typical range
            pIC50 = compound.experimental_data.get("pIC50")
            if pIC50:
                assert 3.0 < pIC50 < 8.0

        assert True
