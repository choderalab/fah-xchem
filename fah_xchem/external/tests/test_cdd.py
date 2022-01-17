"""Tests for CDD data retrieval.

"""
import os
import json
from pathlib import Path

import pytest

from fah_xchem.external.cdd import CDDData


@pytest.mark.skipif((('CDD_VAULT_TOKEN' not in os.environ) and
                     ('CDD_VAULT_NUM' not in os.environ)),
                     reason="require both CDD_VAULT_TOKEN and CDD_VAULT_NUM to run CDD tests")
class TestCDDData:

    @pytest.fixture
    def cdddata(self, tmpdir):
        with tmpdir.as_cwd():
            vault_token = os.environ['CDD_VAULT_TOKEN']
            vault_num = os.environ['CDD_VAULT_NUM']

            cdd = CDDData(data_dir=Path('cdd-data').absolute(), vault_token=vault_token)

        return cdd

    def test_retrieve_molecule_data(self, cdddata):
        cdddata.retrieve_molecule_data()

        assert (cdddata.data_dir / 'molecules.json').exists()

        with open(cdddata.data_dir / 'molecules.json', 'r') as f:
            molecules = json.load(f)

        assert all([mol['class'] == 'molecule' for mol in molecules['objects']])

    def test_retrieve_protocol_data(self, cdddata):
        protocol_ids = [cdddata.fluorescence_IC50_protocol_id]
        cdddata.retrieve_protocol_data(protocol_ids)

        assert (cdddata.data_dir / 'protocols') == cdddata.protocols_dir
        assert (cdddata.data_dir / 'protocols').exists()
        assert (cdddata.data_dir / 'protocols' / cdddata.fluorescence_IC50_protocol_id).exists()

        assert (
                {'protocol-data.json', 'protocol-defs.json'} == 
                {p.name for p in
                    (cdddata.protocols_dir / cdddata.fluorescence_IC50_protocol_id).iterdir()}
                )

        with open(cdddata.protocols_dir / cdddata.fluorescence_IC50_protocol_id / 'protocol-data.json', 'r') as f:
            protocol_data = json.load(f)

        with open(cdddata.protocols_dir / cdddata.fluorescence_IC50_protocol_id / 'protocol-defs.json', 'r') as f:
            protocol_defs = json.load(f)

        assert all(['readouts' in p for p in protocol_data['objects']])
        assert all([p['class'] == 'readout definition' for p in protocol_defs['readout_definitions']])

    def test_retrieve_protocol_data_with_molecules(self, cdddata):
        protocol_ids = [cdddata.fluorescence_IC50_protocol_id]
        cdddata.retrieve_protocol_data(protocol_ids, molecules=True)

        assert (cdddata.data_dir / 'molecules.json').exists()

        with open(cdddata.data_dir / 'molecules.json', 'r') as f:
            molecules = json.load(f)

        assert all([mol['class'] == 'molecule' for mol in molecules['objects']])

        assert (cdddata.data_dir / 'protocols') == cdddata.protocols_dir
        assert (cdddata.data_dir / 'protocols').exists()
        assert (cdddata.data_dir / 'protocols' / cdddata.fluorescence_IC50_protocol_id).exists()

        assert (
                {'protocol-data.json', 'protocol-defs.json'} == 
                {p.name for p in
                    (cdddata.protocols_dir / cdddata.fluorescence_IC50_protocol_id).iterdir()}
                )

        with open(cdddata.protocols_dir / cdddata.fluorescence_IC50_protocol_id / 'protocol-data.json', 'r') as f:
            protocol_data = json.load(f)

        with open(cdddata.protocols_dir / cdddata.fluorescence_IC50_protocol_id / 'protocol-defs.json', 'r') as f:
            protocol_defs = json.load(f)

        assert all(['readouts' in p for p in protocol_data['objects']])
        assert all([p['class'] == 'readout definition' for p in protocol_defs['readout_definitions']])
