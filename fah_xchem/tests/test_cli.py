import os
import json

import pytest
from click.testing import CliRunner

from fah_xchem.cli import cli
from fah_xchem.schema import ExperimentalCompoundDataUpdate

runner = CliRunner()


@pytest.mark.slow
@pytest.mark.skipif(
    (("CDD_VAULT_TOKEN" not in os.environ) and ("CDD_VAULT_NUM" not in os.environ)),
    reason="require both CDD_VAULT_TOKEN and CDD_VAULT_NUM to run CDD tests",
)
class TestCDD:

    protocol_ids = ("5549",)

    # @pytest.fixture(scope='class')
    # def cdddata_dir(self, tmpdir_factory):
    #    return tmpdir_factory.mktemp('cdd-data')

    def test_retrieve_molecule_data(self, tmpdir):
        with tmpdir.as_cwd():

            # download molecule and protocol data
            result = runner.invoke(
                cli,
                [
                    "cdd",
                    "--data-dir",
                    "cdd-data",
                    "retrieve-molecule-data",
                ],
            )

            # check for file presence
            assert os.path.exists("cdd-data/molecules.json")

    def test_retrieve_protocol_data(self, tmpdir):
        with tmpdir.as_cwd():

            # build up args for protocol ids
            protocols = []
            for pid in self.protocol_ids:
                protocols.append("-i")
                protocols.append(pid)

            # download molecule and protocol data
            result = runner.invoke(
                cli,
                [
                    "cdd",
                    "--data-dir",
                    "cdd-data",
                    "retrieve-protocol-data",
                ]
                + protocols,
            )

            # check for file presence
            assert not os.path.exists("cdd-data/molecules.json")
            for pid in self.protocol_ids:
                assert os.path.exists(f"cdd-data/protocols/{pid}")
                assert os.path.exists(f"cdd-data/protocols/{pid}/protocol-defs.json")
                assert os.path.exists(f"cdd-data/protocols/{pid}/protocol-records.json")

    def test_retrieve_protocol_data_with_molecules(self, tmpdir):
        with tmpdir.as_cwd():

            # build up args for protocol ids
            protocols = []
            for pid in self.protocol_ids:
                protocols.append("-i")
                protocols.append(pid)

            # download molecule and protocol data
            result = runner.invoke(
                cli,
                [
                    "cdd",
                    "--data-dir",
                    "cdd-data",
                    "retrieve-protocol-data",
                    "--molecules",
                ]
                + protocols,
            )

            # check for file presence
            assert os.path.exists("cdd-data/molecules.json")
            for pid in self.protocol_ids:
                assert os.path.exists(f"cdd-data/protocols/{pid}")
                assert os.path.exists(f"cdd-data/protocols/{pid}/protocol-defs.json")
                assert os.path.exists(f"cdd-data/protocols/{pid}/protocol-records.json")

    def test_generate_experimental_compound_data(self, tmpdir):

        with tmpdir.as_cwd():

            # build up args for protocol ids
            protocols = []
            for pid in self.protocol_ids:
                protocols.append("-i")
                protocols.append(pid)

            # download molecule and protocol data
            result = runner.invoke(
                cli,
                [
                    "cdd",
                    "--data-dir",
                    "cdd-data",
                    "retrieve-protocol-data",
                    "--molecules",
                ]
                + protocols,
            )

            assert result.exit_code == 0

            # generate downstream experimental data and analyze
            result = runner.invoke(
                cli,
                [
                    "cdd",
                    "--data-dir",
                    "cdd-data",
                    "generate-experimental-compound-data",
                ]
                + protocols
                + ["experimental-data.json"],
            )

            assert result.exit_code == 0

            # load experimental compound data and use schema; will fail if not possible
            with open("experimental-data.json", "r") as f:
                exp_data = ExperimentalCompoundDataUpdate.parse_raw(f.read())
