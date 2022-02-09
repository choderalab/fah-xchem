"""Interface to Collaborative Drug Discovery APIs and data artifacts.

"""
import os
import time
import logging
import json
import shutil
from typing import List, Dict, Tuple, Union
from collections import defaultdict

import requests
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .base import ExternalData
from ..schema import ExperimentalCompoundData, ExperimentalCompoundDataUpdate


class FailedExportError(Exception):
    """Export failed to finish within timeout."""


class CDDData(ExternalData):
    """Collaborative Drug Discovery (CDD) data interface.

    Sourced from: https://github.com/postera-ai/COVID_moonshot_submissions/blob/fixing_stereochem/lib/get_experimental_data.py

    """

    base_url: str = Field(
        "https://app.collaborativedrug.com/api/v1/vaults",
        description="Base URL for CCD Vault API",
    )
    vault_token: str = Field(..., description="API token for access to CDD Vault")
    vault_num: str = Field("5549", description="CDD Vault number")
    fluorescence_IC50_protocol_id: str = Field(
        "49439", description="Protocol ID for fluorescence measurements"
    )

    _protocol_processing_map = {"49439": "_process_fluorescence_IC50"}

    @property
    def protocols_dir(self):
        return self.data_dir.joinpath("protocols")

    def get_available_protocols(self):
        """Return full protocol definitions for all protocols available.

        Returns
        -------
        protocol_defs
            Full response from CDD on available protocols, definitions for all readouts.

        """
        # retrieve protocol definitions
        headers = {"X-CDD-token": self.vault_token}

        url = f"{self.base_url}/{self.vault_num}/protocols/"
        url = f"{url}?async=True"

        response = self._get_async_exports({"protocols": url}, headers=headers)[
            "protocols"
        ]

        protocol_data = response.json()

        return protocol_data

    def _get_async_exports(
        self, async_urls: Dict[str, str], headers: Dict[str, str], timeout: int = 3600
    ):
        from rich.live import Live
        from rich.text import Text
        from rich.console import Console

        responses = {}
        for name, async_url in async_urls.items():
            responses[name] = requests.get(async_url, headers=headers)

        console = Console()
        text = Text("Retrieving urls...")
        for async_url in async_urls.values():
            text.append(f"\n: {async_url}")

        console.print(text)

        logging.info("CDDData : Beginning export(s)")
        text = Text(
            "CDDData : Beginning export(s)",
            style="bold red",
        )

        with Live(text, refresh_per_second=4) as live:

            export_ids = {}
            for name, response in responses.items():
                export_info = response.json()
                export_ids[name] = export_info["id"]

            # CHECK STATUS of Export
            statuses = {name: None for name in async_urls}
            seconds_waiting = 0

            first = True
            while any([status != "finished" for status in statuses.values()]):
                text.remove_suffix(" --> Checking status of export(s)")

                for name, export_id in export_ids.items():
                    url = (
                        f"{self.base_url}/{self.vault_num}/export_progress/{export_id}"
                    )
                    response = requests.get(url, headers=headers)

                    # to view the status, use:
                    statuses[name] = response.json()["status"]

                if first:
                    logging.info("CDDData : Checking status of export(s)")
                text.append(" --> Checking status of export(s)", style="bold yellow")

                time.sleep(5)
                seconds_waiting += 5
                if seconds_waiting > timeout:
                    logging.info("Export Never Finished")
                    break

                first = False

            failed = []
            for name, status in statuses.items():
                if status != "finished":
                    failed.append(async_urls[name])
            if failed:
                raise FailedExportError(
                    f"The following urls failed to deliver an export within the timeout: {failed}"
                )

            logging.info("CDDData : Retrieving finished export(s)")
            text.append(" --> Retrieving finished export(s)", style="bold green")
            exports = {}
            for name, export_id in export_ids.items():
                url = f"{self.base_url}/{self.vault_num}/exports/{export_id}"
                exports[name] = requests.get(url, headers=headers)

            text.append(" --> Done", style="bold blue")

        return exports

    def _get_protocol_data(self, protocol_ids, molecules=False):

        urls = {}
        protocol_defs = {}
        protocol_datas = {}
        for protocol_id in protocol_ids:
            # retrieve protocol definitions
            headers = {"X-CDD-token": self.vault_token}
            url = f"{self.base_url}/{self.vault_num}/protocols/{protocol_id}"
            response = requests.get(url, headers=headers)

            protocol_defs[protocol_id] = response.json()

            urls[protocol_id] = f"{url}/data?async=True"

        if molecules:
            url = f"{self.base_url}/{self.vault_num}/molecules?async=True&no_structures=True"
            urls["molecules"] = url

        responses = self._get_async_exports(urls, headers=headers)

        if molecules:
            molecule_data = responses.pop("molecules").json()

        for protocol_id, response in responses.items():
            protocol_datas[protocol_id] = response.json()

        protocol_results = {
            protocol_id: (protocol_defs[protocol_id], protocol_datas[protocol_id])
            for protocol_id in protocol_ids
        }

        if molecules:
            return protocol_results, molecule_data
        else:
            return protocol_results, None

    def retrieve_protocol_data(
        self, protocol_ids: List[str], molecules: bool = False, return_raw: bool = False
    ) -> Union[dict, Tuple[dict], None]:
        """Retrieve full definitions and data records for the given protocol ids.

        If `molecules=True`, then also retrieve full molecule data.

        Parameters
        ----------
        protocol_ids
            List of protocol ids to retrieve definitions and data records for.
        molecules
            If `True`, retrieve molecule data along with data for given protocol ids.
        return_raw
            If `True`, directly return all data structures retrieved in
            addition to writing them to `data_dir`.

        Returns
        -------
        results
            If `return_raw=True`, then a dict with protocol ids as keys, definitions and data as values is returned.
            If `return_raw=True` and `molecules=True`, then a tuple with the above, followed by the molecule data data is returned.
            Otherwise, `None` is returned.

        """
        results = self._get_protocol_data(protocol_ids, molecules=molecules)

        if molecules:
            protocol_results, molecule_data = results

            with open(self.data_dir.joinpath("molecules.json"), "w") as f:
                json.dump(molecule_data, f)
        else:
            protocol_results, _ = results

        for protocol_id, protocol_result in protocol_results.items():
            protocol_defs, protocol_data = protocol_result

            protocol_dir = self.protocols_dir.joinpath(protocol_id)
            protocol_dir.mkdir(parents=True, exist_ok=True)

            # TODO: add timestamp as metadata layer
            with open(protocol_dir.joinpath("protocol-defs.json"), "w") as f:
                json.dump(protocol_defs, f)

            with open(protocol_dir.joinpath("protocol-data.json"), "w") as f:
                json.dump(protocol_data, f)

        if return_raw and molecules:
            return protocol_results, molecule_data
        elif return_raw:
            return protocol_results

    def _get_molecule_data(self):
        url = (
            f"{self.base_url}/{self.vault_num}/molecules?async=True&no_structures=True"
        )
        headers = {"X-CDD-token": self.vault_token}
        response = self._get_async_exports({"molecules": url}, headers=headers)

        molecules = response["molecules"].json()

        return molecules

    def retrieve_molecule_data(self, return_raw=False):
        """Retrieve full molecule data.

        Parameters
        ----------
        return_raw
            If `True`, directly return molecule data in addition to writing to `data_dir`.

        Returns
        -------
        results
            If `return_raw=True`, then a dict with molecule data is returned.
            Otherwise, `None` is returned.

        """
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # TODO: add timestamp as metadata layer
        datadict = self._get_molecule_data()
        with open(self.data_dir.joinpath("molecules.json"), "w") as f:
            json.dump(datadict, f)

        if return_raw:
            return datadict

    def _generate_experimental_compound_data_molten(
            self,
            protocol_ids: List[str],
            return_map=False):

        # create a dictionary of all selected protocol data, each a dict of
        # keyed by (molecule id, batch id), with values giving raw readout data
        # with definitions merged in
        protocol_data_ns = {}
        protocol_name_id_map = {}
        for protocol_id in protocol_ids:
            protocol_dir = self.protocols_dir.joinpath(protocol_id)

            with open(self.data_dir.joinpath("molecules.json"), "r") as f:
                molecules = json.load(f)

            with open(protocol_dir.joinpath("protocol-defs.json"), "r") as f:
                protocol_defs = json.load(f)

            with open(protocol_dir.joinpath("protocol-data.json"), "r") as f:
                protocol_data = json.load(f)

            readout_definitions = {
                rdef["id"]: rdef for rdef in protocol_defs["readout_definitions"]
            }
            protocol_name = protocol_defs["name"]
            protocol_name_id_map[protocol_id] = protocol_name

            protocol_data_n = defaultdict(list)

            for record in protocol_data["objects"]:
                if "molecule" not in record:
                    continue

                record_n = dict()
                for readout in record["readouts"]:
                    readout_definition = readout_definitions[int(readout)]
                    readout_def_data = {key: readout_definition.get(key)
                            for key in ('name', 'unit_label', 'aggregation', 'precision_type', 'precision_number')}

                    record_n[readout_def_data['name']] = record["readouts"][readout].copy()

                    # add readout definition information to record
                    record_n[readout_def_data['name']].update(readout_def_data)

                protocol_data_n[(record["molecule"], record["batch"])].append(record_n)

            protocol_data_ns[protocol_name] = protocol_data_n

        # create experimental data object that merges protocol data with
        # molecule data into a molten object that can be transformed further
        # downstream
        compound_datas = []
        for mol in molecules["objects"]:
            for batch in mol["batches"]:
                experimental_data = {}
                for protocol_name, protocol_data_n in protocol_data_ns.items():
                    if (mol["id"], batch["id"]) in protocol_data_n:
                        rec_readouts = protocol_data_n[(mol["id"], batch["id"])]
                    else:
                        continue

                    experimental_data.update({protocol_name: rec_readouts})

                if experimental_data:
                    compound_data = dict(
                        compound_id=batch["batch_fields"]["External ID"],
                        suspected_smiles=batch["batch_fields"].get("suspected_SMILES"),
                        experimental_data=experimental_data,
                    )

                    compound_datas.append(compound_data)

        ecd = dict(compounds=compound_datas)

        if return_map:
            return ecd, protocol_name_id_map
        else:
            return ecd

    def generate_experimental_compound_data(
        self, protocol_ids: List[str]
    ) -> ExperimentalCompoundDataUpdate:
        """Generate `ExperimentalCompoundData` from current molecule and selected protocol data.

        Parameters
        ----------
        protocol_ids
            List of protocol ids to retrieve definitions and data records for.

        Returns
        -------
        experimental_compound_data
            A data structure giving a list of compound metadata, each having at least a compound id
            and if present among the given protocol data a dictionary of experimental data

        """
        exp_data_molten, protocol_name_id_map = self._generate_experimental_compound_data_molten(protocol_ids, return_map=True)
        ecds = list()
        
        for compound_data in exp_data_molten['compounds']:
            ecd = dict()

            ecd['compound_id'] = compound_data['compound_id']

            suspected_smiles = compound_data['suspected_smiles']
            if suspected_smiles is None:
                ecd['smiles'] = None
            else:
                smiles = suspected_smiles.split()[0]
                ecd['smiles'] = self._canonicalize_smiles(smiles)

                achiral = self._achiral(smiles)

                if achiral:
                    ecd['achiral'] = True
                else:
                    stereochemistry_certain = self._stereochemistry_is_certain(smiles)
                    if len(suspected_smiles.split()) > 1 and stereochemistry_certain:
                        ecd['relative_stereochemistry_enantiomerically_pure'] = True
                    elif stereochemistry_certain:
                        ecd['relative_stereochemistry_enantiomerically_pure'] = True
                    else:
                        ecd['racemic'] = True

            # extract and add in experimental data
            ecd['experimental_data'] = self._experimental_data_from_molten(
                    compound_data['experimental_data'], protocol_name_id_map, protocol_ids)

            ecds.append(
                ExperimentalCompoundData(**ecd)
            )

        return ExperimentalCompoundDataUpdate(compounds=ecds)

    def _canonicalize_smiles(self, smiles):
        from openeye import oechem

        oemol = oechem.OEGraphMol()
        oechem.OESmilesToMol(oemol, smiles)
        canonicalized_smiles = oechem.OEMolToSmiles(oemol)

        return canonicalized_smiles

    def _achiral(self, smiles):
        """Return True of compound is achiral.

        """
        from rdkit import Chem

        rdmol = Chem.MolFromSmiles(smiles)
        chiral_centers = Chem.FindMolChiralCenters(rdmol, includeUnassigned=True, useLegacyImplementation=False)
        
        return len(chiral_centers) == 0

    def _stereochemistry_is_certain(self, smiles):
        """
        Return True if `smiles` resolves to only a single isomer.
    
        Examples:
        "CNC(=O)CN1Cc2ccc(Cl)cc2[C@@]2(CCN(c3cncc4c3CCCC4)C2=O)C1 |o1:14|" : compound is enantiopure, but stereochemistry is uncertain
        "CNC(=O)CN1Cc2ccc(Cl)cc2[C@@]2(CCN(c3cncc4c3CCCC4)C2=O)C1" : compound is enantiopure, stereochemistry is certain
        "CNC(=O)CN1Cc2ccc(Cl)cc2[C]2(CCN(c3cncc4c3CCCC4)C2=O)C1" : compound is racemic

        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    
        rdmol = Chem.MolFromSmiles(smiles)
        smi_list = []
        opts = StereoEnumerationOptions(unique=True)
        isomers = tuple(EnumerateStereoisomers(rdmol, options=opts))
        for smi in sorted(Chem.MolToSmiles(isomer, isomericSmiles=True) for isomer in isomers):
            smi_list.append(smi)
    
        return len(smi_list) == 1

    def _experimental_data_from_molten(self, experimental_data, protocol_name_id_map, protocol_ids):
        processed = {}
        for protocol_id in protocol_ids:
            protocol_name = protocol_name_id_map[protocol_id]

            records = experimental_data[protocol_name]
            processed.update(getattr(self, self._protocol_processing_map[protocol_id])(records))

        return processed

    def _process_fluorescence_IC50(self, records):
        """Hardcodes and readout processing specific to IC50 protocol data.

        """
        processed = {}

        # we can do this because IC50 is a batch aggregation over all the
        # records; each one has the same value
        # TODO; add inline check and log message for when above is *not true*
        record = records[0]

        # convert to pIC50
        # some records lack CI bounds entirely
        # some records have no value for IC50, with 'note: could not be calculated'
        if 'IC50' in record:
            IC50 = record['IC50'].get('value')
            if IC50:
                processed['pIC50'] = - np.log10(IC50)

        if 'IC50 CI (Lower)' in record:
            IC50_lower = record['IC50 CI (Lower)'].get('value')
            if IC50_lower:
                processed['pIC50_lower'] = - np.log10(IC50_lower)

        if 'IC50 CI (Upper)' in record:
            IC50_upper = record['IC50 CI (Upper)'].get('value')
            if IC50_upper:
                processed['pIC50_upper'] = - np.log10(IC50_upper)

        return processed

    def clear(self, protocols: bool = True, molecules: bool = True):
        """Clear local data.

        Parameters
        ----------
        protocols
            If `True`, clear all local protocol data.
        molecules
            If `True`, clear all local molecule data.

        """

        if protocols:
            shutil.rmtree(self.protocols_dir, ignore_errors=True)

        if molecules:
            os.remove(self.data_dir / "molecules.json")
