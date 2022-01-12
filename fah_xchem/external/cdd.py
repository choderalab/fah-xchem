"""Interface to Collaborative Drug Discovery APIs and data artifacts.

"""
import os
import time
import logging
import json
import shutil
from typing import List, Dict
from collections import defaultdict

import requests
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .base import ExternalData
from ..schema import CompoundMetadata, ExperimentalCompoundData


class FailedExportError(Exception):
    """Export failed to finish within timeout.

    """


class CDDData(ExternalData):
    """Collaborative Drug Discovery (CDD) data interface.

    Sourced from: https://github.com/postera-ai/COVID_moonshot_submissions/blob/fixing_stereochem/lib/get_experimental_data.py

    """

    base_url: str = Field(
        "https://app.collaborativedrug.com/api/v1/vaults",
        description="Base URL for CCD Vault API"
    )
    vault_token: str = Field(
        ...,
        description="API token for access to CDD Vault"
    )
    vault_num: str = Field(
        "5549", 
        description="CDD Vault number"
    )
    fluorescence_IC50_protocol_id: str = Field(
        "49439",
        description="Protocol ID for fluorescence measurements"
    )

    @property
    def protocols_dir(self):
        return self.data_dir.joinpath('protocols')

    def get_available_protocols(self):
        # retrieve protocol definitions
        headers = {"X-CDD-token": self.vault_token}
        
        url = f"{self.base_url}/{self.vault_num}/protocols/"
        url = f"{url}?async=True"

        response = self._get_async_exports({"protocols": url}, headers=headers)["protocols"]
        
        protocol_data = response.json()
        
        return protocol_data

    def _get_async_exports(
            self, 
            async_urls: Dict[str, str],
            headers: Dict[str, str],
            timeout: int = 3600
        ):
        from rich.live import Live
        from rich.text import Text
        from rich.console import Console

        responses = {}
        for name, async_url in async_urls.items():
            responses[name] = requests.get(async_url, headers=headers)

        console = Console()
        text = Text("Retrieving urls...", no_wrap=True)
        for async_url in async_urls.values():
            text.append(f"\n: {async_url}")

        console.print(text)

        logging.info("CDDData : Beginning export(s)")
        text = Text("CDDData : Beginning export(s)", style="bold red", no_wrap=True)

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
                    url = f"{self.base_url}/{self.vault_num}/export_progress/{export_id}"
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
                raise FailedExportError(f"The following urls failed to deliver an export within the timeout: {failed}")
    
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
            molecule_data = responses.pop('molecules').json()

        for protocol_id, response in responses.items():
            protocol_datas[protocol_id] = response.json()


        protocol_results = {protocol_id: (protocol_defs[protocol_id],
                                          protocol_datas[protocol_id])
                            for protocol_id in protocol_ids}

        if molecules:
            return protocol_results, molecule_data
        else:
            return protocol_results, None
    
    def retrieve_protocol_data(self, protocol_ids, molecules=False, return_raw=False):

        results = self._get_protocol_data(protocol_ids, molecules=molecules)

        if molecules:
            protocol_results, molecule_data = results
            self.data_dir.mkdir(parents=True, exist_ok=True)

            with open(self.data_dir.joinpath('molecules.json'), 'w') as f:
                json.dump(molecule_data, f)
        else:
            protocol_results, _ = results

        for protocol_id, protocol_result in protocol_results.items():
            protocol_defs, protocol_data = protocol_result

            protocol_dir = self.protocols_dir.joinpath(protocol_id)
            protocol_dir.mkdir(parents=True, exist_ok=True)

            with open(protocol_dir.joinpath('protocol-defs.json'), 'w') as f:
                json.dump(protocol_defs, f)
                
            with open(protocol_dir.joinpath('protocol-data.json'), 'w') as f:
                json.dump(protocol_data, f)

        if return_raw and molecules:
            return protocol_results, molecule_data
        elif return_raw:
            return protocol_results

    def _get_molecule_data(self):
        url = f"{self.base_url}/{self.vault_num}/molecules?async=True&no_structures=True"
        headers = {"X-CDD-token": self.vault_token}
        response = self._get_async_exports({"molecules": url}, headers=headers)

        molecules = response["molecules"].json()
    
        return molecules 
    
    def retrieve_molecule_data(self, return_raw=False):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        datadict = self._get_molecule_data()
        with open(self.data_dir.joinpath('molecules.json'), 'w') as f:
            json.dump(datadict, f)

        if return_raw:
            return datadict

    def generate_experimental_compound_data(self, protocol_ids):
        protocol_data_ns = []
        for protocol_id in protocol_ids:
            protocol_dir = self.protocols_dir.joinpath(protocol_id)

            with open(self.data_dir.joinpath('molecules.json'), 'r') as f:
                molecules = json.load(f)

            with open(protocol_dir.joinpath('protocol-defs.json'), 'r') as f:
                protocol_defs = json.load(f)
                
            with open(protocol_dir.joinpath('protocol-data.json'), 'r') as f:
                protocol_data = json.load(f)

            readout_definitions = {rdef['id']: rdef for rdef in protocol_defs['readout_definitions']}

            protocol_data_n = defaultdict(dict)
            
            for record in protocol_data['objects']:
                if 'molecule' not in record:
                    continue
                
                for readout in record['readouts']:
                    record['readouts'][readout]['name'] = readout_definitions[int(readout)]['name']
                
                protocol_data_n[record['molecule']][record['batch']] = record

            protocol_data_ns.append(protocol_data_n)
            
        compound_metadatas = []
        for mol in molecules['objects']:
            for batch in mol['batches']:
                experimental_data = {}
                for protocol_data_n in protocol_data_ns:
                    if mol['id'] in protocol_data_n:
                        if batch['id'] in protocol_data_n[mol['id']]:
                            readouts = protocol_data_n[mol['id']][batch['id']]['readouts']
                        else:
                            continue
                    else:
                        continue
                            
                    experimental_data.update({readout['name']: readout for readout in readouts.values()})
                    
                    # remove redundant 'name' field from each readout
                    for readout in experimental_data.values():
                        readout.pop('name')
                
                compound_metadata = CompoundMetadata(
                    compound_id=batch['batch_fields']['External ID'],
                    experimental_data=experimental_data)
                
                compound_metadatas.append(compound_metadata)

        ecd = ExperimentalCompoundData(compounds=compound_metadatas)

        return ecd


    def clear(self, protocols=True, molecules=True):
        """Clear local data

        """

        if protocols:
            shutil.rmtree(self.protocols_dir)

        if molecules:
            os.remove(self.data_dir / 'molecules.json')
