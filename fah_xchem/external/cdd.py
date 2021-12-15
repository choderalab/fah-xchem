"""Interface to Collaborative Drug Discovery APIs and data artifacts.

"""
import sys
import time
import logging
import json

import requests
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .base import ExternalData


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
    rapidfire_IC50_protocol_id: str = Field(
        "49700",
        description="Protocol ID for RapidFire assay IC50 measurements"
    )
    fluorescence_IC50_protocol_id: str = Field(
        "49439",
        description="Protocol ID for fluorescence measurements"
    )

    @property
    def rapidfire_data_dir(self):
        return self.data_dir.joinpath('rapidfire')

    @property
    def fluorescence_data_dir(self):
        return self.data_dir.joinpath('fluorescence')

    def _get_async_export(self, async_url):
        from rich.live import Live
        from rich.text import Text

        headers = {"X-CDD-token": self.vault_token}
        response = requests.get(async_url, headers=headers)
        logging.info("BEGINNING EXPORT")
        text = Text("BEGINNING EXPORT", style="bold red")

        with Live(text, refresh_per_second=4) as live:

            export_info = response.json()
            export_id = export_info["id"]
    
            # CHECK STATUS of Export
            status = None
            seconds_waiting = 0

            first = True
            while status != "finished":
                text.remove_suffix(" --> CHECKING STATUS OF EXPORT")
                headers = {"X-CDD-token": self.vault_token}
                url = f"{self.base_url}/{self.vault_num}/export_progress/{export_id}"
    
                response = requests.get(url, headers=headers)
    
                if first:
                    logging.info("CHECKING STATUS OF EXPORT")
                text.append(" --> CHECKING STATUS OF EXPORT", style="bold yellow")

                # to view the status, use:
                status = response.json()["status"]
    
                time.sleep(5)
                seconds_waiting += 5
                if seconds_waiting > 5000:
                    logging.info("Export Never Finished")
                    break

                first = False
    
            if status != "finished":
                sys.exit("EXPORT IS BROKEN")
    
            headers = {"X-CDD-token": self.vault_token}
            url = f"{self.base_url}/{self.vault_num}/exports/{export_id}"
    
            logging.info("RETRIEVING FINISHED EXPORT")
            text.append(" --> RETRIEVING FINISHED EXPORT", style="bold green")
            response = requests.get(url, headers=headers)
            text.append(" --> DONE", style="bold blue")

        return response

    def get_rapidfire_IC50_data(self):
        url = f"{self.base_url}/{self.vault_num}/protocols/{self.rapidfire_IC50_protocol_id}/data?async=True"
        response = self._get_async_export(url)
        rapidfire_dose_response_dict = response.json()["objects"]
    
        mol_id_list = []
        avg_ic50_list = []
        max_reading_list = []
        min_reading_list = []
        hill_slope_list = []
        r2_list = []
        concentration_list = []
        inhibition_list = []
        curve_ic50_list = []
    
        curve_dict = {}
        for mol_dict in rapidfire_dose_response_dict:
            if "molecule" not in mol_dict:
                continue
    
            if "564286" not in mol_dict["readouts"]:
                continue
    
            if "564283" not in mol_dict["readouts"]:
                continue
    
            if "564285" not in mol_dict["readouts"]:
                continue
    
            mol_id = mol_dict["molecule"]
            if mol_id not in curve_dict:
                curve_dict[mol_id] = {}
    
            run = mol_dict["run"]
            if run not in curve_dict[mol_id]:
                if type(mol_dict["readouts"]["564285"]) != dict:
                    curve_dict[mol_id][run] = {
                        "concentration_um": [mol_dict["readouts"]["564283"]],
                        "percent_inhibition": [mol_dict["readouts"]["564285"]],
                    }
                else:
                    curve_dict[mol_id][run] = {
                        "concentration_um": [mol_dict["readouts"]["564283"]],
                        "percent_inhibition": [mol_dict["readouts"]["564285"]["value"]],
                    }
                if "564286" in mol_dict["readouts"]:
                    if type(mol_dict["readouts"]["564286"]) == dict:
                        if "modifier" in mol_dict["readouts"]["564286"]:
                            ic50 = 99
                        elif ("overridden_intercept" in mol_dict["readouts"]["564286"]) or (
                            "note" in mol_dict["readouts"]["564286"]
                            and "could not be calculated"
                            in mol_dict["readouts"]["564286"]["note"]
                        ):
                            ic50 = np.nan
                        else:
                            ic50 = mol_dict["readouts"]["564286"]["value"]
    
                    else:
                        ic50 = mol_dict["readouts"]["564286"]
                else:
                    ic50 = np.nan
    
                if "564291" in mol_dict["readouts"]:
                    min_reading = mol_dict["readouts"]["564291"]
                else:
                    min_reading = np.nan
    
                if "564292" in mol_dict["readouts"]:
                    max_reading = mol_dict["readouts"]["564292"]
                else:
                    max_reading = np.nan
    
                if "564290" in mol_dict["readouts"]:
                    hill_slope = mol_dict["readouts"]["564290"]
                else:
                    hill_slope = np.nan
    
                if "564294" in mol_dict["readouts"]:
                    r2 = mol_dict["readouts"]["564294"]
                else:
                    r2 = np.nan
    
                curve_dict[mol_id][run]["r_avg_IC50"] = ic50
                curve_dict[mol_id][run]["r_max_inhibition_reading"] = max_reading
                curve_dict[mol_id][run]["r_min_inhibition_reading"] = min_reading
                curve_dict[mol_id][run]["r_hill_slope"] = hill_slope
                curve_dict[mol_id][run]["r_R2"] = r2
                curve_dict[mol_id][run]["r_IC50"] = ic50
            else:
                curve_dict[mol_id][run]["concentration_um"] = curve_dict[mol_id][run][
                    "concentration_um"
                ] + [mol_dict["readouts"]["564283"]]
                if type(mol_dict["readouts"]["564285"]) != dict:
                    curve_dict[mol_id][run]["percent_inhibition"] = curve_dict[mol_id][run][
                        "percent_inhibition"
                    ] + [mol_dict["readouts"]["564285"]]
                else:
                    curve_dict[mol_id][run]["percent_inhibition"] = curve_dict[mol_id][run][
                        "percent_inhibition"
                    ] + [mol_dict["readouts"]["564285"]["value"]]
    
        for mol in curve_dict:
            mol_id_list.append(mol)
            runs_avg_ic50_list = []
            runs_max_reading_list = []
            runs_min_reading_list = []
            runs_hill_slope_list = []
            runs_r2_list = []
            runs_concentration_list = []
            runs_inhibition_list = []
            runs_curve_ic50_list = []
            for run in curve_dict[mol]:
                runs_avg_ic50_list.append(curve_dict[mol][run]["r_avg_IC50"])
                runs_max_reading_list.append(
                    curve_dict[mol][run]["r_max_inhibition_reading"]
                )
                runs_min_reading_list.append(
                    curve_dict[mol][run]["r_min_inhibition_reading"]
                )
                runs_hill_slope_list.append(curve_dict[mol][run]["r_hill_slope"])
                runs_r2_list.append(curve_dict[mol][run]["r_R2"])
                runs_concentration_list.append(curve_dict[mol][run]["concentration_um"])
                runs_inhibition_list.append(curve_dict[mol][run]["percent_inhibition"])
                runs_curve_ic50_list.append(curve_dict[mol][run]["r_IC50"])
    
            avg_ic50_list.append(runs_avg_ic50_list)
            max_reading_list.append(runs_max_reading_list)
            min_reading_list.append(runs_min_reading_list)
            hill_slope_list.append(runs_hill_slope_list)
            r2_list.append(runs_r2_list)
            concentration_list.append(runs_concentration_list)
            inhibition_list.append(runs_inhibition_list)
            curve_ic50_list.append(runs_curve_ic50_list)
    
        rapidfire_df = pd.DataFrame(
            {
                "CDD_mol_ID": mol_id_list,
                "r_avg_IC50": [x[0] for x in avg_ic50_list],
                "r_curve_IC50": curve_ic50_list,
                "r_max_inhibition_reading": max_reading_list,
                "r_min_inhibition_reading": min_reading_list,
                "r_hill_slope": hill_slope_list,
                "r_R2": r2_list,
                "r_concentration_uM": concentration_list,
                "r_inhibition_list": inhibition_list,
            }
        )
        return rapidfire_df, rapidfire_dose_response_dict

    def retrieve_rapidfire_IC50_data(self):
        self.rapidfire_data_dir.mkdir(parents=True, exist_ok=True)

        df, datadict = self.get_rapidfire_IC50_data()
        df.to_csv(self.rapidfire_data_dir.joinpath('rapidfire.csv'), sep=',', header=True)

        with open(self.rapidfire_data_dir.joinpath('rapidfire.json'), 'w') as f:
            json.dump(datadict, f)

    def get_fluorescence_IC50_data(self):
        url = f"{self.base_url}/{self.vault_num}/protocols/{self.fluorescence_IC50_protocol_id}/data?async=True"
        response = self._get_async_export(url)
        fluorescence_response_dict = response.json()["objects"]
    
        mol_id_list = []
        avg_ic50_list = []
        avg_pic50_list = []
        max_reading_list = []
        min_reading_list = []
        hill_slope_list = []
        r2_list = []
        concentration_list = []
        inhibition_list = []
        curve_ic50_list = []
    
        curve_dict = {}
        for mol_dict in fluorescence_response_dict:
            if "molecule" not in mol_dict:
                continue
    
            mol_id = mol_dict["molecule"]
            if mol_id not in curve_dict:
                curve_dict[mol_id] = {}
    
            run = mol_dict["run"]
            if run not in curve_dict[mol_id]:
                curve_dict[mol_id][run] = {
                    "concentration_um": [mol_dict["readouts"]["557072"]],
                    "percent_inhibition": [mol_dict["readouts"]["557073"]],
                }
                if "557736" in mol_dict["readouts"]:
                    avg_ic50 = mol_dict["readouts"]["557736"]["value"]
                else:
                    avg_ic50 = np.nan
    
                if "557738" in mol_dict["readouts"]:
                    if type(mol_dict["readouts"]["557738"]) == dict:
                        avg_pic50 = np.nan
                    else:
                        avg_pic50 = mol_dict["readouts"]["557738"]
                else:
                    avg_pic50 = np.nan
    
                if "557085" in mol_dict["readouts"]:
                    min_reading = mol_dict["readouts"]["557085"]["value"]
                else:
                    min_reading = np.nan
    
                if "557086" in mol_dict["readouts"]:
                    max_reading = mol_dict["readouts"]["557086"]["value"]
                else:
                    max_reading = np.nan
    
                if "557078" in mol_dict["readouts"]:
                    hill_slope = mol_dict["readouts"]["557078"]
                else:
                    hill_slope = np.nan
    
                if "557082" in mol_dict["readouts"]:
                    r2 = mol_dict["readouts"]["557082"]
                else:
                    r2 = np.nan
    
                if "557074" in mol_dict["readouts"]:
                    if type(mol_dict["readouts"]["557074"]) == dict:
                        if avg_ic50 > 99:
                            curve_ic50 = 99
                        else:
                            curve_ic50 = np.nan
                    else:
                        curve_ic50 = mol_dict["readouts"]["557074"]
                else:
                    curve_ic50 = np.nan
    
                curve_dict[mol_id][run]["f_avg_IC50"] = avg_ic50
                curve_dict[mol_id][run]["f_avg_pIC50"] = avg_pic50
                curve_dict[mol_id][run]["f_max_inhibition_reading"] = max_reading
                curve_dict[mol_id][run]["f_min_inhibition_reading"] = min_reading
                curve_dict[mol_id][run]["f_hill_slope"] = hill_slope
                curve_dict[mol_id][run]["f_R2"] = r2
                curve_dict[mol_id][run]["f_IC50"] = curve_ic50
            else:
                if ("557072" in mol_dict["readouts"]) and ("557073" in mol_dict["readouts"]):
                    curve_dict[mol_id][run]["concentration_um"] = curve_dict[mol_id][run][
                        "concentration_um"
                    ] + [mol_dict["readouts"]["557072"]]
                    curve_dict[mol_id][run]["percent_inhibition"] = curve_dict[mol_id][run][
                        "percent_inhibition"
                    ] + [mol_dict["readouts"]["557073"]]
                else:
                    continue
    
        for mol in curve_dict:
            mol_id_list.append(mol)
            runs_avg_ic50_list = []
            runs_avg_pic50_list = []
            runs_max_reading_list = []
            runs_min_reading_list = []
            runs_hill_slope_list = []
            runs_r2_list = []
            runs_concentration_list = []
            runs_inhibition_list = []
            runs_curve_ic50_list = []
            for run in curve_dict[mol]:
                runs_avg_ic50_list.append(curve_dict[mol][run]["f_avg_IC50"])
                runs_avg_pic50_list.append(curve_dict[mol][run]["f_avg_pIC50"])
                runs_max_reading_list.append(
                    curve_dict[mol][run]["f_max_inhibition_reading"]
                )
                runs_min_reading_list.append(
                    curve_dict[mol][run]["f_min_inhibition_reading"]
                )
                runs_hill_slope_list.append(curve_dict[mol][run]["f_hill_slope"])
                runs_r2_list.append(curve_dict[mol][run]["f_R2"])
                runs_concentration_list.append(curve_dict[mol][run]["concentration_um"])
                runs_inhibition_list.append(curve_dict[mol][run]["percent_inhibition"])
                runs_curve_ic50_list.append(curve_dict[mol][run]["f_IC50"])
    
            avg_ic50_list.append(runs_avg_ic50_list)
            avg_pic50_list.append(runs_avg_pic50_list)
            max_reading_list.append(runs_max_reading_list)
            min_reading_list.append(runs_min_reading_list)
            hill_slope_list.append(runs_hill_slope_list)
            r2_list.append(runs_r2_list)
            concentration_list.append(runs_concentration_list)
            inhibition_list.append(runs_inhibition_list)
            curve_ic50_list.append(runs_curve_ic50_list)
    
        fluorescence_df = pd.DataFrame(
            {
                "CDD_mol_ID": mol_id_list,
                "f_avg_IC50": [x[0] for x in avg_ic50_list],
                "f_avg_pIC50": [x[0] for x in avg_pic50_list],
                "f_curve_IC50": curve_ic50_list,
                "f_max_inhibition_reading": max_reading_list,
                "f_min_inhibition_reading": min_reading_list,
                "f_hill_slope": hill_slope_list,
                "f_R2": r2_list,
                "f_concentration_uM": concentration_list,
                "f_inhibition_list": inhibition_list,
            }
        )
        return fluorescence_df, fluorescence_response_dict
    
    def retrieve_fluorescence_IC50_data(self):
        self.fluorescence_data_dir.mkdir(parents=True, exist_ok=True)

        df, datadict = self.get_fluorescence_IC50_data()
        df.to_csv(self.fluorescence_data_dir.joinpath('fluorescence.csv'), sep=',', header=True)

        with open(self.fluorescence_data_dir.joinpath('fluorescence.json'), 'w') as f:
            json.dump(datadict, f)

    def get_current_vault_data(self):
        url = f"{self.base_url}/{self.vault_num}/molecules?async=True&no_structures=True"
        response = self._get_async_export(url)

        current_mols = response.json()
    
        mol_ids = []
        cdd_names = []
        batch_ids = []
        external_ids = []
        canonical_ids = []
        virtual_list = []
        for_synthesis_list = []
        made_list = []
    
        for mol in current_mols['objects']:
            try:
                for i in range(len(mol['batches'])):
                    cdd_names.append(mol['name'])
                    batch_ids.append(mol['batches'][i]['id'])
                    external_ids.append(mol['batches'][i]['batch_fields']['External ID'])
                    if 'Canonical PostEra ID' in mol['batches'][i]['batch_fields']:
                        canonical_ids.append(mol['batches'][i]['batch_fields']['Canonical PostEra ID'])
                    else:
                        canonical_ids.append(np.nan)
                    mol_ids.append(mol['id'])
                    project_names = []
                    for project in mol['batches'][i]['projects']:
                        project_names.append(project['name']) 
                    if 'Compounds_Virtual' in project_names:
                        virtual_list.append(True)
                    else:
                        virtual_list.append(False)
                    if 'Compounds_for Synthesis' in project_names:
                        for_synthesis_list.append(True)
                    else:
                        for_synthesis_list.append(False)
                    if 'Compounds_Made' in project_names:
                        made_list.append(True)
                    else:
                        made_list.append(False)
            except Exception as e:
                print(e)
                pass
    
        current_cdd_df = pd.DataFrame(
            {
                "external_ID": external_ids,
                "CDD_name": cdd_names,
                "molecule_ID": mol_ids,
                "batch_ID": batch_ids,
                "canonical_CID": canonical_ids,
                "virtual_project": virtual_list,
                "for_synthesis_project": for_synthesis_list,
                "made_project": made_list,
            }
        )
        return current_cdd_df, current_mols
    
    def retrieve_current_vault_data(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        df, datadict = self.get_current_vault_data()
        df.to_csv(self.data_dir.joinpath('molecules.csv'), sep=',', header=True)

        with open(self.data_dir.joinpath('molecules.json'), 'w') as f:
            json.dump(datadict, f)

    def generate_compound_series_update(self):
        ...
        # is there a way to use the CompoundSeries model to get the structure right, but only populate with the fields
        # metadata:[compound_id,experimental_data]?

