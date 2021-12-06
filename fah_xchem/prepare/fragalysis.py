import os
from pathlib import Path
from zipfile import ZipFile

import requests
from rich.progress import track
from pydantic import BaseModel, Field

from .constants import FRAGALYSIS_URL


class FragalysisData(BaseModel):

    structures_url: str = Field(
        None,
        description="URL to target structures, (e.g. 'https://fragalysis.diamond.ac.uk/media/targets/Mpro.zip')"
    )
    activity_url: str = Field(
        None,
        description="URL to activity data (e.g. '')"
    )
    data_dir: Path = Field(
        ...,
        description="Data directory; retrieved data will be deposited here, can be pre-existing")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_data_dir = self.data_dir.joinpath('targets')
        self.activity_data_dir = self.data_dir.joinpath('activity')

    @staticmethod
    def _download_url(url, save_path, chunk_size=128):
        """
        Download file from the specified URL to the specified file path, creating base dirs if needed.
        """
        # Create directory
        base_path, filename = os.path.split(save_path)
        os.makedirs(base_path, exist_ok=True)
    
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            nchunks = int(int(r.headers['Content-Length'])/chunk_size)
            for chunk in track(r.iter_content(chunk_size=chunk_size), 'Downloading ZIP archive of Mpro structures...',
                               total=nchunks):
                fd.write(chunk)
    
    def retrieve_target_structures(self) -> None:

        zip_path = self.data_dir.joinpath('target.zip')
        self._download_url(self.structures_url, zip_path)

        with ZipFile(zip_path, 'r') as zip_obj:
            zip_obj.extractall(self.data_dir)
        zip_path.unlink()


    def retrieve_activity_data(self) -> None: 
        ...
