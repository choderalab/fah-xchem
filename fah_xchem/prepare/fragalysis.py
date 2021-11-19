import os
from pathlib import Path
from zipfile import ZipFile

import requests
from rich.progress import track
from pydantic import BaseModel, Field

from .constants import FRAGALYSIS_URL

class FragalysisHarness:

    structures_url: str = Field(
        description="URL to target structures, (e.g. 'https://fragalysis.diamond.ac.uk/media/targets/Mpro.zip')"
    )

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
    
    
    def download_target_structures(self, structures_path: Path) -> None:
        # TODO: remove MPRO-specific reference
        zip_path = structures_path.joinpath('Mpro.zip')
        self._download_url(self.structures_url, zip_path)
        with ZipFile(zip_path, 'r') as zip_obj:
            zip_obj.extractall(structures_path)
        zip_path.unlink()
