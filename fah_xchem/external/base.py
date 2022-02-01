import os
from pathlib import Path

from pydantic import BaseModel, Field
import requests
from rich.progress import track


class ExternalData(BaseModel):
    """Mixin class for external data interfaces."""

    data_dir: Path = Field(
        ...,
        description="Data directory; retrieved data will be deposited here, can be pre-existing",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # create data directory if not present
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _download_url(url, save_path, headers=None, chunk_size=128, message=None):
        """
        Download file from the specified URL to the specified file path, creating base dirs if needed.
        """
        # Create directory
        base_path, filename = os.path.split(save_path)
        os.makedirs(base_path, exist_ok=True)

        r = requests.get(url, stream=True, headers=headers)
        with open(save_path, "wb") as fd:
            nchunks = int(int(r.headers["Content-Length"]) / chunk_size)
            for chunk in track(
                r.iter_content(chunk_size=chunk_size), message, total=nchunks
            ):
                fd.write(chunk)
