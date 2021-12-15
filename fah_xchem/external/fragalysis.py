"""Interface to Fragalysis APIs and data artifacts.

"""

from zipfile import ZipFile

from pydantic import Field

from .base import ExternalData

class FragalysisData(ExternalData):

    structures_url: str = Field(
        None,
        description="URL to target structures (e.g. 'https://fragalysis.diamond.ac.uk/media/targets/Mpro.zip')"
    )

    @property
    def target_data_dir(self):
        return self.data_dir.joinpath('targets')

    def retrieve_target_structures(self) -> None:

        zip_path = self.data_dir.joinpath('target.zip')
        self._download_url(self.structures_url, zip_path, message="Downloading ZIP archive of Fragalysis structures")

        with ZipFile(zip_path, 'r') as zip_obj:
            zip_obj.extractall(self.target_data_dir)
        zip_path.unlink()


