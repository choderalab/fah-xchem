from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..schema import CompoundSeriesMetadata, Compound, Transformation, CompoundMetadata


class CompoundSeriesUpdate(BaseModel):
    """Permissive version of `CompoundSeries`.

    Can include any set of fields as needed to update a CompoundSeries.

    """
    metadata: CompoundSeriesMetadata = None
    compounds: List[Compound] = None
    transformations: List[Transformation] = None


class CompoundSeries(BaseModel):
    """A compound series; the core object of operation in fah-xchem.

    """
    metadata: CompoundSeriesMetadata = Field(
        ...,
        description="Metadata for the whole compound series"
    )
    compounds: List[Compound] = Field(
        ...,
        description="List of compounds comprising the compound series"
    )
    transformations: List[Transformation] = Field(
        ...,
        description="List of transformations performed between compounds within the compound series"
    )

    def update_experimental_data(
            self,
            metadata: List[CompoundMetadata],
        ) -> None:
        """Update experimental data for compounds in the CompoundSeries.

        Compound instances in `self.compounds` will be replaced as necessary.

        Parameters
        ----------
        metadata
            List of CompoundMetadata objects to apply to CompoundSeries.

        """

        metadata_d = {i.compound_id: i for i in metadata}

        new_compounds = []
        for compound in self.compounds:
            metadata_update = metadata_d.get(compound.metadata.compound_id)

            if metadata_update: 
                metadata_current = compound.metadata.dict()
                metadata_current['experimental_data'].update(metadata_update.experimental_data)

                compound_current = compound.dict()
                compound_current['metadata'] = metadata_current

                new_compounds.append(Compound(**compound_current))
            else:
                new_compounds.append(compound)

        self.compounds = new_compounds
