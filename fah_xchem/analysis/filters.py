"""Detailed-chemistry and molecular characterizers used for filtering analysis outputs.

"""

from ..schema import CompoundMicrostate, CompoundSeriesAnalysis


class Racemic:
    """Filter to determine whether a compound microstate belongs to a compound that was assayed as a racemic mixture."""

    def __init__(self, series: CompoundSeriesAnalysis):

        self.series = series

        # Transformations not involving racemates
        # TODO: Consider changing `series.compounds` to be a dict with `compound_id` as keys
        self.microstates = {
            compound.metadata.compound_id: compound.microstates
            for compound in series.compounds
        }

    def compound_microstate(
        self,
        compound_microstate: CompoundMicrostate,
    ):
        return (
            True
            if (len(self.microstates[compound_microstate.compound_id]) > 1)
            else False
        )
