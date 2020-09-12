from typing import Optional
from ..schema import (
    AnalysisConfig,
    CompoundSeries,
    CompoundSeriesAnalysis,
    ServerConfig,
)


def analyze_compound_series(
    compound_series: CompoundSeries,
    config: AnalysisConfig,
    server_config: ServerConfig,
    cache_dir: Optional[str] = None,
    num_procs: Optional[int] = None,
) -> CompoundSeriesAnalysis:
    return CompoundSeriesAnalysis(
        metadata=compound_series.metadata, compounds=[], transformations=[]
    )
