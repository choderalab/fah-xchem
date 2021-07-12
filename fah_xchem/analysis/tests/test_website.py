"""Tests for website generation.

"""

import os
from datetime import datetime

import pytest

from fah_xchem.schema import TimestampedAnalysis
from fah_xchem.analysis.website import WebsiteArtifactory
from fah_xchem.data import get_compound_series_analysis_results


@pytest.fixture
def website_artifactory(tmpdir):

    with tmpdir.as_cwd():
        compound_series_analysis = get_compound_series_analysis_results()
        waf = WebsiteArtifactory(
                base_url="https://localhost:8080",
                path=os.getcwd(),
                series=compound_series_analysis,
                timestamp=datetime(2021, 7, 12, 0, 0, 0),
                fah_ws_api_url=None,
                )

    return waf


class TestWebsiteArtifactory:

    def test_init(self, website_artifactory):
        waf = website_artifactory

        assert waf.environment

    def test_generate_website(self):
        ...
    
    def test_generate_summary(self):
        ...

    def test_generate_compounds(self):
        ...

    def test_generate_microstates(self):
        ...

    def test_generate_transformations(self):
        ...

    def test_generate_retrospective_transformations(self):
        ...
