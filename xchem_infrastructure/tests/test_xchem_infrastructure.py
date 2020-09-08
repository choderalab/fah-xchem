"""
Unit and regression test for the xchem_infrastructure package.
"""

# Import package, test suite, and other packages as needed
import xchem_infrastructure
import pytest
import sys

def test_xchem_infrastructure_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "xchem_infrastructure" in sys.modules
