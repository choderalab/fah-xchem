"""
Unit and regression test for the fah_xchem package.
"""

# Import package, test suite, and other packages as needed
import fah_xchem
import pytest
import sys

def test_fah_xchem_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "fah_xchem" in sys.modules
