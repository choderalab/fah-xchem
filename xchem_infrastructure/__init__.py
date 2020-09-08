"""
xchem-infrastructure
Tools and infrastructure for automated compound discovery using Folding@home
"""

# Add imports here
from .xchem import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
