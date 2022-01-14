"""
The BGlib package.
Submodules
----------
.. autosummary::
    :toctree: _autosummary
"""
from .__version__ import version as __version__
from . import be, gmode, misc

__all__ = ['__version__', 'be', 'gmode', 'misc']