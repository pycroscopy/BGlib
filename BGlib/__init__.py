"""
The BGlib package.

Submodules
----------

.. autosummary::
    :toctree: _autosummary
"""
from .__version__ import version as __version__
from . import be, gmode, trKPFM
from . import bglib_fitter, bglib_guesser, bglib_process

__all__ = ['__version__', 'be', 'gmode', 'trKPFM', 'bglib_guesser', 'bglib_fitter', 'bglib_process']

