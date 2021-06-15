"""
Physical or chemical model-based analysis of data

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    Fitter
    BESHOfitter
    BELoopFitter
    BERelaxFit
    utils

"""

from . import utils
from .be_sho_fitter import BESHOfitter
from .be_relax_fit import BERelaxFit
from .bglib_fitter import LoopFitter

__all__ = ['BESHOfitter', 'utils', 'BERelaxFit','LoopFitter']
