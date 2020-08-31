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
from .fitter import Fitter
from .be_sho_fitter import BESHOfitter
from .be_loop_fitter import BELoopFitter
from .be_relax_fit import BERelaxFit

__all__ = ['Fitter', 'BESHOfitter', 'BELoopFitter', 'utils', 'BERelaxFit']
