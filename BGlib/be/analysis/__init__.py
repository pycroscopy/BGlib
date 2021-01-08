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
from BGlib.be.analysis.utils.fitter import Fitter
from .be_sho_fitter import BESHOfitter
from BGlib.be.analysis.utils.be_loop_fitter import BELoopFitter
from .be_relax_fit import BERelaxFit

__all__ = ['Fitter', 'BESHOfitter', 'BELoopFitter', 'utils', 'BERelaxFit']
