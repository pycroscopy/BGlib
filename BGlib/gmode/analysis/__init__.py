"""
Physical or chemical model-based analysis of data

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    GIVBayesian

"""

from . import utils
from .giv_bayesian import GIVBayesian

__all__ = ['GIVBayesian', 'utils']
