"""
The BGlib package.
Submodules
----------
.. autosummary::
    :toctree: _autosummary
"""
from importlib import import_module

from .__version__ import version as __version__

__all__ = ['__version__', 'be', 'gmode', 'misc']


def __getattr__(name):
    if name in {'be', 'gmode', 'misc'}:
        module = import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
