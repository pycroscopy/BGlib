"""
Translator used to go from .ibw igor files to USID HDF5 files.

Created on Fri Sept 18 2020

@author: Nicole Creange
"""


from .igor_ibw import IgorIBWTranslator
from .trkpfm import TRKPFMTranslator


__all__ = ['IgorIBWTranslator','TRKPFMTranslator']
