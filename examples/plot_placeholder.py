"""
===================================
Hello World
===================================

**Suhas Somnath**

8/28/2020

**UNDER CONSTRUCTION**
"""

# The package for accessing files in directories, etc.:
import importlib
import os
import zipfile

# Warning package in case something goes wrong
from warnings import warn
import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
# Package for downloading online files:
try:
    # This package is not part of anaconda and may need to be installed.
    importlib.import_module('wget')
except ImportError:
    warn('wget not found.  Will install with pip.')
    install('wget')
    importlib.import_module('wget')

# The mathematical computation package:
import numpy as np

# The package used for creating and manipulating HDF5 files:
import h5py

# import sidpy - supporting package for pyUSID:
try:
    importlib.import_module('sidpy')
except ImportError:
    warn('sidpy not found.  Will install with pip.')
    install('sidpy')
    importlib.import_module('sidpy')

###############################################################################
# Better documentation to follow
# ==============================
# Here we will download a compressed data file from Github and unpack it:
print('Hello World')
