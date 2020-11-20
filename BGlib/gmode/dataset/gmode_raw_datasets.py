from pyUSID import USIDataset
from ..analysis import BESHOfitter
from ..analysis import be_sho_fitter as bsho
import os
import numpy as np
import matplotlib.pyplot as plt
from pyUSID.io.hdf_utils import reshape_to_n_dims
from sidpy.viz.plot_utils import plot_curves, plot_map_stack, get_cmap_object, plot_map, set_tick_font_size, \
    plot_complex_spectra
from IPython.display import display
from sidpy.viz.jupyter_utils import save_fig_filebox_button
from sidpy.hdf.hdf_utils import get_attr
import ipywidgets as widgets
import h5py

class RawGmodeDataset(USIDataset):
    """
    Extention of teh USIDataset object for Linear Time-resolved Kelvin Probe Force Microscopy (tr-KPFM) datasets
    This includes various visualization and analysis routines

    Pass the raw_data h5 dataset to make this into a RawTRKPFM_L_Dataset

    Should be able to pass either HDF5dataset or USIDataset
    """

    def __init__(self,h5_dataset):
        super(RawGmodeDataset,self).__init__(h5_ref = h5_dataset)

        #Prepare the datasets
        self.dataset_type = 'RawGmodeDataset'
        self.parm_dict = self.dset.file['/Measurement_000'].attrs


class GmodeDataset(RawGmodeDataset):
    def __init__(self,h5_dataset):
        super(GmodeDataset, self).__init__(h5_dataset)
        self.dataset_type = 'RawGmodeDataset'

    def plot_data(self):
        """
        function to visualize the data
        """
        expt_type = get_attr(self.file,'data_type')
