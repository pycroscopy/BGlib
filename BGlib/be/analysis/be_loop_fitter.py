# -*- coding: utf-8 -*-
"""
:class:`~pycroscopy.analysis.be_loop_fitter.BELoopFitter` that fits Simple
Harmonic Oscillator model data to a parametric model to describe hysteretic
switching in ferroelectric materials

Created on Thu Nov 20 11:48:53 2019

@author: Suhas Somnath, Chris R. Smith, Rama K. Vasudevan

"""

from __future__ import division, print_function, absolute_import, \
    unicode_literals
import dask
import time
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import least_squares
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sidpy.hdf.dtype_utils import stack_real_to_compound, \
    flatten_compound_to_real
from sidpy.hdf.hdf_utils import get_attr, write_simple_attrs
from sidpy.proc.comp_utils import get_MPI, recommend_cpu_cores
from pyUSID.io.usi_data import USIDataset
from .be_sho_fitter import sho32
from BGlib.be.analysis.utils.fitter import Fitter
# import utils
'''
Custom dtypes for the datasets created during fitting.
'''
loop_metrics32 = np.dtype({'names': ['Area', 'Centroid x', 'Centroid y',
                                     'Rotation Angle [rad]', 'Offset'],
                           'formats': [np.float32, np.float32, np.float32,
                                       np.float32, np.float32]})

crit32 = np.dtype({'names': ['AIC_loop', 'BIC_loop', 'AIC_line', 'BIC_line'],
                   'formats': [np.float32, np.float32, np.float32,
                               np.float32]})

__field_names = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'b_0', 'b_1', 'b_2', 'b_3',
                 'R2 Criterion']
loop_fit32 = np.dtype({'names': __field_names,
                       'formats': [np.float32 for name in __field_names]})


class BELoopFitter(Fitter):
    """
    A class that fits Simple Harmonic Oscillator model data to a 9-parameter
    model to describe hysteretic switching in
    ferroelectric materials

    Notes
    -----
    Quantitative mapping of switching behavior in piezoresponse force microscopy, Stephen Jesse, Ho Nyung Lee,
    and Sergei V. Kalinin, Review of Scientific Instruments 77, 073702 (2006); doi: http://dx.doi.org/10.1063/1.2214699

    """

    def __init__(self, h5_main, be_data_type, vs_mode, vs_cycle_frac,
                 **kwargs):
        """

        Parameters
        ----------
        h5_main : h5py.Dataset
            The dataset over which the analysis will be performed. This dataset
            should be linked to the spectroscopic indices and values, and position
            indices and values datasets.
        data_type : str
            Type of data. This is an attribute written to the HDF5 file at the
            root level by either the translator or the acquisition software.
            Accepted values are: 'BEPSData' and 'cKPFMData'
            Default - this function will attempt to extract this metadata from the
            HDF5 file
        vs_mode: str
            Type of measurement. Accepted values are:
             'AC modulation mode with time reversal' or 'DC modulation mode'
             This is an attribute embedded under the "Measurement" group with the
             following key: 'VS_mode'. Default - this function will attempt to
             extract this metadata from the HDF5 file
        vs_cycle_frac : str
            Fraction of the bi-polar triangle waveform for voltage spectroscopy
            used in this experiment
        h5_target_group : h5py.Group, optional. Default = None
            Location where to look for existing results and to place newly
            computed results. Use this kwarg if the results need to be written
            to a different HDF5 file. By default, this value is set to the
            parent group containing `h5_main`
        kwargs : passed onto pyUSID.Process
        """

        super(BELoopFitter, self).__init__(h5_main, "Loop_Fit",
                                           variables=None, **kwargs)

        # This will be reset h5_main to this value before guess / fit
        # Some simple way to guard against failure
        self.__h5_main_orig = USIDataset(h5_main)

        self.parms_dict = None

        self._check_validity(h5_main, be_data_type, vs_mode, vs_cycle_frac)

        # Instead of the variables kwarg to the Fitter. Do check here:
        if 'DC_Offset' in self.h5_main.spec_dim_labels:
            self._fit_dim_name = 'DC_Offset'
        elif 'write_bias' in self.h5_main.spec_dim_labels:
            self._fit_dim_name = 'write_bias'
        else:
            raise ValueError('Neither "DC_Offset", nor "write_bias" were '
                             'spectroscopic dimension in the provided dataset '
                             'which has dimensions: {}'
                             '.'.format(self.h5_main.spec_dim_labels))

        if 'FORC' in self.h5_main.spec_dim_labels:
            self._forc_dim_name = 'FORC'
        else:
            self._forc_dim_name = 'FORC_Cycle'

        # accounting for memory copies
        self._max_raw_pos_per_read = self._max_pos_per_read

        # Declaring attributes here for PEP8 cleanliness
        self.h5_projected_loops = None
        self.h5_loop_metrics = None
        self._met_spec_inds = None
        self._write_results_chunk = None


    def do_be_fitting(self,method='K-Means',NN=2,h5_partial_guess=None, *func_args, **func_kwargs):
        """
        Function to perform BE Loop fitting at each pixel.

        Parameters
        -----
        guess_func : K-Means, optional
            Which fitting method to use. Default is K-means clustering for priors
        NN : 2, optional and only used for 'Neighbor' method
            Number of surrounding shells of pixels to average.  Default is 2 (surrounding 24 pixels)
        h5_partial_guess : h5py.Dataset, optional
            Partial guess results dataset to continue computing on
        """
        self.parms_dict = {'fitting-method': 'K-Means'}

        if method not in ['Random', 'Neighbor', 'K-Means', 'Hierarchical']:
            raise TypeError('Please use one of the following methods: Random, Neighbor, K-Means, Hierarchical.')


        #TODO: define h5_f, dc_vec, PR_mat for future use

        if method == 'K-Means':
            from .utils.be_kmeans_fit import _do_fit
            self.parms_dict.update({'fitting-method': 'K-Means'})

            fig,ax,p0_refs,p0_mat,SumSq,fitted_loops_mat = _do_fit()
            return fig,ax,p0_refs,p0_mat,SumSq,fitted_loops_mat

        if method == 'Neighbor':
            from .utils.be_neighbor_fit import _do_fit
            self.parms_dict.update({'fitting-method': 'Nearest Neighbor',
                                    'number of neighbors': NN})

            fig,ax,p0_refs,p0_mat,SumSq,fitted_loops_mat = _do_fit()
            return fig,ax,p0_refs,p0_mat,SumSq,fitted_loops_mat

        if method == 'Hierarchical':
            # from .utils import be_hierarchical_fit
            # from .utils import fitter

            self.parms_dict.update({'fitting-method': 'Hierarchical'})
            expt_type = usid.hdf_utils.get_attr(h5_f, 'data_type')
            h5_main = usid.hdf_utils.find_dataset(h5_f, 'Raw_Data')[0]
            h5_meas_grp = h5_main.parent.parent
            vs_mode = usid.hdf_utils.get_attr(h5_meas_grp, 'VS_mode')
            vs_cycle_frac = usid.hdf_utils.get_attr(h5_meas_grp, 'VS_cycle_fraction')
            max_cores = 1
            # Do the Loop Fitting on the SHO Fit dataset
            h5_loop_group = None
            # TODO: I don't think loop_fitter is being properly called since I moved the functions into be_hierarchical_fit
            loop_fitter = BELoopFitter(h5_sho_fit, expt_type, vs_mode, vs_cycle_frac,
                                                      cores=max_cores, h5_target_group=h5_loop_group,
                                                      verbose=False)
            loop_fitter.set_up_guess()
            h5_loop_guess = loop_fitter.do_guess(override=True)
            # Calling explicitely here since Fitter won't do it automatically
            h5_guess_loop_parms = loop_fitter.extract_loop_parameters(h5_loop_guess)

            loop_fitter.set_up_fit()
            h5_loop_fit = loop_fitter.do_fit(override=True)
            h5_loop_group = h5_loop_fit.parent





def shift_vdc(vdc_vec):
    """
    Rolls the Vdc vector by a quarter cycle

    Parameters
    ----------
    vdc_vec : 1D numpy array
        DC offset vector

    Returns
    -------
    shift_ind : int
        Number of indices by which the vector was rolled
    vdc_shifted : 1D numpy array
        Vdc vector rolled by a quarter cycle

    """
    shift_ind = int(
        -1 * len(vdc_vec) / 4)  # should NOT be hardcoded like this!
    vdc_shifted = np.roll(vdc_vec, shift_ind)
    return shift_ind, vdc_shifted