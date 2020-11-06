# -*- coding: utf-8 -*-
"""
:class:`~pycroscopy.analysis.be_loop_fitter.BELoopFitter` that fits Simple
Harmonic Oscillator model data to a parametric model to describe hysteretic
switching in ferroelectric materials

Created on Thu Nov 20 11:48:53 2019
Last Updated Fri Oct 23 2020

@author: Suhas Somnath, Chris R. Smith, Rama K. Vasudevan, Nicole C. Creange

"""

from __future__ import division, print_function, absolute_import, \
    unicode_literals
import joblib
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
from pyUSID.io.hdf_utils import get_unit_values, get_sort_order, \
    reshape_to_n_dims, create_empty_dataset, create_results_group, \
    write_reduced_anc_dsets, write_main_dataset
from pyUSID.io.usi_data import USIDataset
from .utils.be_loop import projectLoop, fit_loop, generate_guess, \
    loop_fit_function, calc_switching_coef_vec, switching32
from .utils.tree import ClusterTree
from .be_sho_fitter import sho32
from .fitter import Fitter

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
                 'R^2']
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

    @staticmethod
    def _check_validity(h5_main, data_type, vs_mode, vs_cycle_frac):
        """
        Checks whether or not the provided object can be analyzed by this class

        Parameters
        ----------
        h5_main : h5py.Dataset instance
            The dataset containing the SHO Fit (not necessarily the dataset
            directly resulting from SHO fit)
            over which the loop projection, guess, and fit will be performed.
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
        """
        if h5_main.dtype != sho32:
            raise TypeError('Provided dataset is not a SHO results dataset.')

        if data_type == 'BEPSData':
            if vs_mode not in ['DC modulation mode', 'current mode']:
                raise ValueError('Provided dataset has a mode: "' + vs_mode +
                                 '" is not a "DC modulation" or "current mode"'
                                 ' BEPS dataset')
            elif vs_cycle_frac != 'full':
                raise ValueError('Provided dataset does not have full cycles')

        elif data_type == 'cKPFMData':
            if vs_mode != 'cKPFM':
                raise ValueError('Provided dataset has an unsupported VS_mode:'
                                 ' "' + vs_mode + '"')
        else:
            raise NotImplementedError('Loop fitting not supported for Band '
                                      'Excitation experiment type: {}'
                                      ''.format(data_type))

    def _create_projection_datasets(self):
        """
        Creates the Loop projection and metrics HDF5 dataset & results group
        """

        # Which row in the spec datasets is DC offset?
        _fit_spec_index = self.h5_main.spec_dim_labels.index(
            self._fit_dim_name)

        # TODO: Unkown usage of variable. Waste either way
        # self._fit_offset_index = 1 + _fit_spec_index

        # Calculate the number of loops per position
        cycle_start_inds = np.argwhere(
            self.h5_main.h5_spec_inds[_fit_spec_index, :] == 0).flatten()
        tot_cycles = cycle_start_inds.size
        if self.verbose and self.mpi_rank == 0:
            print('Found {} cycles starting at indices: {}'.format(tot_cycles,
                                                                   cycle_start_inds))

        # Make the results group
        self.h5_results_grp = create_results_group(self.h5_main,
                                                   self.process_name,
                                                   h5_parent_group=self._h5_target_group)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        # If writing to a new HDF5 file:
        # Add back the data_type attribute - still being used in the visualizer
        if self.h5_results_grp.file != self.h5_main.file:
            write_simple_attrs(self.h5_results_grp.file,
                               {'data_type': get_attr(self.h5_main.file,
                                                      'data_type')})

        # Write datasets
        self.h5_projected_loops = create_empty_dataset(self.h5_main,
                                                       np.float32,
                                                       'Projected_Loops',
                                                       h5_group=self.h5_results_grp)

        h5_loop_met_spec_inds, h5_loop_met_spec_vals = write_reduced_anc_dsets(
            self.h5_results_grp, self.h5_main.h5_spec_inds,
            self.h5_main.h5_spec_vals, self._fit_dim_name,
            basename='Loop_Metrics', verbose=False)

        self.h5_loop_metrics = write_main_dataset(self.h5_results_grp,
                                                  (self.h5_main.shape[0], tot_cycles), 'Loop_Metrics',
                                                  'Metrics', 'compound', None,
                                                  None, dtype=loop_metrics32,
                                                  h5_pos_inds=self.h5_main.h5_pos_inds,
                                                  h5_pos_vals=self.h5_main.h5_pos_vals,
                                                  h5_spec_inds=h5_loop_met_spec_inds,
                                                  h5_spec_vals=h5_loop_met_spec_vals)

        # Copy region reference:
        # copy_region_refs(self.h5_main, self.h5_projected_loops)
        # copy_region_refs(self.h5_main, self.h5_loop_metrics)

        self.h5_main.file.flush()
        self._met_spec_inds = self.h5_loop_metrics.h5_spec_inds

        if self.verbose and self.mpi_rank == 0:
            print('Finished creating Guess dataset')

    def _create_guess_datasets(self):
        """
        Creates the HDF5 Guess dataset
        """
        self._create_projection_datasets()

        self._h5_guess = create_empty_dataset(self.h5_loop_metrics, loop_fit32,
                                             'Guess')

        self._h5_guess = USIDataset(self._h5_guess)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        self.h5_main.file.flush()

    def _create_fit_datasets(self):
        """
        Creates the HDF5 Fit dataset
        """

        if self._h5_guess is None:
            raise ValueError('Need to guess before fitting!')

        self._h5_fit = create_empty_dataset(self._h5_guess, loop_fit32, 'Loop Fits')
        self._h5_fit = USIDataset(self._h5_fit)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        self.h5_main.file.flush()

    def _read_data_chunk(self):
        """
        Get the next chunk of SHO data (in the case of Guess) or Projected
        loops (in the case of Fit)

        Notes
        -----
        self.data contains data for N pixels.
        The challenge is that this may contain M FORC cycles
        Each FORC cycle needs its own V DC vector
        So, we can't blindly use the inherited unit_compute.
        Our variables now are Position, Vdc, FORC, all others

        We want M lists of [VDC x all other variables]

        The challenge is that VDC and FORC are inner dimensions -
        neither the fastest nor the slowest (guaranteed)
        """

        # The Process class should take care of all the basic reading
        super(BELoopFitter, self)._read_data_chunk()

        if self.data is None:
            # Nothing we can do at this point
            return

        if self.verbose and self.mpi_rank == 0:
            print('BELoopFitter got data chunk of shape {} from Fitter'
                  '.'.format(self.data.shape))

        spec_dim_order_s2f = get_sort_order(self.h5_main.h5_spec_inds)[::-1]

        self._dim_labels_s2f = list(['Positions']) + list(
            np.array(self.h5_main.spec_dim_labels)[spec_dim_order_s2f])

        self._num_forcs = int(
            any([targ in self.h5_main.spec_dim_labels for targ in
                 ['FORC', 'FORC_Cycle']]))

        order_to_s2f = [0] + list(1 + spec_dim_order_s2f)
        if self.verbose and self.mpi_rank == 0:
            print('Order for reshaping to S2F: {}'.format(order_to_s2f))

        if self.verbose and self.mpi_rank == 0:
            print(self._dim_labels_s2f, order_to_s2f)

        if self._num_forcs:
            forc_pos = self.h5_main.spec_dim_labels.index(self._forc_dim_name)
            self._num_forcs = self.h5_main.spec_dim_sizes[forc_pos]

        if self.verbose and self.mpi_rank == 0:
            print('Num FORCS: {}'.format(self._num_forcs))

        all_but_forc_rows = []
        for ind, dim_name in enumerate(self.h5_main.spec_dim_labels):
            if dim_name not in ['FORC', 'FORC_Cycle', 'FORC_repeat']:
                all_but_forc_rows.append(ind)

        if self.verbose and self.mpi_rank == 0:
            print('All but FORC rows: {}'.format(all_but_forc_rows))

        dc_mats = []

        forc_mats = []

        num_reps = 1 if self._num_forcs == 0 else self._num_forcs
        for forc_ind in range(num_reps):
            if self.verbose and self.mpi_rank == 0:
                print('\nWorking on FORC #{}'.format(forc_ind))

            if self._num_forcs:
                this_forc_spec_inds = \
                    np.where(self.h5_main.h5_spec_inds[forc_pos] == forc_ind)[
                        0]
            else:
                this_forc_spec_inds = np.ones(
                    shape=self.h5_main.h5_spec_inds.shape[1], dtype=np.bool)

            if self._num_forcs:
                this_forc_dc_vec = get_unit_values(
                    self.h5_main.h5_spec_inds[all_but_forc_rows][:,
                    this_forc_spec_inds],
                    self.h5_main.h5_spec_vals[all_but_forc_rows][:,
                    this_forc_spec_inds],
                    all_dim_names=list(np.array(self.h5_main.spec_dim_labels)[
                                           all_but_forc_rows]),
                    dim_names=self._fit_dim_name)
            else:
                this_forc_dc_vec = get_unit_values(self.h5_main.h5_spec_inds,
                                                   self.h5_main.h5_spec_vals,
                                                   dim_names=self._fit_dim_name)
            this_forc_dc_vec = this_forc_dc_vec[self._fit_dim_name]
            dc_mats.append(this_forc_dc_vec)

            this_forc_2d = self.data[:, this_forc_spec_inds]
            if self.verbose and self.mpi_rank == 0:
                print('2D slice shape for this FORC: {}'.format(this_forc_2d.shape))

            this_forc_nd, success = reshape_to_n_dims(this_forc_2d,
                                                      h5_pos=None,
                                                      h5_spec=self.h5_main.h5_spec_inds[
                                                              :,
                                                              this_forc_spec_inds])

            if success != True:
                raise ValueError('Unable to reshape data to N dimensions')

            if self.verbose and self.mpi_rank == 0:
                print(this_forc_nd.shape)

            this_forc_nd_s2f = this_forc_nd.transpose(
                order_to_s2f).squeeze()  # squeeze out FORC
            dim_names_s2f = self._dim_labels_s2f.copy()
            if self._num_forcs > 0:
                dim_names_s2f.remove(
                    self._forc_dim_name)
                # because it was never there in the first place.
            if self.verbose and self.mpi_rank == 0:
                print('Reordered to S2F: {}, {}'.format(this_forc_nd_s2f.shape,
                                                        dim_names_s2f))

            rest_dc_order = list(range(len(dim_names_s2f)))
            _dc_ind = dim_names_s2f.index(self._fit_dim_name)
            rest_dc_order.remove(_dc_ind)
            rest_dc_order = rest_dc_order + [_dc_ind]
            if self.verbose and self.mpi_rank == 0:
                print('Transpose for reordering to rest, DC: {}'
                      ''.format(rest_dc_order))

            rest_dc_nd = this_forc_nd_s2f.transpose(rest_dc_order)
            rest_dc_names = list(np.array(dim_names_s2f)[rest_dc_order])

            self._pre_flattening_shape = list(rest_dc_nd.shape)
            self._pre_flattening_dim_name_order = list(rest_dc_names)

            if self.verbose and self.mpi_rank == 0:
                print('After reodering: {}, {}'.format(rest_dc_nd.shape,
                                                       rest_dc_names))

            dc_rest_2d = rest_dc_nd.reshape(np.prod(rest_dc_nd.shape[:-1]),
                                            np.prod(rest_dc_nd.shape[-1]))

            if self.verbose and self.mpi_rank == 0:
                print('Shape after flattening to 2D: {}'
                      ''.format(dc_rest_2d.shape))

            forc_mats.append(dc_rest_2d)

        self.data = forc_mats, dc_mats

        if self.verbose and self.mpi_rank == 0:
            print('self.data loaded')

    def _read_guess_chunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset.

        Notes
        -----
        Use the same strategy as that used for reading the raw data.
        The technique is slightly simplified since the end result
        per FORC cycle is just a 1D array of loop metrics.
        However, this compound dataset needs to be converted to float
        in order to send to scipy.optimize.least_squares
        """
        # The Fitter class should take care of all the basic reading
        super(BELoopFitter, self)._read_guess_chunk()

        if self.verbose and self.mpi_rank == 0:
            print('_read_guess_chunk got guess of shape: {} from super'
                  '.'.format(self._guess.shape))

        spec_dim_order_s2f = get_sort_order(self._h5_guess.h5_spec_inds)[::-1]

        order_to_s2f = [0] + list(1 + spec_dim_order_s2f)
        if self.verbose and self.mpi_rank == 0:
            print('Order for reshaping to S2F: {}'.format(order_to_s2f))

        dim_labels_s2f = list(['Positions']) + list(
            np.array(self._h5_guess.spec_dim_labels)[spec_dim_order_s2f])

        if self.verbose and self.mpi_rank == 0:
            print(dim_labels_s2f, order_to_s2f)

        num_forcs = int(any([targ in self._h5_guess.spec_dim_labels for targ in
                             ['FORC', 'FORC_Cycle']]))
        if num_forcs:
            forc_pos = self._h5_guess.spec_dim_labels.index(self._forc_dim_name)
            num_forcs = self._h5_guess.spec_dim_sizes[forc_pos]

        if self.verbose and self.mpi_rank == 0:
            print('Num FORCS: {}'.format(num_forcs))

        all_but_forc_rows = []
        for ind, dim_name in enumerate(self._h5_guess.spec_dim_labels):
            if dim_name not in ['FORC', 'FORC_Cycle', 'FORC_repeat']:
                all_but_forc_rows.append(ind)

        if self.verbose and self.mpi_rank == 0:
            print('All but FORC rows: {}'.format(all_but_forc_rows))

        forc_mats = []

        num_reps = 1 if num_forcs == 0 else num_forcs
        for forc_ind in range(num_reps):
            if self.verbose and self.mpi_rank == 0:
                print('\nWorking on FORC #{}'.format(forc_ind))
            if num_forcs:
                this_forc_spec_inds = \
                np.where(self._h5_guess.h5_spec_inds[forc_pos] == forc_ind)[0]
            else:
                this_forc_spec_inds = np.ones(
                    shape=self._h5_guess.h5_spec_inds.shape[1], dtype=np.bool)

            this_forc_2d = self._guess[:, this_forc_spec_inds]
            if self.verbose and self.mpi_rank == 0:
                print('2D slice shape for this FORC: {}'.format(this_forc_2d.shape))

            this_forc_nd, success = reshape_to_n_dims(this_forc_2d,
                                                      h5_pos=None,
                                                      h5_spec=self._h5_guess.h5_spec_inds[
                                                              :,
                                                              this_forc_spec_inds])

            if success != True:
                raise ValueError('Unable to reshape 2D guess to N dimensions')

            if self.verbose and self.mpi_rank == 0:
                print('N dimensional shape for this FORC: {}'.format(this_forc_nd.shape))

            this_forc_nd_s2f = this_forc_nd.transpose(
                order_to_s2f).squeeze()  # squeeze out FORC
            dim_names_s2f = dim_labels_s2f.copy()
            if num_forcs > 0:
                dim_names_s2f.remove(self._forc_dim_name)
                # because it was never there in the first place.
            if self.verbose and self.mpi_rank == 0:
                print('Reordered to S2F: {}, {}'.format(this_forc_nd_s2f.shape,
                                                        dim_names_s2f))

            dc_rest_2d = this_forc_nd_s2f.ravel()
            if self.verbose and self.mpi_rank == 0:
                print('Shape after raveling: {}'.format(dc_rest_2d.shape))

            # Scipy will not understand compound values. Flatten.
            # Ignore the R2 error
            # TODO: avoid memory copies!
            float_mat = np.zeros(shape=list(dc_rest_2d.shape) +
                                       [len(loop_fit32.names)-1],
                                 dtype=np.float32)
            if self.verbose and self.mpi_rank == 0:
                print('Created empty float matrix of shape: {}'
                      '.'.format(float_mat.shape))
            for ind, field_name in enumerate(loop_fit32.names[:-1]):
                float_mat[..., ind] = dc_rest_2d[field_name]

            if self.verbose and self.mpi_rank == 0:
                print('Shape after flattening to float: {}'
                      '.'.format(float_mat.shape))

            forc_mats.append(float_mat)

        self._guess = np.array(forc_mats)
        if self.verbose and self.mpi_rank == 0:
            print('Flattened Guesses to shape: {} and dtype:'
                  '.'.format(self._guess.shape, self._guess.dtype))

    @staticmethod
    def _project_loop(sho_response, dc_offset):
        """
        Projects a provided piezoelectric hysteresis loop

        Parameters
        ----------
        sho_response : numpy.ndarray
            Compound valued array with the SHO response for a single loop
        dc_offset : numpy.ndarray
            DC offset corresponding to the provided loop

        Returns
        -------
        projected_loop : numpy.ndarray
            Projected loop
        ancillary : numpy.ndarray
            Metrics for the loop projection
        """
        # projected_loop = np.zeros(shape=sho_response.shape, dtype=np.float32)
        ancillary = np.zeros(shape=1, dtype=loop_metrics32)

        pix_dict = projectLoop(np.squeeze(dc_offset),
                               sho_response['Amplitude [V]'],
                               sho_response['Phase [rad]'])

        projected_loop = pix_dict['Projected Loop']
        ancillary['Rotation Angle [rad]'] = pix_dict['Rotation Matrix'][0]
        ancillary['Offset'] = pix_dict['Rotation Matrix'][1]
        ancillary['Area'] = pix_dict['Geometric Area']
        ancillary['Centroid x'] = pix_dict['Centroid'][0]
        ancillary['Centroid y'] = pix_dict['Centroid'][1]

        return projected_loop, ancillary

    def set_up_guess(self):
        """
        Performs necessary book-keeping before fitting can be called.
        Also remaps data reading, computation, writing functions to those
        specific to Guess.
        """
        self.h5_main = self.__h5_main_orig
        self.parms_dict = {'projection_method': 'pycroscopy BE loop model',
                           'guess_method': "pycroscopy Nearest Neighbor"}

        # self.compute = self.

    @staticmethod
    def extract_loop_parameters(h5_loop_fit, nuc_threshold=0.03):
        """
        Method to extract a set of physical loop parameters from a dataset of fit parameters
        Parameters
        ----------
        h5_loop_fit : h5py.Dataset
            Dataset of loop fit parameters
        nuc_threshold : float
            Nucleation threshold to use in calculation physical parameters
        Returns
        -------
        h5_loop_parm : h5py.Dataset
            Dataset of physical parameters
        """
        dset_name = h5_loop_fit.name.split('/')[-1] + '_Loop_Parameters'
        h5_loop_parameters = create_empty_dataset(h5_loop_fit,
                                                  dtype=switching32,
                                                  dset_name=dset_name,
                                                  new_attrs={
                                                      'nuc_threshold': nuc_threshold})

        loop_coef_vec = flatten_compound_to_real(
            np.reshape(h5_loop_fit, [-1, 1]))
        switching_coef_vec = calc_switching_coef_vec(loop_coef_vec,
                                                     nuc_threshold)

        h5_loop_parameters[:, :] = switching_coef_vec.reshape(
            h5_loop_fit.shape)

        h5_loop_fit.file.flush()

        return h5_loop_parameters

    # def split_data(self,cycle=1):
    #     main_dsets = usid.hdf_utils.get_all_main(h5_f)
    #     h5_sho_fit = main_dsets[1]
    #     amplitude = h5_sho_fit['Amplitude [V]']
    #     phase = h5_sho_fit['Phase [rad]']
    #     adjust = np.max(phase) - np.min(phase)
    #     phase_wrap = []
    #     for ii in range(phase.shape[0]):
    #         phase_wrap.append([x + adjust if x < -2 else x for x in phase[ii, :]])
    #     phase = np.asarray(phase_wrap)
    #     plt.figure()
    #     plt.hist(phase.ravel(), bins=100)
    #
    #     PR_mat = amplitude * np.cos(phase)
    #     self.PR_mat_full = -PR_mat.reshape(h5_sho_fit.pos_dim_sizes[0], h5_sho_fit.pos_dim_sizes[1], -1)
    #
    #     self.dc_vec_OF = h5_sho_fit.h5_spec_vals[0, :][
    #         np.logical_and(h5_sho_fit.h5_spec_vals[1, :] == 0, h5_sho_fit.h5_spec_vals[2, :] == cycle)]  # off field
    #     self.dc_vec_IF = h5_sho_fit.h5_spec_vals[0, :][
    #         np.logical_and(h5_sho_fit.h5_spec_vals[1, :] == 1, h5_sho_fit.h5_spec_vals[2, :] == cycle)]  # on field
    #
    #     PR_OF2 = PR_mat[:, :, 129::2]  # off field
    #     PR_IF2 = PR_mat[:, :, 128::2]  # on field

    def neighbor_fit(self,dc_vec, PR_mat,NN=1):
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score
        from tqdm import trange
        from copy import deepcopy

        cmap = plt.cm.plasma_r
        scale = (0, 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        cbaxs = fig.add_axes([0.92, 0.125, 0.02, 0.755])
        p0_refs = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        all_mean = np.mean(np.mean(PR_mat, axis=0), axis=0)

        bnds = (-100, 100)
        p0_mat = [[]] * PR_mat.shape[0] * PR_mat.shape[1]  # empty array to store fits from neighboring pixels
        fitted_loops_mat = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        SumSq = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        ref_counts = np.arange(PR_mat.shape[0] * PR_mat.shape[1]).reshape(
            (PR_mat.shape[0], PR_mat.shape[1]))  # reference for finding neighboring pixels
        count = -1
        # SET UP X DATA
        xdata0 = dc_vec
        max_x = np.where(xdata0 == np.max(xdata0))[0]
        if max_x != 0 or max_x != len(xdata0):
            xdata = np.roll(xdata0, -max_x)  # assumes voltages are a symmetric triangle wave
            dum = 1
        else:
            xdata = xdata0  # just in case voltages are already rolled
            dum = 0

        p0_vals = []
        opt_vals = []
        res = []
        if dum == 1:
            all_mean = np.roll(all_mean, -max_x)

        for kk in range(50):
            p0 = np.random.normal(0.1, 5, 9)
            p0_vals.append(p0)
            try:
                vals_min, pcov = curve_fit(loop_fit_function, xdata, all_mean, p0=p0, maxfev=10000)
            except:
                continue
            opt_vals.append(vals_min)
            fitted_loop = loop_fit_function(xdata, *vals_min)
            yres = all_mean - fitted_loop
            res.append(yres @ yres)

        popt = opt_vals[np.argmin(res)]
        popt_mean = deepcopy(popt)
        p0_mat = [popt] * PR_mat.shape[0] * PR_mat.shape[1]
        plt.figure()
        plt.plot(xdata, all_mean, 'ko')
        fitted_loop = loop_fit_function(xdata, *popt)
        plt.plot(xdata, fitted_loop, 'k')
        print('Done with average fit')
        for ii in trange(PR_mat.shape[0]):
            xind = ii
            for jj in range(PR_mat.shape[1]):
                count += 1  # used to keep track of popt vals
                yind = jj
                ydata0 = PR_mat[xind, yind, :]
                if dum == 1:
                    ydata = np.roll(ydata0, -max_x)
                else:
                    ydata = ydata0

                xs = [ii + k for k in range(-NN, NN + 1)]
                ys = [jj + k for k in range(-NN, NN + 1)]
                nbrs = [(n, m) for n in xs for m in ys]
                cond = [all(x >= 0 for x in list(y)) for y in nbrs]
                nbrs = [d for (d, remove) in zip(nbrs, cond) if remove]
                cond2 = [all(x < ref_counts.shape[0] for x in list(y)) for y in
                         nbrs]  # assumes PR_mat is square....
                nbrs = [d for (d, remove) in zip(nbrs, cond2) if remove]
                NN_indx = [ref_counts[v] for v in nbrs]
                prior_coefs = [p0_mat[k] for k in NN_indx if len(p0_mat[k]) != 0]
                if prior_coefs == []:
                    p0 = popt
                else:
                    p0 = np.mean(prior_coefs, axis=0)
                p0_refs[count] = p0
                try:
                    popt, pcov = curve_fit(loop_fit_function, xdata, ydata, p0=p0, maxfev=10000, bounds=bnds)
                except:
                    continue
                p0_mat[count] = popt  # saves fitted coefficients for the index

                fitted_loop = loop_fit_function(xdata, *p0_mat[count])
                fitted_loops_mat[count] = fitted_loop
                ss = r2_score(ydata, fitted_loop)
                SumSq[count] = ss
                sh = np.floor(1 + (2 ** 16 - 1) * ((ss) - scale[0]) / (scale[1] - scale[0]))
                if sh < 1:
                    sh = 1
                if sh > 2 ** 16:
                    sh = 2 ** 16

                ax.plot(ii, jj, c=cmap(sh / (2 ** 16)), marker='s', markersize=7)

        scbar = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r, norm=plt.Normalize(vmin=scale[0], vmax=scale[1]))
        scbar._A = []
        cbar = plt.colorbar(scbar, cax=cbaxs)
        cbar.ax.set_ylabel('$R^2$', rotation=270, labelpad=20)

        return fig, ax, p0_refs, p0_mat, SumSq, fitted_loops_mat


    def kmeans_fit(self,dc_vec, PR_mat):
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score
        from tqdm import trange
        from copy import deepcopy
        from sklearn.cluster import KMeans

        cmap = plt.cm.plasma_r
        scale = (0, 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        cbaxs = fig.add_axes([0.92, 0.125, 0.02, 0.755])
        p0_refs = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        all_mean = np.mean(np.mean(PR_mat, axis=0), axis=0)

        bnds = (-100, 100)
        p0_mat = [[]] * PR_mat.shape[0] * PR_mat.shape[1]  # empty array to store fits from neighboring pixels
        fitted_loops_mat = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        SumSq = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        ref_counts = np.arange(PR_mat.shape[0] * PR_mat.shape[1]).reshape(
            (PR_mat.shape[0], PR_mat.shape[1]))  # reference for finding neighboring pixels
        count = -1
        # SET UP X DATA
        xdata0 = dc_vec
        max_x = np.where(xdata0 == np.max(xdata0))[0]
        if max_x != 0 or max_x != len(xdata0):
            xdata = np.roll(xdata0, -max_x)  # assumes voltages are a symmetric triangle wave
            dum = 1
        else:
            xdata = xdata0  # just in case voltages are already rolled
            dum = 0

        p0_vals = []
        opt_vals = []
        res = []
        if dum == 1:
            all_mean = np.roll(all_mean, -max_x)

        for kk in range(50):
            p0 = np.random.normal(0.1, 5, 9)
            p0_vals.append(p0)
            try:
                vals_min, pcov = curve_fit(loop_fit_function, xdata, all_mean, p0=p0, maxfev=10000)
            except:
                continue
            opt_vals.append(vals_min)
            fitted_loop = loop_fit_function(xdata, *vals_min)
            yres = all_mean - fitted_loop
            res.append(yres @ yres)
        popt = opt_vals[np.argmin(res)]
        popt_mean = deepcopy(popt)
        p0_mat = [popt] * PR_mat.shape[0] * PR_mat.shape[1]
        plt.figure()
        plt.plot(xdata, all_mean, 'ko')
        fitted_loop = loop_fit_function(xdata, *popt)
        plt.plot(xdata, fitted_loop, 'k')
        print('Done with average fit')

        size = PR_mat.shape[0] * PR_mat.shape[1]
        if nclust is empty:
            n_clusters = int(size / 50)
        else:
            n_clusters = nclust
        print('Using ' + str(n_clusters) + ' clusters')
        PR_mat_flat = PR_mat.reshape(size, int(PR_mat.shape[2]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(PR_mat_flat)
        labels = kmeans.labels_
        p0_clusters = []
        cluster_loops = []
        for pp in trange(n_clusters):
            opt_vals = []
            res = []
            clust = PR_mat_flat[labels == pp]
            PR_mean = np.mean(clust, axis=0)
            if dum == 1:
                PR_mean = np.roll(PR_mean, -max_x)
            cluster_loops.append(PR_mean)
            p0 = p0_mat[0]
            try:
                popt, pcov = curve_fit(loop_fit_function, xdata, PR_mean, p0=p0, maxfev=10000)
            except:
                kk = 0
                p0 = np.random.normal(0.1, 5, 9)
                while kk < 20:
                    try:
                        vals_min, pcov = curve_fit(loop_fit_function, xdata, all_mean, p0=p0, maxfev=10000)
                    except:
                        continue
                    kk += 1
                    opt_vals.append(vals_min)
                    fitted_loop = loop_fit_function(xdata, *vals_min)
                    yres = PR_mean - fitted_loop
                    res.append(yres @ yres)
                    popt = opt_vals[np.argmin(res)]
            p0_clusters.append(popt)

        for ii in trange(PR_mat.shape[0]):
            xind = ii
            for jj in range(PR_mat.shape[1]):
                count += 1  # used to keep track of popt vals
                yind = jj
                ydata0 = PR_mat[xind, yind, :]
                if dum == 1:
                    ydata = np.roll(ydata0, -max_x)
                else:
                    ydata = ydata0

                lab = labels[count]
                p0 = p0_clusters[lab]
                try:
                    popt, pcov = curve_fit(loop_fit_function, xdata, ydata, p0=p0, maxfev=10000)
                except:
                    p0 = popt_mean
                    try:
                        popt, pcov = curve_fit(loop_fit_function, xdata, ydata, p0=p0, maxfev=10000)
                    except:
                        continue
                p0_refs[count] = p0
                p0_mat[count] = popt  # saves fitted coefficients for the index

                fitted_loop = loop_fit_function(xdata, *p0_mat[count])
                fitted_loops_mat[count] = fitted_loop
                ss = r2_score(ydata, fitted_loop)
                SumSq[count] = ss
                sh = np.floor(1 + (2 ** 16 - 1) * ((ss) - scale[0]) / (scale[1] - scale[0]))
                if sh < 1:
                    sh = 1
                if sh > 2 ** 16:
                    sh = 2 ** 16

                ax.plot(ii, jj, c=cmap(sh / (2 ** 16)), marker='s', markersize=7)


        scbar = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r, norm=plt.Normalize(vmin=scale[0], vmax=scale[1]))
        scbar._A = []
        cbar = plt.colorbar(scbar, cax=cbaxs)
        cbar.ax.set_ylabel('$R^2$', rotation=270, labelpad=20)

        return fig, ax, p0_refs, p0_mat, SumSq, fitted_loops_mat


    def random_fit(self,dc_vec, PR_mat):
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score
        from tqdm import trange

        cmap = plt.cm.plasma_r
        scale = (0, 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        cbaxs = fig.add_axes([0.92, 0.125, 0.02, 0.755])
        p0_refs = [[]] * PR_mat.shape[0] * PR_mat.shape[1]

        p0_mat = [[]] * PR_mat.shape[0] * PR_mat.shape[1]  # empty array to store fits from neighboring pixels
        fitted_loops_mat = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        SumSq = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        ref_counts = np.arange(PR_mat.shape[0] * PR_mat.shape[1]).reshape(
            (PR_mat.shape[0], PR_mat.shape[1]))  # reference for finding neighboring pixels
        count = -1
        # SET UP X DATA
        xdata0 = dc_vec
        max_x = np.where(xdata0 == np.max(xdata0))[0]
        if max_x != 0 or max_x != len(xdata0):
            xdata = np.roll(xdata0, -max_x)  # assumes voltages are a symmetric triangle wave
            dum = 1
        else:
            xdata = xdata0  # just in case voltages are already rolled
            dum = 0

        for ii in trange(PR_mat.shape[0]):
            xind = ii
            for jj in range(PR_mat.shape[1]):
                count += 1  # used to keep track of popt vals
                yind = jj
                ydata0 = PR_mat[xind, yind, :]
                if dum == 1:
                    ydata = np.roll(ydata0, -max_x)
                else:
                    ydata = ydata0

                opt_vals = []
                res = []
                p0_vals = []
                kk = 0
                while kk < 2:
                    p0 = np.random.normal(0.1, 5, 9)
                    p0_vals.append(p0)
                    try:
                        vals_min, pcov = curve_fit(loop_fit_function, xdata, ydata, p0=p0, maxfev=10000)
                    except:
                        continue
                    kk += 1
                    opt_vals.append(vals_min)
                    fitted_loop = loop_fit_function(xdata, *vals_min)
                    yres = ydata - fitted_loop
                    res.append(yres @ yres)
                popt = opt_vals[np.argmin(res)]
                p0_mat[count] = popt

                fitted_loop = loop_fit_function(xdata, *p0_mat[count])
                fitted_loops_mat[count] = fitted_loop
                ss = r2_score(ydata, fitted_loop)
                SumSq[count] = ss
                sh = np.floor(1 + (2 ** 16 - 1) * ((ss) - scale[0]) / (scale[1] - scale[0]))
                if sh < 1:
                    sh = 1
                if sh > 2 ** 16:
                    sh = 2 ** 16

                ax.plot(ii, jj, c=cmap(sh / (2 ** 16)), marker='s', markersize=7)

        scbar = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r, norm=plt.Normalize(vmin=scale[0], vmax=scale[1]))
        scbar._A = []
        cbar = plt.colorbar(scbar, cax=cbaxs)
        cbar.ax.set_ylabel('$R^2$', rotation=270, labelpad=20)

        return fig, ax, p0_refs, p0_mat, SumSq, fitted_loops_mat



    def fit_loops(self,dc_vec, PR_mat, method='Neighbor',NN=1, nclust = []):
        """
        Fits loops using default nearest neighbor method.

        dc_vec: vector of DC-biases
        PR_mat: matrix of data responses
        method: default nearest neighbor- Neighbor, can be changed to Random or K-Means
        NN: number of nearest neighbors to consider, default 1.  Only needed for Neighbors method
        nclust: number of clusters to use for K-Means, default of data size/100
        """


        if method not in ['Random', 'Neighbor', 'K-Means', 'Hierarchical']:
            print(
                'Please use one of the following methods: "Neighbor", "K-Means", "Hierarchical" or "Random".')
            return
        if method == 'Hierarchical':
            'Do tree fitting'
        if method == 'Neighbor':
            fig, ax, p0_refs, p0_mat, SumSq, fitted_loops_mat = neighbor_fit(self,dc_vec, PR_mat,NN=NN)
        if method == 'K-Means':
            fig, ax, p0_refs, p0_mat, SumSq, fitted_loops_mat = kmeans_fit(self,dc_vec, PR_mat)
        if method == 'Random':
            fig, ax, p0_refs, p0_mat, SumSq, fitted_loops_mat = random_fit(self,dc_vec, PR_mat)
        return fig, ax, p0_refs, p0_mat, SumSq, fitted_loops_mat




def _be_loop_err(coef_vec, data_vec, dc_vec, *args):
    """

    Parameters
    ----------
    coef_vec : numpy.ndarray
    data_vec : numpy.ndarray
    dc_vec : numpy.ndarray
        The DC offset vector
    args : list

    Returns
    -------
    fitness : float
        The 1-r^2 value for the current set of loop coefficients

    """
    if coef_vec.size < 9:
        raise ValueError(
            'Error: The Loop Fit requires 9 parameter guesses!')

    data_mean = np.mean(data_vec)

    func = loop_fit_function(dc_vec, coef_vec)

    ss_tot = sum(abs(data_vec - data_mean) ** 2)
    ss_res = sum(abs(data_vec - func) ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return 1 - r_squared


def guess_loops_hierarchically(vdc_vec, projected_loops_2d):
    """
    Provides loop parameter guesses for a given set of loops

    Parameters
    ----------
    vdc_vec : 1D numpy float numpy array
        DC voltage offsets for the loops
    projected_loops_2d : 2D numpy float array
        Projected loops arranged as [instance or position x dc voltage steps]

    Returns
    -------
    guess_parms : 1D compound numpy array
        Loop parameter guesses for the provided projected loops

    """

    def _loop_fit_tree(tree, guess_mat, fit_results, vdc_shifted,
                       shift_ind):
        """
        Recursive function that fits a tree object describing the cluster results

        Parameters
        ----------
        tree : ClusterTree object
            Tree describing the clustering results
        guess_mat : 1D numpy float array
            Loop parameters that serve as guesses for the loops in the tree
        fit_results : 1D numpy float array
            Loop parameters that serve as fits for the loops in the tree
        vdc_shifted : 1D numpy float array
            DC voltages shifted be 1/4 cycle
        shift_ind : unsigned int
            Number of units to shift loops by

        Returns
        -------
        guess_mat : 1D numpy float array
            Loop parameters that serve as guesses for the loops in the tree
        fit_results : 1D numpy float array
            Loop parameters that serve as fits for the loops in the tree

        """
        # print('Now fitting cluster #{}'.format(tree.name))
        # I already have a guess. Now fit myself
        curr_fit_results = fit_loop(vdc_shifted,
                                    np.roll(tree.value, shift_ind),
                                    guess_mat[tree.name])
        # keep all the fit results
        fit_results[tree.name] = curr_fit_results
        for child in tree.children:
            # Use my fit as a guess for the lower layers:
            guess_mat[child.name] = curr_fit_results[0].x
            # Fit this child:
            guess_mat, fit_mat = _loop_fit_tree(child, guess_mat,
                                                fit_results, vdc_shifted,
                                                shift_ind)
        return guess_mat, fit_results

    num_clusters = max(2, int(projected_loops_2d.shape[
                                  0] ** 0.5))  # change this to 0.6 if necessary
    estimators = KMeans(num_clusters)
    results = estimators.fit(projected_loops_2d)
    centroids = results.cluster_centers_
    labels = results.labels_

    # Get the distance between cluster means
    distance_mat = pdist(centroids)
    # get hierarchical pairings of clusters
    linkage_pairing = linkage(distance_mat, 'weighted')
    # Normalize the pairwise distance with the maximum distance
    linkage_pairing[:, 2] = linkage_pairing[:, 2] / max(
        linkage_pairing[:, 2])

    # Now use the tree class:
    cluster_tree = ClusterTree(linkage_pairing[:, :2], labels,
                               distances=linkage_pairing[:, 2],
                               centroids=centroids)
    num_nodes = len(cluster_tree.nodes)

    # prepare the guess and fit matrices
    loop_guess_mat = np.zeros(shape=(num_nodes, 9), dtype=np.float32)
    # loop_fit_mat = np.zeros(shape=loop_guess_mat.shape, dtype=loop_guess_mat.dtype)
    loop_fit_results = list(
        np.arange(num_nodes, dtype=np.uint16))  # temporary placeholder

    shift_ind, vdc_shifted = shift_vdc(vdc_vec)

    # guess the top (or last) node
    loop_guess_mat[-1] = generate_guess(vdc_vec, cluster_tree.tree.value)

    # Now guess the rest of the tree
    loop_guess_mat, loop_fit_results = _loop_fit_tree(cluster_tree.tree,
                                                      loop_guess_mat,
                                                      loop_fit_results,
                                                      vdc_shifted,
                                                      shift_ind)

    # Prepare guesses for each pixel using the fit of the cluster it belongs to:
    guess_parms = np.zeros(shape=projected_loops_2d.shape[0],
                           dtype=loop_fit32)
    for clust_id in range(num_clusters):
        pix_inds = np.where(labels == clust_id)[0]
        temp = np.atleast_2d(loop_fit_results[clust_id][0].x)
        # convert to the appropriate dtype as well:
        r2 = 1 - np.sum(np.abs(loop_fit_results[clust_id][0].fun ** 2))
        guess_parms[pix_inds] = stack_real_to_compound(
            np.hstack([temp, np.atleast_2d(r2)]), loop_fit32)

    return guess_parms


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