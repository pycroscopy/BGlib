@staticmethod

from .be_loop_fitter import BELoopFitter
from pyUSID.io.hdf_utils import get_unit_values, get_sort_order, \
    reshape_to_n_dims, create_empty_dataset, create_results_group, \
import time
import numpy as np

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
    if h5_main.dtype != 'sho32': #TODO: not sure if this is correct
        raise TypeError('Provided dataset is not a SHO results dataset.')

    if data_type.lower() == 'bepsdata':
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
        print('All but FORC spec rows: {}'.format(all_but_forc_rows))

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
            print('After reshaping to N dimensions: shape: '
                  '{}'.format(this_forc_nd.shape))
            print('Will reorder to slow-to-fast as: '
                  '{} and squeeze FORC out'.format(order_to_s2f))

        this_forc_nd_s2f = this_forc_nd.transpose(
            order_to_s2f).squeeze()  # squeeze out FORC
        # Need to account for niche case when number of positions = 1:
        if self.h5_main.shape[0] == 1:
            # Add back a singular dimension
            this_forc_nd_s2f = this_forc_nd_s2f.reshape([1] + list(this_forc_nd_s2f.shape))

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

def _get_existing_datasets(self):
    """
    The purpose of this function is to allow processes to resume from partly computed results
    Start with self.h5_results_grp
    """
    super(BELoopFitter, self)._get_existing_datasets()
    self.h5_projected_loops = self.h5_results_grp['Projected_Loops']
    self.h5_loop_metrics = self.h5_results_grp['Loop_Metrics']
    try:
        _ = self.h5_results_grp['Guess_Loop_Parameters']
    except KeyError:
        _ = self.extract_loop_parameters(self._h5_guess)
    try:
        # This has already been done by super
        _ = self.h5_results_grp['Fit']
        try:
            _ = self.h5_results_grp['Fit_Loop_Parameters']
        except KeyError:
            _ = self.extract_loop_parameters(self._h5_fit)
    except KeyError:
        pass