import numpy as np
import matplotlib.pyplot as plt

from pyUSID.hdf_utils import reshape_to_n_dims
from .be_raw_dataset import RawBEDataset
from .be_sho_dataset import SHOBEDataset
from sidpy.viz.plot_utils.curve import cbar_for_line_plot

class cKPFMRawDataset(RawBEDataset):
    """

    Extension of the RawBEDataset object to allow for cKPFM datasets
    This includes various visualization and analaysis routines

    Pass the raw_data h5 dataset to make this into a cKPFMRawDataset

    high_voltage_amp: (int) (default = 1) (multiplication factor in case voltage amplifier was used).

    """

    def __init__(self, usid_dataset, high_voltage_amp=1):


        super(cKPFMRawDataset, self).__init__(h5_ref=usid_dataset)

        # Prepare the datasets
        self._dataset_type = 'cKPFM-USIDataset'
        self._parm_dict = self.file['/Measurement_000'].attrs
        self.num_write_steps = self._parm_dict['VS_num_DC_write_steps']
        self.num_read_steps = self._parm_dict['VS_num_read_steps']
        self.num_fields = 2
        self.high_voltage_amp = high_voltage_amp

        return

    @staticmethod


    def plot_spectrogram(self, static = False):
        print('here we can plot out stuff')
        # TODO: add a visualization routine for raw cKPFM datasets here
        # This would basically be similar to the existing RawBEDataset visualizer,
        # except that it would also plot the cKPFM waveform




class cKPFMSHODataset(SHOBEDataset):
    """

    Extension of the SHOBEDataset object to allow for cKPFM datasets
    This includes various visualization and analaysis routines

    Pass the sho-fitted h5 dataset to make this into a cKPFMSHODataset

    high_voltage_amp: (int) (default = 1) (multiplication factor in case voltage amplifier was used).

    """

    def __init__(self, usid_dataset, high_voltage_amp=1):

        self.dset = usid_dataset
        super(cKPFMSHODataset, self).__init__(h5_ref=usid_dataset)

        # Prepare the datasets
        self.dataset_type = 'cKPFM-USIDataset'
        self.parm_dict = self.dset.file['/Measurement_000'].attrs
        self.num_write_steps = self.parm_dict['VS_num_DC_write_steps']
        self.num_read_steps = self.parm_dict['VS_num_read_steps']
        self.num_fields = 2
        self.high_voltage_amp = high_voltage_amp

        return

    def process_sho_data(self, phase_offset=None, option=1):
        '''This method will process SHO data, adjusting the phase and setting up the Ndimensional matrices
        that will make plotting easier down the line
        Option = 1: for plotting only minimum to maximum voltage
        Option = 2: for plotting entire DC voltage range
        '''

        Nd_mat = self.h5_sho_fit.get_n_dim_form()

        phase_offset = det_phase_offset(Nd_mat[:, :, :, :, :]['Phase [rad]'].ravel())

        Nd_mat[:, :, :, :, :]['Phase [rad]'] = Nd_mat[:, :, :, :, :]['Phase [rad]'] - phase_offset

        self.nd_mat = Nd_mat

        self.h5_sho_spec_inds = self.h5_sho_fit.h5_spec_inds
        self.h5_sho_spec_vals = self.h5_sho_fit.h5_spec_vals

        self.sho_spec_labels = self.h5_sho_fit.spec_dim_labels
        self.pos_labels = self.h5_sho_fit.pos_dim_labels

        self.num_fields = self.h5_sho_fit.spec_dim_sizes[self.h5_sho_fit.spec_dim_labels.index('Field')]
        self.num_write_steps = self.h5_sho_fit.spec_dim_sizes[self.h5_sho_fit.spec_dim_labels.index('write_bias')]
        self.num_read_steps = self.h5_sho_fit.spec_dim_sizes[self.h5_sho_fit.spec_dim_labels.index('read_bias')]

        # It turns out that the read voltage index starts from 1 instead of 0
        # Also the VDC indices are NOT repeating. They are just rising monotonically

        self.write_volt_index = self.sho_spec_labels.index('write_bias')
        self.read_volt_index = self.sho_spec_labels.index('read_bias')
        self.h5_sho_spec_inds[self.read_volt_index, :] -= np.min(self.h5_sho_spec_inds[self.read_volt_index, :])
        self.h5_sho_spec_inds[self.write_volt_index, :] = np.tile(np.repeat(np.arange(self.num_write_steps),
                                                                            self.num_fields), self.num_read_steps)

        # Get the bias matrix:
        self.bias_mat, _ = reshape_to_n_dims(self.h5_sho_spec_vals, h5_spec=self.h5_sho_spec_inds)
        self.bias_vec_r_all = self.bias_mat[self.read_volt_index] * self.high_voltage_amp
        self.bias_vec_w_all = self.bias_mat[self.write_volt_index] * self.high_voltage_amp
        self.bias_vec_w = self.bias_vec_w_all.reshape(self.h5_sho_fit.spec_dim_sizes)[1, :, 1]
        self.bias_vec_r = self.bias_vec_r_all.reshape(self.h5_sho_fit.spec_dim_sizes)[1, :, :]

        # Option 1: only show curves from maximum to minimum write voltage:
        if option == 1:
            self.write_step_start = np.argmax(self.bias_vec_w)
            self.write_step_end = np.argmin(self.bias_vec_w)

        # option 2: show all curves from the whole write voltage waveform
        if option == 2:
            self.write_step_start = 0
            self.write_step_end = self.num_write_steps - 1


    @staticmethod
    def plot_line_family(axis, x_vec, line_family, line_names=None, label_prefix='', label_suffix='',
                         y_offset=0, show_cbar=False, **kwargs):
        """
        Plots a family of lines with a sequence of colors

        Parameters
        ----------
        axis : matplotlib.axes.Axes object
            Axis to plot the curve
        x_vec : array-like
            Values to plot against
        line_family : 2D numpy array
            family of curves arranged as [curve_index, features]
        line_names : array-like
            array of string or numbers that represent the identity of each curve in the family
        label_prefix : string / unicode
            prefix for the legend (before the index of the curve)
        label_suffix : string / unicode
            suffix for the legend (after the index of the curve)
        y_offset : (optional) number
            quantity by which the lines are offset from each other vertically (useful for spectra)
        show_cbar : (optional) bool
            Whether or not to show a colorbar (instead of a legend)

        """

        x_vec = np.array(x_vec)

        if not isinstance(line_family, list):
            line_family = np.array(line_family)

        assert line_family.ndim == 2, 'line_family must be a 2D array'

        num_lines = line_family.shape[0]

        if line_names is None:
            # label_prefix = 'Line '
            line_names = [str(line_ind) for line_ind in range(num_lines)]

        line_names = ['{} {} {}'.format(label_prefix, cur_name, label_suffix) for cur_name in line_names]

        print("Line family shape is {}".format(line_family.shape))

        for line_ind in range(num_lines):
            colors = plt.cm.get_cmap('jet', line_family.shape[-1])
            axis.plot(x_vec, line_family[line_ind] + line_ind * y_offset,
                      color=colors(line_ind),
                      )

        if show_cbar:
            # put back the cmap parameter:
            kwargs.update({'cmap': cmap})
            _ = cbar_for_line_plot(axis, num_lines, **kwargs)

    def plot_cKPFM_static(self):
        print('here we can plot out stuff')