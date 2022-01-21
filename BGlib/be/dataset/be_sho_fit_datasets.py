from ..analysis import BELoopFitter
from .be_process_datasets import BEPSLoopsDataset
from sidpy.hdf.hdf_utils import get_auxiliary_datasets, get_attr
from pyUSID import USIDataset
from pyUSID.io.hdf_utils import reshape_to_n_dims
from sidpy.viz.plot_utils import plot_curves, plot_map_stack, get_cmap_object, plot_map, set_tick_font_size, \
    plot_complex_spectra
import os
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

#SHOBEPSDataset
#SHOBELINEDataset
#SHOcKPFMDataset
#SHOBERelaxDataset

#Main parent class

class SHOBEDataset(USIDataset):
    """
      Extension of the USIDataset object for SHO-Fitted Band-excitation (BE) datasets
      This includes various visualization routines
      Pass the sho-fitted USIDataset to make this into a SHOBEDataset
      Should be able to pass either HDF5dataset or USIDataset, because it will make it a USIDataset if not.
      """

    def __init__(self, h5_dataset):
        super(SHOBEDataset, self).__init__(h5_ref=h5_dataset)
        # Populate some data tags
        self.dataset_type = 'SHOBEDataset'
        self.parm_dict = list(self.file['/Measurement_000'].attrs)

class SHOBEPSDataset(SHOBEDataset):
    """
      Extension of the USIDataset object for SHO-Fitted Band-excitation (BE) datasets
      This includes various visualization routines
      Pass the sho-fitted USIDataset to make this into a SHOBEDataset
      Should be able to pass either HDF5dataset or USIDataset, because it will make it a USIDataset if not.
      """

    def __init__(self, h5_dataset):
        super(SHOBEPSDataset, self).__init__(h5_dataset)
        # Populate some data tags
        self.dataset_type = 'SHOBEPSDataset'

    def visualize_results(self, save_plots=True, show_plots=True, cmap=None,
                              expt_type=None, meas_type=None, field_mode=None):
        """
        Plots some loops, amplitude, phase maps for BE-Line and BEPS datasets.\n
        Note: The file MUST contain SHO fit guesses at the very least
        Parameters
        ----------
        self : HDF5 Dataset
            dataset to be plotted
        save_plots : (Optional) Boolean
            Whether or not to save plots to files in the same directory as the h5 file
        show_plots : (Optional) Boolean
            Whether or not to display the plots on the screen
        cmap : String, or matplotlib.colors.LinearSegmentedColormap object (Optional)
            Requested color map
        expt_type : str, Optional
            Type of data. This is an attribute written to the HDF5 file at the
            root level by either the translator or the acquisition software.
            Accepted values are: 'BEPSData', 'BELineData', 'BERelaxData',
            'cKPFMData'
            Default - this function will attempt to extract this metadata from the
            HDF5 file
        meas_type: str, Optional
            Type of measurement. Accepted values are:
             'AC modulation mode with time reversal' or 'DC modulation mode'
             This is an attribute embedded under the "Measurement" group with the
             following key: 'VS_mode'. Default - this function will attempt to
             extract this metadata from the HDF5 file
        field_mode : str, Optional
            Mode in which measurements were made. Accepted values are:
            'in and out-of-field',
            This is an attribute at the "Measurement" group under the following key:
            'VS_measure_in_field_loops'. Default - this function will attempt to
            extract this metadata from the HDF5 file
        Returns
        -------
        None
        """
        # TODO: This function needs to be cleaned up and modularized, not perform as many hard checks for attributes; rather accept attributes as kwargs; return the figure object instead of writing to file, etc.
        cmap = plt.cm.viridis

        def __plot_loops_maps(ac_vec, resp_mat, grp_name, win_title, spec_var_title, meas_var_title, save_plots,
                              folder_path, basename, num_rows, num_cols):
            if isinstance(grp_name, str):
                grp_name = grp_name + '_'
            else:
                grp_name = ''
            plt_title = grp_name + win_title + '_Loops'
            fig, ax = plot_curves(ac_vec, resp_mat, evenly_spaced=True, num_plots=25, x_label=spec_var_title,
                                  y_label=meas_var_title, subtitle_prefix='Position', title=plt_title)
            if save_plots:
                fig.savefig(os.path.join(folder_path, basename + '_' + plt_title + '.png'), format='png', dpi=300)

            plt_title = grp_name + '_' + win_title + '_Snaps'
            fig, axes = plot_map_stack(resp_mat.reshape(num_rows, num_cols, resp_mat.shape[1]),
                                       color_bar_mode="each", evenly_spaced=True, subtitle='UDVS Step #',
                                       title=plt_title, cmap=cmap)
            if save_plots:
                fig.savefig(os.path.join(folder_path, basename + '_' + plt_title + '.png'), format='png', dpi=300)

            return fig

        print('Creating plots of SHO Results from {}.'.format(self.name))

        h5_file = self.file

        if not isinstance(expt_type, str):
            expt_type = get_attr(h5_file, 'data_type')

        if expt_type not in ['BEPSData', 'BELineData', 'BERelaxData', 'cKPFMData']:
            warn('Unsupported data format')
            return

        isBEPS = expt_type != 'BELineData'

        (folder_path, basename) = os.path.split(h5_file.filename)
        basename, _ = os.path.splitext(basename)

        # This is OK
        sho_grp = self.parent

        # TODO: This makes too many assumptions about the file structure
        chan_grp = h5_file['/'.join(sho_grp.name[1:].split('/')[:2])]
        meas_grp = chan_grp.parent

        # TODO: This makes too many assumptions about the file structure
        grp_name = None
        if meas_type is None and field_mode is None:
            grp_name = '_'.join(chan_grp.name[1:].split('/'))
            grp_name = '_'.join([grp_name, sho_grp.name.split('/')[-1].split('-')[0], self.name.split('/')[-1]])

        try:
            h5_pos = self.h5_pos_inds
        except KeyError:
            print('No Position_Indices found as attribute of {}'.format(self.name))
            print('Rows and columns will be calculated from dataset shape.')
            num_rows = int(np.floor((np.sqrt(self.shape[0]))))
            num_cols = int(np.reshape(self, [num_rows, -1, self.shape[1]]).shape[1])
        else:
            num_rows, num_cols = self.pos_dim_sizes

        try:
            h5_spec_vals = h5_file[get_attr(self, 'Spectroscopic_Values')]
        except Exception:
            raise

        # Assume that there's enough memory to load all the guesses into memory
        amp_mat = self['Amplitude [V]'] * 1000  # convert to mV ahead of time
        freq_mat = self['Frequency [Hz]'] / 1000
        q_mat = self['Quality Factor']
        phase_mat = self['Phase [rad]']
        rsqr_mat = self['R2 Criterion']

        fig_list = list()
        if isBEPS:

            if not isinstance(meas_type, str):
                meas_type = meas_grp.attrs['VS_mode']

            # basically 3 kinds for now - DC/current, AC, UDVS - lets ignore this
            if meas_type == 'load user defined VS Wave from file':
                warn('Not handling custom experiments for now')
                # h5_file.close()
                return

            # Plot amplitude and phase maps at one or more UDVS steps
            if meas_type == 'AC modulation mode with time reversal':
                center = int(h5_spec_vals.shape[1] * 0.5)
                ac_vec = np.squeeze(h5_spec_vals[h5_spec_vals.attrs['AC_Amplitude']][:, 0:center])

                forw_resp = np.squeeze(amp_mat[:, slice(0, center)])
                rev_resp = np.squeeze(amp_mat[:, slice(center, None)])

                for win_title, resp_mat in zip(['Forward', 'Reverse'], [forw_resp, rev_resp]):
                    fig_list.append(
                        __plot_loops_maps(ac_vec, resp_mat, grp_name, win_title, 'AC Amplitude', 'Amplitude',
                                          save_plots, folder_path, basename, num_rows, num_cols))
            else:
                # plot loops at a few locations
                dc_vec = np.squeeze(h5_spec_vals[self.spec_dim_descriptors.index('DC_Offset (V)')])

                if not isinstance(field_mode, str):
                    field_mode = meas_grp.attrs['VS_measure_in_field_loops']

                if field_mode == 'in and out-of-field':

                    dc_vec = np.squeeze(dc_vec[slice(0, None, 2)])

                    in_phase = np.squeeze(phase_mat[:, slice(0, None, 2)])
                    in_amp = np.squeeze(amp_mat[:, slice(0, None, 2)])
                    out_phase = np.squeeze(phase_mat[:, slice(1, None, 2)])
                    out_amp = np.squeeze(amp_mat[:, slice(1, None, 2)])

                    for win_title, resp_mat in zip(['In_Field', 'Out_of_Field'],
                                                   [in_phase * in_amp, out_phase * out_amp]):
                        fig_list.append(__plot_loops_maps(dc_vec, resp_mat, grp_name, win_title, 'DC Bias',
                                                          'Piezoresponse (a.u.)', save_plots, folder_path,
                                                          basename, num_rows, num_cols))
                else:
                    fig_list.append(__plot_loops_maps(dc_vec, phase_mat * amp_mat, grp_name, '', 'DC Bias',
                                                      'Piezoresponse (a.u.)', save_plots, folder_path, basename,
                                                      num_rows, num_cols))

        else:  # BE-Line can only visualize the amplitude and phase maps:
            amp_mat = amp_mat.reshape(num_rows, num_cols)
            freq_mat = freq_mat.reshape(num_rows, num_cols)
            q_mat = q_mat.reshape(num_rows, num_cols)
            phase_mat = phase_mat.reshape(num_rows, num_cols)
            rsqr_mat = rsqr_mat.reshape(num_rows, num_cols)

            fig_ms, ax_ms = plot_map_stack(np.dstack((amp_mat, freq_mat, q_mat, phase_mat, rsqr_mat)).T,
                                           num_comps=5, color_bar_mode='each', title=grp_name,
                                           subtitle=['Amplitude (mV)', 'Frequency (kHz)', 'Quality Factor',
                                                     'Phase (deg)',
                                                     'R^2 Criterion'], cmap=cmap)

            fig_list.append(fig_ms)
            if save_plots:
                if grp_name is None:
                    grp_name = ''
                else:
                    grp_name = grp_name + '_'
                plt_path = os.path.join(folder_path, basename + '_' + grp_name + 'Maps.png')
                fig_ms.savefig(plt_path, format='png', dpi=300)

        if show_plots:
            plt.show()

        return fig_list

    def fit_loops(self, override = False):
        """
        Plots some loops, amplitude, phase maps for BE-Line and BEPS datasets.\n
        Note: The file MUST contain SHO fit guesses at the very least
        Parameters
        ----------
        self : SHOBEPSDataset
            SHO-Fitted dataset to be fit
        Returns
        -------
        fit_results:  BEPSFitDataset
            Dataset with results of Loop fitting
        """
        loop_fitter = BELoopFitter(self, be_data_type='BEPSData', vs_mode = 'DC modulation mode', vs_cycle_frac='full',
                                                  cores=None, h5_target_group=None,
                                                  verbose=True)
        loop_fitter.set_up_guess()
        h5_loop_guess = loop_fitter.do_guess(override=override)
        # Calling explicitely here since Fitter won't do it automatically
        h5_guess_loop_parms = loop_fitter.extract_loop_parameters(h5_loop_guess)

        loop_fitter.set_up_fit()
        h5_loop_fit = loop_fitter.do_fit(override=override)
        h5_loop_group = h5_loop_fit.parent

        return BEPSLoopsDataset(h5_loop_fit)

class SHOBELINEDataset(SHOBEDataset):
    """
      Extension of the USIDataset object for SHO-Fitted Band-excitation (BE) datasets
      This includes various visualization routines
      Pass the sho-fitted USIDataset to make this into a SHOBEDataset
      Should be able to pass either HDF5dataset or USIDataset, because it will make it a USIDataset if not.
      """

    def __init__(self, h5_dataset):
        super(SHOBELINEDataset, self).__init__(h5_ref=h5_dataset)
        # Populate some data tags
        self.dataset_type = 'SHOBELINEDataset'

    def visualize_results(self, save_plots=True, show_plots=True, cmap=None,
                              expt_type=None, meas_type=None, field_mode=None):
        """
        Plots some loops, amplitude, phase maps for BE-Line and BEPS datasets.\n
        Note: The file MUST contain SHO fit guesses at the very least
        Parameters
        ----------
        self : HDF5 Dataset
            dataset to be plotted
        save_plots : (Optional) Boolean
            Whether or not to save plots to files in the same directory as the h5 file
        show_plots : (Optional) Boolean
            Whether or not to display the plots on the screen
        cmap : String, or matplotlib.colors.LinearSegmentedColormap object (Optional)
            Requested color map
        expt_type : str, Optional
            Type of data. This is an attribute written to the HDF5 file at the
            root level by either the translator or the acquisition software.
            Accepted values are: 'BEPSData', 'BELineData', 'BERelaxData',
            'cKPFMData'
            Default - this function will attempt to extract this metadata from the
            HDF5 file
        meas_type: str, Optional
            Type of measurement. Accepted values are:
             'AC modulation mode with time reversal' or 'DC modulation mode'
             This is an attribute embedded under the "Measurement" group with the
             following key: 'VS_mode'. Default - this function will attempt to
             extract this metadata from the HDF5 file
        field_mode : str, Optional
            Mode in which measurements were made. Accepted values are:
            'in and out-of-field',
            This is an attribute at the "Measurement" group under the following key:
            'VS_measure_in_field_loops'. Default - this function will attempt to
            extract this metadata from the HDF5 file
        Returns
        -------
        None
        """
        # TODO: This function needs to be cleaned up and modularized, not perform as many hard checks for attributes; rather accept attributes as kwargs; return the figure object instead of writing to file, etc.
        cmap = plt.cm.viridis

        def __plot_loops_maps(ac_vec, resp_mat, grp_name, win_title, spec_var_title, meas_var_title, save_plots,
                              folder_path, basename, num_rows, num_cols):
            if isinstance(grp_name, str):
                grp_name = grp_name + '_'
            else:
                grp_name = ''
            plt_title = grp_name + win_title + '_Loops'
            fig, ax = plot_curves(ac_vec, resp_mat, evenly_spaced=True, num_plots=25, x_label=spec_var_title,
                                  y_label=meas_var_title, subtitle_prefix='Position', title=plt_title)
            if save_plots:
                fig.savefig(os.path.join(folder_path, basename + '_' + plt_title + '.png'), format='png', dpi=300)

            plt_title = grp_name + '_' + win_title + '_Snaps'
            fig, axes = plot_map_stack(resp_mat.reshape(num_rows, num_cols, resp_mat.shape[1]),
                                       color_bar_mode="each", evenly_spaced=True, subtitle='UDVS Step #',
                                       title=plt_title, cmap=cmap)
            if save_plots:
                fig.savefig(os.path.join(folder_path, basename + '_' + plt_title + '.png'), format='png', dpi=300)

            return fig

        print('Creating plots of SHO Results from {}.'.format(self.name))

        h5_file = self.file

        if not isinstance(expt_type, str):
            expt_type = get_attr(h5_file, 'data_type')

        if expt_type not in ['BEPSData', 'BELineData', 'BERelaxData', 'cKPFMData']:
            warn('Unsupported data format')
            return

        isBEPS = expt_type != 'BELineData'

        (folder_path, basename) = os.path.split(h5_file.filename)
        basename, _ = os.path.splitext(basename)

        # This is OK
        sho_grp = self.parent

        # TODO: This makes too many assumptions about the file structure
        chan_grp = h5_file['/'.join(sho_grp.name[1:].split('/')[:2])]
        meas_grp = chan_grp.parent

        # TODO: This makes too many assumptions about the file structure
        grp_name = None
        if meas_type is None and field_mode is None:
            grp_name = '_'.join(chan_grp.name[1:].split('/'))
            grp_name = '_'.join([grp_name, sho_grp.name.split('/')[-1].split('-')[0], self.name.split('/')[-1]])

        try:
            h5_pos = self.h5_pos_inds
        except KeyError:
            print('No Position_Indices found as attribute of {}'.format(self.name))
            print('Rows and columns will be calculated from dataset shape.')
            num_rows = int(np.floor((np.sqrt(self.shape[0]))))
            num_cols = int(np.reshape(self, [num_rows, -1, self.shape[1]]).shape[1])
        else:
            num_rows, num_cols = self.pos_dim_sizes

        try:
            h5_spec_vals = h5_file[get_attr(self, 'Spectroscopic_Values')]
        except Exception:
            raise

        # Assume that there's enough memory to load all the guesses into memory
        amp_mat = self['Amplitude [V]'] * 1000  # convert to mV ahead of time
        freq_mat = self['Frequency [Hz]'] / 1000
        q_mat = self['Quality Factor']
        phase_mat = self['Phase [rad]']
        rsqr_mat = self['R2 Criterion']

        fig_list = list()
        if isBEPS:

            if not isinstance(meas_type, str):
                meas_type = meas_grp.attrs['VS_mode']

            # basically 3 kinds for now - DC/current, AC, UDVS - lets ignore this
            if meas_type == 'load user defined VS Wave from file':
                warn('Not handling custom experiments for now')
                # h5_file.close()
                return

            # Plot amplitude and phase maps at one or more UDVS steps
            if meas_type == 'AC modulation mode with time reversal':
                center = int(h5_spec_vals.shape[1] * 0.5)
                ac_vec = np.squeeze(h5_spec_vals[h5_spec_vals.attrs['AC_Amplitude']][:, 0:center])

                forw_resp = np.squeeze(amp_mat[:, slice(0, center)])
                rev_resp = np.squeeze(amp_mat[:, slice(center, None)])

                for win_title, resp_mat in zip(['Forward', 'Reverse'], [forw_resp, rev_resp]):
                    fig_list.append(
                        __plot_loops_maps(ac_vec, resp_mat, grp_name, win_title, 'AC Amplitude', 'Amplitude',
                                          save_plots, folder_path, basename, num_rows, num_cols))
            else:
                # plot loops at a few locations
                dc_vec = np.squeeze(h5_spec_vals[self.spec_dim_descriptors.index('DC_Offset (V)')])

                if not isinstance(field_mode, str):
                    field_mode = meas_grp.attrs['VS_measure_in_field_loops']

                if field_mode == 'in and out-of-field':

                    dc_vec = np.squeeze(dc_vec[slice(0, None, 2)])

                    in_phase = np.squeeze(phase_mat[:, slice(0, None, 2)])
                    in_amp = np.squeeze(amp_mat[:, slice(0, None, 2)])
                    out_phase = np.squeeze(phase_mat[:, slice(1, None, 2)])
                    out_amp = np.squeeze(amp_mat[:, slice(1, None, 2)])

                    for win_title, resp_mat in zip(['In_Field', 'Out_of_Field'],
                                                   [in_phase * in_amp, out_phase * out_amp]):
                        fig_list.append(__plot_loops_maps(dc_vec, resp_mat, grp_name, win_title, 'DC Bias',
                                                          'Piezoresponse (a.u.)', save_plots, folder_path,
                                                          basename, num_rows, num_cols))
                else:
                    fig_list.append(__plot_loops_maps(dc_vec, phase_mat * amp_mat, grp_name, '', 'DC Bias',
                                                      'Piezoresponse (a.u.)', save_plots, folder_path, basename,
                                                      num_rows, num_cols))

        else:  # BE-Line can only visualize the amplitude and phase maps:
            amp_mat = amp_mat.reshape(num_rows, num_cols)
            freq_mat = freq_mat.reshape(num_rows, num_cols)
            q_mat = q_mat.reshape(num_rows, num_cols)
            phase_mat = phase_mat.reshape(num_rows, num_cols)
            rsqr_mat = rsqr_mat.reshape(num_rows, num_cols)

            fig_ms, ax_ms = plot_map_stack(np.dstack((amp_mat, freq_mat, q_mat, phase_mat, rsqr_mat)).T,
                                           num_comps=5, color_bar_mode='each', title=grp_name,
                                           subtitle=['Amplitude (mV)', 'Frequency (kHz)', 'Quality Factor',
                                                     'Phase (deg)',
                                                     'R^2 Criterion'], cmap=cmap)

            fig_list.append(fig_ms)
            if save_plots:
                if grp_name is None:
                    grp_name = ''
                else:
                    grp_name = grp_name + '_'
                plt_path = os.path.join(folder_path, basename + '_' + grp_name + 'Maps.png')
                fig_ms.savefig(plt_path, format='png', dpi=300)

        if show_plots:
            plt.show()

        return fig_list

class SHOcKPFMDataset(SHOBEDataset):
    """
    Extension of the SHOBEDataset object to allow for cKPFM datasets
    This includes various visualization and analaysis routines
    Pass the sho-fitted h5 dataset to make this into a cKPFMSHODataset
    high_voltage_amp: (int) (default = 1) (multiplication factor in case voltage amplifier was used).
    """

    def __init__(self, usid_dataset, high_voltage_amp=1):

        self.dset = usid_dataset
        super(SHOcKPFMDataset, self).__init__(h5_ref=usid_dataset)

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