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

