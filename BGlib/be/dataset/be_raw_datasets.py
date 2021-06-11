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
from .be_sho_fit_datasets import SHOBEDataset

#The following classes inherit from RawBEDataset
#RawBEPSDataset
#RawBELINEDataset
#RawBERelaxDataset

#This is the MAIN parent class for RawBEDatasets.
class RawBEDataset(USIDataset):

    """

    Extension of the USIDataset object for Raw Band-excitation (BE) datasets

    Pass the raw_data h5 dataset to make this into a RawBEDataset

    Should be able to pass either HDF5dataset or USIDataset

    This is never used on its own, it is simply a parent class to the actual RawBE Classes

    """

    def __init__(self, h5_dataset):

        super(RawBEDataset, self).__init__(h5_ref=h5_dataset)

        # Prepare the datasets
        self.dataset_type = 'RawBEDataset'
        self.parm_dict = list(self.file['/Measurement_000'].attrs)

#----From here, these are the child classes, each for specific modes----#

class RawBEPSDataset(RawBEDataset):

    def __init__(self, h5_dataset):
        super(RawBEPSDataset, self).__init__(h5_dataset)
        # Prepare the datasets
        self.dataset_type = 'RawBEDataset'

    def plot_spectrogram(self, px=None, method='all'):
        """
        This function will plot the spectrogram at a given pixel, or the mean spectrogram if no pixels are given

        Parameters
        ----------
        px : Int, pixel value to be plotted. If None, an average spectrogram will be plotted.
            For BELine datasets, px is the row number

        method: string: -'all': will plot real, imag, amp and phase of the spectrogram
                        - 'amp': will plot amplitude and phase
                        - 'real': will plot only the real and imaginary parts of the spectrogram
        """
        if px is not None:
            assert int(px) >= 0, "Pixel index provided must be positive"
            assert int(px) < self.shape[0], "Pixel index provided is greater than number of pixels in dataset"
            average_spect = False
        else:
            average_spect = True

        expt_type = get_attr(self.file, 'data_type')

        # Get the data
        freq_index = self.spec_dim_labels.index('Frequency')
        freq_len = self.spec_dim_sizes[freq_index]

        if not average_spect:
            real_spec = np.real(self[px, :]).reshape(-1, freq_len)
            imag_spec = np.imag(self[px, :]).reshape(-1, freq_len)
            amp_spec = np.abs(self[px, :]).reshape(-1, freq_len)
            phase_spec = np.angle(self[px, :]).reshape(-1, freq_len)
        else:
            real_spec = np.real(self[:, :]).mean(axis=0).reshape(-1, freq_len)
            imag_spec = np.imag(self[:, :]).mean(axis=0).reshape(-1, freq_len)
            amp_spec = np.abs(self[:, :]).mean(axis=0).reshape(-1, freq_len)
            phase_spec = np.angle(self[:, :]).mean(axis=0).reshape(-1, freq_len)

        data_spec = [real_spec, imag_spec, amp_spec, phase_spec]

        # Plot the data
        # TODO: Use plot map stack to do this!!

        if method == 'all':
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
            titles = ['Real', 'Imaginary', 'Amplitude', 'Phase']
            for ind, ax in enumerate(axes.flat):
                ax.imshow(data_spec[ind].T)
                if average_spect:
                    spec_title = 'Mean ' + titles[ind]
                else:
                    spec_title = titles[ind] + ' at px ' + str(px)

                ax.set_title(spec_title)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
            if method == 'amp':
                titles = ['Amplitude', 'Phase']
                for ind, ax in enumerate(axes.flat):
                    ax.imshow(data_spec[ind + 2].T)
                    if average_spect:
                        spec_title = 'Mean' + titles[ind]
                    else:
                        titles[ind] + 'at px ' + str(px)

                    ax.set_title(spec_title)
            elif method == 'real':
                titles = ['Real', 'Imaginary']
                for ind, ax in enumerate(axes.flat):
                    ax.imshow(data_spec[ind])
                    if average_spect:
                        spec_title = 'Mean' + titles[ind]
                    else:
                        titles[ind] + 'at px ' + str(px)

                    ax.set_title(spec_title)
        fig.tight_layout()
        return fig, data_spec

    def visualize_spectrograms(self, cmap=None):
        """
        Jupyer notebook ONLY function. Sets up a simple visualzier for visualizing raw BE data.
        Sliders for position indices can be used to visualize BE spectrograms (frequency, UDVS step).
        In the case of 2 spatial dimensions, a spatial map will be provided as well

        Parameters
        ----------
        cmap : matplotlib.colors.LinearSegmentedColormap object (Optional)
            Requested color map
        """
        if cmap is None: cmap = plt.cm.viridis

        h5_pos_inds = self.h5_pos_inds
        pos_dims = self.pos_dim_sizes
        pos_labels = self.pos_dim_labels

        h5_spec_vals = self.h5_spec_vals
        h5_spec_inds = self.h5_spec_inds
        spec_dims = self.spec_dim_sizes
        spec_labels = self.spec_dim_labels

        ifreq = spec_labels.index('Frequency')
        freqs_nd = reshape_to_n_dims(h5_spec_vals, h5_spec=h5_spec_inds)[0][ifreq].squeeze()
        freqs_2d = freqs_nd.reshape(freqs_nd.shape[0], -1) / 1000  # Convert to kHz

        num_udvs_steps = int(np.prod([spec_dims[idim] for idim in range(len(spec_dims)) if idim != ifreq]))

        if len(pos_dims) >= 2:
            # Build initial slice dictionaries
            spatial_slice_dict = {'X': slice(None), 'Y': slice(None)}
            for key in pos_labels:
                if key in spatial_slice_dict.keys():
                    continue
                else:
                    spatial_slice_dict[key] = [0]

            spectrogram_slice_dict = {key: [0] for key in pos_labels}

            spatial_slice, _ = self._get_pos_spec_slices(slice_dict=spatial_slice_dict)

            x_size = pos_dims[-1]
            y_size = pos_dims[-2]

            spatial_map = np.abs(np.reshape(self[spatial_slice, 0], (y_size, x_size)))
            spectrogram = np.reshape(self[0], (num_udvs_steps, -1))
            fig, axes = plt.subplots(ncols=3, figsize=(12, 4), subplot_kw={'adjustable': 'box'})
            spatial_img, spatial_cbar = plot_map(axes[0], np.abs(spatial_map), cmap=cmap)
            axes[0].set_aspect('equal')
            axes[0].set_xlabel(pos_labels[-1])
            axes[0].set_ylabel(pos_labels[-2])

            xdata = int(0.5 * x_size)
            ydata = int(0.5 * y_size)
            crosshair = axes[0].plot(xdata, ydata, 'k+')[0]


            amp_img, amp_cbar = plot_map(axes[1], np.abs(spectrogram), show_xy_ticks=True, cmap=cmap,
                                         extent=[freqs_2d[0, 0], freqs_2d[-1, 0], 0, num_udvs_steps])

            phase_img, phase_cbar = plot_map(axes[2], np.angle(spectrogram), show_xy_ticks=True, cmap=cmap,
                                             extent=[freqs_2d[0, 0], freqs_2d[-1, 0], 0, num_udvs_steps])
            phase_img.set_clim(vmin=-np.pi, vmax=np.pi)

            for axis in axes[1:3]:
                axis.set_ylabel('BE step')
                axis.axis('tight')
                x0, x1 = (freqs_2d[0, 0], freqs_2d[-1, 0])
                y0, y1 = (0, num_udvs_steps)
                axis.set_aspect(np.abs(x1 - x0) / np.abs(y1 - y0))

            pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                          pos_labels[-2] + ': ' + str(ydata) + ', '
            for dim_name in pos_labels[-3::-1]:
                pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '

            axes[1].set_title('Amplitude \n' + pos_heading)
            axes[1].set_xlabel('Frequency (kHz)')

            axes[2].set_title('Phase \n' + pos_heading)
            axes[2].set_xlabel('Frequency (kHz)')

            fig.tight_layout()

            fig_filename, _ = os.path.splitext(self.file.filename)
            display(save_fig_filebox_button(fig, fig_filename + '.png'))

            # Build sliders for any extra Position Dimensions
            pos_sliders = dict()
            for ikey, key in enumerate(pos_labels[:-2]):
                pos_sliders[key] = widgets.IntSlider(value=0, min=0, max=pos_dims[ikey] - 1,
                                                     step=1, description='{} Step:'.format(key),
                                                     continuous_update=False)

            def get_spatial_slice():
                xdata, ydata = crosshair.get_xydata().squeeze()
                spatial_slice_dict[pos_labels[-1]] = [int(xdata)]
                spatial_slice_dict[pos_labels[-2]] = [int(ydata)]
                for key in pos_labels[:-2]:
                    spatial_slice_dict[key] = [pos_sliders[key].value]

                spatial_slice, _ = self._get_pos_spec_slices(slice_dict=spatial_slice_dict)

                return spatial_slice

            def spec_index_unpacker(step):
                spatial_slice_dict[pos_labels[-1]] = slice(None)
                spatial_slice_dict[pos_labels[-2]] = slice(None)
                for key in pos_labels[:-2]:
                    spatial_slice_dict[key] = [pos_sliders[key].value]

                spatial_slice, _ = self._get_pos_spec_slices(slice_dict=spatial_slice_dict)

                spatial_map = np.abs(np.reshape(self[spatial_slice, step], (x_size, y_size)))
                spatial_img.set_data(spatial_map)
                spat_mean = np.mean(spatial_map)
                spat_std = np.std(spatial_map)
                spatial_img.set_clim(vmin=spat_mean - 3 * spat_std, vmax=spat_mean + 3 * spat_std)

                spec_heading = ''
                for dim_ind, dim_name in enumerate(spec_labels):
                    spec_heading += dim_name + ': ' + str(h5_spec_vals[dim_ind, step]) + ', '
                axes[0].set_title(spec_heading[:-2])
                fig.canvas.draw()

            def pos_picker(event):
                if not spatial_img.axes.in_axes(event):
                    return

                xdata = int(round(event.xdata))
                ydata = int(round(event.ydata))

                crosshair.set_xdata(xdata)
                crosshair.set_ydata(ydata)

                spatial_slice = get_spatial_slice()

                pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                              pos_labels[-2] + ': ' + str(ydata) + ', '
                for dim_name in pos_labels[-3::-1]:
                    pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '
                axes[1].set_title('Amplitude \n' + pos_heading)
                axes[2].set_title('Phase \n' + pos_heading)

                spectrogram = np.reshape(self[spatial_slice, :], (num_udvs_steps, -1))

                if len(spec_dims) > 1:
                    amp_map = np.abs(spectrogram)
                    amp_img.set_data(np.abs(spectrogram))
                    phase_img.set_data(np.angle(spectrogram))
                    amp_mean = np.mean(amp_map)
                    amp_std = np.std(amp_map)
                    amp_img.set_clim(vmin=amp_mean - 3 * amp_std, vmax=amp_mean + 3 * amp_std)
                else:
                    amp_img.set_ydata(np.abs(spectrogram))
                    phase_img.set_ydata(np.angle(spectrogram))
                amp_cbar.changed()
                phase_cbar.changed()

                fig.canvas.draw()

            def pos_slider_update(slider):
                spatial_slice = get_spatial_slice()
                step = spec_index_slider.value

                spec_index_unpacker(step)

                pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                              pos_labels[-2] + ': ' + str(ydata) + ', '
                for dim_name in pos_labels[-3::-1]:
                    pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '
                axes[1].set_title('Amplitude \n' + pos_heading)
                axes[2].set_title('Phase \n' + pos_heading)

                spectrogram = np.reshape(self[spatial_slice, :], (num_udvs_steps, -1))

                if len(spec_dims) > 1:
                    amp_img.set_data(np.abs(spectrogram))
                    phase_img.set_data(np.angle(spectrogram))
                else:
                    amp_img.set_ydata(np.abs(spectrogram))
                    phase_img.set_ydata(np.angle(spectrogram))
                amp_cbar.changed()
                phase_cbar.changed()

                fig.canvas.draw()

            spec_index_slider = widgets.IntSlider(value=0, min=0, max=self.shape[1], step=1,
                                                  description='Step')
            cid = spatial_img.figure.canvas.mpl_connect('button_press_event', pos_picker)
            widgets.interact(spec_index_unpacker, step=spec_index_slider)
            for key, slider in pos_sliders.items():
                widgets.interact(pos_slider_update, slider=slider)
            # plt.show()

        else:
            def plot_spectrogram(data, freq_vals):
                fig, axes = plt.subplots(ncols=2, figsize=(9, 5), sharey=True)
                im_handles = list()
                im_handles.append(axes[0].imshow(np.abs(data), cmap=cmap,
                                                 extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                                         data.shape[0], 0],
                                                 interpolation='none'))
                axes[0].set_title('Amplitude')
                axes[0].set_ylabel('BE step')
                im_handles.append(axes[1].imshow(np.angle(data), cmap=cmap,
                                                 extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                                         data.shape[0], 0],
                                                 interpolation='none'))
                axes[1].set_title('Phase')
                axes[0].set_xlabel('Frequency index')
                axes[1].set_xlabel('Frequency index')
                for axis in axes:
                    axis.axis('tight')
                    axis.set_ylim(0, data.shape[0])
                fig.tight_layout()
                return fig, axes, im_handles

            fig, axes, im_handles = plot_spectrogram(np.reshape(self[0], (num_udvs_steps, -1)), freqs_2d)

            def position_unpacker(**kwargs):
                pos_dim_vals = range(len(pos_labels))
                for pos_dim_ind, pos_dim_name in enumerate(pos_labels):
                    pos_dim_vals[pos_dim_ind] = kwargs[pos_dim_name]
                pix_ind = pos_dim_vals[0]
                for pos_dim_ind in range(1, len(pos_labels)):
                    pix_ind += pos_dim_vals[pos_dim_ind] * pos_dims[pos_dim_ind - 1]
                spectrogram = np.reshape(self[pix_ind], (num_udvs_steps, -1))
                im_handles[0].set_data(np.abs(spectrogram))
                im_handles[1].set_data(np.angle(spectrogram))
                display(fig)

            pos_dict = dict()
            for pos_dim_ind, dim_name in enumerate(pos_labels):
                pos_dict[dim_name] = (0, pos_dims[pos_dim_ind] - 1, 1)

            widgets.interact(position_unpacker, **pos_dict)
            display(fig)

        return fig

    def perform_SHO_fit(self, sho_fit_points=5, sho_override=False, max_cores=None,
                        h5_sho_targ_grp=None, guess_func=bsho.SHOGuessFunc.complex_gaussian):
        '''h5_path  - group or name of file to write to. If None '''
        if type(h5_sho_targ_grp) == str:
            f_open_mode = 'w'
            if os.path.exists(h5_sho_targ_grp):
                f_open_mode = 'r+'
            h5_sho_file = h5py.File(h5_sho_targ_grp, mode=f_open_mode)
            h5_sho_targ_grp = h5_sho_file

        sho_fit_points = 5  # The number of data points at each step to use when fitting
        sho_override = False  # Force recompute if True

        h5_sho_targ_grp = None

        sho_fitter = BESHOfitter(self, cores=max_cores,
                                 verbose=False, h5_target_group=h5_sho_targ_grp)

        sho_fitter.set_up_guess(guess_func=guess_func,
                                num_points=sho_fit_points)

        h5_sho_guess = sho_fitter.do_guess(override=sho_override)
        sho_fitter.set_up_fit()
        h5_sho_fit = sho_fitter.do_fit(override=sho_override)

        # TODO: return the be_sho_dataset object

        return SHOBEDataset(h5_sho_fit)

class RawBELINEDataset(RawBEDataset):

    def __init__(self, h5_dataset):
        super(RawBELINEDataset, self).__init__(h5_dataset)
        # Prepare the datasets
        self.dataset_type = 'RawBELINEDataset'

    def plot_spectrogram(self, row=None, method='all'):
        """
        This function will plot the spectrogram at a given row, or the meaned (across rows) spectrogram if no rows are given

        Parameters
        ----------
        row : Int, row to be plotted. If None, an average spectrogram will be plotted.


        method: string: -'all': will plot real, imag, amp and phase of the spectrogram
                        - 'amp': will plot amplitude and phase
                        - 'real': will plot only the real and imaginary parts of the spectrogram
        """
        if row is not None:
            assert int(row) >= 0, "Pixel index provided must be positive"
            assert int(row) < self.shape[0], "Pixel index provided is greater than number of pixels in dataset"
            average_spect = False
        else:
            average_spect = True

        expt_type = get_attr(self.file, 'data_type')

        # Get the data
        freq_index = self.spec_dim_labels.index('Frequency')
        freq_len = self.spec_dim_sizes[freq_index]

        if not average_spect:
            real_spec = np.real(self[row * self.pos_dim_sizes[0]:(row + 1) * self.pos_dim_sizes[1], :])
            imag_spec = np.imag(self[row * self.pos_dim_sizes[0]:(row + 1) * self.pos_dim_sizes[1], :])
            amp_spec = np.abs(self[row * self.pos_dim_sizes[0]:(row + 1) * self.pos_dim_sizes[1], :])
            phase_spec = np.angle(self[row * self.pos_dim_sizes[0]:(row + 1) * self.pos_dim_sizes[1], :])
        else:
            dset_ndim = self.get_n_dim_form()
            real_spec = np.real(dset_ndim[:]).mean(axis=0)
            imag_spec = np.imag(dset_ndim[:]).mean(axis=0)
            amp_spec = np.abs(dset_ndim[:]).mean(axis=0)
            phase_spec = np.angle(dset_ndim[:]).mean(axis=0)

        data_spec = [real_spec, imag_spec, amp_spec, phase_spec]

        # Plot the data
        # TODO: Use plot map stack to do this!!

        if method == 'all':
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
            titles = ['Real', 'Imaginary', 'Amplitude', 'Phase']
            for ind, ax in enumerate(axes.flat):
                ax.imshow(data_spec[ind].T)
                if average_spect:
                    spec_title = 'Mean ' + titles[ind]
                else:
                    spec_title = titles[ind] + ' at row ' + str(row)

                ax.set_title(spec_title)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
            if method == 'amp':
                titles = ['Amplitude', 'Phase']
                for ind, ax in enumerate(axes.flat):
                    ax.imshow(data_spec[ind + 2].T)
                    if average_spect:
                        spec_title = 'Mean' + titles[ind]
                    else:
                        titles[ind] + 'at row ' + str(row)
                    ax.set_title(spec_title)
            elif method == 'real':
                titles = ['Real', 'Imaginary']
                for ind, ax in enumerate(axes.flat):
                    ax.imshow(data_spec[ind])
                    if average_spect:
                        spec_title = 'Mean' + titles[ind]
                    else:
                        titles[ind] + 'at row ' + str(row)

                    ax.set_title(spec_title)
        fig.tight_layout()
        return fig, data_spec

    def visualize_spectrograms(self, cmap=None):
        """
        Jupyer notebook ONLY function. Sets up a simple visualzier for visualizing raw BE data.
        Sliders for position indices can be used to visualize BE spectrograms (frequency, UDVS step).
        In the case of 2 spatial dimensions, a spatial map will be provided as well

        Parameters
        ----------
        cmap : matplotlib.colors.LinearSegmentedColormap object (Optional)
            Requested color map
        """
        if cmap is None: cmap = plt.cm.viridis

        h5_pos_inds = self.h5_pos_inds
        pos_dims = self.pos_dim_sizes
        pos_labels = self.pos_dim_labels

        h5_spec_vals = self.h5_spec_vals
        h5_spec_inds = self.h5_spec_inds
        spec_dims = self.spec_dim_sizes
        spec_labels = self.spec_dim_labels

        ifreq = spec_labels.index('Frequency')
        freqs_nd = reshape_to_n_dims(h5_spec_vals, h5_spec=h5_spec_inds)[0][ifreq].squeeze()
        freqs_2d = freqs_nd.reshape(freqs_nd.shape[0], -1) / 1000  # Convert to kHz

        num_udvs_steps = int(np.prod([spec_dims[idim] for idim in range(len(spec_dims)) if idim != ifreq]))

        if len(pos_dims) >= 2:
            # Build initial slice dictionaries
            spatial_slice_dict = {'X': slice(None), 'Y': slice(None)}
            for key in pos_labels:
                if key in spatial_slice_dict.keys():
                    continue
                else:
                    spatial_slice_dict[key] = [0]

            spectrogram_slice_dict = {key: [0] for key in pos_labels}

            spatial_slice, _ = self._get_pos_spec_slices(slice_dict=spatial_slice_dict)

            x_size = pos_dims[-1]
            y_size = pos_dims[-2]

            spatial_map = np.abs(np.reshape(self[spatial_slice, 0], (y_size, x_size)))
            spectrogram = np.reshape(self[0], (num_udvs_steps, -1))
            fig, axes = plt.subplots(ncols=3, figsize=(12, 4), subplot_kw={'adjustable': 'box'})
            spatial_img, spatial_cbar = plot_map(axes[0], np.abs(spatial_map), cmap=cmap)
            axes[0].set_aspect('equal')
            axes[0].set_xlabel(pos_labels[-1])
            axes[0].set_ylabel(pos_labels[-2])

            xdata = int(0.5 * x_size)
            ydata = int(0.5 * y_size)
            crosshair = axes[0].plot(xdata, ydata, 'k+')[0]

            # BE-Line
            axes[1].set_ylabel('Amplitude (a. u.)')
            axes[2].set_ylabel('Phase (rad)')
            spectrogram = np.squeeze(spectrogram)
            amp_img = axes[1].plot(np.abs(spectrogram))[0]
            phase_img = axes[2].plot(np.angle(spectrogram))[0]
            amp_full = np.abs(self[()])
            amp_mean = np.mean(amp_full)
            amp_std = np.std(amp_full)
            st_devs = 4

            axes[1].set_ylim([0, amp_mean + st_devs * amp_std])
            axes[2].set_ylim([-np.pi, np.pi])

            pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                          pos_labels[-2] + ': ' + str(ydata) + ', '
            for dim_name in pos_labels[-3::-1]:
                pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '

            axes[1].set_title('Amplitude \n' + pos_heading)
            axes[1].set_xlabel('Frequency (kHz)')

            axes[2].set_title('Phase \n' + pos_heading)
            axes[2].set_xlabel('Frequency (kHz)')

            fig.tight_layout()

            fig_filename, _ = os.path.splitext(self.file.filename)
            display(save_fig_filebox_button(fig, fig_filename + '.png'))

            # Build sliders for any extra Position Dimensions
            pos_sliders = dict()
            for ikey, key in enumerate(pos_labels[:-2]):
                pos_sliders[key] = widgets.IntSlider(value=0, min=0, max=pos_dims[ikey] - 1,
                                                     step=1, description='{} Step:'.format(key),
                                                     continuous_update=False)

            def get_spatial_slice():
                xdata, ydata = crosshair.get_xydata().squeeze()
                spatial_slice_dict[pos_labels[-1]] = [int(xdata)]
                spatial_slice_dict[pos_labels[-2]] = [int(ydata)]
                for key in pos_labels[:-2]:
                    spatial_slice_dict[key] = [pos_sliders[key].value]

                spatial_slice, _ = self._get_pos_spec_slices(slice_dict=spatial_slice_dict)

                return spatial_slice

            def spec_index_unpacker(step):
                spatial_slice_dict[pos_labels[-1]] = slice(None)
                spatial_slice_dict[pos_labels[-2]] = slice(None)
                for key in pos_labels[:-2]:
                    spatial_slice_dict[key] = [pos_sliders[key].value]

                spatial_slice, _ = self._get_pos_spec_slices(slice_dict=spatial_slice_dict)

                spatial_map = np.abs(np.reshape(self[spatial_slice, step], (x_size, y_size)))
                spatial_img.set_data(spatial_map)
                spat_mean = np.mean(spatial_map)
                spat_std = np.std(spatial_map)
                spatial_img.set_clim(vmin=spat_mean - 3 * spat_std, vmax=spat_mean + 3 * spat_std)

                spec_heading = ''
                for dim_ind, dim_name in enumerate(spec_labels):
                    spec_heading += dim_name + ': ' + str(h5_spec_vals[dim_ind, step]) + ', '
                axes[0].set_title(spec_heading[:-2])
                fig.canvas.draw()

            def pos_picker(event):
                if not spatial_img.axes.in_axes(event):
                    return

                xdata = int(round(event.xdata))
                ydata = int(round(event.ydata))

                crosshair.set_xdata(xdata)
                crosshair.set_ydata(ydata)

                spatial_slice = get_spatial_slice()

                pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                              pos_labels[-2] + ': ' + str(ydata) + ', '
                for dim_name in pos_labels[-3::-1]:
                    pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '
                axes[1].set_title('Amplitude \n' + pos_heading)
                axes[2].set_title('Phase \n' + pos_heading)

                spectrogram = np.reshape(self[spatial_slice, :], (num_udvs_steps, -1))

                if len(spec_dims) > 1:
                    amp_map = np.abs(spectrogram)
                    amp_img.set_data(np.abs(spectrogram))
                    phase_img.set_data(np.angle(spectrogram))
                    amp_mean = np.mean(amp_map)
                    amp_std = np.std(amp_map)
                    amp_img.set_clim(vmin=amp_mean - 3 * amp_std, vmax=amp_mean + 3 * amp_std)
                else:
                    amp_img.set_ydata(np.abs(spectrogram))
                    phase_img.set_ydata(np.angle(spectrogram))
                #amp_cbar.changed()
                #phase_cbar.changed()

                fig.canvas.draw()

            def pos_slider_update(slider):
                spatial_slice = get_spatial_slice()
                step = spec_index_slider.value

                spec_index_unpacker(step)

                pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                              pos_labels[-2] + ': ' + str(ydata) + ', '
                for dim_name in pos_labels[-3::-1]:
                    pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '
                axes[1].set_title('Amplitude \n' + pos_heading)
                axes[2].set_title('Phase \n' + pos_heading)

                spectrogram = np.reshape(self[spatial_slice, :], (num_udvs_steps, -1))

                if len(spec_dims) > 1:
                    amp_img.set_data(np.abs(spectrogram))
                    phase_img.set_data(np.angle(spectrogram))
                else:
                    amp_img.set_ydata(np.abs(spectrogram))
                    phase_img.set_ydata(np.angle(spectrogram))
                #amp_cbar.changed()
                #phase_cbar.changed()

                fig.canvas.draw()

            spec_index_slider = widgets.IntSlider(value=0, min=0, max=self.shape[1], step=1,
                                                  description='Step')
            cid = spatial_img.figure.canvas.mpl_connect('button_press_event', pos_picker)
            widgets.interact(spec_index_unpacker, step=spec_index_slider)
            for key, slider in pos_sliders.items():
                widgets.interact(pos_slider_update, slider=slider)
            # plt.show()

        else:
            def plot_spectrogram(data, freq_vals):
                fig, axes = plt.subplots(ncols=2, figsize=(9, 5), sharey=True)
                im_handles = list()
                im_handles.append(axes[0].imshow(np.abs(data), cmap=cmap,
                                                 extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                                         data.shape[0], 0],
                                                 interpolation='none'))
                axes[0].set_title('Amplitude')
                axes[0].set_ylabel('BE step')
                im_handles.append(axes[1].imshow(np.angle(data), cmap=cmap,
                                                 extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                                         data.shape[0], 0],
                                                 interpolation='none'))
                axes[1].set_title('Phase')
                axes[0].set_xlabel('Frequency index')
                axes[1].set_xlabel('Frequency index')
                for axis in axes:
                    axis.axis('tight')
                    axis.set_ylim(0, data.shape[0])
                fig.tight_layout()
                return fig, axes, im_handles

            fig, axes, im_handles = plot_spectrogram(np.reshape(self[0], (num_udvs_steps, -1)), freqs_2d)

            def position_unpacker(**kwargs):
                pos_dim_vals = range(len(pos_labels))
                for pos_dim_ind, pos_dim_name in enumerate(pos_labels):
                    pos_dim_vals[pos_dim_ind] = kwargs[pos_dim_name]
                pix_ind = pos_dim_vals[0]
                for pos_dim_ind in range(1, len(pos_labels)):
                    pix_ind += pos_dim_vals[pos_dim_ind] * pos_dims[pos_dim_ind - 1]
                spectrogram = np.reshape(self[pix_ind], (num_udvs_steps, -1))
                im_handles[0].set_data(np.abs(spectrogram))
                im_handles[1].set_data(np.angle(spectrogram))
                display(fig)

            pos_dict = dict()
            for pos_dim_ind, dim_name in enumerate(pos_labels):
                pos_dict[dim_name] = (0, pos_dims[pos_dim_ind] - 1, 1)

            widgets.interact(position_unpacker, **pos_dict)
            display(fig)

        return fig

    def perform_SHO_fit(self, sho_fit_points=5, sho_override=False, max_cores=None,
                        h5_sho_targ_grp=None, guess_func=bsho.SHOGuessFunc.complex_gaussian):
        '''h5_path  - group or name of file to write to. If None '''
        if type(h5_sho_targ_grp) == str:
            f_open_mode = 'w'
            if os.path.exists(h5_sho_targ_grp):
                f_open_mode = 'r+'
            h5_sho_file = h5py.File(h5_sho_targ_grp, mode=f_open_mode)
            h5_sho_targ_grp = h5_sho_file

        sho_fit_points = 5  # The number of data points at each step to use when fitting
        sho_override = False  # Force recompute if True

        h5_sho_targ_grp = None

        sho_fitter = BESHOfitter(self, cores=max_cores,
                                 verbose=False, h5_target_group=h5_sho_targ_grp)

        sho_fitter.set_up_guess(guess_func=guess_func,
                                num_points=sho_fit_points)

        h5_sho_guess = sho_fitter.do_guess(override=sho_override)
        sho_fitter.set_up_fit()
        h5_sho_fit = sho_fitter.do_fit(override=sho_override)

        return h5_sho_fit #SHOBELINEDataset(h5_sho_fit)

class RawcKPFMDataset(RawBEDataset):
    """

    Extension of the RawBEDataset object to allow for cKPFM datasets
    This includes various visualization and analaysis routines

    Pass the raw_data h5 dataset to make this into a cKPFMRawDataset

    high_voltage_amp: (int) (default = 1) (multiplication factor in case voltage amplifier was used).

    """

    def __init__(self, h5_dataset, high_voltage_amp=1):
        super(RawcKPFMDataset, self).__init__(h5_dataset)
        # Prepare the datasets
        self.dataset_type = 'RawcKPFMDataset'

        # Prepare the datasets
        self._dataset_type = 'cKPFM-USIDataset'
        self.num_write_steps = self._parm_dict['VS_num_DC_write_steps']
        self.num_read_steps = self._parm_dict['VS_num_read_steps']
        self.num_fields = 2
        self.high_voltage_amp = high_voltage_amp

        return

    def plot_spectrogram(self, static = False):
        print('here we can plot out stuff')
        # TODO: add a visualization routine for raw cKPFM datasets here
        # This would basically be similar to the existing RawBEDataset visualizer,
        # except that it would also plot the cKPFM waveform
