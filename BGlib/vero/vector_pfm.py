# -*- coding: utf-8 -*-
"""
Utiliities to enable plotting of vector PFM data acquired through Asylum Oxford Instruments Vero AFM

Created on Tue Nov 5 14:27:00 2024

@author: Marti Checa, Yongtao Liu, Rama Vasudevan (CNMS/ORNL)
"""
import SciFiReaders as sr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class VectorPFM(object):
    def __init__(self, data_path_dict, parameters_dict, apply_line_correction=False) -> None:
        """
        Vector PFM class that enables analysis of single frequency VERO PFM images taken with four distinct laser spot psitions
        This method is explained in this paper by Proksch and Wagner https://arxiv.org/pdf/2410.03340

        Input:  - data_path_dict (dict): Fields expected are 'left', 'right', 'before', 'after' with values being the path to the files
                                    Accepted file types are '.ibw' and sidpy.Dataset objects.
                - parameters_dict: parameters of the setup
                    #Experimental Parameters
                    r0=(2/3)
                    L= 225e-6
                    x0=14e-6
                    Gx=0.5
                    Gy=1
                - apply_line_correction (bool) (Default=False): Whether to apply line-by-line offset correction to the images

        """
        self.data_path_dict = data_path_dict
        self.parms_dict = parameters_dict
        self.line_correction = apply_line_correction
        self.data_dict = self.open_data()
        #process the data
        self._process_data()
    
    def _process_data(self):
        """
        Add the difference data into the data dictionary
        """
        r0=self.parms_dict['r0']
        L= self.parms_dict['L']
        x0=self.parms_dict['x0']
        Gx=self.parms_dict['Gx']
        Gy=self.parms_dict['Gy']

        # Compute difference and average images for amplitude and piezoresponse
        difference_left_right_amplitude = self.data_dict["Left"]['amplitude'] - self.data_dict["Right"]['amplitude']
        difference_past_before_amplitude = self.data_dict["Past"]['amplitude'] - self.data_dict["Before"]['amplitude']
        average_left_right_amplitude = (self.data_dict["Left"]['amplitude'] + self.data_dict["Right"]['amplitude']) / 2
        average_before_past_amplitude = (self.data_dict["Before"]['amplitude'] + self.data_dict["Past"]['amplitude']) / 2

        difference_left_right_piezoresponse = (self.data_dict["Left"]['piezoresponse'] - self.data_dict["Right"]['piezoresponse'])/(2*Gy)
        difference_past_before_piezoresponse = (((r0*L-x0)*self.data_dict["Past"]['piezoresponse']) - ((r0*L+x0)*self.data_dict["Before"]['piezoresponse']))/(2*r0*L*Gx)
        average_left_right_piezoresponse = (self.data_dict["Left"]['piezoresponse'] + self.data_dict["Right"]['piezoresponse']) / 2
        average_before_past_piezoresponse = (self.data_dict["Before"]['piezoresponse'] + self.data_dict["Past"]['piezoresponse']) / 2


    def plot_difference_data(self):

    def open_data(self):
        """
        Opens data pointed to by self.data_path_dict, converts to piezoresponse, and adds it to a data dictionary
        """
        files = self.data_path_dict
        data_dict = {}
        for label, file in files.items():
            amplitude_data = self.load_ibw_data(file, channel=1)  # Channel 1 for Amplitude
            phase_data = self.load_ibw_data(file, channel=3)     # Channel 3 for Phase

            if amplitude_data is not None and phase_data is not None:
                piezoresponse_data = self.compute_piezoresponse(amplitude_data, phase_data)
                if self.line_correction:

                data_dict[label] = {
                    'amplitude': amplitude_data,
                    'phase': phase_data,
                    'piezoresponse': piezoresponse_data
                }
            else:
                print(f"Failed to load data for {label}.")

        return data_dict
    
    def compute_piezoresponse(self, amplitude_data, phase_data)->np.ndarray:
        """
        Compute the piezoresponse using the formula: Piezoresponse = Amplitude * cos(Phase)

        Inputs:  
            - amplitude_data (np.ndarray): 2D PFM amplitude image as a numpy array
            - phase_data (np.ndarray): 2D PFM phase image as a numpy array
        Output: 
            - piezoresponse (np.ndarray)
        """

        # Convert phase from degrees to radians
        phase_radians = np.radians(phase_data)

        # Compute piezoresponse
        piezoresponse_data = amplitude_data * np.cos(phase_radians)

        return piezoresponse_data
    
    def line_by_line_offset_correction(self, image)->np.ndarray:
        """
        Perform a line by line offset correction on the image
        Input:  - image (np.ndarray).
        Output: - corrected_image: image after offset correction
        """

        return
    
    def _plot_image(self, data, title, cmap='viridis', vmin=None, vmax=None)->matplotlib.figure.Figure:
        """
        Plot a 2D image of the data with optional vmin and vmax for color scaling.
        
        Args:
            data (np.array): 2D array to plot.
            title (str): Title of the plot.
            cmap (str): Colormap to use for the plot. Default is 'inferno'.
            vmin (float, optional): Minimum value for color scaling. If None, it is calculated automatically.
            vmax (float, optional): Maximum value for color scaling. If None, it is calculated automatically.
        Returns:
            - fig: matplotlib.figure object
        """
        rotated_data = np.rot90(data)
        
        # Automatically determine vmin and vmax if not provided
        if vmin is None:
            vmin = np.min(rotated_data)
        if vmax is None:
            vmax = np.max(rotated_data)
        
        fig, axes = plt.subplots(figsize=(5, 5))
        im1 = axes.imshow(rotated_data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im1, label="Amplitude")
        axes.set_title(title)
        axes.set_xlabel("X Axis")
        axes.set_ylabel("Y Axis")

        return fig
