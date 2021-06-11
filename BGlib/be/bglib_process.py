#base packages

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

import sidpy as sid
import pyNSID as nsid
import pyUSID as usid
from bglib_fitter import *
from dask.distributed import Client, progress
import dask

#TODO: Process class should include more def for other modes (gmode, trKPFM)
class BGlibProcess():
    """
    BGlib Process class, branches out to Fitter and Guesser class.
    Must provide h5 dataset and the guess method.
    """
#TODO: Where do the fitters sit within the Process class?
    def __init__(self, h5_main, threads = 4, num_workers = 1):
        self.h5_data = h5_main
        self.threads = threads
        self.num_workers = num_workers
        self.client = Client(threads_per_worker=self.threads, n_workers=self.num_workers)

    def extract_sho(self):
        # set for be currently
        sho = SHO_Fitter()
        main_dsets = [] #TODO: add in SHO fit

        self.h5_sho_fit = main_dsets[1]

    def extract_PR(self,plot=False):
        # set for be currently
        self.amplitude  = self.h5_sho_fit['Amplitude [V']
        self.phase = self.h5_sho_fit['Phase [rad]']
        adjust = np.max(self.phase) - np.min(self.phase)
        phase_wrap = []
        for ii in range(self.phase.shape[0]):
            phase_wrap.append([x + adjust if x < -2 else x for x in self.phase[ii, :]])
        self.phase = np.asarray(phase_wrap)

        self.PR_mat = self.amplitude*np.cos(self.phase)
        self.PR_mat = -self.PR_mat.reshape(self.h5_sho_fit.pos_dim_sizes[0],self.h5_sho_fit.pos_dim_sizes[1],-1) #TODO: make sure h5_sho_fit has pos_dim_sizes

        self.dc_vec_OF = self.h5_sho_fit.h5_spec_vals[1,:][self.h5_sho_fit.h5_spec_vals[0,:]==0] #TODO: get this more general!
        self.dc_vec_IF = self.h5_sho_fit.h5_spec_vals[1,:][self.h5_sho_fit.h5_spec_vals[0,:] == 1]#TODO: get this more general!

        self.PR_OF = self.PR_mat[:, :, ::2]#TODO: get this more general!
        self.PR_IF = self.PR_mat[:, :, 1::2]#TODO: get this more general!

        if plot == True:
            fig,ax = plt.subplots(ncols=2,figsize=(10,5))
            ax[0].hist(self.phase.ravel(),bins=100)
            ax[0].title('Phase distribution')
            ax[1].plot(self.dc_vec_OF,self.PR_OF[11,9,:],'r',label='Off field')
            ax[1].plot(self.dc_vec_IF, self.PR_IF[11, 9, :], 'b',label='On Field')
            ax[1].title('Hysteresis Loops from pixel [11,9]')


