import numpy as np
import os
import matplotlib.pyplot as plt
from bglib_process import BGlibProcess
from bglib_guesser import *

# class BGlibFitter(BGlibProcess):
#     """
#     Fitter class for BGlib, contains fitter for SHO fit of raw data and hysteresis loop fitter.
#     """
#     def __init__(self):
#         super(BGlibFitter, self).__init__(h5_main, "Fitter",
#                                            method='K-Means', **kwargs)  #TODO: still don't quite understand the super()


class SHO_Fitter(BGlibProcess):

    def __init__(self):
        super(SHO_Fitter, self).__init__()

        # TODO: add fitter


class LoopFitter(BGlibProcess):

    def __init__(self):
        super(LoopFitter, self).__init__()

        # TODO: add fitter

    def do_fit(self):

        if self.method == 'K-Means':
            guess =


    def loop_fit_function(self, vdc, *coef_vec): #TODO: change from regular function to contain class variables
        """
        9 parameter fit function

        Parameters
        -----------
        vdc : 1D numpy array or list
            DC voltages
        coef_vec : 1D numpy array or list
            9 parameter coefficient vector

        Returns
        ---------
        loop_eval : 1D numpy array
            Loop values
        """
        from scipy.special import erf
        a = coef_vec[:5]
        b = coef_vec[5:]
        d = 1000

        v1 = np.asarray(vdc[:int(len(vdc) / 2)])
        v2 = np.asarray(vdc[int(len(vdc) / 2):])

        g1 = (b[1] - b[0]) / 2 * (erf((v1 - a[2]) * d) + 1) + b[0]
        g2 = (b[3] - b[2]) / 2 * (erf((v2 - a[3]) * d) + 1) + b[2]

        y1 = (g1 * erf((v1 - a[2]) / g1) + b[0]) / (b[0] + b[1])
        y2 = (g2 * erf((v2 - a[3]) / g2) + b[2]) / (b[2] + b[3])

        f1 = a[0] + a[1] * y1 + a[4] * v1
        f2 = a[0] + a[1] * y2 + a[4] * v2

        loop_eval = np.hstack((f1, f2))
        return loop_eval

    def loop_resid(coef_vec, vdc, ydata):  #TODO: change from regular function to contain class variables
        y = loop_fit_function(vdc, *coef_vec)
        res = ydata - y
        ss = res @ res
        return ss

