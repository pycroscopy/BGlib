from pyUSID import USIDataset
import pyUSID as usid
import numpy as np
from ..viz.be_viz_utils import jupyter_visualize_beps_loops

'''
Here we make dataset objects that are processed further after SHO Fitting
These will all be extensions of BEProcessedDataset
These are
    -BEPSLoopsDataset (after processing of loops, i.e. loop guess and fit)
    -cKPFMLoopsDataset (after processing on cKPFM curves, e.g. j-CPD)
    -RelaxationFitDataset (after processing of relaxation curves, e.g. tau, A0)
'''

class BEProcessedDataset(USIDataset):
    """
      Extension of the USIDataset object for SHO-Fitted and Processed Band-excitation (BE) datasets
      Pass the sho-fitted and subsequently processed USIDataset to make this into a BEProcessedDataset
      """

    def __init__(self, h5_dataset):
        super(BEProcessedDataset, self).__init__(h5_ref=h5_dataset)
        # Populate some data tags
        self.dataset_type = 'BEProcessedDataset'
        self.parm_dict = list(self.file['/Measurement_000'].attrs)


class BEPSLoopsDataset(BEProcessedDataset):
    """
       Extension of the BEProcessedDataset object for SHO-Fitted and Processed Band-excitation (BE) datasets
       Pass the sho-fitted and subsequently loop fitted dataset to make this into a BEPSLoopsDataset
    """
    def __init__(self, h5_dataset):
        super(BEPSLoopsDataset, self).__init__(h5_dataset)
        # Populate some data tags
        self.dataset_type = 'BEPSLoopsDataset'
        h5_projected_loops = usid.USIDataset(self.parent['Projected_Loops'])
        h5_proj_spec_inds = h5_projected_loops.h5_spec_inds
        h5_proj_spec_vals = h5_projected_loops.h5_spec_vals

        # reshape the vdc_vec into DC_step by Loop
        sort_order = usid.hdf_utils.get_sort_order(h5_proj_spec_inds)
        dims = usid.hdf_utils.get_dimensionality(h5_proj_spec_inds[()],
                                                 sort_order[::-1])
        self.vdc_vec = np.reshape(h5_proj_spec_vals[list(h5_proj_spec_vals.attrs['labels']).index(b'DC_Offset')], dims).T

        # Also reshape the projected loops to Positions-DC_Step-Loop
        # Also reshape the projected loops to Positions-DC_Step-Loop
        self.proj_nd = h5_projected_loops.get_n_dim_form()
        self.proj_3d = np.reshape(self.proj_nd, [h5_projected_loops.shape[0],
                                       self.proj_nd.shape[2], -1])
        self.h5_projected_loops = h5_projected_loops
        self.h5_loop_guess = self.parent['Guess']
        self.h5_loop_fit = self.parent['Fit']

    def visualize_results(self):
        return jupyter_visualize_beps_loops(self.h5_projected_loops, self.h5_loop_guess, self.h5_loop_fit)