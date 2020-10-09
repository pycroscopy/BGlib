from pyUSID import USIDataset
from ..analysis import TRKPFM_L_Analyzer

class RawTRKPFM_L_Dataset(USIDataset):
    """
    Extention of teh USIDataset object for Linear Time-resolved Kelvin Probe Force Microscopy (tr-KPFM) datasets
    This includes various visualization and analysis routines

    Pass the raw_data h5 dataset to make this into a RawTRKPFM_L_Dataset

    Should be able to pass either HDF5dataset or USIDataset
    """

    def __init__(self,h5_dataset):
        super(RawTRKPFM_L_Dataset,self).__init__(h5_ref = h5_dataset)

        #Prepare the datasets
        self.dataset_type = 'RawTRKPFM_L_Dataset'
        self.parm_dict = self.dset.file['/Measurement_000'].attrs

        self.analyzer = TRKPFM_L_Analyzer(self)





