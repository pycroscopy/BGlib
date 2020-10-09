from pyUSID import USIDataset

class SHOBEDataset(USIDataset):
    """

    Extension of the USIDataset object for Raw Band-excitation (BE) datasets
    This includes various visualization and analaysis routines

    Pass the raw_data h5 dataset to make this into a RawBEDataset

    """

    def __init__(self, usid_dataset):

        self.dset = usid_dataset
        super(SHOBEDataset, self).__init__(h5_ref=usid_dataset)

        # Prepare the datasets
        self.dataset_type = 'SHOBEDataset'
        self.parm_dict = self.dset.file['/Measurement_000'].attrs

        return


    def plot_spectrogram(self, static = False):
        print('here we can plot out stuff')
        # TODO: Add the BE visualizers here

