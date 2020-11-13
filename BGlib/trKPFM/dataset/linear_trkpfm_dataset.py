from pyUSID import USIDataset
from ..analysis import trKPFM_L_Analyzer

class RawTRKPFM_L_Dataset():
    """
    Extention of teh USIDataset object for Linear Time-resolved Kelvin Probe Force Microscopy (tr-KPFM) datasets
    This includes various visualization and analysis routines

    Pass the raw_data h5 dataset to make this into a RawTRKPFM_L_Dataset

    Should be able to pass either HDF5dataset or USIDataset
    """

    def __init__(self,h5_dataset):
        # super(RawTRKPFM_L_Dataset,self).__init__(h5_ref = h5_dataset)
        self.h5_main = h5_dataset
        #Prepare the datasets
        self.dataset_type = 'RawTRKPFM_L_Dataset'
        # self.parm_dict = self.dset.file['/Measurement_000'].attrs
        # self.analyzer = trKPFM_L_Analyzer(self)

    def plot_raw_data(self):
        """
          Not sure how to get the scan rate and scan size out of the igor file...

          scan_rate: in Hz
          scan_size: in um
        """
        import matplotlib.pyplot as plt

        try:
            test = self.volt[0,0]
        except:
            self.unpack_data()

        fig,ax = plt.subplots(ncols=3,nrows=2,figsize=(12,6))
        ax[0,0].imshow(self.amp,cmap='inferno')
        ax[0,0].set_title('Amplitude')
        ax[0,1].imshow(self.phase,cmap='inferno')
        ax[0,1].set_title('Phase')
        ax[0,2].imshow(self.height,cmap='inferno')
        ax[0,2].set_title('Height')
        ax[1,0].imshow(self.pot,cmap='plasma')
        ax[1,0].set_title('CPD')
        ax[1,1].imshow(self.volt,cmap='plasma')
        ax[1,1].set_title('Applied Voltage')
        ax[1,2].plot(self.volt[:,int(self.ndim_form[1]/2)])
        ax[1,2].set_ylabel('Voltage Profile')

    def unpack_data(self):

        """
          Grabs raw data using the descriptor names ['Amplitude', 'Potential', 'UserIn', 'HeightRetrace', 'Phase'].
          UserIn: Voltage (may need to change as this is jus what I used to collect data)
        """
        import numpy as np

        desr = [ii.data_descriptor for ii in self.h5_main]
        subs = ['Amplitude', 'Potential', 'UserIn', 'HeightRetrace', 'Phase']
        indx0 = [desr.index(list(filter(lambda x: ii in x, desr))[0]) for ii in subs]
        amp_data = self.h5_main[indx0[0]]
        pot_data = self.h5_main[indx0[1]]
        volt_data = self.h5_main[indx0[2]]
        height_data = self.h5_main[indx0[3]]
        phase_data = self.h5_main[indx0[4]]
        self.ndim_form = volt_data.get_n_dim_form().shape
        self.volt = (np.flipud(np.reshape(volt_data, self.ndim_form[:2])))
        self.pot = (np.flipud(np.reshape(pot_data, self.ndim_form[:2])))
        self.height = (np.flipud(np.reshape(height_data, self.ndim_form[:2])))
        self.phase = (np.flipud(np.reshape(phase_data, self.ndim_form[:2])))
        self.amp = (np.flipud(np.reshape(amp_data,self.ndim_form[:2])))

    def time_line_scan(self,d=0):
        """
        Plot a single CPD line scan along a specified distance, d

        """
        import matplotlib.pyplot as plt

        try:
            test = self.volt[0,0]
        except:
            self.unpack_data()

        if d > self.ndim_form[1]:
            print('Please choose a d value less than: '+str(self.ndim_form[1]))

        fig,ax = plt.subplots(figsize=(6,6))
        ax.plot(self.pot[d,:])

    def distance_line_scan(self, t=0):
        """
        Plot a single CPD line scan along a specified time, t

        """
        import matplotlib.pyplot as plt

        try:
            test = self.volt[0, 0]
        except:
            self.unpack_data()

        if t > self.ndim_form[0]:
            print('Please choose a d value less than: '+str(self.ndim_form[0]))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.pot[:,t])

