import unittest
import sys
import os
sys.path.append("../BGlib/")
import BGlib.be as belib
import pyUSID as usid
import h5py

class TestLoadDataset(unittest.TestCase):

    def test_bepfm_dataset(self):
        #this file is actually a corrupted BEPFM file for which the translator has an auto fix
        #So we need to download the original file, see if the following works, then delete
        #TODO: Host hte original file somewhere, use wget
        
        input_file_path = r'/Users/rvv/Downloads/PTO_a32/BEPFM_1um_afterBEPS_0039.h5'
        (data_dir, filename) = os.path.split(input_file_path)
        # No translation here
        h5_path = input_file_path
        force = False # Set this to true to force patching of the datafile.
        tl = belib.translators.LabViewH5Patcher()
        tl.translate(h5_path, force_patch=force)
        h5_file = h5py.File(h5_path, 'r+')
        h5_main = usid.hdf_utils.find_dataset(h5_file, 'Raw_Data')[0]
        #Let's see if we get the right type of file back...
        assert type(h5_main) == usid.io.usi_data.USIDataset

    def test_values_as_length(self):
        print("OK")
        