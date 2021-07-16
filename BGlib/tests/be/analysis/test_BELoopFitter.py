from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest
import sys

from BGlib.be.analysis.BELoopFitter import *

class TestFitFunc(unittest.TestCase):

    def test_fit_fun_inputs(self):
        arr1 = np.asarray([1,3,5])
        arr2 = np.asarray([7,11,13])
        p0 = np.random.normal(0.1, 5, 9)
        self.assertTrue(np.allclose(arr1,arr2,p0,fit_func(arr1,arr2,p0)))

    def test_invalid_values1(self):
        with self.assertRaises(TypeError):
            _ = fit_func('array','array2','values') #test strings

    def test_invalid_values2(self):
        with self.assertRaises(TypeError):
            _ = fit_func(0,1,3) # test integers
    def test_invalid_values3(self):
        with self.assertRaises(TypeError):
            _ = fit_func([0,1,3], [5,7,11], [13,17,23]) #test lists



