from __future__ import division, print_function, unicode_literals, \
    absolute_import
import unittest

from BGlib.be.analysis.BELoopFitter import *

test_data = np.load('test_data_2x2.npy')
test_data_coeff = np.load('test_data_2x2_coeff.npy')
test_data_xvec = np.load('xvec.npy')

class TestFitFunc(unittest.TestCase):
    """
    Tests for fit_func in BELoopFitter
    test for:
    fit_func
    format_xvec
    calc_priors
    calc_mean_fit
    fit_parallel -> TODO
    fit_series -> TODO
    convert_coeff2loop -> TODO
    loop_fit_func2
    """
    def test_fit_func_inputs(self):
        arr1 = np.asarray([1,3,5])
        arr2 = np.asarray([7,11,13])
        p0 = np.random.normal(0.1, 5, 9)
        self.assertTrue(np.allclose(arr1,arr2,p0,fit_func(arr1,arr2,p0)))

    def test_invalid_values1(self):
        """
        input arrays cannot be strings
        """
        with self.assertRaises(TypeError):
            _ = fit_func('array','array2','values') #test strings

    def test_invalid_values2(self):
        """
        Input arrays should have len() > 1
        """
        with self.assertRaises(TypeError):
            _ = fit_func(0,1,3) # test integers
    def test_invalid_values3(self):
        """
        Input arrays cannot be lists, should be np.array
        """
        with self.assertRaises(TypeError):
            _ = fit_func([0,1,3], [5,7,11], [13,17,23]) #test lists

    def test_fit_func(self):
        """
        Check the loop values come back as the pre-defined values
        """
        expected = test_data[0,0]
        p0 = test_data_coeff[0,0]
        loop = fit_func(xvec,*p0)
        self.assertEqual(expected,loop)


class TestFormatXvec(unittest.TestCase):
    def test_format_xvec_input(self):
        """
        Array should be longer than length 1
        """
        arr1 = np.asarray([1])
        with self.assertRaises(TypeError):
            _ = LoopFitter.format_xvec(arr1)

    def test_format_xvec_input2(self):
        """
        Array should be a numeric array
        """
        arr1 = np.asarray(['test','test2'])
        with self.assertRaises(TypeError):
            _ = LoopFitter.format_xvec(arr1)

    def test_max(self):
        """
        Testing array max is correct
        """
        arr1 = np.asarray([1,2,3,4,5,4,3,2,1])
        max_x = np.where(xvec == np.max(arr1))[0]
        expected = 4
        self.assertEqual(expected,max_x)
#TODO: need to test for if max is an integer?  I would think it should be fine

class TestCalcPriors(unittest.TestCase):
    def test_calc_priors_input(self):
        """
        Input should be a np.array not a list
        """
        arr1 = [1,2,3,4]
        with self.assertRaises(TypeError):
            _ = calc_priors(arr1)

    def test_calc_priors_input2(self):
        """
        Input should not contain strings
        """
        arr1 = ['test',2,3,4]
        with self.assertRaises(TypeError):
            _ = calc_priors(arr1)

    def test_calc_priors_input3(self):
        """
        Input should be longer than 1 element
        """
        arr1 = np.asarray([1])
        with self.assertRaises(TypeError):
            _ = calc_priors(arr1)


class TestCalcMeanFit(unittest.TestCase):
    def test_input(self):
        """
        Input should be a np.array not a list
        """
        arr1 = [1, 2, 3, 4]
        with self.assertRaises(TypeError):
            _ = calc_mean_fit(arr1)

    def test_input2(self):
        """
        Input should not contain strings
        """
        arr1 = ['test', 2, 3, 4]
        with self.assertRaises(TypeError):
            _ = calc_mean_fit(arr1)

    def test_input3(self):
        """
        Input should be longer than 1 element
        """
        arr1 = np.asarray([1])
        with self.assertRaises(TypeError):
            _ = calc_mean_fit(arr1)
#TODO: need to make tests for basic functions like append, curve_fit, @, deepcopy, argmin?

class TestFitParallel(unittest.TestCase):
    def test_fit_parallel(self):
        # TODO: mostly uses functions/data already set up in the class
        # TODO: how to make a test to check dask is running? This function just sets up dask and calls other functions
        return

class TestFitSeries(unittest.TestCase): #TODO
    def test_fit_series_method(self):
        method_list = ['Kmeans','Random','Neighbor']
        prior_computation = ['Random','Neighbor','KMeans','text']
        for k in prior_computation:
            if k not in method_list:
                with self.assertRaises(TypeError):
                    _ = fit_series(self)
        return

class TestConvertCoeff2loop(unittest.TestCase): #TODO: how to run tests on dask.compute?
    def test_convert_coeff2loop(self):
        return

class TestLoopFitFunc(unittest.TestCase):
    def test_fit_func_inputs(self):
        arr1 = np.asarray([1, 3, 5])
        arr2 = np.asarray([7, 11, 13])
        p0 = np.random.normal(0.1, 5, 9)
        self.assertTrue(np.allclose(arr1, arr2, p0, loop_fit_func2(arr1, arr2, p0)))

    def test_invalid_values1(self):
        """
        input arrays cannot be strings
        """
        with self.assertRaises(TypeError):
            _ = loop_fit_func2('array', 'array2', 'values')  # test strings

    def test_invalid_values2(self):
        """
        Input arrays should have len() > 1
        """
        with self.assertRaises(TypeError):
            _ = loop_fit_func2(0, 1, 3)  # test integers

    def test_invalid_values3(self):
        """
        Input arrays cannot be lists, should be np.array
        """
        with self.assertRaises(TypeError):
            _ = loop_fit_func2([0, 1, 3], [5, 7, 11], [13, 17, 23])  # test lists



if __name__ == '__main__':
    unittest.main()
