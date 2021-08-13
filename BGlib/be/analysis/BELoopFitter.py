from scipy.optimize import curve_fit
from copy import deepcopy
from sklearn.cluster import KMeans
import numpy as np
from dask.distributed import Client, progress
import dask
import matplotlib.pyplot as plt
import sidpy
from enum import Enum

class GuessMethod(Enum):
    KMeans = 0
    Neighbor = 1
    Random = 2
    # Hierarchical = 3

def fit_func(xvec, yvec, p0, **kwargs):
    from LoopFitter import loop_fit_func #not sure if this will fix the issue....
    dum = kwargs['dum']
    if dum == 1:
        yvec = np.roll(yvec, -kwargs['max_x'])
    try:
        popt, pcov = curve_fit(loop_fit_func, xvec, yvec, p0=p0)
    except:
        popt = np.repeat(np.nan, 9)
        pcov = np.repeat(np.nan, 9)
    return popt, pcov


class LoopFitter():

    def __init__(self, xvec, sidpy_dataset, pos_dims=None, prior_computation=GuessMethod.KMeans, _computation=None,
                 num_workers=1, threads=4, *args, **kwargs):
        from dask import delayed
        # self.dataset = sidpy_dataset
        # self.prior_computation = prior_computation
        # num_workers = num_workers
        # threads = threads
        self.fit_results = []
        self._computation = _computation
        self.results = []

        self.client = Client(threads_per_worker=threads, n_workers=num_workers)

        # num_fit_dims = self.dataset.ndim - len(pos_dims)

        self.pos_dim_shapes = tuple([sidpy_dataset.shape[pos_dim] for pos_dim in range(len(pos_dims))])
        num_computations = np.prod([sidpy_dataset.shape[pos_dim] for pos_dim in range(len(pos_dims))])

        remaining_dataset_shape = [sidpy_dataset.shape[y] for y in np.arange(sidpy_dataset.ndim)
                                   if y not in pos_dims]

        shape_tuple = tuple([num_computations]) + tuple(remaining_dataset_shape)

        print("shape tuple is {}".format(shape_tuple))

        self.dataset_flat = sidpy_dataset.reshape(shape_tuple)
        dataset_flat = sidpy_dataset.reshape(shape_tuple)

        xvec, max_x, dum = LoopFitter.format_xvec(xvec)
        # self.max_x = max_x
        # self.xvec = xvec
        p0_mat_flat, p0_clusters, labels = LoopFitter.calc_priors(xvec, dataset_flat, prior_computation, shape_tuple, max_x, dum)

        # self.prior_mat_flat = np.asarray(p0_mat_flat).reshape((shape_tuple[0], 9))
        prior_mat_flat = np.asarray(p0_mat_flat).reshape((shape_tuple[0], 9))


        if prior_computation == GuessMethod.KMeans or prior_computation == GuessMethod.Random:
            for ind in range(num_computations):
                lazy_result = dask.delayed(self._computation)(xvec, dataset_flat[ind, :],
                                                              p0=prior_mat_flat[ind, :], dum=dum,
                                                              max_x=max_x)
                #                 lazy_result = dask.delayed(self.loop_fit_func2)(xvec, self.dataset_flat[ind,:],p0=self.prior_mat_flat[ind,:])
                self.fit_results.append(lazy_result)

        if prior_computation == GuessMethod.Neighbor:
            # self.xvec = xvec
            self.NN = kwargs['NN']
            self.prior_mat = prior_mat_flat.reshape((sidpy_dataset.shape[0], sidpy_dataset.shape[1], 9))

    # -------------------- Support Functions --------------------#

    @staticmethod # doesn't change state of class, not assigning a self within function
    def format_xvec(xvec):
        max_x = np.where(xvec == np.max(xvec))[0]
        if max_x != 0 or max_x != len(xvec):
            xvec = np.roll(xvec, -max_x)  # assumes voltages are a symmetric triangle wave
            dum = 1
        else:
            xvec = xvec  # just in case voltages are already rolled
            dum = 0
        return xvec, max_x, dum

    @staticmethod
    def calc_priors(xvec, dataset_flat, prior_computation, shape_tuple, max_x, dum):
        # p0_mat = [[]] * self.shape_tuple[0]

        if prior_computation == GuessMethod.KMeans:
            popt_mean, all_mean = LoopFitter.calc_mean_fit(xvec, dataset_flat, max_x, dum)
            n_clusters = int(shape_tuple[0] / 100)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataset_flat)
            labels = kmeans.labels_
            p0_clusters = []
            cluster_loops = []
            for pp in range(n_clusters):
                opt_vals = []
                res = []
                clust = dataset_flat[labels == pp]
                clust = np.asarray(clust)
                PR_mean = np.mean(clust, axis=0)
                if dum == 1:
                    PR_mean = np.roll(PR_mean, -max_x)
                cluster_loops.append(PR_mean)
                p0 = popt_mean
                try:
                    popt, pcov = curve_fit(LoopFitter.loop_fit_func2, xvec, PR_mean, p0=p0, maxfev=10000)
                except:
                    kk = 0
                    p0 = np.random.normal(0.1, 5, 9)
                    while kk < 20:
                        try:
                            vals_min, pcov = curve_fit(LoopFitter.loop_fit_func2, xvec, all_mean, p0=p0, maxfev=10000)
                        except:
                            continue
                        kk += 1
                        opt_vals.append(vals_min)
                        fitted_loop = LoopFitter.loop_fit_func2(xvec, *vals_min)
                        yres = PR_mean - fitted_loop
                        res.append(yres @ yres)
                        popt = opt_vals[np.argmin(res)]
                p0_clusters.append(deepcopy(popt))
                # fitted_loop = LoopFitter.loop_fit_func2(xvec, *popt)

            p0_mat_flat = np.asarray([p0_clusters[k] for k in labels])
            return p0_mat_flat, p0_clusters, labels

        if prior_computation == GuessMethod.Neighbor:
            # TODO: cannot use dask parallel computation
            popt_mean, all_mean = LoopFitter.calc_mean_fit(xvec)
            p0_mat_flat = [popt_mean] * shape_tuple[0]
            return p0_mat_flat

        if prior_computation == GuessMethod.Random:
            p0_mat_flat = [np.random.normal(0.1, 5, 9) for x in range(shape_tuple[0])]
            return p0_mat_flat

    @staticmethod
    def calc_mean_fit(xvec, dataset, max_x, dum):
        opt_vals = []
        res = []
        all_mean = dataset.mean(axis=1).mean(axis=0)
        all_mean = np.asarray(all_mean)
        # self.all_mean = deepcopy(all_mean)
        if dum == 1:
            all_mean = np.roll(all_mean, -max_x)
        for kk in range(20):
            # TODO: convert this fitting to dask
            p0 = np.random.normal(0.1, 5, 9)
            try:
                vals_min, pcov = curve_fit(LoopFitter.loop_fit_func2, xvec, all_mean, p0=p0, maxfev=10000)
            except:
                continue
            opt_vals.append(vals_min)
            fitted_loop = LoopFitter.loop_fit_func2(xvec, *vals_min)
            yres = all_mean - fitted_loop
            res.append(yres @ yres)

        popt = opt_vals[np.argmin(res)]
        popt_mean = deepcopy(popt)

        loop_avg = LoopFitter.loop_fit_func2(xvec, *popt_mean)
        plt.figure()
        plt.plot(xvec, all_mean, 'k')
        plt.plot(xvec, loop_avg, 'r')
        return popt_mean, all_mean

    def fit_parallel(self):  # NEED HELP -- Issues with Dask
        if self.prior_computation == GuessMethod.Neighbor:
            return print('Please use "fit_series" to fit loops rather than fit_parallel')

        self.results = dask.compute(*self.fit_results)
        popt_vals = [v[0] for v in self.results]
        self.pcov_vals = [v[1] for v in self.results]
        self.results_arr = np.asarray(popt_vals)

        self.results_reshaped_shape = self.pos_dim_shapes + tuple([-1])
        self.results_reshaped = np.array(self.results_arr).reshape(self.results_reshaped_shape)
        #         self.pcov_reshaped = np.array(np.asarray(pcov_vals)).reshape(self.results_reshaped_shape)

        self.fit_dataset = sidpy.Dataset.from_array(self.results_reshaped, name='Fitted Coefficients')
        self.fit_dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE
        x_pos = np.arange(0, self.dataset.shape[0])
        y_pos = np.arange(0, self.dataset.shape[1])
        self.fit_dataset.set_dimension(0, sidpy.Dimension(x_pos,
                                                          name='x',
                                                          units='um',
                                                          quantity='x',
                                                          dimension_type='spatial'))
        self.fit_dataset.set_dimension(1, sidpy.Dimension(y_pos,
                                                          name='y',
                                                          units='um',
                                                          quantity='y',
                                                          dimension_type='spatial'))

        return self.fit_dataset, self.pcov_vals

    def fit_series(self):
        xvec = deepcopy(self.xvec)
        count = -1
        SSqRes = []
        fitted_data = []
        ref_counts = np.arange(self.shape_tuple[0]).reshape(self.dataset.shape[:2])
        for ii in range(self.pos_dim_shapes[0]):
            xind = ii
            for jj in range(self.pos_dim_shapes[1]):
                count += 1
                yind = jj
                ydata0 = np.asarray(self.dataset[xind, yind, :])

                if self.prior_computation == GuessMethod.Neighbor:
                    xs = [ii + k for k in range(-self.NN, self.NN + 1)]
                    ys = [jj + k for k in range(-self.NN, self.NN + 1)]
                    nbrs = [(n, m) for n in xs for m in ys]
                    cond = [all(x >= 0 for x in list(y)) for y in nbrs]
                    nbrs = [d for (d, remove) in zip(nbrs, cond) if remove]
                    cond2 = [all(x < self.pos_dim_shapes[0] for x in list(y)) for y in nbrs]
                    nbrs = [d for (d, remove) in zip(nbrs, cond2) if remove]
                    NN_indx = [ref_counts[v] for v in nbrs]
                    #                     prior_coefs = [self.prior_mat_flat[k] for k in NN_indx if len(self.prior_mat[k]) != 0]
                    prior_coefs = [self.prior_mat_flat[k] for k in NN_indx]

                    p0 = np.mean(prior_coefs, axis=0)
                    prior_mat_flat_ref = deepcopy(self.prior_mat_flat)
                    prior_mat_flat_ref[count] = p0
                    try:
                        popt, pcov = self.fit_func(xvec, ydata0, p0)
                    except:
                        try:
                            popt, pcov = self.fit_func(xvec, ydata0, self.popt_mean)
                        except:
                            popt = np.repeat(np.nan, 9)
                    self.prior_mat_flat[count] = popt

                if self.prior_computation == GuessMethod.KMeans or GuessMethod.Random:
                    p0 = self.prior_mat_flat[count]
                    ydata0 = np.asarray(self.dataset_flat[count])
                    try:
                        popt, pcov = self.fit_func(xvec, ydata0, p0)
                    except:
                        try:
                            popt, pcov = self.fit_func(xvec, ydata0, self.popt_mean)
                        except:
                            popt = np.repeat(np.nan, 9)

                if self.prior_computation == GuessMethod.Random:
                    p0 = self.prior_mat_flat[count]
                    ydata0 = np.asarray(self.dataset_flat[count])
                    try:
                        popt, pcov = self.fit_func(xvec, ydata0, p0)
                    except:
                        try:
                            p0 = np.random.normal(0.1, 5, 9)
                            popt, pcov = self.fit_func(xvec, ydata0, p0)
                        except:
                            popt = np.repeat(np.nan, 9)

                self.results.append(deepcopy(popt))
                fitted_loop = LoopFitter.loop_fit_func2(xvec, *popt)
                fitted_data.append(fitted_loop)
                SSqRes.append(np.sum((ydata0 - fitted_loop) ** 2))
        self.results_shaped = np.asarray(self.results).reshape((self.dataset.shape[0], self.dataset.shape[1], 9))
        self.SSqRes = SSqRes

        self.fit_dataset = sidpy.Dataset.from_array(self.results_shaped, name='Fitted Coefficients')
        self.fit_dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE
        x_pos = np.arange(0, self.dataset.shape[0])
        y_pos = np.arange(0, self.dataset.shape[1])
        self.fit_dataset.set_dimension(0, sidpy.Dimension(x_pos,
                                                          name='x',
                                                          units='um',
                                                          quantity='x',
                                                          dimension_type='spatial'))
        self.fit_dataset.set_dimension(1, sidpy.Dimension(y_pos,
                                                          name='y',
                                                          units='um',
                                                          quantity='y',
                                                          dimension_type='spatial'))

        self.fitted_loops = sidpy.Dataset.from_array(np.asarray(fitted_data).reshape(self.dataset.shape),
                                                     name='Fitted Loops')
        self.fitted_loops.data_type = sidpy.DataType.SPECTRAL_IMAGE
        self.fitted_loops.set_dimension(0, sidpy.Dimension(x_pos,
                                                           name='x',
                                                           units='um',
                                                           quantity='x',
                                                           dimension_type='spatial'))
        self.fitted_loops.set_dimension(1, sidpy.Dimension(y_pos,
                                                           name='y',
                                                           units='um',
                                                           quantity='y',
                                                           dimension_type='spatial'))
        self.fitted_loops.set_dimension(2, sidpy.Dimension(xvec,
                                                           name='Voltage',
                                                           units='V',
                                                           quantity='Voltage',
                                                           dimension_type='spectral'))
        return self.fit_dataset, self.fitted_loops  # self.results, self.results_shaped

    @staticmethod
    def convert_coeff2loop(xvec, fit_dataset, num_computations, dataset): #TODO: check if I need dataset or if I can use fit_dataset instead
        # xvec = deepcopy(xvec)
        fit_loops = []
        fit_coeff = fit_dataset.reshape((num_computations, 9))
        for ind in range(num_computations):
            # fit_loops.append(LoopFitter.loop_fit_func2(xvec, *fit_coeff[ind, :]))
                    lazy_result = dask.delayed(LoopFitter.loop_fit_func2)(xvec, *fit_coeff[ind,:])
                    fit_loops.append(lazy_result)
        fitted_loops = dask.compute(*fit_loops)
        # self.fitted_loops = fit_loops
        fitted_loops = sidpy.Dataset.from_array(np.asarray(fitted_loops).reshape(dataset.shape),
                                                     name='Fitted Loops')
        fitted_loops.data_type = sidpy.DataType.SPECTRAL_IMAGE
        x_pos = np.arange(0, dataset.shape[0])
        y_pos = np.arange(0, dataset.shape[1])
        fitted_loops.set_dimension(0, sidpy.Dimension(x_pos,
                                                           name='x',
                                                           units='um',
                                                           quantity='x',
                                                           dimension_type='spatial'))
        fitted_loops.set_dimension(1, sidpy.Dimension(y_pos,
                                                           name='y',
                                                           units='um',
                                                           quantity='y',
                                                           dimension_type='spatial'))
        fitted_loops.set_dimension(2, sidpy.Dimension(xvec,
                                                           name='Voltage',
                                                           units='V',
                                                           quantity='Voltage',
                                                           dimension_type='spectral'))
        return fitted_loops

    @staticmethod
    def loop_fit_func2(vdc, *coef_vec):
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
