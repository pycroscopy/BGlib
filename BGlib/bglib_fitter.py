from scipy.optimize import curve_fit
from copy import deepcopy
from sklearn.cluster import KMeans


def fit_func(xvec, yvec, p0):
    #         if self.dum == 1:
    yvec = np.roll(yvec, -self.max_x)
    popt, pcov = curve_fit(loop_fit_func, xvec, yvec, p0=p0)
    return popt, pcov


class LoopFitter():

    def __init__(self, xvec, sidpy_dataset, pos_dims=None, prior_computation='KMeans', _computation=None,
                 num_workers=1, threads=4, *args, **kwargs):
        from dask import delayed
        self.dataset = sidpy_dataset
        #         self.dataset = self.extract_PR()
        self.prior_computation = prior_computation
        self.num_workers = num_workers
        self.threads = threads
        self.fit_results = []
        self._computation = _computation

        self.client = Client(threads_per_worker=self.threads, n_workers=self.num_workers)

        num_fit_dims = self.dataset.ndim - len(pos_dims)

        self.pos_dim_shapes = tuple([self.dataset.shape[pos_dim] for pos_dim in range(len(pos_dims))])
        self.num_computations = np.prod([self.dataset.shape[pos_dim] for pos_dim in range(len(pos_dims))])

        remaining_dataset_shape = [self.dataset.shape[y] for y in np.arange(self.dataset.ndim)
                                   if y not in pos_dims]

        self.shape_tuple = tuple([self.num_computations]) + tuple(remaining_dataset_shape)

        print("shape tuple is {}".format(self.shape_tuple))

        self.dataset_flat = self.dataset.reshape(self.shape_tuple)

        xvec, max_x = self.format_xvec(xvec)
        self.max_x = max_x
        self.prior_mat_flat = self.calc_priors(xvec)
        self.prior_mat_flat = np.asarray(self.prior_mat_flat).reshape((self.shape_tuple[0], 9))

        pdb.set_trace()

        if self.prior_computation == 'KMeans' or self.prior_computation == 'Random':
            for ind in range(self.num_computations):
                lazy_result = dask.delayed(self._computation)(xvec, self.dataset_flat[ind, :],
                                                              p0=self.prior_mat_flat[ind, :])
                self.fit_results.append(lazy_result)

        if self.prior_computation == 'Neighbor':
            self.xvec = xvec
            self.NN = NN
            self.prior_mat = self.prior_mat_flat.reshape(self.dataset.shape)

    # -------------------- Support Functions --------------------#

    #     def extract_PR(self):
    #         amplitude = self.sho_dataset['Amplitude']
    #         phase = self.sho_dataset['Phase [rad]']
    #         adjust = np.max(phase) - np.min(phase)
    #         phase_wrap = []
    #         for ii in range(phase.shape[0]):
    #             phase_wrap.append([x+adjust if x < -2 else x for x in phase[ii,:]])
    #         phase = np.asarray(phase_wrap)
    #         PR_mat = amplitude*np.cos(phase)
    #         PR_mat = -PR_mat.reshape(h5_sho_fit.pos_dim_sizes[0],h5_sho_fit.pos_dim_sizes[1],-1 )
    #         dc_vec_OF = h5_sho_fit.h5_spec_vals[0,:][np.logical_and(h5_sho_fit.h5_spec_vals[1,:]==0,h5_sho_fit.h5_spec_vals[2,:]==2)]
    #         dc_vec_IF = h5_sho_fit.h5_spec_vals[0,:][np.logical_and(h5_sho_fit.h5_spec_vals[1,:]==1,h5_sho_fit.h5_spec_vals[2,:]==2)]
    #         PR_OF = PR_mat[:,:,129::2] # off field
    #         PR_IF = PR_mat[:,:,128::2] # on field

    #         data = PR_OF
    #         xvec = dc_vec_OF
    #         return PR_mat, xvec

    def format_xvec(self, xvec):
        max_x = np.where(xvec == np.max(xvec))[0]
        if max_x != 0 or max_x != len(xdata0):
            xvec = np.roll(xvec, -max_x)  # assumes voltages are a symmetric triangle wave
            self.dum = 1
        else:
            xvec = xvec  # just in case voltages are already rolled
            self.dum = 0
        return xvec, max_x

    def calc_priors(self, xvec):
        p0_mat = [[]] * self.shape_tuple[0]

        if self.prior_computation == 'KMeans':
            popt_mean = self.calc_mean_fit(xvec)
            self.popt_mean = popt_mean
            n_clusters = int(self.shape_tuple[0] / 100)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.dataset_flat)
            labels = kmeans.labels_
            p0_clusters = []
            cluster_loops = []
            for pp in range(n_clusters):
                opt_vals = []
                res = []
                clust = self.dataset_flat[labels == pp]
                clust = np.asarray(clust)
                PR_mean = np.mean(clust, axis=0)
                if self.dum == 1:
                    PR_mean = np.roll(PR_mean, -self.max_x)
                cluster_loops.append(PR_mean)
                p0 = popt_mean
                try:
                    popt, pcov = curve_fit(self.loop_fit_func2, xvec, PR_mean, p0=p0, maxfev=10000)
                except:
                    kk = 0
                    p0 = np.random.normal(0.1, 5, 9)
                    while kk < 20:
                        try:
                            vals_min, pcov = curve_fit(self.loop_fit_func2, xvec, all_mean, p0=p0, maxfev=10000)
                        except:
                            continue
                        kk += 1
                        opt_vals.append(vals_min)
                        fitted_loop = self.loop_fit_func2(xvec, *vals_min)
                        yres = PR_mean - fitted_loop
                        res.append(yres @ yres)
                        popt = opt_vals[np.argmin(res)]
                p0_clusters.append(popt)
                fitted_loop = self.loop_fit_func(xvec, *popt)

            p0_mat_flat = np.asarray([p0_clusters[k] for k in labels])
            # TODO: make array of associated p0 values
            return p0_mat_flat

        if self.prior_computation == 'Neighbor':
            # TODO: cannot use dask parallel computation
            popt_mean = self.calc_mean_fit(xvec)
            p0_mat_mean = [popt_mean] * self.shape_tuple[0]

        if self.prior_computation == 'Random':
            p0_mat_flat = [np.random.normal(0.1, 5, 9) for x in range(self.shape_tuple[0])]
            return p0_mat_flat

    def calc_mean_fit(self, xvec):
        opt_vals = []
        res = []
        all_mean = self.dataset.mean(axis=1).mean(axis=0)
        all_mean = np.asarray(all_mean)
        if self.dum == 1:
            all_mean = np.roll(all_mean, -self.max_x)
        for kk in range(20):

            # TODO: convert this fitting to dask

            p0 = np.random.normal(0.1, 5, 9)
            try:
                vals_min, pcov = curve_fit(self.loop_fit_func2, xvec, all_mean, p0=p0, maxfev=10000)
            except:
                continue
            opt_vals.append(vals_min)
            fitted_loop = self.loop_fit_func2(xvec, *vals_min)
            yres = all_mean - fitted_loop
            res.append(yres @ yres)

        popt = opt_vals[np.argmin(res)]
        popt_mean = deepcopy(popt)
        return popt_mean

    def fit_parallel(self):
        if self.prior_computation == 'Neighbor':
            return print('Please use "fit_series" to fit loops rather than fit_parallel')

        self.results = dask.compute(*self.fit_results)
        self.results_arr = np.array(self.results)
        pdb.set_trace()
        # convert to sidpy and return it

        self.results_reshaped_shape = self.pos_dim_shapes + tuple([-1])
        self.results_reshaped = np.array(self.results_arr).reshape(self.results_reshaped_shape)

        return self.results, self.results_reshaped

    def fit_series(self, xvec):
        count = 0
        ref_counts = np.arange(self.shape_tuple).reshape(self.dataset.shape)
        for ii in range(self.pos_dim_shapes[0]):
            xind = ii
            for jj in range(self.pos_dim_shapes[1]):
                count += 1
                yind = jj
                ydata0 = self.dataset[xind, yind, :]
                xs = [ii + k for k in range(-self.NN, self.NN + 1)]
                ys = [jj + k for k in range(-self.NN, self.NN + 1)]
                nbrs = [(n, m) for n in xs for m in ys]
                cond = [all(x >= 0 for x in list(y)) for y in nbrs]
                nbrs = [d for (d, remove) in zip(nbrs, cond) if remove]
                cond2 = [all(x < self.pos_dim_shapes[0] for x in list(y)) for y in nbrs]
                nbrs = [d for (d, remove) in zip(nbrs, cond2) if remove]
                NN_indx = [ref_counts[v] for v in nbrs]
                prior_coefs = [self.prior_mat[k] for k in NN_indx if len(self.prior_mat[k]) != 0]
                p0 = np.mean(prior_coefs, axis=0)
                prior_mat_flat_ref = deepcopy(self.prior_mat_flat)
                prior_mat_flat_ref[count] = p0

                popt, pcov = self.fit_func(xvec, ydata0, p0)

                self.prior_mat_flat[count] = popt
                self.results[count] = deepcopy(popt)

        self.results_shaped = self.results.reshape(self.dataset.shape)
        return self.results, self.results_shaped

    def loop_fit_func2(self, vdc, *coef_vec):
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



