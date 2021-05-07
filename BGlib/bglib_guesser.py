import numpy as np
import os
import matplotlib.pyplot as plt
from bglib_process import BGlibProcess
from sklearn.cluster import KMeans
import pyUSID as usid
import sidpy as sid
from scipy.optimize import curve_fit
from bglib_fitter import loop_fit_function
from copy import deepcopy

class BGlibGuesser(BGlibProcess): #TODO: Add validity checks

    def __init__(self):
        super(BGlibGuesser, self).__init__()

        self.p0_mat = [[]]**self.PR_mat.shape[0]*self.PR_mat.shape[1]
        self.all_mean = np.mean(np.mean(self.PR_mat,axis=0),axis=0)

        xdata0 = self.dc_vec_OF
        self.max_x = np.where(xdata0 == np.max(xdata0))[0]
        if max_x != 0 or max_x != len(xdata0):
            self.xdata = np.roll(xdata0, -self.max_x)  # assumes voltages are a symmetric triangle wave
            self.rolled = 1
        else:
            self.xdata = xdata0  # just in case voltages are already rolled
            self.rolled = 0

        self.p0_vals = []
        self.opt_vals = []
        self.res = []
        if self.rolled == 1:
            self.all_mean = np.roll(self.all_mean, -self.max_x)

        self.ref_counts = np.arange(self.PR_mat.shape[0] * self.PR_mat.shape[1]).reshape(
            (self.PR_mat.shape[0], self.PR_mat.shape[1]))  # reference for finding neighboring pixels

    def fit_average(self,N=20,plot=False):
        """
        Function which fits the average hysteresis loop.

        :param N: number of fitting iterations to try. Used to ensure the best fit to the loop is found.
        """
        for kk in range(N):
            p0 = np.random.normal(0.1, 5, 9)
            self.p0_vals.append(p0)
            try:
                vals_min, pcov = curve_fit(loop_fit_function, self.xdata, self.all_mean, p0=p0, maxfev=10000)
            except:
                continue
            self.opt_vals.append(vals_min)
            fitted_loop = loop_fit_function(self.xdata, *vals_min)
            yres = self.all_mean - fitted_loop
            self.res.append(yres @ yres)

        popt = self.opt_vals[np.argmin(self.res)]
        self.popt_mean = deepcopy(popt)
        self.p0_mat = [popt] * self.PR_mat.shape[0] * self.PR_mat.shape[1]

        if plot == True:
            plt.figure()
            plt.plot(self.xdata, self.all_mean, 'ko')
            fitted_loop = loop_fit_function(self.xdata, *popt)
            plt.plot(self.xdata, fitted_loop, 'k')

        print('Done with average fit')

    def k_means_averages(self): # Can be done in parallel but need to set it up that way #TODO: Parallelize
        self.fit_average()

        size = self.PR_mat.shape[0] * self.PR_mat.shape[1]

        PR_mat_flat = self.PR_mat.reshape(size, int(self.PR_mat.shape[2]))
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(PR_mat_flat)
        self.labels = kmeans.labels_
        p0_clusters_dum = []
        cluster_loops = []
        for pp in self.n_clusters:
            opt_vals = []
            res = []
            clust = PR_mat_flat[self.labels == pp]
            PR_mean = np.mean(clust, axis=0)
            if self.rolled == 1:
                PR_mean = np.roll(PR_mean, -self.max_x)
            cluster_loops.append(PR_mean)
            p0 = self.p0_mat[0]
            try:
                popt, pcov = curve_fit(loop_fit_function, self.xdata, PR_mean, p0=p0, maxfev=10000)
            except:
                kk = 0
                p0 = np.random.normal(0.1, 5, 9)
                while kk < 20:
                    try:
                        vals_min, pcov = curve_fit(loop_fit_function, self.xdata, self.all_mean, p0=p0, maxfev=10000)
                    except:
                        continue
                    kk += 1
                    opt_vals.append(vals_min)
                    fitted_loop = loop_fit_function(self.xdata, *vals_min)
                    yres = self.PR_mean - fitted_loop
                    res.append(yres @ yres)
                    popt = opt_vals[np.argmin(res)]
            p0_clusters_dum.append(popt)

        p0_clusters = p0_clusters_dum.reshape((self.PR_mat.shape[0],self.PR_mat.shape[1],9)) #matrix of p0 values corresponding to the associated cluster mean fit
        return p0_clusters

    def k_mean_guess(self,n_clusters = None):
        """
       Function which uses the previously calculated cluster averages and returns the fitted parameters for those average cluster loops

       :param n_clusters: the numbers of clusters to use in k_means clustering
        """
        if n_clusters is None:
            size = self.PR_mat.shape[0] * self.PR_mat.shape[1]
            self.n_clusters = int(size / 100)

        p0_clusters = self.k_means_averages(self,self.n_clusters)

        return p0_clusters

    def neighbor_guess(self,NN=2,ii,jj): # Must be done sequentially!
        """
        Function which uses the matrix of priors and calculates the average prior using a set nubmer of nearest neighbors

        :param NN: number of nearest neighbors to use, default = 2
        :param ii: row index for p0_mat, fed in from Fitter class function
        :param jj: column index for p0_mat, fed in from Fitter class function
        """

        xs = [ii + k for k in range(-NN, NN + 1)]
        ys = [jj + k for k in range(-NN, NN + 1)]
        nbrs = [(n, m) for n in xs for m in ys]
        cond = [all(x >= 0 for x in list(y)) for y in nbrs]
        nbrs = [d for (d, remove) in zip(nbrs, cond) if remove]
        cond2 = [all(x < self.ref_counts.shape[0] for x in list(y)) for y in nbrs]  # assumes PR_mat is square....
        nbrs = [d for (d, remove) in zip(nbrs, cond2) if remove]
        NN_indx = [self.ref_counts[v] for v in nbrs]
        prior_coefs = [self.p0_mat[k] for k in NN_indx if len(self.p0_mat[k]) != 0]
        if prior_coefs == []:
            p0 = self.popt_mean
        else:
            p0 = np.mean(prior_coefs, axis=0)

        return p0

    def random_guess(self):
        """
        Function which uses randomly chooses the piors used for loop fitting.
        """
        p0 = np.random.normal(0.1, 5, 9)

        return p0

    def hierarchical_guess(self):
        #TODO: fill in hierarchical method
        p0 = []
        return p0






