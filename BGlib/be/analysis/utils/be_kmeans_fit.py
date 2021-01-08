
from scipy.optimize import curve_fit
from tqdm import trange
from copy import deepcopy
from sklearn.cluster import KMeans
import pyUSID as usid
from .be_loop_fitter_v2 import BELoopFitter # Not sure if this is the correct way to call this

class KMeans_fitting():
    def __init__(self):
        super(BELoopFitter, self).__init__(h5_main, "Loop_Fit",
                                           variables=None, **kwargs)



    def _create_guess_datasets(self):
        """
        Creates the HDF5 Guess dataset
        """
        self._create_projection_datasets()

        self._h5_guess = create_empty_dataset(self.h5_loop_metrics, loop_fit32,
                                              'Guess')

        self._h5_guess = USIDataset(self._h5_guess)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        self.h5_main.file.flush()


    def loop_fit_function(vdc, *coef_vec):
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
    def loop_resid(coef_vec,vdc, ydata):
        y = loop_fit_function(vdc,*coef_vec)
        res = ydata-y
        ss = res@res
        return ss
    def _avg_fit(self):
        for kk in range(20):
            p0 = np.random.normal(0.1, 5, 9)
            p0_vals.append(p0)
            try:
                vals_min, pcov = curve_fit(loop_fit_function, xdata, all_mean, p0=p0, maxfev=10000)
            except:
                continue
            opt_vals.append(vals_min)
            fitted_loop = loop_fit_function(xdata, *vals_min)
            yres = all_mean - fitted_loop
            res.append(yres @ yres)
        self.popt = opt_vals[np.argmin(res)]
        popt_mean = deepcopy(popt)
        self.p0_mat = [popt] * PR_mat.shape[0] * PR_mat.shape[1]


    def _avg_cluster_fit(self):
        size = PR_mat.shape[0] * PR_mat.shape[1]
        n_clusters = int(size / 100)
        PR_mat_flat = PR_mat.reshape(size, int(PR_mat.shape[2]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(PR_mat_flat)
        labels = kmeans.labels_
        self.p0_clusters = []
        cluster_loops = []
        for pp in trange(n_clusters):
            opt_vals = []
            res = []
            clust = PR_mat_flat[labels == pp]
            PR_mean = np.mean(clust, axis=0)
            if dum == 1:
                PR_mean = np.roll(PR_mean, -max_x)
            cluster_loops.append(PR_mean)
            p0 = p0_mat[0]
            try:
                popt, pcov = curve_fit(loop_fit_function, xdata, PR_mean, p0=p0, maxfev=10000)
            except:
                kk = 0
                p0 = np.random.normal(0.1, 5, 9)
                while kk < 20:
                    try:
                        vals_min, pcov = curve_fit(loop_fit_function, xdata, all_mean, p0=p0, maxfev=10000)
                    except:
                        continue
                    kk += 1
                    opt_vals.append(vals_min)
                    fitted_loop = loop_fit_function(xdata, *vals_min)
                    yres = PR_mean - fitted_loop
                    res.append(yres @ yres)
                    popt = opt_vals[np.argmin(res)]
            self.p0_clusters.append(popt)

    def _setup_vars(self):
        # TODO: identify PR_mat
        self.p0_refs = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        self.all_mean = np.mean(np.mean(PR_mat, axis=0), axis=0)

        self.bnds = (-100, 100)
        self.p0_mat = [[]] * PR_mat.shape[0] * PR_mat.shape[1]  # empty array to store fits from neighboring pixels
        self.fitted_loops_mat = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        self.SumSq = [[]] * PR_mat.shape[0] * PR_mat.shape[1]
        self.ref_counts = np.arange(PR_mat.shape[0] * PR_mat.shape[1]).reshape(
            (PR_mat.shape[0], PR_mat.shape[1]))  # reference for finding neighboring pixels


    def _do_fit(self):
        import matplotlib.pyplot as plt
        # check for data format, set up for inputs: h5_f,dc_vec_OF,PR_mat
        # --> what is already in self?
        cmap = plt.cm.plasma_r
        scale = (0, 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        cbaxs = fig.add_axes([0.92, 0.125, 0.02, 0.755])

        # set up variables for storing data
        _setup_vars() # still need PR_mat


        count = -1
        for ii in trange(PR_mat.shape[0]):
            xind = ii
            for jj in range(PR_mat.shape[1]):
                count += 1  # used to keep track of popt vals
                yind = jj
                ydata0 = PR_mat[xind, yind, :]
                if dum == 1:
                    ydata = np.roll(ydata0, -max_x)
                else:
                    ydata = ydata0

                lab = labels[count]
                p0 = p0_clusters[lab]
                try:
                    popt, pcov = curve_fit(loop_fit_function, xdata, ydata, p0=p0, maxfev=10000)
                except:
                    p0 = popt_mean
                    try:
                        popt, pcov = curve_fit(loop_fit_function, xdata, ydata, p0=p0, maxfev=10000)
                    except:
                        continue
                p0_refs[count] = p0
                p0_mat[count] = popt

                fitted_loop = loop_fit_function(xdata, *p0_mat[count])
                fitted_loops_mat[count] = fitted_loop
                ss = r2_score(ydata, fitted_loop)
                SumSq[count] = ss
                sh = np.floor(1 + (2 ** 16 - 1) * ((ss) - scale[0]) / (scale[1] - scale[0]))
                if sh < 1:
                    sh = 1
                if sh > 2 ** 16:
                    sh = 2 ** 16

                cz = ax.plot(ii, jj, c=cmap(sh / (2 ** 16)), marker='s', markersize=7)

        scbar = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r, norm=plt.Normalize(vmin=scale[0], vmax=scale[1]))
        scbar._A = []
        cbar = plt.colorbar(scbar, cax=cbaxs)
        cbar.ax.set_ylabel('$R^2$', rotation=270, labelpad=20)

        return fig, ax, p0_refs, p0_mat, SumSq, fitted_loops_mat


