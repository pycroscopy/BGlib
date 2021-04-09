from scipy.optimize import curve_fit
from tqdm import trange
from copy import deepcopy
import pyUSID as usid
from .be_loop_fitter_v2 import BELoopFitter # Not sure if this is the correct way to call this



class Neighbor_fitting():
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
        return p0_mat

    def _do_fit(self):
        import matplotlib.pyplot as plt
        # check for data format, set up for inputs: h5_f,dc_vec_OF,PR_mat
        # --> what is already in self?
        cmap = plt.cm.plasma_r
        scale = (0, 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        cbaxs = fig.add_axes([0.92, 0.125, 0.02, 0.755])

        # set up variables for storing data
        _setup_vars()  # still need PR_mat

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

                xs = [ii + k for k in range(-NN, NN + 1)]
                ys = [jj + k for k in range(-NN, NN + 1)]
                nbrs = [(n, m) for n in xs for m in ys]
                cond = [all(x >= 0 for x in list(y)) for y in nbrs]
                nbrs = [d for (d, remove) in zip(nbrs, cond) if remove]
                cond2 = [all(x < ref_counts.shape[0] for x in list(y)) for y in nbrs]  # assumes PR_mat is square....
                nbrs = [d for (d, remove) in zip(nbrs, cond2) if remove]
                NN_indx = [ref_counts[v] for v in nbrs]
                prior_coefs = [p0_mat[k] for k in NN_indx if len(p0_mat[k]) != 0]
                if prior_coefs == []:
                    p0 = popt
                else:
                    p0 = np.mean(prior_coefs, axis=0)
                p0_refs[count] = p0
                try:
                    popt, pcov = curve_fit(loop_fit_function, xdata, ydata, p0=p0, maxfev=10000, bounds=bnds)
                except:
                    fitted_loop = loop_fit_function(xdata, *p0)
                    plt.figure()
                    plt.plot(xdata, fitted_loop, 'r')
                    plt.plot(xdata, ydata, 'k')
                    continue
                p0_mat[count] = popt  # saves fitted coefficients for the index

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