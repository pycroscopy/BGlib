"""
Analysis codes for doing linear tr-KPFM: where slow scan is disabled
"""

class trKPFM_L_Analyzer():
    def __init__(self,usid_dataset):
        """
        h5_main: h5 file
        scan_rate: frequency of scan (ex: 0.5 Hz)
        scan_size: physical dimension of scan (ex: 80 um)
        dim: distance between electrodes in cm (for Efield calculations)
        """
        self.h5_main = usid_dataset
        # self.dataset_type = 'trKPFM-USIDataset'
        super(trKPFM_L_Analyzer, self).__init__(h5_main,'Linear_trKPFM')

    def add_scan_info(self,scan_rate,scan_size,dim):
        # self.h5_main = self.source_h5_dataset
        self.scan_rate = scan_rate
        self.scan_size = scan_size
        self.dim = dim


    def split_voltages(self):
        import numpy as np
        t = np.array([ii for ii in range(self.ndim_form[0])]) * (1 / self.scan_rate) * 2
        y = np.linspace(0, self.scan_size, self.ndim_form[0])

        cnttot = 0
        count = 1
        cntmax = 0
        indx = []
        vs = []
        for ii in range(1, len(t)):
            if np.rint(self.volt[ii, 0]) != np.rint(self.volt[ii - 1, 0]):
                cnttot = cnttot + 1  # counts the total number of voltage switches in the data set
                if np.rint(self.volt[ii, 0]) != np.rint(self.volt[ii, -1]):  # checks to see if the voltage is changed in the middle fo the scan line
                    indx.append(ii + 1)
                    vs.append(self.volt[ii+1,0])
                else:
                    indx.append(ii)
                    vs.append(self.volt[ii, 0])
                if count > cntmax:
                    cntmax = count
                    count = 1
                else:
                    count = 1
            else:
                count = count + 1
        self.vs = vs
        self.cntmax = cntmax
        self.cnttot = cnttot
        self.indx = indx
        self.y = y
        self.t = t
        # return vs, cntmax, cnttot, indx, t, y

    def unpack_data(self):

        """
          Not sure how to get the scan rate and scan size out of the igor file...

          scan_rate: in Hz
          scan_size: in um
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


        # data_dict = {'Voltage': volt, 'CPD': pot, 'Height': height, 'Phase': phase, 'Amplitude': amp_data, 'ndim_form': ndim_form,
        #              'scan_rate': self.scan_rate, 'scan_size': self.scan_size}

        # vs, cntmax, cnttot, indx, t, y = self.split_voltages(data_dict)
        self.split_voltages()
        # data_dict.update({'indx': indx, 'vs': vs, 'count max': cntmax,'count total':cnttot,'t':t,'y':y})
        # return data_dict

    def compute_voltage_averages(self):
        import numpy as np

        avgs = []
        zeroavg = np.mean(self.pot[:self.indx[0] - 1, :], axis=0)
        for ii in range(len(self.indx) - 1):
            avgs.append(np.mean(self.pot[self.indx[ii]:self.indx[ii + 1] + 1, :], axis=0) - zeroavg)

        # data_dict.update({'zeroavg':zeroavg,'avgs':avgs})
        self.zeroavg = zeroavg
        self.avgs = avgs


    def plot_CPD_voltages(self,method='Raw',window=13,poly=3):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import signal as si

        if method not in ['Raw','Static_rm','Efield']:
            print(method+" is not an option. Please choose from either: 'Raw', 'Static_rm', or 'Efield'.")
            return
        cmap = plt.cm.get_cmap('plasma', self.cntmax)
        jj = 0
        fig, axs = plt.subplots(nrows=int(self.cnttot / 2 + 1), ncols=2, sharex='col', figsize=(15, 10))
        axs[0, 0].axis('off')
        axs[1, 0].set_title('Biasing')
        axs[0, 1].set_title('Zero Voltage After Bias')
        axs[0, 1].text(0.3, 0.8, '0 V', transform=axs[0, 1].transAxes)
        axs[int(self.cnttot / 2), 0].set_xlabel('$\mu$m')
        axs[int(self.cnttot / 2), 1].set_xlabel('$\mu$m')
        cbaxs = fig.add_axes([0.91, 0.095, 0.02, 0.71])
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
        axs[0, 1].set_ylabel('CPD (V)', rotation=90, labelpad=2)
        axs[0, 1].axvspan(self.y[0], 0, facecolor='0.5', alpha=0.5)

        col = 1
        row = 0
        lab = 0
        if method == 'Efield':
            if np.rint(self.volt[self.indx[0]-1,-1])!=np.rint(self.volt[self.indx[0]-1,0]):
                lab = 1
            else:
                lab = 0
        for ii in range(len(self.t)):
            if method=='Efield':
                if ii < self.indx[0]:
                    continue
            if np.rint(self.volt[ii, -1]) != np.rint(self.volt[ii, 0]):
                jj = jj + 1
                lab = 1
                continue
            if ii != 0:
                if lab == 1:
                    prev = np.rint(self.volt[ii - 2, 0])
                else:
                    prev = np.rint(self.volt[ii - 1, 0])
                if np.rint(self.volt[ii, 0]) != prev:
                    jj = 0
                    if np.rint(self.volt[ii, 0]) != 0:
                        col = 0
                        row = row + 1
                        if lab==1:
                            zero_pot = self.pot[ii-2,:]
                        else:
                            zero_pot = self.pot[ii-1,:]
                    else:
                        col = 1
                        zero_pot = self.zeroavg
                    S = np.array2string(np.rint(self.volt[ii, 0])) + ' V'
                    axs[row, col].text(0.3, 0.8, S, transform=axs[row, col].transAxes)
                    if method == 'Raw':
                        axs[row, col].set_ylabel('CPD (V)', rotation=90, labelpad=2)
                    if method == 'Static_rm':
                        axs[row, col].set_ylabel('$\Delta$CPD (V)', rotation=90, labelpad=2)
                    if method == 'Efield':
                        axs[row, col].set_ylabel('E (V/m)', rotation=90, labelpad=2)
                        axs[row, col].axhline(y = self.volt[ii,0]/self.dim,color=(0.5,0.5,0.5),linestyle='--')
                if ii == self.indx[0]:
                    zero_pot = self.zeroavg
            if method == 'Raw':
                yy = self.pot[ii, :]
                axs[row, col].plot(self.y, yy, c=cmap(jj))
            elif method == 'Static_rm':
                yy = self.pot[ii, :] - self.zeroavg
                axs[row, col].plot(self.y, yy, c=cmap(jj))
            elif method == 'Efield':
                smooth = si.savgol_filter(self.pot[ii,:]-zero_pot,window,poly)
                yy = np.diff(smooth)/np.diff(self.y)*1e4
                axs[row, col].plot(self.y[:-1],yy,c=cmap(jj))

            lab = 0
            jj = jj + 1
        scbar = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                      norm=plt.Normalize(vmin=0, vmax=self.t[self.cntmax]))
        scbar._A = []
        cbar = plt.colorbar(scbar, cax=cbaxs)
        cbar.ax.set_ylabel('Relaxation Time (s)', rotation=270, labelpad=15, size=12)
        cbar.ax.tick_params(labelsize=12)
        fig.align_ylabels(axs)
        fig.subplots_adjust(wspace=.3)
        return fig, axs

    def calc_relaxation(self,dist):
        """
        dist: list of distances (real space) of wanted relaxtion calculation
        """
        return 'Function under construction'
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.optimize import curve_fit

        t_const = []
        t_const_z = []
        a_vals = []
        b_vals = []
        a_vals_z = []
        b_vals_z = []
        lin0 = []
        lin1 = []
        p = -1
        cmap = plt.cm.get_cmap('plasma', len(self.indx))

        def exp_decay(x, a, b, k):
            return a * np.exp(x * k) + b

        fig, ax = plt.subplots(ncols=len(dist), nrows=2, figsize=(11, 8))
        ax[1, 0].set_ylabel('CPD - Static (V) After Biasing')
        for k in dist:
            jj = (np.abs(self.y - k)).argmin()
            p = p + 1
            t_c = []
            a_0 = []
            b_0 = []
            t_z = []
            a_z = []
            b_z = []
            ax[0, p].set_xlabel('Time (s)')
            ax[1, p].set_xlabel('Time (s)')
            ax[0, p].yaxis.set_ticks_position('both')
            ax[1, p].yaxis.set_ticks_position('both')
            ax[0, p].title.set_text(str(k) + '$\mu$m \n from edge')
            ax[0, p].axhline(y=0, linestyle=':', linewidth=0.5, color='k')
            ax[1, p].axhline(y=0, linestyle=':', linewidth=0.5, color='k')

            if p != 0:
                if k != dist[-1]:
                    ax[0, p].set_yticklabels('')
                    ax[1, p].set_yticklabels('')
                else:
                    ax[0, p].yaxis.tick_right()
                    ax[1, p].yaxis.tick_right()

            for ii in range(len(self.indx)):
                if np.rint(self.volt[self.indx[ii], 0]) == 0:
                    if self.indx[ii] == self.indx[-1]:
                        y_array = self.pot[self.indx[ii]:-1, jj] - self.zeroavg[jj]
                    else:
                        y_array = self.pot[self.indx[ii]:self.indx[ii + 1] - 1, jj] - self.zeroavg[jj]
                    x_array = self.t[:len(y_array)]
                    lin1 += ax[1, p].plot(x_array, y_array, c=cmap(ii),
                                          label=str(int(np.rint(self.volt[self.indx[ii - 1], 0]))) + ' V$_{app}$')
                    if self.volt[self.indx[ii - 1], 0] > 0:
                        yint = np.min(y_array)
                    else:
                        yint = np.max(y_array)
                    popt, pcov = curve_fit(exp_decay, x_array, y_array, p0=[0.5, yint, -0.5])
                    ax[1, p].plot(x_array, exp_decay(x_array, *popt), color='k', linestyle='--')
                    t_z.append(popt[2])
                    b_z.append(popt[1])
                    a_z.append(popt[0])
                else:
                    y_array = self.pot[self.indx[ii]:self.indx[ii + 1] - 1, jj] - self.zeroavg[jj]
                    x_array = self.t[:len(y_array)]
                    lin0 += ax[0, p].plot(x_array, y_array, c=cmap(ii),
                                          label=str(int(np.rint(self.volt[self.indx[ii], 0]))) + ' V')
                    if self.volt[self.indx[ii], 0] > 0:
                        yint = np.max(y_array)
                    else:
                        yint = np.min(y_array)
                    popt, pcov = curve_fit(exp_decay, x_array, y_array, p0=[0.5, yint, -0.5])
                    ax[0, p].plot(x_array, exp_decay(x_array, *popt), color='k', linestyle='--')
                    t_c.append(popt[2])
                    b_0.append(popt[1])
                    a_0.append(popt[0])
            if len(t_const) == 0:
                t_const = np.asarray(t_c)
                a_vals = np.asarray(a_0)
                b_vals = np.asarray(b_0)
            else:
                t_const = np.vstack((t_const, np.asarray(t_c)))
                a_vals = np.vstack((a_vals, np.asarray(a_0)))
                b_vals = np.vstack((b_vals, np.asarray(b_0)))
            if len(t_const_z) == 0:
                t_const_z = np.asarray(t_z)
                a_vals_z = np.asarray(a_z)
                b_vals_z = np.asarray(b_z)
            else:
                t_const_z = np.vstack((t_const_z, np.asarray(t_z)))
                a_vals_z = np.vstack((a_vals_z, np.asarray(a_z)))
                b_vals_z = np.vstack((b_vals_z, np.asarray(b_z)))
        fig.subplots_adjust(wspace=-.1)
        fig.subplots_adjust(hspace=0.3)

        plt.setp(ax[0, :], ylim=(-1.2, 1.2))
        plt.setp(ax[1, :], ylim=(-0.35, 0.35))
        ax[0, 0].legend(ncol=2, handlelength=1.0, handletextpad=0.4, loc='upper left', columnspacing=1,
                              prop={'size': 8})
        ax[1, 0].legend(ncol=2, handlelength=1.0, handletextpad=0.4, loc='upper left', columnspacing=1,
                              prop={'size': 8})
        return fig,ax

    def view_line_scans(self,dist):
        """
        CURRENTLY BROKEN
        dist: distances in real space to plot
        """
        return 'Function under construction'
        import numpy as np
        import matplotlib.pyplot as plt
        avgs = []
        for ii in range(len(self.indx) - 1):
            avgs.append(np.mean(self.pot[self.indx[ii]:self.indx[ii + 1] + 1, :], axis=0) - self.zeroavg)

        cmap = plt.cm.get_cmap('plasma', len(self.indx))
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(self.y, self.zeroavg, 'r--', label='Static CPD')
        ax[0].set_xlabel('Distance ($\mu$m)')
        for ii in range(len(self.avgs)):
            ax[0].plot(self.y, self.avgs[ii], c=cmap(ii))
        ax[0].legend()
        ax[0].axhline(y=0, color=(0.5, 0.5, 0.5), linestyle='--')

        cmap1 = plt.cm.get_cmap('viridis', self.ndim_form[0])

        for ii in dist:
            ii = (np.abs(self.y - ii)).argmin()
            v = np.zeros(len(self.zeroavg))
            ax[1].plot(self.t[:], self.pot[:, ii] - self.zeroavg[ii], c=cmap1(ii), label=str(int(self.y[ii])) + ' $\mu$m',
                       linewidth=1.5)
        ax[1].set_xlabel('Time (s)')
        ax[1].legend(ncol=1, title='Distance from edge', bbox_to_anchor=(1, 1), loc="upper left")
        ax[1].axhline(y=0, color=(0.5, 0.5, 0.5), linestyle='--')
        for k in self.indx:
            j = self.indx.index(k)
            if np.rint(self.vs[j]) == 0:
                continue
            txty = self.pot[k,-1]-self.zeroavg[-1]
            if txty > 0:
                ax[1].text(self.t[k] + 10, txty, str(int(np.rint(self.vs[j]))) + ' V', horizontalalignment='left',
                           verticalalignment='bottom')
            else:
                ax[1].text(self.t[k] + 10, txty, str(int(np.rint(self.vs[j]))) + ' V', horizontalalignment='left',
                           verticalalignment='top')
        fig.subplots_adjust(wspace=.3)

    def calc_decay(self, jj, indx):
        import numpy as np
        from scipy.optimize import curve_fit

        def exp_decay(x, a, b, k):
            return a * np.exp(x * k) + b

        ii = self.indx.index(indx)  #TODO: Make sure this line is correct...
        if indx == self.indx[-1]:
            y_array = self.pot[indx:-1, jj] - self.zeroavg[jj]
        else:
            y_array = self.pot[indx:self.indx[ii + 1] - 1, jj] - self.zeroavg[jj]

        x_array = self.t[:len(y_array)]

        if self.volt[self.indx[ii - 1], 0] > 0:
            yint = np.min(y_array)
        else:
            yint = np.max(y_array)

        popt, pcov = curve_fit(exp_decay, x_array, y_array, p0=[0.5, yint, -0.5])
        return popt, pcov

    def decay_analysis_d(self):
        import numpy as np

        t_c_mat = []
        b_0_mat = []
        a_0_mat = []

        for v in range(len(self.indx)):
            t_c = []
            b_0 = []
            a_0 = []
            if np.rint(self.vs[v]) == 0:
                continue

            for ii in range(len(self.y)):
                ind = self.indx[v]
                try:
                    popt, pcov = self.analyzer.calc_decay(self, ii, ind)  # data, distance, voltage start
                    t_c.append(popt[2])
                    b_0.append(popt[1])
                    a_0.append(popt[0])
                except:
                    popt = np.nan
                    pcov = np.nan
                    t_c.append(popt)
                    b_0.append(popt)
                    a_0.append(popt)

            t_c_mat.append(t_c)
            b_0_mat.append(b_0)
            a_0_mat.append(a_0)
        return t_c_mat, b_0_mat, a_0_mat