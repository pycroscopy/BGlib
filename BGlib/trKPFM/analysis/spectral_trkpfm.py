"""
Analysis code for spectral tr-KPFM
"""

class trKPFM_S_Dataset():
    def __init__(self,h5_file):
        self.h5_file = h5_file

    def load_data(self):
        import numpy as np
        import pyUSID as usid

        h5_main = usid.hdf_utils.find_dataset(self.h5_file, 'Raw_Data')[0]
        self.h5_main = usid.USIDataset(h5_main)
        self.real1 = np.real(self.h5_main[:])
        self.imag1 = np.imag(self.h5_main[:])

        h5_main2 = usid.hdf_utils.find_dataset(self.h5_file,'Raw_Data')[1]
        self.h5_main2 = usid.USIDataset(h5_main2)
        self.real2 = np.real(self.h5_main2[:])
        self.imag2 = np.imag(self.h5_main2[:])

        meas_grp = self.h5_main[:]
        for att in meas_grp.attrs.keys():
            print(att,meas_grp[att])

    def unpack_attributes(self):
        import numpy as np
        import pyUSID as usid

        self.dc_amp_vec = np.squeeze(usid.hdf_utils.find_dataset(self.h5_file,'Spectroscopic_Values')[0][:])
        self.pnts_per_pix = self.dc_amp_vec.shape[1]
        self.num_rows = int(self.h5_main['num_rows'])
        self.num_cols = int(self.h5_main['num_cols'])
        self.num_pix = self.num_cols*self.num_rows

        bias = self.h5_main['dc_amp_vec']
        self.bias_vec = bias[0][1::2]
        bias_positive = bias[0][1::2]
        bias_negative = np.flip(bias[0][3::4].T,0)
        self.bias_vec_flip = np.append(bias_negative,bias_positive)
        self.num_dc_step = np.size(self.bias_vec)
        pts_per_bias = self.pnts_per_pix/self.num_dc_step

        self.io_rate = float(self.h5_main['IO_rate'])
        self.io_time = float(self.h5_main['IO_time'])

        t_per_pix = self.dc_amp_vec.shape[1]/self.io_rate

        self.t_vec_bias = np.linspace(0,self.io_time,nm=self.io_time*self.io_rate,dtype='float')
        self.t_vec_pix = np.linspace(0,2*self.num_dc_step*self.io_time,num=self.pnts_per_pix,dtype='float')

        print("Length of time per bias value:")
        print(self.io_time)
        print("Length of time per pixel:")
        print(t_per_pix)
        print("Number of bias values:")
        print(self.num_dc_step)
        print('Number of fields:')
        print(2)

    def view_specs(self):
        import matplotlib.pyplot as plt
        import pyUSID as usid

        plt.figure(1)
        plt.plot(self.t_vec_pix, self.dc_amp_vec[0], color='b')
        plt.axis('tight')
        plt.title('Spectroscopic Dimension 1', fontsize=12, color='b')
        plt.xlabel('Time per Pixel (S)', size=15, color='b')
        plt.ylabel('Time per Step (S)', fontsize=15, color='b')

        plt.figure(2)
        plt.plot(self.t_vec_pix, self.dc_amp_vec[1], color='orange')
        plt.axis('tight')
        plt.title('Spectroscopic Dimension 2', fontsize=12, color='orange')
        plt.xlabel('Time per Pixel (S)', size=15, color='orange')
        plt.ylabel('Field State', fontsize=15, color='orange')

        plt.figure(3)
        plt.plot(self.t_vec_pix, self.dc_amp_vec[2], color='g')
        plt.axis('tight')
        plt.title('Spectroscopic Dimension 3', fontsize=12, color='g')
        plt.xlabel('Time per Pixel (S)', size=15, color='g')
        plt.ylabel('Applied Bias (V)', fontsize=15, color='g')

        usid.plot_utils.use_nice_plot_params()

    def calc_CPD(self,Xgain=1,vac=4):
        import numpy as np
        import pyUSID as usid

        amp2 = np.abs(self.h5_main2[:])
        self.CPD = (self.real1/amp2)*(vac/4)*Xgain

        self.h5_CPD_grp = usid.hdf_utils.create_indexed_group(self.h5_main.parent.parent,'CPD')

    def add_ancillary_dsets(self):
        import numpy as np
        import pyUSID as usid

        self.row_vals = np.arange(self.num_rows)
        self.col_vals = np.arange(self.num_cols)

        pos_dims = [usid.write_utils.Dimensions('Cols','m',self.col_vals),
                         usid.write_utils.Dimensions('Rows','m',self.row_vals)]
        spec_dims = [usid.write_utils.Dimensions('Time','S',self.t_vec_bias),
                          usid.write_utils.Dimensions('Field','binary',np.array([0,1])),
                          usid.write_utils.Dimensions('Bias','V',self.bias_vec)]

        self.h5_CPD = usid.hdf_utils.write_main_dataset(self.h5_CPD_grp,self.CPD,'CPD Data',
                                                        'CPD','V',pos_dims,spec_dims,
                                                        dtype=np.float32, compression='gzip')

        self.h5_main.file.flush()
        print(usid.hdf_utils.get_attributes(self.h5_CPD))

    def compute_SVD(self,num_components=100):
        import pycroscopy as px
        import numpy as np

        decomposer = px.processing.svd_utils.SVD(self.h5_CPD,num_components=num_components)
        h5_svd_grp = decomposer.compute(override=True)

        abun_maps = np.reshape(h5_svd_grp['U'][:,:25], (self.num_rows,self.num_cols,-1))
        usid.plot_utils.plot_map_stack(abun_maps,num_comps=4,title='SVD Abundance Maps',reverse_dims=True,
                                       color_bar_mode='single',cmap='inferno',title_yoffset=0.95)

        usid.plot_utils.plot_scree(h5_svd_grp['S'],title='Note the exponential drop of variance with number of components')

        first_evecs = h5_svd_grp['V'][:4,:]
        _ = usid.plot_utils.plot_curves(self.t_vec_pix,first_evecs,xlabel='Time (s)',ylabel='Amplitude (a.u)',
                                        title = 'SVD Eigenvectors',evenly_spaced=True)

        usid.plot_utils.use_nice_plot_params()

    def parse_data(self):
        import pyUSID as usid
        import numpy as np

        dset_list = usid.hdf_utils.get_auxiliary_datasets(self.h5_CPD,['Position_Indices','Position_Values',
                                                                       'Spectroscopic_Indices','Spectroscopic_Values'])
        h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals = dset_list

        ndim_CPD, success, labels = usid.hdf_utils.reshape_to_n_dims(self.h5_CPD, get_labels=True)

        OnCPD_stack = np.squeeze(ndim_CPD[:, :, :, 1, :])  ##remove Off Field
        liampos = OnCPD_stack[:, :, :, 0::2]  ##Postive Bias Values
        liamneg = np.flip(OnCPD_stack[:, :, :, 1::2], 3)  ##Negative  Bias Values (flipped)

        OnCPD_stack = np.concatenate((liamneg, liampos), axis=3)  ##Putting back togehter
        OnCPD2D_stack = np.reshape(OnCPD_stack, [self.num_pix, -1])  ##Reshape pixel  x spec

        OffCPD_stack = np.squeeze(ndim_CPD[:, :, :, 0, :])
        liampos = OffCPD_stack[:, :, :, 0::2]
        liamneg = np.flip(OffCPD_stack[:, :, :, 1::2], 3)

        OffCPD_stack = np.concatenate((liamneg, liampos), axis=3)
        OffCPD2D_stack = np.reshape(OffCPD_stack, [self.num_pix, -1])  ##Reshape pixel  x spec

        t_vec_bias_parsed = np.linspace(0, self.io_time, num=self.t_per_bias * self.io_rate, dtype='float')

        pos_dims = [usid.write_utils.Dimension('Cols', 'm', self.cols_vals),
                    usid.write_utils.Dimension('Rows', 'm', self.rows_vals)]

        spec_dims = [usid.write_utils.Dimension('Time', 'S', self.t_vec_bias),
                     usid.write_utils.Dimension('Bias', 'V', self.bias_vec_flip)]

        h5_parsed_group = usid.hdf_utils.create_indexed_group(self.h5_main.parent.parent, 'Parsed_Data')

        h5_CPD_on = usid.hdf_utils.write_main_dataset(h5_parsed_group, OnCPD2D_stack,
                                                      'CPD_on',  # Name of Main dataset
                                                      'CPD',  # Quantity
                                                      'V',
                                                      pos_dims, spec_dims,
                                                      dtype=np.float32,
                                                      compression='gzip')

        h5_CPD_off = usid.hdf_utils.write_main_dataset(h5_parsed_group, OffCPD2D_stack,
                                                       'CPD_off',  # Name of Main dataset
                                                       'CPD',  # Quantity
                                                       'V',
                                                       None, None,
                                                       h5_pos_inds=h5_CPD_on.h5_pos_inds,
                                                       h5_pos_vals=h5_CPD_on.h5_pos_vals,
                                                       h5_spec_inds=h5_CPD_on.h5_spec_inds,
                                                       h5_spec_vals=h5_CPD_on.h5_spec_vals, dtype=np.float32,
                                                       compression='gzip')

        dset_list = usid.hdf_utils.get_auxiliary_datasets(h5_CPD_on, ['Position_Indices', 'Position_Values',
                                                                      'Spectroscopic_Indices', 'Spectroscopic_Values'])
        h5on_pos_inds, h5on_pos_vals, h5on_spec_inds, h5on_spec_vals = dset_list

        h5_CPD_on[:], success = usid.hdf_utils.reshape_from_n_dims(OnCPD_stack, h5_pos=h5on_pos_inds,
                                                                   h5_spec=h5on_spec_inds)
        h5_CPD_off[:], success = usid.hdf_utils.reshape_from_n_dims(OffCPD_stack, h5_pos=h5on_pos_inds,
                                                                    h5_spec=h5on_spec_inds)

        usid.hdf_utils.print_tree(self.h5_file)


    def fit_relaxation_single(self,h5,bias_pnt=0,window=55,polyval=0):
        from scipy.signal import savgol_filter
        from scipy.optimize import curve_fit
        import numpy as np

        def fit_exp(x, A, tau, y0, x0):
            return A * np.exp(-(x-x0)/ tau)+y0

        CPD_vec, success = h5.slice({'Cols':4,'Rows':3,'Bias':bias_pnt})

        sectn_smooth = savgol_filter(CPD_vec,window,polyval)

        p0s = [0.5,50,0.1,0]
        bnds = ([-1,0,-6,-1],[1,500,6,1])

        popt1, pcov = curve_fit(fit_exp,self.t_vec_bias*1e3,self.CPD,bounds=bnds,p0=p0s)
        popt2, pcov2 = curve_fit(fit_exp,self.t_vec_bias*1e3,sectn_smooth,bounds=bnds,p0=p0s)

        perr = np.sqrt(np.diag(pcov))
        perr2 = np.sqrt(np.diag(pcov2))

        fit_vec = fit_exp(self.t_vec_bias*1e3, *popt1)
        ss_res = np.sum((CPD_vec - fit_vec)**2)
        ss_tot = np.sum((CPD_vec - np.mean(CPD_vec))**2)
        r2 = 1 - (ss_res/ss_tot)

        fit_vec2 = fit_exp(self.t_vec_bias * 1e3, *popt2)
        ss_res2 = np.sum((CPD_vec - fit_vec2) ** 2)
        ss_tot2 = np.sum((CPD_vec - np.mean(CPD_vec)) ** 2)
        r22 = 1 - (ss_res2 / ss_tot2)

        print(' Fit Coefficients: (before smoothing)')
        print(' A: ', popt1[0], ' +/- ', perr[0])
        print('tau (ms):', popt1[1], ' +/- ', perr[1])
        print('y0 Offset (V):', popt1[2], ' +/- ', perr[2])
        print('x0 Offset (ms):', popt1[3], ' +/- ', perr[3])
        print("Goodness of Fit:" + str(r2))

        print(' Fit Coefficients:(after smoothing)')
        print(' A: ', popt2[0], ' +/- ', perr2[0])
        print('tau (ms):', popt2[1], ' +/- ', perr2[1])
        print('y0 Offset (V):', popt2[2], ' +/- ', perr2[2])
        print('x0 Offset (ms):', popt2[3], ' +/- ', perr2[3])
        print("Goodness of Fit:" + str(r22))


    def fit_relaxation_all(self,h5,biaspnt,window=55,polyval=0):
        from scipy.signal import savgol_filter
        import numpy as np
        from scipy.optimize import curve_fit

        def fit_exp(x, A, tau, y0, x0):
            return A * np.exp(-(x-x0)/ tau)+y0

        fit = np.zeros([self.num_cols,self.num_rows,4])
        goodness = np.zeros([self.num_cols,self.num_rows])

        p0s = [0.5, 50, 0.1, 0]
        bnds = ([-1, 0, -6, -1], [1, 500, 6, 1])

        for kk in range(self.num_cols):
            for jj in range(self.num_rows):
                CPD_vec, success = h5.slice({'Cols':kk,'Rows':jj,'Bias': biaspnt})
                sectn_smooth = savgol_filter(CPD_vec,window,polyval)
                popt1,_ = curve_fit(fit_exp,self.t_vec_bias*1e3, sectn_smooth,p0 = p0s,bounds=bnds,check_finite=False)
                fit_vec = fit_exp(self.t_vec_bias*1e3,*popt1)

                ss_res = np.sum((sectn_smooth-fit_vec)**2)
                ss_tot = np.sum((sectn_smooth-np.mean(sectn_smooth))**2)
                r2 = np.abs(1-(ss_res/ss_tot))
                fit[kk,jj,:] = popt1
                goodness[kk,jj] = r2

        self.relaxation_fit = {'fit vals':fit,'goodness':goodness}


    def plot_relaxation_fit(self):
        import numpy as np
        import pyUSID as usid
        import matplotlib.pyplot as plt

        fit_mat1 = np.nan_to_num(self.relaxation_fit['fit vals'])
        fit_mat1.shape

        mask = np.where(np.abs(self.relaxation_fit['goodness']) > 0.6, True, False)

        usid.plot_utils.use_nice_plot_params()

        data = np.squeeze(fit_mat1[:, :, 1])

        data_masked = data

        data_masked[mask == False] = 0

        fig5, A5 = plt.subplots(figsize=(10, 5))
        usid.plot_utils.plot_map(A5, np.abs(np.rot90(mask)), cmap='binary', vmin=0, vmax=1, show_xy_ticks=False,
                                 cbar_label='Binary')

        data_masked = np.squeeze(fit_mat1[:, :, 0])
        data_masked[mask == False] = 0
        fig1, A1 = plt.subplots(figsize=(10, 5))
        usid.plot_utils.plot_map(A1, np.squeeze(np.rot90(data_masked)), stdevs=1.5, show_xy_ticks=False, cbar_label='V')

        data_masked = np.squeeze(fit_mat1[:, :, 1])
        data_masked[mask == False] = 0
        fig2, A2 = plt.subplots(figsize=(10, 5))
        usid.plot_utils.plot_map(A2, np.squeeze(np.rot90(data_masked)), stdevs=1.5, show_xy_ticks=False,
                                 cbar_label='ms')

        data_masked = np.squeeze(fit_mat1[:, :, 2])
        data_masked[mask == False] = 0
        fig3, A3 = plt.subplots(figsize=(10, 5))
        usid.plot_utils.plot_map(A3, np.squeeze(np.rot90(data_masked)), stdevs=1.5, show_xy_ticks=False,
                                 cbar_label='Voffset')

        data_masked = np.squeeze(fit_mat1[:, :, 3])
        data_masked[mask == False] = 0
        fig4, A4 = plt.subplots(figsize=(10, 5))
        usid.plot_utils.plot_map(A4, np.squeeze(np.rot90(data_masked)), stdevs=1.5, show_xy_ticks=False,
                                 cbar_label='ms')


    def KMeans(self,n_clusters):
        import pycroscopy as px
        import numpy as np
        from sklearn.cluster import KMeans

        estimator = px.processing.Cluster(self.h5_CPD,KMeans(n_clusters=n_clusters))
        h5_kmeans_grp = estimator.compute(self.h5_CPD,override=True)
        h5_kmeans_labels = h5_kmeans_grp['Labels']
        h5_kmeans_mean_resp = h5_kmeans_grp['Mean_Response']

        t_vec_pix_parsed = np.linspace(0, self.num_dc_step * self.io_time, num=self.pnts_per_pix / 2, dtype='float')

        fig_lab, fig_centroids = self.plot_cluster_results_separately(h5_kmeans_labels[:, :].reshape(self.num_rows, self.num_cols),
                                                                 h5_kmeans_mean_resp[:, :], t_vec_pix_parsed * 1000,
                                                                 legend_mode=2)



    def plot_cluster_results_separately(self,labels_mat, cluster_centroids, bias_vec, legend_mode=1, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np


        num_clusters = cluster_centroids.shape[0]
        fig_lab, axis_lab = plt.subplots(figsize=(5.5,5))
        _, _ = usid.plot_utils.plot_map(axis_lab, labels_mat,
                                                  clim=[0, num_clusters-1],
                                                  cmap=plt.get_cmap('viridis', num_clusters),
                                                  aspect='auto', show_xy_ticks=True, **kwargs)
        axis_lab.set_xlabel('X ($\mu$m)', fontsize=16)
        axis_lab.set_ylabel('Y ($\mu$m)', fontsize=16)
        axis_lab.set_title('K-Means Cluster Labels', fontsize=16)
        fig_lab.tight_layout()

        # Plot centroids
        fig_width = 5.0
        if legend_mode not in [0, 1]:
            fig_width = 5.85
        fig_centroids, axis_centroids = plt.subplots(figsize=(fig_width, 5))
        colors = [ plt.cm.viridis(x) for x in np.linspace(0, 1, cluster_centroids.shape[0]) ]

        # print('Number of pixels in each cluster:')
        for line_ind in range(cluster_centroids.shape[0]):
            # cmap=plt.cm.jet
            line_color=colors[line_ind]
            line_label = 'Cluster ' + str(line_ind)
           # num_of_cluster_members = len(np.where(labels==line_ind)[0])
            # print ("Cluster " + str(line_ind) + ': ' + str(num_of_cluster_members))
            #if num_of_cluster_members > 10:
            axis_centroids.plot(bias_vec, cluster_centroids[line_ind,:],
                                label=line_label, color=line_color) # marker='o',
        axis_centroids.set_xlabel('Time (ms)', fontsize=16)
        axis_centroids.set_ylabel('CPD (V)', fontsize=16)
        axis_centroids.set_title('K-Means Cluster Centroids', fontsize=16)
        if legend_mode==0:
            axis_centroids.legend(loc='lower right', fontsize=14)
        elif legend_mode==1:
            axis_centroids.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
        else:
            sm = usid.plot_utils.make_scalar_mappable(0, num_clusters-1,
                                                   cmap=usid.plot_utils.discrete_cmap(num_clusters))
            plt.colorbar(sm)
        usid.plot_utils.set_tick_font_size(axis_centroids, 14)
        fig_centroids.tight_layout()

        return fig_lab, fig_centroids


    def plot_cluster_results_2D(self,labels_mat, cluster_centroids,vmin,vmax, legend_mode=1, **kwargs):
            import matplotlib.pyplot as plt
            import pyUSID as usid
            import numpy as np

            num_clusters = cluster_centroids.shape[0]
            fig_lab, axis_lab = plt.subplots(figsize=(5.5,5))
            _, _ = usid.plot_utils.plot_map(axis_lab, labels_mat,
                                                      clim=[0, num_clusters],
                                                      cmap=plt.get_cmap('viridis', num_clusters),
                                                      aspect='auto', show_xy_ticks=True, **kwargs)
            axis_lab.set_xlabel('X ($\mu$m)', fontsize=16)
            axis_lab.set_ylabel('Y ($\mu$m)', fontsize=16)
            axis_lab.set_title('K-Means Cluster Labels', fontsize=16)
            fig_lab.tight_layout()

            # Plot centroids
            fig_width = 5.0
            if legend_mode not in [0, 1]:
                fig_width = 5.85

            fig_centroids, axis_centroids = plt.subplots(figsize=(fig_width, 5))
            ndim_test, success, labels = usid.hdf_utils.reshape_to_n_dims(cluster_centroids, get_labels=True)
            fig, axs = plt.subplots(2, 2,figsize=(5.5,5))


            cm = ['jet']
            index=0
            for col in range(2):

                for row in range(2):
                    ax = axs[col, row]
                    pcm = ax.pcolor(self.time_vec,self.bias_vec,np.rot90(ndim_test[index,:,:]),cmap='jet',vmin=vmin, vmax=vmax)

                    #fig.colorbar(pcm, ax=ax)
                    ax.set_title('Cluster: '+str(index+1), fontsize=12)
                    if row==0:
                        ax.set_ylabel('Bias (V)', fontsize=16)
                    if row==0 & col==1:
                        ax.set_ylabel('Bias (V)', fontsize=16)
                    if col==1:
                        ax.set_xlabel('Time (S)', fontsize=16)
                        #ax.set_title('Cluster: '+str(index+1), fontsize=12)
                    print(row,col)

                    index=index+1


            usid.plot_utils.use_nice_plot_params()

            fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.4, hspace=0.1)
            cb_ax = fig.add_axes([1.0, 0.1, 0.05, 0.8])
            cbar = fig.colorbar(pcm, cax=cb_ax)
            plt.tight_layout()
            plt.show()

            def fitexp(x, A, tau, y0, x0):
                return A * np.exp(-(x - x0) / tau) + y0

            def fitbiexp(x, A1, tau1, A2, tau2, y0, x0):
                return A1 * np.exp(-(x - x0) / tau1) + A2 * np.exp(-(x - x0) / tau2) + y0

            return fig_lab, fig_centroids











