
##
import os
# 1) Force Qt-backed Matplotlib (avoid MacOSX backend which must run on main thread)
os.environ.setdefault("MPLBACKEND", "QtAgg")   # or "Qt5Agg"

# 2) Tame inner BLAS/OpenMP thread storms (critical on macOS + Accelerate/OpenBLAS/MKL)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# (optional but helpful) make Qt explicit
os.environ.setdefault("QT_API", "pyqt5")
import sys
import faulthandler
faulthandler.enable()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTabWidget, QFileDialog, QPushButton, QTextEdit, QFrame,
    QGridLayout, QLineEdit, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
import BGlib.be as belib
import numpy as np
import os
import sidpy
import SciFiReaders as sr
from BGlib.be.analysis.utils.sidpy_sho_fitter import SHOestimateGuess, SHOestimateGuess, SHO_fit_flattened
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import QObject, pyqtSignal
from BGlib.be.analysis.utils.be_loop import projectLoop, loop_fit_function, generate_guess, generate_shallow_guess, generate_deep_guess, generate_deepGP_guess, calc_switching_coef_vec


class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        """Called whenever print() outputs text."""
        if text.strip():  # Avoid empty newlines
            self.text_written.emit(text)

    def flush(self):
        """Needed for Python 3 compatibility."""
        pass

class SidpyBandExcitationProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sidpy Band-Excitation Processor")
        self.setGeometry(100, 100, 1000, 600)

        # Simulate frequency vector (placeholder for now)
        self.freq_vec = np.linspace(320e3, 350e3, 90)

        # Create the main tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add three tabs
        self.tabs.addTab(self.create_raw_data_tab(), "Load / View Raw Data")
        self.tabs.addTab(self.create_sho_tab(), "Fit and View SHO")
        self.tabs.addTab(self.create_fit_loops_tab(), "Fit Loops")
        self.setup_output_redirect()          

    def setup_output_redirect(self):
        """Redirect sys.stdout to output_box."""
        self.stdout_stream = EmittingStream()
        self.stdout_stream.text_written.connect(self.append_output_text)
        sys.stdout = self.stdout_stream
        sys.stderr = self.stdout_stream  # Optional: also redirect errors
        
    def append_output_text(self, text):
        """Append new text to the output box and scroll to the end."""
        self.output_box.moveCursor(QTextCursor.End)  # Move to the end before inserting
        self.output_box.insertPlainText(text)
        self.output_box.moveCursor(QTextCursor.End)  # Ensure the view scrolls to bottom
        return
        
    # ==============================
    # Tab 1: Loading / View Raw Data
    # ==============================
    def create_raw_data_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Load your raw data file and view its contents below:")
        layout.addWidget(instructions)

        # File load button
        load_button = QPushButton("Load Raw Data")
        load_button.clicked.connect(self.load_raw_data)
        layout.addWidget(load_button)

        # Text area to display raw data
        self.raw_data_display = QTextEdit()
        self.raw_data_display.setReadOnly(True)
        layout.addWidget(self.raw_data_display)

        tab.setLayout(layout)
        return tab

    def load_raw_data(self):
        """Open a file dialog and load a h5 BEPS file, and convert it to a sidpy dataset """
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Raw HDF5 BEPS Data File", "", "All Files (*);;h5 (*.h5);;hdf5 (*.hdf5)", options=options
        )
        if filename:
            try:
                patcher = belib.translators.LabViewH5Patcher()
                patcher.translate(filename)
                reader = sr.Usid_reader(filename)
                self.beps_raw = reader.read()[:4,:4,:,:,:,:1]#TODO: temporary for testing only!!
                self.freq_axis = self.beps_raw.labels.index('Frequency (Hz)')
                self.freq_vec = self.beps_raw._axes[self.freq_axis].values
                self.all_dims = np.arange(len(self.beps_raw.shape))
                self.ind_dims = np.delete(self.all_dims, self.freq_axis)
                #TODO: Only for BE datasets. Need to check that the vector is correct
                # self.lower_bound_fields[1].setText(f"{self.freq_vec.min() / 1e3:.2f}")
                # self.upper_bound_fields[1].setText(f"{self.freq_vec.max() / 1e3:.2f}")
                
            except Exception as e:
                self.raw_data_display.setText(f"Error loading file:\n{e}")

    # ==============================
    # Tab 2: Fit and View SHO
    # ==============================    
    def create_sho_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()

        # ===== Left side: Controls =====
        controls_layout = QVBoxLayout()

        # ---- Fitting Parameters Frame ----
        fit_frame = QFrame()
        fit_frame.setFrameShape(QFrame.StyledPanel)
        fit_layout = QGridLayout()

        fit_layout.addWidget(QLabel("<b>Fitting Parameters</b>"), 0, 0, 1, 2, Qt.AlignCenter)

        # Column headers
        fit_layout.addWidget(QLabel("Lower Bound"), 1, 1)
        fit_layout.addWidget(QLabel("Upper Bound"), 1, 2)

        self.param_labels = ["Amplitude (a.u.)", "Resonant Frequency (kHz)", "Quality Factor", "Phase (rad.)"]
        self.lower_bound_fields = []
        self.upper_bound_fields = []

        default_lower = [1E-6, self.freq_vec.min()*0.9, 50, -2*np.pi]
        default_upper = [1E-3, self.freq_vec.max()*1.1, 500, 2*np.pi]

        for i, label in enumerate(self.param_labels):
            fit_layout.addWidget(QLabel(label), i + 2, 0)

            if label == "Amplitude (a.u.)":
                # Lower bound editable box
                lower_edit = QLineEdit("{:.6f}".format(default_lower[i]))
                self.lower_bound_fields.append(lower_edit)
                fit_layout.addWidget(lower_edit, i + 2, 1)
    
                # Upper bound editable box
                upper_edit = QLineEdit("{:.6f}".format(default_upper[i]))
                self.upper_bound_fields.append(upper_edit)
                fit_layout.addWidget(upper_edit, i + 2, 2)
            else:
                # Lower bound editable box
                lower_edit = QLineEdit("{:.2f}".format(default_lower[i]))
                self.lower_bound_fields.append(lower_edit)
                fit_layout.addWidget(lower_edit, i + 2, 1)
    
                # Upper bound editable box
                upper_edit = QLineEdit("{:.2f}".format(default_upper[i]))
                self.upper_bound_fields.append(upper_edit)
                fit_layout.addWidget(upper_edit, i + 2, 2)

        fit_frame.setLayout(fit_layout)
        controls_layout.addWidget(fit_frame)

        # ---- Clustering Parameters Frame ----
        cluster_frame = QFrame()
        cluster_frame.setFrameShape(QFrame.StyledPanel)
        cluster_layout = QGridLayout()

        cluster_layout.addWidget(QLabel("<b>Clustering Parameters</b>"), 0, 0, 1, 2, Qt.AlignCenter)

        # Number of clusters
        cluster_layout.addWidget(QLabel("Number of Clusters"), 1, 0)
        self.num_clusters_edit = QLineEdit("2")
        cluster_layout.addWidget(self.num_clusters_edit, 1, 1)

        # Number of workers
        cluster_layout.addWidget(QLabel("Number of Workers"), 2, 0)
        self.num_workers_edit = QLineEdit("4")
        cluster_layout.addWidget(self.num_workers_edit, 2, 1)

        # K-means Guess dropdown
        cluster_layout.addWidget(QLabel("K-means Guess"), 3, 0)
        self.kmeans_dropdown = QComboBox()
        self.kmeans_dropdown.addItems(["Yes", "No"])
        cluster_layout.addWidget(self.kmeans_dropdown, 3, 1)

        # Buttons
        self.do_guess_button = QPushButton("Do Guess")
        self.do_fit_button = QPushButton("Do Fit")
        self.do_guess_button.clicked.connect(self.on_do_guess)
        self.do_fit_button.clicked.connect(self.on_do_fit)

        cluster_layout.addWidget(self.do_guess_button, 4, 0)
        cluster_layout.addWidget(self.do_fit_button, 4, 1)

        cluster_frame.setLayout(cluster_layout)
        controls_layout.addWidget(cluster_frame)

        # Stretch at the bottom to align top
        controls_layout.addStretch()
        self.do_fit_button.setEnabled(False)

    # ===== Right side: visualization + output =====
        self.right_layout = QVBoxLayout()
        
        # Output box for logs
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("Output logs will appear here...")

        self.right_layout.addWidget(self.output_box, 1)

        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(self.right_layout, 2)

        tab.setLayout(main_layout)
        return tab        

    def get_fitting_bounds(self):
        """Get the numeric fitting bounds, converting frequency back to Hz."""
        lower = []
        upper = []
        for i, param in enumerate(self.param_labels):
            lb = float(self.lower_bound_fields[i].text())
            ub = float(self.upper_bound_fields[i].text())

            if param == "Resonant Frequency":
                # Convert back from kHz to Hz
                lb *= 1e3
                ub *= 1e3
            
            lower.append(lb)
            upper.append(ub)
        return lower, upper

    def on_do_guess(self):
        """Called when 'Do Guess' button is clicked. Will setup the SHO Fit calculation and perform the guess"""
        # Retrieve user-entered values
        self.do_fit_button.setEnabled(False)
        num_clusters = int(self.num_clusters_edit.text())
        num_workers = int(self.num_workers_edit.text())
        kmeans_guess = self.kmeans_dropdown.currentText() == "Yes"
        self.fitter = sidpy.proc.fitter.SidFitter(
            self.beps_raw, SHO_fit_flattened, num_workers=num_workers,
            guess_fn = SHOestimateGuess, ind_dims=self.ind_dims, threads=1, 
            return_cov=False, return_fit=True, return_std=False,
            km_guess=kmeans_guess, num_fit_parms = 4, n_clus = num_clusters
        )
        self.fitter.do_guess()
        self.do_fit_button.setEnabled(True)

    def on_do_fit(self):
        self.lower_bounds, self.upper_bounds = self.get_fitting_bounds()
        self.fitter_output = self.fitter.do_fit(bounds=(self.lower_bounds, self.upper_bounds))
        
        self.fit_params = np.array(self.fitter_output[0].compute())
        self.fit_curves = np.array(self.fitter_output[1].compute())
        self.fitter_output[0].data_type = 'spectral_image'

        self.freq_vec = np.array(self.freq_vec)
        self.vdc_vec = self.fitter_output[0]._axes[2].values                
        self.show_visualizer(self.fit_params, self.freq_vec, self.vdc_vec)
        return
    
    # After the SHO fit is complete, create and add the visualizer widget
    def show_visualizer(self, fit_data, freq_vec, dc_vec):
        """
        Embed the visualizer directly into the right side of the Fit and View SHO tab.
        """
        from sho_visualizer_widget import SHOVisualizerWidget            
        if hasattr(self, "sho_visualizer_widget"):
            # Remove old visualizer if it exists
            self.right_layout.removeWidget(self.sho_visualizer_widget)
            self.sho_visualizer_widget.deleteLater()

        # Create new visualizer
        self.sho_visualizer_widget = SHOVisualizerWidget(fit_data, freq_vec, dc_vec)
        self.right_layout.addWidget(self.sho_visualizer_widget)

    # ==============================
    # Tab 3: Fit Loops
    # ==============================
    def create_fit_loops_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
    
        # ===== LEFT SIDE: Controls =====
        controls_layout = QVBoxLayout()
    
        # ---- Bounds Frame ----
        bounds_frame = QFrame()
        bounds_frame.setFrameShape(QFrame.StyledPanel)
        bounds_layout = QGridLayout()
        bounds_layout.addWidget(QLabel("<b>Loop Fit Bounds</b>"), 0, 0, 1, 3, Qt.AlignCenter)
        bounds_layout.addWidget(QLabel("Lower Bound"), 1, 1)
        bounds_layout.addWidget(QLabel("Upper Bound"), 1, 2)
    
        # parameter labels for p0[0] to p0[8]
        self.loop_param_labels = [f"p0{i}" for i in range(9)]
        self.loop_lower_fields, self.loop_upper_fields = [], []
    
        self.loop_default_lower = [-1, -1, -30, -30, -1, -30, -30, -30, -30]
        self.loop_default_upper = [1, 1, 30, 30, 1, 30, 30, 30, 30]
    
        for i, label in enumerate(self.loop_param_labels):
            bounds_layout.addWidget(QLabel(label), i + 2, 0)
            lower_edit = QLineEdit(str(self.loop_default_lower[i]))
            upper_edit = QLineEdit(str(self.loop_default_upper[i]))
            self.loop_lower_fields.append(lower_edit)
            self.loop_upper_fields.append(upper_edit)
            bounds_layout.addWidget(lower_edit, i + 2, 1)
            bounds_layout.addWidget(upper_edit, i + 2, 2)
    
        bounds_frame.setLayout(bounds_layout)
        controls_layout.addWidget(bounds_frame)
    
        # ---- Other parameters ----
        other_frame = QFrame()
        other_frame.setFrameShape(QFrame.StyledPanel)
        other_layout = QGridLayout()
        other_layout.addWidget(QLabel("<b>Loop Fitting Parameters</b>"), 0, 0, 1, 2, Qt.AlignCenter)
    
        # number of clusters
        other_layout.addWidget(QLabel("Number of Clusters"), 1, 0)
        self.loop_num_clusters = QLineEdit("6")
        other_layout.addWidget(self.loop_num_clusters, 1, 1)
    
        # number of workers
        other_layout.addWidget(QLabel("Number of Workers"), 2, 0)
        self.loop_num_workers = QLineEdit("6")
        other_layout.addWidget(self.loop_num_workers, 2, 1)
    
        # KMeans guess
        other_layout.addWidget(QLabel("KMeans Guess"), 3, 0)
        self.loop_kmeans_dropdown = QComboBox()
        self.loop_kmeans_dropdown.addItems(["Yes", "No"])
        other_layout.addWidget(self.loop_kmeans_dropdown, 3, 1)
    
        # Guess function type
        other_layout.addWidget(QLabel("p0 Guess Function"), 4, 0)
        self.loop_guess_dropdown = QComboBox()
        self.loop_guess_dropdown.addItems(["Basic", "Shallow", "Deep"])
        other_layout.addWidget(self.loop_guess_dropdown, 4, 1)
    
        # Buttons
        self.do_loop_guess_button = QPushButton("Do Loop Guess")
        self.do_loop_fit_button = QPushButton("Do Loop Fit")
        self.do_loop_guess_button.clicked.connect(self.on_do_loop_guess)
        self.do_loop_fit_button.clicked.connect(self.on_do_loop_fit)
        self.do_loop_fit_button.setEnabled(False)
    
        other_layout.addWidget(self.do_loop_guess_button, 5, 0)
        other_layout.addWidget(self.do_loop_fit_button, 5, 1)
    
        other_frame.setLayout(other_layout)
        controls_layout.addWidget(other_frame)
        controls_layout.addStretch()
    
        # Right side: vertical stack — logs (top) + visualization (bottom)
        right_layout = QVBoxLayout()
        
        # Output box (logs) on top
        self.loop_output_box = QTextEdit()
        self.loop_output_box.setReadOnly(True)
        self.loop_output_box.setPlaceholderText("Loop fitting output will appear here...")
        right_layout.addWidget(self.loop_output_box, 2)
        
        # Visualization container (bottom)
        self.loop_vis_container = QVBoxLayout()
        self.loop_vis_widget = None  # placeholder for LoopVisualizerWidget
        right_layout.addLayout(self.loop_vis_container, 3)
        
        # Add both sides
        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(right_layout, 2)
        tab.setLayout(main_layout)
        return tab                
    
    def on_do_loop_guess(self):
        try:
            if not hasattr(self, "fitter_output"):
                self.loop_output_box.append("Error: Run SHO fitting first!")
                return

            nx, ny, nfreq, ndc, nfield, ncycle = self.beps_raw.shape
            loops = np.zeros((nx, ny, nfield, ncycle, ndc))   # storage for projected loops
            for i in range(nx):       # row
                for j in range(ny):   # col
                    for f in range(nfield):  # field
                        for c in range(ncycle):  # cycle
                            vdc_vec = self.beps_raw._axes[3].values                
                            amp_vec = self.fit_params[i, j, :, f, c, 0]
                            phase_vec = self.fit_params[i, j, :, f, c, 3]
                            ##
                            vdc_vec = np.roll(vdc_vec, int(1* len(vdc_vec)/4))
                            amp_vec = np.roll(amp_vec, int(1* len(amp_vec)/4))
                            phase_vec = np.roll(phase_vec, int(1* len(phase_vec)/4))
                            ##
                            loop_result = projectLoop(vdc_vec, amp_vec, phase_vec)
                            loops[i, j, f, c, :] = loop_result['Projected Loop']
            self.loop_output_box.append(f'projected loop shape: {loops.shape}')
        
            loop_dset = sidpy.sid.Dataset.from_array(loops, title="Projected Loops")
            loop_dset.data_type = "image_stack"
            loop_dset.set_dimension(0, sidpy.sid.Dimension(np.arange(nx), name="row", units="px"))
            loop_dset.set_dimension(1, sidpy.sid.Dimension(np.arange(ny), name="col", units="px"))
            loop_dset.set_dimension(2, sidpy.sid.Dimension(np.arange(nfield), name="field", units="idx"))
            loop_dset.set_dimension(3, sidpy.sid.Dimension(np.arange(ncycle), name="cycle", units="idx"))
            loop_dset.set_dimension(4, sidpy.sid.Dimension(vdc_vec, name="Vdc", units="V", quantity="bias", dimension_type="spectral"))
            self.loop_dset = loop_dset
            self.vdc_vec = vdc_vec
            self.loop_output_box.append("Loop dataset successfully created.")

            ##
            d_guess_fn = {
                'Basic': generate_guess,
                'Shallow': generate_shallow_guess,
                'Deep': generate_deep_guess,
                'DeepGP': generate_deepGP_guess
            }
            ##
            self.do_loop_fit_button.setEnabled(False)
            loop_num_clusters = int(self.loop_num_clusters.text())
            loop_num_workers = int(self.loop_num_workers.text())
            loop_kmeans_guess = bool(self.loop_kmeans_dropdown.currentText() == "Yes")
            loop_guess_fn = str(self.loop_guess_dropdown.currentText())
            loop_ind_dims = (0, 1, 2, 3)
            
            self.loop_fitter = sidpy.proc.fitter.SidFitter(
                self.loop_dset, loop_fit_function, xvec=self.vdc_vec, num_workers=loop_num_workers,
                guess_fn = d_guess_fn[loop_guess_fn], ind_dims=loop_ind_dims, threads=1, 
                return_cov=False, return_fit=True, return_std=False,
                km_guess=loop_kmeans_guess, num_fit_parms = 9, n_clus = loop_num_clusters
            )
            self.loop_fitter.do_guess()
            prior = self.loop_fitter.prior  # shape (n_guess, 9)
            lower_bounds1 = np.min(prior, axis=0) - 1e-5
            upper_bounds1 = np.max(prior, axis=0) + 1e-5
            lower_bounds2 = np.array(self.loop_default_lower, dtype=float)
            upper_bounds2 = np.array(self.loop_default_upper, dtype=float)
            lower_bounds = np.minimum(lower_bounds1, lower_bounds2)
            upper_bounds = np.maximum(upper_bounds1, upper_bounds2)
            for i in range(len(self.loop_lower_fields)):
                self.loop_lower_fields[i].setText(f"{lower_bounds[i]:.6g}")
                self.loop_upper_fields[i].setText(f"{upper_bounds[i]:.6g}")
            self.loop_output_box.append("Bounds updated from prior distribution.")
            self.loop_output_box.append(f"Guess shape: {self.loop_fitter.prior.shape}")
            self.do_loop_fit_button.setEnabled(True)
    
        except Exception as e:
            self.loop_output_box.append(f"Error in on_do_loop_guess: {e}")

    def get_loop_fitting_bounds(self):
        """Get the numeric fitting bounds"""
        lower = []
        upper = []
        for i, param in enumerate(self.loop_param_labels):
            lb = float(self.loop_lower_fields[i].text())
            ub = float(self.loop_upper_fields[i].text())            
            lower.append(lb)
            upper.append(ub)
        return lower, upper

    def show_loop_visualizer(self):
        """
        Embed the Loop Visualizer (map + loop plot) into the Loop Fit tab,
        similar in structure to the SHO visualizer.
        """
        from loop_visualizer_widget import LoopVisualizerWidget
        try:
            nx, ny, nfield, ncycle, _ = self.loop_fit_curves.shape
    
            # --- Remove previous visualizer if it exists ---
            if hasattr(self, "loop_vis_widget") and self.loop_vis_widget:
                self.loop_vis_container.removeWidget(self.loop_vis_widget)
                self.loop_vis_widget.deleteLater()
                self.loop_vis_widget = None
    
            # --- Create and add the new visualizer ---
            self.loop_vis_widget = LoopVisualizerWidget(
                switching_coef=self.loop_fit_switching_coef,
                nfield=nfield,
                ncycle=ncycle,
                loop_fit_curves=self.loop_fit_curves,
                loop_dset=self.loop_dset,
            )
            self.loop_vis_container.addWidget(self.loop_vis_widget)
    
            self.loop_output_box.append("✅ Loop Map visualization loaded below.")
        except Exception as e:
            self.loop_output_box.append(f"❌ Error in show_loop_visualizer: {e}")

    
    def on_do_loop_fit(self):
        try:
            self.loop_lower_bounds, self.loop_upper_bounds = self.get_loop_fitting_bounds()
            self.loop_fitter_output = self.loop_fitter.do_fit(bounds=(self.loop_lower_bounds, self.loop_upper_bounds))
            self.loop_fit_params = np.array(self.loop_fitter_output[0].compute())
            self.loop_fit_curves = np.array(self.loop_fitter_output[1].compute())
            switch_coef_struct = calc_switching_coef_vec(self.loop_fit_params.reshape(-1, 9))[:, 0]
            self.loop_fit_switching_coef = np.vstack([list(x) for x in switch_coef_struct]).reshape(self.loop_fit_params.shape)           
            self.show_loop_visualizer()            
        except Exception as e:
            self.loop_output_box.append(f"Error in on_do_loop_fit: {e}")            
        return

    
# ==============================
# Main application entry point
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SidpyBandExcitationProcessor()
    window.show()
    sys.exit(app.exec_())