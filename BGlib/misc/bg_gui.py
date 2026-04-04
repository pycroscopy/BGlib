import os
# 1) Force Qt-backed Matplotlib (avoid MacOSX backend which must run on main thread)
os.environ.setdefault("MPLBACKEND", "QtAgg")   # or "Qt5Agg"

# 2) Tame inner BLAS/OpenMP thread storms (critical on macOS + Accelerate/OpenBLAS/MKL)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# (optional but helpful) make Qt explicit
os.environ.setdefault("QT_API", "pyqt5")
import sys  # noqa: E402
import faulthandler  # noqa: E402
faulthandler.enable()

from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTabWidget, QFileDialog, QPushButton, QTextEdit, QFrame,
    QGridLayout, QLineEdit, QComboBox
)
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal  # noqa: E402
import BGlib.be as belib  # noqa: E402
import numpy as np  # noqa: E402
import sidpy  # noqa: E402
import SciFiReaders as sr  # noqa: E402
from BGlib.be.analysis.utils.sidpy_sho_fitter import SHOestimateGuess, SHO_fit_flattened  # noqa: E402
from PyQt5.QtGui import QTextCursor  # noqa: E402

# pip install dask distributed
import time  # noqa: E402
from typing import Optional, Sequence, Tuple  # noqa: E402
from dask.distributed import Client  # noqa: E402

# Helper to normalize log records across dask versions
def _format_sched_record(rec) -> str:
    try:
        level, msg, ts = rec[0], rec[1], rec[2]
        return f"[SCHEDULER] {ts} {level}: {msg}"
    except Exception:
        return f"[SCHEDULER] {rec}"

def _format_worker_record(worker: str, item) -> str:
    # item might be (ts, msg), (ts, level, msg), or (level, msg)
    try:
        if len(item) == 3 and isinstance(item[0], (int, float, str)):
            ts, level, msg = item
            return f"[{worker}] {ts} {level}: {msg}"
        elif len(item) == 2:
            a, b = item
            # Heuristic: timestamp first
            if isinstance(a, (int, float, str)) and not isinstance(b, (int, float)):
                return f"[{worker}] {a}: {b}"
            else:
                return f"[{worker}] {a} {b}"
    except Exception:
        pass
    return f"[{worker}] {item}"

class DaskLogWorker(QObject):
    # Emits batches of log lines to append
    log_lines = pyqtSignal(list)
    error = pyqtSignal(str)
    connected = pyqtSignal(str)   # emits dashboard link when ready

    def __init__(self, scheduler_address: str, poll_interval_ms: int = 1000,
                 worker_tail: int = 2000, parent=None):
        super().__init__(parent)
        self.address = scheduler_address
        self.poll_interval_ms = max(200, poll_interval_ms)
        self.worker_tail = worker_tail
        self._running = False
        self._client: Optional[Client] = None
        self._seen_counts: dict[str, int] = {}

    def start(self):
        """Entry point wired to QThread.started"""
        try:
            # Create the Client *in this thread* so its IO loop lives here.
            self._client = Client(self.address, set_as_default=False, asynchronous=False)
            # Let the UI show a link if desired (useful during dev)
            if getattr(self._client, "dashboard_link", None):
                self.connected.emit(self._client.dashboard_link)
        except Exception as e:
            self.error.emit(f"Failed to connect Client: {e!r}")
            return

        self._running = True
        # First fetch primes scheduler logs; worker logs can be large so use tails
        while self._running:
            try:
                lines: list[str] = []

                # --- Scheduler logs ---
                try:
                    sched_logs: Sequence[Tuple] = self._client.get_scheduler_logs()  # type: ignore
                    for rec in sched_logs:
                        lines.append(_format_sched_record(rec))
                except Exception as e:
                    # Older/newer versions may differ; keep going
                    self.error.emit(f"Scheduler logs error: {e!r}")

                # --- Worker logs (tail only) ---
                try:
                    # n=tail only supported on newer versions; fallback handled below
                    worker_logs = self._client.get_worker_logs(n=self.worker_tail)  # type: ignore
                except TypeError:
                    worker_logs = self._client.get_worker_logs()  # type: ignore

                # worker_logs is {worker_name: list_of_records}
                for worker, records in worker_logs.items():
                    seen = self._seen_counts.get(worker, 0)
                    # If we requested tail, the list is already short; still show only new items
                    new = records[seen:] if seen < len(records) else []
                    if new:
                        for item in new:
                            lines.append(_format_worker_record(worker, item))
                        self._seen_counts[worker] = seen + len(new)

                if lines:
                    self.log_lines.emit(lines)

            except Exception as e:
                self.error.emit(f"Polling error: {e!r}")

            # Sleep without blocking the GUI thread
            time.sleep(self.poll_interval_ms / 1000.0)

        # Cleanup
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass

    def stop(self):
        self._running = False

class DaskLogsPanel(QTextEdit):
    def __init__(self, scheduler_address: str, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._thread = QThread(self)      # owns the worker thread
        self._worker = DaskLogWorker(scheduler_address)

        # Wire signals
        self._worker.log_lines.connect(self._append_lines)
        self._worker.error.connect(self._append_error)
        self._worker.connected.connect(self._on_connected)

        # Move worker to thread and start
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start)
        self._thread.start()

    def _append_lines(self, lines: list[str]):
        # Append efficiently
        self.append("\n".join(lines))

    def _append_error(self, msg: str):
        self.append(f"<span style='color:#b00;'>[ERROR] {msg}</span>")

    def _on_connected(self, dash_link: str):
        self.append(f"<i>Connected. Dashboard: {dash_link}</i>")

    def closeEvent(self, event):
        try:
            self._worker.stop()
            self._thread.quit()
            self._thread.wait(2000)
        finally:
            super().closeEvent(event)

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
                self.beps_raw = reader.read()[:30,:30,:,:,:]
                self.freq_axis = self.beps_raw.labels.index('Frequency (Hz)')
                self.freq_vec = self.beps_raw._axes[self.freq_axis].values
                self.all_dims = np.arange(len(self.beps_raw.shape))
                self.ind_dims = np.delete(self.all_dims, self.freq_axis)
                #TODO: Only for BE datasets. Need to check that the vector is correct
                self.lower_bound_fields[1].setText(f"{self.freq_vec.min() / 1e3:.2f}")
                self.upper_bound_fields[1].setText(f"{self.freq_vec.max() / 1e3:.2f}")
                
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

        default_lower = [1E-6, self.freq_vec.min(), 50, -2*np.pi]
        default_upper = [1E-3, self.freq_vec.max(), 500, 2*np.pi]

        for i, label in enumerate(self.param_labels):
            fit_layout.addWidget(QLabel(label), i + 2, 0)

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
        main_layout = main_layout

        tab.setLayout(main_layout)
        return tab

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
        layout = QVBoxLayout()

        label = QLabel("This tab will allow you to fit loops to your data.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Example placeholder
        layout.addWidget(QLabel("Placeholder for loop fitting controls and visualization."))

        tab.setLayout(layout)
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
        self.lower_bounds, self.upper_bounds = self.get_fitting_bounds()
        
        self.fitter = sidpy.proc.fitter.SidFitter(self.beps_raw, SHO_fit_flattened, num_workers=num_workers,
                                     guess_fn = SHOestimateGuess,ind_dims=self.ind_dims,
                           threads=1, return_cov=False, return_fit=True, return_std=False,
                           km_guess=kmeans_guess,num_fit_parms = 4, n_clus = num_clusters)
        address = self.fitter.client.dashboard_link
        #start outputting the logs of the Dask client here
        self._output_dask_client_logs(address)
        self.fitter.do_guess()
        self.do_fit_button.setEnabled(True)

    def _output_dask_client_logs(self, address):
        """
        Start outputting the code for the stuff here
        """
        import webbrowser
        # This will open the default system browser at that address:
        webbrowser.open(address)
        return

    def append_output_text(self, text):
        """Append new text to the output box and scroll to the end."""
        self.output_box.moveCursor(QTextCursor.End)  # Move to the end before inserting
        self.output_box.insertPlainText(text)
        self.output_box.moveCursor(QTextCursor.End)  # Ensure the view scrolls to bottom
        return
    
    def on_do_fit(self):
        self.lower_bounds, self.upper_bounds = self.get_fitting_bounds()
        self.fitter_output = self.fitter.do_fit()
        self.fitter_output[0].data_type = 'spectral_image'
        dc_vec = self.fitter_output[0]._axes[2].values
        fit_params = np.array(self.fitter_output[0].compute())
        self.show_visualizer(fit_params, np.array(self.freq_vec), dc_vec)
        return
# ==============================
# Main application entry point
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SidpyBandExcitationProcessor()
    window.show()
    sys.exit(app.exec_())
