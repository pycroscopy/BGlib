# visualizer_widget.py
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QGridLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from BGlib.misc.sho_visualizer_core import SHOVisualizerCore
import matplotlib as mpl

class MplCanvas(FigureCanvas):
    """Matplotlib canvas to embed in PyQt."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class SHOVisualizerWidget(QWidget):
    """
    Widget containing the BEPS visualizer for embedding inside a tab.
    """
    def __init__(self, fit_data, freq_vec,dc_vec, parent=None):
        super().__init__(parent)
        

        # Backend logic
        self.core = SHOVisualizerCore(fit_data, freq_vec, dc_vec)

        # Main layout
        main_layout = QHBoxLayout()

        # ========== LEFT SIDE: CONTROLS ==========
        control_layout = QGridLayout()

        self.dc_vec = dc_vec

        # DC Offset Slider
        self.dc_slider = QSlider(Qt.Horizontal)
        self.dc_slider.setMinimum(0)
        self.dc_slider.setMaximum(fit_data.shape[2] - 1)  # DC dimension size
        self.dc_slider.setValue(0)
        self.dc_slider.valueChanged.connect(self.update_dc_index)
        control_layout.addWidget(QLabel("DC Offset"), 0, 0)
        control_layout.addWidget(self.dc_slider, 0, 1)

        # Field Dropdown
        self.field_dropdown = QComboBox()
        self.field_dropdown.addItems([f"Field {i}" for i in range(fit_data.shape[3])])
        self.field_dropdown.currentIndexChanged.connect(self.update_field)
        control_layout.addWidget(QLabel("Field"), 1, 0)
        control_layout.addWidget(self.field_dropdown, 1, 1)

        # Cycle Dropdown
        self.cycle_dropdown = QComboBox()
        self.cycle_dropdown.addItems([f"Cycle {i}" for i in range(fit_data.shape[4])])
        self.cycle_dropdown.currentIndexChanged.connect(self.update_cycle)
        control_layout.addWidget(QLabel("Cycle"), 2, 0)
        control_layout.addWidget(self.cycle_dropdown, 2, 1)

        # Fit Parameter Dropdown
        self.fit_param_dropdown = QComboBox()
        self.fit_param_dropdown.addItems(['Amplitude', 'Resonant Frequency', 'Quality Factor', 'Phase'])
        self.fit_param_dropdown.currentIndexChanged.connect(self.update_fit_param)
        control_layout.addWidget(QLabel("Fit Parameter"), 3, 0)
        control_layout.addWidget(self.fit_param_dropdown, 3, 1)

        # ========== RIGHT SIDE: PLOTS ==========
        self.map_canvas = MplCanvas(self, width=5, height=4)
        self.spectrum_canvas = MplCanvas(self, width=5, height=3)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(QLabel("2D Map"))
        plot_layout.addWidget(self.map_canvas)
        self._map_click_cid = (
        self.map_canvas.figure.canvas.mpl_connect(
            "button_press_event", self._on_map_click
            )
        )
        plot_layout.addWidget(QLabel("Spectrum at Selected Point"))
        plot_layout.addWidget(self.spectrum_canvas)

        # Combine into main layout
        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(plot_layout, 3)
        self.setLayout(main_layout)
      
        # Initial plot
        self.update_visualization()

    def _on_map_click(self, event):
        """Handle clicks on the 2D map and update the spectrum plot."""
        # Only act on the map axes and valid data coords:
        if event.inaxes is not self.map_canvas.axes or event.xdata is None or event.ydata is None:
            return

        # 1) Convert axes coords to integer row/col indices:
        r, c = self._xy_to_indices(event.xdata, event.ydata)
        if r is None or c is None:
            return

        # 2) Plot spectrum for that pixel:
        self._plot_spectrum_at(r, c)

        # 3) (Optional) mark the selected pixel on the map:
        self._update_click_marker(c, r)

    def _xy_to_indices(self, x, y):
        """
        Convert axes data coords (x,y) to array indices (row, col) for the map image.
        This assumes the map is shown with imshow. We read the first image on the axes.
        """
        if len(self.map_canvas.axes.images) == 0:
            return None, None

        im: mpl.image.AxesImage = self.map_canvas.axes.images[0]
        arr = im.get_array()
        if arr is None:
            return None, None

        # Image extent and orientation used by imshow:
        x0, x1, y0, y1 = im.get_extent()
        ny, nx = arr.shape[:2]
        origin = im.origin  # 'upper' or 'lower'

        # Normalize x,y within extent → fractional coord in [0,1]
        # guard against zero division if degenerate extent:
        if (x1 - x0) == 0 or (y1 - y0) == 0:
            return None, None
        fx = (x - x0) / (x1 - x0)
        fy = (y - y0) / (y1 - y0)

        # Convert to pixel indices
        col = int(np.floor(fx * nx))
        if origin == "upper":
            row = int(np.floor((1.0 - fy) * ny))
        else:  # origin == 'lower'
            row = int(np.floor(fy * ny))

        # Clamp to valid range:
        row = np.clip(row, 0, ny - 1)
        col = np.clip(col, 0, nx - 1)

        return row, col

    def _plot_spectrum_at(self, r, c):
        """
        Pull the spectrum for (row=r, col=c) and draw it on spectrum axes.
        Adapt the 'get' part to your data model (examples below).
        """
        ax = self.spectrum_canvas.axes
        ax.cla()

        # ---- EXAMPLE ways to fetch the spectrum (pick the one that matches your widget) ----
        # Case A: you have a 3D cube of shape (rows, cols, freq)
        # spec = self.spectra[r, c, :]
        #
        # Case B: you have a callable / accessor
        # spec = self.get_spectrum(r, c)
        #
        # Case C: data stored in sidpy Dataset (e.g., with spectral dimension last)
        # spec = self.dataset.isel(y=r, x=c).values  # adjust dim names as needed

        # Replace with your actual retrieval:
        spec = self.spectrum = self.core.get_spectrum_at_point(r, c)  # <-- implement or swap in your accessor

        # Frequency or x-axis:
        # If you already have frequency axis (1D) aligned to spec:
        # freq = self.frequency  # shape (nfreq,)
        # Else fall back to index:
        freq = getattr(self, "frequency", None)
        if freq is None or len(freq) != len(spec):
            freq = np.arange(len(spec))

        ax.plot(self.dc_vec, spec)
        ax.set_title(f"Spectrum at (row={r}, col={c})")

        # If you like, auto-scale or set consistent y-lims:
        ax.relim()
        ax.autoscale_view()

        # Redraw just this canvas:
        self.spectrum_canvas.figure.canvas.draw_idle()

    def _update_click_marker(self, x, y):
        """Show a marker at the clicked position on the map."""
        ax = self.map_canvas.axes
        # Remove the old marker if it exists:
        if hasattr(self, "_click_marker") and self._click_marker in ax.lines:
            self._click_marker.remove()

        # Draw a nice white-outlined circle so it’s visible on most colormaps:
        (self._click_marker,) = ax.plot(
            [x], [y],
            marker="o", mfc="none", mec="w", mew=1.5, ms=10, linestyle="None"
        )
        self.map_canvas.figure.canvas.draw_idle()


    # ========================================
    # Control Updates
    # ========================================
    def update_dc_index(self, value):
        self.core.set_dc_index(value)
        self.update_visualization()

    def update_field(self, index):
        self.core.set_field(index)
        self.update_visualization()

    def update_cycle(self, index):
        self.core.set_cycle(index)
        self.update_visualization()

    def update_fit_param(self, index):
        self.core.set_fit_param(index)
        self.update_visualization()

    # ========================================
    # Visualization Update
    # ========================================
    def update_visualization(self):
        """Update both map and spectrum views."""
        # Update 2D Map
        self.map_canvas.axes.clear()
        map_slice = self.core.get_map_slice()
        im = self.map_canvas.axes.imshow(map_slice.T, origin='lower', cmap='viridis', aspect='auto')
        self.map_canvas.axes.set_title("2D Fit Parameter Map")
        self.map_canvas.draw()

        # Update Spectrum
        
        self.spectrum_canvas.axes.clear()
        spectrum = self.core.get_spectrum_at_point()
        self.spectrum_canvas.axes.plot(self.dc_vec, spectrum, 'b-')
        self.spectrum_canvas.axes.set_title("Response vs DC Offset")
        self.spectrum_canvas.draw()
