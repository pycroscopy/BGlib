# visualizer_widget.py
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QGridLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from BGlib.misc.sho_visualizer_core import SHOVisualizerCore

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
    def __init__(self, raw_data, fit_data, freq_vec, parent=None):
        super().__init__(parent)

        # Backend logic
        self.core = SHOVisualizerCore(raw_data, fit_data, freq_vec)

        # Main layout
        main_layout = QHBoxLayout()

        # ========== LEFT SIDE: CONTROLS ==========
        control_layout = QGridLayout()

        # DC Offset Slider
        self.dc_slider = QSlider(Qt.Horizontal)
        self.dc_slider.setMinimum(0)
        self.dc_slider.setMaximum(raw_data.shape[2] - 1)  # DC dimension size
        self.dc_slider.setValue(0)
        self.dc_slider.valueChanged.connect(self.update_dc_index)
        control_layout.addWidget(QLabel("DC Offset"), 0, 0)
        control_layout.addWidget(self.dc_slider, 0, 1)

        # Field Dropdown
        self.field_dropdown = QComboBox()
        self.field_dropdown.addItems([f"Field {i}" for i in range(raw_data.shape[3])])
        self.field_dropdown.currentIndexChanged.connect(self.update_field)
        control_layout.addWidget(QLabel("Field"), 1, 0)
        control_layout.addWidget(self.field_dropdown, 1, 1)

        # Cycle Dropdown
        self.cycle_dropdown = QComboBox()
        self.cycle_dropdown.addItems([f"Cycle {i}" for i in range(raw_data.shape[4])])
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
        plot_layout.addWidget(QLabel("Spectrum at Selected Point"))
        plot_layout.addWidget(self.spectrum_canvas)

        # Combine into main layout
        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(plot_layout, 3)
        self.setLayout(main_layout)

        # Initial plot
        self.update_visualization()

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
        self.spectrum_canvas.axes.plot(np.arange(len(spectrum)), spectrum, 'b-')
        self.spectrum_canvas.axes.set_title("Response vs DC Offset")
        self.spectrum_canvas.draw()
