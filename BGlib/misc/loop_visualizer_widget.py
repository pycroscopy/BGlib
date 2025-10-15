import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LoopVisualizerWidget(QWidget):
    def __init__(self, switching_coef, nfield, ncycle, parent=None):
        super().__init__(parent)
        self.switching_coef = switching_coef
        self.nfield = nfield
        self.ncycle = ncycle
        self.param_names = [
            "V-", "V+", "Imprint", "R+", "R-",
            "Switchable Polarization", "Work of Switching",
            "Nucleation Bias 1", "Nucleation Bias 2"
        ]

        layout = QVBoxLayout(self)

        # --- Dropdowns ---
        controls = QHBoxLayout()
        self.param_dropdown = QComboBox()
        self.param_dropdown.addItems(self.param_names)
        controls.addWidget(QLabel("Parameter:"))
        controls.addWidget(self.param_dropdown)

        self.field_dropdown = QComboBox()
        self.field_dropdown.addItems([f"{i}" for i in range(nfield)])
        controls.addWidget(QLabel("Field:"))
        controls.addWidget(self.field_dropdown)

        self.cycle_dropdown = QComboBox()
        self.cycle_dropdown.addItems([f"{i}" for i in range(ncycle)])
        controls.addWidget(QLabel("Cycle:"))
        controls.addWidget(self.cycle_dropdown)

        layout.addLayout(controls)

        # --- Matplotlib Figure ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # --- Connect change events ---
        self.param_dropdown.currentIndexChanged.connect(self.update_plot)
        self.field_dropdown.currentIndexChanged.connect(self.update_plot)
        self.cycle_dropdown.currentIndexChanged.connect(self.update_plot)

        self.update_plot()

    def update_plot(self):
        param_idx = self.param_dropdown.currentIndex()
        field_idx = int(self.field_dropdown.currentText())
        cycle_idx = int(self.cycle_dropdown.currentText())
    
        # infer spatial dimensions (assuming square scan)
        nxny = int(len(self.switching_coef) / (self.nfield * self.ncycle))
        nx = ny = int(np.sqrt(nxny))
    
        # reshape to (nx, ny, nfield, ncycle)
        coef_reshaped = np.array(self.switching_coef, dtype=object).reshape(nx, ny, self.nfield, self.ncycle)
    
        # extract the parameter of interest (0–8 index from param_idx)
        data = np.vectorize(lambda x: x[param_idx])(coef_reshaped[:, :, field_idx, cycle_idx])
    
        # plot heatmap
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(data, cmap="viridis", origin="lower")
        ax.set_title(self.param_names[param_idx])
        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.canvas.draw()
