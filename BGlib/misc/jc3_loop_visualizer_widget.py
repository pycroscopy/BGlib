import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LoopVisualizerWidget(QWidget):
    """
    Two-panel visualization for loop fitting:
    1. Top: nx×ny map of selected parameter for given field & cycle
    2. Bottom: raw vs fitted loop for selected (x, y, field, cycle)
    """
    def __init__(self, switching_coef, nfield, ncycle, loop_fit_curves=None, loop_dset=None, parent=None):
        super().__init__(parent)

        self.switching_coef = switching_coef
        self.nfield = nfield
        self.ncycle = ncycle
        self.loop_fit_curves = loop_fit_curves
        self.loop_dset = loop_dset

        self.param_names = [
            "V-", "V+", "Imprint", "R+", "R-",
            "Switchable Polarization", "Work of Switching",
            "Nucleation Bias 1", "Nucleation Bias 2"
        ]

        # === Main layout ===
        layout = QVBoxLayout(self)

        # --- Controls ---
        controls = QGridLayout()
        controls.addWidget(QLabel("Parameter:"), 0, 0)
        self.param_dropdown = QComboBox()
        self.param_dropdown.addItems(self.param_names)
        controls.addWidget(self.param_dropdown, 0, 1)

        controls.addWidget(QLabel("Field:"), 0, 2)
        self.field_dropdown = QComboBox()
        self.field_dropdown.addItems([str(i) for i in range(nfield)])
        controls.addWidget(self.field_dropdown, 0, 3)

        controls.addWidget(QLabel("Cycle:"), 0, 4)
        self.cycle_dropdown = QComboBox()
        self.cycle_dropdown.addItems([str(i) for i in range(ncycle)])
        controls.addWidget(self.cycle_dropdown, 0, 5)

        controls.addWidget(QLabel("X index:"), 1, 0)
        self.x_dropdown = QComboBox()
        controls.addWidget(self.x_dropdown, 1, 1)

        controls.addWidget(QLabel("Y index:"), 1, 2)
        self.y_dropdown = QComboBox()
        controls.addWidget(self.y_dropdown, 1, 3)

        layout.addLayout(controls)

        # --- Figure and subplots ---
        self.figure = Figure(figsize=(7, 6))
        self.ax_map = self.figure.add_subplot(211)
        self.ax_loop = self.figure.add_subplot(212)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # === Setup dropdowns ===
        nx, ny, _, _, _ = self.switching_coef.shape
        self.x_dropdown.addItems([str(i) for i in range(nx)])
        self.y_dropdown.addItems([str(i) for i in range(ny)])

        # === Connect events ===
        self.param_dropdown.currentIndexChanged.connect(self.update_map)
        self.field_dropdown.currentIndexChanged.connect(self.update_map)
        self.cycle_dropdown.currentIndexChanged.connect(self.update_map)
        self.x_dropdown.currentIndexChanged.connect(self.update_loop)
        self.y_dropdown.currentIndexChanged.connect(self.update_loop)

        # === Initial plots ===
        self.update_map()
        self.update_loop()

    # ---------------------------------------------------------
    # Top visualization: parameter map
    # ---------------------------------------------------------
    def update_map(self):
        try:
            param_idx = self.param_dropdown.currentIndex()
            field_idx = int(self.field_dropdown.currentText())
            cycle_idx = int(self.cycle_dropdown.currentText())
    
            data = self.switching_coef[:, :, field_idx, cycle_idx, param_idx]
    
            self.ax_map.clear()
            im = self.ax_map.imshow(data, cmap="viridis", origin="lower", aspect="auto")
            self.ax_map.set_title(f"{self.param_names[param_idx]}  (Field {field_idx}, Cycle {cycle_idx})")
            self.ax_map.set_xlabel("Y index")
            self.ax_map.set_ylabel("X index")
    
            # --- Safe colorbar handling ---
            try:
                if hasattr(self, "_colorbar") and self._colorbar:
                    # Safely clear old colorbar axes
                    self._colorbar.ax.clear()
                    self._colorbar.ax.remove()
            except Exception:
                pass
    
            # Create a new colorbar in its own axis (avoids subplotspec issues)
            cax = self.figure.add_axes([0.92, 0.55, 0.02, 0.35])  # [left, bottom, width, height]
            self._colorbar = self.figure.colorbar(im, cax=cax)
            self._colorbar.ax.tick_params(labelsize=8)
    
            self.canvas.draw_idle()
    
        except Exception as e:
            self.ax_map.clear()
            self.ax_map.text(0.5, 0.5, f"Map error:\n{e}", ha="center", va="center")
            self.canvas.draw_idle()

    # ---------------------------------------------------------
    # Bottom visualization: raw vs fitted loop
    # ---------------------------------------------------------
    def update_loop(self):
        try:
            x = int(self.x_dropdown.currentText())
            y = int(self.y_dropdown.currentText())
            field_idx = int(self.field_dropdown.currentText())
            cycle_idx = int(self.cycle_dropdown.currentText())

            vdc_vec = self.loop_dset._axes[4].values
            raw_loop = np.array(self.loop_dset[x, y, field_idx, cycle_idx, :])
            fit_loop = self.loop_fit_curves[x, y, field_idx, cycle_idx, :]

            self.ax_loop.clear()
            self.ax_loop.plot(vdc_vec, raw_loop, "o-", label="Measured")
            self.ax_loop.plot(vdc_vec, fit_loop, "-", label="Fitted")
            self.ax_loop.legend()
            self.ax_loop.set_title(f"Loop at (x={x}, y={y}), Field {field_idx}, Cycle {cycle_idx}")
            self.ax_loop.set_xlabel("DC Bias (V)")
            self.ax_loop.set_ylabel("Response (a.u.)")
            self.canvas.draw_idle()
        except Exception as e:
            self.ax_loop.clear()
            self.ax_loop.text(0.5, 0.5, f"Loop error:\n{e}", ha="center", va="center")
            self.canvas.draw_idle()
