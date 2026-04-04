class SHOVisualizerCore:
    """
    Backend for BEPS visualizer logic.
    Pure computation and data slicing; no UI code.
    """

    def __init__(self, fit_data, freq_vec, dc_vec):
        # shape: [X, Y, DC, Field, Cycle, freq]
        self.fit_data = fit_data
        self.freq_vec = freq_vec  # 1D frequency vector
        self.dc_vec = dc_vec
        self.selected_x = 0
        self.selected_y = 0
        self.selected_field = 0
        self.selected_cycle = 0
        self.selected_dc_index = 0
        self.selected_fit_param = 0

    def set_point(self, x_idx, y_idx):
        self.selected_x = x_idx
        self.selected_y = y_idx

    def set_field(self, field_idx):
        self.selected_field = field_idx

    def set_cycle(self, cycle_idx):
        self.selected_cycle = cycle_idx

    def set_dc_index(self, dc_idx):
        self.selected_dc_index = dc_idx

    def set_fit_param(self, param_idx):
        self.selected_fit_param = param_idx

    def get_map_slice(self):
        """
        Returns a 2D slice of the data at the current DC index, field, and cycle.
        """
        return self.fit_data[
            :,
            :,
            self.selected_dc_index,
            self.selected_field,
            self.selected_cycle,
            self.selected_fit_param,
        ]

    def get_spectrum_at_point(self, row=0, col=0):
        """
        Returns the spectrum at the selected (X, Y) point for all DC indices.
        """
        return self.fit_data[
            row, col, :, self.selected_field, self.selected_cycle, self.selected_fit_param
        ]
