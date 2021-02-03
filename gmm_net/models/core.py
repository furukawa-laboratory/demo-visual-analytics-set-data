import numpy as np
class Space():
    def __init__(self,
                 data: np.ndarray,
                 grid_points=None,
                 grid_mapping=None,
                 graph_whole=None,
                 graph_indv=None,
                 dropdown=None,
                 label_data=None,
                 label_feature=None,
                 is_middle_color_zero=None):
        self.data = data.copy()
        self.n_data = data.shape[0]
        self.n_dim = data.shape[1]
        self.grid_points = grid_points
        self.grid_mapping = grid_mapping
        self.graph_whole = graph_whole
        self.graph_indiv = graph_indv
        self.dropdown = dropdown
        self.label_data = label_data
        self.label_feature = label_feature
        self.is_middle_color_zero = is_middle_color_zero


    # def _set_grid(self, grid_points, n_grid_points):
    #     self.grid_points = grid_points

class LatentSpace(Space):
    pass

class ObservedSpace(Space):
    pass

class FeatureSpace(Space):
    pass
