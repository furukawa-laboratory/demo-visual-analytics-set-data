import numpy as np
class Space():
    def __init__(self, data: np.ndarray, grid_points: np.ndarray, grid_mapping=None, graph=None):
        self.data = data.copy()
        self.grids = grid_points.copy()
        self.grid_mapping = grid_mapping
        self.graph = graph

    # def _set_grid(self, grid_points, n_grid_points):
    #     self.grid_points = grid_points

class LatentSpace(Space):
    pass

class FeatureSpace(Space):
    pass
