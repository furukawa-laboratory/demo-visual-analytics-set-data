import numpy as np
from scipy.spatial.distance import cdist


class KDE():
    def __init__(self):
        pass

    def fit(self, dataset, bandwidth, weights=None):
        self.dataset = dataset.copy()
        self.n_samples, self.n_features = dataset.shape
        self.bandwidth = bandwidth
        if weights is not None:
            if weights.ndim == 1:
                if weights.shape[0] != self.n_samples:
                    raise ValueError("weights length must be same dataset.shape[0]")
                self.weights = weights[None, :]
            elif weights.ndim == 2:
                if weights.shape[1] != self.n_samples:
                    raise ValueError("weights.shape[1] must be same dataset.shape[1]")
                self.weights = weights
            else:
                raise ValueError("weights must be 1d or 2d array")
        else:
            self.weights = np.ones((1, self.n_samples))
        self.weights = self.weights / self.weights.sum(axis=1)[:, None]

    def pdf(self, x):
        sqdist = cdist(x, self.dataset, metric='sqeuclidean')
        gauss_func = np.exp(-sqdist / (self.bandwidth * self.bandwidth * 2.0))  # KxN
        prob = np.einsum("kn,mn->mk", gauss_func, self.weights)
        prob /= np.sqrt((2.0 * np.pi * self.bandwidth * self.bandwidth) ** self.n_features)  # MxK
        return np.squeeze(prob)
