import numpy as np
from scipy.spatial.distance import cdist


def discrete_kde(x, bandwidth, bias, resolution, return_zeta=True):
    n_dim = x.shape[1]
    precision = 1 / (bandwidth * bandwidth)
    # 各軸におけるxの最小値と最大値
    x_range = np.concatenate([x.min(axis=0)[None, :], x.max(axis=0)[None, :]], axis=0)# (2,ndim)
    # zetaの範囲はxの範囲よりも2*bandwidthだけ広くしておく
    zeta_range = x_range + np.array([-2.0 * bandwidth, 2.0 * bandwidth])[:, None] # (2,ndim)
    if n_dim == 1:
        zeta = np.linspace(zeta_range[0,0],zeta_range[1,0],resolution)
    elif n_dim == 2:
        zeta0 = np.linspace(zeta_range[0,0],zeta_range[1,0],resolution)
        zeta1 = np.linspace(zeta_range[0,1],zeta_range[1,1],resolution)
        zeta = np.meshgrid(zeta0,zeta1,indexing='ij')
        zeta = np.dstack(zeta).reshape(-1,n_dim)
    else:
        raise ValueError('x.shape[1] must be 1 or 2.')

    distance = cdist(zeta,x,"sqeuclidean")

    if return_zeta:
        return prob,zeta
    else:
        return prob

if __name__ == '__main__':
    n_samples = 100
    n_dim = 2
    x = np.random.normal(0.0, 1.0, (n_samples, n_dim))
    if n_dim == 2:
        x *= np.array([1.5,1.0])[None,:]
    bandwidth = 0.5
    resolution = 20
    bias = 0.0
    return_zeta = True
    prob, zeta = discrete_kde(x=x, bandwidth=bandwidth,
                              bias=bias, resolution=resolution,
                              return_zeta=return_zeta)
