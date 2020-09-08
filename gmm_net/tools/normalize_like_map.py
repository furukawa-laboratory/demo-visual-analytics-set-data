import numpy as np
def normalize_like_map(X, weights, prior_mean, prior_precision, after_standardization=False):
    # exception
    if X.ndim == 1:
        X = X.reshape(-1,1)
    elif X.ndim == 2:
        pass
    else:
        raise ValueError('X must be 1d or 2d array')

    if weights is None:
        weights = np.ones(X.shape[0])
    elif weights.ndim == 1 and weights.shape[0] == X.shape[0]:
        weights = np.squeeze(weights)
    else:
        raise ValueError('weights must be None or 1d array which has same length of X.shape[0]')
    if isinstance(prior_mean,int) or isinstance(prior_mean, float):
        prior_mean = np.ones(X.shape[1]) * prior_mean
    elif isinstance(prior_mean, np.ndarray):
        if prior_mean.shape[0] == X.shape[1] and prior_mean.ndim == 1:
            pass
        else:
            raise ValueError('prior_mean must be scalar or 1d array which has same length of X.shape[1]')
    else:
        raise ValueError('prior_mean must be scalar or 1d array which has same length of X.shape[1]')

    if isinstance(prior_precision,float) or isinstance(prior_precision,int):
        pass
    else:
        raise ValueError('prior_precision must be scalar')

    x_standardized = weighted_standardize(X=X, weights=weights)
    x_map = (weights[:, None] * x_standardized + prior_precision * prior_mean[None, :])
    x_map = x_map / (weights[:, None] + prior_precision)

    if after_standardization:
        return weighted_standardize(x_map, weights)
    else:
        return x_map

def weighted_standardize(X, weights):
    x_average = np.average(X, axis=0, weights=weights)
    variance = np.average(np.square(X - x_average[None, :]), axis=0, weights=weights)
    x_standardized = (X - x_average[None, :]) / np.sqrt(variance[None, :])
    return x_standardized


if __name__ == '__main__':
    X = np.arange(1,7,dtype=float).reshape(2,3)
    weights = np.array([100,300])
    prior_mean = 0
    prior_precision = 1
    X_map = normalize_like_map(X,weights,prior_mean,prior_precision)
    print('finish!')