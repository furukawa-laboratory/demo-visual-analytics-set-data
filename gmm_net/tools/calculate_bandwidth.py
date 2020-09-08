import math


def calculate_bandwidth(
        n_data_in_one_sigma, n_samples, n_components, width_latent_space
):
    volume_latent_space = width_latent_space ** n_components
    if n_components == 2:
        return math.sqrt(
            (n_data_in_one_sigma * volume_latent_space) / (n_samples * math.pi)
        )
    elif n_components == 3:
        return math.pow(
            (3.0 * n_data_in_one_sigma * volume_latent_space) / (4.0 * math.pi * n_samples)
            , 1.0 / 3.0)
    else:
        raise ValueError("Not implemented n_components={}".format(n_components))
