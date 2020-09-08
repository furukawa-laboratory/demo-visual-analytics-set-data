import numpy as np
import matplotlib.pyplot as plt
import colorsys
def cmap2d_base_and_brightness(value: np.ndarray, brightness: np.ndarray, base_cmap: str,
                               vmin=None, vmax=None, bmin=None, bmax=None,
                               used_bright_range=[0.0, 1.0]) -> np.ndarray:
    # exception handling
    if value.ndim != 1:
        raise ValueError('value must be 1d np.ndarray')
    if brightness.ndim != 1:
        raise ValueError('brightness must be 1d np.ndarray')
    for threshold in used_bright_range:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError('invalid used_bright_range={}'.format(used_bright_range))
    if used_bright_range[0] > used_bright_range[1]:
        raise ValueError('invalid used_bright_range={}'.format(used_bright_range))

    # set min and max value
    if vmin is None:
        vmin = value.min()
    if vmax is None:
        vmax = value.max()
    if bmin is None:
        bmin = brightness.min()
    if bmax is None:
        bmax = brightness.max()

    # get cmap
    cm = plt.get_cmap(base_cmap)
    # get the number of segments of colormap
    n_segments = cm.N
    # normalize 0 to 1
    value_normalized = (value - vmin) / (vmax - vmin)
    # get the segment which the value belongs to
    segment = (value_normalized * n_segments).astype(int)
    # convert rgb to hsv corresponds value
    hsv = np.array([colorsys.rgb_to_hsv(*cm(s)[:3]) for s in segment])

    # normalize brightness value [0, 1]
    brightness_normalized = (brightness - bmin) / (bmax - bmin)
    # normalize brightness value to used_bright_range
    brightness_normalized = brightness_normalized * (used_bright_range[1] - used_bright_range[0]) + used_bright_range[0]
    hsv[:, 2] = brightness_normalized
    rgb = np.array([colorsys.hsv_to_rgb(*h) for h in hsv.tolist()])
    return rgb



if __name__ == "__main__":
    value = np.linspace(0.0, 1.0, 256)
    brightness = np.linspace(0.0, 1.0, 256)
    mesh = np.meshgrid(value, brightness, indexing='ij')
    mesh = np.dstack(mesh).reshape(-1,2)
    rgb = cmap2d_base_and_brightness(mesh[:, 0],
                                     mesh[:, 1],
                                     base_cmap='bwr',
                                     used_bright_range=[0.3, 1.0])
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, aspect='equal')
    rgb_3d = rgb.reshape(brightness.shape[0], value.shape[0], 3)
    ax.imshow(X=rgb_3d, interpolation='spline16')
    #ax.scatter(mesh[:,0],mesh[:,1],s=1,c=rgb)
    plt.show()
