# import packages
from matplotlib.colors import ListedColormap
import numpy as np

def transparent_cmap(cmap, increasing_alpha=True, N=255, max_alpha=1):
    "Copy colormap and set a gradually changing alpha values"
    mycmap = cmap
    mycmap._init()
    if increasing_alpha:
        mycmap._lut[:,-1] = np.linspace(0, max_alpha, N+4)
    else:
        mycmap._lut[:,-1] = np.linspace(max_alpha, 0, N+4)
    return mycmap

def black_gradient(color, num_colors=256, max_alpha=1, transparent=False):
    """Generate a black backed linear color map"""
    color =  np.array(color[:3])
    # initialzie colors
    _colors = np.zeros([num_colors,4])
    for _i, _c in enumerate(color):
        _colors[:,_i] = np.linspace(0, _c, num_colors)
    if transparent:
        _colors[:, -1] = np.linspace(0, max_alpha, num_colors)
    else:
        _colors[:, -1] = max_alpha
    return ListedColormap(_colors)

def transparent_gradient(color, num_colors=256, max_alpha=1):
    """Generate a black backed linear color map"""
    color =  np.array(color[:3])
    # initialzie colors
    _colors = np.zeros([num_colors,4])
    _colors[:,:3] = color[np.newaxis,:]
    # set alpha
    _colors[:, -1] = np.linspace(0, max_alpha, num_colors)

    return ListedColormap(_colors)