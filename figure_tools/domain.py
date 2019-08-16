import sys,os,re,time,glob
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
from matplotlib import cm 
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy
from scipy.signal import fftconvolve
from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter
from scipy import ndimage, stats
from skimage import morphology, restoration, measure
from skimage.segmentation import random_walker
from scipy.ndimage import gaussian_laplace
import cv2
import multiprocessing as mp
from sklearn.decomposition import PCA

from . import _distance_zxy,_sigma_zxy,_allowed_colors

from scipy.stats import linregress
#from astropy.convolution import Gaussian2DKernel,convolve



## Plotting function
def plot_boundary_probability(region_ids, domain_start_list, figure_kwargs={}, plot_kwargs={},
                              xlabel="region_ids", ylabel="probability", fontsize=16,
                              save=False, save_folder='.', save_name=''):
    """Wrapper function to plot boundary probability given domain_start list"""
    if 'plt' not in locals():
        import matplotlib.pyplot as plt
    # summarize
    _x = np.array(region_ids, dtype=np.int)
    _y = np.zeros(np.shape(_x), dtype=np.float)
    for _dm_starts in domain_start_list:
        for _d in _dm_starts:
            if _d > 0 and _d in _x:
                _y[np.where(_x == _d)[0]] += 1
    _y = _y / len(domain_start_list)
    _fig, _ax = plt.subplots(figsize=(15, 5), dpi=200, **figure_kwargs)
    _ax.plot(_x, _y, label=ylabel, **plot_kwargs)
    _ax.set_xlim([0, len(_x)])
    _ax.set_xlabel(xlabel, fontsize=fontsize)
    _ax.set_ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    if save:
        _filename = 'boundary_prob.png'
        if save_name != '':
            _filename = save_name + '_' + _filename
        plt.savefig(os.path.join(save_folder, _filename), transparent=True)
    return _ax

def plot_boundaries(distance_map, boundaries, input_ax=None, plot_limits=[0, 1500],
                    line_width=1.5, figure_dpi=200, figure_fontsize=20, figure_cmap='seismic_r',
                    save=False, save_folder=None, save_name=''):
    boundaries = list(boundaries)
    if 0 not in boundaries:
        boundaries = [0] + boundaries
    if len(distance_map) not in boundaries:
        boundaries += [len(distance_map)]
    # sort
    boundaries = sorted([int(_b) for _b in boundaries])
    if input_ax is None:
        fig = plt.figure(dpi=figure_dpi)
        ax = plt.subplot(1, 1, 1)
    else:
        ax = input_ax
    im = ax.imshow(distance_map, cmap=figure_cmap,
                   vmin=min(plot_limits), vmax=max(plot_limits))
    plt.subplots_adjust(left=0.02, bottom=0.06,
                        right=0.95, top=0.94, wspace=0.05)
    if input_ax is None:
        cb = plt.colorbar(im, ax=ax)
    else:
        cb = plt.colorbar(im, ax=ax, shrink=0.75)
        cb.ax.tick_params(labelsize=figure_fontsize)
        ax.tick_params(labelsize=figure_fontsize)
        ax.yaxis.set_ticklabels([])
        line_width *= 2
    for _i in range(len(boundaries)-1):
        ax.plot(np.arange(boundaries[_i], boundaries[_i+1]), boundaries[_i]*np.ones(
            boundaries[_i+1]-boundaries[_i]), 'y', linewidth=line_width)
        ax.plot(boundaries[_i]*np.ones(boundaries[_i+1]-boundaries[_i]),
                np.arange(boundaries[_i], boundaries[_i+1]), 'y', linewidth=line_width)
        ax.plot(np.arange(boundaries[_i], boundaries[_i+1]), boundaries[_i+1]*np.ones(
            boundaries[_i+1]-boundaries[_i]), 'y', linewidth=line_width)
        ax.plot(boundaries[_i+1]*np.ones(boundaries[_i+1]-boundaries[_i]),
                np.arange(boundaries[_i], boundaries[_i+1]), 'y', linewidth=line_width)
    ax.set_xlim([0, distance_map.shape[0]])
    ax.set_ylim([distance_map.shape[1], 0])

    if save:
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_name == '':
                save_name = 'boundaries.png'
            else:
                if '.png' not in save_name:
                    save_name += '_boundaries.png'
            fig.savefig(os.path.join(save_folder, save_name), transparent=True)

    return ax

