import sys
import os
import re
import time
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
import scipy

import multiprocessing as mp

from . import get_img_info, corrections, visual_tools, alignment_tools, analysis, classes
from .External import Fitting_v3, DomainTools
from . import _correction_folder, _temp_folder, _distance_zxy, _sigma_zxy, _image_size, _allowed_colors

from astropy.convolution import Gaussian2DKernel, convolve
from scipy.signal import find_peaks, fftconvolve

from scipy.stats import normaltest, ks_2samp, ttest_ind

def compare_domains_2(_distmap, _domain_starts, _i1, _i2, gaussian_p_th=0.05, p_th=0.01, ks_th=0.3, make_plot=False, verbose=False):

    _size = len(_distmap)

    _s1 = _domain_starts[_i1]
    if _i1 <= len(_domain_starts) - 2:
        _e1 = _domain_starts[_i1+1]
    else:
        _e1 = _size

    _s2 = _domain_starts[_i2]
    if _i2 <= len(_domain_starts) - 2:
        _e2 = _domain_starts[_i2+1]
    else:
        _e2 = _size

    _b1 = np.ravel(np.triu(_distmap[_s1:_e1, _s1:_e1], 1))
    _b2 = np.ravel(np.triu(_distmap[_s2:_e2, _s2:_e2], 1))

    _intra_dist = np.concatenate([_b1, _b2])
    _intra_dist = _intra_dist[np.isnan(_intra_dist) == False]
    _intra_dist = _intra_dist[_intra_dist > 200]
    _inter_dist = np.ravel(_distmap[_s1:_e1, _s2:_e2])
    _inter_dist = _inter_dist[np.isnan(_inter_dist) == False]

    # first test whether two distributions are gaussian
    if normaltest(_intra_dist)[1] < gaussian_p_th and normaltest(_inter_dist)[1] < gaussian_p_th:
        # now considered as testing normal means
        _tt, _p = ttest_ind(_intra_dist, _inter_dist, equal_var=False)
        _use_gaussian = True

    else:
        # not normal: do KS test
        _ks, _p = ks_2samp(_intra_dist, _inter_dist)
        _use_gaussian = False

    # make plot
    if make_plot:
        plt.figure()
        plt.title(f"Gaussian={_use_gaussian}, p={np.round(_p, 4)}")
        plt.hist(_intra_dist, density=True, alpha=0.5, label='intra')
        plt.hist(_inter_dist, density=True, alpha=0.5, label='inter')
        plt.legend()
        if __name__ == '__main__':
            plt.show()
        print(ks_2samp(_inter_dist, _intra_dist), _p, _use_gaussian)
    if _use_gaussian:
        return _p, _p < p_th
    else:
        return _p, (_p < p_th and _ks > ks_th)


def _fuse_domains_2(distmap, domain_starts, gaussian_p_th=0.05, p_th=0.01, ks_th=0.3, make_plot=False, verbose=False):
    _distmap = distmap.copy()
    _starts = list(domain_starts)
    while len(_starts) > 1:
        _i = 1
        _merge = 0
        while _i < len(_starts)-1:
            _p_left, _split_left = compare_domains_2(_distmap, _starts, _i, _i-1,
                                                     gaussian_p_th=gaussian_p_th,
                                                     p_th=p_th, ks_th=ks_th, make_plot=make_plot)
            _p_right, _split_right = compare_domains_2(_distmap, _starts, _i, _i+1,
                                                       gaussian_p_th=gaussian_p_th,
                                                       p_th=p_th, ks_th=ks_th, make_plot=make_plot)
            if not _split_left and _split_right:
                _starts.pop(_i)
                _merge += 1
            elif _split_left and not _split_right:
                _starts.pop(_i+1)
                _merge += 1
            elif not _split_left and not _split_right:
                if _p_left > _p_right:
                    _starts.pop(_i)
                    _merge += 1
                elif _p_right > _p_left:
                    _starts.pop(_i+1)
                    _merge += 1
            _i += 1

        if _merge == 0:
            break
    return _starts


def domain_calling(distmap, gfilt_size=3, dom_sz=4, p_th=0.001, gaussian_p_th=0.05, ks_th=0.3, make_plot=False):

    _kernel = Gaussian2DKernel(gfilt_size)
    _distmap = distmap.copy()
    # apply gaussian filter
    if gfilt_size > 0:
        for _i in range(len(_distmap)-1):
            _distmap[_i, _i] = np.nan
        _distmap = convolve(distmap, _kernel)
    # calculate candidate domain boundary
    _candidate_boundaries = _get_candidate_boundary(
        _distmap, dom_sz=dom_sz, make_plot=make_plot)
    # fuse candidate domains
    _final_starts = _fuse_domains_2(_distmap, _candidate_boundaries, p_th=p_th,
                                    gaussian_p_th=gaussian_p_th, ks_th=ks_th, make_plot=False)
    _boundaries = _final_starts + [len(distmap)]

    if make_plot:
        _plot_boundaries(_distmap=_distmap, _boundaries=_boundaries)

    return _boundaries


def _get_candidate_boundary(_distmap, dom_sz=4, make_plot=False):
    
    _size = len(_distmap)
    # initialize
    _dists = []
    # get dists
    for _i in range(_size):
        if _i >= dom_sz/2 and _i < _size - dom_sz/2:
            _b1 = np.ravel(
                np.triu(_distmap[max(_i-dom_sz, 0):_i, max(_i-dom_sz, 0):_i], 1))
            _b2 = np.ravel(
                np.triu(_distmap[_i:min(_i+dom_sz, _size), _i:min(_i+dom_sz, _size)], 1))
            _intra = np.concatenate([_b1, _b2])
            _inter = np.ravel(
                _distmap[max(_i-dom_sz, 0):_i, _i:min(_i+dom_sz, _size)])
            _dists.append(np.nanmean(_inter) - np.nanmean(_intra))
        else:
            _dists.append(0)
    # call peaks
    _peaks = find_peaks(_dists, distance=dom_sz, prominence=10)
    # candidate boundaries
    _cand_boundaries = [0] + list(_peaks[0])

    if make_plot:
        plt.figure(figsize=[18, 3])
        plt.plot(_dists)
        plt.xticks(np.arange(0, _size, _size/10))
        if __name__ == '__main__':
            plt.show()
    return _cand_boundaries


def _plot_boundaries(_distmap, _boundaries):
    plt.figure(dpi=200)
    plt.imshow(_distmap, cmap='seismic_r', vmin=0, vmax=np.nanmedian(_distmap)*2)
    plt.colorbar()
    for _i in range(len(_boundaries)-1):
        plt.plot(np.arange(_boundaries[_i], _boundaries[_i+1]), _boundaries[_i]
                 * np.ones(_boundaries[_i+1]-_boundaries[_i]), 'y', linewidth=1)
        plt.plot(_boundaries[_i]*np.ones(_boundaries[_i+1]-_boundaries[_i]),
                 np.arange(_boundaries[_i], _boundaries[_i+1]), 'y', linewidth=1)
        plt.plot(np.arange(_boundaries[_i], _boundaries[_i+1]), _boundaries[_i+1]
                 * np.ones(_boundaries[_i+1]-_boundaries[_i]), 'y', linewidth=1)
        plt.plot(_boundaries[_i+1]*np.ones(_boundaries[_i+1]-_boundaries[_i]),
                 np.arange(_boundaries[_i], _boundaries[_i+1]), 'y', linewidth=1)
    if __name__ == '__main__':
        plt.show()


def extract_sequences(zxy, domain_starts):
    """Function to extract sequences of zxy coordinates given domain start indices"""
    _dm_starts = np.array(domain_starts, dtype=np.int)
    _dm_ends = np.array(list(domain_starts[1:])+[len(zxy)], dtype=np.int)
    _zxy = np.array(zxy)
    _seqs = []
    for _start, _end in zip(_dm_starts, _dm_ends):
        _seqs.append(_zxy[_start:_end])
    return _seqs
