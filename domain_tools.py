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
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


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
        _fig = plot_boundaries(distance_map=_distmap, boundaries=_boundaries)

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


def plot_boundaries(distance_map, boundaries, plot_limits=[0, 1500],
                    line_width=1, figure_dpi=200, figure_cmap='seismic_r',
                    save=False, save_folder=None, save_name=''):
    boundaries = list(boundaries)
    if 0 not in boundaries:
        boundaries = [0] + boundaries
    if len(distance_map) not in boundaries:
        boundaries += [len(distance_map)]
    # sort
    boundaries = sorted(boundaries)
    fig = plt.figure(dpi=figure_dpi)
    plt.imshow(distance_map, cmap=figure_cmap, vmin=min(
        plot_limits), vmax=max(plot_limits))
    plt.colorbar()
    for _i in range(len(boundaries)-1):
        plt.plot(np.arange(boundaries[_i], boundaries[_i+1]), boundaries[_i]*np.ones(
            boundaries[_i+1]-boundaries[_i]), 'y', linewidth=line_width)
        plt.plot(boundaries[_i]*np.ones(boundaries[_i+1]-boundaries[_i]),
                 np.arange(boundaries[_i], boundaries[_i+1]), 'y', linewidth=line_width)
        plt.plot(np.arange(boundaries[_i], boundaries[_i+1]), boundaries[_i+1]*np.ones(
            boundaries[_i+1]-boundaries[_i]), 'y', linewidth=line_width)
        plt.plot(boundaries[_i+1]*np.ones(boundaries[_i+1]-boundaries[_i]),
                 np.arange(boundaries[_i], boundaries[_i+1]), 'y', linewidth=line_width)
    plt.xlim([0, distance_map.shape[0]])
    plt.ylim([distance_map.shape[1], 0])
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
    if __name__ == '__main__':
        plt.show()
    return fig


def extract_sequences(zxy, domain_starts):
    """Function to extract sequences of zxy coordinates given domain start indices"""
    _dm_starts = np.array(domain_starts, dtype=np.int)
    _dm_ends = np.array(list(domain_starts[1:])+[len(zxy)], dtype=np.int)
    _zxy = np.array(zxy)
    _seqs = []
    for _start, _end in zip(_dm_starts, _dm_ends):
        _seqs.append(_zxy[_start:_end])
    return _seqs


def radius_of_gyration(segment):
    segment = np.array(segment)
    segment = np.linalg.norm(segment - np.nanmean(segment, axis=0), axis=1)


def domain_distances(zxy1, zxy2, _measure='median'):
    """Function to measure domain distances between two zxy arrays

    use KS-statistic as a distance:
        citation: https://arxiv.org/abs/1711.00761"""
    _intra_dist = np.concatenate([pdist(zxy1), pdist(zxy2)])
    _inter_dist = np.ravel(cdist(zxy1, zxy2))
    _measure = str(_measure).lower()
    if _measure == 'median':
        return np.abs(np.nanmedian(_inter_dist) - np.nanmedian(_intra_dist)) / np.abs(np.nanmedian(_inter_dist) + np.nanmedian(_intra_dist))
    if _measure == 'mean':
        return np.abs(np.nanmean(_inter_dist) - np.nanmean(_intra_dist)) / np.abs(np.nanmean(_inter_dist) + np.nanmean(_intra_dist))
    if _measure == 'ks':
        if 'ks_2samp' not in locals():
            from scipy.stats import ks_2samp
        return ks_2samp(_inter_dist, _intra_dist)[0]


def domain_pdists(dom_zxys, metric='ks', hierarchy_metric='weighted', 
                  make_dist_plot=False):
    """Calculate domain pair-wise distances, return a vector as the same order as
    scipy.spatial.distance.pdist """
    dom_pdists = []
    for _i in range(len(dom_zxys)):
        for _j in range(_i+1, len(dom_zxys)):
            dom_pdists.append(domain_distances(
                dom_zxys[_i], dom_zxys[_j], _measure=metric))
    dom_pdists = np.array(dom_pdists)
    dom_pdists[dom_pdists < 0] = 0  # remove minus distances, just in case
    
    if make_dist_plot:
        plt.figure(figsize=(6, 5))
        plt.title('Domain distances')
        plt.imshow(squareform(dom_pdists), cmap='seismic_r',
                   vmin=np.min(dom_pdists), vmax=np.max(dom_pdists))
        plt.colorbar()
        plt.show()

    return dom_pdists


def subcompartment_calling(spots, distance_zxy=_distance_zxy, dom_sz=5, gfilt_size=1.,
                           domain_dist_metric='ks', domain_cluster_metric='average',
                           corr_th=0.64, plot_steps=False, plot_results=True,
                           fig_dpi=200,  fig_dim=10, fig_font_size=18,
                           save_result_figs=False, save_folder=None, save_name='',
                           verbose=True):
    """Function to call 'sub-compartments' by thresholding correlation matrices.
    --------------------------------------------------------------------------------
    The idea for subcompartment calling:
    1. call rough domain candidates by maximize local distances.
    2. calculate 'distances' between domain candidates
    3. merge neighboring domains by correlation between domain distance vectors
    4. repeat step 2 and 3 until converge
    --------------------------------------------------------------------------------
    Inputs:
        spots: all sepected spots for this chromosome, np.ndarray
        distance_zxy: transform pixels in spots into nm, np.ndarray-like of 3 (default: [200,106,106])
        dom_sz: domain window size of the first rough domain candidate calling, int (default: 5)
        gfilt_size: filter-size to gaussian smooth picked coordinates, float (default: 1.)
        corr_th: threshold for correlations between neighboring vectors, float (default: 0.64)
        plot_steps: whether make plots during intermediate steps, bool (default: False)
        plot_results: whether make plots for results, bool (default: True)
        fig_dpi: dpi of image, int (default: 200)  
        fig_dim: dimension of subplot of image, int (default: 10)  
        fig_font_size: font size of titles in image, int (default: 18)
        save_result_figs: whether save result image, bool (default: False)
        save_folder: folder to save image, str (default: None, which means not save)
        save_name: filename of saved image, str (default: '', which will add default postfixs)
        verbose: say something!, bool (default: True)
    Output:
        cand_bd_starts: candidate boundary start region ids, list / np.1darray
        """
    ## check inputs
    if not isinstance(distance_zxy, np.ndarray):
        distance_zxy = np.array(distance_zxy)
    if len(distance_zxy) != 3:
        raise ValueError(
            f"size of distance_zxy should be 3, however {len(distance_zxy)} was given!")

    ## convert spots into zxys
    if verbose:
        print(f"-- start calling sub_compartments")
        _start_time = time.time()
    _zxy = np.array(spots)[:, 1:4] * distance_zxy[np.newaxis, :]
    # smooth
    if gfilt_size is not None and gfilt_size > 0:
        if verbose:
            print(f"--- gaussian interpolate chromosome, sigma={gfilt_size}")
        _zxy = DomainTools.interpolate_chr(_zxy, gaussian=gfilt_size)

    ## 1. call candidate domains
    if verbose:
        print(f"--- call initial candidate boundaries")
    _local_dists = []
    for i in range(len(_zxy)):
        if i >= dom_sz and i < len(_zxy)-dom_sz:
            cm1 = np.nanmean(_zxy[max(i-dom_sz, 0):i], axis=0)
            cm2 = np.nanmean(_zxy[i:i+dom_sz], axis=0)
            dist = np.linalg.norm(cm1-cm2)
            _local_dists.append(dist)
        else:
            _local_dists.append(0)
    # call maximums as domain starting boundaries
    cand_bd_starts = DomainTools.get_ind_loc_max(
        _local_dists, cutoff_max=0, valley=dom_sz)
    # append a zero as the start for first domain
    cand_bd_starts = np.concatenate([np.array([0]), cand_bd_starts])

    # initialize interation counter
    _merge_inds = np.array([-1])
    if verbose:
        print(
            f"--- start iterate to merge domains, num of candidates:{len(cand_bd_starts)}")
    # start iteration:
    while len(_merge_inds) > 0:
        # calculate domain pairwise-distances
        _dm_pdists = domain_pdists(extract_sequences(_zxy, cand_bd_starts),
                                   metric=domain_dist_metric, 
                                   hierarchy_metric=domain_cluster_metric,
                                   make_dist_plot=plot_steps)
        _coef_mat = np.corrcoef(squareform(_dm_pdists))
        # update domain_id to be merged (domain 0 will never be merged)
        _merge_inds = np.where(np.diag(_coef_mat, 1) > corr_th)[0]+1
        # if there are any domain to be merged:
        if len(_merge_inds) > 0:
            if verbose:
                print(
                    f"---* merge domain: {np.argmax(np.diag(_coef_mat,1))+1}, start with region:{cand_bd_starts[np.argmax(np.diag(_coef_mat,1))+1]}")
            # remove this domain from domain_starts (candidates)
            cand_bd_starts = np.delete(
                cand_bd_starts, np.argmax(np.diag(_coef_mat, 1))+1)
    if verbose:
        print(f"--- num of domain saved:{len(cand_bd_starts)}")

    # calculate hierarchy clusters
    _hierarchy_clusters = linkage(_dm_pdists, method=domain_cluster_metric)
    # finish up and make plot
    if plot_results:
        _fig = plt.figure(figsize=(2*fig_dim, 2*fig_dim), dpi=fig_dpi)
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        ax1.set_title('Noramlized distances between domains',
                      fontsize=fig_font_size)
        im1 = ax1.imshow(squareform(_dm_pdists), cmap='seismic_r',
                         vmin=np.min(_dm_pdists), vmax=np.max(_dm_pdists))
        cb1 = plt.colorbar(
            im1, ax=ax1, ticks=np.arange(0, 1.2, 0.2), shrink=0.8)
        cb1.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
        ax2.set_title('Correlation matrix between domains',
                      fontsize=fig_font_size)
        im2 = ax2.imshow(_coef_mat, cmap='seismic')
        cb2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cb2.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        ax3.set_title('Hierarchal clustering of domains',
                      fontsize=fig_font_size)
        dn = dendrogram(_hierarchy_clusters, ax=ax3)
        if save_result_figs and save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_name == '':
                _result_save_name = 'subcompartment-calling.png'
            else:
                _result_save_name += '_subcompartment-calling.png'
            _full_result_filename = os.path.join(
                save_folder, _result_save_name)
            if verbose:
                print(
                    f"--- save result image into file:{_full_result_filename}")
            plt.savefig(_full_result_filename, transparent=True)
        if __name__ == '__main__':
            plt.show()
        # boundary plot
        plot_boundaries(squareform(pdist(_zxy)), cand_bd_starts, figure_dpi=fig_dpi,
                        save=save_result_figs, save_folder=save_folder, save_name=save_name)
    if verbose:
        print(
            f"--- total time spent in sub-compartment calling: {time.time()-_start_time}")

    return cand_bd_starts, _hierarchy_clusters
