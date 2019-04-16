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
        if save_folder is not None and input_ax is None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_name == '':
                save_name = 'boundaries.png'
            else:
                if '.png' not in save_name:
                    save_name += '_boundaries.png'
            fig.savefig(os.path.join(save_folder, save_name), transparent=True)

    return ax


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


def domain_distances(zxy1, zxy2, _measure='median', dist_th=1, 
                     _normalization_mat=None, _dom1_bds=None, _dom2_bds=None):
    """Function to measure domain distances between two zxy arrays
    use KS-statistic as a distance:
        citation: https://arxiv.org/abs/1711.00761"""
    _intra_dist = [pdist(zxy1), pdist(zxy2)]
    _inter_dist = np.ravel(cdist(zxy1, zxy2))
    _measure = str(_measure).lower()
    _dom1_bds = [int(_b) for _b in _dom1_bds]
    _dom2_bds = [int(_b) for _b in _dom2_bds]
    # normalization
    if _normalization_mat is not None:
        # check other inputs
        if _dom1_bds is None or _dom2_bds is None:
            raise TypeError(
                f"Domain boundaries not fully given while normalization specified, skip normalization!")
        # normalize!
        else:
            _intra_dist[0] = _intra_dist[0] / squareform(
                _normalization_mat[_dom1_bds[0]:_dom1_bds[1], _dom1_bds[0]:_dom1_bds[1]])
            _intra_dist[1] = _intra_dist[1] / squareform(
                _normalization_mat[_dom2_bds[0]:_dom2_bds[1], _dom2_bds[0]:_dom2_bds[1]])
            _intra_dist = np.concatenate(_intra_dist)
            _inter_dist = _inter_dist / \
                np.ravel(
                    _normalization_mat[_dom1_bds[0]:_dom1_bds[1], _dom2_bds[0]:_dom2_bds[1]])
    else:
        # not normalize? directly concatenate
        _intra_dist = np.concatenate(_intra_dist)

    _kept_inter = _inter_dist[np.isnan(_inter_dist) == False]
    _kept_intra = _intra_dist[np.isnan(_intra_dist) == False]
    if len(_kept_inter) == 0 or len(_kept_intra) == 0:
        return min(dist_th*2, 1)

    if _measure == 'median':
        m_inter, m_intra = np.nanmedian(_inter_dist), np.nanmedian(_intra_dist)
        v_inter, v_intra = np.nanmedian(
            (_inter_dist-m_inter)**2), np.nanmedian((_intra_dist-m_intra)**2)
        return (m_inter-m_intra)/np.sqrt(v_inter+v_intra)
    if _measure == 'mean':
        m_inter, m_intra = np.nanmean(_inter_dist), np.nanmean(_intra_dist)
        v_inter, v_intra = np.nanvar(_inter_dist), np.nanvar(_intra_dist)
        return (m_inter-m_intra)/np.sqrt(v_inter+v_intra)
    if _measure == 'ks':
        if 'ks_2samp' not in locals():
            from scipy.stats import ks_2samp
        return ks_2samp(_kept_inter, _kept_intra)[0]


def domain_pdists(dom_zxys, metric='distance', normalization_mat=None, domain_starts=None):
    """Calculate domain pair-wise distances, return a vector as the same order as
    scipy.spatial.distance.pdist """
    # first check whether do normalzation
    if normalization_mat is not None:
        # check other inputs
        if domain_starts is None:
            raise TypeError(
                f"_domain_starts should be given while normalization specified!")
        elif len(domain_starts) != len(dom_zxys):
            raise ValueError(
                f"number of domain zxys:{len(dom_zxys)} and number of domain_starts:{len(domain_starts)} doesn't match!")
        else:
            domain_ends = np.concatenate(
                [domain_starts[1:], np.array([len(normalization_mat)])])
    else:
        domain_starts = np.zeros(len(dom_zxys))
        domain_ends = np.zeros(len(dom_zxys))
    dom_pdists = []
    for _i in range(len(dom_zxys)):
        for _j in range(_i+1, len(dom_zxys)):
            dom_pdists.append(domain_distances(dom_zxys[_i], dom_zxys[_j], _measure=metric,
                                               _normalization_mat=normalization_mat,
                                               _dom1_bds=[
                                                   domain_starts[_i], domain_ends[_i]],
                                               _dom2_bds=[domain_starts[_j], domain_ends[_j]]))
    dom_pdists = np.array(dom_pdists)
    dom_pdists[dom_pdists < 0] = 0  # remove minus distances, just in case

    return dom_pdists

def call_candidate_boundaries(_zxy, _dom_sz, _method='local'):
    """This is the function to first test call domains"""
    if _method == 'local':
        _local_dists = []
        _start = 1
        for i in range(_start,len(_zxy)):
            cm1 = np.nanmean(_zxy[max(i-_dom_sz, 0):i], axis=0)
            cm2 = np.nanmean(_zxy[i:i+_dom_sz], axis=0)
            dist = np.linalg.norm(cm1-cm2)
            _local_dists.append(dist)

        # call maximums as domain starting boundaries
        _cand_bd_starts = DomainTools.get_ind_loc_max(
            _local_dists, cutoff_max=0, valley=_dom_sz)
        # remove extreme values
        filtered_bd_starts = []
        for _bd in list(_cand_bd_starts):
            if _bd >= _dom_sz and _bd < len(_zxy)-_dom_sz:
                filtered_bd_starts.append(_bd + _start)

        # append a zero as the start for first domain
        if 0 not in filtered_bd_starts:
            filtered_bd_starts = np.concatenate([np.array([0]), 
                np.array(filtered_bd_starts)]).astype(np.int)

    return filtered_bd_starts


def merge_domains(_zxy, cand_bd_starts, norm_mat=None, corr_th=0.64, dist_th=0.2,
                  domain_dist_metric='ks', plot_steps=False, verbose=True):
    """Function to merge domains given zxy coordinates and candidate_boundaries"""
    cand_bd_starts = np.array(cand_bd_starts, dtype=np.int)
    _merge_inds = np.array([-1])
    if verbose:
        print(
            f"--- start iterate to merge domains, num of candidates:{len(cand_bd_starts)}")
    # start iteration:
    while len(_merge_inds) > 0 and len(cand_bd_starts) > 1:
        # calculate domain pairwise-distances
        _dm_pdists = domain_pdists(extract_sequences(_zxy, cand_bd_starts),
                                   metric=domain_dist_metric, 
                                   normalization_mat=norm_mat,
                                   domain_starts=cand_bd_starts)
        _coef_mat = np.corrcoef(squareform(_dm_pdists))
        if plot_steps:
            plt.figure()
            plt.imshow(squareform(_dm_pdists), cmap='seismic_r')
            plt.title(f"Domain distances")
            plt.colorbar()
            plt.show()
            plt.figure()
            plt.imshow(_coef_mat, cmap='seismic')
            plt.title(f"Domain correlations")
            plt.colorbar()
            plt.show()
        # update domain_id to be merged (domain 0 will never be merged)
        if len(cand_bd_starts) > 2:
            _corr_inds = np.where(np.diag(_coef_mat, 1) > corr_th)[0]+1
        else:
            _corr_inds = np.where(np.diag(_coef_mat, 1) >= -1)[0]+1
        _dist_inds =np.where(np.diag(squareform(_dm_pdists), 1) <= dist_th)[0]+1
        if len(_dist_inds) > 0 and len(_corr_inds) > 0:
            _merge_inds = [int(_cid) for _cid in _corr_inds if _cid in _dist_inds]
        else:
            _merge_inds = []
        # if there are any domain to be merged:
        if len(_merge_inds) > 0:
            # find the index with minimum distance bewteen neighboring domains
            # first merge neighbors with high corr and low dist
            _merge_dists = (1-np.diag(squareform(_dm_pdists), 1)) / (1-dist_th) + \
                np.diag(_coef_mat,1) / corr_th 
            # filter
            _merge_dists = _merge_dists[np.array(_merge_inds, dtype=np.int)-1]
            _picked_ind = _merge_inds[np.argmax(_merge_dists)]
            if verbose:
                print(f"---* merge domain:{_picked_ind} starting with region:{cand_bd_starts[_picked_ind]}")
            # remove this domain from domain_starts (candidates)
            cand_bd_starts = np.delete(cand_bd_starts, _picked_ind)
    
    if verbose and len(cand_bd_starts) == 1:
            print(f"--- only 1 domain left, skip plotting.")
    elif verbose:
        print(
            f"--- final neighbor domain dists:{np.diag(squareform(_dm_pdists),1)}")
        print(f"--- final neighbor domain corr:{np.diag(_coef_mat,1)}")
        print(f"--- num of domain kept:{len(cand_bd_starts)}")

    return cand_bd_starts


def basic_domain_calling(spots, save_folder=None,
                         distance_zxy=_distance_zxy, dom_sz=5, gfilt_size=0.5,
                         normalization_matrix=r'Z:\References\normalization_matrix.npy',
                         domain_dist_metric='ks', domain_cluster_metric='ward',
                         corr_th=0.64, dist_th=0.2, plot_steps=False, plot_results=True,
                         fig_dpi=100,  fig_dim=10, fig_font_size=18,
                         save_result_figs=False, save_name='', verbose=True):
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
        normalization_matrix: either path or matrix for normalizing polymer effect, str or np.ndarray 
            * if specified, allow normalization, otherwise put a None. (default:str of path)        corr_th: lower threshold for correlations to merge neighboring vectors, float (default: 0.64)
        dist_th: upper threshold for distance to merge neighboring vectors, float (default: 0.2)
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
    # load normalization if specified
    if isinstance(normalization_matrix, str) and os.path.isfile(normalization_matrix):
        norm_mat = np.load(normalization_matrix)
        _normalization = True
    elif isinstance(normalization_matrix, np.ndarray):
        norm_mat = normalization_matrix.copy()
    else:
        _normalization = False

    ## 0. prepare coordinates
    if verbose:
        print(f"-- start basic domain calling")
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
    cand_bd_starts = call_candidate_boundaries(_zxy, dom_sz, 'local')

    ## 2. get zxy sequences
    cand_bd_starts = merge_domains(_zxy=_zxy, cand_bd_starts=cand_bd_starts,
                                   norm_mat=norm_mat, corr_th=corr_th, dist_th=dist_th,
                                   domain_dist_metric=domain_dist_metric,
                                   plot_steps=plot_steps, verbose=verbose)

    ## 3. finish up and make plot
    if plot_results and len(cand_bd_starts) > 1:
        _dm_pdists = domain_pdists(extract_sequences(_zxy, cand_bd_starts),
                                   metric=domain_dist_metric,
                                   normalization_mat=norm_mat,
                                   domain_starts=cand_bd_starts)
        _coef_mat = np.corrcoef(squareform(_dm_pdists))
        if verbose:
            print(
                f"-- make plot for results with {len(cand_bd_starts)} domains")
        import matplotlib.gridspec as gridspec
        _fig = plt.figure(figsize=(3*(fig_dim+1), 2*fig_dim), dpi=fig_dpi)
        gs = gridspec.GridSpec(2, 5, figure=_fig)
        gs.update(wspace=0.1, hspace=0)
        #ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=1)
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title('Noramlized distances between domains',
                      fontsize=fig_font_size)
        im1 = ax1.imshow(squareform(_dm_pdists), cmap='seismic_r',
                         vmin=0, vmax=1)
        cb1 = plt.colorbar(
            im1, ax=ax1, ticks=np.arange(0, 1.2, 0.2), shrink=0.6)
        cb1.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        #ax2 = plt.subplot2grid((2, 5), (0, 1), colspan=1)
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_title('Correlation matrix between domains',
                      fontsize=fig_font_size)
        im2 = ax2.imshow(_coef_mat, cmap='seismic')
        cb2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
        cb2.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        #ax4 = plt.subplot2grid((2, 5), (1, 0), colspan=2, rowspan=1)
        ax4 = plt.subplot(gs[1, 0:2])
        ax4.set_title('Hierarchal clustering of domains',
                      fontsize=fig_font_size)
        # calculate hierarchy clusters
        _hierarchy_clusters = linkage(_dm_pdists, method=domain_cluster_metric)
        # draw dendrogram
        dn = dendrogram(_hierarchy_clusters, ax=ax4)
        ax4.tick_params(labelsize=fig_font_size)

        #ax3 = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2)
        ax3 = plt.subplot(gs[:, 2:])
        plot_boundaries(squareform(pdist(_zxy)), cand_bd_starts, input_ax=ax3,
                        figure_dpi=fig_dpi,
                        figure_fontsize=fig_font_size, save=save_result_figs,
                        save_folder=save_folder, save_name=save_name)
        # save result figure
        if save_result_figs and save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_name == '':
                _result_save_name = 'basic_domain_calling.png'
            else:
                _result_save_name = save_name + '_basic_domain_calling.png'
            _full_result_filename = os.path.join(
                save_folder, _result_save_name)
            if verbose:
                print(
                    f"--- save result image into file:{_full_result_filename}")
            plt.savefig(_full_result_filename, transparent=True)
        if __name__ == '__main__':
            plt.show()

    if verbose:
        print(
            f"--- total time spent in basic domain calling: {time.time()-_start_time}")

    return cand_bd_starts


def iterative_domain_calling(spots, save_folder=None,
                             distance_zxy=_distance_zxy, dom_sz=5, gfilt_size=0.5,
                             split_level=1, num_iter=5, corr_th_scaling=1., dist_th_scaling=1.,
                             normalization_matrix=r'Z:\References\normalization_matrix.npy',
                             domain_dist_metric='ks', domain_cluster_metric='ward',
                             corr_th=0.6, dist_th=0.2, plot_steps=False, plot_results=True,
                             fig_dpi=100,  fig_dim=10, fig_font_size=18,
                             save_result_figs=False, save_name='', verbose=True):
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
        split_level: number of iterative split candidate domains, int (default: 1)
        num_iter: number of iterations for split-merge domains, int (default: 5)
        corr_th_scaling: threshold scaling for corr_th during split-merge iteration, float (default: 1.)
        dist_th_scaling: threshold scaling for dist_th during split-merge iteration, float (default: 1.)
        normalization_matrix: either path or matrix for normalizing polymer effect, str or np.ndarray 
            * if specified, allow normalization, otherwise put a None. (default:str of path)
        corr_th: lower threshold for correlations to merge neighboring vectors, float (default: 0.64)
        dist_th: upper threshold for distance to merge neighboring vectors, float (default: 0.2)
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
    # load normalization if specified
    if isinstance(normalization_matrix, str) and os.path.isfile(normalization_matrix):
        norm_mat = np.load(normalization_matrix)
        _normalization = True
    elif isinstance(normalization_matrix, np.ndarray):
        _normalization = True
        norm_mat = normalization_matrix.copy()
    else:
        _normalization = False
    if corr_th_scaling <= 0:
        raise ValueError(
            f"corr_th_scaling should be float number in [0,1], while {corr_th_scaling} given!")
    if dist_th_scaling <= 0:
        raise ValueError(
            f"dist_th_scaling should be float number in [0,1], while {dist_th_scaling} given!")
    ## 0. prepare coordinates
    if verbose:
        print(f"- start iterative domain calling")
        _start_time = time.time()
    _zxy = np.array(spots)[:, 1:4] * distance_zxy[np.newaxis, :]
    # smooth
    if gfilt_size is not None and gfilt_size > 0:
        if verbose:
            print(f"--- gaussian interpolate chromosome, sigma={gfilt_size}")
        _zxy = DomainTools.interpolate_chr(_zxy, gaussian=gfilt_size)

    ## 1. do one round of basic domain calling
    cand_bd_starts = basic_domain_calling(spots=spots, distance_zxy=_distance_zxy,
                                          dom_sz=dom_sz, gfilt_size=gfilt_size,
                                          normalization_matrix=normalization_matrix,
                                          domain_dist_metric=domain_dist_metric,
                                          domain_cluster_metric=domain_cluster_metric,
                                          corr_th=corr_th, dist_th=dist_th, plot_steps=False,
                                          plot_results=False, save_result_figs=False,
                                          save_folder=None, save_name='', verbose=verbose)
    ## 2. iteratively update domains
    for _i in range(int(num_iter)):
        cand_bd_ends = np.concatenate([cand_bd_starts[1:], np.array([len(spots)])])
        splitted_starts = cand_bd_starts.copy()
        for _j in range(int(split_level)):
            splitted_starts = list(splitted_starts)
            for _start, _end in zip(splitted_starts, cand_bd_ends):
                if (_end - _start) > 2 * dom_sz:
                    if _normalization:
                        new_bds = basic_domain_calling(spots[_start:_end],
                                                    normalization_matrix=norm_mat[_start:_end,
                                                                                    _start:_end],
                                                    dist_th=dist_th,
                                                    corr_th=corr_th*corr_th_scaling,
                                                    gfilt_size=gfilt_size,
                                                    domain_dist_metric=domain_dist_metric,
                                                    plot_results=False, verbose=False)
                    else:
                        new_bds = basic_domain_calling(spots[_start:_end],
                                                    normalization_matrix=None,
                                                    dist_th=dist_th,
                                                    corr_th=corr_th*corr_th_scaling,
                                                    gfilt_size=gfilt_size,
                                                    domain_dist_metric=domain_dist_metric,
                                                    plot_results=False, verbose=False)
                    # save new boundaries
                    splitted_starts += list(_start+new_bds)
            # summarize new boundaries
            splitted_starts = np.unique(splitted_starts).astype(np.int)
        # merge
        if _normalization:
            # no scaling for dist_th
            new_starts = merge_domains(_zxy, splitted_starts, norm_mat=norm_mat,
                                       corr_th=corr_th,
                                       dist_th=dist_th*dist_th_scaling, 
                                       domain_dist_metric=domain_dist_metric,
                                       plot_steps=False, verbose=verbose)
        else:
            new_starts = merge_domains(_zxy, splitted_starts, norm_mat=None,
                                       corr_th=corr_th,
                                       dist_th=dist_th*dist_th_scaling,
                                       domain_dist_metric=domain_dist_metric,
                                       plot_steps=False, verbose=verbose)
        # check if there is no change at all
        if len(new_starts) == len(cand_bd_starts) and (new_starts == cand_bd_starts).all():
            if verbose:
                print(f"-- iter {_i} finished, all boundaries are kept, exit!")
            break
        # else, update
        else:
            cand_bd_starts = new_starts
            if verbose:
                print(
                    f"-- iter {_i} finished, num of updated boundaries: {len(new_starts)}")

    ## 3. plot results
    if plot_results and len(cand_bd_starts) > 1:
        _dm_pdists = domain_pdists(extract_sequences(_zxy, cand_bd_starts),
                                   metric=domain_dist_metric,
                                   normalization_mat=norm_mat,
                                   domain_starts=cand_bd_starts)
        _coef_mat = np.corrcoef(squareform(_dm_pdists))
        if verbose:
            print(
                f"-- make plot for results with {len(cand_bd_starts)} domains")
        import matplotlib.gridspec as gridspec
        _fig = plt.figure(figsize=(3*(fig_dim+1), 2*fig_dim), dpi=fig_dpi)
        gs = gridspec.GridSpec(2, 5, figure=_fig)
        gs.update(wspace=0.1, hspace=0)
        #ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=1)
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title('Noramlized distances between domains',
                      fontsize=fig_font_size)
        im1 = ax1.imshow(squareform(_dm_pdists), cmap='seismic_r',
                         vmin=0, vmax=1)
        cb1 = plt.colorbar(
            im1, ax=ax1, ticks=np.arange(0, 1.2, 0.2), shrink=0.6)
        cb1.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        #ax2 = plt.subplot2grid((2, 5), (0, 1), colspan=1)
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_title('Correlation matrix between domains',
                      fontsize=fig_font_size)
        im2 = ax2.imshow(_coef_mat, cmap='seismic')
        cb2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
        cb2.ax.tick_params(labelsize=fig_font_size-2, width=0.6, length=1)

        #ax4 = plt.subplot2grid((2, 5), (1, 0), colspan=2, rowspan=1)
        ax4 = plt.subplot(gs[1, 0:2])
        ax4.set_title('Hierarchal clustering of domains',
                      fontsize=fig_font_size)
        # calculate hierarchy clusters
        _hierarchy_clusters = linkage(_dm_pdists, method=domain_cluster_metric)
        # draw dendrogram
        dn = dendrogram(_hierarchy_clusters, ax=ax4)
        ax4.tick_params(labelsize=fig_font_size)

        #ax3 = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2)
        ax3 = plt.subplot(gs[:, 2:])
        plot_boundaries(squareform(pdist(_zxy)), cand_bd_starts, input_ax=ax3,
                        figure_dpi=fig_dpi,
                        figure_fontsize=fig_font_size, save=save_result_figs,
                        save_folder=save_folder, save_name=save_name)
        # save result figure
        if save_result_figs and save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if save_name == '':
                _result_save_name = 'iterative_domain_calling.png'
            else:
                _result_save_name = save_name + '_iterative_domain_calling.png'
            _full_result_filename = os.path.join(
                save_folder, _result_save_name)
            if os.path.isfile(_full_result_filename):
                _full_result_filename.replace('.png', '_.png')
            if verbose:
                print(
                    f"--- save result image into file:{_full_result_filename}")
            plt.savefig(_full_result_filename, transparent=True)
        if __name__ == '__main__':
            plt.show()
    if verbose:
        print(
            f"-- total time for iterative domain calling: {time.time()-_start_time}")

    return cand_bd_starts


def local_domain_calling(spots, save_folder=None,
                         distance_zxy=_distance_zxy, dom_sz=5, gfilt_size=0.5,
                         cutoff_max=0.5, plot_results=True,
                         fig_dpi=100,  fig_dim=10, fig_font_size=18,
                         save_result_figs=False, save_name='', verbose=True):
    """Wrapper for local domain calling """
     
    DomainTools.standard_domain_calling_new(zxy)
