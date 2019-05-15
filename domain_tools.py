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
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, is_valid_linkage


def nan_gaussian_filter(mat, sigma, keep_nan=False):
    from scipy.ndimage import gaussian_filter
    U = np.array(mat)
    Unan = np.isnan(U)
    V = U.copy()
    V[U != U] = 0
    VV = gaussian_filter(V, sigma=sigma)

    W = 0*U.copy()+1
    W[U != U] = 0
    WW = gaussian_filter(W, sigma=sigma)

    Z = VV/WW
    if keep_nan:
        Z[Unan] = np.nan
    return Z


def interp1dnan(A):
    A_ = np.array(A)
    ok = np.isnan(A) == False
    xp = ok.nonzero()[0]
    fp = A[ok]
    x = np.isnan(A).nonzero()[0]
    A_[np.isnan(A)] = np.interp(x, xp, fp)
    return A_

def interpolate_chr(_chr, gaussian=0):
    """linear interpolate chromosome coordinates"""
    _chr = np.array(_chr).copy()
    for i in range(_chr.shape[-1]):
        if gaussian > 0:
            _chr[:, i] = nan_gaussian_filter(_chr[:, i], gaussian)
    # interpolate
    from scipy.interpolate import interp1d
    not_nan_inds = np.where(np.isnan(_chr).sum(1) == 0)[0]
    if len(not_nan_inds) == 0:
        return _chr
    else:
        f = interp1d(np.arange(len(_chr))[not_nan_inds], _chr[not_nan_inds],
                     kind='linear', axis=0, bounds_error=False,
                     fill_value='extrapolate')
        _interpolated_chr = f(np.arange(len(_chr)))
        return _interpolated_chr




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


def domain_distance(coordinates, _dom1_bds, _dom2_bds,
                    _measure='median', _normalization_mat=None ):
    """Function to measure domain distances between two zxy arrays
    use KS-statistic as a distance:
        citation: https://arxiv.org/abs/1711.00761"""
    ## check inputs
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    _measure = str(_measure).lower()
    _dom1_bds = [int(_b) for _b in _dom1_bds]
    _dom2_bds = [int(_b) for _b in _dom2_bds]

    # based on coordinates given, get intra/inter distances
    if len(np.shape(coordinates)) != 2:
        raise ValueError(f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        _mat = coordinates
        _intra1 = _mat[_dom1_bds[0]:_dom1_bds[1], _dom1_bds[0]:_dom1_bds[1]]
        _intra2 = _mat[_dom2_bds[0]:_dom2_bds[1], _dom2_bds[0]:_dom2_bds[1]]
        _intra1 = _intra1[np.triu_indices(len(_intra1),1)]
        _intra2 = _intra2[np.triu_indices(len(_intra2), 1)]

        _intra_dist = [_intra1, _intra2]
        _inter_dist = np.ravel(_mat[_dom1_bds[0]:_dom1_bds[1], _dom2_bds[0]:_dom2_bds[1]])

    elif np.shape(coordinates)[1] == 3:
        # extract sequence
        zxy1 = coordinates[_dom1_bds[0]:_dom1_bds[1]]
        zxy2 = coordinates[_dom2_bds[0]:_dom2_bds[1]]
        # get distances
        _intra_dist = [pdist(zxy1), pdist(zxy2)]
        _inter_dist = np.ravel(cdist(zxy1, zxy2))
    else:
        raise ValueError(f"Input coordinates should be distance-matrix or 3d-coordinates!")

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
        return 0

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
        _f = int((np.nanmedian(_inter_dist) - np.nanmedian(_intra_dist)) > 0)
        return _f * ks_2samp(_kept_inter, _kept_intra)[0]

# function to call domain pairwise distance as scipy.spatial.distance.pdist
def domain_pdists(coordinates, domain_starts, metric='median', normalization_mat=None):
    """Calculate domain pair-wise distances, return a vector as the same order as
    scipy.spatial.distance.pdist 
    Inputs:
        coordinates:
        """
    ## check inputs
    coordinates = np.array(coordinates).copy()
    if len(np.shape(coordinates)) != 2:
        raise ValueError(
            f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        _mat = coordinates
    elif np.shape(coordinates)[1] == 3:
        _mat = squareform(pdist(coordinates))
    else:
        raise ValueError(
            f"Input coordinates should be distance-matrix or 3d-coordinates!")

    domain_starts = np.array(domain_starts, dtype=np.int)
    domain_ends = np.zeros(np.shape(domain_starts))
    domain_ends[:-1] = domain_starts[1:]
    domain_ends[-1] = len(coordinates)

    # first check whether do normalzation
    if normalization_mat is not None:
        if normalization_mat.shape[0] != _mat.shape[0] or normalization_mat.shape[1] != _mat.shape[1]:
            raise ValueError(
                f"Wrong shape of normalization:{normalization_mat.shape}, should be equal to {_mat.shape}")

    dom_pdists = []
    for _i in range(len(domain_starts)):
        for _j in range(_i+1, len(domain_starts)):
            dom_pdists.append(domain_distance(coordinates,
                                              [domain_starts[_i], domain_ends[_i]],
                                              [domain_starts[_j], domain_ends[_j]],
                                              _measure=metric,
                                              _normalization_mat=normalization_mat))

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


def merge_domains(coordinates, cand_bd_starts, norm_mat=None, 
                  corr_th=0.64, dist_th=0.2,
                  domain_dist_metric='ks', plot_steps=False, verbose=True):
    """Function to merge domains given zxy coordinates and candidate_boundaries"""
    cand_bd_starts = np.array(cand_bd_starts, dtype=np.int)
    _merge_inds = np.array([-1])
    if verbose:
        print(
            f"-- start iterate to merge domains, num of candidates:{len(cand_bd_starts)}")
    # start iteration:
    while len(_merge_inds) > 0 and len(cand_bd_starts) > 1:
        # calculate domain pairwise-distances
        _dm_pdists = domain_pdists(coordinates, cand_bd_starts,
                                   metric=domain_dist_metric, 
                                   normalization_mat=norm_mat)
        _dm_dist_mat = squareform(_dm_pdists)
        # remove domain candidates that are purely nans
        _keep_inds = np.isnan(_dm_dist_mat).sum(1) < len(_dm_dist_mat)-1
        if np.sum(_keep_inds) != len(cand_bd_starts) and verbose:
            print(f"---** remove {len(cand_bd_starts)-np.sum(_keep_inds)} domains because of NaNs")
        _dm_dist_mat = _dm_dist_mat[_keep_inds][:,_keep_inds]
        cand_bd_starts = cand_bd_starts[_keep_inds]
        # calculate correlation coefficient matrix
        _coef_mat = np.corrcoef(_dm_dist_mat)
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
        _dist_inds = np.where(np.diag(_dm_dist_mat, 1) <= dist_th)[0]+1
        if len(_dist_inds) > 0 and len(_corr_inds) > 0:
            _merge_inds = [int(_cid) for _cid in _corr_inds if _cid in _dist_inds]
        else:
            _merge_inds = []
        # if there are any domain to be merged:
        if len(_merge_inds) > 0:
            # find the index with minimum distance bewteen neighboring domains
            # first merge neighbors with high corr and low dist
            _merge_dists = (1-np.diag(_dm_dist_mat, 1)) / (1-dist_th) + \
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
            f"--- final neighbor domain dists:{np.diag(_dm_dist_mat,1)}")
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
        norm_mat = None
        _normalization = False

    ## 0. prepare coordinates
    if verbose:
        print(f"-- start basic domain calling")
        _start_time = time.time()
    # get zxy
    _spots = np.array(spots)
    if _spots.shape[1] == 3:
        _zxy = _spots
    else:
        _zxy = np.array(_spots)[:, 1:4] * distance_zxy[np.newaxis, :]
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
    cand_bd_starts = merge_domains(_zxy, cand_bd_starts=cand_bd_starts,
                                   norm_mat=norm_mat, corr_th=corr_th, dist_th=dist_th,
                                   domain_dist_metric=domain_dist_metric,
                                   plot_steps=plot_steps, verbose=verbose)

    ## 3. finish up and make plot
    if plot_results and len(cand_bd_starts) > 1:
        _dm_pdists = domain_pdists(_zxy, cand_bd_starts,
                                   metric=domain_dist_metric,
                                   normalization_mat=norm_mat)
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
        norm_mat = None 
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
    # get zxy
    _spots = np.array(spots)
    if _spots.shape[1] == 3:
        _zxy = _spots
    else:
        _zxy = np.array(_spots)[:, 1:4] * distance_zxy[np.newaxis, :]
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
        _dm_pdists = domain_pdists(_zxy, cand_bd_starts,
                                   metric=domain_dist_metric,
                                   normalization_mat=norm_mat)
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
    """Wrapper for local domain calling in bogdan's code"""
    from .External.DomainTools import standard_domain_calling_new
    ## 0. prepare coordinates
    if verbose:
        print(f"- start local domain calling")
        _start_time = time.time()
    _spots = np.array(spots)
    if _spots.shape[1] == 3:
        _zxy = _spots
    else:
        _zxy = np.array(_spots)[:, 1:4] * distance_zxy[np.newaxis, :]
    # smooth
    if gfilt_size is not None and gfilt_size > 0:
        if verbose:
            print(f"--- gaussian interpolate chromosome, sigma={gfilt_size}")
        _zxy = DomainTools.interpolate_chr(_zxy, gaussian=gfilt_size)
    # call bogdan's function
    cand_bd_starts =  standard_domain_calling_new(_zxy, gaussian=0., 
                                                  dom_sz=dom_sz, 
                                                  cutoff_max=cutoff_max)

    return cand_bd_starts

# Calculate distance given distance_matrix, window_size and metric type
def _sliding_window_dist(_mat, _wd, _dist_metric='median'):
    """Function to calculate sliding-window distance across one distance-map of chromosome"""
    dists = []
    for _i in range(len(_mat)):
        if _i - _wd < 0 or _i + _wd > len(_mat):
            dists.append(0)
        else:
            _crop_mat = _mat[_i-_wd:_i+_wd, _i-_wd:_i+_wd]
            _intra1 = np.triu(_crop_mat[:_wd, :_wd], 1)
            _intra1 = _intra1[np.isnan(_intra1)==False]
            _intra2 = np.triu(_crop_mat[_wd:, _wd:], 1)
            _intra2 = _intra2[np.isnan(_intra2)==False]
            _intra_dist = np.concatenate([_intra1[_intra1 > 0],
                                          _intra2[_intra2 > 0]])
            _inter_dist = _crop_mat[_wd:, :_wd]
            _inter_dist = _inter_dist[np.isnan(_inter_dist) == False]
            if len(_intra_dist) == 0 or len(_inter_dist) == 0:
                # return zero distance if one dist list is empty
                dists.append(0)
            # add dist info
            if _dist_metric == 'ks':
                if 'ks_2samp' not in locals():
                    from scipy.stats import ks_2samp
                _f = int((np.median(_inter_dist) - np.median(_intra_dist)) > 0)
                dists.append(_f * ks_2samp(_intra_dist, _inter_dist)[0])
            elif _dist_metric == 'median':
                m_inter, m_intra = np.median(_inter_dist), np.median(_intra_dist)
                v_inter, v_intra = np.median((_inter_dist-m_inter)**2),\
                                   np.median((_intra_dist-m_intra)**2)
                dists.append((m_inter-m_intra)/np.sqrt(v_inter+v_intra))
            elif _dist_metric == 'mean':
                m_inter, m_intra = np.mean(_inter_dist), np.mean(_intra_dist)
                v_inter, v_intra = np.var(_inter_dist), np.var(_intra_dist)
                dists.append((m_inter-m_intra)/np.sqrt(v_inter+v_intra))
    dists = np.array(dists)
    dists[dists<0] = 0

    return dists


def Domain_Calling_Sliding_Window(coordinates, window_size=5, distance_metric='median',
                                  gaussian=0, normalization=r'Z:\References\normalization_matrix.npy',
                                  min_domain_size=4, min_prominence=0.25, reproduce_ratio=0.6,
                                  merge_candidates=True, corr_th=0.6, dist_th=0.2,
                                  merge_strength_th=1., return_strength=False,
                                  verbose=False):
    """Function to call domain candidates by sliding window across chromosome
    Inputs:
        coordnates: n-by-3 coordinates for a chromosome, or n-by-n distance matrix, np.ndarray
        window_size: size of sliding window for each half, the exact windows will be 1x to 2x of size, int
        distance_metric: type in distance metric in each sliding window, 
        gaussian: size of gaussian filter applied to coordinates, float (default: 0, no gaussian)
        normalization: normalization matrix / path to normalization matrix, np.ndarray or str
        min_domain_size: minimal domain size allowed in calling, int (default: 4)
        min_prominence: minimum prominence of peaks in distances called by sliding window, float (default: 0.25)
        reproduce_ratio: ratio of peaks found near the candidates across different window size, float (default: 0.6)
        merge_candidates: wheather merge candidate domains, bool (default:True)
        corr_th: min corrcoef threshold to merge domains, float (default: 0.6)
        dist_th: max distance threshold to merge domains, float (defaul: 0.2)
        merge_strength_th: min strength to not merge at all, float (default: 1.)     
        return_strength: return boundary strength generated sliding_window, bool (default: False)
        verbose: say something!, bool (default: False)
    Outputs:
        kept_domains: domain starts region-indices, np.ndarray
        kept_strengths (optional): kept domain boundary strength, np.ndarray
    """
    ## check inputs
    coordinates = np.array(coordinates).copy()
    if verbose:
        print(f"-- start sliding-window domain calling with", end=' ')
    if len(np.shape(coordinates)) != 2:
        raise ValueError(
            f"Wrong input shape for coordinates, should be 2d but {len(np.shape(coordinates))} is given")
    elif np.shape(coordinates)[0] == np.shape(coordinates)[1]:
        if verbose:
            print(f"distance map")
        if gaussian > 0:
            from astropy.convolution import Gaussian2DKernel, convolve
            _kernel = _kernel = Gaussian2DKernel(x_stddev=gaussian)
            coordinates = convolve(coordinates, _kernel)
        _mat = coordinates
    elif np.shape(coordinates)[1] == 3:
        if verbose:
            print(f"3d coordinates")
        if gaussian > 0:
            coordinates = interpolate_chr(coordinates, gaussian=gaussian)
        _mat = squareform(pdist(coordinates))
    else:
        raise ValueError(
            f"Input coordinates should be distance-matrix or 3d-coordinates!")
    window_size = int(window_size)

    # load normalization if specified
    if isinstance(normalization, str) and os.path.isfile(normalization):
        normalization = np.load(normalization)
    elif isinstance(normalization, np.ndarray) and np.shape(_mat)[0] == np.shape(normalization)[0]:
        pass
    else:
        normalization = None
    # do normalization if satisfied
    if normalization is not None:
        if verbose:
            print(f"--- applying normalization")
        _mat = _mat / normalization
    ## Start slide window to generate a vector for distance
    dist_list = []
    if verbose:
        print(
            f"--- calcualte distances with sliding window from {window_size} to {2*window_size-1}")
    # loop through real window_size between 1x to 2x of window_size
    for _wd in np.arange(window_size, 2*window_size):
        dist_list.append(_sliding_window_dist(_mat, _wd, distance_metric))

    ## call peaks
    if verbose:
        print(
            f"--- call peaks with minimum domain size={min_domain_size}, prominence={min_prominence}")
    peak_list = [scipy.signal.find_peaks(_dists, distance=min_domain_size,
                                         prominence=(min_prominence, None))[0] for _dists in dist_list]

    ## summarize peaks
    if verbose:
        print(
            f"--- summarize domains with {reproduce_ratio} reproducing rate.")
    cand_peaks = peak_list[0]
    _peak_coords = np.ones([len(peak_list), len(cand_peaks)]) * np.nan
    _peak_coords[0] = peak_list[0]
    _r = int(np.ceil(min_domain_size/2))
    for _i, _peaks in enumerate(peak_list[1:]):
        for _j, _p in enumerate(cand_peaks):
            _matched_index = np.where((_peaks >= _p-_r) * (_peaks <= _p+_r))[0]
            if len(_matched_index) > 0:
                # record whether have corresponding peaks in other window-size cases
                _peak_coords[_i+1, _j] = _peaks[_matched_index[0]]

    # only select peaks which showed up in more than reproduce_ratio*number_of_window cases
    _keep_flag = np.sum((np.isnan(_peak_coords) == False).astype(
        np.int), axis=0) >= reproduce_ratio*len(peak_list)
    # summarize selected peaks by mean of all summarized peaks
    sel_peaks = np.round(np.nanmean(_peak_coords, axis=0)
                         ).astype(np.int)[_keep_flag]
    # concatenate a zero
    domain_starts = np.concatenate([np.array([0]), sel_peaks])
    # calculate strength
    _strengths = np.nanmean([_dists[domain_starts]
                             for _dists in dist_list], axis=0)
    if verbose:
        print(f"--- domain called by sliding-window: {len(domain_starts)}")

    if merge_candidates:
        merged_starts = merge_domains(coordinates, domain_starts, 
                                      norm_mat=normalization, corr_th=corr_th,
                                      dist_th=dist_th, domain_dist_metric=distance_metric,
                                      plot_steps=False, verbose=False)    
    kept_domains = np.array([_d for _i,_d in enumerate(domain_starts)  
                             if _d in merged_starts or _strengths[_i] > merge_strength_th])
    if verbose:
        print(f"--- domain after merging: {len(kept_domains)}")
    # return_strength
    if return_strength:
        kept_strengths = np.array([_s for _i, _s in enumerate(_strengths)
                                    if domain_starts[_i] in merged_starts or _s > merge_strength_th])
        return kept_domains.astype(np.int), kept_strengths
    else:
        return kept_domains.astype(np.int)


def Batch_Domain_Calling_Sliding_Window(coordinate_list, window_size=5, distance_metric='median',
                                        num_threads=12, gaussian=0,
                                        normalization=r'Z:\References\normalization_matrix.npy',
                                        min_domain_size=4, min_prominence=0.25, reproduce_ratio=0.6,
                                        merge_candidates=True, corr_th=0.8, dist_th=0.2,
                                        merge_strength_th=1., return_strength=False,
                                        verbose=False):
    """Function to call domain candidates by sliding window across chromosome
    Inputs:
        coordinate_list: list of coordinates:
            n-by-3 coordinates for a chromosome, or n-by-n distance matrix, np.ndarray
        window_size: size of sliding window for each half, the exact windows will be 1x to 2x of size, int
        distance_metric: type in distance metric in each sliding window, 
        num_threads: number of threads to multiprocess domain calling, int (default: 12)
        gaussian: size of gaussian filter applied to coordinates, float (default: 0, no gaussian)
        normalization: normalization matrix / path to normalization matrix, np.ndarray or str
        min_domain_size: minimal domain size allowed in calling, int (default: 4)
        min_prominence: minimum prominence of peaks in distances called by sliding window, float (default: 0.25)
        reproduce_ratio: ratio of peaks found near the candidates across different window size, float (default: 0.6)
        merge_candidates: wheather merge candidate domains, bool (default:True)
        corr_th: min corrcoef threshold to merge domains, float (default: 0.6)
        dist_th: max distance threshold to merge domains, float (defaul: 0.2)
        merge_strength_th: min strength to not merge at all, float (default: 1.)     
        return_strength: return boundary strength generated sliding_window, bool (default: False)
        verbose: say something!, bool (default: False)
    Outputs:
        domain_start_list: list of domain start indices:
            domain starts region-indices, np.ndarray
        strength_list: list of strengths:
            (optional): kept domain boundary strength, np.ndarray
    """
    ## inputs
    if verbose:
        _start_time = time.time()
        print(f"- Start batch domain calling with sliding window.")
    # check coordinate_list
    if isinstance(coordinate_list, list) and len(np.shape(coordinate_list[0]))==2:
        pass
    elif isinstance(coordinate_list, np.ndarray) and len(np.shape(coordinate_list)) == 3:
        pass
    else:
        raise ValueError(f"Input coordinate_list should be a list of 2darray or 3dim merged coordinates")
    # load normalization if specified
    if isinstance(normalization, str) and os.path.isfile(normalization):
        normalization = np.load(normalization)
    elif isinstance(normalization, np.ndarray) and len(coordinate_list[0]) == np.shape(normalization)[0]:
        pass
    else:
        normalization = None
    # num_threads
    num_threads = int(num_threads)
    ## init
    domain_args = []
    # loop through coordinates
    for coordinates in coordinate_list:
        domain_args.append((coordinates, window_size, distance_metric,
                            gaussian, normalization,
                            min_domain_size, min_prominence, reproduce_ratio,
                            merge_candidates, corr_th, dist_th,
                            merge_strength_th, True, verbose))
    # multi-processing
    if verbose:
        print(
            f"-- multiprocessing of {len(domain_args)} domain calling with {num_threads} threads")
    with mp.Pool(num_threads) as domain_pool:
        results = domain_pool.starmap(
            Domain_Calling_Sliding_Window, domain_args)
        domain_pool.close()
        domain_pool.join()
        domain_pool.terminate()
    domain_start_list = [_r[0] for _r in results]

    if verbose:
        print(f"-- time spent in domain-calling:{time.time()-_start_time}")

    if return_strength:
        strength_list = [_r[1] for _r in results]
        return domain_start_list, strength_list
    else:
        return domain_start_list



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
