import numpy as np
import multiprocessing as mp
import time
import pandas as pd
# functions
from itertools import combinations_with_replacement, permutations
from scipy.spatial.distance import pdist, squareform, cdist
from tqdm import tqdm
# shared parameters
default_num_threads = 12
# For a pair of chromosome, summarize
def Chr2ZxysList_2_summaryDist_by_key(chr_2_zxys_list, _c1, _c2, codebook_df,
                                 function='nanmedian', axis=0, 
                                 verbose=False):
    _out_dist_dict = {}
    if _c1 != _c2:
        _out_dist_dict[(_c1,_c2)] = []
    else:
        _out_dist_dict[f"cis_{_c1}"] = []
        _out_dist_dict[f"trans_{_c1}"] = []
    for _chr_2_zxys in chr_2_zxys_list:
        # skip if not all info exists
        if _c1 not in _chr_2_zxys or _c2 not in _chr_2_zxys or _chr_2_zxys[_c1] is None or _chr_2_zxys[_c2] is None:
            continue
        else:
            # if not from the same chr label, calcluate trans-chr with cdist
            if _c1 != _c2:
                for _zxys1 in _chr_2_zxys[_c1]:
                    for _zxys2 in _chr_2_zxys[_c2]:
                        _out_dist_dict[(_c1,_c2)].append(cdist(_zxys1, _zxys2))
            # if from the same chr label, calculate both cis and trans
            else:
                # cis
                _out_dist_dict[f"cis_{_c1}"].extend([squareform(pdist(_zxys)) for _zxys in _chr_2_zxys[_c1]])
                # trans
                if len(_chr_2_zxys[_c1]) > 1:
                    # loop through permutations
                    for _i1, _i2 in permutations(np.arange(len(_chr_2_zxys[_c1])), 2):
                        _out_dist_dict[f"trans_{_c1}"].append(
                            cdist(_chr_2_zxys[_c1][_i1], _chr_2_zxys[_c1][_i2])
                        )
    #for _key in _out_dist_dict:
    #    print(_key, len(_out_dist_dict[_key]))
    #return _out_dist_dict
    # summarize
    _summary_dict = {}
    all_chrs = [str(_chr) for _chr in np.unique(codebook_df['chr'])]
    all_chr_sizes = {_chr:np.sum(codebook_df['chr']==_chr) for _chr in all_chrs}
    for _key, _dists_list in _out_dist_dict.items():
        
        if len(_dists_list) > 0:
            # summarize
            if isinstance(function, str):
                _summary_dict[_key] = getattr(np, function)(_dists_list, axis=axis)
            elif callable(function):
                _summary_dict[_key] = function(_dists_list, axis=axis)
            else:
                raise TypeError("Wrong input type for input: function")
        else:
            if isinstance(_key, str): # cis or trans
                _chr = _key.split('_')[-1] 
                _summary_dict[_key]= np.nan * np.ones([all_chr_sizes[_chr], all_chr_sizes[_chr]])
            else:
                _chr1, _chr2 = _key
                _summary_dict[_key]= np.nan * np.ones([all_chr_sizes[_chr1], all_chr_sizes[_chr2]])
    
    return _summary_dict
# call previous function to calculate all pair-wise chromosomal distance
def Chr2ZxysList_2_summaryDict(
    chr_2_zxys_list, total_codebook,
    function='nanmedian', axis=0, 
    parallel=True, num_threads=default_num_threads,
    verbose=False):
    """Function to batch process chr_2_zxys_list into summary_dictionary"""
    if verbose:
        print(f"-- preparing chr_2_zxys from {len(chr_2_zxys_list)} cells", end=' ')
        _start_prepare = time.time()
    _summary_args = []
    # prepare args
    _all_chrs = np.unique(total_codebook['chr'].values)
    #sorted(_all_chrs, key=lambda _c:sort_chr(_c))
    for _chr1, _chr2 in combinations_with_replacement(_all_chrs, 2):
        if _chr1 != _chr2:
            _sel_chr_2_zxys = [
                {_chr1: _d.get(_chr1, None),
                 _chr2: _d.get(_chr2, None)} for _d in chr_2_zxys_list
            ]
        else:
            _sel_chr_2_zxys = [
                {_chr1: _d.get(_chr1, None)} for _d in chr_2_zxys_list
            ]
        _summary_args.append(
            (_sel_chr_2_zxys, _chr1, _chr2, total_codebook, function, axis, verbose)
        )
    if verbose:
        print(f"in {time.time()-_start_prepare:.3f}s.")
    # process
    _start_time = time.time()
    if parallel:
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances with {num_threads} threads", end=' ')
        with mp.Pool(num_threads) as _summary_pool:
            all_summary_dicts = _summary_pool.starmap(
                Chr2ZxysList_2_summaryDist_by_key, 
                _summary_args, chunksize=1)
            _summary_pool.close()
            _summary_pool.join()
            _summary_pool.terminate()
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    else:
        from tqdm import tqdm
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances sequentially", end=' ')
        all_summary_dicts = [Chr2ZxysList_2_summaryDist_by_key(*_args) for _args in tqdm(_summary_args)]
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    # summarize into one dict
    _summary_dict = {}
    for _dict in all_summary_dicts:
        _summary_dict.update(_dict)
    return _summary_dict

# sort chromosome order
def sort_chr(_chr):
    try:
        out_key = int(_chr)
    except:
        if _chr == 'X':
            out_key = 23
        elif _chr == 'Y':
            out_key = 24
    return out_key

# Generate a chromosome plot order by either region id in codebook, or by chromosome
def Generate_PlotOrder(total_codebook, sel_codebook, sort_by_region=True):
    """Function to cleanup plot_order given total codebook and selected codebook"""
    chr_2_plot_indices = {}
    chr_2_chr_orders = {}
    _sel_Nreg = 0
    for _chr in sorted(np.unique(total_codebook['chr']), key=lambda _c:sort_chr(_c)):
        _chr_codebook = total_codebook[total_codebook['chr']==_chr]

        _reg_ids = _chr_codebook['id'].values 
        _orders = _chr_codebook['chr_order'].values
        _chr_sel_inds, _chr_sel_orders = [], []
        for _id, _ord in zip(_reg_ids, _orders):
            if _id in sel_codebook['id'].values:
                _chr_sel_inds.append(sel_codebook[sel_codebook['id']==_id].index[0])
                _chr_sel_orders.append(_ord)
        if len(_chr_sel_inds) == 0:
            continue
        # append
        if sort_by_region:
            chr_2_plot_indices[_chr] = np.array(_chr_sel_inds)
            chr_2_chr_orders[_chr] = np.array(_chr_sel_orders)
        else: # sort by chr
            chr_2_plot_indices[_chr] = np.arange(_sel_Nreg, _sel_Nreg+len(_chr_sel_inds))
            chr_2_chr_orders[_chr] = np.arange(len(_chr_sel_inds))
        # update num of selected regions
        _sel_Nreg += len(_chr_sel_inds)
    return chr_2_plot_indices, chr_2_chr_orders
# Summarize summary_dict into a matrix, using plot order generated by previous function
def assemble_ChrDistDict_2_Matrix(dist_dict, 
                                  total_codebook, sel_codebook=None,
                                  use_cis=True, use_trans=False,
                                  sort_by_region=True):
    """Assemble a dist_dict into distance matrix shape"""
    if sel_codebook is None:
        sel_codebook = total_codebook
    # get indices
    chr_2_plot_inds, chr_2_chr_orders = Generate_PlotOrder(
        total_codebook, sel_codebook, sort_by_region=sort_by_region)
    # init plot matrix
    _matrix = np.ones([len(sel_codebook),len(sel_codebook)]) * np.nan
    # loop through chr
    for _chr1 in sorted(np.unique(total_codebook['chr']), key=lambda _c:sort_chr(_c)):
        for _chr2 in sorted(np.unique(total_codebook['chr']), key=lambda _c:sort_chr(_c)):
            if _chr1 not in chr_2_plot_inds or _chr2 not in chr_2_plot_inds:
                continue
            # get plot_inds
            _ind1, _ind2 = chr_2_plot_inds[_chr1], chr_2_plot_inds[_chr2]
            _ords1, _ords2 = chr_2_chr_orders[_chr1].astype(np.int32), chr_2_chr_orders[_chr2].astype(np.int32)
            # if the same chr, decide using cis/trans
            if _chr1 == _chr2:
                if use_cis and f"cis_{_chr1}" in dist_dict:
                    #print(_ind1, _ind2, _ords1, _ords2)
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[f"cis_{_chr1}"][_ords1[:, np.newaxis], _ords2]
                elif use_trans and f"trans_{_chr1}" in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[f"trans_{_chr1}"][_ords1[:, np.newaxis], _ords2]
                else:
                    continue
            # if not the same chr, get trans_chr_mat
            else:
                if (_chr1, _chr2) in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[(_chr1, _chr2)][_ords1[:, np.newaxis], _ords2]
                    _matrix[_ind2[:, np.newaxis], _ind1] = dist_dict[(_chr1, _chr2)][_ords1[:, np.newaxis], _ords2].transpose()
                elif (_chr2, _chr1) in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[(_chr2, _chr1)][_ords2[:, np.newaxis], _ords1].transpose()
                    _matrix[_ind2[:, np.newaxis], _ind1] = dist_dict[(_chr2, _chr1)][_ords2[:, np.newaxis], _ords1]
    # assemble references
    _chr_edges, _chr_names = generate_plot_chr_edges(sel_codebook, chr_2_plot_inds, sort_by_region)
    return _matrix, _chr_edges, _chr_names



def generate_plot_chr_edges(sel_codebook, chr_2_plot_inds=None, sort_by_region=True):
    if chr_2_plot_inds is None or not isinstance(chr_2_plot_inds, dict):
        chr_2_plot_inds,_ = Generate_PlotOrder(sel_codebook, sel_codebook, sort_by_region=sort_by_region)
    # assemble references
    _chr_edges, _chr_names = [], []
    if sort_by_region:
        # loop through regions
        prev_chr = None
        for _ind, _chr in zip(sel_codebook.index, sel_codebook['chr']):
            if _chr != prev_chr:
                _chr_edges.append(_ind)
                _chr_names.append(_chr)
            prev_chr = _chr
        _chr_edges.append(len(sel_codebook))
    else:
        # loop through chr
        for _chr, _inds in chr_2_plot_inds.items():
            _chr_edges.append(_inds[0])
            _chr_names.append(_chr)
        _chr_edges.append(len(sel_codebook))
    _chr_edges = np.array(_chr_edges)
    return _chr_edges, _chr_names


def contact_prob(mat, contact_th=0.6, axis=0):
    return np.sum(np.array(mat) <= contact_th, axis=axis) / np.sum(np.isfinite(mat), axis=axis)