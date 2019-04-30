import sys,glob,os,time,copy
import numpy as np
import pickle as pickle
import multiprocessing as mp
import psutil
from scipy import ndimage, stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, cdist, squareform
from functools import partial
import matplotlib.pyplot as plt

from . import *
from .External import Fitting_v3, DomainTools

# default init to make it a package
def __init__():
    pass

def Calculate_BED_to_Region(data_filename, region_dic, data_format='tagAlign', stat_type='count', 
                            overwrite=False, save=False, save_folder=None, verbose=True):
    """Function to formulate a BED-like format data alignment result into statistics in given regions
    -------------------------------------------------------------------------------------------------
    Inputs:
        data_filename: full filename for BED-like data, str 
        region_dic: dictionary of regions that have chr,start,end info, dict 
            (For example, dict from get_img_info.Load_Region_Positions)
        data_format: format of this file, str (default: 'tagAlign')
        stat_type: type of statistics to collect, str ({'sum', 'count'}) 
        overwrite: whether overwrite saved stat dict, bool (default: False)
        save: whether save stat_dictionary into a file, bool (default: False)
        save_folder: where to save this result, str (default: None, which means same folder as data)
        verbose: say something!, bool (default: True)
    Outputs:
        _region_stat: region statistics, dict of region id (save from region_dic) 
            to the statistics at this region
    -------------------------------------------------------------------------------------------------
    """
    ## check inputs
    _start_time = time.time()
    # filename
    if not isinstance(data_filename, str):
        raise TypeError(f"Wrong input type for data_filename, should be str, but {type(data_filename)} is given.")
    elif not os.path.isfile(data_filename):
        raise IOError(f"Input data_file: {data_filename} doesn't exist, exit!")
    # region_dic
    if not isinstance(region_dic, dict):
        raise TypeError(f"Wrong input type for region_dic, should be dict, but {type(region_dic)} is given!")
    # data_format:
    _allowed_formats = ['bed', 'tagalign']
    if not isinstance(data_format, str):
        raise TypeError(f"Wrong input type for data_format, should be str, but {type(data_format)} is given.")
    elif data_format.lower() not in _allowed_formats:
        raise ValueError(f"data_format:{data_format} should be among {_allowed_formats}")
    # stat_type
    _allowed_stats = ['count', 'sum']
    if not isinstance(stat_type, str):
        raise TypeError(f"Wrong input type for stat_type, should be str, but {type(stat_type)} is given.")
    elif stat_type.lower() not in _allowed_stats:
        raise ValueError(f"data_format:{stat_type} should be among {_allowed_stats}")
    
    # sort region dic
    _region_info = sorted(region_dic.items())
    
    # not overwrite?
    _save_filename = data_filename.replace(data_format, f'_{stat_type}_region.pkl')
    if save_folder is not None:
        _save_filename = os.path.join(save_folder, os.path.basename(_save_filename))
    if os.path.isfile(_save_filename) and not overwrite:
        _region_stat = pickle.load(open(_save_filename,'rb'))
        return _region_stat
    # otherwise initialize with 0
    else:
        if verbose:
            print(f"-- Calculate {os.path.basename(data_filename)} for {len(_region_info)} regions!")
        _region_stat = {_k:0 for _k in region_dic}

    # load info
    _content = []
    _reg_index = 0
    # loop through lines and add to stat
    if verbose:
        print(f"--- start iterate through data file.")
    with open(data_filename, 'r') as _handle:
        for line in _handle:
            _c = line.strip().split()
            if data_format.lower() == 'tagalign':
                _chr = _c[0]
                _start = int(_c[1])
                _end = int(_c[2])
                _mid = int((_start+_end)/2)
            # go to next region if:
            if _chr != _region_info[_reg_index][1]['chr']:
                continue
            if _mid < _region_info[_reg_index][1]['start']:
                continue
            if _mid > _region_info[_reg_index][1]['end'] and _chr == _region_info[_reg_index][1]['chr']:
                _reg_index += 1
                continue
            elif _mid >= _region_info[_reg_index][1]['start'] \
                and _mid < _region_info[_reg_index][1]['end'] \
                and _chr == _region_info[_reg_index][1]['chr']:
                if stat_type == 'sum':
                    _region_stat[_region_info[_reg_index][0]] += \
                        min(_end, _region_info[_reg_index][1]['end']-1) - \
                        max(_start, _region_info[_reg_index][1]['start'])
                elif stat_type == 'count':
                    _region_stat[_region_info[_reg_index][0]] += 1
            #_content.append(line.strip().split())
    
    if save:
        if verbose:
            print(f"--- save result into file: {_save_filename}")
        pickle.dump(_region_stat, open(_save_filename, 'wb'))    
    
    if verbose:
        print(f"--- time spent in {data_format} loading: {time.time()-_start_time}")
        
    return _region_stat
    
    
## Compartment Analysis
#-----------------------------------------------------------------------------------
def is_in_hull(ref_zxys, zxy):
    """Check if point zxy in ref_zxys
    either zxy or ref_zxys should be 3d ZXY coordinates"""
    if 'ConvexHull' not in locals():
        from scipy.spatial import ConvexHull
    if len(np.shape(zxy)) != 1:
        raise ValueError(f"Wrong input dimension for p, should be 1d")
    # Remove Nan in ref_zxys
    ref_zxys = np.array(ref_zxys) # convert to array
    _kept_rows = np.isnan(ref_zxys).sum(axis=1) == 0
    _kept_ref_zxys = ref_zxys[_kept_rows]
    if len(_kept_ref_zxys) <= 3:
        print('Not enough points to create convex hull.')
        return False
    # create hull for ref_zxys
    _hull = ConvexHull(_kept_ref_zxys)
    # create hull for ref_zxys + zxy
    _extend_zxys = np.concatenate([_kept_ref_zxys, zxy[np.newaxis,:]])
    _extend_hull = ConvexHull(np.array(_extend_zxys))
    if list(_hull.vertices) == list(_extend_hull.vertices):
        return True
    else:
        return False

# basic function to wrap is_in_hull to do bootstrap
def _bootstrap_region_in_domain(_dm_zxys, _reg_zxy, _sampling_size, _n_iter=100):
    if np.isnan(_reg_zxy).any():
        return np.nan
    else:
        _p = []
        for _i in range(_n_iter):
            _bt_inds = np.random.choice(len(_dm_zxys), _sampling_size, replace=False)
            _bt_zxys = _dm_zxys[np.sort(_bt_inds)]
            # check if inhull
            _p.append( is_in_hull(_bt_zxys, _reg_zxy) )
        return np.nanmean(_p)


def Bootstrap_regions_in_domain(chrom_zxy_list, region_index, domain_indices, 
                               p_bootstrap=0.25, n_iter=100, num_threads=12,
                               verbose=True):
    """Estimate how much a region is enclosed by domain/compartments,
        across all chromosomes in chrom_zxy_list
    Inputs:
        chrom_zxy_list: list of chromosome zxy coordinates, list of 2darray
        region_index: index of region in chromosome to be calculated, int
        domain_indices: array-like region indices for a certain domain/compartment, array-like of ints
        p_bootstrap: subsample percentage for bootstap steps, float (default:0.25)
        n_iter: number of bootstrap iterations, int (default: 100)
        verbose: say something!, bool (default: True)
    Outputs:
        _region_probs: proabilities of region in domain, np.ndarray (1d)
        """
    ## check inputs
    _start_time = time.time()
    if not isinstance(chrom_zxy_list, list):
        raise TypeError(f"Wrong input type for chrom_zxy_list, should be list but {type(chrom_zxy_list)} is given.")
    if not isinstance(region_index, int):
        try:
            region_index = int(region_index)
        except:
            raise TypeError(f"Wrong input region_index:{region_index}, cannot convert to int.")
    # convert domain indices into array with ints
    domain_indices = np.array(domain_indices, dtype=np.int)
    if np.max(domain_indices) > len(chrom_zxy_list[0]):
        raise ValueError(f"Wrong input for domain_indices, no indices should be larger than zxy length")
    if verbose:
        print(f"-- Start boostrap sample for region:{region_index} in regions group of {len(domain_indices)}")
    # check p_bootstrap and n_iter
    if not isinstance(p_bootstrap, float):
        p_bootstrap = float(p_bootstrap)
    elif p_bootstrap <= 0 or p_bootstrap >= 1:
        raise ValueError(f"Wrong p_bootstrap={p_bootstrap}, should be float between 0 and 1")
    _sampling_size = int(np.ceil(len(domain_indices)*p_bootstrap))
    if _sampling_size == len(domain_indices):
        _sampling_size -= 1
    if verbose:
        print(f"--- boostrap sampling p={p_bootstrap}, size={_sampling_size}")
    if not isinstance(n_iter, int):
        try:
            n_iter = int(n_iter)
        except:
            raise TypeError(f"Wrong input n_iter:{n_iter}, cannot convert to int.")
    
    ## Start iteration
    _boostrap_args = []
    
    # loop through chromosomes to get args
    for _chrom_zxys in chrom_zxy_list:
        _dm_zxys = np.array(_chrom_zxys)[domain_indices]
        _reg_zxy = np.array(_chrom_zxys)[region_index]
        _boostrap_args.append( (_dm_zxys, _reg_zxy, _sampling_size, n_iter))
    
        #_p = postanalysis._bootstrap_region_in_domain(_dm_zxys,_reg_zxy,_sampling_size,n_iter)
    # calculate through multi-processing
    with mp.Pool(num_threads) as _pool:
        if verbose:
            print(f"--- {len(_boostrap_args)} chromosomes processing by {num_threads} threads.")
        _region_probs = _pool.starmap(_bootstrap_region_in_domain, 
                                      _boostrap_args, chunksize=1)
        _pool.close()
        _pool.join()
        _pool.terminate()    
    if verbose:
        print(f"--- time spent in boostrap: {time.time()-_start_time}")
    return _region_probs