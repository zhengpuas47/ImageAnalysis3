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
    
    