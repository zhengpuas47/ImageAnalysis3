# from packages
import numpy as np
import time
from scipy.spatial.distance import cdist, pdist, squareform
# from local
from .distance import domain_pdists
from .. import visual_tools

# mark off-diagonal features
def inter_domain_markers(coordiantes, domain_starts, norm_mat=None, metric='median', 
                         off_diagonal_th=-0.5, keep_triu=False, 
                         marker_type='center', marker_param=1., keep_intensity=True, 
                         exclude_neighbors=True, exclude_edges=True, verbose=True):
    """Get off-diagonal markers for a given coordinates
    Inputs:
        
    Outouts:
    
    """
    ## check inputs
    _coordinates = np.array(coordiantes)
    _domain_starts = np.array(domain_starts, dtype=np.int)
    if len(_coordinates.shape) != 2:
        raise IndexError(f"Wrong input shape for coordinates, it should be 2d-array but shape:{_coordinates.shape} is given.")
    
    
    _dm_pds = domain_pdists(_coordinates, _domain_starts, metric=metric, 
                            normalization_mat=norm_mat, allow_minus_dist=True)

    _dx, _dy = np.where(squareform(_dm_pds) < off_diagonal_th)
    if not keep_triu:
        _dx, _dy = _dx[_dx!=_dy], _dy[_dx!=_dy]
        _unique_dxy = np.stack([_dx, _dy]).transpose()

    else:
        _dxy = [(min(_x,_y),max(_x,_y)) for _x,_y in zip(_dx, _dy) if min(_x,_y) != max(_x,_y)]
        _unique_dxy = []
        for _xy in _dxy:
            if _xy not in _unique_dxy:
                _unique_dxy.append(_xy)
        _unique_dxy = np.array(_unique_dxy)
    if exclude_neighbors:
        _kept_dxy = []
        for _dxy in _unique_dxy:
            if np.abs(_dxy[0] - _dxy[1]) > 1:
                _kept_dxy.append(_dxy)
        _unique_dxy = np.array(_kept_dxy)
    if exclude_edges:
        _kept_dxy = []
        for _dxy in _unique_dxy:
            if 0 not in _dxy and len(_domain_starts)-1 not in _dxy:
                _kept_dxy.append(_dxy)
        _unique_dxy = np.array(_kept_dxy)
    # generate markers  
    _marker_map = _generate_inter_domain_markers(_coordinates, _domain_starts, _dm_pds, _unique_dxy,
                                                 _marker_type=marker_type, _marker_param=marker_param, 
                                                 _keep_intensity=keep_intensity)
    return _unique_dxy, _marker_map

def _generate_inter_domain_markers(_coordinates, _domain_starts, _domain_pdists, _domain_xy, 
                                   _marker_type='center', _marker_param=1., _keep_intensity=True):
    """transform domain_xy into marker format"""
    # get domain features
    _dm_starts = np.array(_domain_starts)
    _dm_ends = np.concatenate([_dm_starts[1:], np.array([len(_coordinates)])])
    _dm_centers = ((_dm_starts + _dm_ends)/2).astype(np.int)
    _domain_xy = np.array(_domain_xy, dtype=np.int)
    # initialize marker-map
    _marker_map = np.zeros([len(_coordinates), len(_coordinates)])
    if len(_domain_xy) == 0:
        print("empty_map")
        return _marker_map
    else:
        # get intensities
        if _keep_intensity:
            _intensities = np.abs(squareform(_domain_pdists)[_domain_xy[:,0], _domain_xy[:,1]])
        else:
            _intensities = np.ones(len(_domain_xy))

        if _marker_type == 'center':
            for _dxy, _int in zip(_domain_xy, _intensities):
                _marker_map[_dm_centers[_dxy[0]], _dm_centers[_dxy[1]]] = _int
        elif _marker_type == 'gaussian':      
            for _dxy, _int in zip(_domain_xy, _intensities):
                _marker_map = visual_tools.add_source(_marker_map, pos=[_dm_centers[_dxy[0]], _dm_centers[_dxy[1]]], 
                                                      h=_int, sig=[_marker_param,_marker_param])
        elif _marker_type == 'area':
            for _dxy, _int in zip(_domain_xy, _intensities):
                _area_slice = tuple(slice(_dm_starts[_d],_dm_ends[_d]) for _d in _dxy)
                _marker_map[_area_slice] = _int
        else:
            raise ValueError(f"Wrong input for _marker_type:{_marker_type}")
    return _marker_map

def _loop_out_metric(_coordinates, _position, _domain_starts, metric='median', 
                     _loop_out_th=0., _exclude_boundaries=True, _exclude_edges=True):
    _dm_starts = np.array(_domain_starts)
    _dm_ends = np.concatenate([_dm_starts[1:], np.array([len(_coordinates)])])
    # position
    _pos = int(_position)
    # exclude domain boudaries
    if _exclude_boundaries:
        if _pos in _dm_starts:
            return []
    # exclude edges if specified
    if _exclude_edges:
        if _pos > _dm_starts[-1] or _pos < _dm_ends[0]:
            return []
    # initialize
    _self_dists = []
    _self_id = -1
    _dist_list = []
    for _start, _end in zip(_dm_starts, _dm_ends):
        # calculate the metric
        _dist_list.append(_coordinates[_start:_end, _pos])
        if _position > _start and _position < _end:
            _self_dists.append(_coordinates[_start:_end, _pos])
            
    if len(_self_dists) != 1:
        return []
    else:
        _loop_out_hits = []
        _self_dists = _self_dists[0]
        for _i, _dists in enumerate(_dist_list):
            if np.isnan(_dists).sum() == np.size(_dists):
                _d = np.inf
            else:
                if metric == 'median':
                    m_dist, m_self = np.nanmedian(_dists), np.nanmedian(_self_dists)
                    v_dist, v_self = np.nanmedian((_dists-m_dist)**2),\
                                    np.nanmedian((_self_dists-m_self)**2)
                    if v_dist+v_self == 0:
                        _d = np.inf
                    else:
                        _d = (m_dist-m_self) / np.sqrt(v_dist+v_self)
                elif metric == 'mean':
                    m_dist, m_self = np.nanmean(_dists), np.nanmean(_self_dists)
                    v_dist, v_self = np.nanvar(_dists), np.nanvar(_self_dists)
                    if v_dist+v_self == 0:
                        _d = np.inf
                    else:
                        _d = (m_dist-m_self) / np.sqrt(v_dist+v_self)                
                elif metric == 'ks':
                    if 'ks_2samp' not in locals():
                        from scipy.stats import ks_2samp
                    _f = np.sign((np.nanmedian(_dists) - np.nanmedian(_self_dists)))
                    _d = _f * ks_2samp(_self_dists, _dists)[0]
                else:
                    raise ValueError(f"unsupported metric:{metric}")
            # decide if its a hit
            if _d < _loop_out_th:
                _loop_out_hits.append(_i)
                
        return _loop_out_hits
                
def _generate_loop_out_markers(_coordinates, _domain_starts, _loop_regions, _loop_domains, 
                               _marker_type='center', _marker_param=1., _keep_triu=True, 
                               _verbose=True):
    """transform domain_xy into marker format"""
    # get domain features
    _dm_starts = np.array(_domain_starts)
    _dm_ends = np.concatenate([_dm_starts[1:], np.array([len(_coordinates)])])
    _dm_centers = ((_dm_starts + _dm_ends)/2).astype(np.int)
    _loop_regions = np.array(_loop_regions, dtype=np.int)
    _loop_domains = np.array(_loop_domains, dtype=np.int)
    # initialize marker-map
    _marker_map = np.zeros([len(_coordinates), len(_coordinates)])
    if len(_loop_regions) == 0 or len(_loop_domains)==0:
        if _verbose:
            print(f"---- no loop given, return empty marker map.")
        return _marker_map
    else:
        if _verbose:
            print(f"--- generate loop-out marker for {len(_loop_regions)} loops")
        
        if _marker_type == 'center':
            for _reg, _dm in zip(_loop_regions, _loop_domains):
                _marker_map[_reg, _dm_centers[_dm]] = 1
                if not _keep_triu:
                    _marker_map[_dm_centers[_dm], _reg] = 1
        elif _marker_type == 'gaussian':      
            for _reg, _dm in zip(_loop_regions, _loop_domains):
                _marker_map = visual_tools.add_source(_marker_map, pos=[_reg, _dm_centers[_dm]], 
                                                          h=1, sig=[_marker_param,_marker_param])
                if not _keep_triu:
                    _marker_map = visual_tools.add_source(_marker_map, pos=[_dm_centers[_dm], _reg], 
                                                          h=1, sig=[_marker_param,_marker_param])
        elif _marker_type == 'area':
            for _reg, _dm in zip(_loop_regions, _loop_domains):
                _marker_map[_reg, _dm_starts[_dm]:_dm_ends[_dm]] = 1
                if not _keep_triu:
                    _marker_map[_dm_starts[_dm]:_dm_ends[_dm], _reg] = 1
        else:
            raise ValueError(f"Wrong input _marker_type:{_marker_type}")
    return _marker_map

    
def loop_out_markers(coordinates, domain_starts, norm_mat=None, metric='median',
                     loop_out_th=0., marker_type='center', marker_param=1., keep_triu=True,
                     exclude_boundaries=True, exclude_edges=True, verbose=True):
    ## check inputs
    _coordinates = np.array(coordinates)
    _domain_starts = np.array(domain_starts, dtype=np.int)
    if verbose:
        _start_time = time.time()
        print(f"-- calculate loop-out for {len(_domain_starts)} domains in {len(_coordinates)} regions")
    if len(_coordinates.shape) != 2:
        raise IndexError(f"Wrong input shape for coordinates, it should be 2d-array but shape:{_coordinates.shape} is given.")
    if _coordinates.shape[1] == 3: # in zxy format
        _coordinates = squareform(pdist(_coordinates))
    
    # initialze
    _loop_regions = []
    _loop_domains = []
    for _pos in range(len(_coordinates)):
        _loop_dms = _loop_out_metric(_coordinates, _pos, domain_starts, 
                          _loop_out_th=loop_out_th, metric=metric, 
                          _exclude_boundaries=exclude_boundaries, _exclude_edges=exclude_edges)
        if len(_loop_dms) > 0:
            _loop_regions += [_pos]*len(_loop_dms)
            _loop_domains += _loop_dms
    # generate marker
    _loop_marker = _generate_loop_out_markers(_coordinates, _domain_starts, 
                                              _loop_regions, _loop_domains, _marker_type=marker_type,
                                              _marker_param=marker_param, _keep_triu=keep_triu, _verbose=verbose)
    if verbose:
        print(f"--- {len(_loop_regions)} loops identified, time:{time.time()-_start_time:2.3}")
    
    return _loop_regions, _loop_marker