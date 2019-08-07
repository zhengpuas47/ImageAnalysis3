# from packages
import numpy as np
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
        _unique_dxy = np.stack([_dx, _dy]).transpose()
    else:
        _dxy = [(min(_x,_y),max(_x,_y)) for _x,_y in zip(_dx, _dy)]
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
                
    _marker_map = _generate_inter_domain_markers(_coordinates, _domain_starts, _dm_pds, _unique_dxy,
                                                 _marker_type=marker_type, _marker_param=marker_param, 
                                                 _keep_intensity=keep_intensity)
    
    return _marker_map

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
        return _marker_map
    else:
        # get intensities
        if _keep_intensity:
            _intensities = np.abs(squareform(_domain_pdists)[_domain_xy[:,0], _domain_xy[:,1]])
        else:
            _intensities = np.ones(len(_domain_xy))
        if _marker_type is 'center':
            for _dxy, _int in zip(_domain_xy, _intensities):
                _marker_map[_dm_centers[_dxy[0]], _dm_centers[_dxy[1]]] = _int
        elif _marker_type is 'gaussian':      
            for _dxy, _int in zip(_domain_xy, _intensities):
                _marker_map = visual_tools.add_source(_marker_map, pos=[_dm_centers[_dxy[0]], _dm_centers[_dxy[1]]], 
                                                      h=_int, sig=[_marker_param,_marker_param])
                _marker_map[_dm_centers[_dxy[0]], _dm_centers[_dxy[1]]] = _int
        elif _marker_type is 'area':
            for _dxy, _int in zip(_domain_xy, _intensities):
                _area_slice = tuple(slice(_dm_starts[_d],_dm_ends[_d]) for _d in _dxy)
                _marker_map[_area_slice] = _int
                
    return _marker_map



def _loop_out_metric(_coordinates, _position, _domain_starts, metric='median', loop_out_th=0.):
    _dm_starts = np.array(_domain_starts)
    _dm_ends = np.concatenate([_dm_starts[1:], np.array([len(_coordinates)])])
    # position
    _pos = int(_position)
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
            if metric == 'median':
                m_dist, m_self = np.median(_dists), np.median(_self_dists)
                v_dist, v_self = np.median((_dists-m_dist)**2),\
                                   np.median((_self_dists-m_self)**2)
                _d = (m_dist-m_self) / np.sqrt(v_dist+v_self)
            elif metric == 'mean':
                m_dist, m_self = np.mean(_dists), np.mean(_self_dists)
                v_dist, v_self = np.var(_dists), np.var(_self_dists)
                _d = (m_dist-m_self) / np.sqrt(v_dist+v_self)
            elif metric == 'ks':
                if 'ks_2samp' not in locals():
                    from scipy.stats import ks_2samp
                _f = np.sign((np.median(_dists) - np.median(_self_dists)))
                _d = _f * ks_2samp(_self_dists, _dists)[0]
            else:
                raise ValueError(f"unsupported metric:{metric}")
            if _d < loop_out_th:
                _loop_out_hits.append(_i)
                
        return _loop_out_hits
                
