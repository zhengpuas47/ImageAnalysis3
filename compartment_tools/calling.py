import numpy as np
from scipy.spatial import ConvexHull
## Compartment Analysis
#-----------------------------------------------------------------------------------

# basic function to determine if a point is in convex hull or not
def is_in_hull(ref_zxys, zxy, remove_vertices=True):
    """Check if point zxy in ref_zxys
    either zxy or ref_zxys should be 3d ZXY coordinates"""
    if len(np.shape(zxy)) != 1:
        raise ValueError(f"Wrong input dimension for p, should be 1d")

    # Remove Nan in ref_zxys
    ref_zxys = np.array(ref_zxys) # convert to array
    _kept_rows = np.isnan(ref_zxys).sum(axis=1) == 0
    # remove itself
    if remove_vertices:
        for _i, _ref_zxy in enumerate(ref_zxys):
            if (_ref_zxy == np.array(zxy)).all():
                _kept_rows[_i] = False
    # apply remove
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
def _bootstrap_region_in_domain(_dm_zxys, _reg_zxy, 
                                _sampling_indices=None, 
                                _sampling_size=None, 
                                _n_iter=100, _remove_self=True):
    ## check inputs
    if np.isnan(_reg_zxy).any():
        return np.nan
    _dm_zxys = np.array(_dm_zxys)
    # determine if _reg_zxy is within _dm_zxys
    self_id = -1
    if _remove_self:
        for _i, _zxy in enumerate(_dm_zxys):
            if (_zxy == np.array(_reg_zxy)).all():
                self_id = _i
                break
    # if sampling indices directly given, do a modification
    if _sampling_indices is not None:
        _n_iter = len(_sampling_indices)
        _sampling_indices = np.array(_sampling_indices, dtype=np.int)
    # else do the sampling de novo
    else:
        if _sampling_size is None:
            raise ValueError(f"_sampling_size should be given if no sampling indices directly provided.")
        if _remove_self and self_id >= 0:
            _sampling_indices = [np.random.choice(len(_dm_zxys)-1, _sampling_size, replace=False)
                                 for _i in range(_n_iter)]
        else:
            _sampling_indices = [np.random.choice(len(_dm_zxys)-1, _sampling_size, replace=False)
                                 for _i in range(_n_iter)]
        _sampling_indices = np.array(_sampling_indices, dtype=np.int)
    # adjust sampling if shouldn't include self point
    if _remove_self and self_id >= 0:
        _sampling_indices[_sampling_indices>=self_id] += 1

    ## do sampling
    _p = [is_in_hull(_dm_zxys[_inds], _reg_zxy, remove_vertices=_remove_self)
          for _inds in _sampling_indices]

    return np.nanmean(_p)