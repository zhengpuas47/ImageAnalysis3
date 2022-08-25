import numpy as np 
#
def find_interaction_groups(_chr2Zxys, codebook, search_radius=0.5, min_chrs=3):
    """Function to find multi-way-contact loop"""
    from ..figure_tools.plot_decode import summarize_chr2Zxys
    from sklearn.neighbors import KDTree
    from scipy.spatial.distance import pdist
    # merge
    _cell_zxys, _cell_rids = summarize_chr2Zxys(_chr2Zxys, codebook, keep_valid=True)
    # kdtree query
    interacting_groups = [tuple(np.sort(_g)) 
                          for _g in KDTree(_cell_zxys).query_radius(_cell_zxys, search_radius)
                          if len(_g) >= min(min_chrs, 3)] # look for at least 3-way contacts
    # keep unique
    interacting_groups = list(set(interacting_groups))
    # init selection
    cell_inter_coords = []
    cell_inter_rids = []
    cell_inter_chrs = []
    # collect only trans-chr
    for _g in interacting_groups:
        _g_zxys = _cell_zxys[np.array(_g)]
        # make sure all of them are close enough to each other
        if (pdist(_g_zxys) < search_radius).all():
            _g_rids = _cell_rids[np.array(_g)]
            _g_chrs = codebook.iloc[_g_rids]['chr'].values
            if len(np.unique(_g_chrs)) >= min_chrs:
                #cell_inter_groups.append(np.array(_g))
                cell_inter_coords.append(_g_zxys)
                cell_inter_rids.append(_g_rids)
                cell_inter_chrs.append(_g_chrs)
    return cell_inter_coords, cell_inter_rids, cell_inter_chrs
#
## define zigzag
def find_loopout_regions(zxys, region_ids=None, 
                         method='neighbor', dist_th=1500, 
                         neighbor_region_num=5):
    """Function to find loopout, or zig-zag features within chromosomes.
    Inputs:

    Outputs:

    """

    # convert inputs
    from ..spot_tools.scoring import _local_distance
    from ..spot_tools.scoring import _neighboring_distance
    _zxys = np.array(zxys)

    # if region ids not specified, presume it is continuous
    if region_ids is None:
        region_ids = np.arange(len(_zxys))
    else:
        region_ids = np.array(region_ids)
    
    # identify distance to neighbors
    if method == 'neighbor':
        _nb_dists = _neighboring_distance(zxys, spot_ids=region_ids, neighbor_step=1)[:-1]
        _loopout_flags = np.zeros(len(zxys))
        _loopout_flags[1:] += (_nb_dists >= dist_th) * (1-np.isnan(_nb_dists))
        _loopout_flags[:-1] += (_nb_dists >= dist_th) * (1-np.isnan(_nb_dists))
        
        return _loopout_flags == 2

    elif method == 'local':
        _lc_dists = _local_distance(zxys, spot_ids=region_ids,
                                    sel_zxys=zxys, sel_ids=region_ids, local_size=neighbor_region_num)
        return _lc_dists >= dist_th

    else:
        raise ValueError(f"wrong input method:{method}, exit.")