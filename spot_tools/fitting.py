import numpy as np 
import os, time
from . import _seed_th
from .. import _sigma_zxy
from ..External import Fitting_v3
from ..visual_tools import get_seed_points_base

def __init__():
    pass

def fit_fov_image(im, channel, max_num_seeds=500,
                  seed_gft_size=0.75, background_gft_size=10, local_ft_size=3,
                  th_seed=None, hot_pix_th=5, remove_edge=True, 
                  fit_radius=5, init_sigma=_sigma_zxy, weight_sigma=0, 
                  fitting_args={}, verbose=True):
    """Function to merge seeding and fitting for the whole fov image"""
    ## check inputs
    if th_seed is None:
        _th_seed = _seed_th[str(channel)]
    else:
        _th_seed = float(th_seed)
    if verbose:
        print(f"-- start fitting spots in channel:{channel}, ", end='')
        _fit_time = time.time()
    ## seeding
    _seeds = get_seed_points_base(im, seed_gft_size, background_gft_size,
                                  local_ft_size, th_seed=_th_seed, hot_pix_th=hot_pix_th)
    if max_num_seeds is not None and max_num_seeds > 0:
        _seeds = _seeds[:, :int(max_num_seeds)]
    if remove_edge:
        _seeds = remove_edge_seeds(im, _seeds, np.ceil(local_ft_size/2))
    if verbose:
        print(f"{len(_seeds.T)} seeded, ", end='')
    ## fitting
    _fitter = Fitting_v3.iter_fit_seed_points(
        im, _seeds, radius_fit=fit_radius, 
        init_w=init_sigma, weight_sigma=weight_sigma,
        **fitting_args,
    )    
    # fit
    _fitter.firstfit()
    # check
    _fitter.repeatfit()
    # get spots
    _spots = np.array(_fitter.ps)
    if verbose:
        print(f"{len(_spots)} fitted in {time.time()-_fit_time:.3f}s.")
    return _spots


def remove_edge_seeds(im, T_seeds, distance=2):
    im_size = np.array(np.shape(im))
    _seeds = np.array(T_seeds[:len(im_size),:]).transpose()
    flags = []
    for _seed in _seeds:
        _f = ((_seed >= distance) * (_seed <= im_size)).all()
        flags.append(_f)
    _kept_seeds = T_seeds[:, np.array(flags, dtype=np.bool)]
 
    return _kept_seeds
