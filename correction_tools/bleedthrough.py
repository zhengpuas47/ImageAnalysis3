# import functions from packages
import os
import time 
import pickle 
import numpy as np
import scipy
import matplotlib.pyplot as plt 
# import local variables
from .. import _allowed_colors, _image_size, _correction_folder

# import local functions
from ..io_tools.load import correct_fov_image
from ..spot_tools.fitting import fit_fov_image
from ..io_tools.crop import crop_neighboring_area
from .chromatic import generate_polynomial_data



_bleedthrough_channels=['750', '647', '561']

_bleedthrough_default_correction_args = {
    'correction_folder': _correction_folder,
    'single_im_size':_image_size,
    'all_channels':_allowed_colors,
    'bleed_corr':False,
    'illumination_corr':False,
    'chromatic_corr':False,
}

_bleedthrough_default_fitting_args = {'max_num_seeds':500,
    'th_seed': 300,
    'use_dynamic_th':True,
}


def find_bleedthrough_pairs(filename, channel,
                            corr_channels=_bleedthrough_channels,
                            correction_args=_bleedthrough_default_correction_args,
                            fitting_args=_bleedthrough_default_fitting_args, 
                            intensity_th=1.,
                            crop_size=9, rsq_th=0.9, 
                            save_temp=True, save_name=None,
                            overwrite=True, verbose=True,
                            ):
    """Function to generate bleedthrough spot pairs"""
    if 'LinearRegression' not in locals():
        from sklearn.linear_model import LinearRegression
        
    ## check inputs
    _channel = str(channel)
    if _channel not in corr_channels:
        raise ValueError(f"{_channel} should be within {corr_channels}")
    
    _info_dict = {}
    _load_flags = []
    for _ch in corr_channels:
        if _ch != _channel:
            _basename = os.path.basename(filename).replace('.dax', f'_ref_{_channel}_to_{_ch}.pkl')
            _basename = 'bleedthrough_'+_basename
            _temp_filename = os.path.join(os.path.dirname(filename), _basename)
            if os.path.isfile(_temp_filename) and not overwrite:
                _info_dict[f"{_channel}_to_{_ch}"] = pickle.load(open(_temp_filename, 'rb'))
                _load_flags.append(1)
            else:
                _info_dict[f"{_channel}_to_{_ch}"] = []
                _load_flags.append(0)
    
    if np.mean(_load_flags) == 1:
        if verbose:
            print(f"-- directly load from saved tempfile in folder: {os.path.dirname(filename)}")
        return _info_dict
        
    ## 1. load this file
    _ims, _ = correct_fov_image(filename, corr_channels,
                                calculate_drift=False, warp_image=False,
                                **correction_args, 
                                return_drift=False, verbose=verbose,
                                )
    ## 2. fit centers for target channel
    _ref_im = _ims[list(corr_channels).index(_channel)]
    _tar_channels = [_ch for _ch, _im in zip(corr_channels, _ims) if _ch != _channel]
    _tar_ims = [_im for _ch, _im in zip(corr_channels, _ims) if _ch != _channel]
    
    _ref_spots = fit_fov_image(_ref_im, _channel, normalize_backgroud=True,
                               **fitting_args, verbose=verbose)
    # threshold intensities
    _ref_spots = _ref_spots[_ref_spots[:,0] >= intensity_th]
    ## crop
    for _ch, _im in zip(_tar_channels, _tar_ims):
        _key = f"{_channel}_to_{_ch}"
        if verbose:
            print(f"--- finding matched bleedthrough pairs for {_key}")
        # loop through centers to crop
        for _spot in _ref_spots:
            _rim = crop_neighboring_area(_ref_im, _spot[1:4], crop_sizes=crop_size)
            _cim = crop_neighboring_area(_im, _spot[1:4], crop_sizes=crop_size)
            # calculate r-square
            _x = np.array([np.ravel(_rim), np.ones(np.size(_rim))]).transpose()
            _y = np.ravel(_cim)
            _reg = LinearRegression().fit(_x,_y)        
            _rsq = _reg.score(_x,_y)
    
            print(_reg.coef_, _rsq)
            if _rsq >= rsq_th:
                _info = {
                    'coord': _spot[1:4],
                    'ref_im': _rim,
                    'bleed_im': _cim,
                    'rsquare': _rsq,
                    'slope': _reg.coef_,
                    'intercept': _reg.intercept_,
                    'file':filename,
                }
                _info_dict[_key].append(_info)

    if save_temp:
        for _ch in corr_channels:
            if _ch != _channel:
                _key = f"{_channel}_to_{_ch}"
                _basename = os.path.basename(filename).replace('.dax', f'_ref_{_channel}_to_{_ch}.pkl')
                _basename = 'bleedthrough_'+_basename
                _temp_filename = os.path.join(os.path.dirname(filename), _basename)

            if verbose:
                print(f"--- saving {len(_info_dict[_key])} points to file:{_temp_filename}")
            pickle.dump(_info_dict[_key], open(_temp_filename, 'wb'))

    return _info_dict
            


def interploate_bleedthrough_correction_from_channel(
    info_dicts, ref_channel, target_channel, 
    min_num_spots=200, 
    single_im_size=_image_size, ref_center=None,
    fitting_order=2, 
    save_temp=True, save_folder=None, 
    make_plots=True, save_plots=True,
    overwrite=False, verbose=True,
):
    """Function to interpolate and generate the bleedthrough correction profiles between two channels
    
    """
    _key = f"{ref_channel}_to_{target_channel}"
    _coords = []
    _slopes = []
    _intercepts = []
    
    for _dict in info_dicts:
        if _key in _dict:
            _infos = _dict[_key]
            for _info in _infos:
                _coords.append(_info['coord'])
                _slopes.append(_info['slope'][0])
                _intercepts.append(_info['intercept'])
    
    if len(_coords) < min_num_spots:
        if verbose:
            print(f"-- not enough spots from {ref_channel} to {target_channel}")
        return np.zeros(single_im_size), np.zeros(single_im_size)
    else:
        if verbose:
            print(f"-- {len(_coords)} spots are used to generate profiles from {ref_channel} to {target_channel}")
        _coords = np.array(_coords)
        _slopes = np.array(_slopes)
        _intercepts = np.array(_intercepts)

        # adjust ref_coords with ref center
        if ref_center is None:
            _ref_center = np.array(single_im_size)[:np.shape(_coords)[1]] / 2
        else:
            _ref_center = np.array(ref_center)[:np.shape(_coords)[1]]
        _ref_coords = _coords - _ref_center[np.newaxis, :]

        print(_coords.shape, _slopes.shape, _intercepts.shape)

        # generate_polynomial_data
        _X = generate_polynomial_data(_coords, fitting_order)
        # do the least-square optimization for slope
        _C_slope, _r,_r2,_r3 = scipy.linalg.lstsq(_X, _slopes)   
        _rsq_slope =  1 - np.sum((_X.dot(_C_slope) - _slopes)**2)\
                    / np.sum((_slopes-np.mean(_slopes))**2) # r2 = 1 - SSR/SST   
        print(_C_slope, _rsq_slope)
        
        # do the least-square optimization for intercept
        _C_intercept, _r,_r2,_r3 = scipy.linalg.lstsq(_X, _intercepts)   
        _rsq_intercept =  1 - np.sum((_X.dot(_C_intercept) - _intercepts)**2)\
                    / np.sum((_intercepts-np.mean(_intercepts))**2) # r2 = 1 - SSR/SST   
        print(_C_intercept, _rsq_intercept)
        
        ## generate profiles
        _pixel_coords = np.indices(single_im_size)
        _pixel_coords = _pixel_coords.reshape(np.shape(_pixel_coords)[0], -1)
        _pixel_coords = _pixel_coords - _ref_center[:, np.newaxis]
        # generate predictive pixel coordinates
        _pX = generate_polynomial_data(_pixel_coords.transpose(), fitting_order)
        _p_slope = np.dot(_pX, _C_slope).reshape(single_im_size)
        _p_intercept = np.dot(_pX, _C_intercept).reshape(single_im_size)
        ## save temp if necessary
        
        if save_temp:
            if save_folder is not None and os.path.isdir(save_folder):
                if verbose:
                    print(f"-- saving bleedthrough temp profile from channel: {ref_channel} to channel: {target_channel}.")
                
            else:
                print(f"-- save_folder is not given or not valid, skip.")
            
        ## make plots if applicable
        if make_plots:
            plt.figure(dpi=150, figsize=(4,3))
            plt.imshow(_p_slope.mean(0))
            plt.colorbar()
            plt.title(f"{ref_channel} to {target_channel}, slope")
            if save_plots and (save_folder is not None and os.path.isdir(save_folder)):
                plt.savefig(os.path.join(save_folder, f'bleedthrough_profile_{ref_channel}_to_{target_channel}_slope.png'),
                            transparent=True)
            plt.show()

            plt.figure(dpi=150, figsize=(4,3))
            plt.imshow(_p_intercept.mean(0))
            plt.colorbar()
            plt.title(f"{ref_channel} to {target_channel}, intercept")
            if save_plots and (save_folder is not None and os.path.isdir(save_folder)):
                plt.savefig(os.path.join(save_folder, f'bleedthrough_profile_{ref_channel}_to_{target_channel}_intercept.png'),
                            transparent=True)
            plt.show()

        return _p_slope, _p_intercept