## Load other sub-packages
from .. import visual_tools, get_img_info

## Load shared parameters
from . import _distance_zxy, _image_size, _allowed_colors 
from . import _num_buffer_frames, _num_empty_frames
from .crop import decide_starting_frames, translate_crop_by_drift

## load packages
import numpy as np
import scipy
import os, sys, glob, time

from scipy import ndimage

def multi_crop_image_fov(filename, channels, crop_limit_list,
                         all_channels=_allowed_colors, single_im_size=_image_size,
                         num_buffer_frames=10, num_empty_frames=0,
                         drift=np.array([0,0,0]), return_limits=False, 
                         verbose=False):
    """Function to load images for multiple cells in a fov
    Inputs:
        filename: .dax filename for given image, string of filename
        channels: color_channels for the specific data, list of int or str
        crop_limit_list: list of 2x2 or 3x2 array specifying where to crop, list of np.ndarray
        all_channels: all allowed colors in given data, list (default: _allowed_colors)
        single_im_size: image size for single color full image, list/array of 3 (default:[30,2048,2048])
        num_buffer_frame: number of frames before z-scan starts, int (default:10)
        num_empty_frames: number of empty frames at beginning of zscan, int (default: 0)
        drift: drift to ref-frame of this image, np.array of 3 (default:[0,0,0])
        return_limits: whether return drifted limits for cropping, bool (default: False)
        verbose: say something!, bool (default:False)
    Outputs:
        _cropped_im_list: cropped image list by crop_limit_list x channels
            list of len(crop_limit_list) x size of channels
        (optional) _drifted_limits: drifted list of crop limits
    """
    # load
    if 'DaxReader' not in locals():
        from ..visual_tools import DaxReader
    if 'get_num_frame' not in locals():
        from ..get_img_info import get_num_frame
    ## 0. Check inputs
    # filename
    if not os.path.isfile(filename):
        raise ValueError(f"file {filename} doesn't exist!")
    # channels 
    if isinstance(channels, list):
        _channels = [str(ch) for ch in channels]
    elif isinstance(channels, int) or isinstance(channels, str):
        _channels = [str(channels)]
    else:
        raise TypeError(f"Wrong input type for channels:{type(channels)}, should be list/str/int")
    # check channel values in all_channels
    for _ch in _channels:
        if _ch not in all_channels:
            raise ValueError(f"Wrong input for channel:{_ch}, should be among {all_channels}")
    # check num_buffer_frames and num_empty_frames
    num_buffer_frames = int(num_buffer_frames)
    num_empty_frames = int(num_empty_frames)

    ## 1. Load image
    if verbose:
        print(f"-- crop {len(crop_limit_list)} images with channels:{_channels}")
    # extract image info
    _full_im_shape, _num_channels = get_num_frame(filename,
                                                  frame_per_color=single_im_size[0],
                                                  buffer_frame=num_buffer_frames)
    # load the whole image
    if verbose:
        print(f"--- load image from file:{filename}")
    _full_im = DaxReader(filename, verbose=verbose).loadAll()
    # splice buffer frames
    _start_frames = decide_starting_frames(_channels, _num_channels, all_channels=all_channels,
                                           num_buffer_frames=num_buffer_frames, num_empty_frames=num_empty_frames,
                                           verbose=verbose)
    _splitted_ims = [_full_im[_sf:-num_buffer_frames:_num_channels] for _sf in _start_frames]

    ## 2. Prepare crops
    if verbose:
        print(f"-- start cropping: ", end='')
    _old_crop_list = []
    _drift_crop_list = []
    for _crop in crop_limit_list:
        if len(_crop) == 2:
            _n_crop = np.array([np.array([0, single_im_size[0]])]+list(_crop), dtype=np.int)
        elif len(_crop) == 3:
            _n_crop = np.array(_crop, dtype=np.int)
        else:
            raise ValueError(f"Wrong input _crop, should be 2d or 3d crop but {_crop} is given.")
        # append
        _old_crop_list.append(_n_crop)
        _drift_crop_list.append(translate_crop_by_drift(_n_crop, drift, single_im_size=single_im_size))
    # 2.1 Crop image
    _cropped_im_list = []
    _drifted_limits = []
    for _crop, _n_crop in zip(_old_crop_list, _drift_crop_list):
        _cims = []
        for _ch, _im in zip(_channels, _splitted_ims):
            # translate
            _cim = ndimage.interpolation.shift(_im[tuple(slice(_c[0],_c[1]) for _c in _crop)], 
                                               -drift, mode='nearest')
            # revert to original crop size
            _diffs = (_crop - _n_crop).astype(np.int)
            _cims.append(_cim[_diffs[0, 0]: _diffs[0, 0]+_crop[0, 1]-_crop[0, 0],
                              _diffs[1, 0]: _diffs[1, 0]+_crop[1, 1]-_crop[1, 0],
                              _diffs[2, 0]: _diffs[2, 0]+_crop[2, 1]-_crop[2, 0]])
        # save cropped ims
        if isinstance(channels, list):
            _cropped_im_list.append(_cims)
        elif isinstance(channels, int) or isinstance(channels, str):
            _cropped_im_list.append(_cims[0])
        # save drifted limits
        _d_limits = np.array([_n_crop[:, 0]+_diffs[:, 0],
                              _n_crop[:, 0]+_diffs[:, 0]+_crop[:, 1]-_crop[:, 0]]).T
        _drifted_limits.append(_d_limits)
        if verbose:
            print("*", end='')
    
    if verbose:
        print("done")

    if return_limits:
        return _cropped_im_list, _drifted_limits
    else:
        return _cropped_im_list