## Load other sub-packages
from .. import visual_tools, get_img_info

## Load shared parameters
from . import _distance_zxy, _image_size, _allowed_colors 
from . import _num_buffer_frames, _num_empty_frames

## load packages
import numpy as np
import scipy
import os, sys, glob, time


def decide_starting_frames(channels, num_channels=None, all_channels=_allowed_colors, 
                           num_buffer_frames=10, num_empty_frames=0, verbose=False):
    """Function to decide starting frame ids given channels"""
    if num_channels is None:
        if verbose:
            print(f"num_channels is not given, thus use default: {len(all_channels)}")
        num_channels = len(all_channels)
    else:
        all_channels = all_channels[:num_channels]
    # check channels
    if not isinstance(channels, list):
        raise TypeError(f"channels should be a list but {type(channels)} is given.")
    _channels = [str(_ch) for _ch in channels]
    for _ch in _channels:
        if _ch not in all_channels:
            raise ValueError(f"Wrong input channel:{_ch}, should be among {all_channels}")\
    # check num_buffer_frames and num_empty_frames
    num_buffer_frames = int(num_buffer_frames)
    num_empty_frames = int(num_empty_frames)
    
    # get starting frames
    _start_frames = [(all_channels.index(_ch)-num_buffer_frames+num_empty_frames)%num_channels\
                     + num_buffer_frames for _ch in _channels]
    
    return _start_frames
    

    
def translate_crop_by_drift(crop3d, drift3d=np.array([0,0,0]), single_im_size=_image_size):
    
    crop3d = np.array(crop3d, dtype=np.int)
    drift3d = np.array(drift3d)
    single_im_size = np.array(single_im_size, dtype=np.int)
    # deal with negative upper limits    
    for _i, (_lims, _s) in enumerate(zip(crop3d, single_im_size)):
        if _lims[1] < 0:
            crop3d[_i,1] += _s
    _drift_limits = np.zeros(crop3d.shape, dtype=np.int)
    # generate drifted crops
    for _i, (_d, _lims) in enumerate(zip(drift3d, crop3d)):
        _drift_limits[_i, 0] = max(_lims[0]-np.ceil(np.abs(_d)), 0)
        _drift_limits[_i, 1] = min(_lims[1]+np.ceil(np.abs(_d)), single_im_size[_i])
    return _drift_limits


def generate_neighboring_crop(coord, crop_size=5, 
                              single_im_size=_image_size,
                              sub_pixel_precision=False):
    """Function to generate crop given coord coordinate and crop size
    Inputs:
    Output:
    """
    from ..classes.preprocess import ImageCrop
    ## check inputs
    _coord =  np.array(coord)[:len(single_im_size)]
    if isinstance(crop_size, int) or isinstance(crop_size, np.int32):
        _crop_size = np.ones(len(single_im_size),dtype=np.int32) * crop_size
    else:
        _crop_size = np.array(crop_size)[:len(single_im_size)]
        
    _single_image_size = np.array(single_im_size, dtype=np.int32)
    # find limits for this crop
    if sub_pixel_precision:
        _left_lims = np.max([_coord-_crop_size, np.zeros(len(_single_image_size))], axis=0)
        _right_lims = np.min([_coord+_crop_size+1, _single_image_size], axis=0)
    else:
        # limits
        _left_lims = np.max([np.round(_coord-_crop_size), np.zeros(len(_single_image_size))], axis=0)
        _right_lims = np.min([np.round(_coord+_crop_size+1), _single_image_size], axis=0)

    _crop = ImageCrop(len(single_im_size), 
                      np.array([_left_lims, _right_lims]).transpose(), 
                      single_im_size=single_im_size)

    return _crop

def batch_crop(im, coord, crop_size=5, ):
    single_im_size = np.array(im.shape)
    _crop = generate_neighboring_crop(
        coord=coord, crop_size=crop_size, 
        single_im_size=single_im_size, sub_pixel_precision=False,
    )
    # crop locally
    try:
        _signals = im[_crop.to_slices()]
        if np.size(_signals) == 0:
            _signals = -1 * np.ones(1, dtype=im.dtype)
    except:
        _signals = -1 * np.ones(1, dtype=im.dtype)

    return _signals


def crop_neighboring_area(im, center, crop_sizes, 
                          extrapolate_mode='nearest'):
    
    """Function to crop neighboring area of a certain coordiante
    Args:
        im: image, np.ndarray
        center: zxy coordinate for the center of this crop area
        crop_sizes: dimension(s) of the cropping area, int or np.ndarray
        extrapolate_mode: mode in map_coordinate, str
    Return:
        _cim: cropped image, np.ndarray
    """
    if 'map_coordinates' not in locals():
        from scipy.ndimage.interpolation import map_coordinates
    if not isinstance(im, np.ndarray):
        raise TypeError(f"wrong input image, should be np.ndarray")
    
    _dim = len(np.shape(im))
    _center = np.array(center)[:_dim]
    # crop size
    if isinstance(crop_sizes, int) or isinstance(crop_sizes, np.int32):
        _crop_sizes = np.ones(_dim, dtype=np.int32)*int(crop_sizes)
    elif isinstance(crop_sizes, list) or isinstance(crop_sizes, np.ndarray):
        _crop_sizes = np.array(crop_sizes)[:_dim]
    
    # generate a rough crop, to save RAM
    _rough_left_lims = np.max([np.zeros(_dim), 
                               np.floor(_center-_crop_sizes/2)], axis=0)
    _rough_right_lims = np.min([np.array(np.shape(im)), 
                                np.ceil(_center+_crop_sizes/2)], axis=0)
    _rough_center = _center - _rough_left_lims
    
    _rough_crop = tuple([slice(int(_l),int(_r)) for _l,_r in zip(_rough_left_lims, _rough_right_lims)])
    _rough_cropped_im = im[_rough_crop]
    
    # generate coordinates to be mapped
    _pixel_coords = np.indices(_crop_sizes) + np.expand_dims(_rough_center - (_crop_sizes-1)/2, 
                                                       tuple(np.arange(_dim)+1))
    #return _pixel_coords
    # map coordiates
    _cim = map_coordinates(_rough_cropped_im, _pixel_coords.reshape(_dim, -1),
                           mode=extrapolate_mode, cval=np.min(_rough_cropped_im))
    _cim = _cim.reshape(_crop_sizes) # reshape back to original shape
    
    
    return _cim