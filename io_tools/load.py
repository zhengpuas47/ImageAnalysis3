## Load other sub-packages
from .. import visual_tools, get_img_info, corrections

## Load shared parameters
from . import _distance_zxy, _image_size, _allowed_colors, _corr_channels, _correction_folder
from . import _num_buffer_frames, _num_empty_frames
from .crop import decide_starting_frames, translate_crop_by_drift

## load packages
import numpy as np
import scipy
import os, sys, glob, time

from scipy.ndimage.interpolation import shift, map_coordinates

def multi_crop_image_fov(filename, channels, crop_limit_list,
                         all_channels=_allowed_colors, single_im_size=_image_size,
                         num_buffer_frames=10, num_empty_frames=0,
                         drift=np.array([0,0,0]), shift_order=1,
                         return_limits=False, verbose=False):
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
        print(f"--- load image from file:{filename}", end=', ')
        _load_start = time.time()
    _full_im = DaxReader(filename, verbose=verbose).loadAll()
    # splice buffer frames
    _start_frames = decide_starting_frames(_channels, _num_channels, all_channels=all_channels,
                                           num_buffer_frames=num_buffer_frames, num_empty_frames=num_empty_frames,
                                           verbose=verbose)
    _splitted_ims = [_full_im[_sf:-num_buffer_frames:_num_channels] for _sf in _start_frames]
    if verbose:
        print(f"in {time.time()-_load_start}s")
    ## 2. Prepare crops
    if verbose:
        print(f"-- start cropping: ", end='')
        _start_time = time.time()
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
    for _old_crop, _n_crop in zip(_old_crop_list, _drift_crop_list):
        _cims = []
        for _ch, _im in zip(_channels, _splitted_ims):
            if drift.any():
                # translate
                _cim = shift(_im, -drift, order=shift_order, mode='nearest')
            else:
                _cim = _im.copy()
            # revert to original crop size
            _diffs = (_old_crop - _n_crop).astype(np.int)
            _cims.append(_cim[_diffs[0, 0]: _diffs[0, 0]+_old_crop[0, 1]-_old_crop[0, 0],
                              _diffs[1, 0]: _diffs[1, 0]+_old_crop[1, 1]-_old_crop[1, 0],
                              _diffs[2, 0]: _diffs[2, 0]+_old_crop[2, 1]-_old_crop[2, 0]])
        # save cropped ims
        if isinstance(channels, list):
            _cropped_im_list.append(_cims)
        elif isinstance(channels, int) or isinstance(channels, str):
            _cropped_im_list.append(_cims[0])
        # save drifted limits
        _d_limits = np.array([_n_crop[:, 0]+_diffs[:, 0],
                              _n_crop[:, 0]+_diffs[:, 0]+_old_crop[:, 1]-_old_crop[:, 0]]).T
        _drifted_limits.append(_d_limits)
        if verbose:
            print("*", end='')
    
    if verbose:
        print(f"done in {time.time()-_start_time}s.")

    if return_limits:
        return _cropped_im_list, _drifted_limits
    else:
        return _cropped_im_list

def correct_fov_image(dax_filename, sel_channels, 
                      single_im_size=_image_size, all_channels=_allowed_colors,
                      num_buffer_frames=10, num_empty_frames=0, drift=np.array([0,0,0]),
                      corr_channels=_corr_channels, correction_folder=_correction_folder,
                      hot_pixel_corr=True, hot_pixel_th=4, z_shift_corr=False,
                      illumination_corr=True, illumination_profile=None, 
                      bleed_corr=True, bleed_profile=None, ref_channel='647',
                      chromatic_corr=True, chromatic_profile=None, 
                      normalization=False, output_dtype=np.uint16,
                      verbose=True):
    """Function to correct one whole field-of-view image in proper manner"""
    ## check inputs
    # dax_filename
    if not os.path.isfile(dax_filename):
        raise IOError(f"Dax file: {dax_filename} is not a file, exit!")
    if not isinstance(dax_filename, str) or dax_filename[-4:] != '.dax':
        raise IOError(f"Dax file: {dax_filename} has wrong data type, exit!")
    if verbose:
        print(f"- correct the whole fov for image: {dax_filename}")
    # selected channels
    sel_channels = [str(ch) for ch in sel_channels]
    # image size etc
    single_im_size = np.array(single_im_size, dtype=np.int)
    all_channels = [str(ch) for ch in all_channels]
    num_buffer_frames = int(num_buffer_frames)
    num_empty_frames = int(num_empty_frames)
    drift = np.array(drift[:3], dtype=np.float)

    ## correction channels and profiles
    corr_channels = [str(ch) for ch in sorted(corr_channels, key=lambda v:-int(v))]    
    for _ch in corr_channels:
        if _ch not in all_channels:
            raise ValueError(f"Wrong correction channel:{_ch}, should be within {all_channels}")
    
    ## determine loaded channels
    _overlap = [_ch for _ch in corr_channels if _ch in sel_channels]
    if len(_overlap) > 0 and bleed_corr:
        _load_channels = [_ch for _ch in corr_channels]
    else:
        _load_channels = []
    # append sel_channels
    for _ch in sel_channels:
        if _ch not in _load_channels:
            _load_channels.append(_ch)

    ## check profiles
    if illumination_corr:
        if illumination_profile is None:
            illumination_profile = load_correction_profile('illumination', corr_channels=_load_channels, 
                                correction_folder=correction_folder, all_channels=all_channels,
                                ref_channel=ref_channel, im_size=single_im_size, verbose=verbose)
        else:
            illumination_profile = np.array(illumination_profile, dtype=np.float)
            if (illumination_profile.shape != (single_im_size[-2], single_im_size[-1])).any():
                raise IndexError(f"Wrong input shape for chromatic_profile: {illumination_profile.shape}")
    if bleed_corr:
        if bleed_profile is None:
            bleed_profile = load_correction_profile('bleedthrough', corr_channels=corr_channels, 
                                correction_folder=correction_folder, all_channels=all_channels,
                                ref_channel=ref_channel, im_size=single_im_size, verbose=verbose)
        else:
            bleed_profile = np.array(bleed_profile, dtype=np.float)
            if (bleed_profile.shape != (len(corr_channels),len(corr_channels),single_im_size[-2], single_im_size[-1])).any():
                raise IndexError(f"Wrong input shape for bleed_profile: {bleed_profile.shape}")
    if chromatic_corr:
        if chromatic_profile is None:
            chromatic_profile = load_correction_profile('chromatic', corr_channels=corr_channels, 
                                correction_folder=correction_folder, all_channels=all_channels,
                                ref_channel=ref_channel, im_size=single_im_size, verbose=verbose)
        else:
            chromatic_profile = np.array(chromatic_profile, dtype=np.float)
            if (chromatic_profile.shape != (len(single_im_size), single_im_size[-2], single_im_size[-1])).any():
                raise IndexError(f"Wrong input shape for chromatic_profile: {chromatic_profile.shape}")

    ## check output data-type
    # if normalization, output should be float
    if normalization and output_dtype==np.uint16:
        output_dtype = np.float 
    # otherwise keep original dtype
    else:
        pass

    ## Load image
    if verbose:
        print(f"-- loading image from file:{dax_filename}", end=' ')
        _load_time = time.time()
    from ..visual_tools import DaxReader
    _im = DaxReader(dax_filename).loadAll()
    # get number of colors and frames
    from ..get_img_info import get_num_frame, split_channels
    _full_im_shape, _num_color = get_num_frame(dax_filename,
                                               frame_per_color=single_im_size[0],
                                               buffer_frame=num_buffer_frames)
    _ims = split_im_by_channels(_im, _load_channels, all_channels[:_num_color], 
                                single_im_size=single_im_size, 
                                num_buffer_frames=num_buffer_frames,
                                num_empty_frames=num_empty_frames)
    # clear memory
    del(_im)
    if verbose:
        print(f" in {time.time()-_load_time:.3f}s")
    ## hot-pixel removal
    if hot_pixel_corr:
        if verbose:
            print(f"-- removing hot pixels for channels:{_load_channels}", end=' ')
            _hot_time = time.time()
        # loop through and correct
        for _i, (_ch, _im) in enumerate(zip(_load_channels, _ims)):
            _nim = corrections.Remove_Hot_Pixels(_im.astype(np.float),
                dtype=output_dtype, hot_th=hot_pixel_th)
            _ims[_i] = _nim
        if verbose:
            print(f"in {time.time()-_hot_time:.3f}s")

    ## Z-shift correction
    if z_shift_corr:
        if verbose:
            print(f"-- correct Z-shifts for channels:{_load_channels}", end=' ')
            _z_time = time.time()
        _ims[_i] = corrections.Z_Shift_Correction(_im.astype(np.float),
            dtype=output_dtype, normalization=False)
        if verbose:
            print(f"in {time.time()-_z_time:.3f}s")

    ## illumination correction
    if illumination_corr:
        if verbose:
            print(f"-- illumination correction for channels:", end=' ')
            _illumination_time = time.time()
        for _i, (_im,_ch) in enumerate(zip(_ims, _load_channels)):
            if verbose:
                print(f"{_ch}", end=', ')
            _ims[_i] = (_im.astype(np.float) / illumination_profile[_ch][np.newaxis,:]).astype(output_dtype)
        # clear
        del(illumination_profile)
        if verbose:
            print(f"in {time.time()-_illumination_time:.3f}s")

    ## bleedthrough correction
    if len(_overlap) > 0 and bleed_corr:
        if verbose:
            print(f"-- bleedthrough correction for channels: {corr_channels}", end=' ')
            _bleed_time = time.time()
        _bld_ims = [_ims[_load_channels.index(_ch)] for _ch in corr_channels]
        _bld_corr_ims = []
        for _i, _ch in enumerate(corr_channels):
            _nim = np.sum([_im * bleed_profile[_i, _j] 
                            for _j,_im in enumerate(_bld_ims)],axis=0)
            _bld_corr_ims.append(_nim)
        # update images
        for _nim, _ch in zip(_bld_corr_ims, corr_channels):
            # restore output_type
            _nim[_nim > np.iinfo(output_dtype).max] = np.iinfo(output_dtype).max
            _nim[_nim < np.iinfo(output_dtype).min] = np.iinfo(output_dtype).min
            _ims[_load_channels.index(_ch)] = _nim.astype(output_dtype)
        # clear
        del(_bld_ims, _bld_corr_ims, bleed_profile)
        if verbose:
            print(f"in {time.time()-_bleed_time:.3f}s")

    ## chromatic abbrevation
    if chromatic_corr and sum([_ch in corr_channels for _ch in _load_channels]):

        if verbose:
            print(f"-- chromatic correction for channels: {corr_channels}", end=' ')
            _chromatic_time = time.time()
        for _i, _ch in enumerate(_load_channels):
            if _ch in corr_channels:
                if _ch == ref_channel and not drift.any():
                    if verbose:
                        print(f"{_ch}-skipped", end=', ')
                    continue
                else:
                    if verbose:
                        print(f"{_ch}", end=', ')
                    # 0. get old image
                    _im = _ims[_load_channels.index(_ch)].copy().astype(np.float)
                    # 1. get coordiates to be mapped
                    _coords = np.meshgrid( np.arange(single_im_size[0]), 
                            np.arange(single_im_size[1]), 
                            np.arange(single_im_size[2]), )
                    _coords = np.stack(_coords).transpose((0, 2, 1, 3)) # transpose is necessary
                    # 2. calculate corrected coordinates as a reference
                    if _ch != ref_channel:
                        _coords = _coords + chromatic_profile[_ch][:,np.newaxis,:,:]
                    # 3. apply drift if necessary
                    if drift.any():
                        _coords += drift[:, np.newaxis,np.newaxis,np.newaxis]
                    # 4. map coordinates
                    _corr_im = map_coordinates(_im, _coords.reshape(_coords.shape[0], -1), mode='nearest')
                    _corr_im = _corr_im.reshape(np.shape(_im))
                    # append 
                    _ims[_load_channels.index(_ch)] = _corr_im.astype(output_dtype)
                    # local clear
                    del(_coords, _im)
        # clear
        del(_corr_im, chromatic_profile)
        if verbose:
            print(f"in {time.time()-_chromatic_time:.3f}s")

    ## normalization
    if normalization:
        for _i, _im in enumerate(_ims):
            _ims[_i] = _im.astype(np.float) / np.median(_im)
    
    ## summarize and report selected_ims
    _sel_ims = []
    for _ch in sel_channels:
        _sel_ims.append(_ims[_load_channels.index(_ch)].astype(output_dtype).copy())
    # clear
    del(_ims)

    return _sel_ims


def split_im_by_channels(im, sel_channels, all_channels, single_im_size=_image_size,
                         num_buffer_frames=10, num_empty_frames=0, skip_frame0=True):
    """Function to split a full image by channels"""
    _num_colors = len(all_channels)
    _sel_channels = [str(_ch) for _ch in sel_channels]
    _all_channels = [str(_ch) for _ch in all_channels]
    for _ch in _sel_channels:
        if _ch not in _all_channels:
            raise ValueError(f"Wrong input channel:{_ch}, should be within {_all_channels}")
    _ch_inds = [_all_channels.index(_ch) for _ch in _sel_channels]
    _ch_starts = [num_buffer_frames \
                    + (_i + num_empty_frames-num_buffer_frames) %_num_colors 
                    for _i in _ch_inds]
    if skip_frame0:
        for _i,_s in enumerate(_ch_starts):
            if _s == _num_buffer_frames:
                _ch_starts[_i] += _num_colors

    _splitted_ims = [im[_s:_s+single_im_size[0]*_num_colors:_num_colors].copy() for _s in _ch_starts]

    return _splitted_ims


def load_correction_profile(corr_type, corr_channels=_corr_channels, 
                            correction_folder=_correction_folder, all_channels=_allowed_colors,
                            ref_channel='647', im_size=_image_size, verbose=False):
    """Function to load chromatic/illumination correction profile
    Inputs:
        corr_type: type of corrections to be loaded
        corr_channels: all correction channels to be loaded
    Outputs:
        _pf: correction profile, np.ndarray for bleedthrough, dict for illumination/chromatic
    """
    ## check inputs
    # type
    _allowed_types = ['chromatic', 'illumination', 'bleedthrough']
    _type = str(corr_type).lower()
    if _type not in _allowed_types:
        raise ValueError(f"Wrong input corr_type, should be one of {_allowed_types}")
    # channel
    _all_channels = [str(_ch) for _ch in all_channels]
    _corr_channels = [str(_ch) for _ch in corr_channels]
    for _channel in _corr_channels:
        if _channel not in _all_channels:
            raise ValueError(f"Wrong input channel, should be one of {_all_channels}")
    _ref_channel = str(ref_channel).lower()
    if _ref_channel not in _all_channels:
        raise ValueError(f"Wrong input ref_channel, should be one of {_all_channels}")

    ## start loading file
    if verbose:
        print(f"-- loading {_type} correction profile from file", end=':')
    if _type == 'bleedthrough':
        _basename = _type+'_correction' \
            + '_' + '_'.join(sorted(_corr_channels, key=lambda v:-int(v))) \
            + '_' + str(im_size[-2])+'x'+str(im_size[-1])+'.npy'
        if verbose:
            print(_basename)
        _pf = np.load(os.path.join(correction_folder, _basename))
        _pf = _pf.reshape(len(_corr_channels), len(_corr_channels), im_size[-2], im_size[-1])
    elif _type == 'chromatic':
        if verbose:
            print('')
        _pf = {}
        for _channel in _corr_channels:
            if _channel != _ref_channel:
                _basename = _type+'_correction' \
                + '_' + str(_channel) + '_' + str(_ref_channel) \
                + '_' + str(im_size[-2])+'x'+str(im_size[-1])+'.npy'
                if verbose:
                    print('\t',_channel,_basename)
                _pf[_channel] = np.load(os.path.join(correction_folder, _basename))
            else:
                if verbose:
                    print('\t',_channel, None)
                _pf[_channel] = None
    elif _type == 'illumination':
        if verbose:
            print('')
        _pf = {}
        for _channel in _corr_channels:
            _basename = _type+'_correction' \
            + '_' + str(_channel) \
            + '_' + str(im_size[-2])+'x'+str(im_size[-1])+'.npy'
            if verbose:
                print('\t',_channel,_basename)
            _pf[_channel] = np.load(os.path.join(correction_folder, _basename))

    return _pf 
