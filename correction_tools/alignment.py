import os,time
import numpy as np
from .. import _allowed_colors, _image_size, _num_buffer_frames, _num_empty_frames, _image_dtype
from .. import _correction_folder


def generate_drift_crops(single_im_size=_image_size, coord_sel=None, drift_size=None):
    """Function to generate drift crop from a selected center and given drift size
    keywards:
        single_im_size: single image size to generate crops, np.ndarray like;
        coord_sel: selected center coordinate to split image into 4 rectangles, np.ndarray like;
        drift_size: size of drift crop, int or np.int;
    returns:
        crops: 4x3x2 np.ndarray. 
    """
    # check inputs
    _single_im_size = np.array(single_im_size)
    if coord_sel is None:
        coord_sel = np.array(_single_im_size/2, dtype=np.int)
    if coord_sel[-2] >= _single_im_size[-2] or coord_sel[-1] >= _single_im_size[-1]:
        raise ValueError(f"wrong input coord_sel:{coord_sel}, should be smaller than single_im_size:{single_im_size}")
    if drift_size is None:
        drift_size = int(np.max(_single_im_size)/3)
        
    # generate boundaries
    crop_0_bds = np.array([[0, _single_im_size[0]],
                           [0, coord_sel[-2]],
                           [0, coord_sel[-1]],
                          ], dtype=np.int)
    crop_1_bds = np.array([[0, _single_im_size[0]],
                           [0, coord_sel[-2]],
                           [coord_sel[-1], _single_im_size[-1]],
                          ], dtype=np.int)
    crop_2_bds = np.array([[0, _single_im_size[0]],
                           [coord_sel[-2], _single_im_size[-2]],
                           [0, coord_sel[-1]],
                          ], dtype=np.int)
    crop_3_bds = np.array([[0, _single_im_size[0]],
                           [coord_sel[-2], _single_im_size[-2]],
                           [coord_sel[-1], _single_im_size[-1]],
                          ], dtype=np.int)
    # generate crops
    crops = []
    for _bd in [crop_0_bds, crop_1_bds, crop_2_bds, crop_3_bds]:
        _ct = np.mean(_bd, axis=1).astype(np.int)
        
        _crop = np.array([_bd[0],
                      [max(_ct[-2]-drift_size/2, _bd[-2,0]), 
                       min(_ct[-2]+drift_size/2, _bd[-2,1])],
                      [max(_ct[-1]-drift_size/2, _bd[-1,0]), 
                       min(_ct[-1]+drift_size/2, _bd[-1,1])], 
                     ], dtype=np.int)
        crops.append(_crop)
        
    return np.array(crops)



def align_beads(tar_cts, ref_cts, 
                tar_im=None, ref_im=None,
                use_fft=True, fft_filt_size=0, 
                match_distance_th=2., 
                check_paired_cts=True,
                outlier_sigma=1.5,
                return_paired_cts=True,
                verbose=True):
    """Align single bead image to return drifts
        with two options:
            not use_fft: slow: enumerate all possible combinations of center pairs, keep good ones, canculate drift between pairs.
            use_fft: use FFT to get pixel-level traslation, then calculate finer drift.
    Inputs:
        tar_cts: target centers from target image, 2d-numpy array
        ref_cts: reference centers from reference image, 2d-numpy array
        tar_im: target image, np.ndarray, required if use_fft
        ref_im: reference image, np.ndarray, required if use_fft
        fft_filt_size: blurring filter size before FFT, int(default: 0)
        match_distance_th: threshold to uniquely match centers, float (default: 2.)
        check_paired_cts: do extra checking for a bead pair, whether it is close to its neighboring drifts, bool (default: True)
        outlier_sigma: gaussian sigma for outlier bead shift, float (default: 1.5 (over this threshold))
        return_paired_cts: whether return paired centers, bool (default: True)
        verbose: say something!, bool (default: True)
    Outputs:
        _mean_shift: mean drift calculated from paired centers, 1d numpy array of dim,
    conditional outputs:
        _paired_tar_cts: paired target centers, 2d numpy arrray of n_spots*dim
        _paired_ref_cts: paired reference centers, 2d numpy arrray of n_spots*dim    
    """
    # convert inputs
    _tar_cts = np.array(tar_cts)
    _ref_cts = np.array(ref_cts)
    _distance_th = float(match_distance_th)
    from ..alignment_tools import fft3d_from2d
    from ..spot_tools.matching import find_paired_centers, check_paired_centers
    # case 1: directly align centers by brute force
    if not use_fft:
        from ..alignment_tools import translation_align_pts

        # calculate drift
        _drift, _paired_ref_cts, _paired_tar_cts = translation_align_pts(
            _ref_cts, _tar_cts, 
            cutoff=_distance_th, return_pts=True,
            verbose=verbose,
        )
    # case 2: fft align images and match centers
    else:
        if tar_im is None or ref_im is None:
            raise ValueError(f"both tar_im and ref_im should be given if use FFT!")
        if np.shape(tar_im) != np.shape(ref_im):
            raise IndexError(f"tar_im shape:{np.shape(tar_im)} should match ref_im shape:{np.shape(ref_im)}")
        # do rough alignment
        _rough_drift = fft3d_from2d(tar_im, ref_im, 
                                    gb=fft_filt_size,
                                    max_disp=np.max(np.shape(tar_im))/2)
        # matche centers
        _drift, _paired_tar_cts, _paired_ref_cts = find_paired_centers(
            _tar_cts, _ref_cts, _rough_drift,
            cutoff=_distance_th, return_paired_cts=True,
            verbose=verbose,
        )
    print("before check:", _drift, len(_paired_ref_cts))
    # check paired centers
    if check_paired_cts and len(_paired_ref_cts) > 3:
        _drift, _paired_tar_cts, _paired_ref_cts = check_paired_centers(
            _paired_tar_cts, _paired_ref_cts, 
            outlier_sigma=outlier_sigma,
            return_paired_cts=True,
            verbose=verbose,
        )
    
    # return
    _return_args = [_drift]
    if return_paired_cts:
        _return_args.append(_paired_tar_cts)
        _return_args.append(_paired_ref_cts)

    return tuple(_return_args)

# basic function to align single image
def align_single_image(filename, crop_list, bead_channel='488',
                       all_channels=_allowed_colors, 
                       single_im_size=_image_size,
                       num_buffer_frames=_num_buffer_frames,
                       num_empty_frames=_num_empty_frames,
                       illumination_corr=True, 
                       correction_folder=_correction_folder,
                       ref_filename=None, ref_all_channels=None,
                       ref_centers=None, ref_ims=None,
                       th_seed=100, th_seed_per=98, use_percentile=False,
                       max_num_seeds=None, min_num_seeds=50, 
                       fitting_kwargs={}, 
                       use_fft=True, fft_filt_size=0, 
                       match_distance_th=2., 
                       check_paired_cts=True,
                       outlier_sigma=1.5,
                       good_drift_th=1.,
                       return_target_ims=False,
                       return_paired_cts=False,
                       verbose=False,                       
                       ):
    """Function to align one single image
    Inputs:
    
    Outputs:
    """
    from scipy.spatial.distance import cdist, pdist, squareform
    from ..io_tools.load import correct_fov_image
    from ..alignment_tools import fft3d_from2d
    from ..spot_tools.fitting import get_centers
    ## check inputs
    # check crop_list:
    if len(crop_list) < 2:
        raise IndexError(f"crop_list should at least have 2 elements")
    elif len(crop_list[0]) != len(single_im_size):
        raise IndexError("dimension of crop_list should match single_im_size")
    # check channels:
    _all_channels = [str(_ch) for _ch in all_channels]
    # check bead_channel
    _bead_channel = str(bead_channel)
    if _bead_channel not in all_channels:
        raise ValueError(f"bead channel {_bead_channel} not exist in all channels given:{_all_channels}")
    # check ref_all_channels
    if ref_all_channels is None:
        _ref_all_channels = _all_channels
    else:
        _ref_all_channels = [str(_ch) for _ch in ref_all_channels]

    # check filename file type   
    if isinstance(filename, np.ndarray):
        if verbose:
            print(f"-- start aligning given image to", end=' ')
        _bead_im = filename
        if np.shape(_bead_im) != tuple(single_im_size):
            raise IndexError(f"shape of target image:{np.shape(_bead_im)} and single_im_size:{single_im_size} doesn't match!")
    elif isinstance(filename, str):
        if verbose:
            print(f"-- start aligning file {filename} to", end=' ')
        if not os.path.isfile(filename) or filename.split('.')[-1] != 'dax':
            raise IOError(f"input filename: {filename} should be a .dax file!")
        _bead_im = correct_fov_image(filename, [_bead_channel], 
                                     single_im_size=single_im_size, 
                                     all_channels=all_channels,
                                     num_buffer_frames=num_buffer_frames, 
                                     num_empty_frames=num_empty_frames, 
                                     calculate_drift=False, 
                                     correction_folder=correction_folder,
                                     illumination_corr=illumination_corr,
                                     bleed_corr=False, chromatic_corr=False,
                                     z_shift_corr=False, hot_pixel_corr=True,
                                     normalization=False, return_drift=False,
                                     verbose=False,
                                     )[0]
    else:
        raise IOError(f"Wrong input file type, {filename} should be .dax file or np.ndarray")
    # crop target image:
    _tar_ims = [_bead_im[tuple([slice(_s[0], _s[-1]) for _s in _c])] for _c in crop_list]
    # get centers
    _tar_ct_list = [get_centers(_im, th_seed=th_seed,
                                    th_seed_per=th_seed_per, 
                                    use_percentile=use_percentile,
                                    max_num_seeds=max_num_seeds, 
                                    min_num_seeds=min_num_seeds,
                                    **fitting_kwargs,
                                    ) for _im in _tar_ims]

    ## acquire references
    # case 1: ref_centers and ref_ims are given:
    if ref_centers is not None and ref_ims is not None:
        if verbose:
            print(f"given ref_centers and images, n={len(ref_centers)}")
        if len(ref_centers) != len(ref_ims):
            raise IndexError(f"length of ref_centers:{len(ref_centers)} should match length of ref_ims:{len(ref_ims)}")
        elif len(crop_list) != len(ref_centers):
            raise IndexError(f"length of crop_list:{len(crop_list)} should match length of ref_centers:{len(ref_centers)}")
        _ref_ims = ref_ims
        _ref_ct_list = ref_centers
    # case 2: ref_filename is given:
    elif ref_filename is not None:
        if isinstance(ref_filename, np.ndarray):
            if verbose:
                print(f"ref image directly given")
            _ref_bead_im = ref_filename
        elif isinstance(ref_filename, str):
            if verbose:
                print(f"ref_file: {ref_filename}")
            _ref_bead_im = old_correct_fov_image(ref_filename, [_bead_channel], 
                                        single_im_size=single_im_size, 
                                        all_channels=all_channels,
                                        num_buffer_frames=num_buffer_frames,
                                        num_empty_frames=num_empty_frames, 
                                        calculate_drift=False, 
                                        correction_folder=correction_folder,
                                        illumination_corr=illumination_corr,
                                        warp_image=False,
                                        bleed_corr=False, 
                                        chromatic_corr=False,
                                        z_shift_corr=False, 
                                        hot_pixel_corr=True,
                                        normalization=False, 
                                        return_drift=False,
                                        verbose=False,
                                        )[0][0]
        _ref_ims = []
        for _c in crop_list:
            _crop = tuple([slice(int(_s[0]), int(_s[-1])) for _s in _c])
            _ref_ims.append(_ref_bead_im[_crop])
        # collect ref_ct_list
        from ..spot_tools.fitting import select_sparse_centers
        _ref_ct_list = []
        for _im in _ref_ims:
            _cand_cts = get_centers(_im, th_seed=th_seed,
                                    th_seed_per=th_seed_per, 
                                    use_percentile=use_percentile,
                                    max_num_seeds=max_num_seeds, 
                                    min_num_seeds=min_num_seeds,
                                    **fitting_kwargs,
                                    )
            _ref_ct_list.append(select_sparse_centers(_cand_cts, 
                                distance_th=match_distance_th))
    else:
        raise ValueError(f"ref_filename or ref_centers+ref_ims should be given!")
    
    # Do alignment
    _drift_list = []
    _paired_tar_ct_list = []
    _paired_ref_ct_list = []
    # iterate until find good drifts or calculated all cropped images
    while len(_drift_list) < len(crop_list):
        # get image
        _cid = len(_drift_list)
        # calculate drift
        _drift, _paired_tar_cts, _paired_ref_cts = align_beads(
            _tar_ct_list[_cid], _ref_ct_list[_cid], 
            _tar_ims[_cid], _ref_ims[_cid],
            use_fft=use_fft, 
            fft_filt_size=fft_filt_size, 
            match_distance_th=match_distance_th, 
            check_paired_cts=check_paired_cts,
            outlier_sigma=outlier_sigma,
            return_paired_cts=True,
            verbose=verbose,
        )
        # judge whether this matching is successful
        if len(_paired_tar_cts) == 0:
            _drift = np.inf * np.ones(len(single_im_size))
        # append
        _drift_list.append(_drift)
        _paired_tar_ct_list.append(_paired_tar_cts)
        _paired_ref_ct_list.append(_paired_ref_cts)

        # check if matched well: 
        if len(_drift_list) >=2:
            if (cdist(_drift[np.newaxis,:], _drift_list[:-1])[0] < good_drift_th).any():
                break
    ## select drifts
    _dists = squareform(pdist(_drift_list))
    _dists[np.arange(len(_dists)), np.arange(len(_dists))] = np.inf
    _inds = np.unravel_index(np.argmin(_dists, axis=None), _dists.shape)
    # get the two that are closest
    if _dists[_inds] > good_drift_th:
        _success_flag = False
        print(f"-- Suspicious Failure: selcted drifts: {_drift_list[_inds[0]]}, {_drift_list[_inds[1]]} are not close enough.")
    else:
        _success_flag = True
    # extract _final_drift and return
    _final_drift = np.nanmean([_drift_list[_inds[0]], _drift_list[_inds[1]]], axis=0)

    # return
    _return_args = [_final_drift, _success_flag]
    if return_target_ims:
        _return_args.append(_tar_ims)
    if return_paired_cts:
        _return_args.append(_paired_tar_ct_list)
        _return_args.append(_paired_ref_ct_list)
    
    return tuple(_return_args)


# basic function to align single image
def cross_correlation_align_single_image(im, ref_im, precision_fold=100,
                                         all_channels=_allowed_colors, 
                                         ref_all_channels=None, drift_channel='488',
                                         single_im_size=_image_size,
                                         num_buffer_frames=_num_buffer_frames,
                                         num_empty_frames=_num_empty_frames,
                                         correction_folder=_correction_folder,
                                         correction_args={},
                                         return_all=False,
                                         verbose=True, detailed_verbose=False,                      
                       ):
    """Function to align one single image by FFT
    Inputs:
        im: 
    Outputs:
    """
    
    if verbose:
        print(f"-- aligning image", end=' ')
        if isinstance(im, str):
            print(os.path.join(os.path.basename(os.path.dirname(im))), os.path.basename(im), end=' ')
        if isinstance(ref_im, str):
            print('to '+os.path.join(os.path.basename(os.path.dirname(ref_im))), os.path.basename(ref_im), end=' ')
    
    # set default correction args
    _correction_args = {
        'hot_pixel_corr':True,
        'hot_pixel_th':4,
        'z_shift_corr':False,
        'illumination_corr':True,
        'illumination_profile':None,
        'bleed_corr':False,
        'chromatic_corr':False,
        'normalization':False,
    }
    _correction_args.update(correction_args)
    
    # check im file type   
    if isinstance(im, np.ndarray):
        if verbose:
            print(f"-> directly use image")
        _im = im.copy()
        if np.shape(_im) != tuple(np.array(single_im_size)):
            raise IndexError(f"shape of im:{np.shape(_im)} and single_im_size:{single_im_size} doesn't match!")
    elif isinstance(im, str):
        if 'correct_fov_image' not in locals():
            from ..io_tools.load import correct_fov_image
        #  load image
        _im = correct_fov_image(im, [drift_channel], 
                                  single_im_size=single_im_size, all_channels=all_channels,
                                  num_buffer_frames=num_buffer_frames, num_empty_frames=num_empty_frames, 
                                  drift=[0,0,0], calculate_drift=False, drift_channel=drift_channel, 
                                  correction_folder=correction_folder, warp_image=False, verbose=detailed_verbose, 
                                  **_correction_args)[0][0]
    # check ref_im file type   
    if isinstance(ref_im, np.ndarray):
        if verbose:
            print(f"-- directly use ref_image")
        _ref_im = ref_im
        if np.shape(_ref_im) != tuple(np.array(single_im_size)):
            raise IndexError(f"shape of ref_im:{np.shape(_ref_im)} and single_im_size:{single_im_size} doesn't match!")
    elif isinstance(ref_im, str):
        if 'correct_fov_ref_image' not in locals():
            from ..io_tools.load import correct_fov_image
        _ref_im = correct_fov_image(ref_im, [drift_channel], 
                                      single_im_size=single_im_size, all_channels=all_channels,
                                      num_buffer_frames=num_buffer_frames, num_empty_frames=num_empty_frames, 
                                      drift=[0,0,0], calculate_drift=False, drift_channel=drift_channel, 
                                      correction_folder=correction_folder, warp_image=False, verbose=detailed_verbose, 
                                      **_correction_args)[0][0]
    
    # align by cross-correlation
    from skimage.feature import register_translation
    _start_time = time.time()
    _drift, _error, _phasediff = register_translation(_ref_im, _im, 
                                                      upsample_factor=precision_fold)
    
    # return
    if return_all:
        return _drift, _error, _phasediff
    else:
        return _drift
    
def generate_translation_from_DAPI(old_dapi_im, new_dapi_im, 
                                   old_to_new_rotation,
                                   drift=None,
                                   fft_gb=0, fft_max_disp=200, 
                                   image_dtype=_image_dtype,
                                   verbose=True):
    """Function to generate translation matrix required in cv2. 
        - Only allow X-Y translation """
    ## check inputs
    from math import pi
    import cv2
    from ..alignment_tools import fft3d_from2d
    if np.shape(old_to_new_rotation)[0] != 2 or np.shape(old_to_new_rotation)[1] != 2 or len(np.shape(old_to_new_rotation)) != 2:
        raise IndexError(f"old_to_new_rotation should be a 2x2 rotation matrix!, but {np.shape(old_to_new_rotation)} is given.")
    if drift is None:
        drift = np.zeros(len(old_dapi_im.shape))
    else:
        drift = np.array(drift)
    ## 1. rotate new dapi im at its center
    if verbose:
        print(f"-- start calculating drift between DAPI images")
    # get dimensions
    _dz,_dx,_dy = np.shape(old_dapi_im)
    # calculate cv2 rotation inputs from given rotation_mat
    
    _rotation_angle = np.arcsin(old_to_new_rotation[0,1])/pi*180
    _temp_new_rotation_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), _rotation_angle, 1) # 
    # generated rotated image by rotation at the x-y center
    _rot_new_im = np.array([cv2.warpAffine(_lyr, _temp_new_rotation_M, 
                                           _lyr.shape, borderMode=cv2.BORDER_DEFAULT) 
                            for _lyr in new_dapi_im], dtype=image_dtype)
    ## 2. calculate drift by FFT
    _dapi_shift = fft3d_from2d(old_dapi_im, _rot_new_im, max_disp=fft_max_disp, gb=fft_gb)
    if verbose:
        print(f"-- start generating translated segmentation labels")
    # define mat to translate old mat into new ones
    _rotate_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), -_rotation_angle, 1)
    _rotate_M[:,2] -= np.flipud(_dapi_shift[-2:])
    _rotate_M[:,2] -= np.flipud(drift[-2:])

    return _rot_new_im, _rotate_M


def calculate_translation(reference_im:np.ndarray, 
                          target_im:np.ndarray,
                          ref_to_tar_rotation:np.ndarray=None,
                          cross_corr_align:bool=True,
                          alignment_kwargs:dict={},
                          verbose:bool=True,
                          ):
    """Calculate translation between two images with rotation """
    from math import pi
    import cv2
    ## quality check
    # images
    if np.shape(reference_im) != np.shape(target_im):
        raise IndexError(f"two images should be of the same shape")
    # rotation matrix
    if ref_to_tar_rotation is None:
        ref_to_tar_rotation = np.diag([1,1])
    elif np.shape(ref_to_tar_rotation) != tuple([2,2]):
        raise IndexError(f"wrong shape for rotation matrix, should be 2x2. ")
    # get dimensions
    _dz,_dx,_dy = np.shape(reference_im)
    # calculate angle
    if verbose:
        print(f"-- start calculating drift with rotation between images")
    _rotation_angle = np.arcsin(ref_to_tar_rotation[0,1])/pi*180
    _temp_new_rotation_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), _rotation_angle, 1) # temporary rotation angle
    # rotate image
    if _rotation_angle != 0:
        _rot_target_im = np.array([cv2.warpAffine(_lyr, _temp_new_rotation_M, 
                                                  _lyr.shape, borderMode=cv2.BORDER_DEFAULT) 
                                   for _lyr in target_im], dtype=reference_im.dtype)
    else:
        _rot_target_im = target_im
    # calculate drift
    if cross_corr_align:
        _drift = cross_correlation_align_single_image(_rot_target_im,
                                                      reference_im,
                                                      single_im_size=np.array(np.shape(reference_im)),
                                                      **alignment_kwargs,)
    else:
        _drift_crops = generate_drift_crops()
        _drift = align_single_image(reference_im, _drift_crops,
                                    single_im_size=np.array(np.shape(reference_im)), 
                                    ref_filename=_rot_target_im, **alignment_kwargs)
    if verbose:
        print(f"--- drift: {np.round(_drift,2)} pixels")
        
    return _rot_target_im, ref_to_tar_rotation, _drift

# updated function to align simge image:

_default_align_corr_args={
    'single_im_size':_image_size,
    'num_buffer_frames':_num_buffer_frames,
    'num_empty_frames':_num_empty_frames,
    'correction_folder':_correction_folder,
    'illumination_corr':True,
    'bleed_corr': False, 
    'chromatic_corr': False,
    'z_shift_corr': False, 
    'hot_pixel_corr': True,
    'normalization': False,
}

_default_align_fitting_args={
    'th_seed': 300,
    'th_seed_per': 95, 
    'use_percentile': False,
    'use_dynamic_th': True,
    'min_dynamic_seeds': 10,
    'max_num_seeds': 200,
}


def align_image(
    src_im:np.ndarray, 
    ref_im:np.ndarray, 
    crop_list=None,
    use_autocorr=True, precision_fold=100, 
    drift_diff_th=1.,
    all_channels=_allowed_colors, 
    ref_all_channels=None, 
    drift_channel='488',
    correction_args={},
    fitting_args={},
    match_distance_th=2.,
    return_all=False,
    verbose=True, detailed_verbose=False,                      
    ):
    """Function to align one image by either FFT or spot_finding"""
    
    from scipy.spatial.distance import cdist, pdist, squareform
    from ..io_tools.load import correct_fov_image
    from ..alignment_tools import fft3d_from2d
    from ..spot_tools.fitting import fit_fov_image
    from ..spot_tools.fitting import select_sparse_centers
    from skimage.feature import register_translation
    ## check inputs
    # correciton keywords
    _correction_args = {_k:_v for _k,_v in _default_align_corr_args.items()}
    _correction_args.update(correction_args)
    # fitting keywords
    _fitting_args = {_k:_v for _k,_v in _default_align_fitting_args.items()}
    _fitting_args.update(fitting_args)
    
    # check crop_list:
    if crop_list is None:
        crop_list = generate_drift_crops(_correction_args['single_im_size'])
    for _crop in crop_list:
        if np.shape(np.array(_crop)) != (3,2):
            raise IndexError(f"crop should be 3x2 np.ndarray.")
    # check channels
    _all_channels = [str(_ch) for _ch in all_channels]
    # check bead_channel
    _drift_channel = str(drift_channel)
    if _drift_channel not in all_channels:
        raise ValueError(f"bead channel {_drift_channel} not exist in all channels given:{_all_channels}")
    # check ref_all_channels
    if ref_all_channels is None:
        _ref_all_channels = _all_channels
    else:
        _ref_all_channels = [str(_ch) for _ch in ref_all_channels]
    
    ## process source image
    if isinstance(src_im, np.ndarray):
        if verbose:
            print(f"-- start aligning given source image to", end=' ')
        _src_im = src_im
        if np.shape(_src_im) != tuple(_correction_args['single_im_size']):
            _size = tuple(_correction_args['single_im_size'])
            raise IndexError(f"shape of target image:{np.shape(_ref_im)} and single_im_size:{_size} doesnt match!")
    elif isinstance(src_im, str):
        if verbose:
            print(f"-- start aligning file {src_im}.", end=' ')
        if not os.path.isfile(src_im) or src_im.split('.')[-1] != 'dax':
            raise IOError(f"input src_im: {src_im} should be a .dax file!")
        _src_im = correct_fov_image(src_im, [_drift_channel], 
                                    all_channels=_all_channels,
                                    calculate_drift=False, 
                                    return_drift=False, verbose=detailed_verbose,
                                    **_correction_args)[0]
    else:
        raise IOError(f"Wrong input file type, {type(src_im)} should be .dax file or np.ndarray")
    
    ## process reference image
    if isinstance(ref_im, np.ndarray):
        if verbose:
            print(f"given reference image.")
        _ref_im = ref_im
        if np.shape(_ref_im) != tuple(_correction_args['single_im_size']):
            _size = tuple(_correction_args['single_im_size'])
            raise IndexError(f"shape of reference image:{np.shape(_ref_im)} and single_im_size:{_size} doesnt match!")
    elif isinstance(ref_im, str):
        if verbose:
            print(f"reference file:{ref_im}.")
        if not os.path.isfile(ref_im) or ref_im.split('.')[-1] != 'dax':
            raise IOError(f"input ref_im: {ref_im} should be a .dax file!")
        _ref_im = correct_fov_image(ref_im, [_drift_channel], 
                                    all_channels=_ref_all_channels,
                                    calculate_drift=False, 
                                    return_drift=False, verbose=detailed_verbose,
                                    **_correction_args)[0]
    else:
        raise IOError(f"Wrong input ref file type, {type(ref_im)} should be .dax file or np.ndarray")
    
    ## crop images
    _crop_src_ims, _crop_ref_ims = [], []
    for _crop in crop_list:
        _s = ([slice(*np.array(_c,dtype=np.int)) for _c in _crop])
        _crop_src_ims.append(_src_im[_s])
        _crop_ref_ims.append(_ref_im[_s])
        
    ## align two images
    _drifts, _errors, _diffs = [], [], []
    if use_autocorr:
        if verbose:
            print("--- use auto correlation to calculate drift.")
        for _i, (_sim, _rim) in enumerate(zip(_crop_src_ims, _crop_ref_ims)):
            _start_time = time.time()
            _drift, _error, _phasediff = register_translation(_sim, _rim, 
                                                              upsample_factor=precision_fold)
            # append
            _drifts.append(_drift)
            _errors.append(_error)
            _diffs.append(_phasediff)
            if verbose:
                print(f"--- align image {_i} in {time.time()-_start_time:.3f}s.")
    else:
        if verbose:
            print("--- use spot finding to calculate drift.")
        for _i, (_sim, _rim) in enumerate(zip(_crop_src_ims, _crop_ref_ims)):
            _start_time = time.time()
            # source
            _src_spots = fit_fov_image(_sim, _drift_channel, 
                verbose=detailed_verbose,
                **_fitting_args) # fit source spots
            _sp_src_cts = select_sparse_centers(_src_spots[:,1:4], match_distance_th) # select sparse source spots
            # reference
            _ref_spots = fit_fov_image(_rim, _drift_channel, 
                verbose=detailed_verbose,
                **_fitting_args)
            _sp_ref_cts = select_sparse_centers(_ref_spots[:,1:4], match_distance_th) # select sparse ref spots
            #print(_sp_src_cts, _sp_ref_cts)
            
            # align
            _dft, _paired_src_cts, _paired_ref_cts = align_beads(
                _sp_src_cts, _sp_ref_cts,
                _sim, _rim,
                use_fft=True,
                match_distance_th=match_distance_th, 
                return_paired_cts=True,
                verbose=detailed_verbose,
            )
            _drifts.append(_dft)
            _errors.append(0)
            _diffs.append(0)
            if verbose:
                print(f"--- align image {_i} in {time.time()-_start_time:.3f}s.")
                print(_dft)
                
    return np.nanmean(_drifts, axis=1) #, _errors, _diffs
