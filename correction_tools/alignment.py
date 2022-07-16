import os,time
import numpy as np
from .. import _allowed_colors, _image_size, _num_buffer_frames, _num_empty_frames, _image_dtype
from .. import _correction_folder

## alignment between re-labeled sample
def align_manual_points(pos_file_before, pos_file_after,
    save=True, save_folder=None, save_filename='', 
    overwrite=False, verbose=True):
    """Function to align two manually picked position files, 
    they should follow exactly the same order and of same length.
    Inputs:
        pos_file_before: full filename for positions file before translation
        pos_file_after: full filename for positions file after translation
        save: whether save rotation and translation info, bool (default: True)
        save_folder: where to save rotation and translation info, None or string (default: same folder as pos_file_before)
        save_filename: filename specified to save rotation and translation points
        verbose: say something! bool (default: True)
    Outputs:
        R: rotation for positions, 2x2 array
        T: traslation of positions, array of 2
    Here's example for how to translate points
        translated_ps_before = np.dot(ps_before, R) + t
    """
    # load position_before
    if os.path.isfile(pos_file_before):
        ps_before = np.loadtxt(pos_file_before, delimiter=',')
    else:
        raise IOError(
            f"- Position file:{pos_file_before} file doesn't exist, exit!")
    # load position_after
    if os.path.isfile(pos_file_after):
        ps_after = np.loadtxt(pos_file_after, delimiter=',')
    else:
        raise IOError(
            f"- Position file:{pos_file_after} file doesn't exist, exit!")
    # do SVD decomposition to get best fit for rigid-translation
    c_before = np.mean(ps_before, axis=0)
    c_after = np.mean(ps_after, axis=0)
    H = np.dot((ps_before - c_before).T, (ps_after - c_after))
    U, _, V = np.linalg.svd(H)  # do SVD
    # calcluate rotation
    R = np.dot(V, U.T).T
    if np.linalg.det(R) < 0:
        R[:, -1] = -1 * R[:, -1]
    # calculate translation
    t = - np.dot(c_before, R) + c_after
    # here's example for how to translate points
    # translated_ps_before = np.dot(ps_before, R) + t
    if verbose:
        print(
            f"- Manually picked points aligned, rotation:\n{R},\n translation:{t}")
    # save
    if save:
        if save_folder is None:
            save_folder = os.path.dirname(pos_file_before)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if len(save_filename) > 0:
            save_filename += '_'
        rotation_name = os.path.join(save_folder, save_filename+'rotation')
        if not os.path.exists(rotation_name+'.npy') or overwrite:
            np.save(rotation_name, R)
            if verbose:
                print(f'-- rotation matrix saved to file:{rotation_name}.npy')
        else:
            if verbose:
                print(f'-- {rotation_name}.npy exists, skip.')
        translation_name = os.path.join(save_folder, save_filename+'translation')        
        if not os.path.exists(translation_name+'.npy') or overwrite:      
            np.save(translation_name, t)
            if verbose:
                print(f'-- translation matrix saved to file:{translation_name}.npy')
        else:
            if verbose:
                print(f'-- {translation_name}.npy exists, skip.')
    return R, t


def _find_boundary(_ct, _radius, _im_size):
    _bds = []
    for _c, _sz in zip(_ct, _im_size):
        _bds.append([max(_c-_radius, 0), min(_c+_radius, _sz)])
    
    return np.array(_bds, dtype=np.int)

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
        drift_size = int(np.max(_single_im_size)/4)
        
    # generate crop centers
    crop_cts = [
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1],]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1],]),                               
    ]
    # generate boundaries
    crops = [_find_boundary(_ct, _radius=drift_size/2, _im_size=single_im_size) for _ct in crop_cts]
        
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
    from skimage.registration import phase_cross_correlation
    _start_time = time.time()
    _drift, _error, _phasediff = phase_cross_correlation(_ref_im, _im, 
                                                         upsample_factor=precision_fold)
    
    # return
    if return_all:
        return _drift, _error, _phasediff
    else:
        return _drift

# updated function to align drift of single image:

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
    min_good_drifts=3, drift_diff_th=1.,
    all_channels=_allowed_colors, 
    ref_all_channels=None, 
    drift_channel='488',
    correction_args={},
    fitting_args={},
    match_distance_th=2.,
    verbose=True, 
    detailed_verbose=False,                      
    ):
    """Function to align one image by either FFT or spot_finding"""
    
    from ..io_tools.load import correct_fov_image
    from ..spot_tools.fitting import fit_fov_image
    from ..spot_tools.fitting import select_sparse_centers
    from skimage.registration import phase_cross_correlation
    #print("**", type(src_im), type(ref_im))
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
    # define result flag
    _result_flag = 0
    # process image
    if isinstance(src_im, np.ndarray):
        if verbose:
            print(f"-- start aligning given source image to", end=' ')
        _src_im = src_im
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
    elif isinstance(ref_im, str):
        if verbose:
            print(f"reference file:{ref_im}.")
        if not os.path.isfile(ref_im) or ref_im.split('.')[-1] != 'dax':
            raise IOError(f"input ref_im: {ref_im} should be a .dax file!")
        _ref_im = correct_fov_image(ref_im, [_drift_channel], 
                                    all_channels=_ref_all_channels,
                                    calculate_drift=False, 
                                    return_drift=False, verbose=detailed_verbose,
                                    **_correction_args)[0][0]
    else:
        raise IOError(f"Wrong input ref file type, {type(ref_im)} should be .dax file or np.ndarray")

    if np.shape(_src_im) != np.shape(_ref_im):
        raise IndexError(f"shape of target image:{np.shape(_src_im)} and reference image:{np.shape(_ref_im)} doesnt match!")

    ## crop images
    _crop_src_ims, _crop_ref_ims = [], []
    for _crop in crop_list:
        _s = tuple([slice(*np.array(_c,dtype=np.int)) for _c in _crop])
        _crop_src_ims.append(_src_im[_s])
        _crop_ref_ims.append(_ref_im[_s])
    ## align two images
    _drifts = []
    for _i, (_sim, _rim) in enumerate(zip(_crop_src_ims, _crop_ref_ims)):
        _start_time = time.time()
        if use_autocorr:
            if detailed_verbose:
                print("--- use auto correlation to calculate drift.")
            # calculate drift with autocorr
            _dft, _error, _phasediff = phase_cross_correlation(_rim, _sim, 
                                                               upsample_factor=precision_fold)
        else:
            if detailed_verbose:
                print("--- use beads fitting to calculate drift.")
            # source
            _src_spots = fit_fov_image(_sim, _drift_channel, 
                verbose=detailed_verbose,
                **_fitting_args) # fit source spots
            _sp_src_cts = select_sparse_centers(_src_spots[:,1:4], match_distance_th) # select sparse source spots
            # reference
            _ref_spots = fit_fov_image(_rim, _drift_channel, 
                verbose=detailed_verbose,
                **_fitting_args)
            _sp_ref_cts = select_sparse_centers(_ref_spots[:,1:4], match_distance_th, 
                                                verbose=detailed_verbose) # select sparse ref spots
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
            _dft = _dft * -1 # beads center is the opposite as cross correlation
        # append 
        _drifts.append(_dft) 
        if verbose:
            print(f"-- drift {_i}: {np.around(_dft, 2)} in {time.time()-_start_time:.3f}s.")

        # detect variance within existing drifts
        _mean_dft = np.nanmean(_drifts, axis=0)
        if len(_drifts) >= min_good_drifts:
            _dists = np.linalg.norm(_drifts-_mean_dft, axis=1)
            _kept_drift_inds = np.where(_dists <= drift_diff_th)[0]
            if len(_kept_drift_inds) >= min_good_drifts:
                _updated_mean_dft = np.nanmean(np.array(_drifts)[_kept_drift_inds], axis=0)
                _result_flag += 0
                if verbose:
                    print(f"--- drifts for crops:{_kept_drift_inds} pass the thresold, exit cycle.")
                break
    
    if '_updated_mean_dft' not in locals():
        if verbose:
            print(f"-- return a sub-optimal drift")
        _drifts = np.array(_drifts)
        # select top 3 drifts
        from scipy.spatial.distance import pdist, squareform
        _dist_mat = squareform(pdist(_drifts))
        np.fill_diagonal(_dist_mat, np.inf)
        # select closest pair
        _sel_inds = np.array(np.unravel_index(np.argmin(_dist_mat), np.shape(_dist_mat)))
        _sel_drifts = list(_drifts[_sel_inds])
        # select closest 3rd drift
        _sel_drifts.append(_drifts[np.argmin(_dist_mat[:, _sel_inds].sum(1))])
        if detailed_verbose:
            print(f"--- select drifts: {np.round(_sel_drifts, 2)}")
        # return mean
        _updated_mean_dft = np.nanmean(_sel_drifts, axis=0)
        _result_flag += 1

    return  _updated_mean_dft, _result_flag

# Function to calculate translation for alignment of re-mounted samples
def calculate_translation(reference_im:np.ndarray, 
                          target_im:np.ndarray,
                          ref_to_tar_rotation:np.ndarray=None,
                          use_autocorr:bool=True,
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
    _drift, _drift_flag = align_image(
        _rot_target_im,
        reference_im,
        precision_fold=10,
        use_autocorr=use_autocorr,
        verbose=verbose,
        #detailed_verbose=verbose,
        **alignment_kwargs,)

    if verbose:
        print(f"--- drift: {np.round(_drift,2)} pixels")
        
    return _rot_target_im, ref_to_tar_rotation, _drift