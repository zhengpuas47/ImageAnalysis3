from . import get_img_info, visual_tools, alignment_tools, analysis, classes
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy,_image_size,_allowed_colors
from .External import Fitting_v3
import numpy as np
import scipy
import pickle
import matplotlib.pylab as plt
import os, glob, sys, time
from scipy.stats import scoreatpercentile
import multiprocessing as mp
import ctypes
from scipy.ndimage.interpolation import map_coordinates
def __init__():
    pass

# merged function to calculate bead drift directly from files


def Calculate_Bead_Drift(folders, fovs, fov_id, num_threads=12, drift_size=500, ref_id=0,
                         sequential_mode=False, bead_channel='488', all_channels=_allowed_colors,
                         illumination_corr=True, correction_folder=_correction_folder,
                         coord_sel=None, single_im_size=_image_size, num_buffer_frames=10,
                         match_distance=3, match_unique=True, rough_drift_gb=0,
                         max_ref_points=500, ref_seed_per=95, drift_cutoff=1,
                         save=True, save_folder=None, save_postfix='_current_cor.pkl',
                         overwrite=False, verbose=True):
    """Function to generate drift profile given a list of corrected bead files
    Inputs:
        folders: hyb-folder names planned for drift-correction, list of folders
        fovs: filenames for all field-of-views, list of filenames
        fov_id: selected fov_id to do drift_corrections, int
        num_threads: number of threads used for drift correction, int (default: 12)
        drift_size: selected sub-figure size for bead-fitting, int (default: 500)
        ref_id: id for reference hyb-folder, int (default: 0)
            (only effective if sequential_mode is False)
        sequential_mode: whether align drifts sequentially, bool (default: False)
            (if sequential_mode is false that means align all ims to ref_id im)
        bead_channel: channel for beads, int or str (default: '488')
        all_channels: all allowed channels, list of str (default: _alowed_colors)
        illumination_corr: whether do illumination correction, bool (default: True)
        correction_folder: where to find correction_profile, str for folder path (default: default)
        coord_sel: selected coordinate to pick windows nearby, array of 2 (default: None, center of image)
        single_im_size: image size of single channel 3d image, array of 3 (default: _image_size)
        num_buffer_frames: number of buffer frames in zscan, int (default: 10)
        max_ref_points: maximum allowed reference points, int (default: 500)
        ref_seed_per: seeding intensity percentile for ref-ims, float (default: 95)
        save: whether save drift result dictionary to pickle file, bool (default: True)
        save_folder: folder to save drift result, required if save is True, string of path (default: None)
        save_postfix: drift file postfix, string (default: '_sequential_current_cor.pkl')
        overwrite: whether overwrite existing drift_dic, bool (default: False)
        verbose: say something during the process! bool (default:True)
    Outputs:
        drift_dic: dictionary for ref_hyb_name -> 3d drift array
        fail_count: number of suspicious failures in drift correction, int
    """
    ## check inputs
    # check folders
    if not isinstance(folders, list):
        raise ValueError("Wrong input type of folders, should be a list")
    if len(folders) == 0:  # check folders length
        raise ValueError("Kwd folders should have at least one element")
    if not isinstance(fovs, list):
        raise ValueError("Wrong input type of fovs, should be a list")
    # check if all images exists
    _fov_name = fovs[fov_id]
    for _fd in folders:
        _filename = os.path.join(_fd, _fov_name)
        if not os.path.isfile(_filename):
            raise IOError(
                f"one of input file:{_filename} doesn't exist, exit!")
    # check ref-id
    if ref_id >= len(folders) or ref_id <= -len(folders):
        raise ValueError(
            f"Ref_id should be valid index of folders, however {ref_id} is given")
    # check save-folder
    if save_folder is None:
        save_folder = os.path.join(
            os.path.dirname(folders[0]), 'Analysis', 'drift')
        print(
            f"No save_folder specified, use default save-folder:{save_folder}")
    elif not os.path.exists(save_folder):
        if verbose:
            print(f"Create drift_folder:{save_folder}")
        os.makedirs(save_folder)
    # check save_name
    if sequential_mode:  # if doing drift-correction in a sequential mode:
        save_postfix = '_sequential'+save_postfix

    # check coord_sel
    if coord_sel is None:
        coord_sel = np.array(
            [int(single_im_size[-2]/2), int(single_im_size[-1]/2)], dtype=np.int)
    else:
        coord_sel = np.array(coord_sel, dtype=np.int)
    # collect crop coordinates (slices)
    crop0 = np.array([[0, single_im_size[0]],
                      [max(coord_sel[-2]-drift_size, 0), coord_sel[-2]],
                      [max(coord_sel[-1]-drift_size, 0), coord_sel[-1]]], dtype=np.int)
    crop1 = np.array([[0, single_im_size[0]],
                      [coord_sel[-2], min(coord_sel[-2] +
                                          drift_size, single_im_size[-2])],
                      [coord_sel[-1], min(coord_sel[-1]+drift_size, single_im_size[-1])]], dtype=np.int)
    crop2 = np.array([[0, single_im_size[0]],
                      [coord_sel[-2], min(coord_sel[-2] +
                                          drift_size, single_im_size[-2])],
                      [max(coord_sel[-1]-drift_size, 0), coord_sel[-1]]], dtype=np.int)
    # merge into one array which is easier to feed into function
    selected_crops = np.stack([crop0, crop1, crop2])
    
    ## start loading existing profile
    _save_name = _fov_name.replace('.dax', save_postfix)
    _save_filename = os.path.join(save_folder, _save_name)
    # try to load existing profiles
    if not overwrite and os.path.isfile(_save_filename):
        if verbose:
            print(f"- loading existing drift info from file:{_save_filename}")
        old_drift_dic = pickle.load(open(_save_filename, 'rb'))
        _exist_keys = [os.path.join(os.path.basename(
            _fd), _fov_name) in old_drift_dic for _fd in folders]
        if sum(_exist_keys) == len(folders):
            if verbose:
                print("-- all frames exists in original drift file, exit.")
            return old_drift_dic, 0

    # no existing profile or force to do de novo correction:
    else:
        if verbose:
            print(
                f"- starting a new drift correction for field-of-view:{_fov_name}")
        old_drift_dic = {}
    ## initialize drift correction
    _start_time = time.time()  # record start time
    # whether drift for reference frame changes
    if len(old_drift_dic) > 0:
        old_ref_frames = [_hyb_name for _hyb_name,
                          _dft in old_drift_dic.items() if (np.array(_dft) == 0).all()]
        if len(old_ref_frames) > 1 or len(old_ref_frames) == 0:
            print("-- ref frame not unique, start over!")
            old_drift_dic = {}
        else:
            old_ref_frame = old_ref_frames[0]
            # if ref-frame not overlapping, remove the old one for now
            if old_ref_frame != os.path.join(os.path.basename(folders[ref_id]), _fov_name):
                if verbose:
                    print(
                        f"-- old-ref:{old_ref_frame}, delete old refs because ref doesn't match")
                del old_drift_dic[old_ref_frame]
    else:
        old_ref_frame = None
    if not sequential_mode:
        if verbose:
            print(
                f"- Start drift-correction with {num_threads} threads, image mapped to image:{ref_id}")
        ## get all reference information
        # ref filename
        _ref_filename = os.path.join(
            folders[ref_id], _fov_name)  # get ref-frame filename
        _ref_keyname = os.path.join(
            os.path.basename(folders[ref_id]), _fov_name)
        _ref_ims = []
        _ref_centers = []
        if verbose:
            print("--- loading reference images and centers")
        for _crop in selected_crops:
            _ref_im = correct_single_image(_ref_filename, bead_channel, crop_limits=_crop,
                                        single_im_size=single_im_size,
                                        all_channels=all_channels, num_buffer_frames=num_buffer_frames,
                                        illumination_corr=illumination_corr, verbose=verbose)
            _ref_center = visual_tools.get_STD_centers(_ref_im, dynamic=True, th_seed_percentile=ref_seed_per,
                                                       sort_by_h=True, verbose=verbose)
            # limit ref points
            if max_ref_points > 0:
                _ref_center = np.array(_ref_center)[:max(
                    max_ref_points, len(_ref_center)), :]
            # append
            _ref_ims.append(_ref_im)
            _ref_centers.append(_ref_center)
        ## retrieve files requiring drift correction
        args = []
        new_keynames = []
        for _i, _fd in enumerate(folders):
            # extract filename
            _filename = os.path.join(_fd, _fov_name)
            _keyname = os.path.join(os.path.basename(_fd), _fov_name)
            # append all new frames except ref-frame. ref-frame will be assigned to 0
            if _keyname not in old_drift_dic and _keyname != _ref_keyname:
                new_keynames.append(_keyname)
                args.append((_filename, selected_crops, None, _ref_ims, _ref_centers,
                             bead_channel, all_channels, single_im_size,
                             num_buffer_frames, ref_seed_per * 0.99**_i, illumination_corr,
                             correction_folder, match_distance, match_unique, 
                             rough_drift_gb, drift_cutoff, verbose))
    
    ## sequential mode
    else:
        # retrieve files requiring drift correction
        args = []
        new_keynames = []
        for _ref_fd, _fd in zip(folders[:-1], folders[1:]):
            # extract filename
            _filename = os.path.join(_fd, _fov_name)  # full filename for image
            _keyname = os.path.join(os.path.basename(
                _fd), _fov_name)  # key name used in dic
            # full filename for ref image
            _ref_filename = os.path.join(_ref_fd, _fov_name)
            if _keyname not in old_drift_dic:
                # append
                new_keynames.append(_keyname)
                args.append((_filename, selected_crops, _ref_filename, None, None,
                             bead_channel, all_channels, single_im_size,
                             num_buffer_frames, ref_seed_per, illumination_corr,
                             correction_folder, match_distance, match_unique,
                             rough_drift_gb, drift_cutoff, verbose))

    ## multiprocessing
    if verbose:
        print(
            f"-- Start multi-processing drift correction with {num_threads} threads")
    with mp.Pool(num_threads) as drift_pool:
        align_results = drift_pool.starmap(alignment_tools.align_single_image, args)
        drift_pool.close()
        drift_pool.join()
        drift_pool.terminate()
    # clear
    del(args)
    classes.killchild()
    # convert to dict
    if not sequential_mode:
        new_drift_dic = {_ref_name: _ar[0] for _ref_name, _ar in zip(
            new_keynames, align_results)}
    else:
        _total_dfts = [_ar[0][0] for _ar in zip(align_results)]
        _total_dfts = [sum(_total_dfts[:i+1])
                       for i, _d in enumerate(_total_dfts)]
        new_drift_dic = {_ref_name: _dft for _ref_name,
                         _dft in zip(new_keynames, _total_dfts)}
    # calculate failed count
    fail_count = sum([_ar[1] for _ar in align_results])
    # append ref frame drift
    _ref_keyname = os.path.join(os.path.basename(folders[ref_id]), _fov_name)
    new_drift_dic[_ref_keyname] = np.zeros(3)  # for ref frame, assign to zero
    if old_ref_frame is not None and old_ref_frame in new_drift_dic:
        ref_drift = new_drift_dic[old_ref_frame]
        if verbose:
            print(f"-- drift of reference: {ref_drift}")
        # update drift in old ones
        for _old_name, _old_dft in old_drift_dic.items():
            if _old_name not in new_drift_dic:
                new_drift_dic[_old_name] = _old_dft + ref_drift
    ## save
    if save:
        if verbose:
            print(f"- Saving drift to file:{_save_filename}")
        if not os.path.exists(os.path.dirname(_save_filename)):
            if verbose:
                print(
                    f"-- creating save-folder: {os.path.dirname(_save_filename)}")
            os.makedirs(os.path.dirname(_save_filename))
        pickle.dump(new_drift_dic, open(_save_filename, 'wb'))
    if verbose:
        print(
            f"-- Total time cost in drift correction: {time.time()-_start_time}")
        if fail_count > 0:
            print(f"-- number of failed drifts: {fail_count}")

    return new_drift_dic, fail_count


# illumination correction for one image

def Illumination_correction(im, correction_channel, crop_limits=None, all_channels=_allowed_colors,
                            single_im_size=_image_size, correction_folder=_correction_folder,
                            profile_dtype=np.float, image_dtype=np.uint16,
                            ic_profile_name='illumination_correction', correction_power=1, verbose=True):
    """Function to do fast illumination correction in a RAM-efficient manner
    Inputs:
        im: 3d image, np.ndarray or np.memmap
        channel: the color channel for given image, int or string (should be in single_im_size list)
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        all_channels: allowed channels to be corrected, list 
        single_im_size: full image size before any slicing, list of 3 (default:[30,2048,2048])
        correction_folder: correction folder to find correction profile, string of path (default: Z://Corrections/)
        correction_power: power for correction factor, float (default: 1)
        verbose: say something!, bool (default: True)
    Outputs:
        corr_im: corrected image
    """
    ## check inputs
    # im
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise ValueError(
            f"Wrong input type for im: {type(im)}, np.ndarray or np.memmap expected")
    if verbose:
        print(f"-- correcting illumination for image size:{im.shape} for channel:{correction_channel}")
    # channel
    channel = str(correction_channel)
    if channel not in all_channels:
        raise ValueError(
            f"Input channel:{channel} is not in allowed channels:{allowed_channels}")
    # check correction profile exists:
    ic_filename = os.path.join(correction_folder,
                               ic_profile_name+'_'+str(channel)+'_'
                               + str(single_im_size[-2])+'x'+str(single_im_size[-1])+'.npy')
    if not os.path.isfile(ic_filename):
        raise IOError(
            f"Illumination correction profile file:{ic_filename} doesn't exist, exit!")
    # image shape and ref-shape
    im_shape = np.array(im.shape, dtype=np.int)
    single_im_size = np.array(single_im_size, dtype=np.int)
    # check crop_limits
    if crop_limits is None:
        if (im_shape[-2:]-single_im_size[-2:]).any():
            raise ValueError(
                f"Test image is not of full image size:{single_im_size}, while crop_limits are not given, exit!")
        else:
            # change crop_limits into full image size
            crop_limits = np.stack([np.zeros(3), im_shape]).T
    elif len(crop_limits) <= 1 or len(crop_limits) > 3:
        raise ValueError("crop_limits should have 2 or 3 elements")
    elif len(crop_limits) == 2:
        crop_limits = np.stack(
            [np.array([0, im_shape[0]])] + list(crop_limits)).astype(np.int)
    else:
        crop_limits = np.array(crop_limits, dtype=np.int)
    # convert potential negative values to positive for further calculation
    for _s, _lims in zip(im_shape, crop_limits):
        if _lims[1] < 0:
            _lims[1] += _s
    crop_shape = crop_limits[:, 1] - crop_limits[:, 0]
    # crop image if necessary
    if not (im_shape[-2:]-single_im_size[-2:]).any():
        cim = im[crop_limits[0, 0]:crop_limits[0, 1],
                 crop_limits[1, 0]:crop_limits[1, 1],
                 crop_limits[2, 0]:crop_limits[2, 1], ]
    elif not (im_shape-crop_shape).any():
        cim = im
    else:
        raise IndexError(
            f"Wrong input size for im:{im_shape} compared to crop_limits:{crop_limits}")

    ## do correction
    # get cropped correction profile
    cropped_profile = visual_tools.slice_2d_image(
        ic_filename, single_im_size[-2:], crop_limits[1], crop_limits[2], image_dtype=profile_dtype)
    # correction
    corr_im = (cim / cropped_profile**correction_power).astype(image_dtype)

    return corr_im


def Chromatic_abbrevation_correction(im, correction_channel, target_channel='647', crop_limits=None, 
                                     all_channels=_allowed_colors, single_im_size=_image_size,
                                     correction_folder=_correction_folder, 
                                     profile_dtype=np.float, image_dtype=np.uint16,
                                     cc_profile_name='chromatic_correction', verbose=True):
    """Chromatic abbrevation correction for given image and crop
        im: 3d image, np.ndarray or np.memmap
        correction_channel: the color channel for given image, int or string (should be in single_im_size list)
        target_channel: the target channel that image should be correct to, int or string (default: '647')
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        all_channels: allowed channels to be corrected, list 
        single_im_size: full image size before any slicing, list of 3 (default:[30,2048,2048])
        correction_folder: correction folder to find correction profile, string of path (default: Z://Corrections/)
        profile_dtype: data type for correction profile, numpy datatype (default: np.float)
        image_dtype: image data type, numpy datatype (default: np.uint16)
        cc_profile_name: chromatic correction file basename, str (default: 'chromatic_correction')
        verbose: say something!, bool (default: True)"""
    
    ## check inputs
    # im
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise ValueError(
            f"Wrong input type for im: {type(im)}, np.ndarray or np.memmap expected")
    if verbose:
        print(f"-- correcting chromatic abbrevation for iamge size:{im.shape}, channel:{correction_channel}")
    # correction channel
    correction_channel = str(correction_channel)
    if correction_channel not in all_channels:
        raise ValueError(f"Input channel:{correction_channel} is not in allowed channels:{all_channels}")
    # target channel
    target_channel = str(target_channel)
    if target_channel not in all_channels:
        raise ValueError(f"Input channel:{target_channel} is not in allowed channels:{all_channels}")
    # if no correction required, directly return
    if correction_channel == target_channel:
        if verbose:
            print(f"--- no chromatic abbrevation required for channel:{correction_channel}")
        return im
    
    # check correction profile exists:
    cc_filename = os.path.join(correction_folder,
                               cc_profile_name+'_'+str(correction_channel)+'_'+str(target_channel)+'_'
                               + str(single_im_size[-2])+'x'+str(single_im_size[-1])+'.npy')
    if not os.path.isfile(cc_filename):
        raise IOError(
            f"Chromatic correction profile file:{cc_filename} doesn't exist, exit!")
    
    # image shape and ref-shape
    im_shape = np.array(im.shape, dtype=np.int)
    single_im_size = np.array(single_im_size, dtype=np.int)
    
    # check crop_limits
    if crop_limits is None:
        if (im_shape[-2:]-single_im_size[-2:]).any():
            raise ValueError(
                f"Test image is not of full image size:{single_im_size}, while crop_limits are not given, exit!")
        else:
            # change crop_limits into full image size
            crop_limits = np.stack([np.zeros(3), im_shape]).T
    elif len(crop_limits) <= 1 or len(crop_limits) > 3:
        raise ValueError("crop_limits should have 2 or 3 elements")
    elif len(crop_limits) == 2:
        crop_limits = np.stack(
            [np.array([0, im_shape[0]])] + list(crop_limits)).astype(np.int)
    else:
        crop_limits = np.array(crop_limits, dtype=np.int)
    
    # convert potential negative values to positive for further calculation
    for _s, _lims in zip(im_shape, crop_limits):
        if _lims[1] < 0:
            _lims[1] += _s
    crop_shape = crop_limits[:, 1] - crop_limits[:, 0]
    
    # crop image if necessary
    if not (im_shape[-2:]-single_im_size[-2:]).any():
        cim = im[crop_limits[0, 0]:crop_limits[0, 1],
                 crop_limits[1, 0]:crop_limits[1, 1],
                 crop_limits[2, 0]:crop_limits[2, 1] ]
    elif not (im_shape-crop_shape).any():
        cim = im
    else:
        raise IndexError(f"Wrong input size for im:{im_shape} compared to crop_limits:{crop_limits}")
    
    ## do correction
    # 1. get coordiates to be mapped
    _coords = np.meshgrid( np.arange(crop_limits[0][1]-crop_limits[0][0]), 
                           np.arange(crop_limits[1][1]-crop_limits[1][0]), 
                           np.arange(crop_limits[2][1]-crop_limits[2][0]))
    _coords = np.stack(_coords).transpose((0, 2, 1, 3)) # transpose is necessary
    # 2. load chromatic profile
    _cropped_cc_profile = visual_tools.slice_image(cc_filename, [3, single_im_size[1], single_im_size[2]],
                                                   [0,3], [crop_limits[1][0],crop_limits[1][1]],
                                                   [crop_limits[2][0],crop_limits[2][1]], image_dtype=profile_dtype)
    # 3. calculate corrected coordinates as a reference
    _corr_coords = _coords + _cropped_cc_profile[:,np.newaxis]
    # 4. map coordinates
    _corr_im = map_coordinates(cim, _corr_coords.reshape(_corr_coords.shape[0], -1), mode='nearest')
    _corr_im = _corr_im.reshape(np.shape(cim))

    return _corr_im

# correct for illumination _shifts across z layers
def Z_Shift_Correction(im, dtype=np.uint16, normalization=False, verbose=False):
    '''Function to correct for each layer in z, to make sure they match in term of intensity'''
    if verbose:
        print("-- correcting Z axis illumination shifts.")
    if not normalization:
        _nim = im / np.median(im, axis=(1, 2))[:,np.newaxis,np.newaxis] * np.median(im)
    else:
        _nim = im / np.median(im, axis=(1, 2))[:,np.newaxis,np.newaxis] * np.median(im)
    return _nim.astype(dtype)

# remove hot pixels
def Remove_Hot_Pixels(im, dtype=np.uint16, hot_pix_th=0.50, hot_th=4, 
                      interpolation_style='nearest', verbose=False):
    '''Function to remove hot pixels by interpolation in each single layer'''
    if verbose:
        print("-- removing hot pixels")
    # create convolution matrix, ignore boundaries for now
    _conv = (np.roll(im,1,1)+np.roll(im,-1,1)+np.roll(im,1,2)+np.roll(im,1,2))/4
    # hot pixels must be have signals higher than average of neighboring pixels by hot_th in more than hot_pix_th*total z-stacks
    _hotmat = im > hot_th * _conv
    _hotmat2D = np.sum(_hotmat,0)
    _hotpix_cand = np.where(_hotmat2D > hot_pix_th*np.shape(im)[0])
    # if no hot pixel detected, directly exit
    if len(_hotpix_cand[0]) == 0:
        return im
    # create new image to interpolate the hot pixels with average of neighboring pixels
    _nim = im.copy()
    if interpolation_style == 'nearest':
        for _x, _y in zip(_hotpix_cand[0],_hotpix_cand[1]):
            if _x > 0 and  _y > 0 and _x < im.shape[1]-1 and  _y < im.shape[2]-1:
                _nim[:,_x,_y] = (_nim[:,_x+1,_y]+_nim[:,_x-1,_y]+_nim[:,_x,_y+1]+_nim[:,_x,_y-1])/4
    return _nim.astype(dtype)


# fast function to generate illumination profiles
def generate_illumination_correction(color, data_folder, correction_folder, image_type='H', 
                                     all_colors=_allowed_colors,  num_of_images=50,
                                    folder_id=0, buffer_frame=10, frame_per_color=30, target_color_ind=-1,
                                    gaussian_sigma=40, seeding_th_per=99.5, seeding_th_base=300, seeding_crop_size=9,
                                    remove_cap=False, cap_th_per=99.5,
                                    force=False, save=True, save_name='illumination_correction_', make_plot=False, verbose=True):
    """Function to generate illumination correction profile from hybridization type of image or bead type of image
    Inputs:
        color: 
        data_folder: master folder for images, string of path
        correction_folder: folder to save correction_folder, string of path
        image_type: type of images to generate this profile, int (default: 50)
        gaussian_sigma: sigma of gaussian filter used to smooth image, int (default: 40)
    Outputs:
        ic_profile: 2d illumination correction profile 
        """
    from scipy.stats import scoreatpercentile
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve
    ## check inputs
    # color
    _color = str(color)
    _allowed_colors = all_colors
    if _color not in _allowed_colors:
        raise ValueError(f"Wrong color input, {color} is given, color among {_allowed_colors} is expected")
    # extract index of this color
    if target_color_ind == -1:
        target_color_ind = _allowed_colors.index(_color)
    # image type
    image_type = image_type[0].upper()
    if image_type not in ['H', 'B']:
        raise ValueError(f"Wrong input image_type, {image_type} is given, 'H' or 'B' expected!")
    ## get images
    _folders, _fovs = get_img_info.get_folders(data_folder, feature=image_type,verbose=verbose)
    if len(_folders)==0 or len(_fovs)==0:
        raise IOError(f"No folders or fovs detected with given data_folder:{data_folder} and image_type:{image_type}!")
    ## load image info
    _im_filename = os.path.join(_folders[folder_id], _fovs[0])
    _info_filename = _im_filename.replace('.dax', '.inf')
    with open(_info_filename, 'r') as _info_hd:
        _infos = _info_hd.readlines()
    # get frame number and color information
    _num_frame, _num_color = 0, 0
    _dx, _dy = 0, 0
    for _line in _infos:
        _line = _line.rstrip()
        if "number of frames" in _line:
            _num_frame = int(_line.split('=')[1])
            _num_color = (_num_frame - 2*buffer_frame) / frame_per_color
            if _num_color != int(_num_color):
                raise ValueError("Wrong num_color, should be integer!")
            _num_color = int(_num_color)
        if "frame dimensions" in _line:
            _dx = int(_line.split('=')[1].split('x')[0])
            _dy = int(_line.split('=')[1].split('x')[1])
    if not _num_frame or not _num_color:
        raise ValueError(f"No frame number info exists in {_info_filename}")
    if not _dx or not _dy:
        raise ValueError(f"No frame dimension info exists in {_info_filename}")    
    # save filename    
    save_filename = os.path.join(correction_folder, save_name+str(_color)+'_'+str(_dx)+'x'+str(_dy))
    if os.path.isfile(save_filename+'.npy') and not force:
        if verbose:
            print(f"- directly loading illumination correction profile from file:{save_filename}.npy")
        _mean_profile = np.load(save_filename+'.npy')
    else:
        # initialize 
        _picked_profiles = []
        _fd = _folders[folder_id]
        for _fov in _fovs:
            # exit if enough images loaded
            if len(_picked_profiles) >= num_of_images:
                break
            # start slicing image
            if os.path.isfile(os.path.join(_fd, _fov)):
                ## get info
                _im_filename = os.path.join(_fd, _fov)
                if verbose:
                    print(f"-- loading {_color} from image file {_im_filename}, color_ind:{target_color_ind}")
                _im = visual_tools.slice_image(_im_filename, [_num_frame, _dx, _dy], [buffer_frame, _num_frame-buffer_frame], [0, _dx],
                                                [0, _dy], _num_color, target_color_ind)
                # do corrections
                _im = Remove_Hot_Pixels(_im, verbose=False)
                _im = Z_Shift_Correction(_im, verbose=False)
                _im = np.array(_im, dtype=np.float)
                # remove top values if necessary
                if remove_cap:
                    _cap_thres = scoreatpercentile(_im, cap_th_per)
                    _im[_im > _cap_thres] = np.nan
                # seed and exclude these blocks
                _seed_thres = scoreatpercentile(_im, seeding_th_per) - np.median(_im)
                _seeds = visual_tools.get_seed_points_base(_im, th_seed=_seed_thres)
                for _sd in np.transpose(_seeds):
                    _l_lims = np.array(_sd - int(seeding_crop_size/2), dtype=np.int)
                    _l_lims[_l_lims<0] = 0
                    _r_lims = [_sd+int(seeding_crop_size/2), np.array(_im.shape)]
                    _r_lims = np.min(np.array(_r_lims, dtype=np.int), axis=0)
                    _im[_l_lims[0]:_r_lims[0], _l_lims[1]:_r_lims[1], _l_lims[2]:_r_lims[2]] = np.nan
                # append
                _picked_profiles.append(_im)
        # generate averaged profile
        if verbose:
            print("- generating averaged profile")
        _mean_profile = np.nanmean(np.concatenate(_picked_profiles), axis=0)
        _mean_profile[_mean_profile == 0] = np.nan
        ## gaussian filter
        if verbose:
            print("- applying gaussian filter to averaged profile")
        # set gaussian kernel
        _kernel = Gaussian2DKernel(x_stddev=gaussian_sigma)
        # convolution, which will interpolate any NaN numbers
        _mean_profile = convolve(_mean_profile, _kernel, boundary='extend')
        _mean_profile = _mean_profile / np.max(_mean_profile)
        if save:
            if verbose:
                print(f"-- saving correction profile to file:{save_filename}.npy")
            if not os.path.exists(os.path.dirname(save_filename)):
                os.makedirs(os.path.dirname(save_filename))
            np.save(save_filename, _mean_profile)
    if make_plot:
        plt.figure()
        plt.imshow(_mean_profile)
        plt.colorbar()
        plt.show()

    return _mean_profile


def generate_chromatic_abbrevation_from_spots(corr_spots, ref_spots, corr_channel, ref_channel, 
                                            image_size=_image_size, fitting_order=2,
                                            correction_folder=_correction_folder, make_plot=False, 
                                            save=True, save_name='chromatic_correction_',force=False, verbose=True):
    """Code to generate chromatic abbrevation from fitted and matched spots"""
    ## check inputs
    if len(corr_spots) != len(ref_spots):
        raise ValueError("corr_spots and ref_spots are of different length, so not matched")
    # color 
    _allowed_colors = ['750', '647', '561', '488', '405']
    if str(corr_channel) not in _allowed_colors:
        raise ValueError(f"corr_channel given:{corr_channel} is not valid, {_allowed_colors} are expected")
    if str(ref_channel) not in _allowed_colors:
        raise ValueError(f"corr_channel given:{ref_channel} is not valid, {_allowed_colors} are expected")
        
    ## savefile
    filename_base = save_name + str(corr_channel)+'_'+str(ref_channel)+'_'+str(image_size[1])+'x'+str(image_size[2])
    saved_profile_filename = os.path.join(correction_folder, filename_base+'.npy')
    saved_const_filename = os.path.join(correction_folder, filename_base+'_const.npy')
    # whether have existing profile
    if os.path.isfile(saved_profile_filename) and os.path.isfile(saved_const_filename) and not force:
        _cac_profiles = np.load(saved_profile_filename)
        _cac_consts = np.load(saved_const_filename)
        if make_plot:
            for _i,_grid_shifts in enumerate(_cac_profiles):
                plt.figure()
                plt.imshow(_grid_shifts)
                plt.colorbar()
                plt.title(f"chromatic-abbrevation {corr_channel} to {ref_channel}, axis-{_i}")
                plt.show()
    else:
        ## start correction
        _cac_profiles = []
        _cac_consts = []
        # variables used in polyfit
        ref_spots, corr_spots = np.array(ref_spots), np.array(corr_spots) # convert to array
        _x = ref_spots[:,1]
        _y = ref_spots[:,2]
        _data = [] # variables in polyfit
        for _order in range(fitting_order+1): # loop through different orders
            for _p in range(_order+1):
                _data.append(_x**_p * _y**(_order-_p))
        _data = np.array(_data).transpose()

        for _i in range(3): # 3D
            if verbose:
                print(f"-- fitting chromatic-abbrevation in axis {_i} with order:{fitting_order}")
            _value =  corr_spots[:,_i] - ref_spots[:,_i] # target-value for polyfit
            _C,_r,_r2,_r3 = scipy.linalg.lstsq(_data, _value)    # coefficients and residues
            _cac_consts.append(_C) # store correction constants
            _rsquare =  1 - np.var(_data.dot(_C) - _value)/np.var(_value)
            if verbose:
                print(f"--- fitted rsquare:{_rsquare}")

            ## generate correction function
            def _get_shift(coords):
                # traslate into 2d
                if len(coords.shape) == 1:
                    coords = coords[np.newaxis,:]
                _cx = coords[:,1]
                _cy = coords[:,2]
                _corr_data = []
                for _order in range(fitting_order+1):
                    for _p in range(_order+1):
                        _corr_data.append(_cx**_p * _cy**(_order-_p))
                _shift = np.dot(np.array(_corr_data).transpose(), _C)
                return _shift

            ## generate correction_profile
            _xc_t, _yc_t = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[2])) # initialize meshgrid
            _xc, _yc = _xc_t.transpose(), _yc_t.transpose()
            # transpose to the shape compatible with function
            _grid_coords = np.array([np.zeros(np.size(_xc)), _xc.reshape(-1), _yc.reshape(-1)]).transpose() 
            # calculate shift and trasform back
            _grid_shifts = _get_shift(_grid_coords)
            _grid_shifts = _grid_shifts.reshape(np.shape(_xc))
            _cac_profiles.append(_grid_shifts) # store correction profile across 2d

            ## make plot
            if make_plot:
                plt.figure()
                plt.imshow(_grid_shifts)
                plt.colorbar()
                plt.title(f"chromatic-abbrevation {corr_channel} to {ref_channel}, axis-{_i}")
                plt.show()
        _cac_profiles = np.array(_cac_profiles)
        _cac_consts = np.array(_cac_consts)
        # save 
        if save:
            if verbose:
                print("-- save profiles to file:{saved_profile_filename}")
            np.save(saved_profile_filename.split('.npy')[0], _cac_profiles)
            if verbose:
                print("-- save shift functions to file:{saved_const_filename}")
            np.save(saved_const_filename.split('.npy')[0], _cac_consts)
    
    def _cac_func(coords, consts=_cac_consts, max_order=fitting_order):
        # traslate into 2d
        if len(coords.shape) == 1:
            coords = coords[np.newaxis,:]
        _cx = coords[:,1]
        _cy = coords[:,2]
        _corr_data = []
        for _order in range(max_order+1):
            for _p in range(_order+1):
                _corr_data.append(_cx**_p * _cy**(_order-_p))
        _shifts = []
        for _c in consts:
            _shifts.append(np.dot(np.array(_corr_data).transpose(), _c))
        _shifts = np.array(_shifts).transpose()
        _corr_coords = coords - _shifts
        return _corr_coords    
    
    return _cac_profiles, _cac_func

def fast_chromatic_abbrevation_correction(im, correction_channel, target_channel='647', single_im_size=_image_size,
                                          crop_limits=None, all_channels=_allowed_colors,
                                          buffer_frame=10, frame_per_color=30, target_color_ind=-1,
                                          correction_folder=_correction_folder, cac_profile=None,
                                          verbose=True):
    """Chromatic abbrevation correction"""
    from scipy.ndimage.interpolation import map_coordinates
    ## check inputs
    # color
    _color = str(correction_channel)
    all_channels = ['750', '647', '561', '488', '405']
    if _color not in all_channels:
        raise ValueError(
            f"Wrong color input, {_color} is given, color among {all_channels} is expected")
    if target_color_ind == -1:
        target_color_ind = all_channels.index(_color)
    # crop_limits
    if crop_limits is not None:
        if len(crop_limits) <= 1 or len(crop_limits) > 3:
            raise ValueError("crop_limits should have 2 elements")
        else:
            for _lims in crop_limits:
                if len(_lims) < 2:
                    raise ValueError(
                        f"given limit {_lims} has less than 2 elements, exit")
    # if no correction required, directly return
    if _color == target_channel:
        if verbose:
            print(f"-- no chromatic abbrevation required for channel:{_color}")
        return im
    else:
        if verbose:
            print(f"-- chromatic abbrevation for image with size:{im.shape}")
    if cac_profile is not None and (isinstance(cac_profile, list) or isinstance(cac_profile, np.ndarray)):
        # directly adopt from input profile
        _cac_profile = cac_profile
        print("direct chromatic")
    elif cac_profile is not None and "multiprocessing" in str(type(cac_profile)):
        _cac_shape = np.array([3, single_im_size[1], single_im_size[2]], dtype=int)
        _cac_profile = np.frombuffer(cac_profile).reshape(_cac_shape)
    else:
        # load profile from correction file
        _cac_profile_file = os.path.join(correction_folder,
                                        'chromatic_correction_'+str(_color)+'_'+str(target_channel)+'_'+str(single_im_size[1])+'x'+str(single_im_size[2])+'.npy')
        if not os.path.isfile(_cac_profile_file):
            raise IOError(
                f"chromatic abbreviation correction file:{_cac_profile_file} doesn't exist, exit! ")
        else:
            _cac_profile = np.load(_cac_profile_file)
    # check image
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise ValueError(
            "Wrong input type for im, should be np.ndarray or memmap")

    ## process the whole image
    if crop_limits is None and (np.array(np.shape(im))[:3] == np.array(single_im_size)).all():
        if verbose:
            print(f"-- chromatic abbrevation for the whole image")
        # initialize grid-points
        _coords = np.meshgrid(np.arange(im.shape[0]), np.arange(
            single_im_size[1]), np.arange(single_im_size[2]))
        _coords = np.stack(_coords).transpose((0, 2, 1, 3))
        _corr_coords = _coords + _cac_profile[:, np.newaxis, :, :]
        # map coordinates
        _corr_im = map_coordinates(im, _corr_coords.reshape(
            _corr_coords.shape[0], -1), mode='nearest')
        _corr_im = _corr_im.reshape(single_im_size)
    ## process cropped image
    elif crop_limits is not None:
        if len(crop_limits) == 2:
            if (np.array(np.shape(im))[:3] == np.array(single_im_size)).all():
                _cropped_im = im[:, crop_limits[0][0]:crop_limits[0]
                                [1], crop_limits[1][0]:crop_limits[1][1]]
            elif (np.array(np.shape(im))[:3] == np.array([single_im_size[0], crop_limits[0][1]-crop_limits[0][0], crop_limits[1][1]-crop_limits[1][0]])).all():
                _cropped_im = im.copy()
            else:
                raise ValueError(
                    "Input image should be of original-image-size or cropped image size given by crop_limits")
            if verbose:
                print(f"-- chromatic abbrevation for cropped image")
            # initialize grid-points for cropped region
            _coords = np.meshgrid(np.arange(_cropped_im.shape[0]), np.arange(
                crop_limits[0][1]-crop_limits[0][0]), np.arange(crop_limits[1][1]-crop_limits[1][0]))
            _coords = np.stack(_coords).transpose((0, 2, 1, 3))
            _corr_coords= _coords + _cac_profile[:, np.newaxis,
                                                 crop_limits[0][0]:crop_limits[0][1],
                                                 crop_limits[1][0]:crop_limits[1][1]]
        else: # 3 limits are given:
            if (np.array(np.shape(im))[:3] == np.array(single_im_size)).all():
                _cropped_im = im[crop_limits[0][0]:crop_limits[0][1], 
                                 crop_limits[1][0]:crop_limits[1][1],
                                 crop_limits[2][0]:crop_limits[2][1]
                                 ]
            elif (np.array(np.shape(im))[:3] == np.array([crop_limits[0][1]-crop_limits[0][0], crop_limits[1][1]-crop_limits[1][0], crop_limits[2][1]-crop_limits[2][0]])).all():
                _cropped_im = im.copy()
            else:
                raise ValueError(
                    "Input image should be of original-image-size or cropped image size given by crop_limits")
            if verbose:
                print(f"-- chromatic abbrevation for cropped image")
            # initialize grid-points for cropped region
            _coords = np.meshgrid( np.arange(crop_limits[0][1]-crop_limits[0][0]), 
                                   np.arange(crop_limits[1][1]-crop_limits[1][0]), 
                                   np.arange(crop_limits[2][1]-crop_limits[2][0]))

            _coords = np.stack(_coords).transpose((0, 2, 1, 3))
            _corr_coords = _coords + _cac_profile[:, np.newaxis, 
                                                  crop_limits[1][0]:crop_limits[1][1], 
                                                  crop_limits[2][0]:crop_limits[2][1]]
        # map coordinates
        _corr_im = map_coordinates(_cropped_im, _corr_coords.reshape(
            _corr_coords.shape[0], -1), mode='nearest')
        _corr_im = _corr_im.reshape(np.shape(_cropped_im))
 
    return _corr_im

def load_correction_profile(channel, corr_type, correction_folder=_correction_folder, ref_channel='647', 
                            im_size=_image_size, verbose=False):
    """Function to load chromatic/illumination correction profile"""
    ## check inputs
    # type
    _allowed_types = ['chromatic', 'illumination']
    _type = str(corr_type).lower()
    if _type not in _allowed_types:
        raise ValueError(f"Wrong input corr_type, should be one of {_allowed_types}")
    # channel
    _allowed_channels = ['750', '647', '561', '488', '405']
    _channel = str(channel).lower()
    _ref_channel = str(ref_channel).lower()
    if _channel not in _allowed_channels:
        raise ValueError(f"Wrong input channel, should be one of {_allowed_channels}")
    if _ref_channel not in _allowed_channels:
        raise ValueError(f"Wrong input ref_channel, should be one of {_allowed_channels}")
    ## start loading file
    _basename = _type+'_correction_'+_channel+'_'
    if _type == 'chromatic':
        _basename += _ref_channel+'_'
    _basename += str(im_size[1])+'x'+str(im_size[2])+'.npy'
    # filename 
    _corr_filename = os.path.join(correction_folder, _basename)
    if os.path.isfile(_corr_filename):
        _corr_profile = np.load(_corr_filename)
    elif _type == 'chromatic' and _channel == _ref_channel:
        return None
    else:
        raise IOError(f"File {_corr_filename} doesn't exist, exit!")

    return np.array(_corr_profile)


# merged function to crop and correct single image
def correct_single_image(filename, channel, crop_limits=None, seg_label=None, extend_dim=20,
                         single_im_size=_image_size, all_channels=_allowed_colors, num_buffer_frames=10,
                         drift=np.array([0, 0, 0]), correction_folder=_correction_folder, normalization=False,
                         z_shift_corr=True, hot_pixel_remove=True, illumination_corr=True, chromatic_corr=True,
                         return_limits=False, verbose=False):
    """wrapper for all correction steps to one image, used for multi-processing
    Inputs:
        filename: full filename of a dax_file or npy_file for one image, string
        channel: channel to be extracted, str (for example. '647')
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        seg_label: segmentation label, 2D array
        extend_dim: extension pixel number if doing cropping, int (default: 20)
        single_im_size: z-x-y size of the image, list of 3
        all_channels: all_channels used in this image, list of str(for example, ['750','647','561'])
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        drift: 3d drift vector for this image, 1d-array
        correction_folder: path to find correction files
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        return_limits: whether return cropping limits
        verbose: whether say something!, bool (default:True)
        """
    ## check inputs
    # color channel
    channel = str(channel)
    all_channels = [str(ch) for ch in all_channels]
    if channel not in all_channels:
        raise ValueError(
            f"Target channel {channel} doesn't exist in all_channels:{all_channels}")
    # only 3 all_channels requires chromatic correction
    if channel not in ['750', '647', '561']:
        chromatic_corr = False
    # check filename
    if not isinstance(filename, str):
        raise ValueError(
            f"Wrong input of filename {filename}, should be a string!")
    if not os.path.isfile(filename):
        raise IOError("Input filename:{filename} doesn't exist, exit!")
    elif '.dax' not in filename:
        raise IOError("Input filename should be .dax format!")
    # decide crop_limits
    if crop_limits is not None:
        _limits = np.array(crop_limits, dtype=np.int)
    elif seg_label is not None:
        _limits = visual_tools.Extract_crop_from_segmentation(seg_label, extend_dim=extend_dim,
                                                              single_im_size=single_im_size)
    else: # no crop-limit specified
        _limits = None
    # check drift
    if len(drift) != 3:
        raise ValueError(f"Wrong input drift:{drift}, should be an array of 3")

    ## load image
    _ref_name = os.path.join(filename.split(
        os.sep)[-2], filename.split(os.sep)[-1])
    if verbose:
        print(f"- Start correcting {_ref_name} for channel:{channel}")
    _full_im_shape, _num_color = get_img_info.get_num_frame(filename,
                                                            frame_per_color=single_im_size[0],
                                                            buffer_frame=num_buffer_frames)
    # crop image
    _cropped_im, _dft_limits = visual_tools.crop_single_image(filename, channel, crop_limits=_limits,
                                                          all_channels=all_channels,
                                                          drift=drift, single_im_size=single_im_size,
                                                          num_buffer_frames=num_buffer_frames,
                                                          return_limits=True, verbose=verbose)
    ## corrections
    _corr_im = _cropped_im.copy()
    if z_shift_corr:
        # correct for z axis shift
        _corr_im = Z_Shift_Correction(
            _corr_im,  normalization=normalization, verbose=verbose)
    elif not z_shift_corr and normalization:
        # normalize if not doing z_shift_corr
        _corr_im = _corr_im / np.median(_corr_im)
    if hot_pixel_remove:
        # correct for hot pixels
        _corr_im = Remove_Hot_Pixels(_corr_im, hot_th=3, verbose=verbose)
    if illumination_corr:
        # illumination correction
        _corr_im = Illumination_correction(_corr_im, channel,
                                           crop_limits=_dft_limits,
                                           correction_folder=correction_folder,
                                           single_im_size=single_im_size,
                                           verbose=verbose)
    if chromatic_corr:
        # chromatic correction
        _corr_im = Chromatic_abbrevation_correction(_corr_im, channel,
                                                    single_im_size=single_im_size,
                                                    crop_limits=_dft_limits,
                                                    correction_folder=correction_folder,
                                                    verbose=verbose)
    ## return
    if return_limits:
        return _corr_im, _dft_limits
    else:
        return _corr_im

# generate bleedthrough 
def generate_bleedthrough_info(filename, ref_channel, bld_channel, single_im_size=_image_size, 
                               all_channels=_allowed_colors, num_buffer_frames=10,
                               normalization=False, illumination_corr=True, 
                               th_seed=2000, crop_window=9,
                               remove_boundary_pts=True, rsq_th=0.81, verbose=True):
    """Generate bleedthrough coefficient
    Inputs:
        filename: full filename for image, str
        ref_channel: channel for labeling, int or str (example: 750)
        bld_channel: channel for checking bleedthrough, int or str (example: 647)
        single_im_size: image size for single channel, array of 3 (default:[30,2048,2048])
        all_channels: all allowed channel list, list of channels (default:[750,647,561,488,405])
        num_buffer_frame: number of frames that is not used in zscan, int (default: 10)
        illumination_corr: whether do illumination correction, bool (default: Ture)
        th_seed: seeding threshold for getting candidate centers, int (default: 3000)
        crop_window: window size for cropping, int (default: 9)
        remove_boundary_pts: whether remove points that too close to boundary, bool (default: True)
        rsq_th: threshold for rsquare from linear regression between bldthrough and ref image, float (default: 0.81)
        verbose: say something!, bool (default: True)
    Outputs:
        picked_list: list of dictionary containing all info for bleedthrough correction"""
    from sklearn.linear_model import LinearRegression
    ## generate ref-image and bleed-through-image
    ref_channel = str(ref_channel)
    bld_channel = str(bld_channel)
    all_channels = [str(ch) for ch in all_channels]
    if ref_channel not in all_channels:
        raise ValueError(f"ref_channel:{ref_channel} should be in all_channels:{all_channels}")
    if bld_channel not in all_channels:
        raise ValueError(f"bld_channel:{bld_channel} should be in all_channels:{all_channels}")

    if verbose:
        print(f"-- acquiring ref_im and bld_im from file:{filename}")
    ref_im = correct_single_image(filename, ref_channel, single_im_size=single_im_size, 
                                  all_channels=all_channels, num_buffer_frames=num_buffer_frames,
                                  normalization=normalization, illumination_corr=illumination_corr, 
                                  chromatic_corr=False, verbose=verbose)
    bld_im = correct_single_image(filename, bld_channel, single_im_size=single_im_size, 
                                  all_channels=all_channels, num_buffer_frames=num_buffer_frames,
                                  normalization=normalization, illumination_corr=illumination_corr,
                                  chromatic_corr=False, verbose=verbose)
    # get candidate centers
    centers = visual_tools.get_STD_centers(ref_im, th_seed=th_seed, verbose=True)
    # pick sparse centers
    sel_centers = visual_tools.select_sparse_centers(centers, crop_window)
    
    ## crop images
    cropped_refs, cropped_blds = [], []
    # local parameter, cropping radius
    _radius = int((crop_window-1)/2)
    if _radius < 1:
        raise ValueError(f"Crop radius should be at least 1!")
    # loop through all centers
    for ct in sel_centers:
        if len(ct) != 3:
            raise ValueError(f"Wrong input dimension of centers, only expect [z,x,y] coordinates in center:{ct}")
        crop_l = np.array([np.zeros(3), np.round(ct-_radius)], dtype=np.int).max(0)
        crop_r = np.array([np.array(np.shape(ref_im)), 
                           np.round(ct-_radius)+crop_window], dtype=np.int).min(0)
        cropped_refs.append(ref_im[crop_l[0]:crop_r[0], crop_l[1]:crop_r[1], crop_l[2]:crop_r[2]])
        cropped_blds.append(bld_im[crop_l[0]:crop_r[0], crop_l[1]:crop_r[1], crop_l[2]:crop_r[2]])
    # remove centers that too close to boundary
    sel_centers = list(sel_centers)
    for _i, (_rim, _bim, _ct) in enumerate(zip(cropped_refs, cropped_blds, sel_centers)):
        if remove_boundary_pts and (_rim.shape-np.array([crop_window, crop_window, crop_window], dtype=np.int)).any():
            # pop points at boundary
            cropped_refs.pop(_i)
            cropped_blds.pop(_i)
            sel_centers.pop(_i)
    # check cropped image shape    
    cropped_shape = np.array([np.array(_cim.shape) for _cim in cropped_refs]).max(0)
    if (cropped_shape > crop_window).any():
        raise ValueError(f"Wrong dimension for cropped images:{cropped_shape}, should be of crop_window={crop_window} size")

    if verbose:
        print(f"-- {len(sel_centers)} centers are selected.")
    ## final picked list
    picked_list = []
    for _i, (_rim, _bim, _ct) in enumerate(zip(cropped_refs, cropped_blds, sel_centers)):
        _x = np.ravel(cropped_refs[_i])[:,np.newaxis]
        _y = np.ravel(cropped_blds[_i])
        _reg = LinearRegression().fit(_x,_y)
        if _reg.score(_x,_y) > rsq_th:
            _pair_dic = {'zxy': _ct,
                         'ref_im': _rim,
                         'bld_im': _bim,
                         'rsquare': _reg.score(_x,_y),
                         'slope': _reg.coef_,
                         'intercept': _reg.intercept_}
            picked_list.append(_pair_dic)
    if verbose:
        print(f"-- {len(picked_list)} pairs are saved.")

    return picked_list

# bleedthrough correction
def Bleedthrough_correction(input_im, crop_limits=None, all_channels=_allowed_colors,
                            correction_channels=None, single_im_size=_image_size,
                            num_buffer_frames=10, drift=np.array([0, 0, 0]), 
                            correction_folder=_correction_folder,
                            z_shift_corr=True, hot_pixel_remove=True,
                            profile_basename='Bleedthrough_correction_matrix',
                            profile_dtype=np.float, image_dtype=np.uint16,
                            return_limits=False, verbose=True):
    """Bleedthrough correction for a composite image
    Inputs:
        input_im: input image filename or list of images, str or list
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        all_channels: allowed channels to be corrected, list 
        single_im_size: full image size before any slicing, list of 3 (default:[30,2048,2048])
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        drift: 3d drift vector for this image, 1d-array (default:np.array([0,0,0]))
        correction_folder: correction folder to find correction profile, string of path (default: Z://Corrections/)
        profile_basename: base filename for bleedthrough correction profile file, str (default: 'Bleedthrough_correction_matrix')
        profile_dtype: data type for correction profile, numpy datatype (default: np.float)
        image_dtype: image data type, numpy datatype (default: np.uint16)
        return_limits: return modified limits, 3x2 np.ndarray
        verbose: say something!, bool (default: True)
    Outputs:
        _corr_ims: list of corrected images
    """
    ## check inputs
    # input_im
    _start_time = time.time()
    if isinstance(input_im, str):
        if not os.path.isfile(input_im):
            raise IOError(f"input image file:{input_im} doesn't exist, exit!")
    elif isinstance(input_im, list):
        if len(input_im) != 3:
            raise ValueError(
                f"input images should be 3, but {len(input_im)} are given!")
    else:
        raise ValueError(
            f"input_im should be str or list, but {type(input_im)} is given!")
    # crop_limits:
    if crop_limits is None:
        crop_limits = np.array([np.zeros(len(single_im_size)), np.array(
            single_im_size)], dtype=np.int).transpose()[:3]
    if len(crop_limits) != 2 and len(crop_limits) != 3:
        raise ValueError(
            f"crop_limits should be 2d or 3d, but {len(crop_limits)} is given!")
    # correction_channels:
    if correction_channels is None:
        correction_channels = all_channels[:3]
    elif len(correction_channels) != 3:
        raise ValueError("correction_channels should have 3 elements!")
    # correction profile
    _basename = profile_basename
    for _ch in correction_channels:
        _basename += '_'+str(_ch)
    profile_filename = os.path.join(correction_folder, _basename+'.npy')
    if not os.path.isfile(profile_filename):
        raise IOError(f"Bleedthrough correction profile:{profile_filename} doesn't exist, exit!")\

    ## start correction
    # load image if necessary
    if isinstance(input_im, str):
        _ims, _dft_limits = visual_tools.crop_multi_channel_image(input_im, correction_channels, 
                                                     crop_limits,
                                                     num_buffer_frames, all_channels, single_im_size,
                                                     drift, return_limits=True, verbose=verbose)
        # do zshift and hot-pixel correction
        # correct for z axis shift
        _ims = [Z_Shift_Correction(_cim, verbose=False) for _cim in _ims]
        # correct for hot pixels
        _ims = [Remove_Hot_Pixels(_cim, hot_th=3, verbose=False) for _cim in _ims]
    elif isinstance(input_im, list):
        if verbose:
            print(f"-- correcting bleedthrough for images")
        _ims = input_im
        _dft_limits = crop_limits
    _ims = [_im.astype(np.float) for _im in _ims]
    #print(time.time()-_start_time)
    # load profile
    if verbose:
        print("--- loading bleedthrough profile")
    _bld_profile = visual_tools.slice_image(profile_filename, [9, single_im_size[-2], single_im_size[-1]],
                                            [0, 9], crop_limits[-2], crop_limits[-1], image_dtype=profile_dtype)
    _bld_profile = _bld_profile.reshape(
        (3, 3, _bld_profile.shape[-2], _bld_profile.shape[-1]))
    #print(time.time()-_start_time)
    # do bleedthrough correction
    if verbose:
        print("--- applying bleedthrough correction")
    _corr_ims = []
    for _i in range(3):
        _nim = _ims[0]*_bld_profile[_i, 0] + _ims[1] * \
            _bld_profile[_i, 1] + _ims[2]*_bld_profile[_i, 2]
        _nim[_nim > np.iinfo(image_dtype).max] = np.iinfo(image_dtype).max
        _nim[_nim < np.iinfo(image_dtype).min] = np.iinfo(image_dtype).min
        _corr_ims.append(_nim.astype(image_dtype))
    #print(time.time()-_start_time)
    if return_limits:
        return _corr_ims, _dft_limits
    else:
        return _corr_ims

## correct selected channels from one dax file
def correct_one_dax(filename, sel_channels=None, crop_limits=None, seg_label=None,
                    extend_dim=20, single_im_size=_image_size, all_channels=_allowed_colors,
                    num_buffer_frames=10, drift=np.array([0, 0, 0]),
                    correction_folder=_correction_folder, bleed_corr=True,
                    z_shift_corr=True, hot_pixel_remove=True, illumination_corr=True, chromatic_corr=True,
                    return_limits=False, verbose=False):
    """wrapper for all correction steps to one image, used for multi-processing
    Inputs:
        filename: full filename of a dax_file or npy_file for one image, string
        sel_channels: selected channels to be extracted, list of str (for example. ['647'])
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        seg_label: segmentation label, 2D array
        extend_dim: extension pixel number if doing cropping, int (default: 20)
        single_im_size: z-x-y size of the image, list of 3
        all_channels: all_channels used in this image, list of str(for example, ['750','647','561'])
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        drift: 3d drift vector for this image, 1d-array
        correction_folder: path to find correction files
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        return_limits: whether return cropping limits, bool (default: False)
        verbose: whether say something!, bool (default:True)
    Outputs:
        _corr_ims: list of corrected images, list
        """
    ## check inputs
    # all_channels:
    all_channels = [str(_ch) for _ch in all_channels]
    # sel_channels:
    if sel_channels is None:
        sel_channels = all_channels[:3]
    else:
        sel_channels = [str(_ch) for _ch in sel_channels]
        for _ch in sel_channels:
            if _ch not in all_channels:
                raise ValueError(
                    f"All channels in selected channels should be in all_channels, but {_ch} is given")
    # correction_channels:
    if len(sel_channels) < 3:
        correction_channels = all_channels[:3]
    elif len(sel_channels) > 3:
        raise ValueError("correction_channels should have 3 elements!")
    else:
        correction_channels = sel_channels
    # decide crop_limits
    if crop_limits is not None:
        _limits = np.array(crop_limits, dtype=np.int)
    elif seg_label is not None:  # if segmentation_label is provided, then use this info
        _limits = visual_tools.Extract_crop_from_segmentation(seg_label, extend_dim=extend_dim,
                                                              single_im_size=single_im_size)
    else:  # no crop-limit specified
        _limits = None
    # check drift
    if len(drift) != 3:
        raise ValueError(f"Wrong input drift:{drift}, should be an array of 3")

    ## Start correction
    _ref_name = os.path.join(filename.split(os.sep)[-2],filename.split(os.sep)[-1])
    if verbose:
        print(f"- Start correct one dax file: {_ref_name}")
    # load image by bleedthrough correction
    if bleed_corr:
        _corr_ims, _dft_limits = Bleedthrough_correction(filename, _limits, all_channels=all_channels,
                                                         correction_channels=correction_channels, single_im_size=single_im_size,
                                                         num_buffer_frames=num_buffer_frames, drift=drift,
                                                         correction_folder=correction_folder,
                                                         z_shift_corr=z_shift_corr, hot_pixel_remove=hot_pixel_remove,
                                                         return_limits=True, verbose=verbose)
    else:
        if verbose:
            print(f"- loading image from {_ref_name} for channels:{correction_channels}")
        _corr_ims, _dft_limits = visual_tools.crop_multi_channel_image(filename, correction_channels, 
                                                                       _limits, num_buffer_frames, 
                                                                       all_channels, single_im_size,
                                                                       drift, return_limits=True, 
                                                                       verbose=verbose)


    # proceed with only selected channels
    _corr_ims = [_im for _im, _ch in zip(
        _corr_ims, correction_channels) if str(_ch) in sel_channels]
    # do z-shift and hot-pixel correction
    if not bleed_corr:
        # correct for z axis shift
        if z_shift_corr:
            # correct for z axis shift
            _corr_ims = [Z_Shift_Correction(
                _cim, verbose=verbose) for _cim in _corr_ims]
        if hot_pixel_remove:
            # correct for hot pixels
            _corr_ims = [Remove_Hot_Pixels(
                _cim, hot_th=3, verbose=verbose) for _cim in _corr_ims]
    # illumination and chromatic correction
    if illumination_corr:
        # illumination correction
        _corr_ims = [Illumination_correction(_cim, _ch,
                                             crop_limits=_dft_limits,
                                             correction_folder=correction_folder,
                                             single_im_size=single_im_size,
                                             verbose=verbose) for _cim, _ch in zip(_corr_ims, sel_channels)]
    if chromatic_corr:
        # chromatic correction
        _corr_ims = [Chromatic_abbrevation_correction(_cim, _ch,
                                                      single_im_size=single_im_size,
                                                      crop_limits=_dft_limits,
                                                      correction_folder=correction_folder,
                                                      verbose=verbose) for _cim, _ch in zip(_corr_ims, sel_channels)]
    if return_limits:
        return _corr_ims, _dft_limits
    else:
        return _corr_ims
