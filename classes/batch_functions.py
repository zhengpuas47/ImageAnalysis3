# Functions used in batch processing
import os, h5py, pickle, psutil, time
import numpy as np
from scipy.sparse.extract import find
from scipy import ndimage
from . import _allowed_kwds, _image_dtype
from ..io_tools.load import correct_fov_image
from ..spot_tools.fitting import fit_fov_image, get_centers

Channel_2_SeedTh = {
    '750':600,
    '647':600,
    '561':600,
    '748':1000,
    '637':1000,
    '545':1000,    
}

## Process managing
def killtree(pid, including_parent=False, verbose=False):
    """Function to kill all children of a given process"""
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        if verbose:
            print ("child", child)
        child.kill()
    if including_parent:
        parent.kill()

def killchild(verbose=False):
    """Easy function to kill children of current process"""
    _pid = os.getpid()
    killtree(_pid, False, verbose)

## Parsing existing files
def _color_dic_stat(color_dic, channels, _type_dic=_allowed_kwds):
    """Extract number of targeted datatype images in color_dic"""
    _include_types = {}
    for _name, _k in _type_dic.items():
        for _fd, _infos in color_dic.items():
            for _ch, _info in zip(channels, _infos):
                if len(_info) > 0 and _info[0] == _k and 'chrom' not in _info:
                    if _name not in _include_types:
                        _include_types[_name] = {'ids':[], 'channels':[]}
                    # append
                    _include_types[_name]['ids'].append(int(_info.split(_k)[1]))
                    _include_types[_name]['channels'].append(_ch)
    # sort
    for _name, _dict in _include_types.items():
        _ids = _dict['ids']
        _chs = _dict['channels']
        _sorted_ids = [_id for _id in sorted(_ids)]
        _sorted_chs = [_ch for _id,_ch in sorted(zip(_ids, _chs))]
        _include_types[_name]['ids'] = _sorted_ids
        _include_types[_name]['channels'] = _sorted_chs
        
    return _include_types


def batch_process_image_to_spots(dax_filename, 
                                 sel_channels, 
                                 save_filename, 
                                 data_type, 
                                 region_ids,
                                 ref_filename, 
                                 load_file_lock=None, 
                                 warp_image=True, 
                                 correction_args={}, 
                                 save_image=True, 
                                 empty_value=0,
                                 fov_savefile_lock=None, 
                                 overwrite_image=False, 
                                 drift_args={}, 
                                 save_drift=True, 
                                 drift_filename=None, 
                                 drift_file_lock=None, 
                                 overwrite_drift=False, 
                                 fit_spots=True, 
                                 fit_in_mask=False, 
                                 fitting_args={}, 
                                 save_spots=True, 
                                 spot_file_lock=None, 
                                 overwrite_spot=False, 
                                 verbose=False):
    """run by multi-processing to batch process images to spots
    Inputs:

    Outputs:
        _spots: fitted spots for this image
    """
    ## check inputs
    # dax_filename
    if not os.path.isfile(dax_filename):
        raise IOError(f"Dax file: {dax_filename} is not a file, exit!")
    if not isinstance(dax_filename, str) or dax_filename[-4:] != '.dax':
        raise IOError(f"Dax file: {dax_filename} has wrong data type, exit!")
    # selected channels
    sel_channels = [str(ch) for ch in sel_channels]
    if verbose:
        print(f"+ batch process image: {dax_filename} for channels:{sel_channels}")
    # save filename
    if not os.path.isfile(save_filename):
        raise IOError(f"HDF5 file: {save_filename} is not a file, exit!")
    if not isinstance(save_filename, str) or save_filename[-5:] != '.hdf5':
        raise IOError(f"HDF5 file: {save_filename} has wrong data type, exit!")
    # ref_Filename 
    if isinstance(ref_filename, str):
        if not os.path.isfile(ref_filename):
            raise IOError(f"Dax file: {ref_filename} is not a file, exit!")
        elif ref_filename[-4:] != '.dax':
            raise IOError(f"Dax file: {ref_filename} has wrong data type, exit!")
    elif isinstance(ref_filename, np.ndarray):
        pass
    else:
        raise TypeError(f"ref_filename should be np.ndarray or string of path, but {type(ref_filename)} is given")
    ## region-ids
    if len(region_ids) != len(sel_channels):
        raise ValueError(f"Wrong input region_ids:{region_ids}, should of same length as sel_channels:{sel_channels}.")
    region_ids = [int(_id) for _id in region_ids] # convert to ints
    
    # judge if images exist
    # initiate lock
    if 'fov_savefile_lock' in locals() and fov_savefile_lock is not None:
        fov_savefile_lock.acquire()
    _ims, _warp_flags, _drifts = load_image_from_fov_file(save_filename, 
                                                    data_type, region_ids,
                                                    load_drift=True, 
                                                    verbose=verbose)
    # release lock
    if 'fov_savefile_lock' in locals() and fov_savefile_lock is not None:
        fov_savefile_lock.release()

    # determine which image should be processed
    # initialize processing images and channels
    _process_flags = []
    _process_sel_channels = []
    # initialzie carried over images and channels
    _carryover_ims = []
    _carryover_sel_channels = []
    for _im, _flg, _drift, _rid, _ch in zip(_ims, _warp_flags, _drifts, region_ids, sel_channels):
        # if decided to overwrite image or overwrite drift, proceed
        if overwrite_image or overwrite_drift:
            _process_flags.append(True)
            _process_sel_channels.append(_ch)
        else:
            if (_im != empty_value).any()  and _flg-1 == int(warp_image): # and (_drift!= empty_value).any() # remove this drift requirement, because it could be zero
                # image exist, no need to process from beginning
                _process_flags.append(False)
                _carryover_ims.append(_im.copy() )
                _carryover_sel_channels.append(_ch)
            else:
                _process_flags.append(True)
                _process_sel_channels.append(_ch)
    # release RAM
    del(_ims)

    # convert this processed drifts
    _process_drift = list(set([tuple(_dft) for _dft in _drifts]))

    # one unique non-zero drift exist, directly use it
    if len(_process_drift) == 1 and np.array(_process_drift[0]).any() and not overwrite_drift:
        _process_drift = np.array(_process_drift[0])
        _corr_drift = False
    # no drift
    else: 
        _process_drift = np.zeros(len(_process_drift[0]))
        _corr_drift = True
        
    ## if any image to be processed:
    if np.sum(_process_flags) > 0:

        if verbose:
            print(f"-- {_process_sel_channels} images are required to process, {_carryover_sel_channels} images are loaded from save file: {save_filename}")

        ## correct images
        if warp_image:
            _processed_ims, _drift, _drift_flag = correct_fov_image(
                dax_filename, 
                _process_sel_channels, 
                load_file_lock=load_file_lock,
                calculate_drift=_corr_drift, 
                drift=_process_drift,
                ref_filename=ref_filename, 
                warp_image=warp_image,
                return_drift=True, verbose=verbose, 
                **correction_args, **drift_args)
        else:
            _processed_ims, _processed_warp_funcs, _drift, _drift_flag = correct_fov_image(
                                    dax_filename, 
                                    _process_sel_channels, 
                                    load_file_lock=load_file_lock,
                                    calculate_drift=_corr_drift, 
                                    drift=_process_drift,
                                    ref_filename=ref_filename, 
                                    warp_image=warp_image,
                                    return_drift=True, verbose=verbose, 
                                    **correction_args, **drift_args)
    # nothing processed, create empty list
    else:
        _processed_ims = []
        if not warp_image:
            _processed_warp_funcs = []
        _drift = np.array(_process_drift) # use old drift
        _drift_flag = 0

    ## merge processed and carryover images
    _sel_ims = []
    for _ch, _flg in zip(sel_channels, _process_flags):
        if not _flg:
            _sel_ims.append(_carryover_ims.pop(0))
        else:
            _sel_ims.append(_processed_ims.pop(0))

    if not warp_image:
        _warp_funcs = []
        for _ch, _flg in zip(sel_channels, _process_flags):
            if not _flg:
                from ..correction_tools.chromatic import generate_chromatic_function
                _warp_funcs.append(
                    generate_chromatic_function(correction_args['chromatic_profile'][str(_ch)], _drift)
                )
            else:
                _warp_funcs.append(
                    _processed_warp_funcs.pop(0)
                )

    ## save image if specified
    if save_image:
        # initiate lock
        if 'fov_savefile_lock' in locals() and fov_savefile_lock is not None:
            fov_savefile_lock.acquire()
        # run saving
        _save_img_success = save_image_to_fov_file(
            save_filename, _sel_ims, data_type, region_ids, 
            warp_image, _drift, _drift_flag, 
            overwrite_image, verbose) # this step also save drift
        # release lock
        if 'fov_savefile_lock' in locals() and fov_savefile_lock is not None:
            fov_savefile_lock.release()

    ## multi-fitting
    if fit_spots:
        # check fit_in_mask
        if fit_in_mask:
            if 'seed_mask' not in fitting_args or fitting_args['seed_mask'] is None:
                raise KeyError(f"seed_mask should be given if fit_in_mask specified")

            if warp_image:
                _shifted_mask = fitting_args['seed_mask']
            else:
                # translate this mask according to drift
                if verbose:
                    print(f"-- start traslating seed_mask by drift: {_drift}", end=' ')
                    _translate_start = time.time()
                _shifted_mask = ndimage.shift(fitting_args['seed_mask'], 
                                            -_drift, 
                                            mode='constant', 
                                            cval=0)
            # store seed_mask
            fitting_args['seed_mask'] = _shifted_mask
            
            if verbose:
                print(f"-- in {time.time()-_translate_start:.2f}s.")
                _translate_start = time.time()
        _raw_spot_list = []
        _spot_list = []
        # get threshold
        
        for _ich, (_im, _ch) in enumerate(zip(_sel_ims, sel_channels)):
            fitting_args['th_seed'] = Channel_2_SeedTh[str(_ch)]

            _raw_spots = fit_fov_image(
                _im, _ch, verbose=verbose, 
                **fitting_args,
            )
            if not warp_image:
                # update spot coordinates given warp functions, if image was not warpped.
                _func = _warp_funcs[_ich]
                _spots = _func(_raw_spots)
                #print(f"type: {type(_spots)} for {dax_filename}, region {region_ids[_ich]} channel {_ch}, {_func}")
            else:
                _spots = _raw_spots.copy()
            # append 
            _spot_list.append(_spots)
            _raw_spot_list.append(_raw_spots)
        ## save fitted_spots if specified
        if save_spots:
            # initiate lock
            if spot_file_lock is not None:
                spot_file_lock.acquire()
            # run saving
            _save_spt_success = save_spots_to_fov_file(
                save_filename, _spot_list, data_type, region_ids, 
                raw_spot_list=_raw_spot_list,
                overwrite=overwrite_spot, verbose=verbose)
            # release lock
            if spot_file_lock is not None:
                spot_file_lock.release()
    else:
        _spot_list = np.array([])

    return

# save image to fov file
def save_image_to_fov_file(filename, ims, data_type, region_ids, 
                           warp_image=False, drift=None, drift_flag=None,
                           overwrite=False, verbose=True):
    """Function to save image to fov-standard savefile(hdf5)
    Inputs:
        filename: fov class hdf5 saving filename, string of file path
        ims: images to be saved, list of np.ndarray 
        data_type: data type used to load, string
        region_ids: corresponding region ids of given data_type, 
            should match length of ims, list of ints
        warp_image: whether image was warpped or not, bool (default: False)
        drift: whether drift exist and whether we are going to save it, bool (default: None, not saving)
        overwrite: whether overwrite existing data, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
    """
    ## check inputs
    if not os.path.isfile(filename):
        raise IOError(f"save file: {filename} doesn't exist!")
    if data_type not in _allowed_kwds:
        raise ValueError(f"Wrong input data_type:{data_type}, should be among {_allowed_kwds}.")
    if len(ims) != len(region_ids):
        raise ValueError(f"Wrong input region_ids:{region_ids}, should of same length as ims, len={len(ims)}.")
    if drift is not None:
        if len(np.shape(drift)) == 1:
            _all_drifts = [drift for _im in ims]
        elif len(drift) == len(ims):
            _all_drifts = drift
        else:
            raise IndexError(f"Length of drift should match ims")
    if verbose:
        print(f"- writting {data_type} info to file:{filename}")
        _save_start = time.time()
    _updated_ims = []
    _updated_drifts = []
    _saving_flag = False 
    ## start saving
    with h5py.File(filename, "a", libver='latest') as _f:
        _grp = _f.require_group(data_type) # change to require_group
        for _i, (_id, _im) in enumerate(zip(region_ids, ims)):
            _index = list(_grp['ids'][:]).index(_id)
            _flag = _grp['flags'][_index]
            # if not been written or overwrite:
            if _flag == 0 or overwrite:
                _saving_flag = True 
                _grp['ims'][_index] = _im
                # warpping image flag
                if not warp_image:
                    _grp['flags'][_index] = 1 # 1 as markers of un-wrapped iamges
                else:
                    _grp['flags'][_index] = 2 # 2 as markers of warpped images
                _updated_ims.append(_id)
                if drift is not None:
                    _grp['drifts'][_index,:] = _all_drifts[_i]
                    _updated_drifts.append(_id)
                
    if verbose:
        if _saving_flag:
            print(f"-- updated ims for id:{_updated_ims}, drifts for id:{_updated_drifts} in {time.time()-_save_start:.3f}s")
        else:
            print(f"-- images and drifts already exist, skip.")

    # return success flag    
    return _saving_flag

# load image from fov file
def load_image_from_fov_file(filename, data_type, region_ids,
                             image_dtype=_image_dtype, load_drift=False, verbose=True):
    """Function to load images from fov class file
    Inputs:
        filename: fov class hdf5 saving filename, string of file path
        data_type: data type used to load, string
        region_ids: corresponding region ids of given data_type, list of ints
        verbose: say something!, bool (default: True)
    Outputs:
        _ims: images in the order of region_ids provided, list of np.ndarray
        _flags: whether these images were warpped (==2), list of ints
    """
    ## check inputs
    if not os.path.isfile(filename):
        raise IOError(f"load file: {filename} doesn't exist!")
    if data_type not in _allowed_kwds:
        raise ValueError(f"Wrong input data_type:{data_type}, should be among {_allowed_kwds}.")
    if isinstance(region_ids, int) or isinstance(region_ids, np.int):
        _region_ids = [int(region_ids)]
    elif isinstance(region_ids, list) or isinstance(region_ids, np.ndarray):
        _region_ids = [int(_id) for _id in region_ids]
    else:
        raise TypeError(f"Wrong input type for region_ids:{region_ids}")
    
    if verbose:
        print(f"- loading {data_type} info from file:{os.path.basename(filename)}", end=' ')
        _load_start = time.time()
    ## start loading
    _ims = []
    _flags = []
    if load_drift:
        _drifts = []
    with h5py.File(filename, "a", libver='latest') as _f:
        # get the group
        _grp = _f[data_type]
        # get index
        for _i, _id in enumerate(_region_ids):
            _index = list(_grp['ids'][:]).index(_id)
            # extract images and flag
            _ims.append(_grp['ims'][_index])
            _flags.append(_grp['flags'][_index])
            if load_drift:
                _drifts.append(_grp['drifts'][_index,:])
    if verbose:
        print(f"in {time.time()-_load_start:.3f}s.")
    if load_drift:
        return _ims, _flags, _drifts
    else:
        return _ims, _flags

# save image to fov file
def save_spots_to_fov_file(filename, spot_list, data_type, region_ids, 
                           raw_spot_list=None,
                           overwrite=False, verbose=True):
    """Function to save image to fov-standard savefile(hdf5)
    Inputs:
    
    Outputs:
    """
    ## check inputs
    if not os.path.isfile(filename):
        raise IOError(f"save file: {filename} doesn't exist!")
    if data_type not in _allowed_kwds:
        raise ValueError(f"Wrong input data_type:{data_type}, should be among {_allowed_kwds}.")
    if len(spot_list) != len(region_ids):
        raise ValueError(f"Wrong input region_ids:{region_ids}, should of same length as spots, len={len(spot_list)}.")
    if raw_spot_list is not None and len(raw_spot_list) != len(spot_list):
        raise IndexError(f"length of input spot_list and raw_spot list should match, {len(spot_list)}, {len(raw_spot_list)}")

    if verbose:
        print(f"- writting {data_type} spots into file:{filename}")
        _save_start = time.time()
    _updated_spots = []
    ## start saving
    with h5py.File(filename, "a", libver='latest') as _f:
        _grp = _f[data_type]
        for _i, (_id, _spots) in enumerate(zip(region_ids, spot_list)):
            # check size of this spot save
            _saved_shape = _grp['spots'].shape
            _max_shape = _grp['spots'].maxshape
            # if not large enough with maxshape, recreate this saving buffer
            if _saved_shape[1] < len(_spots) and _max_shape[1] is not None and _max_shape[1]< len(_spots):
                if verbose:
                    print(f"-- recreate {data_type}_spots and {data_type}_raw_spots from {_saved_shape[1]} to {len(_spots)}.")
                # retrieve existing values
                _existing_spots = _grp['spots'][:]
                _existing_raw_spots = _grp['raw_spots'][:]
                if verbose:
                    print(f"--- deleting spots and raw_spots")
                # delete datasets
                del(_grp['spots'])
                del(_grp['raw_spots'])
                # resave existing spots
                if verbose:
                    print(f"--- recreating spots and raw_spots")
                _grp.create_dataset('spots',
                                    (_saved_shape[0], len(_spots), _saved_shape[2]), 
                                    dtype='f', maxshape=(_saved_shape[0], None, _saved_shape[2]), chunks=True)
                _grp['spots'][:,:_saved_shape[1],:] = _existing_spots
                # resave existing raw_spots
                _grp.create_dataset('raw_spots',
                                    (_saved_shape[0], len(_spots), _saved_shape[2]), 
                                    dtype='f', maxshape=(_saved_shape[0], None, _saved_shape[2]), chunks=True)
                _grp['raw_spots'][:,:_saved_shape[1],:] = _existing_raw_spots
            # if maxshape allowed, simply reshape
            elif _saved_shape[1] < len(_spots):
                if verbose:
                    print(f"-- resize {data_type}_spots and {data_type}_raw_spots from {_saved_shape[1]} to {len(_spots)}.")
                _grp['spots'].resize(len(_spots), 1)
                _grp['raw_spots'].resize(len(_spots), 1)

            _index = list(_grp['ids'][:]).index(_id)
            if np.sum(_grp['spots'][_index])==0 or overwrite:
                _grp['spots'][_index, :len(_spots), :] = _spots
                _updated_spots.append(_id)
            if 'raw_spots' in _grp.keys():
                if np.sum(_grp['raw_spots'][_index])==0 or overwrite:
                    _grp['raw_spots'][_index, :len(raw_spot_list[_i]), :] = raw_spot_list[_i]

    if verbose:
        print(f"-- updated spots for id:{_updated_spots} in {time.time()-_save_start:.3f}s")
    # return success flag    
    return True

# save drift to file
def save_drift_to_file(drift_filename, image_filename, drift, overwrite=False, verbose=True):
    """Save drift for one image to drift file"""
    ## check inputs
    if os.path.isfile(drift_filename):
        drift_dict = pickle.load(open(drift_filename, 'rb'))
    else:
        drift_dict = {}
    # update drift
    _update = False
    _key = os.path.join(os.path.basename(os.path.dirname(image_filename)),
                        os.path.basename(image_filename))
    if _key not in drift_dict or overwrite:
        drift_dict[_key] = drift
        _update = True
    # save
    if _update:
        if verbose:
            print(f"-- update drift of {_key} into file:{drift_filename}")
        pickle.dump(drift_dict, open(drift_filename, 'wb'))
    else:
        if verbose:
            print(f"-- no updates in drift, skip.")
    # return success flag
    return True
    

# create drift file
def create_drift_file(drift_filename, ref_filename, 
                      n_dim=3,
                      overwrite=False, verbose=True):
    """Function to create drift save file"""
    ## check inputs
    if os.path.isfile(drift_filename) and not overwrite:
        drift_dict = pickle.load(open(drift_filename, 'rb'))
    else:
        drift_dict = {}
    _ref_key = os.path.join(os.path.basename(os.path.dirname(ref_filename)),
                            os.path.basename(ref_filename))
    if _ref_key not in drift_dict:
        drift_dict[_ref_key] = np.zeros(n_dim)
        _update = True
    else:
        _update = False
    if _update:
        # create folder 
        if not os.path.isdir(os.path.dirname(drift_filename)):
            if verbose:
                print(f"--- creating folder:{os.path.dirname(drift_filename)}")
            os.makedirs(os.path.dirname(drift_filename))
        # save initialized drift_dict
        if verbose:
            print(f"-- create drift file:{drift_filename} with reference:{_ref_key}")
        pickle.dump(drift_dict, open(drift_filename, 'wb'))
    else:
        if verbose:
            print(f"-- no updates in drift file:{drift_filename}, skip.")
    
    return True 



