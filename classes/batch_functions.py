# Functions used in batch processing
import os, h5py, pickle, psutil
import numpy as np
from . import _allowed_kwds
from ..io_tools.load import correct_fov_image

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
                if len(_info) > 0 and _info[0] == _k:
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



def batch_process_image_to_spots(dax_filename, sel_channels, ref_filename, 
                                 correction_args={}, 
                                 save_image=False, save_filename=None, 
                                 data_type=None, region_ids=None,
                                 image_file_lock=None, overwrite_image=False, 
                                 drift_args={}, save_drift=True, drift_folder=None, 
                                 drift_file_lock=None, overwrite_drift=False, 
                                 fitting_args={}, save_spot=True, 
                                 spot_file_lock=None, overwrite_spot=False, 
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
    # ref_Filename 
    if not os.path.isfile(ref_filename):
        raise IOError(f"Dax file: {ref_filename} is not a file, exit!")
    if not isinstance(ref_filename, str) or ref_filename[-4:] != '.dax':
        raise IOError(f"Dax file: {ref_filename} has wrong data type, exit!")
    ## judge if drift correction is required
    if drift_folder is None:
        drift_folder = os.path.join(os.path.dirname(os.path.dirname(dax_filename)),
                            'Analysis', 'drift')
    _drift_filename = os.path.join(drift_folder, 
                        os.path.basename(dax_filename).replace('.dax', '_current_cor.pkl'))
    _key = os.path.join(os.path.basename(os.path.dirname(dax_filename)),
                        os.path.basename(dax_filename))
    # try to load drift                 
    _drift_dict = pickle.load(open(_drift_filename, 'rb'))
    if _key in _drift_dict and not overwrite_drift:
        if verbose:
            print(f"-- load drift from drift_dict: {_drift_filename}")
        _drift = _drift_dict[_key]
        print(_drift)
        _corr_drift = False 
    else:
        if verbose:
            print(f"-- no existing drift loaded, initialize drift.")
        _drift = np.array([0.,0.,0.])
        if ref_filename == dax_filename:
            _corr_drift = False
        else:
            _corr_drift = True
    # check save_image parameters
    if save_image:
        if data_type not in _allowed_kwds:
            raise ValueError(f"Wrong input data_type:{data_type}, should be among {_allowed_kwds}.")
        if save_filename is None:
            raise ValueError(f"Input save_filename:{save_filename} should be given.")
        if region_ids is None:
            raise ValueError(f"Input region_ids:{region_ids} should be given.")
        if len(region_ids) != len(sel_channels):
            raise ValueError(f"Wrong input region_ids:{region_ids}, should of same length as sel_channels:{sel_channels}.")
        region_ids = [int(_id) for _id in region_ids] # convert to ints
    ## correct images
    _sel_ims, _drift = correct_fov_image(dax_filename, sel_channels, 
                            calculate_drift=_corr_drift, drift=_drift,
                            ref_filename=ref_filename, 
                            return_drift=True, verbose=verbose, 
                            **correction_args, **drift_args)
    
    ## save image if specified
    # initiate lock
    if image_file_lock is not None:
        image_file_lock.acquire()
    # run saving
    _save_img_success = save_image_to_fov_file(
        save_filename, _sel_ims, data_type, region_ids, 
        _drift, image_file_lock,
        overwrite_image, verbose)
    # release lock
    if image_file_lock is not None:
        image_file_lock.release()

    ## save drift if specified
    # initiate lock
    if drift_file_lock is not None:
        drift_file_lock.acquire()
    # run saving
    _save_drift_success = save_drift_to_file(_drift_filename,
                                             dax_filename, _drift, 
                                             overwrite_drift, verbose)
    # release lock
    if drift_file_lock is not None:
        drift_file_lock.release()

    ## multi-fitting

    ## save fitted_spots if specified
    _spot_list = []

    return _spot_list
    
# save image to fov file
def save_image_to_fov_file(filename, ims, data_type, region_ids, drift=None,
                           lock=None, overwrite=False, verbose=True):
    """Function to save image to fov-standard savefile(hdf5)
    Inputs:
    
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
    _updated_ims = []
    _updated_drifts = []
    ## start saving
    with h5py.File(filename, "a", libver='latest') as _f:
        _grp = _f[data_type]
        for _i, (_id, _im) in enumerate(zip(region_ids, ims)):
            _index = list(_grp['ids'][:]).index(_id)
            _flag = _grp['flags'][_index]
            if _flag == 0 or overwrite:
                _grp['ims'][_index] = _im
                _grp['flags'][_index] = 1
                _updated_ims.append(_id)
                if drift is not None:
                    _grp['drifts'][_index] = _all_drifts[_i]
                    _updated_drifts.append(_id)
    if verbose:
        print(f"-- updated ims for id:{_updated_ims}, drifts for id:{_updated_drifts}")
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
    
