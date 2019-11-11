# Functions used in batch processing
import os
import psutil
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
                                 correction_args={}, save_image=False, 
                                 image_manager=None, overwrite_image=False, 
                                 drift_args={}, save_drift=True, 
                                 drift_manager=None, overwrite_drift=False, 
                                 fitting_args={}, save_spot=True, 
                                 spot_namager=None, overwrite_spot=False, 
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
    # judge if drift correction is required
    if ref_filename == dax_filename:
        _corr_drift = False
        _drift = np.array([0,0,0])
    else:
        _corr_drift = True
    # correct images


         
        
    _sel_ims = correct_fov_image(dax_filename, sel_channels, 
                                 verbose=verbose, **correction_args)



    return 
    