import numpy as np
import time
from scipy.ndimage.interpolation import map_coordinates

def warp_3d_image(image, drift, chromatic_profile=None, 
                  warp_order=1, border_mode='constant', 
                  verbose=False):
    """Warp image given chromatic profile and drift"""
    _start_time = time.time()
    # 1. get coordiates to be mapped
    single_im_size = np.array(image.shape)
    _coords = np.meshgrid( np.arange(single_im_size[0]), 
            np.arange(single_im_size[1]), 
            np.arange(single_im_size[2]), )
    _coords = np.stack(_coords).transpose((0, 2, 1, 3)) # transpose is necessary 
    # 2. calculate corrected coordinates if chormatic abbrev.
    if chromatic_profile is not None:
        _coords = _coords + chromatic_profile 
    # 3. apply drift if necessary
    _drift = np.array(drift)
    if _drift.any():
        _coords = _coords - _drift[:, np.newaxis,np.newaxis,np.newaxis]
    # 4. map coordinates
    _corr_im = map_coordinates(image, 
                               _coords.reshape(_coords.shape[0], -1),
                               order=warp_order,
                               mode=border_mode, cval=np.min(image))
    _corr_im = _corr_im.reshape(np.shape(image)).astype(image.dtype)
    if verbose:
        print(f"-- finish warp image in {time.time()-_start_time:.3f}s. ")
    return _corr_im