import numpy as np
from .. import _image_size

class ImageCrop():
    """"""
    def __init__(self, 
                 ndim, 
                 crop_array=None,
                 single_im_size=_image_size,
                 ):
        _shape = (ndim, 2)
        
        self.array = np.zeros(_shape, dtype=np.int32)
        if crop_array is None:
            self.array[:,1] = np.array(single_im_size)
        else:
            self.update(crop_array)
    
    def update(self, 
               crop_array, 
               ):
        _arr = np.array(crop_array, dtype=np.int32)
        if np.shape(_arr) == np.shape(self.array):
            self.array = _arr
        return
    
    def to_slices(self):
        return tuple([slice(_s[0], _s[1]) for _s in self.array])

        