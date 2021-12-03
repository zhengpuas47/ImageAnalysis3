from .. import _image_size
import numpy as np


class ImageCrop():
    """ """
    def __init__(self, 
                 ndim, 
                 crop_array=None,
                 single_im_size=_image_size,
                 ):
        _shape = (ndim, 2)
        self.ndim = ndim
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

    def inside(self, coords):
        """Check whether given coordinate is in this crop"""
        _coords = np.array(coords)
        if len(np.shape(_coords)) == 1:
            _coords = _coords[np.newaxis,:]
        elif len(np.shape(_coords)) > 2:
            raise IndexError("Only support single or multiple coordinates")
        # find kept spots
        _masks = [(_coords[:,_d] >= self.array[_d,0]) *\
                  (_coords[:,_d] <= self.array[_d,1])
                  for _d in range(self.ndim)]
        _mask = np.prod(_masks, axis=0).astype(np.bool)

        return _mask

    def distance_to_edge(self, coord):
        """Check distance of a coordinate to the edge of this crop"""
        _coord = np.array(coord)[:self.ndim]
        return np.min(np.abs(_coord[:,np.newaxis] - self.array))


    def crop_coords(self, coords):
        """ """
        _coords = np.array(coords)
        _mask = self.inside(coords)
        _cropped_coords = _coords[_mask] - self.array[:,0][np.newaxis,:]
        
        return _cropped_coords

    def overlap(self, crop2):
        
        # find overlaps
        _llim = np.max([self.array[:,0], crop2.array[:,0]], axis=0)
        
        _rlim = np.min([self.array[:,1], crop2.array[:,1]], axis=0)

        if (_llim > _rlim).any():
            return None
        else:
            return ImageCrop(len(_llim), np.array([_llim, _rlim]).transpose())

    def relative_overlap(self, crop2):
        _overlap = self.overlap(crop2)
        if _overlap is not None:
            _overlap.array = _overlap.array - self.array[:,0][:, np.newaxis]

        return _overlap



class ImageCrop_3d(ImageCrop):
    """ """
    def __init__(self, 
                 crop_array=None,
                 single_im_size=_image_size,
                 ):
    
        super().__init__(3, crop_array, single_im_size)

    def crop_spots(self, spots):
        """ """
        _spots = np.array(spots)
        _coords = _spots[:,1:4]
        _mask = self.inside(_coords)
        _cropped_spots = _spots[_mask]
        _cropped_spots[:,1:4] = _cropped_spots[:,1:4] - self.array[:,0][np.newaxis,:]
        
        return _cropped_spots

    def overlap(self, crop2):
        _returned_crop = super().overlap(crop2)
        if _returned_crop is None:
            return None
        else:
            return ImageCrop_3d(_returned_crop.array)



class Spots3D(np.ndarray):
    """Class for fitted spots in 3D"""
    def __new__(cls, 
                input_array, 
                bits=None,
                pixel_size=None,
                info=None,
                copy_data=True):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        if copy_data:
            input_array = np.array(input_array).copy()
        if len(np.shape(input_array)) == 1:
            obj = np.asarray([input_array]).view(cls)
        elif len(np.shape(input_array)) == 2:
            obj = np.asarray(input_array).view(cls)
        else:
            raise IndexError('Spots3D class only creating 2D-array')
        # add the new attribute to the created instance
        if isinstance(bits, int):
            obj.bits = np.ones(len(obj), dtype=np.int32) * int(bits)
        elif bits is not None and np.size(bits) == 1:
            obj.bits = np.ones(len(obj), dtype=np.int32) * int(bits[0])
        elif bits is not None and len(bits) == len(obj):
            obj.bits = np.array(bits, dtype=np.int32) 
        else:
            obj.bits = bits

        obj.pixel_size = np.array(pixel_size)
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __str__(self):
        """ """
        return ""

    def __array_finalize__(self, obj):
        """ """
        # see InfoArray.__array_finalize__ for comments
        if obj is None: 
            return
        else:
            self.info = getattr(obj, 'info', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """ """
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Spots3D):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Spots3D):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], Spots3D):
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(Spots3D)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], Spots3D):
            results[0].info = info

        return results[0] if len(results) == 1 else results

    def to_coords(self):
        """ convert into 3D coordinates in pixels """
        return np.array(self[:,1:4])
    
    def to_positions(self, pixel_size=None):
        """ convert into 3D spatial positions"""
        _saved_pixel_size = getattr(self, 'pixel_size', None)
        if _saved_pixel_size is not None and _saved_pixel_size.any():
            return self.to_coords() * np.array(_saved_pixel_size)[np.newaxis,:]
        elif pixel_size is None:
            raise ValueError('pixel size not given')
        else:
            return self.to_coords() * np.array(pixel_size)[np.newaxis,:]

    def to_intensities(self):
        """ """
        return np.array(self[:,0])



# scoring spot Tuple
class SpotTuple():
    """Tuple of coordinates"""
    def __init__(self, 
                 spots_tuple:Spots3D,
                 bits:np.ndarray,
                 pixel_size:np.ndarray or list,
                 spots_inds=None,
                 tuple_id=None,
                 ):
        # add spot Tuple
        self.spots = spots_tuple[:].copy()
        # add information for bits
        # add the new attribute to the created instance
        if isinstance(bits, int):
            self.bits = np.ones(len(self.spots), dtype=np.int32) * int(bits)
        elif bits is not None and np.size(bits) == 1:
            self.bits = np.ones(len(self.spots), dtype=np.int32) * int(bits[0])
        elif bits is not None:
            self.bits = np.array(bits[:len(self.spots)], dtype=np.int32) 
        elif spots_tuple.bits is not None:
            self.bits = spots_tuple.bits[:len(self.spots)]
        else:
            self.bits = bits

        self.pixel_size = np.array(pixel_size)
        
        self.spots_inds = spots_inds
        self.tuple_id = tuple_id
        

    def dist_internal(self):
        from scipy.spatial.distance import pdist
        _self_coords = self.spots.to_positions(self.pixel_size)
        return pdist(_self_coords)

    def intensities(self):
        return self.spots.to_intensities()
    def intensity_mean(self):
        return np.mean(self.spots.to_intensities())

    def centroid_spot(self):
        self.centroid = np.mean(self.spots, axis=0, keepdims=True)
        return self.centroid

    def dist_centroid_to_spots(self, spots:Spots3D):
        from scipy.spatial.distance import cdist
        if not hasattr(self, 'centroid'):
            _cp = self.centroid_spot()
        else:
            _cp = getattr(self, 'centroid')
        _centroid_coords = _cp.to_positions(pixel_size=self.pixel_size)
        _target_coords = spots.to_positions(pixel_size=self.pixel_size)
        return cdist(_centroid_coords, _target_coords)[0]

    def dist_to_spots(self, 
                      spots:Spots3D):
        from scipy.spatial.distance import cdist
        _self_coords = self.spots.to_positions(pixel_size=self.pixel_size)
        _target_coords = spots.to_positions(pixel_size=self.pixel_size)
        return cdist(_self_coords, _target_coords)

    def dist_chromosome(self):
        pass

