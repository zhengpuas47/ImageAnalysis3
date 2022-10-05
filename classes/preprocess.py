import numpy as np
import pickle
import warnings
# functions
from scipy.spatial.distance import cdist, pdist
from scipy.ndimage.interpolation import map_coordinates
from tqdm import tqdm
# default info for spots
from .. import _image_size
#3from ..spot_tools import _3d_infos, _3d_spot_infos, _spot_coord_inds

# default params
_3d_spot_infos = ['height', 'z', 'x', 'y', 'background', 'sigma_z', 'sigma_x', 'sigma_y', 'sin_t', 'sin_p', 'eps']
_3d_infos = ['z', 'x', 'y']
_spot_coord_inds = [_3d_spot_infos.index(_info) for _info in _3d_infos]

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
        if len(single_im_size) == ndim:
            self.image_sizes = np.array(single_im_size, dtype=np.int32)
        
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

    def crop_spots(self, spots_3d):
        """ """
        _spots = spots_3d.copy()
        _coords = _spots[:,1:4]
        _mask = self.inside(_coords)
        _cropped_spots = _spots[_mask].copy()
        _cropped_spots[:,1:4] = np.array(_cropped_spots[:,1:4]) - self.array[:,0][np.newaxis,:]
        
        return _cropped_spots

    def overlap(self, crop2):
        _returned_crop = super().overlap(crop2)
        if _returned_crop is None:
            return None
        else:
            return ImageCrop_3d(_returned_crop.array)

    def translate_drift(self, drift=None):
        if drift is None:
            _drift = np.zeros(self.ndim, dtype=np.int32)
        else:
            _drift = np.round(drift).astype(np.int32)
        _new_box = []
        for _limits, _d, _sz in zip(self.array, _drift, self.image_sizes):
            _new_limits = [
                max(0, _limits[0]-_d),
                min(_sz, _limits[1]-_d),
            ]
            _new_box.append(np.array(_new_limits, dtype=np.int32))
        _new_box = np.array(_new_box)
        # generate new crop
        _new_crop = ImageCrop_3d(_new_box, self.image_sizes)
        #print(_drift, _new_box)
        return _new_crop

class Spots3D(np.ndarray):
    """Class for fitted spots in 3D"""
    def __new__(cls, 
                input_array, 
                bits=None,
                pixel_sizes=None,
                channels=None,
                copy_data=True,
                intensity_index=0,
                coordinate_indices=[1,2,3]):
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
        if isinstance(bits, (int, np.int32)):
            obj.bits = np.ones(len(obj), dtype=np.int32) * int(bits)
        elif bits is not None and np.size(bits) == 1:
            obj.bits = np.ones(len(obj), dtype=np.int32) * int(bits[0])
        elif bits is not None and len(bits) == len(obj):
            obj.bits = np.array(bits, dtype=np.int32) 
        else:
            obj.bits = bits
        # channels
        if isinstance(channels, bytes):
            channels = channels.decode()
        if isinstance(channels, (int, np.int32)):
            obj.channels = np.ones(len(obj), dtype=np.int32) * int(channels)
        elif channels is not None and isinstance(channels, str):
            obj.channels = np.array([channels]*len(obj))
        elif channels is not None and len(channels) == len(obj):
            obj.channels = np.array(channels) 
        else:
            obj.channels = channels
        # others
        obj.pixel_sizes = np.array(pixel_sizes)
        obj.intensity_index = int(intensity_index)
        obj.coordinate_indices = np.array(coordinate_indices, dtype=np.int32)
        # default parameters
        obj._3d_infos = _3d_infos
        obj._3d_spot_infos = _3d_spot_infos
        obj._spot_coord_inds = np.array(_spot_coord_inds)
        # Finally, we must return the newly created object:
        return obj

    #    def __str__(self):
    #        """Spots3D object with dimension"""
    #        return ""

    def __getitem__(self, key):
        """Modified getitem to allow slicing of bits as well"""
        #print(f" getitem {key}, {type(key)}", self.shape)
        new_obj = super().__getitem__(key)
        # if slice, slice bits as well
        if hasattr(self, 'bits') and getattr(self, 'bits') is not None and len(np.shape(getattr(self, 'bits')))==1:
            if isinstance(key, slice) or isinstance(key, np.ndarray) or isinstance(key, int):
                setattr(new_obj, 'bits', getattr(self, 'bits')[key] )
        if hasattr(self, 'channels') and getattr(self, 'channels') is not None and len(np.shape(getattr(self, 'channels')))==1:
            if isinstance(key, slice) or isinstance(key, np.ndarray) or isinstance(key, int):
                setattr(new_obj, 'channels', getattr(self, 'channels')[key] )
        #print(new_obj, type(new_obj))
        return new_obj

    def __setitem__(self, key, value):
        #print(f" setitem {key}, {type(key)}")
        return super().__setitem__(key, value)

    def __array_finalize__(self, obj):
        """
        Reference: https://numpy.org/devdocs/user/basics.subclassing.html 
        """
        if obj is None: 
            return
        else:
            if hasattr(obj, 'shape') and len(getattr(obj, 'shape')) != 2:
                obj = np.array(obj)
            # other attributes
            setattr(self, 'bits', getattr(obj, 'bits', None))
            setattr(self, 'channels', getattr(obj, 'channels', None))
            setattr(self, 'pixel_sizes', getattr(obj, 'pixel_sizes', None))
            setattr(self, 'intensity_index', getattr(obj, 'intensity_index', None))
            setattr(self, 'coordinate_indices', getattr(obj, 'coordinate_indices', None))
            setattr(self, '_3d_infos', getattr(obj, '_3d_infos', None))
            setattr(self, '_3d_spot_infos', getattr(obj, '_3d_spot_infos', None))
            setattr(self, '_spot_coord_inds', getattr(obj, '_spot_coord_inds', None))
        #print(f"**finalizing, {obj}, {type(obj)}")
        return obj

    def to_coords(self):
        """ convert into 3D coordinates in pixels """
        _coordinate_indices = getattr(self, 'coordinate_indices', np.array([1,2,3]))
        return np.array(self[:,_coordinate_indices])
    
    def to_positions(self, pixel_sizes=None):
        """ convert into 3D spatial positions"""
        _saved_pixel_sizes = getattr(self, 'pixel_sizes', None)
        if _saved_pixel_sizes is not None and _saved_pixel_sizes.any():
            return self.to_coords() * np.array(_saved_pixel_sizes)[np.newaxis,:]
        elif pixel_sizes is None:
            raise ValueError('pixel_sizes not given')
        else:
            return self.to_coords() * np.array(pixel_sizes)[np.newaxis,:]

    def to_intensities(self):
        """ """
        _intensity_index = getattr(self, 'intensity_index', 0 )
        return np.array(self[:,_intensity_index])

# scoring spot Tuple
class SpotTuple():
    """Tuple of coordinates"""
    def __init__(self, 
                 spots_tuple:Spots3D,
                 bits:np.ndarray=None,
                 pixel_sizes:np.ndarray or list=None,
                 spots_inds=None,
                 tuple_id=None,
                 ):
        # add spot Tuple
        self.spots = spots_tuple[:].copy()
        # add information for bits
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
        if pixel_sizes is None:
            self.pixel_sizes = getattr(self.spots, 'pixel_sizes', None)
        else:
            self.pixel_sizes = np.array(pixel_sizes)
        
        self.spots_inds = spots_inds
        self.tuple_id = tuple_id
        
    def dist_internal(self):
        _self_coords = self.spots.to_positions(self.pixel_sizes)
        return pdist(_self_coords)

    def intensities(self):
        return self.spots.to_intensities()
    def intensity_mean(self):
        return np.mean(self.spots.to_intensities())

    def centroid_spot(self):
        self.centroid = np.mean(self.spots, axis=0, keepdims=True)
        self.centroid.pixel_sizes = self.pixel_sizes
        return self.centroid

    def dist_centroid_to_spots(self, spots:Spots3D):
        """Calculate distance from tuple centroid to given spots"""
        if not hasattr(self, 'centroid'):
            _cp = self.centroid_spot()
        else:
            _cp = getattr(self, 'centroid')
        _centroid_coords = _cp.to_positions(pixel_sizes=self.pixel_sizes)
        _target_coords = spots.to_positions(pixel_sizes=self.pixel_sizes)
        return cdist(_centroid_coords, _target_coords)[0]

    def dist_to_spots(self, 
                      spots:Spots3D):
        _self_coords = self.spots.to_positions(pixel_sizes=self.pixel_sizes)
        _target_coords = spots.to_positions(pixel_sizes=self.pixel_sizes)
        return cdist(_self_coords, _target_coords)

    def dist_chromosome(self):
        pass



default_im_size=np.array([50,2048,2048])
default_pixel_sizes=np.array([250,108,108])
default_channels = ['750','647','561','488','405']
default_ref_channel = '647'
default_dapi_channel = '405'
default_num_buffer_frames = 0
default_num_empty_frames = 0
default_seed_th = 1000
from ..io_tools.load import split_im_by_channels,load_correction_profile
from ..correction_tools.filter import gaussian_high_pass_filter
from ..correction_tools.alignment import align_image
from ..visual_tools import DaxReader
import re
import os
import xml.etree.ElementTree as ET
import time
import h5py

class DaxProcesser():
    """Major image processing class for 3D image in DNA-MERFISH,
    including two major parts:
        1. image corrections
        2. spot finding
    """
    def __init__(self, 
                 ImageFilename, 
                 CorrectionFolder=None,
                 Channels=None,
                 DriftChannel=None,
                 DapiChannel=None,
                 verbose=True,
                 ):
        """Initialize DaxProcessing class"""
        if isinstance(ImageFilename, str) \
            and os.path.isfile(ImageFilename)\
            and ImageFilename.split(os.extsep)[-1] == 'dax':
            self.filename = ImageFilename
        elif not isinstance(ImageFilename, str):
            raise TypeError(f"Wrong input type ({type(ImageFilename)}) for ImageFilename.")
        elif ImageFilename.split(os.extsep)[-1] != 'dax':
            raise TypeError(f"Wrong input file extension, should be .dax")
        else:
            raise OSError(f"image file: {ImageFilename} doesn't exist, exit.")
        if verbose:
            print(f"Initialize DaxProcesser for file:{ImageFilename}")
        # other files together with dax
        self.inf_filename = self.filename.replace('.dax', '.inf') # info file
        self.off_filename = self.filename.replace('.dax', '.off') # offset file
        self.power_filename = self.filename.replace('.dax', '.power') # power file
        self.xml_filename = self.filename.replace('.dax', '.xml') # xml file
        # Correction folder
        self.correction_folder = CorrectionFolder
        # Channels
        if Channels is None:
            _loaded_channels = DaxProcesser._FindDaxChannels(self.filename)
            if _loaded_channels is None:
                self.channels = default_channels
            else:
                self.channels = _loaded_channels
        elif isinstance(Channels, list) or isinstance(Channels, np.ndarray):
            self.channels = list(Channels)
        else:
            raise TypeError(f"Wrong input type for Channels")
        if DriftChannel is not None and str(DriftChannel) in self.channels:
            setattr(self, 'drift_channel', str(DriftChannel))
        if DapiChannel is not None and str(DapiChannel) in self.channels:
            setattr(self, 'dapi_channel', str(DapiChannel))
        # Log for whether corrections has been done:
        self.correction_log = {_ch:{} for _ch in self.channels}
        # verbose
        self.verbose=verbose
        
    def _check_existance(self):
        """Check the existance of the full set of Dax file"""
        # return True if all file exists
        return os.path.isfile(self.filename) \
            and os.path.isfile(self.inf_filename) \
            and os.path.isfile(self.off_filename) \
            and os.path.isfile(self.power_filename) \
            and os.path.isfile(self.xml_filename) \

    def _load_image(self, 
                    sel_channels=None,
                    ImSize=None, 
                    NbufferFrame=default_num_buffer_frames,
                    NemptyFrame=default_num_empty_frames,
                    save_attrs=True, overwrite=False,
                    ):
        """Function to load and parse images by channels,
            assuming that for each z-layer, all channels has taken a frame in the same order
        """
        _load_start = time.time()
        # init loaded channels
        if not hasattr(self, 'loaded_channels'):
            setattr(self, 'loaded_channels', [])
        
        # get selected channels
        if sel_channels is None:
            _sel_channels = self.channels
        elif isinstance(sel_channels, list):
            _sel_channels = [str(_ch) for _ch in sel_channels]
        elif isinstance(sel_channels, str) or isinstance(sel_channels, int):
            _sel_channels = [str(sel_channels)]
        else:
            raise ValueError(f"Invalid input for sel_channels")
        # choose loading channels
        _loading_channels = []
        for _ch in sorted(_sel_channels, key=lambda v: self.channels.index(v)):
            if hasattr(self, f"im_{_ch}") and not overwrite:
                continue
            else:
                _loading_channels.append(_ch)
        # get image size
        if ImSize is None:
            self.image_size = DaxProcesser._FindImageSize(self.filename,
                                                 channels=self.channels,
                                                 NbufferFrame=NbufferFrame,
                                                 verbose=self.verbose,
                                                 )
        else:
            self.image_size = np.array(ImSize, dtype=np.int32)
        # Load dax file
        _reader = DaxReader(self.filename, verbose=self.verbose)
        _raw_im = _reader.loadAll()
        # split by channel
        _ims = split_im_by_channels(
            _raw_im, _loading_channels,
            all_channels=self.channels, single_im_size=self.image_size,
            num_buffer_frames=NbufferFrame, num_empty_frames=NemptyFrame,
        )
        if self.verbose:
            print(f"- Loaded images for channels:{_loading_channels} in {time.time()-_load_start:.3f}s.")
        # save attributes
        if save_attrs:
            for _ch, _im in zip(_loading_channels, _ims):
                setattr(self, f"im_{_ch}", _im)
            setattr(self, 'num_buffer_frames', NbufferFrame)
            setattr(self, 'num_empty_frames', NemptyFrame)
            self.loaded_channels.extend(_loading_channels)
            # sort loaded
            self.loaded_channels = [_ch for _ch in sorted(self.loaded_channels, key=lambda v: self.channels.index(v))]
            return
        else:
            return _ims, _loading_channels

    def _corr_bleedthrough(self,
                           correction_channels=None,
                           correction_pf=None, 
                           correction_folder=None,
                           rescale=True,
                           save_attrs=True,
                           overwrite=False,
                           ):
        """Apply bleedthrough correction to remove crosstalks between channels
            by a pre-measured matrix"""
        # find correction channels
        _total_bleedthrough_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels 
            if str(_ch) != getattr(self, 'drift_channel', None) \
                and str(_ch) != getattr(self, 'dapi_channel', None)
            ]
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_bleedthrough', False) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Correct bleedthrough already finished, skip. ")
            return 
        ## if not finished, do process
        if self.verbose:
            print(f"- Start bleedthrough correction for channels:{_correction_channels}.")
        if correction_folder is None:
            correction_folder = self.correction_folder
        if correction_pf is None:
            correction_pf = load_correction_profile(
                'bleedthrough', _correction_channels,
                correction_folder=correction_folder,
                ref_channel=_correction_channels[0],
                all_channels=self.channels,
                im_size=self.image_size,
                verbose=self.verbose,
            )
        _corrected_ims = []
        for _ich, _ch1 in enumerate(_correction_channels):
            # new image is the sum of all intensity contribution from images in corr_channels
            _bleedthrough_start = time.time()
            if getattr(self, f"im_{_ch1}", None) is None:
                if self.verbose:
                    print(f"-- skip bleedthrough correction for channel {_ch1}, image not detected.")
                _corrected_ims.append(None)
                continue
            else:
                _dtype = getattr(self, f"im_{_ch1}", None).dtype
                _min,_max = np.iinfo(_dtype).min, np.iinfo(_dtype).max
                # init image
                _im = np.zeros(self.image_size)
                for _jch, _ch2 in enumerate(_correction_channels):
                    if hasattr(self, f"im_{_ch2}"):
                        _im += getattr(self, f"im_{_ch2}") * correction_pf[_ich, _jch]
                # rescale
                if rescale: # (np.max(_im) > _max or np.min(_im) < _min)
                    _im = (_im - np.min(_im)) / (np.max(_im) - np.min(_im)) * _max + _min
                _im = np.clip(_im, a_min=_min, a_max=_max).astype(_dtype)
                _corrected_ims.append(_im)
                # release RAM
                del(_im)
                # print time
                if self.verbose:
                    print(f"-- corrected bleedthrough for channel {_ch1} in {time.time()-_bleedthrough_start:.3f}s.")
        # after finish, save attr
        if self.verbose:
            print(f"-- finish bleedthrough correction in {time.time()-_total_bleedthrough_start:.3f}s. ")
        if save_attrs:
            for _ch, _im in zip(_correction_channels, _corrected_ims):
                setattr(self, f"im_{_ch}", _im)
            del(_corrected_ims)
            # update log
            for _ch in _correction_channels:
                self.correction_log[_ch]['corr_bleedthrough'] = True
            return 
        else:
            return _corrected_ims, _correction_channels
    # remove hot pixels
    def _corr_hot_pixels_3D(
        self, 
        correction_channels=None,
        hot_pixel_th:float=0.5, 
        hot_pixel_num_th:float=4, 
        save_attrs:bool=True,
        )->None:
        """Remove hot pixel by interpolation"""
        _total_hot_pixel_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels]
        if self.verbose:
            print(f"- Correct hot_pixel for channels: {_correction_channels}")
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_hot_pixel', False) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Correct hot_pixel already finished, skip. ")
            return 
        ## update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for corr_hot_pixel.")
        _corrected_ims = []
        # apply correction
        for _ch in _correction_channels:
            _hot_pixel_time = time.time()
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip hot_pixel correction for channel {_ch}, image not detected.")
                if not save_attrs:
                    _corrected_ims.append(None)
                continue
            else:
                from ..correction_tools.filter import Remove_Hot_Pixels
                _dtype = _im.dtype
                _im = Remove_Hot_Pixels(_im, _dtype, 
                    hot_pix_th=hot_pixel_th,
                    hot_th=hot_pixel_num_th,
                )
                # save attr
                if save_attrs:
                    setattr(self, f"im_{_ch}", _im.astype(_dtype), )
                    # update log
                    self.correction_log[_ch]['corr_hot_pixel'] = True
                else:
                    _corrected_ims.append(_im)
                # release RAM
                del(_im)
                # print time
                if self.verbose:
                    print(f"-- corrected hot_pixel for channel {_ch} in {time.time()-_hot_pixel_time:.3f}s.")
        # finish
        if self.verbose:
            print(f"- Finished hot_pixel correction in {time.time()-_total_hot_pixel_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _corrected_ims, _correction_channels
    # illumination correction                    
    def _corr_illumination(self, 
                           correction_channels=None,
                           correction_pf=None, 
                           correction_folder=None,
                           rescale=True,
                           save_attrs=True,
                           overwrite=False,
                           ):
        """Apply illumination correction to flatten field-of-view illumination
            by a pre-measured 2D-array"""
        _total_illumination_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels]
        if self.verbose:
            print(f"- Correct illumination for channels: {_correction_channels}")
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_illumination', False) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Correct illumination already finished, skip. ")
            return 
        ## update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for corr_illumination.")
        # load profile
        if correction_folder is None:
            correction_folder = self.correction_folder
        if correction_pf is None:
            correction_pf = load_correction_profile(
                'illumination', _correction_channels,
                correction_folder=correction_folder,
                ref_channel=_correction_channels[0],
                all_channels=self.channels,
                im_size=self.image_size,
                verbose=self.verbose,
            )
        _corrected_ims = []
        # apply correction
        for _ch in _correction_channels:
            _illumination_time = time.time()
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip illumination correction for channel {_ch}, image not detected.")
                if not save_attrs:
                    _corrected_ims.append(None)
                continue
            else:
                _dtype = _im.dtype
                _min,_max = np.iinfo(_dtype).min, np.iinfo(_dtype).max
                # apply corr
                _im = _im.astype(np.float32) / correction_pf[_ch][np.newaxis,:]
                if rescale: # (np.max(_im) > _max or np.min(_im) < _min)
                    _im = (_im - np.min(_im)) / (np.max(_im) - np.min(_im)) * _max + _min
                _im = np.clip(_im, a_min=_min, a_max=_max)
                # save attr
                if save_attrs:
                    setattr(self, f"im_{_ch}", _im.astype(_dtype), )
                    # update log
                    self.correction_log[_ch]['corr_illumination'] = True
                else:
                    _corrected_ims.append(_im)
                # release RAM
                del(_im)
                # print time
                if self.verbose:
                    print(f"-- corrected illumination for channel {_ch} in {time.time()-_illumination_time:.3f}s.")
        # finish
        if self.verbose:
            print(f"- Finished illumination correction in {time.time()-_total_illumination_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _corrected_ims, _correction_channels

    def _corr_chromatic_functions(self, 
        correction_channels=None,
        correction_pf=None, 
        correction_folder=None,
        ref_channel=default_ref_channel,
        save_attrs=True,
        overwrite=False,
        ):
        """Generate chromatic_abbrevation functions for each channel"""
        from ..correction_tools.chromatic import generate_chromatic_function
        _total_chromatic_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels 
            if str(_ch) != getattr(self, 'drift_channel', None) and str(_ch) != getattr(self, 'dapi_channel', None)]
        if self.verbose:
            print(f"- Generate corr_chromatic_functions for channels: {_correction_channels}")
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_chromatic', False) or self.correction_log[_ch].get('corr_chromatic_function', False)
            for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Correct chromatic function already finished, skip. ")
            return 
        # update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"- Keep channels: {_correction_channels} for corr_chromatic_functions.")
        ## if not finished, do process
        if self.verbose:
            print(f"- Start generating chromatic correction for channels:{_correction_channels}.")
        if correction_folder is None:
            correction_folder = self.correction_folder
        if correction_pf is None:
            correction_pf = load_correction_profile(
                'chromatic_constants', _correction_channels,
                correction_folder=correction_folder,
                all_channels=self.channels,
                ref_channel=ref_channel,
                im_size=self.image_size,
                verbose=self.verbose,
            )
        ## loop through channels to generate functions
        # init corrected_funcs
        _image_size = getattr(self, 'image_size')
        _drift = getattr(self, 'drift', np.zeros(len(_image_size)))
        _corrected_funcs = []
        # apply
        for _ch in _correction_channels:
            if self.verbose:
                _chromatic_time = time.time()
                print(f"-- generate chromatic_shift_function for channel: {_ch}", end=' ')
            _func = generate_chromatic_function(correction_pf[_ch], _drift)
            if save_attrs:
                setattr(self, f"chromatic_func_{_ch}", _func)
            else:
                _corrected_funcs.append(_func)
            if self.verbose:
                print(f"in {time.time()-_chromatic_time:.3f}s")
        if self.verbose:
            print(f"-- finish generating chromatic functions in {time.time()-_total_chromatic_start:.3f}s")
        if save_attrs:
            # update log
            for _ch in _correction_channels:
                self.correction_log[_ch]['corr_chromatic_function'] = True
            return 
        else:
            return _corrected_funcs
        
    def _calculate_drift(
        self, 
        RefImage, DriftChannel='488', 
        precise_align=True,
        use_autocorr=True, 
        drift_kwargs={},
        save_attr=True, 
        save_ref_im=False,
        overwrite=False,
        ):
        """Calculate drift given reference image"""
        if hasattr(self, 'drift') and hasattr(self, 'drift_flag') and not overwrite:
            if self.verbose:
                print(f"- Drift already calculated, skip.")
            return self.drift, self.drift_flag
        # Load drift image
        if DriftChannel is None and hasattr(self, 'drift_channel'):
            DriftChannel = getattr(self, 'drift_channel')
        elif DriftChannel is not None:
            DriftChannel = str(DriftChannel)
        else:
            raise ValueError(f"Wrong input value for DriftChannel: {DriftChannel}")
        # get _DriftImage
        if DriftChannel in self.channels and hasattr(self, f"im_{DriftChannel}"):
            _DriftImage = getattr(self, f"im_{DriftChannel}")
        elif DriftChannel in self.channels and not hasattr(self, f"im_{DriftChannel}"):
            _DriftImage = self._load_image(sel_channels=[DriftChannel], 
                                           ImSize=self.image_size,
                                           NbufferFrame=self.num_buffer_frames, NemptyFrame=self.num_empty_frames,
                                           save_attrs=False)[0][0]
            
        else:
            raise AttributeError(f"DriftChannel:{DriftChannel} image doesn't exist, exit.")
        if self.verbose:
            print(f"+ Calculate drift with drift_channel: {DriftChannel}")
        if isinstance(RefImage, str) and os.path.isfile(RefImage):
            # if come from the same file, skip
            if RefImage == self.filename:
                if self.verbose:
                    print(f"-- processing ref_image itself, skip.")
                _drift = np.zeros(len(self.image_size))
                _drift_flag = 0
                if save_attr:
                    # drift channel
                    setattr(self, 'drift_channel', DriftChannel)
                    # ref image
                    if save_ref_im:
                        setattr(self, 'ref_im', getattr(self, f"im_{DriftChannel}"))
                    # drift results
                    setattr(self, 'drift', _drift)
                    setattr(self, 'drift_flag', _drift_flag)
                    return 
                else:
                    return _drift, _drift_flag
            # load RefImage from file and get this image
            _dft_dax_cls = DaxProcesser(RefImage, CorrectionFolder=self.correction_folder,
                                        Channels=None, verbose=self.verbose)
            _dft_dax_cls._load_image(sel_channels=[DriftChannel], ImSize=self.image_size,
                                     NbufferFrame=self.num_buffer_frames, NemptyFrame=self.num_empty_frames)
            RefImage = getattr(_dft_dax_cls, f"im_{DriftChannel}")

        elif isinstance(RefImage, np.ndarray) and (np.array(RefImage.shape)==np.array(_DriftImage.shape)).all():
            # directly add
            if save_ref_im:
                setattr(self, 'ref_im', RefImage)
            pass
        else:
            raise ValueError(f"Wrong input of RefImage, should be either a matched sized image, or a filename")
        # align_image
        if precise_align:
            _drift, _drift_flag = align_image(
                _DriftImage,
                RefImage, 
                use_autocorr=use_autocorr, drift_channel=DriftChannel,
                verbose=self.verbose, **drift_kwargs,
            )
        else:
            if self.verbose:
                print("-- use auto correlation to calculate rough drift.")
            # calculate drift with autocorr
            from skimage.registration import phase_cross_correlation
            _start_time = time.time()
            _drift, _error, _phasediff = phase_cross_correlation(
                RefImage, _DriftImage, 
                )
            _drift_flag = 2
            if self.verbose:
                print(f"-- calculate rough drift: {_drift} in {time.time()-_start_time:.3f}s. ")
        if save_attr:
            # drift channel
            setattr(self, 'drift_channel', DriftChannel)
            # ref image
            if save_ref_im:
                setattr(self, 'ref_im', RefImage)
            # drift results
            setattr(self, 'drift', _drift)
            setattr(self, 'drift_flag', _drift_flag)
        return _drift, _drift_flag
    # warp image
    def _warp_image(self,
                    drift=None,
                    correction_channels=None,
                    corr_chromatic=True, chromatic_pf=None,
                    correction_folder=None,
                    ref_channel=default_ref_channel,
                    save_attrs=True, overwrite=False,
                    ):
        """Warp image in 3D, this step must give a drift"""
        _total_warp_start = time.time()
        # get drift
        if drift is not None:
            _drift = np.array(drift)
        elif hasattr(self, 'drift'):
            _drift = getattr(self, 'drift')
        else:
            _drift = np.zeros(len(self.image_size))
            warnings.warn(f"drift not given to warp image. ")
        # get channels
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels]
        _chromatic_channels = [_ch for _ch in _correction_channels 
                            if _ch != getattr(self, 'drift_channel', None) and _ch != getattr(self, 'dapi_channel', None)]
        # Log
        _ch_2_finish_warp = {_ch: self.correction_log[_ch].get('corr_drift', False) or  not _drift.any() for _ch in _correction_channels}
        _ch_2_finish_chromatic = {_ch: self.correction_log[_ch].get('corr_chromatic', False) for _ch in _chromatic_channels}
        _logs = [_ch_2_finish_warp.get(_ch) and _ch_2_finish_chromatic.get(_ch, True) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"- Warp drift and chromatic already finished, skip. ")
            return 
        # start warpping
        if self.verbose:
            print(f"- Start warpping images channels:{_correction_channels}.")
        if correction_folder is None:
            correction_folder = self.correction_folder
        # load chromatic warp
        if corr_chromatic and chromatic_pf is None:
            chromatic_pf = load_correction_profile(
                'chromatic', _chromatic_channels,
                correction_folder=correction_folder,
                all_channels=self.channels,
                ref_channel=ref_channel,
                im_size=self.image_size,
                verbose=self.verbose,
            )
        # init corrected_ims
        _corrected_ims = []
        # do warpping
        for _ch in _correction_channels:
            _chromatic_time = time.time()
            # get flag for this channel
            _finish_warp = _ch_2_finish_warp.get(_ch)
            _finish_chromatic = _ch_2_finish_chromatic.get(_ch, True)
            print(_ch, _finish_warp, _finish_chromatic)
            # skip if not required
            if _finish_warp and (_finish_chromatic or not corr_chromatic):
                if self.verbose:
                    print(f"-- skip warpping image for channel {_ch}, no drift or chromatic required.")
                continue
            # get image
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip warpping image for channel {_ch}, image not detected.")
                continue
            # 1. get coordiates to be mapped
            _coords = np.meshgrid(np.arange(self.image_size[0]), 
                                np.arange(self.image_size[1]), 
                                np.arange(self.image_size[2]), 
                                )
            # transpose is necessary  
            _coords = np.stack(_coords).transpose((0, 2, 1, 3)) 
            _note = f"-- warp image"
            # 2. apply drift if necessary
            if not _finish_warp:
                _coords = _coords - _drift[:, np.newaxis,np.newaxis,np.newaxis]
                _note += f' with drift:{_drift}'
                # update flag
                self.correction_log[_ch]['corr_drift'] = True
            # 3. aaply chromatic if necessary
            if not _finish_chromatic and corr_chromatic:
                _note += ' with chromatic abbrevation' 
                if chromatic_pf[_ch] is None and str(_ch) == ref_channel:
                    pass
                else:                 
                    _coords = _coords + chromatic_pf[_ch]
                # update flag
                self.correction_log[_ch]['corr_chromatic'] = True
            # 4. map coordinates
            if self.verbose:
                print(f"{_note} for channel: {_ch}")
            _im = map_coordinates(_im, 
                                _coords.reshape(_coords.shape[0], -1),
                                mode='nearest').astype(_im.dtype)
            _im = _im.reshape(tuple(self.image_size))

            # 5. save
            if save_attrs:
                setattr(self, f"im_{_ch}", _im,)
            else:
                _corrected_ims.append(_im)
            # release RAM
            del(_im)
            # print time
            if self.verbose:
                print(f"-- finish warpping channel {_ch} in {time.time()-_chromatic_time:.3f}s.")
        
        # print time
        if self.verbose:
            print(f"-- finish warpping in {time.time()-_total_warp_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _corrected_ims, _correction_channels
    # Gaussian highpass for high-background images
    def _gaussian_highpass(self,                            
                        correction_channels=None,
                        gaussian_sigma=3,
                        gaussian_truncate=2,
                        save_attrs=True,
                        overwrite=False,
                        ):
        """Function to apply gaussian highpass for selected channels"""
        _total_highpass_start = time.time()
        if correction_channels is None:
            correction_channels = self.loaded_channels
        _correction_channels = [str(_ch) for _ch in correction_channels 
                                if str(_ch) != getattr(self, 'dapi_channel', None)]
        if self.verbose:
            print(f"- Apply Gaussian highpass for channels: {_correction_channels}")
        ## if finished ALL, directly return
        _logs = [self.correction_log[_ch].get('corr_highpass', False) for _ch in _correction_channels]
        if np.array(_logs).all():
            if self.verbose:
                print(f"-- Gaussian_highpass for channel:{_correction_channels} already finished, skip. ")
            return 
        # update _correction_channels based on log
        _correction_channels = [_ch for _ch, _log in zip(_correction_channels, _logs) if not _log ]
        if self.verbose:
            print(f"-- Keep channels: {_correction_channels} for gaussian_highpass.")
        # loop through channels
        _corrected_ims = []
        for _ch in _correction_channels:
            if self.verbose:
                print(f"-- applying gaussian highpass, channel={_ch}, sigma={gaussian_sigma}", end=' ')
                _highpass_time = time.time()
            # get image
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip gaussian_highpass for channel {_ch}, image not detected.")
                if not save_attrs:
                    _corrected_ims.append(None)
                continue
            else:
                _dtype = _im.dtype
                _min,_max = np.iinfo(_dtype).min, np.iinfo(_dtype).max
                # apply gaussian highpass filter
                _im = gaussian_high_pass_filter(_im, gaussian_sigma, gaussian_truncate)
                _im = np.clip(_im, a_min=_min, a_max=_max).astype(_dtype)
                # save attr
                if save_attrs:
                    setattr(self, f"im_{_ch}", _im)
                    # update log
                    self.correction_log[_ch]['corr_highpass'] = True
                else:
                    _corrected_ims.append(_im)
                # release RAM
                del(_im)
                # print time
                if self.verbose:
                    print(f"in {time.time()-_highpass_time:.3f}s")
        # finish
        if self.verbose:
            print(f"- Finished gaussian_highpass filtering in {time.time()-_total_highpass_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _corrected_ims, _correction_channels
            
    # Spot_fitting:
    def _fit_spots(self, fit_channels=None, 
                th_seed=1000, num_spots=None, fitting_kwargs={},
                save_attrs=True, overwrite=False):
        """Fit spots for the entire image"""
        from ..spot_tools.fitting import fit_fov_image
        # total start time
        _total_fit_start = time.time()
        if fit_channels is None:
            fit_channels = self.loaded_channels
        _fit_channels = [str(_ch) for _ch in fit_channels 
                        if str(_ch) != getattr(self, 'drift_channel', None) \
                            and str(_ch) != getattr(self, 'dapi_channel', None)]
        if self.verbose:
            print(f"- Fit spots in channels:{_fit_channels}")
        _fit_logs = [hasattr(self, f'spots_{_ch}') and not overwrite for _ch in _fit_channels]
        ## if finished ALL, directly return
        if np.array(_fit_logs).all():
            if self.verbose:
                print(f"-- Fitting for channel:{_fit_channels} already finished, skip. ")
            return 
        # update _fit_channels based on log
        _fit_channels = [_ch for _ch, _log in zip(_fit_channels, _fit_logs) if not _log ]
        # th_seeds
        if isinstance(th_seed, int) or isinstance(th_seed, float):
            _ch_2_thSeed = {_ch:th_seed for _ch in _fit_channels}
        elif isinstance(th_seed, dict):
            _ch_2_thSeed = {str(_ch):_th for _ch,_th in th_seed.items()}
        if self.verbose:
            print(f"-- Keep channels: {_fit_channels} for fitting.")
        _spots_list = []
        for _ch in _fit_channels:
            if self.verbose:
                print(f"-- fitting channel={_ch}", end=' ')
                _fit_time = time.time()
            # get image
            _im = getattr(self, f"im_{_ch}", None)
            if _im is None:
                if self.verbose:
                    print(f"-- skip fitting for channel {_ch}, image not detected.")
                continue
            # fit
            _th_seed = _ch_2_thSeed.get(_ch, default_seed_th)
            _spots = fit_fov_image(_im, _ch, 
                                th_seed=_th_seed, max_num_seeds=num_spots, 
                                verbose=self.verbose,
                                **fitting_kwargs)
            _cell_ids = np.ones(len(_spots),dtype=np.int32) -1
            if save_attrs:
                setattr(self, f"spots_{_ch}", _spots)
                setattr(self, f"spots_cell_ids_{_ch}", _cell_ids)
            else:
                _spots_list.append(_spots)
        # return
        if self.verbose:
            print(f"-- finish fitting in {time.time()-_total_fit_start:.3f}s.")
        if save_attrs:
            return
        else:
            return _spots_list
    # Spot Fitting 2, by segmentation
    def _fit_spots_by_segmentation(self, channel, seg_label, 
                                   th_seed=500, num_spots=None, fitting_kwargs={},
                                   segment_search_radius=3, 
                                   save_attrs=True, verbose=False):
        """Function to fit spots within each segmentation
        Necessary numbers:
            th_seed: default seeding threshold
            num_spots: number of expected spots (within each segmentation mask)
        """
        from ..segmentation_tools.cell import segmentation_mask_2_bounding_box
        from ..spot_tools.fitting import fit_fov_image
        from .partition_spots import Spots_Partition
        
        # get drift
        _drift = getattr(self, 'drift', np.zeros(len(self.image_size)))
        # get cell_id
        if self.verbose:
            print(f"- Start fitting spots in each segmentation")
        _cell_ids = np.unique(seg_label)
        _cell_ids = _cell_ids[_cell_ids>0]

        _all_spots, _all_cell_ids = [], []
        for _cell_id in tqdm(_cell_ids):
            _cell_mask = (seg_label==_cell_id)
            _crop = segmentation_mask_2_bounding_box(_cell_mask, 3)

            _local_mask = _cell_mask[_crop.to_slices()]
            _drift_crop = _crop.translate_drift(drift=_drift)
            _drift_local_im = getattr(self, f'im_{channel}')[_drift_crop.to_slices()]
            # fit
            _spots = fit_fov_image(_drift_local_im, str(channel), 
                                th_seed=th_seed, max_num_seeds=num_spots, 
                                verbose=verbose,
                                **fitting_kwargs)
            if len(_spots) > 0:
                # adjust to absolute coordinate per fov
                _spots = Spots3D(_spots)
                _spots[:,_spots.coordinate_indices] = _spots[:,_spots.coordinate_indices] + _drift_crop.array[:,0]
                # keep spots within mask
                _kept_flg = Spots_Partition.spots_to_labels(_cell_mask, _spots, 
                                                            search_radius=segment_search_radius,verbose=False)
                _spots = _spots[_kept_flg>0]
                # append
                if len(_spots) > 0:
                    _all_spots.append(_spots)
                    _all_cell_ids.append(np.ones(len(_spots), dtype=np.int32)*_cell_id)
        # concatenate
        if len(_all_spots) > 0:
            _all_spots = np.concatenate(_all_spots)
            _all_cell_ids = np.concatenate(_all_cell_ids)
        else:
            _all_spots = np.array([])
            _all_cell_ids = np.array([])
            print(f"No spots detected.")
        # save attr
        if save_attrs:
            setattr(self, f"spots_{channel}", _all_spots)
            setattr(self, f"spots_cell_ids_{channel}", _all_cell_ids)
            return
        return _all_spots, _all_cell_ids

    # Saving:
    def _save_to_hdf5(self):
        pass
    def _save_to_npy(self, save_channels, save_folder=None, save_basenames=None):
        if save_folder is None:
            pass

        
    # Loading:
    def _load_from_hdf5(self):
        pass
    @staticmethod
    def _FindDaxChannels(dax_filename,
                         verbose=True,
                         ):
        """Find channels"""
        _xml_filename = dax_filename.replace('.dax', '.xml') # xml file
        try:
            _hal_info = ET.parse(_xml_filename).getroot()
            _shutter_filename = _hal_info.findall('illumination/shutters')[0].text
            _shutter_channels = os.path.basename(_shutter_filename).split(os.extsep)[0].split('_')
            # select all digit names which are channels
            _true_channels = [_ch for _ch in _shutter_channels
                              if len(re.findall(r'^[0-9]+$', _ch))]
            if verbose:
                print(f"-- all used channels: {_true_channels}")
            return _true_channels
        except:
            return None
    @staticmethod
    def _FindGlobalPosition(dax_filename:str,
                            verbose=True) -> np.ndarray:
        """Function to find global coordinates in micron"""
        _xml_filename = dax_filename.replace('.dax', '.xml') # xml file
        try:
            _hal_info = ET.parse(_xml_filename).getroot()
            _position_micron = np.array(_hal_info.findall('acquisition/stage_position')[0].text.split(','), dtype=np.float64)
            return _position_micron
        except:
            raise ValueError(f"Positions not properly parsed")


    @staticmethod
    def _LoadInfFile(inf_filename):
        with open(inf_filename, 'r') as _info_hd:
            _infos = _info_hd.readlines()
        _info_dict = {}
        for _line in _infos:
            _line = _line.rstrip()#.replace(' ','')
            _key, _value = _line.split(' = ')
            _info_dict[_key] = _value
        return _info_dict
    @staticmethod
    def _FindImageSize(dax_filename, 
                       channels=None,
                       NbufferFrame=default_num_buffer_frames,
                       verbose=True,
                       ):
        _inf_filename = dax_filename.replace('.dax', '.inf') # info file
        if channels is None:
            channels = DaxProcesser._FindDaxChannels(dax_filename)                
        try:
            _info_dict = DaxProcesser._LoadInfFile(_inf_filename)
            # get image shape
            _dx,_dy = _info_dict['frame dimensions'].split('x')
            _dx,_dy = int(_dx),int(_dy)
            # get number of frames in z
            _n_frame = int(_info_dict['number of frames'])
            _dz = (_n_frame - 2 * NbufferFrame) / len(channels)
            if _dz == int(_dz):
                _dz = int(_dz)
                _image_size = np.array([_dz,_dx,_dy],dtype=np.int32)
                if verbose:
                    print(f"-- single image size: {_image_size}")
                return _image_size
            else:
                raise ValueError("Wrong num_color, should be integer!")
        except:
            return np.array(default_im_size)
    @staticmethod
    def _LoadSegmentation(segmentation_filename,
                          fov_id=None,
                          verbose=True):
        """Function to load segmentation from file"""
        # check existance
        if not isinstance(segmentation_filename, str) or not os.path.isfile(segmentation_filename):
            raise ValueError(f"invalid segmentation_filename: {segmentation_filename}")
        #load
        if verbose:
            print(f"-- load segmentation from: {segmentation_filename}")
        if segmentation_filename.split(os.extsep)[-1] == 'npy':
            _seg_label = np.load(segmentation_filename)
        elif segmentation_filename.split(os.extsep)[-1] == 'pkl':
            _seg_label = pickle.load(open(segmentation_filename, 'rb'))
        elif segmentation_filename.split(os.extsep)[-1] == 'hdf5' or segmentation_filename.split(os.extsep)[-1] == 'h5':
            with h5py.File(segmentation_filename, 'r') as _f:
                if fov_id is None:
                    fov_id = list(_f.keys())[0]
                _seg_label = _f[str(fov_id)]['dna_mask'][:]
        # return
        return _seg_label


def batch_process_image_quick(
    dax_filename, correction_folder,
    sel_channels,
    drift_channel='488', dapi_channel='405',
    corr_hot_pixels=True,
    corr_illumination=True,
    verbose=True,
    ):
    """Function to quickly apply DaxProcesser"""
    # create class
    _cls = DaxProcesser(dax_filename, correction_folder,
        Channels=None, DriftChannel=drift_channel, DapiChannel=dapi_channel, verbose=verbose)
    # load image
    _cls._load_image(sel_channels=sel_channels)
    # correct illumination if applicable
    if corr_hot_pixels:
        _cls._corr_hot_pixels_3D(correction_channels=sel_channels)
    if corr_illumination:
        _cls._corr_illumination(correction_channels=sel_channels)
    # get images
    return [getattr(_cls, f"im_{_ch}") for _ch in sel_channels]