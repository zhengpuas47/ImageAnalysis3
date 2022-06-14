# import common packages
import copy
import numpy as np
import os, glob, sys, time
# import from parental packages
from . import _distance_zxy, _image_size
from ..classes.preprocess import Spots3D
from ..io_tools.parameters import _read_microscope_json

## Change scaling-------------------------------------------------------
# center and PCA transfomr spots in 3d
def normalize_center_spots(spots, distance_zxy=_distance_zxy, 
                           center_zero=True, scale_variance=False,
                           pca_align=True, return_pca=False, scaling=1.):
    """Function to adjust gaussian fitted spots into standardized 3d situation
    Inputs: 
        spots: list of spots that that generated from Fitting_V3, 2d np.ndarray or list of 1d np.ndarray
        distance_zxy: transformation from pixel to nm, np.ndarray of 3 (default: global setting _distance_zxy)
        center: whether center chromosome to 0, bool (default: True)
        scale_variance: whether apply variance scaling so all directions have the same variance, bool (default: False)
        pca_align: whether align spots into PCA space, bool (default: True)
        scaling: scaling factor for the chromosome, float (default: 1., which means scale as pixel)
    Output:
        _spots: same spots after transformation, 2d np.ndarray
    """
    ## check inputs
    # spots
    _spots = np.array(spots).copy() # make copy of spots
    if len(spots.shape) != 2:
        raise ValueError(f"Input spots should be 2d-array like structure, but shape:{spots.shape} is given!")
    # case 1, already converted to zxy format
    if np.shape(_spots)[1] == 3:
        _coords = _spots
        _stds = np.ones(np.shape(_coords))
    # case 2, already converted to hzxy format (intensity with coordinates)
    elif np.shape(_spots)[1] == 4:
        _coords = _spots[:,-3:]
        _stds = np.ones(np.shape(_coords))    
    # case 3, full spots info, extract and adjust scaling according to pixel size
    else:
        #distance_zxy
        distance_zxy = np.array(distance_zxy)[:3]
        _adjust_scaling = distance_zxy / np.min(distance_zxy)
        # start convert distances
        _coords = _spots[:,1:4] * _adjust_scaling[np.newaxis,:] 
        _stds = _spots[:,5:8] * _adjust_scaling[np.newaxis,:]

    # centering
    _coord_center = np.nanmean(_coords, axis=0)
    if center_zero:
        #print('centering')
        _coords = _coords - _coord_center
        _coord_center = np.zeros(np.shape(_coords)[1])
    # normalize total variance to 1
    if scale_variance:
        _total_scale = np.sqrt(np.nanvar(_coords,axis=0).sum())    
        _coords = _coords / _total_scale * scaling
        _stds = _stds / _total_scale * scaling
    else:
        _coords = _coords * scaling
        _stds = _stds * scaling

    # pca align
    if pca_align:
        if 'PCA' not in locals():
            from sklearn.decomposition import PCA
        # extract value and indices for valid spots
        _clean_coords = np.array([_c-_coord_center for _c in _coords if not np.isnan(_c).any()]) # here needs to reduce center to zero for PCA
        _clean_inds = np.array([_i for _i,_c in enumerate(_coords) if not np.isnan(_c).any()],dtype=np.int)
        # do PCA
        _model = PCA(3)
        _model.fit(_clean_coords)
        _trans_coords = _model.fit_transform(_clean_coords)
        _coords[_clean_inds] = _trans_coords + _coord_center # add center back if necessary
        #print(_model.explained_variance_ratio_)
    else:
        _model = None 
    # return
    if np.shape(_spots)[1] == 3:
        _spots = _coords
    elif np.shape(_spots)[1] == 4:
        _spots[:,-3:] = _coords
    else:
        # save then back to spots
        _spots[:,1:4] = _coords
        _spots[:,5:8] = _stds
    
    if return_pca:
        return _spots, _model
    else:
        return _spots 

## translate spots with microscope.json

def MicroscopeTranslate_Spots(
    fov_spots:Spots3D,
    microscope_param_file:str,
    image_size:np.ndarray=_image_size,
    )->Spots3D:
    """Translate spots given microscope"""
    _fov_spots = copy.copy(fov_spots)
    # load microscope.json
    microscope_param = _read_microscope_json(microscope_param_file)
    _coords = _fov_spots.to_coords()
    # transpose
    if microscope_param.get('transpose', False):
        _coords = _coords[:, np.array([0,2,1])]
    # flip_x
    if microscope_param.get('flip_horizontal', False):
        _coords[:,2] = -1 * (_coords[:,2] - image_size[2]/2) + image_size[2]/2
    # flip y
    if microscope_param.get('flip_vertical', False):
        _coords[:,1] = -1 * (_coords[:,1] - image_size[1]/2) + image_size[1]/2
    # update coordinates
    _fov_spots[:, _fov_spots.coordinate_indices] = _coords
    return _fov_spots
        

## Affine Warpping------------------------------------------------------- 

def translate_spots(spots, rotation_mat=None, drift=None, 
                    single_im_size=_image_size):
    _spots = np.array(spots, dtype=np.float)
    _single_im_size = np.array(single_im_size, dtype=np.float)
    _rot_center = _single_im_size / 2
    if len(np.shape(_spots)) == 1:
        _spots = _spots[np.newaxis,:]
    if np.shape(_spots)[1] == 3:
        _coords = _spots
    elif np.shape(_spots)[1] == 4:
        _coords = _spots[:,1:]
    else:
        _coords = _spots[:,1:4]
    
    if rotation_mat is None:
        rotation_mat = np.diag([1,1])
    else:
        rotation_mat = np.array(rotation_mat)
    # rotation
    _coords[:,1:] = np.dot(_coords[:,1:]-_rot_center[1:3], 
                           np.transpose(rotation_mat)) \
                    + _rot_center[1:3]
        
    if drift is None:
        drift = np.zeros(3)
    else:
        drift = np.array(drift)
    # drift
    _coords += -drift[:3]
    
    return _coords