from warnings import WarningMessage
import numpy as np
from numpy.lib.npyio import save
import cv2
import os,sys,time
import skimage 
import math
import h5py
import json
import copy 
from scipy.ndimage import grey_dilation

# internal function
from ..figure_tools.plot_segmentation import plot_segmentation
from ..io_tools.parameters import _read_microscope_json

default_cellpose_kwargs = {
    'anisotropy': 1,
    'diameter': 60,
    'min_size': 200,
    'stitch_threshold': 0.1,
    'do_3D':True,
}
default_alignment_params = {
    'dialation_size':4,
}
default_pixel_sizes = [250,108,108]
default_Zcoords = np.arange(13)
default_dna_Zcoords = np.round(np.arange(0,12.5,0.25),2)

class Cellpose_Segmentation_Psedu3D:
    """"""
    def __init__(self, _im, data_type='DAPI', 
                 save_filename=None, verbose=True):
        """"""
        # inherit from superclass
        super().__init__()
    
        # save images
        self.raw_im = _im
        self.data_type = data_type
        self.allowed_types = ['DAPI', 'polyT']
        if self.data_type not in self.allowed_types:
            raise ValueError(f"input datatype {self.data_type} not in {self.allowed_types}")
        # save 
        self.save_filename = save_filename
        self.verbose = verbose

    def run(self):
        """"""
        _lys, _sel_ids = Cellpose_Segmentation_Psedu3D.pick_Z_stacks(self.raw_im)
        
        _mask = Cellpose_Segmentation_Psedu3D.run_segmentation(_lys)
        
        _clean_mask = Cellpose_Segmentation_Psedu3D.merge_3d_masks(_mask)
        
        _z = Cellpose_Segmentation_Psedu3D.convert_layer_list_to_um(_sel_ids)
        _full_mask = interploate_z_masks(_clean_mask, _z)
        
        return _full_mask
        
    @staticmethod
    def pick_Z_stacks(im:np.ndarray, 
                      num_layer_project:int=5,
                      num_layer_overlap:int=1,
                      projection_method:'function'=np.mean,
                      verbose=True,
                      ):
        
        _im = im.copy()
        # projection on z
        _sel_layers = []
        for _i, _ly in enumerate(_im):
            if _i < num_layer_project-1:
                continue
            if len(_sel_layers) == 0 or min(_sel_layers[-1][-1*num_layer_overlap-1:]) + num_layer_project <= _i:
                _sel_layers.append(np.arange(_i-num_layer_project+1, _i+1))
                
        # generate max projections
        _max_proj_layers = np.array([projection_method(_im[np.array(_lys)],axis=0) for _lys in _sel_layers])
        
        if verbose:
            print(f"- {len(_max_proj_layers)} layers selected with {projection_method} projection.")
        return _max_proj_layers, _sel_layers
    
    @staticmethod
    def run_segmentation(_projected_im, 
                          model_type='nuclei', 
                          use_gpu=True, 
                          diameter=60, min_size=10, 
                          cellprob_threshold=0.5, stitch_threshold=0.2,
                          flow_threshold=1.0,
                          verbose=True,
                          ):
        from cellpose import models
        # segmentation
        seg_model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        masks, _, _, _ = seg_model.eval(
            np.stack([_projected_im, _projected_im], axis=3),
            diameter=diameter, 
            channels=[0,0], 
            min_size=min_size,
            cellprob_threshold=cellprob_threshold, # -6 to 6, positively correlate with number of masks
            stitch_threshold=stitch_threshold,
            flow_threshold=flow_threshold,
            do_3D=False)
        # clear ram
        del(seg_model)
        
        return masks
    
    @staticmethod
    def merge_3d_masks(masks, overlap_th=0.9, verbose=True):
        
        import time
        # copy masks
        _masks = np.array(masks).copy()
        all_mask_ids = np.unique(_masks)
        all_mask_ids = all_mask_ids[all_mask_ids>0]

        xy_projections = [(_masks==_i).any(0) for _i in all_mask_ids]

        kept_masks = np.zeros(np.shape(_masks), dtype=np.uint16)
        kept_ids = []

        # intialize
        if verbose:
            print(f"- start merging 3d masks")
            _start_time = time.time()
        unprocessed_ids = list(all_mask_ids)

        while len(unprocessed_ids) > 0:
            # default: kept this cell
            _kept_flag = True
            # extract i
            _i = unprocessed_ids.pop(0)
            _i_msk = xy_projections[list(all_mask_ids).index(_i)]

            # calculate j percentage to see whether merge this into _j
            for _j in unprocessed_ids:
                # extract j
                _j_msk = xy_projections[list(all_mask_ids).index(_j)]

                # compare these two masks
                _i_percent = np.sum(_i_msk*_j_msk) / np.sum(_i_msk)
                _j_percent = np.sum(_i_msk*_j_msk) / np.sum(_j_msk)
                if _i_percent > 0 or _j_percent > 0:
                    if verbose:
                        print(f"-- overlap found for cell:{_i} to {_j}, {_i_percent:.4f}, {_j_percent:.4f}")

                # remove i, merge into j
                if _i_percent > overlap_th:
                    _kept_flag = False
                    # update mask, i already removed by continue
                    _masks[_masks==_i] = _j
                    xy_projections[list(all_mask_ids).index(_j)] = (_masks==_j).any(0)
                    if verbose:
                        print(f"--- skip {_i}")
                    break
                    
                # remove j, merge into i
                elif _j_percent > overlap_th:
                    _kept_flag = False
                    # remove j
                    unprocessed_ids.pop(unprocessed_ids.index(_j))
                    # update mask
                    _masks[_masks==_j] = _i
                    xy_projections[list(all_mask_ids).index(_i)] = (_masks==_i).any(0)
                    # redo i
                    unprocessed_ids = [_i] + unprocessed_ids
                    if verbose:
                        print(f"--- redo {_i}")
                    break

            # save this mask if there's no overlap
            if _kept_flag:
                kept_masks[_masks==_i] = np.max(np.unique(kept_masks))+1
                kept_ids.append(_i)
        if verbose:
            print(f"- {np.max(kept_masks)} labels kept.")
            print(f"- finish in {time.time()-_start_time:.2f}s. ")
            
        return kept_masks
    
    @staticmethod
    def convert_layer_list_to_um(layer_lists:list, 
                                 step_sizes:float=0.2, 
                                 select_method:'function'=np.median):
        return step_sizes * np.array([select_method(_lys) for _lys in layer_lists])
    

class Cellpose_Segmentation_3D():
    """Do 3D cellpose segmentation to DAPI image, and additionally watershed on polyT iamge given the DAPI seeds
    Minimal usage:
    seg_cls = Cellpose_Segmentation_3D(dapi_im, polyt_im, pixel_size, save_filename=filename) # create class
    labels = seg_cls.run() # run segmentation 
    seg_cls.save() # save segmentation results
    """
    def __init__(self, dapi_im, polyt_im=None,
                 pixel_sizes=default_pixel_sizes,
                 cellpose_kwargs={},
                 watershed_beta=1, 
                 save_filename=None, 
                 load_from_savefile=True,
                 verbose=True):
        """Create CellposeSegmentation3D class"""
        # inherit from superclass
        #super().__init__()
        # save images
        self.dapi_im = dapi_im
        self.polyt_im = polyt_im
        # parameters
        self.pixel_sizes = pixel_sizes
        self.cellpose_kwargs = {_k:_v for _k,_v in default_cellpose_kwargs.items()}
        self.cellpose_kwargs.update(cellpose_kwargs)
        self.watershed_beta = watershed_beta
        # save info
        self.save_filename = save_filename
        self.verbose = verbose
        # load 
        if load_from_savefile and save_filename is not None and os.path.exists(save_filename):
            self.load()
    def run(self, model_type='nuclei', use_gpu=True, overwrite=False,):
        """Composite segmentation with reshaped image"""
        if hasattr(self, 'segmentation_masks') and not overwrite:
            if self.verbose:
                print(f"-- segmentation_masks already exist, skip.")
            return self.segmentation_masks
        else:
            # 
            _resized_shape = self.generate_resize_shape(self.dapi_im.shape, self.pixel_sizes)
            #
            _resized_dapi_im = self.reshape_raw_images(self.dapi_im, _resized_shape)
            if hasattr(self, 'polyt_im') and getattr(self, 'polyt_im') is not None:
                _resized_polyt_im = self.reshape_raw_images(self.polyt_im, _resized_shape)
            else:
                _resized_polyt_im = None
            #
            _resized_masks = self.run_segmentation(_resized_dapi_im, _resized_polyt_im,
                                model_type=model_type,
                                use_gpu=use_gpu,
                                cellpose_kwargs=self.cellpose_kwargs)
            # revert mask size
            _masks = self.reshape_masks(_resized_masks, np.shape(self.dapi_im))
            # watershed
            if hasattr(self, 'polyt_im') and getattr(self, 'polyt_im') is not None:
                _extended_masks = self.watershed_with_mask(self.polyt_im, _masks, self.watershed_beta)
            else:
                _extended_masks = self.watershed_with_mask(self.dapi_im, _masks, self.watershed_beta)
            # add to attributes
            setattr(self, 'segmentation_masks', _extended_masks)
            return _extended_masks
    def save(self, save_filename=None, overwrite=False):
        # decide save_filename
        if save_filename is None and self.save_filename is None:
            WarningMessage(f"save_filename not given.")
        elif save_filename is not None:
            _save_filename = save_filename
        else:
            _save_filename = self.save_filename
        # save
        if not os.path.exists(_save_filename) or overwrite:
            # save
            if self.verbose:
                print(f"-- saving mask into file: {_save_filename}")
            np.save(_save_filename.split(os.path.extsep+_save_filename.split(os.path.extsep)[-1])[0], 
                    self.segmentation_masks)
        else:
            if self.verbose:
                print(f"-- save_file:{_save_filename} already exists, skip. ")
    def load(self, save_filename=None, overwrite=False):
        # decide save_filename
        if save_filename is None and self.save_filename is None:
            WarningMessage(f"save_filename not given.")
        elif save_filename is not None:
            _save_filename = save_filename
        else:
            _save_filename = self.save_filename
        # load
        if not hasattr(self, 'segmentation_masks') or overwrite:
            if os.path.exists(_save_filename):
                if self.verbose:
                    print(f"-- loading mask from file: {_save_filename}")
                self.segmentation_masks = np.load(_save_filename)
            else:
                if self.verbose:
                    print(f"-- file: {_save_filename} doesn't exist, skip. ")
        else:
            if self.verbose:
                print(f"-- segmentation_masks already exists, skip. ")
    def clear(self):
        if self.verbose:
            print(f"-- removing segmentation_masks from class")
        if hasattr(self, 'segmentation_masks'):
            delattr(self, 'segmentation_masks')
    
    @staticmethod    
    def generate_resize_shape(image_shape, pixel_sizes):
        resize_shape = np.floor(np.array(image_shape)[1:] * np.array(pixel_sizes)[1:] \
                                / np.array(pixel_sizes)[0]).astype(np.int32)
        return resize_shape

    @staticmethod
    def reshape_raw_images(raw_im, 
                           resize_shape,
                           ):
        """Reshape raw image into smaller image to fit-in GPU"""
        _reshaped_im = np.array([cv2.resize(_lr, tuple(resize_shape[-2:]), 
                                 interpolation=cv2.INTER_AREA) for _lr in raw_im])
        return _reshaped_im

    @staticmethod
    def reshape_masks(masks, 
                      resize_shape,
                      ):
        """Reshape raw image into smaller image to fit-in GPU"""
        _reshaped_masks = np.array([cv2.resize(_lr, tuple(resize_shape[-2:]), 
                                    interpolation=cv2.INTER_NEAREST) for _lr in masks])
        return _reshaped_masks

    @staticmethod
    def run_segmentation(small_dapi_im, 
                         small_polyt_im=None, 
                         model_type='nuclei', 
                         use_gpu=True, 
                         cellpose_kwargs={},
                         verbose=True,
                         ):
        from cellpose import models
        import torch
        # check inputs
        _start_time = time.time()
        if small_polyt_im is None:
            small_polyt_im = np.zeros(np.shape(small_dapi_im), dtype=small_dapi_im.dtype)
        # create model
        seg_model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        # parameters
        _cellpose_kwargs = {_k:_v for _k,_v in default_cellpose_kwargs.items()}
        _cellpose_kwargs.update(cellpose_kwargs)

        # run segmentation
        masks, _, _, _ = seg_model.eval(
            np.stack([small_polyt_im, small_dapi_im], axis=3),
            channels=[0,0], 
            resample=True,
            **_cellpose_kwargs)
        # clear ram
        del(seg_model)
        torch.cuda.empty_cache()
        # change background into -1
        new_masks = masks.copy().astype(np.int16)
        new_masks[new_masks==0] = -1
        # return
        if verbose:
            print(f"-- finish segmentation in {time.time()-_start_time:.3f}s. ")
        return new_masks
    
    @staticmethod
    def watershed_with_mask(target_im, seed_labels, beta=1.):
        from skimage.segmentation import random_walker
        _extended_masks = random_walker(target_im, seed_labels, beta=beta, tol=0.001, )
        return _extended_masks


class Align_Segmentation():
    """
    Align segmentation from remounted RNA-DNA sample
    """
    def __init__(self, 
        rna_feature_file:str, 
        rna_dapi_file:str, 
        dna_save_file:str,
        rna_microscope_file:str,
        dna_microscope_file:str,
        rotation_mat:np.ndarray, #
        parameters:dict={},
        overwrite:bool=False,
        debug:bool=False,
        verbose:bool=True,
        ):
        self.rna_feature_file = rna_feature_file
        self.rna_dapi_file = rna_dapi_file
        self.dna_save_file = dna_save_file
        self.rna_microscope_file = rna_microscope_file
        self.dna_microscope_file = dna_microscope_file
        self.rotation_mat = rotation_mat
        # params
        self.parameters = {_k:_v for _k,_v in default_alignment_params.items()}
        self.parameters.update(parameters)
        self.overwrite = overwrite
        self.debug = debug
        self.verbose = verbose

    @staticmethod
    def _load_rna_feature(rna_feature_file:str, _z_coords=default_Zcoords):
        """Load RNA feature from my MERLIN output"""
        _fovcell_2_uid = {}
        with h5py.File(rna_feature_file, 'r') as _f:
            _label_group = _f['labeldata']
            rna_mask = _label_group['label3D'][:] # read
            #rna_mask = Align_Segmentation._correct_image3D_by_microscope_param(rna_mask, microscpe_params) # transpose and flip
            if np.max(rna_mask) <= 0:
                print(f'No cell found in feature file: {rna_feature_file}')
                return rna_mask, _fovcell_2_uid
            else:
                # load feature info
                _feature_group = _f['featuredata']
                for _cell_uid in _feature_group.keys():
                    _cell_group = _feature_group[_cell_uid]
                    _z_coords = _cell_group['z_coordinates'][:]
                    _fovcell_2_uid[(_cell_group.attrs['fov'], _cell_group.attrs['label'])] = _cell_uid
        return rna_mask, _z_coords, _fovcell_2_uid

    @staticmethod
    def _load_dna_info(dna_save_file:str, microscpe_params:dict):
        # Load DAPI
        with h5py.File(dna_save_file, "r", libver='latest') as _f:
            _fov_id = _f.attrs['fov_id']
            _fov_name = _f.attrs['fov_name']
            # load DAPI
            if 'dapi_im' in _f.attrs.keys():
                _dapi_im = _f.attrs['dapi_im']
                # translate DNA
                _dapi_im = Align_Segmentation._correct_image3D_by_microscope_param(_dapi_im, microscpe_params) # transpose and flip
            else:
                _dapi_im = None
        return _dapi_im, _fov_id, _fov_name
    @staticmethod
    def _load_rna_dapi(rna_dapi_file:str, microscpe_params:dict):
        _rna_dapi = np.load(rna_dapi_file)
        _rna_dapi = Align_Segmentation._correct_image3D_by_microscope_param(_rna_dapi, microscpe_params) # transpose and flip
        return _rna_dapi
    @staticmethod
    def _read_microscope_json(_microscope_file:str,):
        return _read_microscope_json(_microscope_file)

    @staticmethod
    def _correct_image3D_by_microscope_param(image3D:np.ndarray, microscope_params:dict):
        """Correct 3D image with microscopy parameter"""
        _image = copy.copy(image3D)
        if not isinstance(microscope_params, dict):
            raise TypeError(f"Wrong inputt ype for microscope_params, should be a dict")
        # transpose
        if 'transpose' in microscope_params and microscope_params['transpose']:
            _image = _image.transpose((0,2,1))
        if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
            _image = np.flip(_image, 2)
        if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
            _image = np.flip(_image, 1)
        return _image

    @staticmethod
    def _correct_image2D_by_microscope_param(image2D:np.ndarray, microscope_params:dict):
        """Correct 3D image with microscopy parameter"""
        _image = copy.copy(image2D)
        # transpose
        if 'transpose' in microscope_params and microscope_params['transpose']:
            _image = _image.transpose((1,0))
        if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
            _image = np.flip(_image, 1)
        if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
            _image = np.flip(_image, 0)
        return _image

    def _generate_dna_mask(self, target_dna_Zcoords=default_dna_Zcoords, save_dtype=np.uint16):
        # process microscope.json
        _rna_mparam = _read_microscope_json(self.rna_microscope_file)
        _dna_mparam = _read_microscope_json(self.dna_microscope_file)
        # load RNA
        _rna_mask, _rna_Zcoords, _fovcell_2_uid = self._load_rna_feature(self.rna_feature_file,)
        _rna_dapi = self._load_rna_dapi(self.rna_dapi_file, _rna_mparam)
        # generate full
        _full_rna_mask = interploate_z_masks(_rna_mask, _rna_Zcoords, 
                                             target_dna_Zcoords, verbose=self.verbose)
        # load DNA
        _dna_dapi, _fov_id, _fov_name = self._load_dna_info(self.dna_save_file, _dna_mparam)
        # decide rotation matrix
        #f _dna_mparam.get('transpose', True):
        #    _dna_rot_mat = self.rotation_mat.transpose()
            
        # translate
        _dna_mask, _rot_dna_dapi = translate_segmentation(
            _rna_dapi, _dna_dapi, self.rotation_mat, 
            label_before=_full_rna_mask, 
            return_new_dapi=True, verbose=self.verbose)
        # Do dialation
        if 'dialation_size' in self.parameters:
            _dna_mask = grey_dilation(_dna_mask, size=self.parameters['dialation_size'])
        _dna_mask = np.clip(_dna_mask, np.iinfo(save_dtype).min, np.iinfo(save_dtype).max,)
        _dna_mask = _dna_mask.astype(save_dtype)
        # add to attribute
        self.dna_mask = _dna_mask
        self.fov_id = _fov_id
        self.fov_name = _fov_name
        self.fovcell_2_uid = _fovcell_2_uid
        if self.debug:
            return _dna_mask, _full_rna_mask, _rna_dapi, _rot_dna_dapi, _dna_dapi
        else:
            return _dna_mask,

    def _save(self, save_hdf5_file:str)->None:
        if self.verbose:
            print(f"-- saving segmentation info from fov:{self.fov_id} into file: {save_hdf5_file}")
        with h5py.File(save_hdf5_file,'a') as _f:
            _fov_group = _f.require_group(str(self.fov_id))
            _fov_group.attrs['fov_id'] = self.fov_id
            _fov_group.attrs['fov_name'] = self.fov_name
            # add dataset:
            if 'dna_mask' in _fov_group.keys() and self.overwrite:
                del(_fov_group['dna_mask'])
            if 'dna_mask' not in _fov_group.keys():
                _mask_dataset = _fov_group.create_dataset('dna_mask', data=self.dna_mask)
            # add uid info
            _uid_group = _fov_group.require_group('cell_2_uid')
            for (_fov_id, _cell_id), _uid in self.fovcell_2_uid.items():
                if str(_cell_id) in _uid_group.keys() and self.overwrite:
                    del(_uid_group[str(_cell_id)])
                if str(_cell_id) not in _uid_group.keys():
                    _uid_group.create_dataset(str(_cell_id), data=_uid, shape=(1,))
        return

    def _load(self, save_hdf5_file:str)->bool:
        # load DNA
        _dna_mparam = _read_microscope_json(self.dna_microscope_file)
        _, _fov_id, _fov_name = self._load_dna_info(self.dna_save_file, _dna_mparam)
        self.fov_id = _fov_id
        self.fov_name = _fov_name
        if self.verbose:
            print(f"-- loading segmentation info from fov:{self.fov_id} into file: {save_hdf5_file}")
        if not os.path.exists(save_hdf5_file):
            print(f"--- sav_hdf5_file:{save_hdf5_file} does not exist, skip loading.")
            return False
        # load
        with h5py.File(save_hdf5_file, 'r') as _f:
            if str(self.fov_id) not in _f.keys():
                return False
            _fov_group = _f[str(self.fov_id)]
            # mask
            self.dna_mask = _fov_group['dna_mask'][:]
            # uid
            self.fovcell_2_uid = {}
            _uid_group = _fov_group['cell_2_uid']
            for _cell_id in _uid_group.keys():
                self.fovcell_2_uid[(self.fov_id, int(_cell_id))] = _uid_group[_cell_id][:][0]
        return True


def translate_segmentation(dapi_before, dapi_after, before_to_after_rotation,
                           label_before=None, label_after=None,
                           return_new_dapi=False,
                           verbose=True,
                           ):
    """ """
    from ..correction_tools.alignment import calculate_translation
    from ..correction_tools.translate import warp_3d_image
    # calculate drift
    _rot_dapi_after, _rot, _dft = calculate_translation(dapi_before, dapi_after, before_to_after_rotation,)
    # get dimensions
    _dz,_dx,_dy = np.shape(dapi_before)
    _rotation_angle = np.arcsin(_rot[0,1])/math.pi*180
    
    if label_before is not None:
        _seg_labels = np.array(label_before)
        _rotation_angle = -1 * _rotation_angle
        _dft = -1 * _dft
    elif label_after is not None:
        _seg_labels = np.array(label_after)
    else:
        ValueError('Either label_before or label_after should be given!')
    # generate rotation matrix in cv2
    if verbose:
        print('- generate rotation matrix')
    _rotation_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), _rotation_angle, 1)
    # rotate segmentation
    if verbose:
        print('- rotate segmentation label with rotation matrix')
    _rot_seg_labels = np.array(
        [cv2.warpAffine(_seg_layer,
                        _rotation_M, 
                        _seg_layer.shape, 
                        flags=cv2.INTER_NEAREST,
                        #borderMode=cv2.BORDER_CONSTANT,
                        borderMode=cv2.BORDER_REPLICATE,
                        #borderValue=int(np.min(_seg_labels)
                        )
            for _seg_layer in _seg_labels]
        )
    # warp the segmentation label by drift
    _dft_rot_seg_labels = warp_3d_image(_rot_seg_labels, _dft, 
        warp_order=0, border_mode='nearest')
    if return_new_dapi:
        return _dft_rot_seg_labels, _rot_dapi_after
    else:
        return _dft_rot_seg_labels


# generate bounding box
def segmentation_mask_2_bounding_box(mask, cell_id=None, extend_pixel=1):
    from ..classes.preprocess import ImageCrop_3d
    if cell_id is not None and (mask==cell_id).any():
        _mask = (mask==cell_id)
    else:
        _mask = mask
    extend_pixel = int(extend_pixel)
    _crop = []
    for _i, _sz in enumerate(_mask.shape):
        _inds = np.where(np.max(_mask, axis=tuple(np.setdiff1d(np.arange(len(_mask.shape)), _i)) ) )[0]
        _crop.append([max(np.min(_inds)-extend_pixel, 0), 
                      min(np.max(_inds)+1+extend_pixel, _sz)])
    _crop = ImageCrop_3d(_crop, _mask.shape)
    return _crop

# interpolate matrices
def interploate_z_masks(z_masks, 
                        z_coords, 
                        target_z_coords=default_dna_Zcoords,
                        mode='nearest',
                        verbose=True,
                        ):

    # target z
    _final_mask = []
    _final_coords = np.round(target_z_coords, 3)
    for _fz in _final_coords:
        if _fz in z_coords:
            _final_mask.append(z_masks[np.where(z_coords==_fz)[0][0]])
        else:
            if mode == 'nearest':
                _final_mask.append(z_masks[np.argmin(np.abs(z_coords-_fz))])
                continue
            # find nearest neighbors
            if np.sum(z_coords > _fz) > 0:
                _upper_z = np.min(z_coords[z_coords > _fz])
            else:
                _upper_z = np.max(z_coords)
            if np.sum(z_coords < _fz) > 0:
                _lower_z = np.max(z_coords[z_coords < _fz])
            else:
                _lower_z = np.min(z_coords)

            if _upper_z == _lower_z:
                # copy the closest mask to extrapolate
                _final_mask.append(z_masks[np.where(z_coords==_upper_z)[0][0]])
            else:
                # interploate
                _upper_mask = z_masks[np.where(z_coords==_upper_z)[0][0]].astype(np.float32)
                _lower_mask = z_masks[np.where(z_coords==_lower_z)[0][0]].astype(np.float32)
                _inter_mask = (_upper_z-_fz)/(_upper_z-_lower_z) * _lower_mask 
                #_final_mask.append(_inter_mask)
                
    if verbose:
        print(f"- reconstruct {len(_final_mask)} layers")
    
    return np.array(_final_mask)


def _batch_align_segmentation(
    _fov_id:int, 
    target_dna_Zcoords:np.ndarray,
    rna_feature_file:str,
    rna_dapi_file:str,
    dna_save_file:str,
    rna_microscope_file:str,
    dna_microscope_file:str,
    rotation_mat:np.ndarray, 
    segmentation_save_file, save=True, 
    save_file_lock=None,
    align_parameters={},
    make_plot=True, 
    overwrite:bool=False,
    debug:bool=False,
    return_mask:bool=False,
    verbose:bool=True,
    )->np.ndarray:
    """Batch function to align the segmentation"""
    if verbose:
        print(f"- Aligning segmentation for fov: {_fov_id}")
    _align_seg = Align_Segmentation(
        rna_feature_file=rna_feature_file, 
        rna_dapi_file=rna_dapi_file,
        dna_save_file=dna_save_file,
        rna_microscope_file=rna_microscope_file, 
        dna_microscope_file=dna_microscope_file,
        rotation_mat=rotation_mat,
        parameters=align_parameters,
        overwrite=overwrite, debug=debug, verbose=verbose,
        )
    # test load if not overwrite
    if not overwrite:
        # initiate lock
        if 'save_file_lock' in locals() and save_file_lock is not None:
            save_file_lock.acquire()
        _exist_flag = _align_seg._load(segmentation_save_file)
        # release lock
        if 'save_file_lock' in locals() and save_file_lock is not None:
            save_file_lock.release()
    else:
        _exist_flag = False
        
    if not _exist_flag:
        # generate segmentation
        _align_seg._generate_dna_mask(target_dna_Zcoords=target_dna_Zcoords)
        # save
        if save:
            # initiate lock
            if 'save_file_lock' in locals() and save_file_lock is not None:
                save_file_lock.acquire()
            _align_seg._save(segmentation_save_file)
            # release lock
            if 'save_file_lock' in locals() and save_file_lock is not None:
                save_file_lock.release()
    # make plot
    if make_plot:
        MaskFig_SaveFile = os.path.join(os.path.dirname(segmentation_save_file), 
                                        os.path.basename(_align_seg.fov_name).replace('.dax', '_SegmentationMask.png'))
        if not os.path.exists(os.path.dirname(MaskFig_SaveFile)):
            os.makedirs(os.path.dirname(MaskFig_SaveFile))
        ax = plot_segmentation(_align_seg.dna_mask, save_filename=MaskFig_SaveFile, verbose=verbose)

    if return_mask:
        return _align_seg.dna_mask
    else:
        return None