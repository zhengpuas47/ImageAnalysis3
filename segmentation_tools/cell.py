from warnings import WarningMessage
import numpy as np
from numpy.lib.npyio import save
from cellpose import models
import torch
import cv2
import os,sys,time
import skimage 

default_cellpose_kwargs = {
    'anisotropy': 1,
    'diameter': 40,
    'min_size': 500,
    'stitch_threshold': 0.1,
}
default_pixel_sizes = [250,108,108]

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
        _full_mask = Cellpose_Segmentation_Psedu3D.interploate_z_masks(_clean_mask, _z)
        
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
            np.array([_projected_im,_projected_im]),
            z_axis=1,
            channel_axis=0,
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
                        print(f"-- overlap found for cell:{_i} to {_j}", _i_percent, _j_percent)

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
    
    @staticmethod
    def interploate_z_masks(z_masks, 
                            z_coords, 
                            target_z_coords=np.round(np.arange(0,12,0.2),2),
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
                    
        if verbose:
            print(f"- reconstruct {len(_final_mask)} layers")
        
        return np.array(_final_mask)

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
            do_3D=True,
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

import math
import cv2

def translate_segmentation(dapi_before, dapi_after, before_to_after_rotation,
                           label_before=None, label_after=None,
                           verbose=True,
                           ):
    """ """
    from ..correction_tools.alignment import calculate_translation
    from ..correction_tools.translate import warp_3d_image
    # calculate drift
    rot_dna_dapi_im, _rot, _dft = calculate_translation(dapi_before, dapi_after, before_to_after_rotation,)
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
    _rotation_M = cv2.getRotationMatrix2D((_dx/2, _dy/2), _rotation_angle, 1)
    # rotate segmentation
    _rot_seg_labels = np.array([cv2.warpAffine(_seg_layer,
                                              _rotation_M, 
                                              _seg_layer.shape, 
                                              flags=cv2.INTER_NEAREST,
                                              borderMode=cv2.BORDER_CONSTANT)
                               for _seg_layer in _seg_labels])
    # warp the segmentation label by drift
    _dft_rot_seg_labels = warp_3d_image(_rot_seg_labels, _dft, warp_order=0)
    
    return _dft_rot_seg_labels#.astype(np.int32)