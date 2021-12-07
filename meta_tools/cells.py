import os
import time
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
from scipy.ndimage import center_of_mass, find_objects

def SegmentationMask3D_2_CellMetaData(segmentation_mask, 
                                      fov_id, fov_position, 
                                      image_sizes, pixel_sizes,
                                      save=True, save_filename=None,
                                      overwrite=False, verbose=True,
                                      ):
    """Use Segmentation mask in each field-of-view and other spatial information,
    generate a metadata for all the cells inside of this mask"""
    
    _start_time = time.time()
    if not os.path.exists(save_filename) or overwrite:

        n_dim = 3
        axis_names = ['z', 'x', 'y']
        
        # load segmentation if necessary
        if isinstance(segmentation_mask, str):
            segmentation_mask =  np.load(segmentation_mask)
        elif isinstance(segmentation_mask, np.ndarray):
            pass
        else:
            raise TypeError(f"Wrong input type for segmentation_mask")

        fov_position = np.array(fov_position)
        if len(fov_position) == n_dim -1:
            fov_position = np.concatenate([[0], fov_position])
        elif len(fov_position) == n_dim:
            pass
        else:
            raise IndexError(f"")
            
        image_sizes = np.array(image_sizes)[:n_dim]
        pixel_sizes = np.array(pixel_sizes)[:n_dim]
        
        cell_meta_df = pd.DataFrame()

        for _cell in np.unique(segmentation_mask)[1:]:
            _mask = segmentation_mask == _cell
            _ct = (np.array(center_of_mass(_mask)) - (image_sizes /2)) * pixel_sizes / 1000 + fov_position
            _box = find_objects(_mask)[0]
            _volume = np.sum(_mask)
            
            # create dict for this cell
            _info_dict = {
                'fov_id': int(fov_id),
                'cell_id': int(_cell),
                'volume': _volume,
            }
            _center_dict = {f"center_{_ax}":_c for _ax, _c in zip(axis_names, _ct)}
            
            _border_dict = {}
            for _i, (_ax, _bd) in enumerate(zip(axis_names, _box)):
                _border_dict[f"min_{_ax}"] = (_bd.start - (image_sizes/2)[_i]) * (pixel_sizes/1000)[_i] + fov_position[_i]
                _border_dict[f"max_{_ax}"] = (_bd.stop - (image_sizes/2)[_i]) * (pixel_sizes/1000)[_i] + fov_position[_i]
            
            # merge
            _info_dict.update(_center_dict)
            _info_dict.update(_border_dict)
            # append
            cell_meta_df = cell_meta_df.append(_info_dict, ignore_index=True)

        if verbose:
            print(f"- {len(cell_meta_df)} cells converted into MetaData")

        if save and save_filename is not None:
            if verbose:
                print(f"-- save {len(cell_meta_df)} cells into file:{save_filename}")
            cell_meta_df.to_csv(save_filename, index=False, header=True)



    else:
        print(f"- directly load file: {save_filename}")
        cell_meta_df = pd.read_csv(save_filename, header=0)

    if verbose:
        _execute_time = time.time() - _start_time
        print(f"-- generate metadata in {_execute_time:.3f}s")

    return cell_meta_df


def Segmentation3D_2_Spots_Partition():
    pass