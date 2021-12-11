import os
import time
import numpy as np
import json

from numpy.lib.arraysetops import isin
import pandas as pd
from scipy.ndimage import center_of_mass, find_objects

default_dim = 3
default_axis_names = ['z', 'x', 'y']

def SegmentationMask3D_2_CellLocations(segmentation_mask, 
                                       fov_id, #fov_position, 
                                       image_sizes, pixel_sizes,
                                       save=True, save_filename=None,
                                       overwrite=False, verbose=True,
                                       ):
    """Use Segmentation mask in each field-of-view and other spatial information,
    generate a relative locations for all the cells to the center of field-of-view inside of this mask"""
    
    _start_time = time.time()

    if save_filename is None or not os.path.exists(save_filename) or overwrite:

        if verbose:
            print(f"- Start process segmentation into cell locations.")
        
        # load segmentation if necessary
        if isinstance(segmentation_mask, str):
            segmentation_mask =  np.load(segmentation_mask)
        elif isinstance(segmentation_mask, np.ndarray):
            pass
        else:
            raise TypeError(f"Wrong input type for segmentation_mask")

        #fov_position = np.array(fov_position)
        #if len(fov_position) == default_dim -1:
        #    fov_position = np.concatenate([[0], fov_position])
        #elif len(fov_position) == default_dim:
        #    pass
        #else:
        #    raise IndexError(f"fov position should either be 2d or 3d")
            
        image_sizes = np.array(image_sizes)[:default_dim]
        pixel_sizes = np.array(pixel_sizes)[:default_dim]
        
        cell_location_df = pd.DataFrame()

        # calculate center of mass
        _cells = np.unique(segmentation_mask)[1:]
        _centers = center_of_mass(segmentation_mask, labels=segmentation_mask, index=_cells)

        for _center, _cell in zip(_centers, _cells):
            _mask = segmentation_mask == _cell
            _ct_in_um = (np.array(_center) - (image_sizes /2)) * pixel_sizes / 1000 #+ fov_position
            _box = find_objects(_mask)[0]
            _volume = np.sum(_mask)
            
            # create dict for this cell
            _info_dict = {
                'fov_id': int(fov_id),
                'cell_id': int(_cell),
                'volume': _volume,
            }
            _center_dict = {f"center_{_ax}":_c for _ax, _c in zip(default_axis_names, _ct_in_um)}
            
            _border_dict = {}
            for _i, (_ax, _bd) in enumerate(zip(default_axis_names, _box)):
                _border_dict[f"min_{_ax}"] = (_bd.start - (image_sizes/2)[_i]) * (pixel_sizes/1000)[_i] #+ fov_position[_i]
                _border_dict[f"max_{_ax}"] = (_bd.stop - (image_sizes/2)[_i]) * (pixel_sizes/1000)[_i] #+ fov_position[_i]
            
            # merge
            _info_dict.update(_center_dict)
            _info_dict.update(_border_dict)
            # append
            cell_location_df = cell_location_df.append(_info_dict, ignore_index=True)

        if verbose:
            print(f"-- {len(cell_location_df)} cells converted into MetaData")

        if save and save_filename is not None:
            if verbose:
                print(f"-- save {len(cell_location_df)} cells into file:{save_filename}")
            cell_location_df.to_csv(save_filename, index=False, header=True)

    else:
        cell_location_df = pd.read_csv(save_filename, header=0)
        print(f"- directly load {len(cell_location_df)} cells file: {save_filename}")

    if verbose:
        _execute_time = time.time() - _start_time
        print(f"-- generated metadata in {_execute_time:.3f}s")


    return cell_location_df


def Translate_CellLocations(metadata, microscope_info,
                            fov_position=None,
                            ):
    """Translate current metadata with microscope information"""
    
    if isinstance(metadata, str):
        metadata =  pd.read_csv(metadata, header=0)
    elif isinstance(metadata, pd.DataFrame):
        pass
    else:
        raise TypeError(f"Wrong input type for metadata")
        
    if fov_position is None:
        fov_position = np.zeros(default_dim)
    elif len(fov_position) == default_dim -1:
        fov_position = np.concatenate([[0], np.array(fov_position)])
    elif len(fov_position) == default_dim:
        fov_position = np.array(fov_position)
    else:
        raise IndexError(f"fov position should either be 2d or 3d")
        
    # check microscope info
    if isinstance(microscope_info, str):
        microscope_info =  json.load(open(microscope_info, 'r'))
    elif isinstance(microscope_info, dict):
        pass
    else:
        raise TypeError(f"Wrong input type for microscopy_info")
        
    #print(metadata)
    #print(fov_position)
    #print(microscope_info)
    
    _centers = np.array(metadata[['center_z', 'center_x', 'center_y']])
    _relative_centers = _centers - fov_position
    
    _mins = np.array(metadata[['min_z', 'min_x', 'min_y']])
    _relative_mins = _mins - fov_position
    
    _maxs = np.array(metadata[['max_z', 'max_x', 'max_y']])
    _relative_maxs = _maxs - fov_position
    
    #print(_relative_centers[0])
    #print(_relative_mins)
    #print(_relative_maxs)
    
    new_metadata = metadata.copy()
    
    if microscope_info['transpose']:
        #print('transpose')
        # swap relative x and y
        _relative_centers[:,np.array([-2,-1])] = _relative_centers[:,np.array([-1,-2])]
        _relative_mins[:,np.array([-2,-1])] = _relative_mins[:,np.array([-1,-2])]
        _relative_maxs[:,np.array([-2,-1])] = _relative_maxs[:,np.array([-1,-2])]
    #print(_relative_centers[0])
    
    if microscope_info['flip_horizontal']:
        #print('flip_horizontal')
        _relative_centers[:,-2]  = -1 * _relative_centers[:,-2]
        _relative_mins[:,-2], _relative_maxs[:,-2] = -1*_relative_maxs[:,-2], -1*_relative_mins[:,-2]
    
    #print(_relative_centers[0])
    
    if microscope_info['flip_vertical']:
        #print('flip_vertical')
        _relative_centers[:,-1]  = -1 * _relative_centers[:,-1]
        _relative_mins[:,-1], _relative_maxs[:,-1] = -1*_relative_maxs[:,-1], -1*_relative_mins[:,-1]
    #print(_relative_centers[0])

    # assign new values
    new_metadata[['center_z', 'center_x', 'center_y']] = _relative_centers + fov_position
    new_metadata[['min_z', 'min_x', 'min_y']] = _relative_mins + fov_position
    new_metadata[['max_z', 'max_x', 'max_y']] = _relative_maxs + fov_position
    
    return new_metadata

def Adjust_CellLocations_by_Position(cell_loc, position_df):
    """"""
    _new_cell_loc = cell_loc.copy()
    _fov_center = np.array(position_df.iloc[np.unique(cell_loc['fov_id'])])[0]
    # adjust fov_center length
    if len(_fov_center) == default_dim -1:
        _fov_center = np.concatenate([[0], np.array(_fov_center)])
    elif len(_fov_center) == default_dim:
        _fov_center = np.array(_fov_center)
    else:
        raise IndexError(f"fov position should either be 2d or 3d")
    
    _new_cell_loc[['center_z', 'center_x', 'center_y']] = cell_loc[['center_z', 'center_x', 'center_y']] + _fov_center
    _new_cell_loc[['min_z', 'min_x', 'min_y']] = cell_loc[['min_z', 'min_x', 'min_y']] + _fov_center
    _new_cell_loc[['max_z', 'max_x', 'max_y']] = cell_loc[['max_z', 'max_x', 'max_y']] + _fov_center
    
    return _new_cell_loc

def Merge_CellLocations(cell_location_list, 
                        microscope_info, 
                        position_df, 
                        fov_ids=None,
                        save=True, save_filename=None,
                        overwrite=False, verbose=True,
                        ):
    """Merge cell-locations from multiple field-of-views"""
    _start_time = time.time()
    
    if save_filename is None or not os.path.exists(save_filename) or overwrite:
        if verbose:
            print(f"- Start merging {len(cell_location_list)} cell locations")
        # initialize
        merged_cell_loc_df = pd.DataFrame()
        if fov_ids is None:
            fov_ids = np.arange(len(position_df))
        # loop through each cell-location file
        for _cell_loc in cell_location_list:
            if isinstance(_cell_loc, str):
                _cell_loc =  pd.read_csv(_cell_loc, header=0)
            elif isinstance(_cell_loc, pd.DataFrame):
                _cell_loc = _cell_loc.copy()
            else:
                raise TypeError(f"Wrong input type for _cell_loc")

            # translate by microscope info
            if microscope_info is not None:
                _cell_loc = Translate_CellLocations(_cell_loc, microscope_info)
            # adjust by position_df
            if position_df is not None:
                _cell_loc = Adjust_CellLocations_by_Position(_cell_loc, position_df)
            # merge
            merged_cell_loc_df = pd.concat([merged_cell_loc_df, _cell_loc],
                                                ignore_index=True)

        if verbose:
            print(f"-- {len(merged_cell_loc_df)} cells converted into MetaData")

        if save and (save_filename is not None or overwrite):
            if verbose:
                print(f"-- save {len(merged_cell_loc_df)} cells into file:{save_filename}")
            merged_cell_loc_df.to_csv(save_filename, index=False, header=True)
    else:
        merged_cell_loc_df = pd.read_csv(save_filename, header=0)
        print(f"- directly load {len(merged_cell_loc_df)} cells file: {save_filename}")

    if verbose:
        _execute_time = time.time() - _start_time
        print(f"-- merge cell-locations in {_execute_time:.3f}s")
            
    return merged_cell_loc_df

