import os, glob, sys, time
import numpy as np 
import pandas as pd

Axis3D_infos = ['z', 'x', 'y']
Spot3D_infos = ['height', 'z', 'x', 'y', 'background', 'sigma_z', 'sigma_x', 'sigma_y', 'sin_t', 'sin_p', 'eps']

Pixel3D_infos = [f"pixel_{_ax}" for _ax in Axis3D_infos]
from ..classes.decode import default_pixel_sizes

def CellSpotsDf_2_CandSpots(_cell_spots_df, 
    spot_infos=Spot3D_infos, 
    pixel_info_names=Pixel3D_infos):
    """Convert spot_df of a cell into cand_spot format"""
    from ..classes.preprocess import Spots3D
    return Spots3D(_cell_spots_df[spot_infos], 
                   bits=_cell_spots_df['bit'].values,
                   channels=_cell_spots_df['channel'].values,
                   pixel_sizes=np.unique(_cell_spots_df[pixel_info_names].values.astype(np.int32), axis=0)[0])

def FovCell2Spots_2_DataFrame(cell_2_spots:dict,
                              fov_id=None,
                              bit_2_channel=None,
                              fovcell_2_uid=None,
                              spot_info_names=Spot3D_infos,
                              pixel_sizes=default_pixel_sizes,
                              pixel_info_names=Pixel3D_infos,
                              save=True, save_filename=None,
                              verbose=True,
                              ):
    """Convert cell_2_spots into pandas.DataFrame"""
    if verbose:
        print(f"- Converting spots from {len(cell_2_spots)} cells into DataFrame")
    # assemble columns
    _columns = ['fov_id', 'cell_id',]
    # add spot info
    _columns.extend(spot_info_names)
    # add bits
    _columns.extend(['bit', 'channel', 'uid'])
    # add pixel
    _columns.extend(pixel_info_names)

    # loop through cell_2_spots
    _spot_info_list = []
    for _cell_id, _spot_dict in cell_2_spots.items():
        for _bit, _spots in _spot_dict.items():
            for _spot, _bit in zip(_spots, _spots.bits):
                _spot_info = [fov_id, _cell_id]
                _spot_info.extend(list(_spot))
                # bit
                _spot_info.append(_bit)
                # channel
                if isinstance(bit_2_channel, dict) and _bit in bit_2_channel:
                    _spot_info.append(bit_2_channel[_bit])
                else:
                    _spot_info.append(None)
                # uid
                if isinstance(fovcell_2_uid, dict) and (fov_id, _cell_id) in fovcell_2_uid:
                    _spot_info.append(fovcell_2_uid[(fov_id, _cell_id)])
                else:
                    _spot_info.append(None)

                _spot_info.extend(list(pixel_sizes))
                # append
                _spot_info_list.append(_spot_info)
    # create dataframe
    _spots_df = pd.DataFrame(_spot_info_list, columns=_columns)
    # save
    if save and save_filename is not None:
        if verbose:
            print(f"- Save DataFrame to file: {save_filename}")
        _spots_df.to_csv(save_filename, index=False)
    # return
    return _spots_df


