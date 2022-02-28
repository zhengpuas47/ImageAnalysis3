import os, glob, sys, time
import numpy as np 
import pandas as pd
import re
import copy

Axis3D_infos = ['z', 'x', 'y']
Spot3D_infos = ['height', 'z', 'x', 'y', 'background', 'sigma_z', 'sigma_x', 'sigma_y', 'sin_t', 'sin_p', 'eps']

Pixel3D_infos = [f"pixel_{_ax}" for _ax in Axis3D_infos]
from ..classes.decode import default_pixel_sizes
from ..classes.preprocess import Spots3D, SpotTuple

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
            for _ispot, (_spot, _bit) in enumerate(zip(_spots, _spots.bits)):
                _spot_info = [fov_id, _cell_id]
                _spot_info.extend(list(_spot))
                # bit
                _spot_info.append(_bit)
                # channel
                if hasattr(_spots, 'channels') and len(_spots.channels) == len(_spots):
                    _spot_info.append(_spots.channels[_ispot])
                elif isinstance(bit_2_channel, dict) and _bit in bit_2_channel:
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


def SpotTuple_2_Dict(spot_tuple, 
                     fov_id=None, cell_id=None, cell_uid=None,
                     bit_2_channel=None, codebook=None,
                     spot_infos=Spot3D_infos, pixel_infos=Pixel3D_infos,
                     ):
    """Convert SpotTuple class into a dict"""
    _tuple_info_dict = {'fov_id':fov_id, 
                        'cell_id':cell_id, 
                        'uid':cell_uid}
    for _i, _spot in enumerate(spot_tuple.spots):
        _bit = spot_tuple.spots.bits[_i]
        if bit_2_channel is None:
            _ch = None
        else:
            _ch = bit_2_channel.get(_bit, None)
        for _info, _s in zip(spot_infos, _spot.astype(np.float32)):
            _tuple_info_dict[f"{_info}_{_i}"] = _s
        # append bit and channel
        _tuple_info_dict[f"bit_{_i}"] = _bit
        _tuple_info_dict[f"channel_{_i}"] = _ch
        # spot_inds
        _tuple_info_dict[f"cand_spot_ind_{_i}"] = spot_tuple.spots_inds[_i]
    # pixelsize
    for _pinfo, _p in zip(pixel_infos, spot_tuple.pixel_sizes):
        _tuple_info_dict[f"{_pinfo}"] = _p
    # tuple_id
    _tuple_info_dict['region_id'] = getattr(spot_tuple, 'tuple_id', None)
    # region information
    if codebook is None or _tuple_info_dict['region_id'] is None:
        _region_dict = {'start':None, 'end':None, 'chr':None, 'chr_order':None}
    else:
        _region_dict = {}
        _reg_info = codebook.loc[codebook['id']==_tuple_info_dict['region_id'], ['name', 'chr', 'chr_order']].values[0]
        _region_dict['start'], _region_dict['end'] = _reg_info[0].split(':')[1].split('-')
        _region_dict['chr'] = _reg_info[1]
        _region_dict['chr_order'] = _reg_info[2]
    _tuple_info_dict.update(_region_dict)
    return _tuple_info_dict

def Dataframe_2_SpotGroups(decoder_group_df, spot_infos=Spot3D_infos, pixel_infos=Pixel3D_infos,):
    _spot_groups = []

    for _ind, _grp_row in decoder_group_df.iterrows():
        # get pixels
        _pixel_sizes = np.array(_grp_row[pixel_infos], dtype=np.float32)
        # find spot_ids
        _spot_ids = []
        for _name in _grp_row.keys():
            _matches = re.findall(r'^(.+)_([0-9]+)$', _name)
            if len(_matches) > 0:
                _spot_ids.append(int(_matches[0][-1]))
        # get spots
        _spots, _spot_bits, _spot_channels = [], [], []
        _spot_inds = []
        for _sid in np.unique(_spot_ids):
            _spot_keys = [f"{_k}_{_sid}" for _k in spot_infos]
            _spot = np.array(_grp_row[_spot_keys]).astype(np.float32)
            # skip NaN spot
            if np.isnan(_spot).any():
                continue
            # append
            _spots.append(_spot)
            _spot_bits.append(_grp_row[f"bit_{_sid}"])
            _spot_channels.append(_grp_row[f"channel_{_sid}"])
            # append ind
            _spot_inds.append(_grp_row[f"cand_spot_ind_{_sid}"])
        _spot3d = Spots3D(
            _spots, bits=_spot_bits, 
            pixel_sizes=_pixel_sizes,
            channels=_spot_channels,
        )
        _spot_inds = np.array(_spot_inds, dtype=np.int32)
        # assemble tuple
        _gp = SpotTuple(
            _spot3d, bits=_spot_bits, pixel_sizes=_pixel_sizes,
            spots_inds=_spot_inds, tuple_id=_grp_row['region_id'],
        )
        # append other information if applicable
        _gp.fov_id = _grp_row.get('fov_id', None)
        _gp.cell_id = _grp_row.get('cell_id', None)
        _gp.uid = _grp_row.get('uid', None)
        # chr info
        _gp.chr = _grp_row.get('chr', None)
        _gp.chr_order = _grp_row.get('chr_order', None)

        # append
        _spot_groups.append(_gp)

    return _spot_groups