import os, glob, sys, time
import numpy as np 
import pandas as pd
import re
import h5py
from tqdm import tqdm
from scipy.spatial import KDTree

Axis3D_infos = ['z', 'x', 'y']
Spot3D_infos = ['height', 'z', 'x', 'y', 'background', 'sigma_z', 'sigma_x', 'sigma_y', 'sin_t', 'sin_p', 'eps']

Pixel3D_infos = [f"pixel_{_ax}" for _ax in Axis3D_infos]
from ..classes import default_pixel_sizes
from ..classes.preprocess import Spots3D, SpotTuple

def CellSpotsDf_2_CandSpots(
    _cell_spots_df, 
    spot_infos=Spot3D_infos, 
    pixel_info_names=Pixel3D_infos
    ):
    """Convert spot_df of a cell into cand_spot format"""
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
                if hasattr(_spots, 'channels') and _spots.channels != None and len(_spots.channels) == len(_spots):
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
                     fov_id=None, cell_id=None, cell_uid=None, homolog=None, sel_ind=None,
                     bit_2_channel=None, codebook=None,
                     spot_infos=Spot3D_infos, pixel_infos=Pixel3D_infos,
                     ):
    """Convert SpotTuple class into a dict"""
    if spot_tuple is None:
        return {}
    # init
    _tuple_info_dict = {'fov_id':fov_id, 
                        'cell_id':cell_id, 
                        'uid':cell_uid,
                        'homolog':homolog,
                        'sel_index':sel_ind,}
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
        _region_dict = {'region_name':None, 'start':None, 'end':None, 'chr':None, 'chr_order':None}
    else:
        _region_dict = {}
        _reg_info = codebook.loc[codebook['id']==_tuple_info_dict['region_id'], ['name', 'chr']].values[0]
        _region_dict['region_name'] = _reg_info[0]
        _region_dict['start'], _region_dict['end'] = _reg_info[0].split(':')[1].split('-')
        _region_dict['chr'] = _reg_info[1]
        if 'chr_order' in codebook.columns:
            _region_dict['chr_order'] = codebook.loc[codebook['id']==_tuple_info_dict['region_id'], ['chr_order']].values[0][0]
    _tuple_info_dict.update(_region_dict)
    return _tuple_info_dict

def spotTuple_2_positionDict(spot_tuple, axes_infos=Axis3D_infos):
    _posDict = {f"center_{_name}":_pos
                for _name, _pos in zip(axes_infos, spot_tuple.centroid_spot().to_positions()[0])}
    _posDict["center_intensity"] = np.mean(spot_tuple.intensities())
    _posDict["center_intensity_var"] = np.std(spot_tuple.intensities())/np.mean(spot_tuple.intensities())
    _posDict["center_internal_dist"] = np.median(spot_tuple.dist_internal())
    return _posDict

def spotTupleList_2_DataFrame(spotTuple_list, 
                              fov_id=None, cell_id=None, 
                              cell_uid=None, homolog=None, 
                              bit_2_channel=None, codebook=None, include_position=True,
                              spot_infos=Spot3D_infos, pixel_infos=Pixel3D_infos, 
                              axes_infos=Axis3D_infos):
    _dict_list = []
    for _g in spotTuple_list:
        _info_dict = SpotTuple_2_Dict(_g,
                                      fov_id=fov_id, cell_id=cell_id, cell_uid=cell_uid,
                                      homolog=homolog, sel_ind=getattr(_g, 'sel_ind',None), bit_2_channel=bit_2_channel,
                                      codebook=codebook, spot_infos=spot_infos, pixel_infos=pixel_infos)
        if include_position:
            _pos_dict = spotTuple_2_positionDict(_g, axes_infos=axes_infos)
            _info_dict.update(_pos_dict)
        _dict_list.append(_info_dict)
    return pd.DataFrame(_dict_list)

def CandSpotDf_add_positions(candSpotDf, intensity_name='height', axes_infos=Axis3D_infos, pixel_infos=Pixel3D_infos, ):
    _ext_candSpotDf = candSpotDf.copy()
    for _name, _pixel_name in zip(axes_infos, pixel_infos):
        if _name in _ext_candSpotDf.columns:
            _ext_candSpotDf[f"center_{_name}"] = _ext_candSpotDf[_name] * _ext_candSpotDf[_pixel_name]
    # intensity
    _ext_candSpotDf["center_intensity"] = _ext_candSpotDf[intensity_name]
    return _ext_candSpotDf


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
        # if all-spots are invalid, return NOne
        if len(_spots) == 0:
            _spot_groups.append(None)
            continue
        # otherwise create tuple
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
        _gp.homolog = _grp_row.get('homolog', None)
        _gp.sel_ind = _grp_row.get('sel_index',None)
        # chr info
        _gp.chr = _grp_row.get('chr', None)
        _gp.chr_order = _grp_row.get('chr_order', None)

        # append
        _spot_groups.append(_gp)

    return _spot_groups

def load_preprocess_spots(
    savefile:str, 
    data_type:str, 
    sel_bits:list=None,
    pixel_sizes=default_pixel_sizes
    ):
    """Load spots from pre-processed output hdf5"""
    _spots_list = []
    with h5py.File(savefile, 'r') as _f:
        # load pixel size
        
        if data_type in _f.keys():
            _grp = _f[data_type]
            # load infos
            _ids = _grp['ids'][:]
            # determine sel_bits if not given
            if sel_bits is None:
                sel_bits = _ids
            # loop through sel_bits
            for _bit in sel_bits:
                _ind = list(_ids).index(_bit)
                _spots = _grp['spots'][_ind]
                _spots = _spots[_spots[:,0] > 0]
                _channel = _grp['channels'][_ind].decode()
                _spots_obj = Spots3D(_spots, bits=_bit, channels=_channel, pixel_sizes=pixel_sizes)
                # append
                _spots_list.append(_spots_obj)
            return _spots_list, sel_bits
        else:
            return None, sel_bits


def merge_Spots3DList(
    spots_list, 
    pixel_sizes=default_pixel_sizes
    ):
    """Merge a list of Spots3D objects"""
    _combined_spots = np.concatenate(spots_list)
    _combined_bits = np.concatenate([getattr(_spots, 'bits', [None]*len(_spots)) for _spots in spots_list])
    _combined_channels = np.concatenate([getattr(_spots, 'channels', [None]*len(_spots)) for _spots in spots_list])
    _combined_pixel_sizes = [getattr(_spots, 'pixel_sizes', pixel_sizes) for _spots in spots_list]
    if len(np.unique(_combined_pixel_sizes, axis=0)) > 1:
        raise ValueError(f"pixel sizes not consistent, exit")
    else:
        _pixel_sizes = np.unique(_combined_pixel_sizes, axis=0)[0]
    # generate spots
    return Spots3D(_combined_spots,
                   bits=_combined_bits,
                   channels=_combined_channels,
                   pixel_sizes=_pixel_sizes,
                  )

def merge_RelabelSpots(
    old_spots:Spots3D, 
    new_spots:Spots3D,
    search_radius=150,
    pixel_sizes=default_pixel_sizes,
    ):
    """Merge two spots objects"""
    _combined_bits = np.concatenate([old_spots.bits, new_spots.bits])
    if hasattr(old_spots, 'channels') and hasattr(new_spots, 'channels') and old_spots.channels is not None:
        _combined_channels = np.concatenate([old_spots.channels, new_spots.channels])
    else:
        _combined_channels = None
    # generate combined spots
    _combined_spots = Spots3D(np.concatenate([old_spots,new_spots]), 
                              bits=_combined_bits, channels=_combined_channels,
                              pixel_sizes=pixel_sizes)
    # create flags to keep
    _spot_flags = np.ones(len(_combined_spots), dtype=bool)
    # create kdtree to query neighbors
    _tree = KDTree(_combined_spots.to_positions())
    # loop through spots from highest intensities
    for _spot_ind in np.argsort(_combined_spots.to_intensities())[::-1]:
        _sel_position = _combined_spots.to_positions()[_spot_ind]
        # search neighbors 
        _nb_spot_inds = _tree.query_ball_point(_sel_position,search_radius)
        _nb_spot_inds = np.setdiff1d(_nb_spot_inds, [_spot_ind])
        # don't use these spots later
        if len(_nb_spot_inds) > 0:
            _spot_flags[_nb_spot_inds] = False
    
    # return kept_spots
    _kept_spots = _combined_spots[_spot_flags]
    return _kept_spots

def FovSpots3D_2_DataFrame(
    spots:Spots3D,
    fov_id:int, 
    cell_ids:list,
    fovcell_2_uid:dict={},
    pixel_sizes=default_pixel_sizes,
    spot_info_names=Spot3D_infos,
    pixel_info_names=Pixel3D_infos,
    ignore_spots_out_cell=True,
    ):
    # Define two sub-function for names and infos
    def _assemble_df_names(    
        spot_info_names=spot_info_names,
        pixel_info_names=pixel_info_names,
    ):
        # assemble columns
        _columns = ['fov_id', 'cell_id',]
        # add spot info
        _columns.extend(spot_info_names)
        # add bits
        _columns.extend(['bit', 'channel', 'uid'])
        # add pixel
        _columns.extend(pixel_info_names)
        return _columns

    def _assemble_spot_info(
        _fov_id, _cell_id,
        _spot, 
        _bit, _channel, _uid,
        _pixel_sizes,
    ):
        _spot_info = [_fov_id, _cell_id]
        _spot_info.extend(list(_spot))
        _spot_info.extend([_bit, _channel, _uid])
        _spot_info.extend(list(_pixel_sizes))
        return _spot_info
    # apply _assemble_df_names
    _columns = _assemble_df_names(spot_info_names=spot_info_names,
                                  pixel_info_names=pixel_info_names)
    _infos = []
    for _i, _spot in tqdm(enumerate(spots)):
        # cell_id
        _cell_id = cell_ids[_i]
        if ignore_spots_out_cell and \
            (_cell_id is None or _cell_id <= 0 or np.isnan(_cell_id)):
            continue
        # uid
        _uid = fovcell_2_uid.get((fov_id,_cell_id), None)
        # bit
        _bit = getattr(_spot, 'bits', None)
        # channel
        _channel = getattr(_spot, 'channels', None)
        # generate spot_info
        _info = _assemble_spot_info(
            fov_id, _cell_id, 
            _spot,
            _bit, _channel, _uid,
            pixel_sizes,
        )
        #print(_info)
        # append
        _infos.append(_info)
        
    # convert into DataFrame
    return pd.DataFrame(_infos, columns=_columns)
