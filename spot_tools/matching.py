import numpy as np
import os, sys
from . import _distance_zxy
from ..visual_tools import translate_spot_coordinates

def convert_pick_RNA_spots(rna_cell, dna_cell, rotation_mat=None,
                           intensity_th=1,
                           tss_ref_attr='EM_picked_gene_spots', tss_dist_th=500., 
                           dna_ref_attr='EM_picked_unique_spots', dna_dist_th=500., 
                           chr_ref_attr='chrom_coords', chr_dist_th=2000.,
                           add_attr=True, attr_name='ts_RNA_spots', verbose=False):
    if rotation_mat is None:
        rotation_mat = np.load(os.path.join(rna_cell.experiment_folder, 'rotation.npy'))
    
    # initiate
    sel_spot_list = []
    for _i,_chrom_coord in enumerate(dna_cell.chrom_coords):
        _sel_spots = np.zeros([len(getattr(rna_cell, 'rna-unique_spots')), 11])
        _sel_spots[:,0] = 0
        _sel_spots[:,1:] = np.nan
        sel_spot_list.append(_sel_spots)
        
    for _rid, (_spot_list, _k) in enumerate(zip(getattr(rna_cell, 'rna-unique_spots'), sorted(getattr(rna_cell, 'rna-info_dic').keys())) ):
        _info = getattr(rna_cell, 'rna-info_dic')[_k]

        # loop through each chromosome
        for _cid, (_spots, _chrom_coord) in enumerate(zip(_spot_list, dna_cell.chrom_coords)):
            # check if there are any candidate spots
            _cand_spots = _spots[_spots[:,0] >= intensity_th]
            # if there are no candidate spots, directly continue
            if len(_cand_spots) == 0:
                continue
            else:
                # do translation first
                _ts_spots = translate_spot_coordinates(rna_cell, dna_cell, 
                                                    _cand_spots, 
                                                    rotation_mat=rotation_mat, 
                                                    rotation_order='forward', verbose=False)
                ## now find ref_targets
                # if no ref_dna, check distance to center
                if 'DNA_id' not in _info or _k not in dna_cell.gene_dic:
                    _kept_spots = _ts_spots[np.linalg.norm((_ts_spots[:,1:4] - _chrom_coord) \
                                                * _distance_zxy, axis=1) <= chr_dist_th]
                # check if there are ref gene spots
                else:
                    _gene_ind = list(sorted(dna_cell.gene_dic.keys())).index(_k)
                    _tss_ref_spot = getattr(dna_cell, 'EM_picked_gene_spots')[_cid][_gene_ind]
                    _dna_ref_spot = getattr(dna_cell, 'EM_picked_unique_spots')[_cid][_info['DNA_id']]
                    
                    _keep_flags = np.zeros(len(_ts_spots),dtype=np.bool)
                    if not np.isnan(_tss_ref_spot).any():
                        _keep_flags += (np.linalg.norm((_ts_spots - _tss_ref_spot)[:,1:4] \
                                                * _distance_zxy, axis=1) <= tss_dist_th)
                        #print('tss', sum(_keep_flags))
                    if not np.isnan(_dna_ref_spot).any():
                        _keep_flags += (np.linalg.norm((_ts_spots - _dna_ref_spot)[:,1:4] \
                                                * _distance_zxy, axis=1) <= dna_dist_th)
                        #print('dna', sum(_keep_flags))
                    if np.isnan(_tss_ref_spot).any() and np.isnan(_dna_ref_spot).any():
                        _keep_flags += (np.linalg.norm((_ts_spots - _dna_ref_spot)[:,1:4] \
                                                * _distance_zxy, axis=1) <= chr_dist_th)
                        #print('chr', sum(_keep_flags))
                    _kept_spots = _ts_spots[_keep_flags]
                
                # then keep the brightest one
                if len(_kept_spots) > 0:
                    _sel_spot = _kept_spots[np.argmax(_kept_spots[:,0])]
                    sel_spot_list[_cid][_rid] = _sel_spot
    if add_attr:
        if verbose:
            print(f"-- add attribute: {attr_name} to DNA cell")
        setattr(dna_cell, attr_name, sel_spot_list)

    return sel_spot_list