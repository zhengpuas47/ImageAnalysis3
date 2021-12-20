import os, glob, sys, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
# local
from .preprocess import Spots3D, SpotTuple
default_pixel_sizes=[250,108,108]
default_search_th = 150
default_search_eps = 0.25

class Merfish_Decoder():
    """Class to decode spot-based merfish results"""    

    def __init__(self, 
                 cand_spots_list, 
                 codebook_filename,
                 bits=None,
                 pixel_sizes=default_pixel_sizes,
                 auto=True,
                 verbose=True,
                 ):
        """Inttialize"""
        # parameters
        self.codebook_filename = codebook_filename
        self.pixel_sizes = np.array(pixel_sizes)
        # bits
        if bits is None:
            _bits = np.arange(1, len(cand_spots_list)+1)
        else:
            _bits = np.array(bits, dtype=np.int32)
        self.bits = _bits
        # spots
        if len(cand_spots_list) != len(_bits):
            raise IndexError(f"lengh of _bits: {len(_bits)} doesn't match length of cand_spots_list: {len(cand_spots_list)}")
        _cand_spots_list = []
        for _spots, _bit in zip(cand_spots_list, _bits):
            if not isinstance(_spots, Spots3D):
                _cand_spots_list.append(Spots3D(_spots, _bit, pixel_sizes=pixel_sizes) )
            else:
                _cand_spots_list.append(_spots)
        #merge cand_spots
        cand_spots = Spots3D(np.concatenate(_cand_spots_list), 
                            bits=np.concatenate([_spots.bits for _spots in _cand_spots_list]), 
                            pixel_sizes=self.pixel_sizes)
        self.cand_spots = cand_spots
        # load codebook if automatic
        if auto:
            self.load_codebook()
            self.find_valid_pairs_in_codebook()
            self.find_valid_tuples_in_codebook()
        # other attributes
        self.verbose = verbose

    def load_codebook(self):
        codebook_df = pd.read_csv(self.codebook_filename, header=0)
        codebook_df.set_index('id')
        # add to attribute
        self.codebook = codebook_df
        # save species names and ids
        if 'name' in codebook_df.columns:
            self.names = codebook_df['name'].values
        if 'id' in codebook_df.columns:
            self.ids = codebook_df['id'].values
        # select columns with info
        self.bit_names = [_c for _c in codebook_df.columns if _c.lower() not in ['id','name','chr','chr_order']]
        self.bit_ids = np.arange(1, len(self.bit_names)+1)
        self.codebook_matrix = codebook_df[self.bit_names].values

    def find_valid_pairs_in_codebook(self):
        """Find valid 2-bit pairs given in the codebook """
        from itertools import combinations
        valid_pair_bits = []
        valid_pair_ids = []
        for _icode, _code in enumerate(self.codebook_matrix):
            for _p in combinations(np.where(_code)[0], 2):
                _bs = tuple(np.sort(self.bits[np.array(_p)]))
                if _bs not in valid_pair_bits:
                    valid_pair_bits.append(_bs)
                    valid_pair_ids.append(self.ids[_icode])
        # attributes
        self.valid_pair_bits = valid_pair_bits
        self.valid_pair_ids = valid_pair_ids
        
        return valid_pair_bits, valid_pair_ids

    def find_valid_tuples_in_codebook(self):
        valid_tuples = {}
        for _icode, _code in enumerate(self.codebook_matrix):
            _bs = tuple(np.sort(self.bits[np.where(_code>0)[0]]))
            if _bs not in valid_tuples:
                valid_tuples[self.ids[_icode]] = _bs
        # attribute
        self.valid_tuples = valid_tuples
        return valid_tuples

    def find_spot_pairs_in_radius(self, search_th=default_search_th, eps=default_search_eps, keep_valid=True):
        """Build a KD tree to find pairs given a radius threshold in nm"""
        # extract all coordinatesa
        _cand_positions = self.cand_spots.to_positions()
        # build kd-tree
        _tree = KDTree(_cand_positions)
        self.kdtree = _tree
        # find pairs
        self.search_th = search_th
        self.search_eps = eps
        _pair_inds_list = list(_tree.query_pairs(self.search_th, eps=self.search_eps))
        # only keep the valid pairs
        if keep_valid:
            _kept_pair_inds_list = []
            for _inds in _pair_inds_list:
                _bts = tuple( np.sort(self.cand_spots.bits[np.array(_inds)]) )
                if _bts in self.valid_pair_bits:
                    _kept_pair_inds_list.append(_inds)
            self.pair_inds_list = _kept_pair_inds_list
        else:
            self.pair_inds_list = _pair_inds_list
        if self.verbose:
            print(f"{len(self.pair_inds_list)} pairs kept given search radius {search_th} nm.")
            
    def assemble_complete_codes(self, search_th=None, search_eps=None):
        # initialize a record
        _spot_groups = []
        _spot_group_inds = []
        _group_id = 0

        if search_th is None:
            _search_th = getattr(self, 'search_th', default_search_th)
        else:
            _search_th = search_th
        if search_eps is None:
            _search_eps = getattr(self, 'search_eps', default_search_eps)
        else:
            _search_eps = search_eps
        # append group
        _group_spots_list = [self.cand_spots[np.array(_inds)] for _inds in self.pair_inds_list]
        # query all targets
        _target_spot_inds = self.kdtree.query_ball_point([_spots.to_positions().mean(0) for _spots in _group_spots_list], 
                                                         _search_th, eps=_search_eps)
        _target_spot_inds = [np.setdiff1d(_target_inds, _inds) 
                            for _target_inds, _inds in zip(_target_spot_inds, self.pair_inds_list)]
        
        for _inds, _targets in tqdm(zip(self.pair_inds_list, _target_spot_inds)):
            if len(_targets) == 0:
                continue
            if np.sum([set(_inds).issubset(_s) for _s in _spot_group_inds]):
                continue
            # select spots
            _group_spots = self.cand_spots[np.array(_inds)]
            # get tuples (species) containing these bits
            for _id, _tuple in self.valid_tuples.items():
                if set(_group_spots.bits).issubset(set(_tuple)):
                    _related_bits = list(set(_tuple).difference(set(_group_spots.bits)))
                    # for this species, try to merge spots
                    _target_bits = self.cand_spots.bits[np.array(_targets)]
                    _kept_targets = np.array([_i for _i, _b in zip(_targets,_target_bits) if _b in _related_bits])
                    # if match found, add
                    if len(_kept_targets) > 0:
                        _kept_spots = self.cand_spots[_kept_targets]
                        _kept_spots_by_bits = [_kept_spots[_kept_spots.bits==_b] for _b in _related_bits]
                        _kept_targets_by_bits = [_kept_targets[_kept_spots.bits==_b] for _b in _related_bits]
                        # append the best fits
                        _spots_dists = [np.max(cdist(_spots.to_positions(), _group_spots.to_positions()), axis=1) for _spots in _kept_spots_by_bits]
                        _sel_kept_spots_by_bits = [_spots[np.argmax(_dists)] for _spots, _dists in zip(_kept_spots_by_bits, _spots_dists)]
                        _sel_targets_by_bits = [_ts[np.argmax(_dists)] for _ts, _dists in zip(_kept_targets_by_bits, _spots_dists)]
                        # merge
                        _merged_inds = np.concatenate([_inds, _sel_targets_by_bits])
                        _merged_spots = Spots3D(np.concatenate([_group_spots, _sel_kept_spots_by_bits])[np.argsort(_merged_inds)],
                                                bits=np.concatenate([_group_spots.bits, np.array(_related_bits)]),
                                                pixel_sizes=_group_spots.pixel_sizes,
                                                )   
                        _merged_inds = np.sort(_merged_inds)
                        # update
                        _spot_tuple = SpotTuple(_merged_spots,
                                                bits=_merged_spots.bits,
                                                pixel_sizes=_merged_spots.pixel_sizes,
                                                spots_inds=_merged_inds,
                                                tuple_id=_id,
                                                )
                    # No match but error-correctionable, do correction:
                    elif len(_related_bits) == 1:
                        _spot_tuple = SpotTuple(_group_spots,
                                                bits=_group_spots.bits,
                                                pixel_sizes=_group_spots.pixel_sizes,
                                                spots_inds=np.array(_inds),
                                                tuple_id=_id,
                                                )
                    # append
                    #print(_spot_tuple.bits, _id, _inds, _spot_tuple.spots_inds)
                    _spot_groups.append(_spot_tuple)
                    _group_id += 1
                    _spot_group_inds.append(set(_spot_tuple.spots_inds))
                    del(_spot_tuple)
        
        # add to attribute
        self.spot_groups = _spot_groups
        if self.verbose:
            print(f"- {len(self.spot_groups)} spot groups selected. ")
        return _spot_groups
    
    @staticmethod
    def find_unused_spots(cand_spots, spot_groups):
        # init
        _spot_usage = np.zeros(len(cand_spots))
        # record spots_inds
        for _g in spot_groups:
            _spot_usage[_g.spots_inds] = _spot_usage[_g.spots_inds] + 1
        # unused spots
        _unused_spots = cand_spots[_spot_usage==0]
        return _unused_spots
    @staticmethod
    def collect_invalid_pairs(unused_spots, num_on_bits=3):
        # build kd-tree
        _tree = KDTree(unused_spots.to_positions())
        _nb_dists, _nb_inds = list(_tree.query(unused_spots.to_positions(), num_on_bits) )

    @staticmethod
    def collect_invalid_pairs(unused_spots, num_on_bits=3):
        # build kd-tree
        _tree = KDTree(unused_spots.to_positions())
        _nb_dists, _nb_inds = list(_tree.query(unused_spots.to_positions(), num_on_bits) )
        _invalid_pairs = [SpotTuple(unused_spots[_inds[:2]]) for _inds in _nb_inds]
        return _invalid_pairs


class DNA_Merfish_Decoder(Merfish_Decoder):
    """DNA MERFISH decoder, based on merfish decoder but allow some special features"""
    def __init__(self, 
                 cand_spots_list, 
                 codebook_filename,
                 bits=None,
                 pixel_sizes=default_pixel_sizes,
                 auto=True,
                 verbose=True,
                 ):
        super().__init__(cand_spots_list=cand_spots_list, codebook_filename=codebook_filename, bits=bits,
            pixel_sizes=pixel_sizes, auto=auto, verbose=verbose)

