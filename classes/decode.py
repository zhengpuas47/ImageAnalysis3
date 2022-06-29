import os, glob, sys, time
import pickle
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy
from itertools import combinations
from scipy import stats
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# local
from ..io_tools.spots import CellSpotsDf_2_CandSpots, SpotTuple_2_Dict, Dataframe_2_SpotGroups
from .preprocess import Spots3D, SpotTuple
from . import default_pixel_sizes, default_search_th,default_search_eps
from ..figure_tools.plot_decode import plot_spot_stats
from ..figure_tools.distmap import GenomeWide_DistMap

default_weights = np.array([1,1,1,1,1,])
default_metric_names = ['int. mean', 'int. CV', 'inner dist median', 'local dist mean', 'homolog dist']


class Merfish_Decoder():
    """Class to decode spot-based merfish results"""    

    def __init__(self, 
                 codebook_df,
                 cand_spots_df,
                 fov_id=None, cell_id=None,
                 bits=None,
                 savefile=None,
                 pixel_sizes=default_pixel_sizes,
                 inner_dist_factor=-1,
                 intensity_factor=1,
                 auto=True,
                 load_from_file=True,
                 overwrite=False,
                 verbose=True,
                 ):
        """Inttialize"""
        # parameters
        self.codebook_df = codebook_df
        self.fov_id = fov_id
        self.cell_id = cell_id
        self.pixel_sizes = np.array(pixel_sizes)
        self.inner_dist_factor = float(inner_dist_factor)
        self.intensity_factor = float(intensity_factor)
        self.process_parameters = {}
        # savefile
        self.savefile = savefile
        # other attributes
        self.verbose = verbose
        # cand_spots
        if not isinstance(cand_spots_df, pd.DataFrame):
            raise TypeError(f"Wrong input type for cand_spots_df: {type(cand_spots_df)}, should be DataFrame")
        self.cand_spots_df = cand_spots_df
        self.cand_spots = CellSpotsDf_2_CandSpots(self.cand_spots_df) # convert into cand_spots

        # update fov_id and cell_id if not given
        if self.fov_id is None:
            try:
                self.fov_id = np.unique(self.cand_spots_df['fov_id'])[0]
            except:
                pass
        if self.cell_id is None:
            try:
                self.cell_id = np.unique(self.cand_spots_df['cell_id'])[0]
            except:
                pass
        # load from savefile
        if load_from_file and os.path.isfile(self.savefile):
            self._load_basic()
        # define bits
        if bits is None:
            self.bits = np.unique(cand_spots_df['bit'].values)
        else:
            self.bits = np.array(bits, dtype=np.int32)
        # load codebook if automatic
        if auto:
            self._load_codebook()
            self._find_valid_pairs_in_codebook()
            self._find_valid_tuples_in_codebook()

    def _create_bit_2_channel(self, save_attr=True):
        """Create bit_2_channel dict"""
        # try create bit_2_channel if possible
        try:
            _bit_2_channel = {}
            for _bit in self.bits:
                _chs = np.unique(self.cand_spots_df.loc[self.cand_spots_df['bit']==_bit, 'channel'].values)
                if len(_chs) == 1:
                    _bit_2_channel[_bit] = str(_chs[0])
        except:
            _bit_2_channel = None
        if save_attr:
            # assign attribute
            self.bit_2_channel = _bit_2_channel
        return _bit_2_channel

    def _load_basic(self):
        if self.verbose:
            print(f"- Loading decoder info from file: {self.savefile}")
        # load basic attrs
        with h5py.File(self.savefile, 'r') as _f:
            self.fov_id = _f.attrs['fov_id']
            self.cell_id = _f.attrs['cell_id']
            self.pixel_sizes = _f.attrs['pixel_sizes']
            # spot_usage
            if 'spot_usage' in _f.keys():
                self.spot_usage = _f['spot_usage'][:].astype(np.int32)
        # cand_spots
        self.cand_spots_df = pd.read_hdf(self.savefile, 'cand_spots')
        self.cand_spots = CellSpotsDf_2_CandSpots(self.cand_spots_df) # convert into cand_spots
        # spot_groups
        _group_df = pd.read_hdf(self.savefile, 'spot_groups')
        self.spot_groups = Dataframe_2_SpotGroups(_group_df)
        # codebook
        self.codebook_df = pd.read_hdf(self.savefile, 'codebook')

    def _save_basic(self, _uid=None, _complevel=1, _complib='blosc:zstd',
        _overwrite=True):
        """Save all information into an hdf5 file (with overwrite)"""
        if self.verbose:
            print(f"- Saving decoder into file: {self.savefile}")
        # save basic attributes
        with h5py.File(self.savefile, 'a') as _f:
            _f.attrs['fov_id'] = self.fov_id
            _f.attrs['cell_id'] = self.cell_id
            _f.attrs['pixel_sizes'] = self.pixel_sizes
            _f.attrs['savefile'] = self.savefile
            _exist_groups = list(_f.keys())
        #print(_exist_groups, _overwrite)
        # save cand_spots_df
        if hasattr(self, 'cand_spots'):
            if self.verbose:
                print(f"-- save cand_spots into: {self.savefile}")
        self.cand_spots_df.to_hdf(self.savefile, 'cand_spots', complevel=_complevel, complib=_complib)
        # save spot_groups
        if hasattr(self, 'spot_groups'):
            if self.verbose:
                print(f"-- save spot_groups into: {self.savefile}")
            _bit_2_channel = self._create_bit_2_channel()
            _infoDict_list = [SpotTuple_2_Dict(_g, self.fov_id, self.cell_id, 
                                               _uid, sel_ind=getattr(_g, 'sel_ind',None), 
                                               bit_2_channel=_bit_2_channel, codebook=self.codebook_df) 
                            for _g in self.spot_groups]
            decoder_group_df = pd.DataFrame(_infoDict_list,)
            decoder_group_df.to_hdf(self.savefile, 'spot_groups', complevel=_complevel, complib=_complib)
        # save spot_usage
        if hasattr(self, 'spot_usage'):
            if self.verbose:
                print(f"-- save spot_groups into: {self.savefile}")
        with h5py.File(self.savefile, 'a') as _f:
            if _overwrite and 'spot_usage' in _f.keys():
                del(_f['spot_usage'])
            if 'spot_usage' not in _f.keys():
                _f.create_dataset('spot_usage', data=self.spot_usage)
        # save codebook
        self.codebook_df.to_hdf(self.savefile, 'codebook', complevel=_complevel, complib=_complib)

    def _load_codebook(self):
        codebook_df = getattr(self, 'codebook_df')
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
        self.codebook_matrix = codebook_df[self.bit_names].values

    def _find_valid_pairs_in_codebook(self):
        """Find valid 2-bit pairs given in the codebook """
        from itertools import combinations
       # valid_pair_bits = []
        #valid_pair_ids = []
        valid_bit_pair_2_region = {}
        for _icode, _code in enumerate(self.codebook_matrix):
            for _p in combinations(np.where(_code > 0)[0], 2):
                _bs = tuple(np.sort(self.bits[np.array(_p)]))
                if _bs not in valid_bit_pair_2_region:
                    #valid_pair_bits.append(_bs)
                    #valid_pair_ids.append(self.ids[_icode])
                    valid_bit_pair_2_region[_bs] = self.ids[_icode]
        # attributes
        #self.valid_pair_bits = valid_pair_bits
        #self.valid_pair_ids = valid_pair_ids
        self.valid_bit_pair_2_region = valid_bit_pair_2_region
        return self.valid_bit_pair_2_region
        #return valid_pair_bits, valid_pair_ids

    def _find_valid_tuples_in_codebook(self):
        valid_tuples = {}
        for _icode, _code in enumerate(self.codebook_matrix):
            _bs = tuple(np.sort(self.bits[np.where(_code>0)[0]]))
            if _bs not in valid_tuples:
                valid_tuples[self.ids[_icode]] = _bs
        # attribute
        self.valid_region_2_bits = valid_tuples
        return valid_tuples

    def _find_spot_pairs_in_radius(self, search_th=default_search_th, eps=default_search_eps, 
        keep_valid=True, overwrite=False):
        """Build a KD tree to find pairs given a radius threshold in nm"""
        if hasattr(self, 'pair_inds_list') and not overwrite:
            return self.pair_inds_list
        else:
            self.pair_inds_list = []
        # extract all coordinatesa
        _cand_positions = self.cand_spots.to_positions()
        # build kd-tree
        _tree = KDTree(_cand_positions)
        self.kdtree = _tree
        # find pairs
        self.search_th = search_th
        self.search_eps = eps
        _pair_inds_list = list(_tree.query_pairs(self.search_th, eps=self.search_eps))
        # loop through all pairs
        _kept_pair_inds_list = []
        for _inds in _pair_inds_list:
            # only keep the valid pairs
            if keep_valid: 
                _bts = tuple( np.sort(self.cand_spots.bits[np.array(_inds)]) )
                if _bts in self.valid_bit_pair_2_region:
                    self.pair_inds_list.append(_inds)
            # otherwise allow all possible pairs
            else:
                self.pair_inds_list.append(_inds)

        if self.verbose:
            print(f"- {len(self.pair_inds_list)} pairs kept given search radius {search_th} nm.")

        # generate pairs
        self.cand_pair_list = self.create_tuples_from_inds(self.cand_spots, self.pair_inds_list, self.valid_bit_pair_2_region)

        return self.pair_inds_list

    def select_spot_tuples_old(self, max_usage=1, keep_pairs=True,
        search_th=default_search_th, eps=default_search_eps, weights=np.ones(5),
        overwrite=False):
        """Function to select spot tuples given spot_pairs found previously"""
        # initialize _spot_usage and tuples
        _spot_usage = np.zeros(len(self.cand_spots))
        # check spot_groups
        if hasattr(self, 'spot_groups') and not overwrite:
            for _g in self.spot_groups:
                _spot_usage[_g.spots_inds] += 1
            if self.verbose:
                print(f"-- directly return {len(self.spot_groups)} spot_groups.")
            # save spot_usage
            setattr(self, 'spot_usage', _spot_usage)
            return self.spot_groups, self.spot_usage
        # initlaize otherwise
        else:
            if self.verbose:
                print(f"-- search spot_groups given search radius {search_th} nm, max_usage={max_usage}")
            self.spot_groups = [] 

        # decide whether generate pair_ind_list
        if not hasattr(self, 'pair_inds_list'):
            self._find_spot_pairs_in_radius(search_th=search_th, eps=eps)
        else:
            self.search_th = search_th
            self.search_eps = eps

        # create pair_list
        if not hasattr(self, 'cand_pair_list') or overwrite:
            _pair_list = self.create_tuples_from_inds(self.cand_spots, self.pair_inds_list, self.valid_bit_pair_2_region)
            setattr(self, 'cand_pair_list', _pair_list)
            if len(_pair_list) == 0:
                return 
        else:
            _pair_list = self.cand_pair_list
        # create pair_spot_usage
        _pair_spot_usage = np.zeros(len(self.cand_spots))
        for _pair in _pair_list:
            _pair.select = False
            _pair_spot_usage[_pair.spots_inds] += 1

        # generate scores for pair_list
        _pair_ref_metrics = generate_score_metrics(self.cand_pair_list,)
        _pair_ref_metrics = np.concatenate(_pair_ref_metrics, axis=0)
        _pair_scores,_ = generate_scores(self.cand_pair_list, _pair_ref_metrics,)
        _final_scores = summarize_score(self.cand_pair_list, weights=weights)
        
        # First iteration, try to merge other points, save if successful
        for _pair in tqdm(sorted(self.cand_pair_list, key=lambda _p:-_p.final_score)): # start from highest scored pairs
            # search neighborhood whether the 3rd point exist
            _nb_spot_inds = self.kdtree.query_ball_point(_pair.centroid_spot().to_positions()[0],
                                                        self.search_th, eps=self.search_eps)
            # skip for now if no neighboring spots detected
            if len(_nb_spot_inds) == 0:
                #print("--- no neighbors detected, skip")
                continue
            # skip if spots are used
            if (_spot_usage[_pair.spots_inds] >= max_usage).any():
                #print("--- spot used, skip")
                continue
            # extract bits
            _nb_spot_bits = self.cand_spots.bits[np.array(_nb_spot_inds)]
            
            # the region_id
            _reg_id = _pair.tuple_id
            # on-bit tuple for this
            _on_bits = self.valid_region_2_bits[_reg_id]
            # find related bits
            _related_bits = list(set(_on_bits).difference(set(_pair.bits)))
            
            #print(_nb_spot_bits, _related_bits)
            
            # generate tentative tuples
            _temp_tuples = []
            for _ind, _b in zip(_nb_spot_inds, _nb_spot_bits):
                if _b in _related_bits:
                    _merged_inds = np.concatenate([_pair.spots_inds, [_ind]])
                    # skip if the additional spot has been used.
                    #if (_spot_usage[_merged_inds] >= max_usage).any():
                    if (_spot_usage[_pair.spots_inds] >= max_usage).any():
                        continue
                    # assemble tentative tuples
                    _merged_bits = np.concatenate([_pair.bits, [_b]])
                    _merged_spots = Spots3D(np.concatenate([_pair.spots, self.cand_spots[_ind][np.newaxis,:]], axis=0),
                                            bits=_merged_bits,
                                            pixel_sizes=_pair.pixel_sizes,)
                    _merged_tuple = SpotTuple(_merged_spots,
                                            bits=_merged_bits,
                                            pixel_sizes=_pair.pixel_sizes,
                                            spots_inds=_merged_inds,
                                            tuple_id=_pair.tuple_id,
                                            )
                    _temp_tuples.append(_merged_tuple)
            # if temp_tuple exists, pick the best one
            if len(_temp_tuples) > 0:
                _temp_metrics = generate_score_metrics(_temp_tuples,)
                _temp_scores,_ = generate_scores(_temp_tuples, _pair_ref_metrics,)
                _temp_final_scores = summarize_score(_temp_tuples)
                _max_tp_ind, _max_ihomo = np.unravel_index(np.argmax(_temp_final_scores), np.shape(_temp_final_scores))
                # append the best match
                self.spot_groups.append( copy(_temp_tuples[_max_tp_ind]) )
                _spot_usage[_temp_tuples[_max_tp_ind].spots_inds] += 1
            
        ## second round of pair
        if keep_pairs:
            for _pair in tqdm(sorted(self.cand_pair_list, key=lambda _p:-_p.final_score)):
                # skip if spots are used
                if (_spot_usage[_pair.spots_inds] >= max_usage).any():
                    #print("--- spot used, skip")
                    continue
                # append the pair
                self.spot_groups.append(copy(_pair))
                _spot_usage[_pair.spots_inds] += 1 
    
        # keep basic_score_metric, remove score_metrics and scores
        for _g in self.spot_groups:
            delattr(_g, 'scores')
            delattr(_g, 'score_metrics')
            delattr(_g, 'final_score')
        if self.verbose:
            print(f"- {len(self.spot_groups)} spot_groups detected")
        # add select orders as attribute
        for _i, _g in enumerate(self.spot_groups):
            _g.sel_ind = _i
        # save spot_usage
        setattr(self, 'spot_usage', _spot_usage)
        return self.spot_groups, self.spot_usage

    def select_spot_tuples(self, max_usage=np.inf, 
        region_2_expect_num=None, 
        search_th=default_search_th, eps=default_search_eps, 
        weights=np.ones(5),
        overwrite=False):
        """Function to select spot tuples given spot_pairs found previously"""
        # initialize _spot_usage and tuples
        _spot_usage = np.zeros(len(self.cand_spots))
        # check spot_groups
        if hasattr(self, 'spot_groups') and not overwrite:
            for _g in self.spot_groups:
                _spot_usage[_g.spots_inds] += 1
            if self.verbose:
                print(f"- directly return {len(self.spot_groups)} spot_groups.")
            return self.spot_groups, _spot_usage
        # initlaize otherwise
        else:
            if self.verbose:
                print(f"- search spot_groups given search radius {search_th} nm, max_usage={max_usage}")
            self.spot_groups = [] 

        # decide whether generate pair_ind_list
        if not hasattr(self, 'pair_inds_list'):
            self._find_spot_pairs_in_radius(search_th=search_th, eps=eps)
        else:
            self.search_th = search_th
            self.search_eps = eps

        # create pair_list
        if not hasattr(self, 'cand_pair_list') or overwrite:
            _pair_list = self.create_tuples_from_inds(self.cand_spots, self.pair_inds_list, self.valid_bit_pair_2_region)
            setattr(self, 'cand_pair_list', _pair_list)
            if len(_pair_list) == 0:
                return 
        else:
            _pair_list = self.cand_pair_list
        # create pair_spot_usage
        _pair_spot_usage = np.zeros(len(self.cand_spots))
        for _pair in _pair_list:
            _pair.select = False
            _pair_spot_usage[_pair.spots_inds] += 1
        # generate scores for pair_list
        _pair_ref_metrics = generate_score_metrics(self.cand_pair_list,)
        _pair_ref_metrics = np.concatenate(_pair_ref_metrics, axis=0)
        _pair_scores,_ = generate_scores(self.cand_pair_list, _pair_ref_metrics,)
        _final_scores = summarize_score(self.cand_pair_list, weights=weights)
        
        # first iteration, try to save non-overlapping spot_pairs without overlapping from high-scores to low-scores
        if self.verbose:
            print(f"-- select unique pairs.")
        for _pair in sorted(self.cand_pair_list, key=lambda _p: -1*_p.final_score):
            # skip if spots are used 
            if (_spot_usage[_pair.spots_inds] > 0).any():
                #print("--- spot used, skip")
                continue
            # append the pair
            _pair.select = True
            self.spot_groups.append( copy(_pair) )
            _spot_usage[_pair.spots_inds] += 1 
        # second iteration
        if isinstance(region_2_expect_num, dict):
            # loop through regions to search for possible candidates
            all_region_ids = np.unique([_p.tuple_id for _p in self.cand_pair_list])
            if self.verbose:
                print(f"-- search minimal candidates for {len(all_region_ids)} regions. ")
            for _rid in all_region_ids:
                _region_cand_pairs = [_p for _p in self.cand_pair_list if _p.tuple_id == _rid]
                _num_selected = np.sum([_p.select for _p in _region_cand_pairs ])
                _region_expect_num = region_2_expect_num.get(_rid, 0) # dont add if not specified in region_2_expect_num
                # if enough spot_groups already selected, skip
                if _num_selected >= _region_expect_num:
                    continue
                # append pairs given number
                else:
                    #print('not enough spots', _rid, _num_selected)
                    for _pair in sorted(_region_cand_pairs, key=lambda _p: -1*_p.final_score):
                        #print("***", _rid, _num_selected, _region_expect_num)
                        if _num_selected >= _region_expect_num:
                            break
                        # skip if this pair selected already
                        if _pair.select == True:
                            continue
                        # append the pair
                        _pair.select = True
                        self.spot_groups.append( copy(_pair) )
                        _spot_usage[_pair.spots_inds] += 1 
                        _num_selected += 1
        
        # third iteration: for kept pairs, try to merge the third region
        if self.verbose:
            print(f"-- upgrade pairs.")
        for _ipair, _pair in tqdm(enumerate(self.spot_groups)):
            # search neighborhood whether the 3rd point exist
            _nb_spot_inds = self.kdtree.query_ball_point(_pair.centroid_spot().to_positions()[0],
                                                        self.search_th, eps=self.search_eps)
            # skip for now if no neighboring spots detected
            if len(_nb_spot_inds) == 0:
                #print("--- no neighbors detected, skip")
                continue
            # skip if spots are used
            if (_spot_usage[_pair.spots_inds] >= max_usage).any():
                #print("--- spot used, skip")
                continue
            # extract bits
            _nb_spot_bits = self.cand_spots.bits[np.array(_nb_spot_inds)]
            # the region_id
            _reg_id = _pair.tuple_id
            # on-bit tuple for this
            _on_bits = self.valid_region_2_bits[_reg_id]
            # find related bits
            _related_bits = list(set(_on_bits).difference(set(_pair.bits)))            
            # generate tentative tuples
            _temp_tuples = []
            for _ind, _b in zip(_nb_spot_inds, _nb_spot_bits):
                if _b in _related_bits:
                    _merged_inds = np.concatenate([_pair.spots_inds, [_ind]])
                    #print("**", _pair_spot_usage[_ind])
                    # skip if the additional spot has been used
                    if (_spot_usage[_merged_inds] >= max_usage).any():
                        continue
                    # assemble tentative tuples
                    _merged_bits = np.concatenate([_pair.bits, [_b]])
                    _merged_spots = Spots3D(np.concatenate([_pair.spots, self.cand_spots[_ind][np.newaxis,:]], axis=0),
                                            bits=_merged_bits,
                                            pixel_sizes=_pair.pixel_sizes,)
                    _merged_tuple = SpotTuple(_merged_spots,
                                            bits=_merged_bits,
                                            pixel_sizes=_pair.pixel_sizes,
                                            spots_inds=_merged_inds,
                                            tuple_id=_pair.tuple_id,
                                            )
                    _temp_tuples.append(_merged_tuple)
            # if temp_tuple exists, pick the best one
            if len(_temp_tuples) > 0:
                _temp_metrics = generate_score_metrics(_temp_tuples,)
                _temp_scores,_ = generate_scores(_temp_tuples, _pair_ref_metrics,)
                _temp_final_scores = summarize_score(_temp_tuples)
                _max_tp_ind, _max_ihomo = np.unravel_index(np.argmax(_temp_final_scores), np.shape(_temp_final_scores))
                # append the best match
                self.spot_groups[_ipair] = copy( _temp_tuples[_max_tp_ind] )
                # modify spot_usage
                _spot_usage[_pair.spots_inds] -= 1
                _spot_usage[_temp_tuples[_max_tp_ind].spots_inds] += 1
                print("**", _ipair)
        self.spot_usage = _spot_usage

        # keep basic_score_metric, remove score_metrics and scores
        for _g in self.spot_groups:
            #delattr(_g, 'scores')
            #delattr(_g, 'score_metrics')
            #delattr(_g, 'final_score')
            ##DEBUG
            pass 

        if self.verbose:
            print(f"- {len(self.spot_groups)} spot_groups detected")
        return self.spot_groups, _spot_usage



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
            for _id, _tuple in self.valid_region_2_bits.items():
                if set(_group_spots.bits).issubset(set(_tuple)):
                    _related_bits = list(set(_tuple).difference(set(_group_spots.bits)))
                    # for this species, try to merge spots
                    _target_bits = self.cand_spots.bits[np.array(_targets)]
                    print(_target_bits, _related_bits)
                    _kept_targets = np.array([_t for _t, _b in zip(_targets,_target_bits) if _b in _related_bits])
                    # if match found, add
                    if len(_kept_targets) > 0:
                        print("**",len(_kept_targets))
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
                                                bits=np.concatenate([_group_spots.bits, np.array(_related_bits)])[np.argsort(_merged_inds)],
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
    def create_tuples_from_inds(cand_spots, pair_inds_list, valid_bit_pair_2_region,):
        """Convert indices list with cand_spots into list of SpotTuple"""
        _spotpair_list = []
        _pair_list = []
        for _inds in pair_inds_list:
            _inds = np.array(_inds)
            
            _spots = cand_spots[_inds]
            _spots.bits = cand_spots.bits[_inds]
            _spots.pixel_sizes = cand_spots.pixel_sizes
            # append
            _spotpair_list.append(_spots)
            
            _reg_id = valid_bit_pair_2_region[tuple(_spots.bits)]
            
            _pair = SpotTuple(_spots,
                            bits=_spots.bits,
                            pixel_sizes=_spots.pixel_sizes,
                            spots_inds=_inds,
                            tuple_id=_reg_id,
                            )
            _pair_list.append(_pair)
        
        return _pair_list

    @staticmethod
    def find_seeding_groups(cand_spots, spot_groups, num_cand_pre_region=2):
        # init
        _spot_usage = np.zeros(len(cand_spots))
        # record spots_inds
        for _g in spot_groups:
            _spot_usage[_g.spots_inds] = _spot_usage[_g.spots_inds] + 1
        
        seeding_groups = []
        
        for _g in spot_groups:
            if (_spot_usage[_g.spots_inds] <= num_cand_pre_region).all():
                seeding_groups.append(_g)
        return seeding_groups

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
        _invalid_pairs = [SpotTuple(unused_spots[_inds[:2]]) for _inds in _nb_inds]
        return _invalid_pairs
    @staticmethod
    def collect_invalid_pair_bits(all_bits, valid_bit_pair_2_region, num_on_bits=2):
        from itertools import combinations
        _invalid_pairs = []
        for _c in combinations(all_bits, num_on_bits):
            if _c not in valid_bit_pair_2_region:
                _invalid_pairs.append(_c)
        
        return _invalid_pairs

    @staticmethod
    def generate_reference(spot_groups,
                        intensity_metric='mean',
                        dist_metric='min',):
        # inteisities
        _ints = [getattr(np, intensity_metric)(_g.spots.to_intensities()) for _g in spot_groups]
        # internal dists
        _inter_dists = [getattr(np, dist_metric)(_g.dist_internal()) for _g in spot_groups]
        return _ints, _inter_dists


class DNA_Merfish_Decoder(Merfish_Decoder):
    """DNA MERFISH decoder, based on merfish decoder but allow some special features"""
    def __init__(self, 
                 codebook_df,
                 cand_spots_df,
                 fov_id=None, cell_id=None,
                 bits=None,
                 savefile=None,
                 pixel_sizes=default_pixel_sizes,
                 inner_dist_factor=-1,
                 intensity_factor=1,
                 ct_dist_factor=3.5,
                 local_dist_factor=0.5,
                 valid_score_th=-10,
                 metric_names=default_metric_names,
                 metric_weights=np.array([1,1,1,1,1]),
                 chr_2_copy_num=None, male=True,
                 auto=True, load_from_file=True,
                 verbose=True,
                 ):
        super().__init__(codebook_df=codebook_df, cand_spots_df=cand_spots_df, 
            fov_id=fov_id, cell_id=cell_id, bits=bits,
            savefile=savefile,
            pixel_sizes=pixel_sizes, 
            inner_dist_factor=inner_dist_factor, intensity_factor=intensity_factor,
            auto=auto, load_from_file=load_from_file, verbose=verbose)
        if self.verbose:
            print(f"Creating DNA_Merfish_Decoder class.")
        # extra parameters
        self.ct_dist_factor = float(ct_dist_factor)
        self.local_dist_factor = float(local_dist_factor)
        self.valid_score_th = float(valid_score_th)
        ##NEW scoring metric information
        self.metric_names = metric_names
        self.metric_weights = metric_weights
        ##NEW number of expected copy number for each chr
        if not isinstance(chr_2_copy_num, dict):
            self.chr_2_copy_num = generate_default_chr_2_copy_num(self.codebook, male=male)
        else:
            self.chr_2_copy_num = chr_2_copy_num
        # generate
        self.region_2_expect_num = self.generate_region_2_copy_num(self.codebook, self.chr_2_copy_num, )
        
        ## Load DNA decode results if applicable
        if load_from_file and os.path.isfile(self.savefile):
            _chr_2_zxys_list = self._load_picked_results()

    def _save_picked_results(self, _complevel=1, _complib='blosc:zstd', _overwrite=False):
        """Save chr_2_zxys_list into hdf5 group"""
        # save chr_2_spot_tuple
        if hasattr(self, 'chr_2_assigned_tuple_list'): 
            if self.verbose:
                print(f"- Save chr_2_assigned_tuple_list into file: {self.savefile}")
            # create bit_2_channel
            _bit_2_channel = self._create_bit_2_channel(save_attr=False)
            for _chr, _groups_list in getattr(self, 'chr_2_assigned_tuple_list',{}).items():
                for _homolog, _groups in enumerate(_groups_list):
                    # check existence
                    _key = f"chr_2_assigned_tuple_list/{_chr}/{_homolog}"
                    with h5py.File(self.savefile, 'a') as _f: 
                        if _key in _f and _overwrite:
                            del(_f[_key])
                        _write_homolog = (_key not in _f)
                    if _write_homolog:
                        # convert to DataFrame
                        _df = [SpotTuple_2_Dict(
                            _g, self.fov_id, self.cell_id, 
                            getattr(self,'uid',None), _homolog,
                            sel_ind=getattr(_g, 'sel_ind',None), 
                            bit_2_channel=_bit_2_channel, codebook=self.codebook_df)
                            for _g in _groups]
                        _df = pd.DataFrame(_df, )
                        #print(_chr, _homolog, len(_df))
                        # write
                        if self.verbose:
                            print(f"-- save tuples for :{_key}")
                        _df.to_hdf(self.savefile, _key, complevel=_complevel, complib=_complib)
                    else:
                        if self.verbose:
                            print(f"-- skip :{_key}")
        # save chr_2_indices_list
        self.chr_2_indices_list = extract_group_indices(self.chr_2_assigned_tuple_list)
        save_hdf5_dict(self.savefile, 'chr_2_indices_list', self.chr_2_indices_list,
            _overwrite=_overwrite, _verbose=self.verbose)
        # save chr_2_zxys_list
        save_hdf5_dict(self.savefile, 'chr_2_zxys_list', self.chr_2_zxys_list,
            _overwrite=_overwrite, _verbose=self.verbose)
        # save chr_2_chr_centers
        save_hdf5_dict(self.savefile, 'chr_2_chr_centers', self.chr_2_chr_centers,
            _overwrite=_overwrite, _verbose=self.verbose)     
        return

    def _load_picked_results(self, _save_attr=True):
        """Load decode-picking results"""
        with h5py.File(self.savefile, 'r') as _f:
            ## chr_2_assigned_tuple_list
            if self.verbose:
                print(f"- Load chr_2_assigned_tuple_list from file: {self.savefile}")
            _chr_2_assigned_tuple_list = {}
            if 'chr_2_assigned_tuple_list' in _f.keys():
                _grp = _f['chr_2_assigned_tuple_list']
                for _chr in _grp.keys():
                    _chr_group = _grp[_chr]
                    _chr_tuple_list = []
                    for _homolog in _chr_group.keys():
                        _key = f"chr_2_assigned_tuple_list/{_chr}/{_homolog}"
                        # load
                        _group_df = pd.read_hdf(self.savefile, _key, )
                        _chr_tuple_list.append(Dataframe_2_SpotGroups(_group_df))
                    # append chr
                    _chr_2_assigned_tuple_list[_chr] = _chr_tuple_list
                if self.verbose:
                    print(f"-- {len(_chr_2_assigned_tuple_list)} chromosome loaded.")
                # save attributes
                if _save_attr and len(_chr_2_assigned_tuple_list) > 0:
                    setattr(self, 'chr_2_assigned_tuple_list', _chr_2_assigned_tuple_list)
            ## chr_2_zxys_list
            if self.verbose:
                print(f"- Load chr_2_zxys_list from file: {self.savefile}")
            _chr_2_zxys_list = {}
            if 'chr_2_zxys_list' in _f.keys():
                _grp = _f['chr_2_zxys_list']
                for _chr in _grp.keys():
                    _chr_2_zxys_list[_chr] = _grp[_chr][:].astype(np.float32)
                if self.verbose:
                    print(f"-- {len(_chr_2_zxys_list)} chromosome loaded.")
                # save attributes
                if _save_attr and len(_chr_2_zxys_list) > 0:
                    setattr(self, 'chr_2_zxys_list', _chr_2_zxys_list)
            ## chr_2_chr_centers
            if self.verbose:
                print(f"- Load chr_2_chr_centers from file: {self.savefile}")
            _chr_2_chr_centers = {}
            if 'chr_2_chr_centers' in _f.keys():
                _grp = _f['chr_2_chr_centers']
                for _chr in _grp.keys():
                    _chr_2_chr_centers[_chr] = _grp[_chr][:].astype(np.float32)
                if self.verbose:
                    print(f"-- {len(_chr_2_chr_centers)} chromosome loaded.")
                # save attributes
                if _save_attr and len(_chr_2_chr_centers) > 0:
                    setattr(self, 'chr_2_chr_centers', _chr_2_chr_centers)  
                    
        return _chr_2_assigned_tuple_list, _chr_2_zxys_list, _chr_2_chr_centers


    def prepare_spot_tuples(self, pair_search_radius=200, 
        max_spot_usage=1, force_search_for_region=True,
        min_chr_spot_num=2, 
        generate_invalid_groups=True, invalid_num=500,
        generate_chr_trees=True, 
        overwrite=False):
        """ NEW
        Collect spot_groups
        Add chromosomal information
        Split spot_groups into chromosomes
        Generate invalid randomized control
        Generate chromosomal kdtree
         """
        ## If everything exist, skip the entire process.
        if hasattr(self, 'spot_groups') \
            and hasattr(self, 'chr_2_groups') \
            and hasattr(self, 'chr_2_invalid_groups') \
            and hasattr(self, 'chr_2_kdtree') \
            and not overwrite:
            print("spot_groups already exists.")
            return 

        # required params
        if force_search_for_region:
            _region_2_expect_num = getattr(self, 'region_2_expect_num', None)
        else:
            _region_2_expect_num = None
        # Generate spot_groups
        if hasattr(self, 'spot_groups') and hasattr(self, 'spot_usage') and len(getattr(self, 'spot_groups')) > 0:
            if self.verbose:
                print(f"- spot_groups and spot_usage already exist, skip.")
        else:
            # find pairs
            self._find_spot_pairs_in_radius(search_th=pair_search_radius, overwrite=overwrite)
            # update chromosome info for cand_pair_list
            self.update_spot_groups_chr_info(self.cand_pair_list, self.codebook)
            # assemble tuples
            spot_groups, spot_usage = self.select_spot_tuples_old(max_usage=max_spot_usage, 
                search_th=pair_search_radius, 
                #region_2_expect_num=_region_2_expect_num, 
                weights=self.metric_weights,
                overwrite=overwrite)
            # update chromosome info
            self.update_spot_groups_chr_info(self.spot_groups, self.codebook)
        # update used parameters


        #for _ig, _g in enumerate(self.spot_groups):
        #    _g.chr = self.codebook.loc[self.codebook['id']==_g.tuple_id, 'chr'].values[0]
        #    _g.chr_order = self.codebook.loc[self.codebook['id']==_g.tuple_id, 'chr_order'].values[0]
        # split tuples into chromosome
        if self.verbose:
            print("- split found tuples into chromosomes. ")
        self.chr_2_groups = {}
        self.chr_2_spot_inds = {}
        for _chr_name in np.unique(self.codebook['chr'].values):
            _chr_tuples = [_g for _g in self.spot_groups if _g.chr == _chr_name]
            _chr_spot_inds = [_g.spots_inds for _g in self.spot_groups if _g.chr == _chr_name]
            if len(_chr_tuples) > min_chr_spot_num:
                self.chr_2_groups[_chr_name] = _chr_tuples
                self.chr_2_spot_inds[_chr_name] = np.unique(np.concatenate(_chr_spot_inds))
        # use chr_2_spot_inds to generate random spots
        if generate_invalid_groups:
            if self.verbose:
                print("- generate randomized spot pairs ")
            # generate invalid pair_bits
            _invalid_pair_bits = self.collect_invalid_pair_bits(self.bits, self.valid_bit_pair_2_region)

            self.chr_2_invalid_groups = {}
            for _chr_name, _spot_inds in self.chr_2_spot_inds.items():
                # select chromosomal spots 
                _chr_cand_spots = self.cand_spots[_spot_inds]
                _chr_cand_spots.bits = self.cand_spots.bits[_spot_inds]
                _chr_cand_spots.pixel_sizes = self.cand_spots.pixel_sizes
                # generate invalid groupd
                _invalid_groups = self.generate_random_invalid_pairs(_chr_cand_spots, 
                    _invalid_pair_bits, total_num=invalid_num, )
                if len(_invalid_groups) > min_chr_spot_num:
                    self.chr_2_invalid_groups[_chr_name] = _invalid_groups
        
        # generate chr_trees
        if generate_chr_trees:
            if self.verbose:
                print("- generate chr_2_kdtree. ")
            self.chr_2_kdtree = self.prepare_chr_trees(self.chr_2_groups)


    def init_homolog_centers(self, method="BB", overwrite=False):
        """NEW Initialize homologous chromosome centers by the method from BB
        has to be 2_fixed
        
        """
        if hasattr(self, 'chr_2_homolog_centers') and not overwrite:
            if self.verbose:
                print(f"- directly return chr_2_homolog_centers")
            return self.chr_2_homolog_centers

        if method == 'BB':
            # chr_2_init_centers
            self.chr_2_homolog_centers = {}
            for _chr_name in self.chr_2_groups:
                #print(_chr_name)
                _chr_zxys = np.array([_g.centroid_spot().to_positions()[0] for _g in self.chr_2_groups[_chr_name]])
                _chr_inds = np.array([_g.chr_order for _g in self.chr_2_groups[_chr_name]])
                _centers = np.array(init_homolog_centers_BB(_chr_zxys, _chr_inds))
                self.chr_2_homolog_centers[_chr_name] = _centers


        return self.chr_2_homolog_centers
    
    ##NEW
    def iterative_assign_spot_groups_2_homologs(self, max_num_iter=10, score_th_percentile=1,
                                                flag_diff_th=0.005, make_plots=False,
                                                ):
        """NEW"""
        # iniitialize 
        self.chr_2_zxys_list = {}
        self.chr_2_homolog_flags = {_chr_name:-1 * np.ones(len(_gs),dtype=np.int32) 
                                    for _chr_name, _gs in self.chr_2_groups.items()}
        self.chr_2_flag_diff = {_chr_name:1 for _chr_name in self.chr_2_groups}
        total_flag_diff = np.sum([len(_gs)*self.chr_2_flag_diff[_chr_name]
                                for _chr_name, _gs in self.chr_2_groups.items()]) \
                            / np.sum([len(_gs) for _chr_name, _gs in self.chr_2_groups.items()])
        # metrics
        self.chr_2_score_metrics = {_chr_name:collect_metrics(_gs) 
                                        for _chr_name, _gs in self.chr_2_groups.items()}
        # loop through chromosomes
        _n_iter = 0
        while total_flag_diff >= flag_diff_th: 
            if _n_iter > max_num_iter:
                break
                
            for _chr_name, _chr_groups in self.chr_2_groups.items():

                if self.chr_2_flag_diff[_chr_name] > 0:
                    # get required inputs
                    _homolog_centers = self.chr_2_homolog_centers.get(_chr_name)
                    _invalid_chr_groups = self.chr_2_invalid_groups.get(_chr_name, None)
                    _chr_region_ids = self.extract_chr_region_ids(self.codebook, _chr_name)
                    # assign by calling function
                    _new_zxys_list, _new_homolog_flags, _new_homolog_centers = \
                        self.assign_spot_groups_2_homologs(_chr_groups, _homolog_centers, 
                            _chr_region_ids, _invalid_chr_groups=_invalid_chr_groups,
                            weights=self.metric_weights, 
                            score_th_percentile=score_th_percentile,
                            make_plots=make_plots,)
                    # compare diff
                    _flag_diff = self.compare_flags_diff(self.chr_2_homolog_flags[_chr_name],
                                                                _new_homolog_flags)
                    print(f"- chr:{_chr_name}, diff={_flag_diff:.4f}")
                    # update
                    self.chr_2_flag_diff[_chr_name] = _flag_diff
                    self.chr_2_zxys_list[_chr_name] = _new_zxys_list
                    self.chr_2_homolog_flags[_chr_name] = _new_homolog_flags
                    self.chr_2_homolog_centers[_chr_name] = _new_homolog_centers
                    # regenerate homolog_based_trees
                    _new_homolog_trees = []
                    for _zxys in _new_zxys_list:
                        if np.sum((np.isnan(_zxys)==0).any(1)) == 0:
                            _new_homolog_trees.append(self.chr_2_kdtree[_chr_name])
                        else:
                            _new_homolog_trees.append( KDTree(_zxys[(np.isnan(_zxys)==0).any(1)]) )
                    # re-calculate metrics
                    self.chr_2_score_metrics[_chr_name] = \
                        generate_score_metrics(_chr_groups, _new_homolog_trees,
                            _new_homolog_centers, 
                            update_attr=True, overwrite=True).reshape(-1, self.chr_2_score_metrics[_chr_name].shape[-1])

            # collect updated metrics
            _total_score_metrics = np.concatenate(list(self.chr_2_score_metrics.values()),
                                                axis=0)
            
            print('mean_metrics', _total_score_metrics.mean(0))
            # re-calculate scores
            _chr_2_scores = self.calculate_scores(_total_score_metrics, overwrite=True)
            # re-calculate total_diff
            total_flag_diff = np.sum([len(_gs)*self.chr_2_flag_diff[_chr_name]
                            for _chr_name, _gs in self.chr_2_groups.items()]) \
                        / np.sum([len(_gs) for _chr_name, _gs in self.chr_2_groups.items()])
            print(f"- iter:{_n_iter}, total_diff={total_flag_diff}")
            _n_iter += 1


    ## new
    def calculate_score_metrics(self, _chr_2_kdtree=None, 
        _chr_2_homolog_centers=None, _n_neighbors=10, overwrite=False):
        """ NEW
        Calculate score_metrics, given"""
        if _chr_2_kdtree is None:
            _chr_2_kdtree = getattr(self, 'chr_2_kdtree', {_c:None for _c in self.chr_2_groups})
        if _chr_2_homolog_centers is None:
            _chr_2_homolog_centers = getattr(self, 'chr_2_homolog_centers', {_c:None for _c in self.chr_2_groups})
        # calculate score_metrics within each chromosome
        _valid_metrics_list = []
        for _chr_name in self.chr_2_groups:
            if self.verbose:
                print(f"-- generate scoring metrics for chr:{_chr_name}")
            generate_score_metrics(
                self.chr_2_groups[_chr_name], 
                _chr_2_kdtree[_chr_name], 
                _chr_2_homolog_centers[_chr_name], 
                n_neighbors=_n_neighbors, update_attr=True,
                overwrite=overwrite,
                )
            _valid_metrics_list.append(collect_metrics(self.chr_2_groups[_chr_name]))
            # invalid
            if hasattr(self, 'chr_2_invalid_groups') and _chr_name in self.chr_2_invalid_groups:
                generate_score_metrics(
                    self.chr_2_invalid_groups[_chr_name], 
                    _chr_2_kdtree[_chr_name], 
                    _chr_2_homolog_centers[_chr_name], 
                    n_neighbors=_n_neighbors, update_attr=True,
                    overwrite=overwrite)
        _valid_metrics_list = np.concatenate(_valid_metrics_list, axis=0)
        return _valid_metrics_list


    def calculate_scores(self, _valid_metrics_list=None, overwrite=False):
        """ NEW
        Calculate scores based on metrics_list from calculate_score_metrics"""
        # _valid_metrics_list
        if _valid_metrics_list is None:
            _valid_metrics_list = self.calculate_score_metrics()
        # calculate scores
        chr_2_valid_scores = {}
        if hasattr(self, 'chr_2_invalid_groups'):
            chr_2_invalid_scores = {}
        else:
            chr_2_invalid_scores = None
        for _chr_name in self.chr_2_groups:
            if self.verbose:
                print(f"-- scoring spot_groups for chr:{_chr_name}")
            chr_2_valid_scores[_chr_name], _ = generate_scores(
                self.chr_2_groups[_chr_name], 
                _valid_metrics_list, update_attr=True)
            # invalid
            if hasattr(self, 'chr_2_invalid_groups') and _chr_name in self.chr_2_invalid_groups:
                chr_2_invalid_scores[_chr_name], _ = generate_scores(
                    self.chr_2_invalid_groups[_chr_name], 
                    _valid_metrics_list, update_attr=True)
        # add to attr
        self.chr_2_valid_scores = chr_2_valid_scores
        self.chr_2_invalid_scores = chr_2_invalid_scores
        # return 
        return chr_2_valid_scores, chr_2_invalid_scores

    ## OLD
    def calculate_self_scores(self, use_invalid_control=True, make_plots=True, overwrite=False):
        from ..spot_tools.scoring import generate_cdf_scores

        if np.sum([hasattr(_g, 'self_score') for _g in self.spot_groups]) == len(self.spot_groups) and not overwrite:
            return np.array([getattr(_g, 'self_score') for _g in self.spot_groups])

        # generate metrics
        valid_ints, valid_inner_dists = self.generate_reference(self.spot_groups)
        if use_invalid_control:
            unused_spots = self.find_unused_spots(self.cand_spots, self.spot_groups)
            invalid_pairs = self.collect_invalid_pairs(unused_spots)
            # generate metrics for negative
            invalid_ints, invalid_inner_dists = self.generate_reference(invalid_pairs)
        else:
            invalid_ints, invalid_inner_dists = None, None
        # scores
        inner_dist_scores = generate_cdf_scores(valid_inner_dists, valid_inner_dists, invalid_inner_dists)
        int_scores = generate_cdf_scores(valid_ints, valid_ints, invalid_ints)
        tuple_self_scores = self.inner_dist_factor * inner_dist_scores + self.intensity_factor * int_scores
        # update tuples
        for _ig, _g in enumerate(self.spot_groups):
            _g.self_score = tuple_self_scores[_ig]
        if make_plots:
            plt.figure()
            plt.scatter(valid_ints, tuple_self_scores)
            plt.show()
            plt.figure()
            plt.scatter(valid_inner_dists, tuple_self_scores)
            plt.show()

        return tuple_self_scores

    def init_homolog_assignment(self, min_cand_number=10, num_homologs=2, 
                                overwrite=False, verbose=True):
        """Use function initial_assign_homologs_by_chr to identify chromosome centers"""

        if not overwrite and hasattr(self, 'chrs_2_init_centers'):
            return getattr(self, 'chrs_2_init_centers')
        
        # initlaize dict
        chrs_2_init_centers = {}
        #chrs_2_init_zxys_list = {}
        # identify isolated spot_groups as seeds
        seeding_groups = self.find_seeding_groups(self.cand_spots, self.spot_groups,)
        # loop through all chromosomes:
        for _chr_name in np.unique(self.codebook['chr']):

            _chr_region_ids = self.extract_chr_region_ids(self.codebook, _chr_name)
            
            # extract seeding chromosome spots
            _chr_seeding_groups = [_g for _g in seeding_groups if _g.chr == _chr_name]
            if len(_chr_seeding_groups) < min_cand_number * num_homologs:
                _chr_seeding_groups = [_g for _g in self.spot_groups if _g.chr == _chr_name]
                if verbose:
                    print(f"-- seeding chr:{_chr_name} with {len(_chr_region_ids)} regions with *{len(_chr_seeding_groups)} spot_groups")
            else:
                if verbose:
                    print(f"-- seeding chr:{_chr_name} with {len(_chr_region_ids)} regions with {len(_chr_seeding_groups)} spot_groups")
            if len(_chr_seeding_groups) < min_cand_number * num_homologs:
                continue
            # get num of homologs
            _num_homologs = self.chr_2_copy_num[_chr_name]
            # run initialize decoding
            init_zxys_list, init_homolog_labels, init_homolog_centers =\
                DNA_Merfish_Decoder.initial_assign_homologs_by_chr(_chr_seeding_groups, _chr_region_ids, _num_homologs)
            # save info
            chrs_2_init_centers[_chr_name] = init_homolog_centers
            #chrs_2_init_zxys_list[_chr_name] = init_zxys_list
        
        # add attribute
        setattr(self, 'chrs_2_init_centers', chrs_2_init_centers)
        
        return chrs_2_init_centers

    def finish_homolog_assignment(self, allow_overlap=False, max_n_iter=15, plot_stats=False, 
                                  overwrite=False, verbose=False):
        """chr_2_assigned_tuple_list stores all assigned spot_groups despite valid_score_th, 
            wihch allows further thresholding"""

        if not overwrite and hasattr(self, 'chr_2_assigned_tuple_list')\
            and hasattr(self, 'chr_2_zxys_list') \
            and hasattr(self, 'chr_2_chr_centers'):
            if self.verbose:
                print("- Directly return assigned homolog results.")
            return getattr(self, 'chr_2_assigned_tuple_list'),\
                getattr(self, 'chr_2_zxys_list'), \
                getattr(self, 'chr_2_chr_centers')

        chr_2_assigned_tuple_list = {}
        chr_2_zxys_list = {}
        chr_2_chr_centers = {}
        # initialize if not exist
        if not hasattr(self, 'chrs_2_init_centers'):
            self.init_homolog_assignment(overwrite=overwrite)

        for _chr_name, _initial_centers in self.chrs_2_init_centers.items():
            if verbose:
                print(f"- processing chr:{_chr_name}")
            # assign
            _chr_groups = [_g for _g in self.spot_groups if _g.chr == _chr_name]
            _chr_region_ids = self.extract_chr_region_ids(self.codebook, _chr_name)
            _zxys_list_steps, _flags_steps, _assigned_tuple_list = \
                DNA_Merfish_Decoder.assign_homologs_by_chr(
                    _chr_groups, _initial_centers, _chr_region_ids,
                    _init_homolog_flags=None,_init_homolog_zxys_list=None,
                    allow_overlap=allow_overlap,
                    ct_dist_f=self.ct_dist_factor, local_dist_f=self.local_dist_factor,
                    valid_score_th=self.valid_score_th,
                    max_n_iter=max_n_iter,
                    plot_stats=plot_stats,
                    verbose=verbose,
                )
            # update center
            _chr_centers = DNA_Merfish_Decoder.calculate_homolog_centroids(_chr_groups, _flags_steps[-1]) 

            chr_2_assigned_tuple_list[_chr_name] = _assigned_tuple_list
            chr_2_zxys_list[_chr_name] = _zxys_list_steps[-1]
            chr_2_chr_centers[_chr_name] = _chr_centers
            
        # add attribute
        setattr(self, 'chr_2_assigned_tuple_list', chr_2_assigned_tuple_list)
        setattr(self, 'chr_2_zxys_list', chr_2_zxys_list)
        setattr(self, 'chr_2_chr_centers', chr_2_chr_centers)
        
        return chr_2_assigned_tuple_list, chr_2_zxys_list, chr_2_chr_centers

    @staticmethod
    def summarize_zxys_all_chromosomes(chr_2_zxys_list, codebook, 
                                       num_homologs=2, keep_valid=True, order_by_chr=True):
        """Summarization of ."""
        # summarize chromosome information from codebook
        chr_2_indices = {_chr:np.array(codebook.loc[codebook['chr']==_chr].index)
                        for _chr in np.unique(codebook['chr'].values)}

        _ordered_chr_names = []
        for _chr_name, _chr_reg_id in zip(codebook['chr'], codebook['chr_order']):
            if _chr_name not in _ordered_chr_names:
                _ordered_chr_names.append(_chr_name)
        
        # assemble sorted-zxys for each homolog
        total_zxys_dict = [{_rid:np.ones(3)*np.nan 
                            for _rid in codebook.index} for _ihomo in np.arange(num_homologs)]
        for _chr_name, _zxys_list in chr_2_zxys_list.items():
            for _ihomo, _zxys in enumerate(_zxys_list):
                _chr_region_ids = DNA_Merfish_Decoder.extract_chr_region_ids(codebook, _chr_name)
                for _i_chr_reg, _zxy in zip(_chr_region_ids, _zxys):
                    #print(_chr_name, _ihomo, _i_chr_reg)
                    _rid = codebook.loc[(codebook['chr']==_chr_name) \
                                              & (codebook['chr_order']==_i_chr_reg)].index[0]
                    total_zxys_dict[_ihomo][_rid] = _zxy
                
        # convert to zxy coordinates in um
        homolog_zxys = [np.array(list(_dict.values()))/1000 for _dict in total_zxys_dict]
        # initalize
        _curr_pos = 0
        figure_zxys_list = []
        figure_labels = []
        figure_label_ids = [_curr_pos]
        # loop through chromosomes and homologs
        if order_by_chr:
            for _chr in _ordered_chr_names:
                for _ihomo, _zxys in enumerate(homolog_zxys):
                    _chr_zxys = _zxys[chr_2_indices[_chr]]
                    if np.sum(~np.isnan(_chr_zxys).any(1)) == 0:
                        continue
                    # update figure_labels
                    figure_labels.append(f"{_chr}_{_ihomo}")
                    # update id
                    if keep_valid:
                        _curr_pos = _curr_pos + np.sum(~np.isnan(_chr_zxys).any(1))
                        figure_zxys_list.append(_chr_zxys[~np.isnan(_chr_zxys).any(1)])
                    else:
                        _curr_pos = _curr_pos + len(_chr_zxys)
                        figure_zxys_list.append(_chr_zxys)
                    # append
                    figure_label_ids.append(_curr_pos)

        else: # order by homolog
            for _ihomo, _zxys in enumerate(homolog_zxys):
                for _chr in _ordered_chr_names:
                    _chr_zxys = _zxys[chr_2_indices[_chr]]
                    if np.sum(~np.isnan(_chr_zxys).any(1)) == 0:
                        continue
                    # update figure_labels
                    figure_labels.append(f"{_chr}_{_ihomo}")
                    # update id
                    if keep_valid:
                        _curr_pos = _curr_pos + np.sum(~np.isnan(_chr_zxys).any(1))
                        figure_zxys_list.append(_chr_zxys[~np.isnan(_chr_zxys).any(1)])
                    else:
                        _curr_pos = _curr_pos + len(_chr_zxys)
                        figure_zxys_list.append(_chr_zxys)
                    # append
                    figure_label_ids.append(_curr_pos)
        # summarize
        figure_label_ids = np.array(figure_label_ids)            
        return figure_zxys_list, figure_labels, figure_label_ids

    @staticmethod
    def extract_chr_region_ids(codebook, _chr_name):
        return codebook.loc[codebook['chr']==_chr_name, 'chr_order'].values

    @staticmethod
    def update_spot_groups_chr_info(spot_groups, codebook):
        for _ig, _g in enumerate(spot_groups):
            _g.chr = codebook.loc[codebook['id']==_g.tuple_id, 'chr'].values[0]
            _g.chr_order = codebook.loc[codebook['id']==_g.tuple_id, 'chr_order'].values[0]
    @staticmethod
    def generate_region_2_copy_num(codebook_df, chr_2_copy_num):
        _region_2_expect_num = {}
        for _chr_name, _num in chr_2_copy_num.items():
            _region_ids = codebook_df.loc[codebook_df['chr']==_chr_name, 'id'].values
            for _rid in _region_ids:
                _region_2_expect_num[_rid] = _num
        return _region_2_expect_num

    @staticmethod
    def prepare_chr_trees(chr_2_groups, ):
        chr_2_kdtree = {}
        # generate kdtree
        for _chr_name, _chr_tuples in chr_2_groups.items():
            _chr_zxys = np.array([_g.centroid_spot().to_positions()[0] for _g in _chr_tuples])
            chr_2_kdtree[_chr_name] = KDTree(_chr_zxys)
        # return
        return chr_2_kdtree

    @staticmethod
    def generate_random_invalid_pairs(cand_spots, invalid_pair_bits, total_num=2000, random=True, 
        ):
        # collect invalid pairs
        _invalid_groups = []
        if random:
            invalid_pair_bits = np.random.permutation(invalid_pair_bits)
        # loop through invalid pairs        
        for _pair in invalid_pair_bits:
            if len(_invalid_groups) >= total_num:
                break
            # randomly select spot pairs from this invalid bit-pair
            _num_sel = int(np.ceil(total_num / len(invalid_pair_bits)))
            _match_spots_list = []
            for _b in _pair:
                _spots = cand_spots[cand_spots.bits==_b]
                if len(_spots) < _num_sel:
                    break
                _sel_inds = np.random.choice(np.arange(len(_spots)), _num_sel)
                _match_spots_list.append(_spots[_sel_inds])
            if len(_match_spots_list) < len(_pair):
                continue

            for _i in range(_num_sel):
                _group = SpotTuple(Spots3D([_spots[_i] for _spots in _match_spots_list]),
                                   bits=np.array(_pair), 
                                   pixel_sizes=cand_spots.pixel_sizes)
                _invalid_groups.append(_group)

        return _invalid_groups

    @staticmethod
    def convert_chr_dict_to_homolog_dict(_chr_dict):
        _homolog_dict = {}
        for _chr, _values in _chr_dict.items():
            for _i, _value in enumerate(_values):
                _homolog_dict[f"{_chr}_{_i}"] = _value
        return _homolog_dict
    @staticmethod
    def collect_homolog_flags(_tuples, _score_th=None):
        _homolog_flags = []
        for _tp in _tuples:
            if _score_th is None or (hasattr(_tp, 'final_score') and getattr(_tp, 'final_score') > _score_th):
                _homolog_flags.append(_tp.homolog)
            else:
                _homolog_flags.append(-1)
        return np.array(_homolog_flags, dtype=np.int16)
    @staticmethod
    def tuple_list_to_zxys(tuple_list, dimension=3, score_th=None,):
        _positions = []
        for _tp in tuple_list:
            if _tp is None:
                _positions.append(np.nan * np.ones(dimension))
            elif score_th is not None and hasattr(_tp, 'final_score') and getattr(_tp, 'final_score') < score_th:
                _positions.append(np.nan * np.ones(dimension))
            else:
                _positions.append(_tp.centroid_spot().to_positions()[0])
        return np.array(_positions)
    @staticmethod
    def compare_flags_diff(old_flags, new_flags):
        return np.mean(old_flags != new_flags)
    @staticmethod
    def calculate_homolog_centroids(_spot_tuples, _labels):
        _valid_labels = np.unique(_labels)
        _valid_labels = _valid_labels[_valid_labels >= 0]
        _homolog_centers = []
        for _lb in _valid_labels:
            _homolog_centers.append(np.median([_g.centroid_spot().to_positions()[0] 
                                            for _l, _g in zip(_labels, _spot_tuples) if _l==_lb], axis=0))
        return np.array(_homolog_centers)
    @staticmethod
    def assign_homologs_by_chr(_chr_tuples, _chr_centers, _chr_region_ids,
                            _init_homolog_flags=None, _init_homolog_zxys_list=None,
                            allow_overlap=False, ct_dist_f=3, local_dist_f=0.5,
                            valid_score_th=-15, max_n_iter=15,
                            plot_stats=True, verbose=True):
        """Assign spot_groups into given number of homologs"""
        from ..spot_tools.scoring import log_distance_scores, exp_distance_scores, _local_distance, _center_distance
        from itertools import permutations, product
        # distribute tuples into region_ids
        _cand_tuples_list = [[_g for _g in _chr_tuples if _g.chr_order==_i] 
                            for _i in np.unique(_chr_region_ids)]
                            #for _i in np.arange(0, np.max(_chr_region_ids)+1)]
        
        # initialize tuple to homolog assignment
        for _tp in _chr_tuples:
            _tp.homolog = -1
        _homolog_flags = DNA_Merfish_Decoder.collect_homolog_flags(_chr_tuples)

        _assignment_diff = 1
        # store homolog flags in each step
        _homolog_centers_steps = [_chr_centers]
        _homolog_flags_steps = []
        if _init_homolog_zxys_list is not None:
            _homolog_zxys_list_steps = [_init_homolog_zxys_list]
        else:
            _homolog_zxys_list_steps = []

        while _assignment_diff > 0.002:
            # assignment within this iteration
            if len(_homolog_flags_steps) > max_n_iter:
                break

            #print("chr_center:", _homolog_centers_steps[-1])
            _assigned_homolog_tuple_list = [[] for _ct in _homolog_centers_steps[-1]]

            # loop through regions
            for _ireg, _cand_tuples in enumerate(_cand_tuples_list):
                #print(f"-- region:{_ireg} has {len(_cand_tuples)} candidates.")   
                # initialize tuple to homolog assignment
                for _tp in _cand_tuples:
                    _tp.homolog = -1
                # Case 1: if no candidates, append None for all homologs 
                if len(_cand_tuples) == 0:
                    for _homolog_tuple in _assigned_homolog_tuple_list:
                        _homolog_tuple.append(None)
                    continue
                _cand_zxys = np.array([_g.centroid_spot().to_positions()[0] for _g in _cand_tuples])
                _cand_self_scores = np.array([_g.self_score for _g in _cand_tuples])
                # calcuate distance to centers
                _cand_ct_dists = np.array([_center_distance(_cand_zxys, _ct) 
                                        for _ct in _homolog_centers_steps[-1]])
                #_cand_ct_scores = ..spot_tools.scoring.log_distance_scores(_cand_ct_dists)
                _cand_ct_scores = exp_distance_scores(_cand_ct_dists)
                # add to score
                _cand_final_scores = _cand_self_scores + np.array(_cand_ct_scores) * ct_dist_f

                # Case 2: if has previous assignment, use this information
                if len(_homolog_zxys_list_steps) > 0 and local_dist_f != 0:
                    _cand_local_dists = np.array([_local_distance(
                        _cand_zxys, _ireg * np.ones(len(_cand_zxys), dtype=np.int32),
                        _zxys_list, np.arange(len(_cand_tuples_list)),
                        local_size=4,
                        ) for _zxys_list in _homolog_zxys_list_steps[-1] ])
                    _cand_local_scores = exp_distance_scores(_cand_local_dists)
                    #_cand_local_scores = ..spot_tools.scoring.log_distance_scores(_cand_local_dists)
                    _cand_final_scores = _cand_final_scores + _cand_local_scores * local_dist_f

                # Case 3: assign one spot for each chr
                if len(_cand_tuples) >= len(_homolog_centers_steps[-1]):
                    # for each assignment, calculate total scores
                    if allow_overlap:
                        _assigns = list(product(np.arange(len(_cand_tuples)), len(_homolog_centers_steps[-1])))
                    else:
                        _assigns = list(permutations(np.arange(len(_cand_tuples)), len(_homolog_centers_steps[-1])))
                    _assign_scores = []
                    for _assign in _assigns:
                        _assign_scores.append( np.nansum([_scores[_j] for _scores, _j in zip(_cand_final_scores, _assign)]) )
                    # select the best
                    _best_assign = _assigns[np.argmax(_assign_scores)]

                    for _ihomo, _ituple in enumerate(_best_assign):
                        if _cand_final_scores[_ihomo, _ituple] >= valid_score_th:
                            _tuple = _cand_tuples[_ituple]
                            _tuple.homolog = _ihomo
                            _tuple.chr_ct_dist = _cand_ct_dists[_ihomo, _ituple]
                            _tuple.chr_ct_score = _cand_ct_scores[_ihomo, _ituple]
                            _tuple.final_score = _cand_final_scores[_ihomo, _ituple]
                            # append
                            _assigned_homolog_tuple_list[_ihomo].append(_tuple)
                        else:
                            _assigned_homolog_tuple_list[_ihomo].append(None)

                else:
                    if allow_overlap:
                        _assigns = list(product(np.arange(len(_homolog_centers_steps[-1])), len(_cand_tuples)))
                    else:
                        _assigns = list(permutations(np.arange(len(_homolog_centers_steps[-1])), len(_cand_tuples)))        
                    _assign_scores = []
                    for _assign in _assigns:
                        _assign_scores.append( np.nansum([_cand_final_scores[_ihomo, _ituple] 
                                                    for _ituple, _ihomo in enumerate(_assign)]) )
                    # select the best
                    _best_assign = _assigns[np.argmax(_assign_scores)]
                    for _ituple, _ihomo in enumerate(_best_assign):
                        if _cand_final_scores[_ihomo, _ituple] >= valid_score_th:
                            _tuple = _cand_tuples[_ituple]
                            _tuple.homolog = _ihomo
                            _tuple.chr_ct_dist = _cand_ct_dists[_ihomo, _ituple]
                            _tuple.chr_ct_score = _cand_ct_scores[_ihomo, _ituple]
                            _tuple.final_score = _cand_final_scores[_ihomo, _ituple]
                            # append
                            _assigned_homolog_tuple_list[_ihomo].append(_tuple)
                        else:
                            _assigned_homolog_tuple_list[_ihomo].append(None)
                    # assign None for not used homolog
                    for _ihomo, _tuple_list in enumerate(_assigned_homolog_tuple_list):
                        if _ihomo not in list(_best_assign):
                            _tuple_list.append(None)

            # update chr_centers
            _homolog_zxys_list = [DNA_Merfish_Decoder.tuple_list_to_zxys(_tuple_list, score_th=valid_score_th) 
                                    for _tuple_list in _assigned_homolog_tuple_list]
            _homolog_centers_steps.append( np.array([np.nanmean(_zxys, axis=0) for _zxys in _homolog_zxys_list]) )

            # update chromosome homolog assignment
            _new_homolog_flags = DNA_Merfish_Decoder.collect_homolog_flags(_chr_tuples, _score_th=valid_score_th)

            # compare homolog assignment flags
            _assignment_diff = DNA_Merfish_Decoder.compare_flags_diff(_new_homolog_flags, _homolog_flags)
            if verbose:
                print(f"- diff in iter-{len(_homolog_flags_steps)}: {_assignment_diff:.4f}")
            #print([_tp.final_score for _tp in _assigned_homolog_tuple_list[0] if _tp is not None])
            _homolog_flags = _new_homolog_flags
            # append
            _homolog_zxys_list_steps.append(_homolog_zxys_list)
            _homolog_flags_steps.append(_homolog_flags)

            if plot_stats:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot([_tp.tuple_id for _tp in _assigned_homolog_tuple_list[0] if _tp is not None],
                        [_tp.final_score for _tp in _assigned_homolog_tuple_list[0] if _tp is not None],
                        )
                plt.plot([_tp.tuple_id for _tp in _assigned_homolog_tuple_list[1] if _tp is not None],
                        [_tp.final_score for _tp in _assigned_homolog_tuple_list[1] if _tp is not None],
                        )
                plt.show()

        return _homolog_zxys_list_steps, _homolog_flags_steps, _assigned_homolog_tuple_list

    @staticmethod
    def initial_assign_homologs_by_chr(_chr_tuples, 
                                    _chr_region_ids, 
                                    _num_homologs=2,
                                    _verbose=True,
                                    ):
        from sklearn.cluster import KMeans
        if _verbose:
            print(f"-- init {_num_homologs} homologs for chr:{_chr_tuples[0].chr}")
        # get coordinates
        _chr_coords = np.array([_g.centroid_spot().to_positions()[0] for _g in _chr_tuples])
        # K-means
        _model = KMeans(n_clusters=_num_homologs, random_state=0)
        _model.fit(_chr_coords)
        _init_labels = _model.labels_
        _init_centers = _model.cluster_centers_

        # initialize assignment with special parameters
        _init_zxys_list, _init_flags, _init_tuple_list = \
            DNA_Merfish_Decoder.assign_homologs_by_chr(
                _chr_tuples, _init_centers, _chr_region_ids, 
                local_dist_f=0,
                plot_stats=False,
                verbose=_verbose,
            )
        # update return variables
        homolog_zxys_list = _init_zxys_list[-1]
        homolog_labels = _init_flags[-1]
        # update homolog centers
        homolog_centers = DNA_Merfish_Decoder.calculate_homolog_centroids(_chr_tuples, homolog_labels) 
        
        return homolog_zxys_list, homolog_labels, homolog_centers

    @staticmethod
    def summarize_zxys_by_regions(chr_2_zxys_list, codebook_df, num_homologs=2,     
        keep_valid=False):
        """Summarization of ."""
        # summarize chromosome information from codebook
        chr_2_indices = {_chr:np.array(codebook_df.loc[codebook_df['chr']==_chr].index)
                        for _chr in np.unique(codebook_df['chr'].values)}

        _ordered_chr_names = []
        for _chr_name, _chr_reg_id in zip(codebook_df['chr'], codebook_df['chr_order']):
            if _chr_name not in _ordered_chr_names:
                _ordered_chr_names.append(_chr_name)

        # assemble sorted-zxys for each homolog
        total_zxys_dict = [{_rid:np.ones(3)*np.nan 
                            for _rid in codebook_df.index} for _ihomo in np.arange(num_homologs)]
        for _chr_name, _zxys_list in chr_2_zxys_list.items():
            for _ihomo, _zxys in enumerate(_zxys_list):
                _chr_region_ids = DNA_Merfish_Decoder.extract_chr_region_ids(codebook_df, _chr_name)
                for _i_chr_reg, _zxy in zip(_chr_region_ids, _zxys):
                    #print(_chr_name, _ihomo, _i_chr_reg)
                    _rid = codebook_df.loc[(codebook_df['chr']==_chr_name) \
                                            & (codebook_df['chr_order']==_i_chr_reg)].index[0]
                    total_zxys_dict[_ihomo][_rid] = _zxy

        # convert to zxy coordinates in um
        homolog_zxys = [np.array(list(_dict.values()))/1000 for _dict in total_zxys_dict]
        return homolog_zxys

    ##NEW
    @staticmethod
    def assign_spot_groups_2_homologs(_chr_groups, _homolog_centers, _chr_region_ids, _invalid_chr_groups=None,
                                    weights = np.array([1,1,1,1,1]), score_th_percentile = 1,
                                    make_plots=True):
        """Assign spot_groups into homologs for a specific chromosome"""
        valid_final_scores = summarize_score(_chr_groups, weights)
        if _invalid_chr_groups is not None:
            invalid_final_scores = summarize_score(_invalid_chr_groups, weights)

            score_th = max(stats.scoreatpercentile(invalid_final_scores, 100-score_th_percentile),
                        stats.scoreatpercentile(valid_final_scores, score_th_percentile))
        else:
            score_th = stats.scoreatpercentile(valid_final_scores, score_th_percentile)
        if make_plots:
            plt.figure()
            plt.hist(np.ravel(valid_final_scores), bins=np.arange(-20,1,0.5), density=True, alpha=0.5)
            if _invalid_chr_groups is not None:
                plt.hist(np.ravel(invalid_final_scores), bins=np.arange(-20,1,0.5), density=True, alpha=0.5)
            plt.title(f"score_th={score_th:.3f}")
            plt.show()
            print(f"-- score_th={score_th:.3f}")
        # initialize homolog flags
        for _g in _chr_groups:
            _g.homolog = -1

        # assign groups into homologs
        homolog_2_groups = {_i:[] for _i,_ct in enumerate(_homolog_centers)}
        for _g in _chr_groups:
            _final_h_sc = summarize_score(_g.scores, weights=weights)
            _g.final_scores = _final_h_sc
            ihomo = np.argmax(_final_h_sc)
            _g.final_score = np.max(_final_h_sc)
            homolog_2_groups[ihomo].append(_g)
        # sort by chromosomal orders
        for _ihomo, _tps in homolog_2_groups.items():
            homolog_2_groups[_ihomo] = [_g for _g in sorted(_tps, key=lambda v:v.chr_order)]

        # generate homolog_2_groups_list
        homolog_2_groups_list = {}
        for _ihomo, _tps in homolog_2_groups.items():
            _tp_list = []
            # loop through each region in codebook
            for _chr_id in _chr_region_ids:
                _matched_tps = [_g for _g in _tps if _g.chr_order==_chr_id]
                if len(_matched_tps) == 0:
                    _tp_list.append(None)
                #print(len(_matched_tps))
                #print([_g.final_scores[_ihomo] for _g in _matched_tps], np.argmax([_g.final_scores[_ihomo] for _g in _matched_tps]))
                else:
                    _best_group = _matched_tps[np.argmax([_g.final_scores[_ihomo] for _g in _matched_tps])]
                    # update homolog flag for this group
                    _best_group.homolog = _ihomo
                    _tp_list.append(_best_group)
            homolog_2_groups_list[_ihomo] = _tp_list

        # convert into zxys 
        homolog_zxys_list = [DNA_Merfish_Decoder.tuple_list_to_zxys(_tps, score_th=score_th) 
                            for _ihomo, _tps in homolog_2_groups_list.items()]
        # collect homolog_flags
        new_chr_homolog_flags = DNA_Merfish_Decoder.collect_homolog_flags(_chr_groups, _score_th=score_th)
        # update homolog centers
        new_homolog_centers = DNA_Merfish_Decoder.calculate_homolog_centroids(_chr_groups, new_chr_homolog_flags)

        return homolog_zxys_list, new_chr_homolog_flags, new_homolog_centers


def batch_decode_DNA(cand_spots_df, codebook_df, decoder_filename, bits,
                     normalize_intensity=False, refine_chromatic=True, id_2_channel=None,
                     pixel_sizes=default_pixel_sizes, num_homologs=2, keep_ratio_th=0.5,
                     pair_search_radius=200,
                     inner_dist_factor=-1,
                     intensity_factor=1,
                     ct_dist_factor=5,
                     local_dist_factor=0.,
                     valid_score_th=-25,
                     load_from_file=True, overwrite=False,
                     return_decoder=False, make_plots=False, verbose=True,
                    ):
    """Batch process DNA-MERFISH decoding"""
    # load cand_spots
    if not isinstance(cand_spots_df, pd.DataFrame):
        raise TypeError("Wrong input type for cand_spots")

    codebook = np.array(codebook_df[[_name for _name in codebook_df.columns 
                                    if 'name' not in _name and  'id' not in _name and 'chr' not in _name]])
    if len(cand_spots_df) < num_homologs * codebook.sum() * keep_ratio_th:
        print(f"Not enough cand_spots ({len(cand_spots_df)}) found, skip.")
        return
            
    # create decoder folder
    if not os.path.exists(os.path.dirname(decoder_filename)):
        print(os.path.dirname(decoder_filename))
        os.makedirs(os.path.dirname(decoder_filename))

    #try:
    # create decoder class
    decoder = DNA_Merfish_Decoder(
        codebook_df, cand_spots_df,
        bits=bits, 
        pixel_sizes=pixel_sizes, 
        savefile=decoder_filename, 
        load_from_file=load_from_file,
        inner_dist_factor=inner_dist_factor,
        intensity_factor=intensity_factor,
        ct_dist_factor=ct_dist_factor,
        local_dist_factor=local_dist_factor,
        valid_score_th=valid_score_th,
        metric_weights=[1,1,3,2,2],
        )
    # assemble spot_groups
    decoder.prepare_spot_tuples(pair_search_radius=pair_search_radius, 
        force_search_for_region=False,
        overwrite=overwrite)
    # save 
    decoder._save_basic()
    # calculate scores
    self_scores = decoder.calculate_self_scores(make_plots=make_plots, overwrite=True)
    # spot picking
    chr_2_assigned_tuple_list, chr_2_zxys_list, chr_2_chr_centers= \
        decoder.finish_homolog_assignment(plot_stats=make_plots, overwrite=overwrite, 
        verbose=verbose)
    # save picked results
    decoder._save_picked_results(_overwrite=overwrite)
    ## Plot
    # distmap
    distmap_filename = decoder_filename.replace('Decoder.hdf5', 'AllChrDistmap.png')
    _zxys_list, _fig_labels, _fig_label_bds = decoder.summarize_zxys_all_chromosomes(
        chr_2_zxys_list, 
        decoder.codebook_df,
        keep_valid=True)
    GenomeWide_DistMap(_zxys_list, _fig_labels, _fig_label_bds,
        save_filename=distmap_filename, show_image=False, verbose=verbose)
    # stats
    SpotStat_filename = decoder_filename.replace('Decoder.hdf5', 'DecodeSpotStats.png')
    plot_spot_stats(decoder.spot_groups, decoder.spot_usage, 
        show_image=False, save_filename=SpotStat_filename, verbose=verbose)

    if return_decoder:
        return decoder
    else:
        return None
    #except:
    #    print(f"failed for decoding: {decoder_filename}")
    #    return None

def batch_load_attr(decoder_savefile, attr):
    """Batch load one attribute from decoder file"""    
    try:
        _cls = pickle.load(open(decoder_savefile, 'rb'))
        return getattr(_cls, attr)
    except:
        print(f"Loading failed.")
        return None

def load_hdf5_dict(_filename, data_key):
    _dict = {}
    with h5py.File(_filename, 'r') as _f:
        if data_key not in _f.keys():
            return None
        _grp = _f[data_key]
        for _key in _grp.keys():
            _dict[_key] = _grp[_key][:]
    return _dict

def load_hdf5_DataFrame(_filename, data_key):
    try:
        return pd.read_hdf(_filename, data_key)
    except:
        return None

def load_hdf5_array(_filename, data_key):
    try:
        with h5py.File(_filename, 'r') as _f:
            return _f[data_key][:]
    except:
        return None
def save_hdf5_dict(_filename, _data_key, _dict, _overwrite=False, _verbose=False):
    if _verbose:
        print(f"- Update {_data_key} into {_filename}")
    
    with h5py.File(_filename, 'a') as _f:
        _grp = _f.require_group(_data_key)
        for _key, _array in _dict.items():
            if _key in _grp.keys() and _overwrite:
                if _verbose:
                    print(f"-- overwrite {_key} in {_data_key}")
                del(_grp[_key])
            if _key not in _grp.keys():
                if _verbose:
                    print(f"-- saving {_key} in {_data_key}")
                _grp.create_dataset(_key, data=_array)
            else:
                if _verbose:
                    print(f"-- skip {_key} in {_data_key}")      
    return 

def spots_dict_to_cand_spots(cand_spot_filename, 
    pixel_sizes=default_pixel_sizes, 
    normalize_intensity=False, 
    refine_chromatic=False, ref_channel='647',
    id_2_channel=None):
    if isinstance(cand_spot_filename, str):
        cand_spots_dict = pickle.load(open(cand_spot_filename, 'rb'))
    elif isinstance(cand_spot_filename, dict):
        cand_spots_dict = cand_spot_filename
    # check number of spots
    _num_spots = np.sum([len(_spots) for _id, _spots in cand_spots_dict.items()])
    if _num_spots == 0:
        return Spots3D([], bits=[])
    try:
        if normalize_intensity:
            if id_2_channel is None:
                raise ValueError(f"id_2_channel not given.")
            cand_spots_dict = normalize_ch_2_channels(
                cand_spots_dict, id_2_channel,
            )
        if refine_chromatic:
            cand_spots_dict = refine_chromatic_by_channel_center(
                cand_spots_dict, id_2_channel, ref_channel=ref_channel,
            )
    except:
        return Spots3D([], bits=[])
    # assemble spots
    _all_spots, _all_bits = [], []
    for _bit, _spots in cand_spots_dict.items():
        if len(_spots) > 0:
            _all_spots.append(_spots)
            _all_bits.append(np.ones(len(_spots), dtype=np.uint16) * _bit)
            
    # concatenate
    return Spots3D(np.concatenate(_all_spots), 
                   bits=np.concatenate(_all_bits),
                   pixel_sizes=pixel_sizes)
        

def normalize_ch_2_channels(id_2_spots, id_2_channel):
    
    import copy

    _ch_2_ints = {_ch:[] for _ch in np.unique(list(id_2_channel.values()))}
    
    for _id in id_2_spots:
        _ch = id_2_channel[_id]
        _ch_2_ints[_ch].append( id_2_spots[_id].to_intensities() )
    
    _ch_2_mean_int = { _ch:np.mean(np.concatenate(_int_list)) for _ch, _int_list in _ch_2_ints.items() }
    
    id_2_norm_spots = {}
    for _id, _spots in id_2_spots.items():
        _new_spots = copy.copy(_spots)
        ##WORKING change this into spot3D.intensity_index
        _new_spots[:,0] = _new_spots.to_intensities() / _ch_2_mean_int[id_2_channel[_id]]
        id_2_norm_spots[_id] = _new_spots
        
    return id_2_norm_spots

def refine_chromatic_by_channel_center(id_2_spots, id_2_channel, ref_channel='647'):
    """refine chromatic abbreviation by forcing median of all"""
    _ch_2_coords = {_ch:[] for _ch in np.unique(list(id_2_channel.values()))}
    #
    for _id in id_2_spots:
        _ch = id_2_channel[_id]
        _ch_2_coords[_ch].append( id_2_spots[_id].to_coords() )
    #
    _ch_2_centers = { _ch:np.mean(np.concatenate(_int_list),axis=0) for _ch, _int_list in _ch_2_coords.items() }
    #print(_ch_2_centers)
    if ref_channel in _ch_2_centers:
        _ref_center = _ch_2_centers[ref_channel]
    else:
        print(list(_ch_2_centers.keys())[0] )
        _ref_center = list(_ch_2_centers.values())[0]
    #print(_ref_center)
    # adjust ch
    id_2_translated_spots = {}
    for _id, _spots in id_2_spots.items():
        _ch = id_2_channel[_id]
        _new_spots = copy(_spots)
        _new_spots[:,1:4] = _new_spots[:,1:4] - _ch_2_centers[_ch] + _ref_center
        id_2_translated_spots[_id] = _new_spots
    return id_2_translated_spots

def adjust_spots_by_chromatic_center(
    cand_spots:Spots3D, 
    bit_2_channel:dict, 
    ref_channel='647'):
    _channels = np.unique(list(bit_2_channel.values()))
    _spot_channels = np.array([bit_2_channel[_b] for _b in cand_spots.bits])
    _ch_2_centers = {
        _ch: np.mean(cand_spots.to_coords()[_spot_channels==_ch], axis=0)
        for _ch in _channels
    }
    # get ref
    if ref_channel not in _ch_2_centers:
        ref_channel = list(_ch_2_centers.keys())[0]
    # correct dict
    _ch_2_shift = {_ch:_ct-_ch_2_centers[ref_channel] for _ch,_ct in _ch_2_centers.items()}
    # correct coordinates
    _new_cand_spots = copy(cand_spots)
    for _ch in _ch_2_shift:
        _new_cand_spots[_spot_channels==_ch, 1:4] -= _ch_2_shift[_ch]

    return _new_cand_spots

def generate_score_metrics(spot_groups, chr_tree=None, homolog_centers=None,
                          n_neighbors=10, 
                          update_attr=True, 
                          overwrite=False):
    """Five metrics:
    [mean_intensity, COV_intensity, median_internal_distance, distance_to_neighbor, distance_to_chr_center]"""
    # check inputs
    if chr_tree is None or isinstance(chr_tree, list) or isinstance(chr_tree, KDTree):
        pass
    else:
        raise TypeError(f"Wrong input type for chr_tree")
    if homolog_centers is None or isinstance(homolog_centers, np.ndarray):
        pass
    else:
        raise TypeError(f"Wrong input type for homolog_centers")
    
    # collect basic metrics
    _basic_metrics_list = []
    for _g in spot_groups:
        #
        if hasattr(_g, 'basic_score_metrics') and not overwrite:
            _metrics = list(getattr(_g, 'basic_score_metrics'))
        else:
            _metrics = [np.mean(_g.spots.to_intensities()), # mean intensity
                    np.std(_g.intensities())/np.mean(_g.intensities()), # Coefficient of variation of intensity
                    np.median(_g.dist_internal()), # median of internal distance
                    ]
            if update_attr:
                _g.basic_score_metrics = np.array(_metrics)
        # append
        _basic_metrics_list.append(_metrics)
        
    def neighboring_dists(_spot_groups, _chr_tree, _n_neighbors=10):
        if _chr_tree is None or _chr_tree.n < _n_neighbors:
            return np.nan * np.ones(len(_spot_groups))
        return np.mean(_chr_tree.query([_g.centroid_spot().to_positions()[0] for _g in _spot_groups], 
                                       _n_neighbors)[0], axis=1)
    def homolog_center_dists(_spot_groups, _homolog_center):
        if _homolog_center is None:
            return np.nan * np.ones(len(_spot_groups))
        #print(np.array([_g.centroid_spot().to_positions()[0] for _g in _spot_groups]).shape)
        #print(_homolog_center.shape)
        return np.linalg.norm([_g.centroid_spot().to_positions()[0] - _homolog_center 
                               for _g in _spot_groups], axis=1)
                         
    # no homologs:
    group_metrics_list = []
    if homolog_centers is None:
        if isinstance(chr_tree, list):
            _tree = chr_tree[0]
        else:
            _tree = chr_tree
        
        _nb_dists = neighboring_dists(spot_groups, _tree, _n_neighbors=n_neighbors)
        _ct_dists = homolog_center_dists(spot_groups, None)
        for _g, _bm, _nb_dist, _ct_dist in zip(spot_groups, _basic_metrics_list, _nb_dists, _ct_dists):
            _metric_list = np.array([_bm+[_nb_dist, _ct_dist]])
            group_metrics_list.append(_metric_list)

    # with homologs
    ## with different chr_tree:
    elif isinstance(chr_tree, list) and isinstance(homolog_centers, np.ndarray):
        _metrics_by_homologs = []
        _nb_dists, _ct_dists = [], []
        
        for _tree, _ct in zip(chr_tree, homolog_centers):
            _nb_dists.append(neighboring_dists(spot_groups, _tree, _n_neighbors=n_neighbors))
            _ct_dists.append( homolog_center_dists(spot_groups, _ct) )
        # merge homologs
        _nb_dists = np.array(_nb_dists).transpose()
        _ct_dists = np.array(_ct_dists).transpose()
        for _g, _bm, _nb_dist, _ct_dist in zip(spot_groups, _basic_metrics_list, _nb_dists, _ct_dists):
            _metric_list = np.array([_bm+[_nd, _cd] for _nd,_cd in zip(_nb_dist,_ct_dist)])
            group_metrics_list.append(_metric_list)

    ## with the same chr_tree:
    elif isinstance(homolog_centers, np.ndarray):
        _metrics_by_homologs = []
        _nb_dists = neighboring_dists(spot_groups, chr_tree, _n_neighbors=n_neighbors)
        _ct_dists = []
        for _ct in homolog_centers:
            _ct_dists.append( homolog_center_dists(spot_groups, _ct) )
        # merge homologs
        _nb_dists = np.array(_nb_dists).transpose()
        _ct_dists = np.array(_ct_dists).transpose()
        for _g, _bm, _nb_dist, _ct_dist in zip(spot_groups, _basic_metrics_list, _nb_dists, _ct_dists):
            _metric_list = np.array([_bm+[_nb_dist, _cd] for _cd in _ct_dist])
            group_metrics_list.append(_metric_list)

    group_metrics_list = np.array(group_metrics_list)
    if update_attr:
        for _g, _metrics_list in zip(spot_groups, group_metrics_list):
            _g.score_metrics = _metrics_list

    return group_metrics_list

def collect_metrics(spot_groups, homo_index=4):
    _metrics_list = []
    for _g in spot_groups:
        if hasattr(_g, 'score_metrics'):
            _metrics = getattr(_g, 'score_metrics')
            if np.isnan(_metrics[:, homo_index]).all():
                _metrics_list.append(_metrics[0])
            else:
                _metrics_list.extend(list(_metrics))
    return np.array(_metrics_list)

def cdf_scores(values, refs, greater=True):
    from scipy import stats
    if np.isnan(refs).all():
        return np.nan * np.ones(np.shape(values))
    if greater:
        return np.array([stats.percentileofscore(refs, _v, kind='weak')/100 + 0.5/len(refs) 
                         for _v in values])
    else:
        return np.array([1 - stats.percentileofscore(refs, _v, kind='weak')/100 + 0.5/len(refs) 
                         for _v in values])

def generate_scores(spot_groups, ref_metrics_list, 
    greater_flags=[True,False,False,False,False], update_attr=True):

    _metrics_list = []
    for _g in spot_groups:
        if np.shape(ref_metrics_list)[1] != np.shape(_g.score_metrics)[1]:
            raise IndexError(f"scoring metrics not given for this spot_group")
    _metrics_list = collect_metrics(spot_groups)
    ref_metrics_list = np.array(ref_metrics_list)

    _scores_list = []
    for _i, (_metrics, _ref_metrics) in enumerate(zip(_metrics_list.transpose(), ref_metrics_list.transpose())):
        # get scores
        _scores_list.append(cdf_scores(_metrics, _ref_metrics, greater=greater_flags[_i]))
    # transpose back
    _scores_list = np.log(np.array(_scores_list).transpose())
    _curr_pt = 0
    _scores_by_group = []
    if update_attr:
        for _g in spot_groups:
            _g.scores = _scores_list[_curr_pt:_curr_pt+len(_g.score_metrics), :]
            _scores_by_group.append(_g.scores)
            _curr_pt += len(_g.score_metrics)

    return _scores_list, _scores_by_group

def summarize_score(spot_groups, weights=np.ones(5), 
                    normalize_spot_num=True, update_attr=True):
    if isinstance(spot_groups, list) and isinstance(spot_groups[0], SpotTuple):
        final_scores = []
        for _g in spot_groups:
            _final_score = np.nansum(_g.scores * weights, axis=-1)
            if normalize_spot_num:
                _final_score *= 1 / len(_g.spots)  
            # append
            final_scores.append(_final_score)
            if update_attr:
                _g.final_score = _final_score
    # if directly providing scores
    elif isinstance(spot_groups, np.ndarray) or isinstance(spot_groups, list):
        scores = np.array(spot_groups)
        final_scores = np.nansum(scores * np.ones(5), axis=-1)


    return np.array(final_scores)

def extract_group_indices(chr_2_assigned_tuple_list):
    _chr_2_tuple_indices = {}
    for _chr, _tuples_list in chr_2_assigned_tuple_list.items():
        _chr_2_tuple_indices[_chr] = []
        for _ihomolog, _tuples in enumerate(_tuples_list):
            _homolog_indices = []
            for _g in _tuples:
                if _g is None:
                    _homolog_indices.append(-1)
                else:
                    _homolog_indices.append(getattr(_g, 'sel_ind', -1))
            _chr_2_tuple_indices[_chr].append(_homolog_indices)
        _chr_2_tuple_indices[_chr] = np.array(_chr_2_tuple_indices[_chr], dtype=np.int32)
    return _chr_2_tuple_indices

def init_homolog_centers_BB(xyz_all,chr_all):
    """
    blackbox from BB to intialize 2 homologous centers
    Inputs: 
        zxy_all: zxys for specific chr,
        chr_all: corresponding chromosome region id for zxys,
    """
    ### usefull inner functions
    def compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds):
        dist1 = dist_mat[i1]
        dist2 = dist_mat[i2]
        keep = dist1>dist2
        nonkeep = ~keep
        rg_mean = np.mean(dist1[nonkeep])+np.mean(dist2[keep])
        cov = np.sum(np.dot(chr_inds,nonkeep)&np.dot(chr_inds,keep))
        #print(cov, rg_mean, keep.shape)
        return cov,rg_mean
    def get_cum_sums(ar_):
        ar__=[]
        count=1
        for i in range(len(ar_)):
            change = ar_[i]!=ar_[i+1] if i+1<len(ar_) else True
            if change:
                ar__.extend([i+1]*count)
                count=1
            else:
                count+=1
        return np.array(ar__)/float(ar__[-1])
    def get_cumprobs(vals,bigger=True):
        invert = 1 if bigger else -1
        asort_ = np.argsort(vals)[::invert]
        cumsums = get_cum_sums(vals[asort_])[np.argsort(asort_)]
        return cumsums

    ### Pool all candidates together
    chr_all = np.array(chr_all)
    chr_all_u = np.unique(chr_all)
    chr_inds = np.array([chr_all==chr_u for chr_u in chr_all_u])
    #print("&&", chr_inds.shape)
    dist_mat = squareform(pdist(xyz_all))
    ### If not enough data return empty
    if len(xyz_all)<2:
        return [np.nan]*3, [np.nan]*3
        #return c1,c2
    
    ### Compute best centers that first maximize coverage and then minimize radius of gyration
    rgs,covs,pairs=[],[],[]
    for i1 in range(len(xyz_all)):
        for i2 in range(i1):
            covT,rg_meanT = compute_cov_rg_fast(xyz_all,chr_all,i1,i2,dist_mat,chr_inds)
            covs.append(covT)
            rgs.append(rg_meanT)
            pairs.append([i1,i2])
    rgs,covs = np.array(rgs,dtype=np.float32),np.array(covs,dtype=np.float32)
    ibest = np.argmax(get_cumprobs(rgs,bigger=False)*get_cumprobs(covs,bigger=True))
    i1,i2 = pairs[ibest]
    c1=xyz_all[i1]
    c2=xyz_all[i2]
    return c1,c2

def batch_decode_BB_like(spot_filename, codebook_df, decoder_filename=None,
                         normalize_intensity=False, refine_chromatic=True, id_2_channel=None,
                         pixel_sizes=default_pixel_sizes, num_homologs=2, keep_ratio_th=0.5,
                         pair_search_radius=250,
                         metric_weights=np.array([1,1,1,1,1]),
                         max_num_iter=10, score_th_percentile=1, flag_diff_th=0.005,
                         load_from_file=True, overwrite=False,
                         make_plots=True, verbose=True,
                        ):
    """NEW"""
    # load spots
    cand_spots = spots_dict_to_cand_spots(
        spot_filename, pixel_sizes=pixel_sizes, 
        normalize_intensity=normalize_intensity, 
        refine_chromatic=refine_chromatic,
        id_2_channel=id_2_channel,
    )
    # check whether this cell is a good candidate
    codebook = np.array(codebook_df[[_name for _name in codebook_df.columns 
                                    if 'name' not in _name and  'id' not in _name and 'chr' not in _name]])
    if len(cand_spots) < num_homologs * codebook.sum() * keep_ratio_th:
        return
        
    # create decoder folder
    if decoder_filename is None:
        decoder_filename = spot_filename.replace('CandSpots', 'BBDecoder')
    if not os.path.exists(os.path.dirname(decoder_filename)):
        print(os.path.dirname(decoder_filename))
        os.makedirs(os.path.dirname(decoder_filename))

    # create class
    new_decoder = DNA_Merfish_Decoder(
        codebook_df, cand_spots,
        savefile=decoder_filename,
        pixel_sizes=pixel_sizes, 
        metric_weights=metric_weights,
        load_from_file=load_from_file, verbose=verbose,
    )

    # decode and get spot_groups
    new_decoder.prepare_spot_tuples(pair_search_radius=pair_search_radius, 
        overwrite=overwrite)
    # init chr centers
    chr_2_homolog_centers = new_decoder.init_homolog_centers()
    # scoring
    score_metrics = new_decoder.calculate_score_metrics()
    chr_2_scores = new_decoder.calculate_scores(score_metrics)
    # select spots
    new_decoder.iterative_assign_spot_groups_2_homologs(
        max_num_iter=max_num_iter,
        score_th_percentile=score_th_percentile,
        flag_diff_th=flag_diff_th,
    )
    # summarize into one matrix
    figure_zxys_list, figure_labels, figure_label_ids = new_decoder.summarize_zxys_all_chromosomes()
    # distmap
    distmap_filename = decoder_filename.replace('Decoder.pkl', 'AllDistmap.png')
    _ax = new_decoder.summarize_to_distmap(color_limits=[0,5], save_filename=distmap_filename) 
    # save
    new_decoder.save(overwrite=overwrite)

def generate_default_chr_2_copy_num(codebook_df, male=True):
    chr_2_copy_num = {_chr:2
        for _chr in np.unique(codebook_df['chr'])
    }
    if male:
        # overwrite X and Y for male
        chr_2_copy_num['X'] = 1
        chr_2_copy_num['Y'] = 1
    else:
        # overwrite X and Y for male
        chr_2_copy_num['X'] = 2
        chr_2_copy_num['Y'] = 0
    
    return chr_2_copy_num

