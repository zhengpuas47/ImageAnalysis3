import os, glob, sys, time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy
from scipy import stats
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# local
from .preprocess import Spots3D, SpotTuple
default_pixel_sizes=[250,108,108]
default_search_th = 150
default_search_eps = 0.25

class Merfish_Decoder():
    """Class to decode spot-based merfish results"""    

    def __init__(self, 
                 codebook_df,
                 cand_spots,
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
        self.pixel_sizes = np.array(pixel_sizes)
        self.inner_dist_factor = float(inner_dist_factor)
        self.intensity_factor = float(intensity_factor)
        # savefile
        self.savefile = savefile
        # cand_spots
        if not isinstance(cand_spots, Spots3D):
            if len(cand_spots) != len(bits):
                raise IndexError(f"lengh of _bits: {len(_bits)} doesn't match length of cand_spots: {len(cand_spots)}")
            self.cand_spots = Spots3D(cand_spots, bits=bits, pixel_sizes=pixel_sizes)
        else:
            self.cand_spots = cand_spots
        # load codebook if automatic
        if auto:
            self.load_codebook()
            self.find_valid_pairs_in_codebook()
            self.find_valid_tuples_in_codebook()
        # load from savefile
        if load_from_file:
            self.load(overwrite=overwrite)
        # other attributes
        self.verbose = verbose

    def load(self, overwrite=False):
        if self.savefile is not None:
            if os.path.isfile(self.savefile):
                print(f"- Loading decoder into file: {self.savefile}")
                _cls = pickle.load(open(self.savefile, 'rb'))
                for _attr in dir(_cls):
                    if _attr[0] == '_':
                        continue
                    elif not hasattr(self, _attr) or overwrite:
                        setattr(self, _attr, getattr(_cls, _attr))
            else:
                print(f"- Skip loading because file: {self.savefile} doesn't exist")
        else:
            print(f"- Decoder.savefile is not given, skip loading.")

    def save(self, overwrite=False):
        if self.savefile is not None:
            if not os.path.isfile(self.savefile) or overwrite:
                print(f"- Saving decoder into file: {self.savefile}")
                pickle.dump(self, open(self.savefile, 'wb'))
            else:
                print(f"- Skip saving because file: {self.savefile} already exists")
        else:
            print(f"- Decoder.savefile is not given, skip saving")


    def load_codebook(self):
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
        self.bit_ids = np.arange(1, len(self.bit_names)+1)
        self.codebook_matrix = codebook_df[self.bit_names].values

    def find_valid_pairs_in_codebook(self):
        """Find valid 2-bit pairs given in the codebook """
        from itertools import combinations
        valid_pair_bits = []
        valid_pair_ids = []
        for _icode, _code in enumerate(self.codebook_matrix):
            for _p in combinations(np.where(_code > 0)[0], 2):
                _bs = tuple(np.sort(self.bit_ids[np.array(_p)]))
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
            _bs = tuple(np.sort(self.bit_ids[np.where(_code>0)[0]]))
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
    def generate_reference(spot_groups,
                        intensity_metric='mean',
                        dist_metric='min',):
        # inteisities
        _ints = [getattr(np, intensity_metric)(_g.spots.to_intensities()) for _g in spot_groups]
        # internal dists
        _inter_dists = [getattr(np, dist_metric)(_g.dist_internal()) for _g in spot_groups]
        return _ints, _inter_dists
    @staticmethod
    def extract_chr_region_ids(codebook, _chr_name):
        return codebook.loc[codebook['chr']==_chr_name, 'chr_order'].values

class DNA_Merfish_Decoder(Merfish_Decoder):
    """DNA MERFISH decoder, based on merfish decoder but allow some special features"""
    def __init__(self, 
                 codebook_df,
                 cand_spots,
                 bits=None,
                 savefile=None,
                 pixel_sizes=default_pixel_sizes,
                 inner_dist_factor=-1,
                 intensity_factor=1,
                 ct_dist_factor=3.5,
                 local_dist_factor=0.5,
                 valid_score_th=-10,
                 auto=True,
                 load_from_file=True,
                 verbose=True,
                 ):
        super().__init__(codebook_df=codebook_df, cand_spots=cand_spots, bits=bits,
            savefile=savefile,
            pixel_sizes=pixel_sizes, 
            inner_dist_factor=inner_dist_factor, intensity_factor=intensity_factor,
            auto=auto, load_from_file=load_from_file, verbose=verbose)
        
        # extra parameters
        self.ct_dist_factor = float(ct_dist_factor)
        self.local_dist_factor = float(local_dist_factor)
        self.valid_score_th = float(valid_score_th)


    def prepare_spot_tuples(self, pair_search_radius=200, overwrite=False):
        """ """
        if hasattr(self, 'spot_groups') and not overwrite:
            print("spot_groups already exists.")
        else:
            # find pairs
            self.find_spot_pairs_in_radius(search_th=pair_search_radius,)
            # assemble tuples
            spot_groups = self.assemble_complete_codes()
        # update chromosome info
        for _ig, _g in enumerate(self.spot_groups):
            _g.chr = self.codebook.loc[self.codebook['id']==_g.tuple_id, 'chr'].values[0]
            _g.chr_order = self.codebook.loc[self.codebook['id']==_g.tuple_id, 'chr_order'].values[0]


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

            # run initialize decoding
            init_zxys_list, init_homolog_labels, init_homolog_centers =\
                DNA_Merfish_Decoder.initial_assign_homologs_by_chr(_chr_seeding_groups, _chr_region_ids, num_homologs)
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

            return getattr(self, 'chr_2_assigned_tuple_list'),\
                getattr(self, 'chr_2_zxys_list'), \
                getattr(self, 'chr_2_chr_centers')

        chr_2_assigned_tuple_list = {}
        chr_2_zxys_list = {}
        chr_2_chr_centers = {}

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

    def summarize_zxys_all_chromosomes(self, num_homologs=2, keep_valid=True, order_by_chr=True):
        """Summarization of ."""
        # summarize chromosome information from codebook
        chr_2_indices = {_chr:np.array(self.codebook.loc[self.codebook['chr']==_chr].index)
                        for _chr in np.unique(self.codebook['chr'].values)}

        _ordered_chr_names = []
        for _chr_name, _chr_reg_id in zip(self.codebook['chr'], self.codebook['chr_order']):
            if _chr_name not in _ordered_chr_names:
                _ordered_chr_names.append(_chr_name)
        
        # assemble sorted-zxys for each homolog
        total_zxys_dict = [{_rid:np.ones(3)*np.nan 
                            for _rid in self.codebook.index} for _ihomo in np.arange(num_homologs)]
        for _chr_name, _zxys_list in self.chr_2_zxys_list.items():
            for _ihomo, _zxys in enumerate(_zxys_list):
                for _i_chr_reg, _zxy in enumerate(_zxys):
                    #print(_chr_name, _ihomo, _i_chr_reg)
                    _rid = self.codebook.loc[(self.codebook['chr']==_chr_name) \
                                              & (self.codebook['chr_order']==_i_chr_reg)].index[0]
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
        # update attributes
        setattr(self, 'final_zxys_list', figure_zxys_list)
        setattr(self, 'final_zxys_names', figure_labels)
        setattr(self, 'final_zxys_boundaries', figure_label_ids)

        return figure_zxys_list, figure_labels, figure_label_ids

    def summarize_to_distmap(self, distmap_plot_kwargs={}, 
                             color_limits=[1,5],
                             figsize=(6,5),
                             dpi=150,
                             save_filename=None):
        
        cmap = copy(cm.seismic_r)
        cmap.set_bad([0.5,0.5,0.5])
        
        _label_bds = getattr(self, 'final_zxys_boundaries')
        _labels = getattr(self, 'final_zxys_names')
        
        try:
            _distmap = squareform(pdist(np.concatenate(getattr(self, 'final_zxys_list'))))
        except:
            print(len( getattr(self, 'final_zxys_list') ))
            return None

        fig, ax = plt.subplots(figsize=figsize,dpi=dpi)
        _pf = ax.imshow(_distmap, cmap=cmap, vmin=min(color_limits), vmax=max(color_limits))
        plt.colorbar(_pf, label=f'Pairwise distance (\u03bcm)')

        ax.set_xticks((_label_bds[1:] + _label_bds[:-1])/2)
        ax.set_xticklabels(_labels, fontsize=6, rotation=60,)
        ax.set_yticks((_label_bds[1:] + _label_bds[:-1])/2)
        ax.set_yticklabels(_labels, fontsize=6,)
        ax.tick_params(axis='both', which='major', pad=1)

        ax.hlines(_label_bds, 0, len(_distmap), color='black', linewidth=0.5)
        ax.vlines(_label_bds, 0, len(_distmap), color='black', linewidth=0.5)
        ax.set_xlim([0, len(_distmap)])
        ax.set_ylim([len(_distmap), 0])
        ax.set_title(f"kept_spots: { np.sum(np.isnan(np.concatenate(getattr(self, 'final_zxys_list'))).any(1)==0) }")        
        if save_filename is not None:
            fig.savefig(save_filename, dpi=200, transparent=True)
            # add attribute
            self.savefile_distmap = save_filename
        
        return ax

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
                            for _i in np.arange(0, np.max(_chr_region_ids)+1)]
        
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
                for _i_chr_reg, _zxy in enumerate(_zxys):
                    #print(_chr_name, _ihomo, _i_chr_reg)
                    _rid = codebook_df.loc[(codebook_df['chr']==_chr_name) \
                                            & (codebook_df['chr_order']==_i_chr_reg)].index[0]
                    total_zxys_dict[_ihomo][_rid] = _zxy

        # convert to zxy coordinates in um
        homolog_zxys = [np.array(list(_dict.values()))/1000 for _dict in total_zxys_dict]
        return homolog_zxys

def batch_decode_DNA(spot_filename, codebook_df, decoder_filename=None,
                     pixel_sizes=default_pixel_sizes, num_homologs=2, keep_ratio_th=0.5,
                     pair_search_radius=150,
                     inner_dist_factor=-1,
                     intensity_factor=1,
                     ct_dist_factor=4,
                     local_dist_factor=0.,
                     valid_score_th=-30,
                     load_from_file=True, overwrite=False,
                    ):
    """Batch process DNA-MERFISH decoding"""

    codebook = np.array(codebook_df[[_name for _name in codebook_df.columns 
                                    if 'name' not in _name and  'id' not in _name and 'chr' not in _name]])
    # load cand_spots
    cand_spots = spots_dict_to_cand_spots(spot_filename)

    cand_spots_dict = pickle.load(open(spot_filename, 'rb'))
    #print(spot_filename, len(cand_spots))

    if len(cand_spots) < num_homologs * codebook.sum() * keep_ratio_th:
        return
            
    # create decoder folder
    if decoder_filename is None:
        decoder_filename = spot_filename.replace('CandSpots', 'Decoder')
    if not os.path.exists(os.path.dirname(decoder_filename)):
        print(os.path.dirname(decoder_filename))
        os.makedirs(os.path.dirname(decoder_filename))

    # create decoder class
    decoder = DNA_Merfish_Decoder(codebook_df, cand_spots,
                                  pixel_sizes=pixel_sizes, 
                                  savefile=decoder_filename, 
                                  load_from_file=load_from_file,
                                  inner_dist_factor=inner_dist_factor,
                                  intensity_factor=intensity_factor,
                                  ct_dist_factor=ct_dist_factor,
                                  local_dist_factor=local_dist_factor,
                                  valid_score_th=valid_score_th,
                                  )

    decoder.prepare_spot_tuples(pair_search_radius=pair_search_radius, overwrite=overwrite)

    self_scores = decoder.calculate_self_scores(make_plots=False, overwrite=overwrite)

    chrs_2_init_centers = decoder.init_homolog_assignment(overwrite=overwrite)

    chr_2_assigned_tuple_list, chr_2_zxys_list, chr_2_chr_centers= \
        decoder.finish_homolog_assignment(plot_stats=False, overwrite=overwrite, verbose=True)

    figure_zxys_list, figure_labels, figure_label_ids = decoder.summarize_zxys_all_chromosomes()

    # distmap
    distmap_filename = decoder_filename.replace('Decoder.pkl', 'AllDistmap.png')
    _ax = decoder.summarize_to_distmap(decoder, save_filename=distmap_filename) 

    decoder.save(overwrite=overwrite)

def batch_load_attr(decoder_savefile, attr):
    """Batch load one attribute from decoder file"""    
    try:
        _cls = pickle.load(open(decoder_savefile, 'rb'))
        return getattr(_cls, attr)
    except:
        print(f"Loading failed.")
        return None


def spots_dict_to_cand_spots(cand_spot_filename, pixel_sizes=default_pixel_sizes):
    cand_spots_dict = pickle.load(open(cand_spot_filename, 'rb'))
    _all_spots, _all_bits = [], []
    for _bit, _spots in cand_spots_dict.items():
        if len(_spots) > 0:
            _all_spots.append(_spots)
            _all_bits.append(np.ones(len(_spots), dtype=np.uint16) * _bit)
    
    if len(_all_spots) == 0:
        return Spots3D([], bits=[])
            
    # concatenate
    return Spots3D(np.concatenate(_all_spots), 
                   bits=np.concatenate(_all_bits),
                   pixel_sizes=pixel_sizes)
        
    