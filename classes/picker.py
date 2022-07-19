from cmath import nan
import enum
import os, time
import h5py
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
# local
Axis3D_infos = ['z', 'x', 'y']
default_weights = [5,2,1]
default_score_th=np.log(0.05)
default_coords_columns = ['region_name', 'chr','start','end','center_z', 'center_x', 'center_y', 'center_intensity', 'center_internal_dist']


class SpotPicker():
    """Class to pick spots decoded from combinatorial or fitted from sequential"""
    def __init__(self, 
                 decoded_file:str,
                 metric_weights:np.ndarray=default_weights,
                 valid_score_th:float=default_score_th,
                 chr_2_copyNum=None,
                 male=True,
                 save_file=None,
                 autoRun:bool=True,
                 preLoad:bool=True,
                 overwrite:bool=False,
                 verbose:bool=True,
                 ):
        # File information
        self.decodedFile = decoded_file
        self.saveFile = save_file
        # chr info
        if isinstance(chr_2_copyNum, dict):
            self.chr_2_copyNum = chr_2_copyNum
        # parameters
        self.male = male
        self.metricWeights = metric_weights
        self.validScoreTh = valid_score_th
        self.overwrite = overwrite
        self.verbose = verbose
        # general buffer
        self.history_chr_2_homolog_centers = []
        self.history_chr_2_homolog_hzxys = []
        self.history_chr_2_homolog_inds = []
        
        if preLoad:
            _exist_flag = self._load_picked()
        else:
            _exist_flag = False
        # if needs to run:
        if autoRun and _exist_flag:
            pass
        return
    # Load codebooks and decoded files
    def _load_decoded(self, sel_columns=default_coords_columns):
        _codebook_keys = []
        _load_keys = []
        _load_dtype = []
        if self.verbose:
            print(f"- Read savefile to load codebooks and coordinates")
        with h5py.File(self.decodedFile, 'r') as _f:
            for _name in _f.keys():
                # Skip picked spots
                if _name == 'picked':
                    continue
                # for the rest append info
                if 'spotGroups' in _f[_name].keys():
                    _load_keys.append(f"{_name}/spotGroups")
                    _load_dtype.append('combo')
                    _codebook_keys.append(f"{_name}/codebook")
                elif 'candSpots' in _f[_name].keys():
                    _load_keys.append(f"{_name}/candSpots")
                    _load_dtype.append('unique')
                    _codebook_keys.append(f"{_name}/codebook")

        # Load
        self.codebook_list = []
        self.coords_list = []
        for _ck, _lk, _dtype in zip(_codebook_keys, _load_keys, _load_dtype):
            if self.verbose:
                print(f"-- load {_dtype} of {_ck.split('/codebook')[0]}")
            # codebook
            _codebook = pd.read_hdf(self.decodedFile, _ck)
            _codebook['library'] = _ck.split('/codebook')[0]
            _codebook['dtype'] = _dtype
            self.codebook_list.append(_codebook)
            # coordiates
            _coords_df = pd.read_hdf(self.decodedFile, _lk)
            if len(_coords_df) > 0:
                # sel among sel_col
                _kept_sel_columns = [_c for _c in sel_columns if _c in _coords_df.columns]
                _miss_sel_columns = [_c for _c in sel_columns if _c not in _coords_df.columns]
                _sel_df = _coords_df[_kept_sel_columns].copy() # get existing columns
                _sel_df.loc[:,_miss_sel_columns] = nan # fill nan
                _sel_df.loc[:,'codebook_name'] = _ck.split('/codebook')[0]
                _sel_df.loc[:,'data_type'] = _dtype
                # add number of spots info
                _sel_df.loc[:,'num_spots'] = (np.isnan(_coords_df[[_c for _c in _coords_df.columns if 'height' in _c]])==False).sum(1)
                self.coords_list.append(_sel_df)
        return
    # Merge coodinates and sort along chromosomes 
    def _merge_decoded(self):
        if self.verbose:
            print(f"- Merge codebooks and coordinates")
        if not hasattr(self, 'codebook_list') or not hasattr(self, 'coords_list'):
            self._load_decoded()
        # skip if no codebooks exists
        if len(self.codebook_list) == 0:
            print(f"-- No codebook exists, skip.")
            return
        # skip if no codebooks exists
        if len(self.coords_list) == 0:
            print(f"-- No coordinate exists, skip.")
            return
        # Merge codebooks
        merged_codebook = pd.concat(self.codebook_list, axis=0, join='outer',ignore_index=True).fillna(0)
        merged_codebook['reg_start'] = [int(_name.split(':')[1].split('-')[0]) for _name in merged_codebook['name']]
        merged_codebook['reg_end'] = [int(_name.split(':')[1].split('-')[1]) for _name in merged_codebook['name']]
        merged_codebook['reg_mid'] = (merged_codebook['reg_start'] + merged_codebook['reg_end'])/2

        for _chr in np.unique(merged_codebook['chr']):
            _chr_codebook = merged_codebook[merged_codebook['chr']==_chr]
            _reg_order = np.argsort(merged_codebook.loc[merged_codebook['chr']==_chr, 'reg_mid'])
            merged_codebook.loc[_chr_codebook.index[_reg_order], 'chr_order'] = np.arange(len(_chr_codebook)).astype(np.int32)

        # cleanup 
        self.merged_codebook = merged_codebook[[_c for _c in merged_codebook.columns if 'reg_' not in _c]]
        # summarize
        _region_2_ind = {}
        _region_2_chr_order = {}
        for _i, _name, _chr_order in zip(self.merged_codebook.index, self.merged_codebook['name'],self.merged_codebook['chr_order']):
            _region_2_ind[_name] = _i
            _region_2_chr_order[_name] = int(_chr_order)
            
        ## spots
        self.merged_coords = pd.concat(self.coords_list, axis=0, join='outer',ignore_index=True)
        self.merged_coords['index'] = [_region_2_ind[_r] for _r in self.merged_coords['region_name']]
        self.merged_coords['chr_order'] = [_region_2_chr_order[_r] for _r in self.merged_coords['region_name']]
        if self.verbose:
            print(f"{len(self.merged_coords)} candidate cooridnates for {len(self.merged_codebook)} regions in total. ")
        return
    # Default copynumber
    def _generate_default_chr_copyNum(self):
        self.chr_2_copyNum = {_chr:2
            for _chr in np.unique(self.merged_codebook['chr'])
        }
        if self.male:
            # overwrite X and Y for male
            self.chr_2_copyNum['X'] = 1
            self.chr_2_copyNum['Y'] = 1
        else:
            # overwrite X and Y for male
            self.chr_2_copyNum['X'] = 2
            self.chr_2_copyNum['Y'] = 0
        return
    # initialize chromosome centers
    ## TODO: fix copynumber contraints
    def _init_homolog_centers(self, method="kmeans", min_spot_num=2, axis_infos=Axis3D_infos):
        """NEW Initialize homologous chromosome centers by selected methods
        """
        if hasattr(self, 'chr_2_homolog_centers') and not self.overwrite:
            if self.verbose:
                print(f"- directly return chr_2_homolog_centers")
            return
        if method == 'kmeans':
            from sklearn.cluster import KMeans
        # chr_2_init_centers
        self.chr_2_homolog_centers = {}
        self.chr_2_cand_hzxys = {}
        self.chr_2_cand_ids = {}
        # loop through chrs
        for _chr_name, _exp_num in self.chr_2_copyNum.items():
            _chr_coords_df = self.merged_coords.loc[self.merged_coords['chr']==str(_chr_name)]
            # if not spots exists, skip
            if len(_chr_coords_df) < min_spot_num:
                continue
            # get coordinates
            _chr_hzxys = _chr_coords_df[['center_intensity']+[f"center_{_x}" for _x in axis_infos]].values
            _chr_ids = _chr_coords_df['chr_order'].values
            # append
            self.chr_2_cand_hzxys[_chr_name] = _chr_hzxys
            self.chr_2_cand_ids[_chr_name] = _chr_ids
            # calculate weights
            _uinds, _uind_counts = np.unique(_chr_ids, return_counts=True)
            _ind_2_weight = {_i:1/_c for _i,_c in zip(_uinds, _uind_counts)}
            _chr_weights = np.array([_ind_2_weight[_i] for _i in _chr_ids])
            # K-means
            if method =='kmeans':
                _model = KMeans(n_clusters=_exp_num, random_state=0)
                _model.fit(_chr_hzxys[:,1:], sample_weight=_chr_weights)
                #_init_labels = _model.labels_
                _init_centers = _model.cluster_centers_
                # save for now
                self.chr_2_homolog_centers[_chr_name] = _init_centers
    # calculate scoring metrics
    def _prepare_score_metrics(self, local_range=5, axis_infos=Axis3D_infos):
        """Calculate chromosome wise scoring metrics"""
        if self.verbose:
            print(f"- Calculate scoring metrics")
        self.chr_2_metrics = {}
        if not hasattr(self, 'chr_2_cand_hzxys') or not hasattr(self, 'chr_2_cand_ids'):
            _chr_2_cand_hzxys = {}
            _chr_2_cand_ids = {}

        for _chr_name, _chr_centers in self.chr_2_homolog_centers.items():
            if hasattr(self, 'chr_2_cand_hzxys') and hasattr(self, 'chr_2_cand_ids') :
                _chr_hzxys = self.chr_2_cand_hzxys[_chr_name]
                _chr_ids = self.chr_2_cand_ids[_chr_name]
            else:
                # get coordinates
                _chr_coords_df = self.merged_coords.loc[self.merged_coords['chr']==str(_chr_name)]
                _chr_hzxys = _chr_coords_df[['center_intensity']+[f"center_{_x}" for _x in axis_infos]].values
                _chr_ids = _chr_coords_df['chr_order'].values
                _chr_2_cand_hzxys[_chr_name] = _chr_hzxys
                _chr_2_cand_ids[_chr_name] = _chr_ids
            # calculate metrics
            if hasattr(self, 'chr_2_homolog_hzxys_list'):
                _ref_hzxys_list = self.chr_2_homolog_hzxys_list.get(_chr_name, None)
            else:
                _ref_hzxys_list = None
            self.chr_2_metrics[_chr_name] = prepare_score_metrics_by_chr(
                _chr_hzxys, _chr_ids, _chr_centers, 
                prev_homolog_hzxys=_ref_hzxys_list, 
                local_range=local_range)
        # add this attribute if not given previously
        if not hasattr(self, 'chr_2_cand_hzxys') or not hasattr(self, 'chr_2_cand_ids'):
            self.chr_2_cand_hzxys = _chr_2_cand_hzxys
            self.chr_2_cand_ids = _chr_2_cand_ids
        return
    # calculate scores
    def _calculate_scores(self, num_metrics=3,):
        if self.verbose:
            print(f"- Calculate scores")
        if not hasattr(self, 'chr_2_metrics') or len(self.chr_2_metrics) == 0:
            self._prepare_score_metrics()
        # if still no metric calculated, skip
        if len(self.chr_2_metrics) == 0:
            print(f"No chromosome metrics calculated, exit.")
            return
        # init
        self.chr_2_scores = {}
        # summarize all metrics
        _all_metrics = np.concatenate([_ms.reshape(num_metrics, -1).transpose() 
                                       for _ms in self.chr_2_metrics.values()]).transpose()
        # calculate cdf scores
        for _chr_name, _metrics in self.chr_2_metrics.items():
            _scores = [ # this length matches num_metrics
                np.log(cdf_scores(np.ravel(_metrics[0]), _all_metrics[0][np.isreal(_all_metrics[0])], 
                                  greater=True).reshape(len(_metrics[0]),-1)) * self.metricWeights[0], # int
                np.log(cdf_scores(np.ravel(_metrics[1]), _all_metrics[1][np.isreal(_all_metrics[1])], 
                                  greater=False).reshape(len(_metrics[1]),-1)) * self.metricWeights[1], # ct dist 
                np.log(cdf_scores(np.ravel(_metrics[2]), _all_metrics[2][np.isreal(_all_metrics[2])], 
                                  greater=False).reshape(len(_metrics[2]),-1)) * self.metricWeights[2], # local dist
            ]
            self.chr_2_scores[_chr_name] = np.nansum(_scores, axis=0)
            # Save to dataframe
            _num_homolog = self.chr_2_copyNum[_chr_name]
            for _ihomo in range(_num_homolog):
                # init score column if not exists
                if f"score_h{_ihomo}" not in self.merged_coords.columns:
                    self.merged_coords[f"score_h{_ihomo}"] = np.nan
                self.merged_coords.loc[self.merged_coords['chr']==_chr_name, 
                                        f"score_h{_ihomo}"] = self.chr_2_scores[_chr_name][_ihomo]
        return
    # assign based on scores
    def _assign_homologs_by_scores(self, allow_overlap=False, detailed_verbose=False):
        if self.verbose:
            print(f"- Assign candidates into homologs")
        from itertools import permutations, product
        if not hasattr(self, 'chr_2_scores') or len(self.chr_2_scores)==0:
            self._calculate_scores()
            # if still not working, skip
            if not hasattr(self, 'chr_2_scores'):
                print(f"No chromosomes have been scored, skip.")
                return
        # save existing
        if hasattr(self, 'chr_2_homolog_hzxys_list') and len(self.chr_2_homolog_hzxys_list):
            self.history_chr_2_homolog_hzxys.append(
                {_chr:_hzxys_list for _chr, _hzxys_list in self.chr_2_homolog_hzxys_list.items()})
            self.history_chr_2_homolog_inds.append(
                {_chr:_inds for _chr, _inds in self.chr_2_homolog_inds_list.items()})
        self.chr_2_homolog_hzxys_list = {}
        self.chr_2_homolog_inds_list = {}
        # assign in each chr
        for _chr_name, _scores in self.chr_2_scores.items():
            # if not supposed to change, directly copy
            if hasattr(self, 'chr_2_change') and not self.chr_2_change[_chr_name]:
                self.chr_2_homolog_hzxys_list[_chr_name] = self.history_chr_2_homolog_hzxys[-1][_chr_name]
                self.chr_2_homolog_inds_list[_chr_name] = self.history_chr_2_homolog_inds[-1][_chr_name]
                if self.verbose and detailed_verbose:
                    print(f"-- skip chr:{_chr_name}")
                continue
            # get chr_info
            _chr_hzxys, _chr_ids = self.chr_2_cand_hzxys[_chr_name], self.chr_2_cand_ids[_chr_name]
            _chr_codebook = self.merged_codebook.loc[self.merged_codebook['chr']==_chr_name]
            _num_homologs = len(self.chr_2_homolog_centers[_chr_name])
            _num_regions = len(_chr_codebook)
            if self.verbose and detailed_verbose:
                print(f"-- process chr:{_chr_name}")
            # store directly coordinates
            homolog_hzxys_list = np.ones([_num_homologs, _num_regions, 4]) * np.nan 
            # store indices relative to DataFrame: merged_coords
            homolog_inds_list = np.ones([_num_homologs, _num_regions], dtype=np.int32) * -1 
            # loop through each indices
            for _i in range(len(_chr_codebook)):
                _cand_hzxys = _chr_hzxys[np.where(_chr_ids==_i)[0]]
                _cand_scores = _scores[:, np.where(_chr_ids==_i)[0]]
                # if no candidate spots, skip
                if len(_cand_hzxys) == 0:
                    continue
                # case 1. if enough spots exists, try to best assign
                if len(_cand_hzxys) >= _num_homologs:
                    if allow_overlap:
                        _cand_assigns = list(product(np.arange(len(_cand_hzxys)), _num_homologs))
                    else:
                        _cand_assigns = list(permutations(np.arange(len(_cand_hzxys)), _num_homologs))
                    _assign_scores = [np.nanmean([_scores[_j] for _scores, _j in zip(_cand_scores, _assign)])
                                      for _assign in _cand_assigns]
                    # select the best
                    _best_cand_assign = _cand_assigns[np.argmax(_assign_scores)]
                    # asign
                    for _ihomo, _j in enumerate(_best_cand_assign):
                        homolog_hzxys_list[_ihomo, _i] = _cand_hzxys[_j]
                        homolog_inds_list[_ihomo, _i] = np.where(self.merged_coords['chr']==_chr_name)[0][np.where(_chr_ids==_i)[0][_j]]
                    #print('case1', _i, len(_cand_hzxys), _best_cand_assign)
                # case 2: if not enough spots assign, let homologs pick:
                else:
                    if allow_overlap:
                        _homo_assigns = list(product(np.arange(_num_homologs), len(_cand_hzxys)))
                    else:
                        _homo_assigns = list(permutations(np.arange(_num_homologs), len(_cand_hzxys)))        
                    _assign_scores = [np.nanmean([_cand_scores[_ihomo, _j] 
                                                 for _j, _ihomo in enumerate(_assign)]) 
                                      for _assign in _homo_assigns]
                    _best_homo_assign = _homo_assigns[np.argmax(_assign_scores)]
                    #print('case2', _i, len(_cand_hzxys), _best_homo_assign)
                    for _j, _ihomo in enumerate(_best_homo_assign):
                        homolog_hzxys_list[_ihomo, _i] = _cand_hzxys[_j]
                        homolog_inds_list[_ihomo, _i] = np.where(self.merged_coords['chr']==_chr_name)[0][np.where(_chr_ids==_i)[0][_j]]
            # save
            self.chr_2_homolog_hzxys_list[_chr_name] = homolog_hzxys_list
            self.chr_2_homolog_inds_list[_chr_name] = homolog_inds_list
            
    def _update_homolog_centers(self, change_shrink=0.8):
        # no previous assignment, exit
        if self.verbose:
            print(f"- Update homolog centers")
        if not hasattr(self, 'chr_2_homolog_hzxys_list'):
            raise AttributeError(f"chr_2_homolog_hzxys_list doesn't exists")
        # store the old centers
        self.history_chr_2_homolog_centers.append({_chr:_cts for _chr, _cts in self.chr_2_homolog_centers.items()})
        for _chr_name, _homolog_hzxys_list in self.chr_2_homolog_hzxys_list.items():
            _old_centers = self.chr_2_homolog_centers[_chr_name].copy()
            _change_vectors = np.nanmean(_homolog_hzxys_list[:,:,1:], axis=(1,)) - _old_centers
            _new_centers = change_shrink * _change_vectors + _old_centers
            self.chr_2_homolog_centers[_chr_name] = _new_centers
    def _determine_selection_changes(self, change_th=0.01):
        # initialize chr_changes
        if not hasattr(self, 'chr_2_change_fraction'):
            self.chr_2_change_fraction = {_chr:1. for _chr in self.chr_2_homolog_centers}
            self.chr_2_change = {_chr:True for _chr in self.chr_2_homolog_centers}
        if len(self.history_chr_2_homolog_hzxys) == 0:
            print("No previous assignment detected, cannot compare.")
            return
        for _chr_name, _inds_list in self.chr_2_homolog_inds_list.items():
            _change_frac = np.mean(self.history_chr_2_homolog_inds[-1][_chr_name] != _inds_list)
            self.chr_2_change_fraction[_chr_name] = _change_frac
            self.chr_2_change[_chr_name] = self.chr_2_change_fraction[_chr_name] > change_th
        return
    def _filter_selected_by_scores(self, detailed_verbose=False):
        if not hasattr(self, 'chr_2_homolog_hzxys_list') or not hasattr(self, 'chr_2_homolog_inds_list'):
            raise AttributeError(f"No selected cooridnates detected, exit!")
        # init
        _score_th = np.sum(self.metricWeights) * self.validScoreTh
        if self.verbose:
            print(f"- Filter selected hzxys by score th={_score_th:.4f}.")
        self.chr_2_filtered_hzxys_list = {}
        self.chr_2_filtered_inds_list = {}
        # process
        for _chr_name in self.chr_2_homolog_hzxys_list:
            _hzxys_list = self.chr_2_homolog_hzxys_list[_chr_name]
            _inds_list = self.chr_2_homolog_inds_list[_chr_name]
            # get picked spot scores
            _chr_scores = np.nan * np.ones_like(_inds_list)
            for _ihomo, _inds in enumerate(_inds_list):
                for _i, _ind in enumerate(_inds):
                    if _ind >= 0:
                        _chr_scores[_ihomo, _i] = self.merged_coords.loc[_ind, f"score_h{_ihomo}"]
            # check whether its pass threshold
            _negative_flags = _chr_scores < _score_th
            if self.verbose and detailed_verbose:
                print(f"-- chr: {_chr_name}, {np.mean(_negative_flags):.3f} removed.")
            # filtered
            _ft_hzxys_list = _hzxys_list.copy()
            _ft_hzxys_list[_negative_flags] = np.nan
            self.chr_2_filtered_hzxys_list[_chr_name] = _ft_hzxys_list
            _ft_inds_list = _inds_list.copy()
            _ft_inds_list[_negative_flags] = -1
            self.chr_2_filtered_inds_list[_chr_name] = _ft_inds_list
        return
    ## Composite functions
    def _first_assignment(self, 
                          init_method='kmeans', min_spot_num=2,
                          local_range=5, 
                          allow_overlap=False):
        # prepare required variables
        self._load_decoded()
        self._merge_decoded()
        if not hasattr(self, 'merged_coords') or len(self.merged_coords) == 0:
            print(f"No coordinates detected, exit.")
            return
        self._generate_default_chr_copyNum()
        # init homologs
        self._init_homolog_centers(method=init_method, min_spot_num=min_spot_num)
        self._prepare_score_metrics(local_range=local_range)
        self._calculate_scores()
        # first round assignment
        self._assign_homologs_by_scores(allow_overlap=allow_overlap,)
        return
    # One step of updating assignment
    def _update_assignment(self, 
                           change_shrink=0.8,
                           local_range=5, 
                           allow_overlap=False,
                           change_th=0.01,
                          ):
        self._update_homolog_centers(change_shrink=change_shrink)
        self._prepare_score_metrics(local_range=local_range)
        self._calculate_scores()
        self._assign_homologs_by_scores(allow_overlap=allow_overlap,)
        self._determine_selection_changes(change_th=change_th)
        return
    # Run iteratively
    def _iterative_assignment(self, 
                              max_niter=10,
                              init_method='kmeans', min_spot_num=2,
                              change_shrink=0.8,
                              local_range=5, 
                              allow_overlap=False,
                              change_th=0.01,
                              filter_by_score=True,
                              detailed_verbose=False,
                             ):
        if not hasattr(self, 'chr_2_homolog_hzxys_list'):
            self._first_assignment(init_method=init_method, min_spot_num=min_spot_num, 
                                   local_range=local_range,
                                   allow_overlap=allow_overlap)
            # Check after first assignment
            if not hasattr(self, 'merged_coords') or len(self.merged_coords) == 0:
                print(f"No coordinates detected, skip iterative assignment.")
                return
            if not hasattr(self, 'chr_2_homolog_hzxys_list') or len(self.merged_coords) == 0:
                print(f"failed to assign homologs, skip iterative assignment.")
                return
        # iters
        for _iter in range(max_niter):
            _iter_start = time.time()
            self._update_assignment(change_shrink=change_shrink,
                                    local_range=local_range,
                                    allow_overlap=allow_overlap,
                                    change_th=change_th)
            # check if termination
            _continue_flags = np.array(list(self.chr_2_change.values()))
            if self.verbose:
                print(f"- Iteration {_iter} in {time.time()-_iter_start:.3f}s. ")
            if not _continue_flags.any():
                break
        if filter_by_score:
            self._filter_selected_by_scores(detailed_verbose=detailed_verbose)
        return
    # Save
    def _save_picked(self, _strict=True, detailed_verbose=False):
        if self.verbose:
            print(f"- Save picked coordiantes into file: {self.saveFile}")
        if self.saveFile is None:
            print(f"saveFile not given, skip saving!")
            if _strict:
                raise ValueError(f"saveFile not given, cannot save anything!")
        # save picked dicts
        save_hdf5_dict(self.saveFile, f"picked/chr_2_homolog_hzxys_list", 
            self.chr_2_homolog_hzxys_list, _overwrite=self.overwrite, _verbose=detailed_verbose)
        save_hdf5_dict(self.saveFile, f"picked/chr_2_homolog_inds_list", 
            self.chr_2_homolog_inds_list, _overwrite=self.overwrite, _verbose=detailed_verbose)
        save_hdf5_dict(self.saveFile, f"picked/chr_2_homolog_centers", 
            self.chr_2_homolog_centers, _overwrite=self.overwrite, _verbose=detailed_verbose)
        save_hdf5_dict(self.saveFile, f"picked/chr_2_scores", 
            self.chr_2_scores, _overwrite=self.overwrite, _verbose=detailed_verbose)
        save_hdf5_dict(self.saveFile, f"picked/chr_2_copyNum", 
            {_chr:np.array([_n]) for _chr,_n in self.chr_2_copyNum.items()}, _overwrite=self.overwrite, _verbose=detailed_verbose)
        # filtered coordinates
        try:
            save_hdf5_dict(self.saveFile, f"picked/chr_2_filtered_hzxys_list", 
                self.chr_2_filtered_hzxys_list, _overwrite=self.overwrite, _verbose=detailed_verbose)
            save_hdf5_dict(self.saveFile, f"picked/chr_2_filtered_inds_list", 
                self.chr_2_filtered_inds_list, _overwrite=self.overwrite, _verbose=detailed_verbose)
        except:
            print(f"failed to save filtered coordinates!")
        # save ref coords and codebook
        try:
            self.merged_codebook.to_hdf(self.saveFile, f"picked/merged_codebook", index=False)
            self.merged_coords.to_hdf(self.saveFile, f"picked/merged_coords", index=False)
        except:
            print(f"failed to save merged decoded info!")
        return
    # Load
    def _load_picked(self):
        if self.verbose:
            print(f"- Load picker from file: {self.saveFile}")
        if self.saveFile is None:
            if self.verbose:
                print(f"saveFile not given, skip loading!")
            return
        if not os.path.exists(self.saveFile):
            print(f"-- savefile:{self.saveFile} not exist, skip")
            return
        try:
            # dicts
            self.chr_2_homolog_hzxys_list = load_hdf5_dict(self.saveFile, f"picked/chr_2_homolog_hzxys_list",)
            self.chr_2_homolog_inds_list = load_hdf5_dict(self.saveFile, f"picked/chr_2_homolog_inds_list",)
            self.chr_2_homolog_centers = load_hdf5_dict(self.saveFile, f"picked/chr_2_homolog_centers",)
            self.chr_2_scores = load_hdf5_dict(self.saveFile, f"picked/chr_2_scores",)
            self.chr_2_copyNum = load_hdf5_dict(self.saveFile, f"picked/chr_2_copyNum",)
            self.chr_2_copyNum = {_chr:_n[0] for _chr,_n in self.chr_2_copyNum.items()}
            # dataframes
            self.merged_codebook = pd.read_hdf(self.saveFile, f"picked/merged_codebook", )
            if self.verbose:
                print(f"-- loading merged_codebook")
            self.merged_coords = pd.read_hdf(self.saveFile, f"picked/merged_coords", )
            # filtered coordinates
            self.chr_2_filtered_hzxys_list = load_hdf5_dict(self.saveFile, f"picked/chr_2_filtered_hzxys_list",)
            self.chr_2_filtered_inds_list = load_hdf5_dict(self.saveFile, f"picked/chr_2_filtered_inds_list",)
            if self.verbose:
                print(f"-- loading merged_coords")
            return True
        except:
            print(f"failed to read picked info.")
            return False

def batch_pick_spots(_decoder_filename, _picked_filename, num_expected_lib=3, 
    weights=default_weights, score_th=default_score_th, max_niter=10, overwrite=False):
    with h5py.File(_decoder_filename, 'r') as _f:
        if len(list(_f.keys())) != num_expected_lib:
            print(f"not enough libraries loaded, {len(list(_f.keys()))} exists, {num_expected_lib} expected. skip.")
            return
    _picker = SpotPicker(_decoder_filename, metric_weights=weights, valid_score_th=score_th,
        save_file=_picked_filename, overwrite=overwrite)
    _picker._iterative_assignment(max_niter=max_niter)
    # save if properly picked
    if hasattr(_picker, 'chr_2_homolog_hzxys_list') and \
        hasattr(_picker, 'chr_2_homolog_hzxys_list') and \
        len(_picker.merged_coords) > 0:
        _picker._save_picked()
    return

def prepare_score_metrics_by_chr(
    hzxys, region_ids, homolog_center_zxys, 
    prev_homolog_hzxys=None, local_range=5,
    ) -> np.ndarray:
    """Function to prepare metrics for scores [intensity, chr_dist, local_dist]"""
    from scipy.spatial.distance import cdist
    # return empty of empty coords
    if len(hzxys) == 0:
        return np.array([])
    if prev_homolog_hzxys is not None and len(prev_homolog_hzxys) != len(homolog_center_zxys):
        raise IndexError(f"length of prev_homolog_hzxys doesn't match")
    # init metric
    _metrics = np.ones([3, len(homolog_center_zxys), len(hzxys)]) * np.nan
    # 1. intensities
    _metrics[0,:,:] = hzxys[:,0]
    # 2. chr_center distance
    _metrics[1,:,:] = cdist(homolog_center_zxys, hzxys[:,1:])
    # 3. chr_local distance
    ## no prev zxys, then assign everything nearby
    if prev_homolog_hzxys is None:
        for _i, _id in enumerate(region_ids):
            _sel_inds = np.where((region_ids>=_id-local_range) & (region_ids<=_id+local_range))[0]
            _sel_inds = np.setdiff1d(_sel_inds, [_i])
            if len(_sel_inds) > 0:
                _zxy = hzxys[_i,1:]
                _dist = np.linalg.norm(_zxy - np.nanmean(hzxys[_sel_inds,1:], axis=0))
                _metrics[2,:,_i] = _dist
    else:
        for _i, _id in enumerate(region_ids):
            _sel_ref_inds = np.arange(max(0,_id-local_range), min(len(prev_homolog_hzxys[0]), _id+local_range+1))
            _sel_ref_inds = np.setdiff1d(_sel_ref_inds, [_i])
            if len(_sel_ref_inds) == 0:
                continue
            for _ihomo, _ref_hzxys in enumerate(prev_homolog_hzxys):
                _zxy = hzxys[_i,1:]
                _dist = np.linalg.norm(_zxy - np.nanmean(_ref_hzxys[_sel_ref_inds,1:], axis=0))
                _metrics[2, _ihomo, _i] = _dist
    return _metrics

def cdf_scores(values, refs, greater=True):
    """Cdf, make sure return value between (0,1) never goes to 0 or 1"""
    if np.isnan(refs).all():
        return np.nan * np.ones(np.shape(values))
    if greater:
        return np.array([percentileofscore(refs, _v, kind='weak')/100 * len(refs) / (len(refs)+2) + 1/(len(refs)+2) 
                         for _v in values])
    else:
        return np.array([1 - percentileofscore(refs, _v, kind='weak')/100 * len(refs) / (len(refs)+2) - 1/(len(refs)+2) 
                         for _v in values])

# required functions to load/save hidf5 dicts         
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
def load_hdf5_dict(_filename, data_key):
    _dict = {}
    with h5py.File(_filename, 'r') as _f:
        if data_key not in _f.keys():
            return None
        _grp = _f[data_key]
        for _key in _grp.keys():
            _dict[_key] = _grp[_key][:]
    return _dict