
import sys,glob,os, time
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm

from .. import get_img_info, corrections, visual_tools
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy
from . import scoring, checking
from scipy.spatial.distance import pdist, squareform




### Scripts in progress for the 2-tier picking method with ndarray as input and output
### Most functions below are expected to run sequentially to match array dimension



### Function to clustered chromosome(s) using given distance  @ Shiwei Liu
### Initially, distance (radius) for the cluster can be roughly set by plotting all spots in the graph; 
### Optioinally, distance can be iteraitvely modifed until the ratio of good_region peaks (using other functions below)
### Chromosome (regions) beyond the radius is considered unlikely to interact
### After this, for chromosome cluster containing a single chromosome (aka, isolated chromosme), spot picking can be done independent of other chromosomes from other clusters;
### for chromosome cluster containing multiple chromosomes, spot picking can be done using K-means since K is known.
def label_and_cluster_chromosome_by_distance (_chrom_azyxi_array, 
                                           _distance =2000, 
                                           _verbose = True):
    
    '''Function to label individual chrom and cluster chrom azyxi array
       
        -Input: 
             _chrom_azyxi_array: a pre-processed nd array; each row is a chrom object, with a (pixel area), z (nm), y (nm), x (nm), i (gene_id) as element in each column
             _distance: the radius distance (nm) to find chrom neighbor to construct chrom clusters 
             
        - Output: 
             _chrom_azyxiuc_array: u (chr unique id) and c (chr cluster id, which is the smallest unqie id if the chr within the cluster) are appended in the input nd array in addition to azyxi
             Note for usage: the u and c can later be used for slicing accordingly to allow spot assigning and picking'''    
    
    if not isinstance (_chrom_azyxi_array, np.ndarray):
        if _verbose:
            print ('-- chrom input should be np.ndarray.')
        return None
    else:
        if len(_chrom_azyxi_array.shape) != 2 or  _chrom_azyxi_array.shape[-1] != 5:
            if _verbose:
                print ('-- each chrom should have 5 elements, which are [area, z, y, x, gene_id].')
            return None
        else:
            if _verbose:
                print ('-- finding neighbor and labeling each chrom with unique chr id as the 6th element.')
                # add result cols to store unique chr id and the found chr cluster id
                _result_cols = np.zeros([len(_chrom_azyxi_array), 2])
                _chrom_azyxi_array_to_cluster = np.hstack((_chrom_azyxi_array.copy(), _result_cols))              
    
    ### 1. loop through chrom to find neighbor(s) (including itself) within the given distance
    # initiate flag to record chr i
    _flag_1 = 0
    _all_neigh_ids = []  # chr id of all neighbor(s) for each chr
    for _chrom_coord_i in _chrom_azyxi_array_to_cluster:
        # flag to record the current chr i
        _flag_1 +=1
        _chrom_coord_i [-2] = _flag_1  # add as unique chr id to the 6th column of the chrom array
        # initiate flag to record chr j
        _flag_2 = 0
        # initiate count for chr neighbors
        _num_neigh = 0
        _neigh_ids = []
        for _chrom_coord_j in _chrom_azyxi_array_to_cluster:
            # flag to record the current chr j
            _flag_2 +=1
            # test if eculidiean distance smaller than the given distance
            if np.linalg.norm (_chrom_coord_i [1:4] - _chrom_coord_j [1:4]) < _distance:
                _num_neigh += 1
                _neigh_ids.append(_flag_2)
        
        _all_neigh_ids.append(_neigh_ids)
        
    ### 2. join redundant chr cluster with shared nodes (chrs) using networkx so each cluster is mutually exclusive to other cluster
    # for example for chr 1,2,3,4,5,6,7,8, if their neigh (including themselves) are [[1,2],[2,1,5],[3],[4],[5,2],[6],[7,8],[8,7]], 
    # then the clusters would be [{1, 2, 5}, {3}, {4}, {6}, {7,8}]
    import networkx 
    from networkx.algorithms.components.connected import connected_components
    if _verbose:
        print ('-- constructing chromosome clusters')

    def ids_to_edges (_ids):
        _iter = iter(_ids)
        _last = next (_iter)  
        for _curr in _iter:
            yield _last, _curr
            _last = _curr

    # find nodes in the network
    _G = networkx.Graph()
    for _ids in _all_neigh_ids:
        # each chr is a node with given edges (connection)
        _G.add_nodes_from (_ids)
        _G.add_edges_from(ids_to_edges(_ids))  # use list_to_edges generator above
    
    # cluster set where each unique chr has at least one neighbor within the given distance
    _chr_cluster_set = [_component for _component in connected_components (_G)]
        
    ### 3. label each chr cluster using the containing chr with smallest index
    if _verbose:
        print ('-- labeleling chromosome cluster id as the 7th element')
    for _chrom in _chrom_azyxi_array_to_cluster:
        for _cluster in _chr_cluster_set:
            if _chrom [-2] in _cluster:
                _chrom [-1] = min(_cluster)      
    
    _chrom_azyxiuc_array = _chrom_azyxi_array_to_cluster
    
    return _chrom_azyxiuc_array





### Function to append to the closest chromosome id to the spot info array; the chromosome id is the same as the unique chr id generated in the function above.
def find_closest_chromosome_for_spots (_chrom_azyxiuc_array,_spots_hzxyi_array, _verbose = True):
    
    '''Function to find the closest (ecludiean distance) chromosome center for each spot
    
       -Inputs: 
           _chrom_azyxiuc_array: pre-processed chrom nd array with info below; 
           each row is a chrom object, with a (pixel area), z (nm), y (nm), x (nm), i (gene_id), u (unique chr id), c(cluster id) as elements in each column

           _spots_hzxyi_array: pre-processed spot nd array with info below;
           each row is a spot object, with h(intensity), z (nm), y (nm), x (nm), i (region_id) as elements in each column

        -Output: 
           _spots_hzxyid_array: d (the chr unique id for the closest chr found) as the 6th element in addition to azyxi'''
            
    
    if not isinstance (_chrom_azyxiuc_array, np.ndarray) or not isinstance (_spots_hzxyi_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        if _chrom_azyxiuc_array.shape[-1] == 7 and _spots_hzxyi_array.shape[-1] == 5:   
            # add empty column to store the results for the closest chrom for each spot
            _result_cols = np.zeros([len(_spots_hzxyi_array), 1])
            _spots_hzxyi_array_closest = np.hstack((_spots_hzxyi_array.copy(), _result_cols))  
        else:
            if _verbose:
                print ('-- chrom and spot inputs are not valid.')
            return None
    ### 1. calculate element-wise distance map
    from scipy.spatial.distance import cdist
    _dist_matrix =  cdist(_spots_hzxyi_array[:,1:4],_chrom_azyxiuc_array[:,1:4])
    
    if _verbose:
        print('-- add the chromosome id of the closest chromosome for each spot')
    ### 2. append the closest chromosome id
    for _index, _spot in enumerate(_spots_hzxyi_array_closest):
        _spot [-1] = _chrom_azyxiuc_array[np.argmin(_dist_matrix[_index])][-2]   # u in chr azxyiuc
        
    _spots_hzxyid_array = _spots_hzxyi_array_closest # d means the closest chrom id by distance

    return _spots_hzxyid_array



### Function to assign spots to a given chromosome cluster with spot region  
def assign_spots_to_chromosome_cluster_by_distance (_chromosome_cluster, 
                                                    _spots_hzxyid_array, 
                                                    #_region_ids = None, 
                                                    _distance = 2000,
                                                    _verbose = True):
    
    """Function to assign spots to given chromosome cluster by distance
    
        Input:
           _chromosome_cluster: selected chromsome cluster, which should be a subset of _chrom_azyxiuc_array with the same cluster id
           _spots_hzxyid_array: a preprocessed spot array after closet chr is found (see function directly above);
                                each row is a spot object, with h(intensity), z (nm), y (nm), x (nm), i (region_id) as elements in each column;
                                to enable the use of function during loop, this can also be a _spots_hzxyida_array, where spots may have been assigned (a)
           _region_ids: the list/np.ndarray for all the combo/region ids
           _distance: the given distance to find all spots within the distance radius; use the same distance as the one used for chromosome clustering to avoid missing/repeating spots
        
        Output:
           _spots_hzxyida_array: a (assigned chr cluster id) is appended or modified as the 7th element in addition to hzxyid"""
    
    if not isinstance (_chromosome_cluster, np.ndarray) or not isinstance (_spots_hzxyid_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        # chrom cluster with single chromosome
        if _chromosome_cluster.shape == (7,): 
            _chromosome_clusters = np.array([_chromosome_cluster])
        # chrom cluster with more than one chromosome
        elif len(_chromosome_cluster) >= 1 and _chromosome_cluster.shape[-1] == 7:
            _chromosome_clusters = _chromosome_cluster
        else:
            if _verbose:
                print ('-- chrom inputs are not valid.')
            return None

    ### 0. preprocess spots array if needed
    # for spot array that has never been assigned
    if _spots_hzxyid_array.shape[-1] == 6: 
        # add empty column to store results for spot's chromosome assignment 
        _result_cols = np.zeros([len(_spots_hzxyid_array), 1])
        _spots_hzxyid_array_to_assign = np.hstack((_spots_hzxyid_array.copy(),_result_cols))
    # for spot array that has been assigned, -> hzxyida array
    # where non-assigned spots should still remain zero as their 'a (assign cluster id)' column   
    elif _spots_hzxyid_array.shape[-1] == 7: 
        _spots_hzxyid_array_to_assign = _spots_hzxyid_array.copy()

    else:
        if _verbose:
            print ('-- spot inputs are not valid.')
        return None

    
    ### 1. assing spots to chromosome cluster by given distance
    # loop through all chr within the cluster
    _cluster_id = _chromosome_clusters [0,-1]
    if _verbose:
        print (f'-- assigning spots to the chromosome cluster {int(_cluster_id)}.')
    
    for _chr in  _chromosome_clusters:  # _azyxiuc array subset
        
        for _spot in _spots_hzxyid_array_to_assign:
            if np.linalg.norm (_spot[1:4] - _chr [1:4]) < _distance:
                # append the cluster id;
                # and this would not be overwritten once generated since all chr here has the same cluster id
                # additionaly, spots outside cluster would remain zero as each cluster is independent as long as the same distance used
                _spot[-1] = _cluster_id
    
    _spots_hzxyida_array = _spots_hzxyid_array_to_assign
    
    return _spots_hzxyida_array




### Function to estimate the distance radius setting and staining quality using the assigned spots for a given chromosome cluster.
def calculate_region_quality (_chromosome_cluster, 
                              _spots_hzxyida_array, 
                              _region_ids = None,                       
                              _isolated_chr_only = False,
                              _verbose = True):

    """Function to_report_basic quality of staining etc
        Input:
           _chromosome_cluster: selected chromsome cluster, which should be a subset of _chrom_azyxiuc_array with the same cluster id
           _spots_hzxyida_array: a preprocessed and assigned spot array after spot assignment (see function directly above);
                                 each row is a spot object, with h(intensity), z (nm), y (nm), x (nm), i (region_id), a (assigned chr cluster id) as elements in each column
           _region_ids: the list/np.ndarray for all the combo/region ids
           _isolated_chr_only: assess only isolated chromosome or not
        Output: 
            if _isolated_chr_only: True, only isolated chromosme will be used to report quality; mutlti-chrom cluster will return None"""

    if not isinstance (_chromosome_cluster, np.ndarray) or not isinstance (_spots_hzxyida_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        # chrom cluster with single chromosome
        if _chromosome_cluster.shape == (7,): 
            _chromosome_clusters = np.array([_chromosome_cluster])
        # chrom cluster with more than one single chromosome
        elif len(_chromosome_cluster) > 1 and _chromosome_cluster.shape[-1] == 7:
            _chromosome_clusters = _chromosome_cluster
        else:
            if _verbose:
                print ('-- chrom inputs are not valid.')
            return None
    
    if _spots_hzxyida_array.shape[-1] != 7: 
        if _verbose:
            print ('-- spot inputs are not valid.')
        return None

    ### 1. process the spots assigned to the specified chromosome cluster
    _cluster_id = _chromosome_clusters [0,-1]
    
    if _verbose:
        print (f'-- calculating region quality for chromosome cluster {_cluster_id}', end=' ')

        # slice only spots belong to the processed cluster
    _spots_sel_cluster = _spots_hzxyida_array[_spots_hzxyida_array[:,-1] == _cluster_id]   # a in spot hzxyida
        
    _confident_region =0  #  region where number of spots match the number of chr 
    _ambiguous_region = 0  # region where number of spots <= 2* number of chr
    _bad_region = 0      # region where number of spots > 2* number of chr
    _loss_region = 0    #  region where number of spots < the number of chr 
            
    # assess regions that are specified 
    if isinstance(_region_ids, list) or isinstance (_region_ids, np.ndarray):
        if len(_region_ids) > 0: 
            if _verbose:
                print ('using specified region ids.')
            _region_ids = _region_ids
    else:
    # assess all regions that has at least one spot
        if _verbose:
            print ('using all region ids from the assigned spots.')
            print ("-- note if one region is completely missed, it cannot be reported.")
        _region_ids = np.unique(_spots_sel_cluster[:,4])      # i in hzxyida
           
            
    for _region_id in _region_ids:
            # slice spots in selected regions
        _spots_sel_region =_spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id]   # i in spot hzxyida
            # record number for each category 
        if len(_spots_sel_region) == len(_chromosome_cluster): # if distance radius was given well, this number should plateau and decrease if keeping increasing the distance
            _confident_region +=1
        elif len(_spots_sel_region) <= 2* len(_chromosome_cluster):
            _ambiguous_region +=1
        elif len(_spots_sel_region) > 2* len(_chromosome_cluster):  # if this is the majority, suggesting a false negative (undetected) chromosome center in this cluster
            _bad_region +=1
        elif len(_spots_sel_region) < len(_chromosome_cluster):  # if this is the majority, suggesting a false postive chromosome center or the distance radius was given too small
            _loss_region +=1
        
    if _isolated_chr_only and len(_chromosome_cluster) !=1:
        if _verbose:
            print ('-- not an isolated chromosome, region reporting skipped')
        return np.array([_cluster_id, np.nan,np.nan,np.nan,np.nan])

    else:
        return np.array([_cluster_id, _confident_region,_ambiguous_region,_bad_region,_loss_region])




### Function to pick assigned spots to the given isolated chromosome
def pick_spots_for_isolated_chromosome (_isolated_chromosome, 
                                        _spots_hzxyida_array, 
                                        _region_ids = None, 
                                        _neighbor_len = None, 
                                        _iter_num = 2, 
                                        #_dist_ratio = 0.8, 
                                        #_h_ratio = 0.8,
                                        _local_dist_th = 2000,
                                        _verbose = True):
    """Function to pick spots for isolated chromosome
       
          Input: 
              _isolated_chromosome: selected chromsome (cluster) which should be in a format of _chrom_azyxiuc_array 
              _spots_hzxyida_array: preprocessed spot array, each row is a spot object;
                                    with h(intensity), z (nm), y (nm), x (nm), i (region_id), d (chr id by closest distance), a (assigned chr cluster id) as elements in each column
                                alternatively, this array can also be hzyxidap that is picked.
              _region_ids: the list/np.ndarray for all the combo/region ids
              _neighbor_len: the number of neighboring region on each side for the region being picked
              _iter_num: number of interation to re-pick spots from multiple candidates
              #_dist_ratio: the ratio used to compare distance when multiple candidates are very close 
              #_h_ratio: the ratio used to compare intensity when multiple candidates are very close 
              _local_dist_th: the biggest distance between a spot and its local neighbor center
          
          Output:
              _spots_hzxyidap_array: the spots array where the picked chr id is added as the 8th element (p)"""

    
    if not isinstance (_isolated_chromosome, np.ndarray) or not isinstance (_spots_hzxyida_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        # chrom cluster should contain one single chromosome;
        if _isolated_chromosome.shape == (7,):
            _isolated_chromosome = np.array([_isolated_chromosome])
            _isolated_chromosome_id = _isolated_chromosome [0,-1]
        elif _isolated_chromosome.shape == (1,7):   
            _isolated_chromosome_id = _isolated_chromosome [0,-1]    # azyxiuc  chr id is the same as cluster id for isolated chromosome
        else:
            if _verbose:
                print ('-- chrom inputs are not valid.')
            return None

    ### 0. preprocess spots hzyxida array if necessary
    # for spot array that has never been picked
    if _spots_hzxyida_array.shape[-1] == 7: 
        # add empty column to store results for spot's chromosome assignment 
        _result_cols = np.zeros([len(_spots_hzxyida_array), 1])
        _spots_hzxyida_array_to_pick = np.hstack((_spots_hzxyida_array.copy(),_result_cols))
    # for spot array that has been pikced -> hzxyidap array
    # where picked spots from other chromosome cluster should still remain zero in 'p (picked chr id)' column   
    elif _spots_hzxyida_array.shape[-1] == 8: 
        _spots_hzxyida_array_to_pick = _spots_hzxyida_array.copy()
    else:
        if _verbose:
            print ('-- spot inputs are not valid.')
        return None
    
    # slice only spots belong to the processed cluster/isolated chromosome to modify
    _spots_sel_cluster = _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,-2]== _isolated_chromosome_id]

    # if regions that are specified 
    if isinstance(_region_ids, list) or isinstance (_region_ids, np.ndarray):
        if len(_region_ids) > 0: 
            if _verbose:
                print ('using specified region ids.')
            _region_ids = _region_ids
    else:
    # if not specified, use all regions that has at least one spot
        if _verbose:
            print ('using all region ids from the assigned spots.')
            print ("-- note if one region is completely missed, it cannot be reported.")
        _region_ids = np.unique(_spots_sel_cluster[:,4])      # i in hzxyida

    if _neighbor_len is None:
        _neighbor_len = round(len(_region_ids/10))   # use 10 + 10 % of region length for local reference
    
    ### 1. initial picking
    # identify good regions as initial ref regions
    _ref_ids = []
    for _region_id in _region_ids:
        # slice spots copy with given region id
        _spots_sel_region =_spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id]   # i in spot hzxyida(p)
        if len(_spots_sel_region) == 1:   # pick for the good region where any one candidate spot
            # modify the _spots_sel_region 
            _spots_sel_region [:,-1]= _isolated_chromosome_id
            # modify the _spots_sel_cluster
            (_spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id]) = _spots_sel_region  # add the picked chr id to the initial _spots_sel_cluster 
            _ref_ids.append(_region_id)
    _ref_ids =np.array(_ref_ids)
    # pick for regions with multiple candidates
    for _region_id in _region_ids:
        # slice spots with given region id
        _spots_sel_region =(_spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id])   # i in spot hzxyida(p)
        # pick spots for regions with more than one candidate spot
        if len(_spots_sel_region) > 1:  
            _start, _end = max(_region_id - _neighbor_len, min (_region_ids)), min(_region_id + _neighbor_len, max(_region_ids))   # from 1 to max(_region_ids)
            _sel_local_ids = np.concatenate([np.arange(_start, _region_id), np.arange(_region_id +1, _end+1)])
            _shared_local_ids = np.intersect1d(_sel_local_ids, _ref_ids)
            # if there are more than 2 good/confident regions around, use the mean as the ref point
            if len(_shared_local_ids) >= 3:
                _neighbor_spots = []
                for _shared_local_id in _shared_local_ids:
                    _spot_ref_region =_spots_sel_cluster[_spots_sel_cluster[:,4] == _shared_local_id]
                    _neighbor_spots.append(_spot_ref_region[0])   
                _ref_center = np.nanmean(_neighbor_spots,axis=0) [1:4]    # zyx only
            # if there less than 3 good/confident regions around, use the mean of all good currently picked regions if there are more than 1/3 of the total regions
            else:
                _picked_spots = _spots_sel_cluster[_spots_sel_cluster [:,-1] ==_isolated_chromosome_id]
                if len(_picked_spots) > round(len(_region_ids)/3):
                    _ref_center = np.nanmean(_picked_spots,axis=0) [1:4] 
                # else, use the chrom center
                else:
                    _ref_center = _isolated_chromosome[0,1:4].copy()  # zyx only
            
            # compare candidate spots and pick the one closest to the ref center: the picked id should be saved in the parental _spots_hzxyida_array_to_pick array
            _dist_to_ref_list = []
            for _cand_spot in _spots_sel_region:
                _dist_to_ref = np.linalg.norm (_cand_spot[1:4] - _ref_center)
                _dist_to_ref_list.append(_dist_to_ref)
            _dist_to_ref_list = np.array(_dist_to_ref_list)
            # use VIEW to modify the _spots_sel_region and then the initial _spots_sel_cluster 
            if min(_dist_to_ref_list) < _local_dist_th:   # use local distance th for the closest spot
                _spots_sel_region[np.argmin(_dist_to_ref_list),-1]= _isolated_chromosome_id 
            (_spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id]) = _spots_sel_region # add the picked chr id to the _spots_sel_cluster 

    ### 2. secondary picking
    _iter = 0
    while _iter < _iter_num:
        if _verbose:
            print (f'-- start iteration round {_iter + 1} using all picked spots.')

        for _region_id in _region_ids:
             # slice spots with given region id
            _spots_sel_region =_spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id]   # i in spot hzxyidap
             # load currently picked spots
            _picked_spots = _spots_sel_cluster[_spots_sel_cluster [:,-1] ==_isolated_chromosome_id]    # spots that are picked (p) in hzyxdap array
            _ref_ids_from_picked = np.unique(_picked_spots[:,4])  # use all picked region from the last picking

            if len(_spots_sel_region) >= 1:   # good region will also be assessed so it is not too far away from other local neighbors
                _start, _end = max(_region_id - _neighbor_len, min (_region_ids)), min(_region_id + _neighbor_len, max(_region_ids))   # from 1 to max(_region_ids)
                _sel_local_ids = np.concatenate([np.arange(_start, _region_id), np.arange(_region_id +1, _end+1)])
                _shared_local_ids = np.intersect1d(_sel_local_ids, _ref_ids_from_picked)
                # use local ref center
                if len(_shared_local_ids) >= 3:
                    _neighbor_spots = []
                    for _shared_local_id in _shared_local_ids:
                        _spot_ref_region =_picked_spots[_picked_spots[:,4] == _shared_local_id]
                        _neighbor_spots.append(_spot_ref_region[0])
                    _ref_center = np.nanmean(_neighbor_spots,axis=0) [1:4]     # zyx only
               
                else:
                     # use picked spots if more than 33% of the total regions
                    if len(_picked_spots) > round(len(_region_ids)/3):
                        _ref_center = np.nanmean(_picked_spots,axis=0) [1:4]     # zyx only 
                     # use chrom center
                    else:
                        _ref_center = _isolated_chromosome[0,1:4].copy()  # zyx only 

                 # compare candidate spots and pick the one closest to the ref center: the picked id should be saved in the parental _spots_hzxyida_array_to_pick array
                _dist_to_ref_list = []
                for _cand_spot in _spots_sel_region:
                    _dist_to_ref = np.linalg.norm (_cand_spot[1:4] - _ref_center)
                    _dist_to_ref_list.append(_dist_to_ref)
                _dist_to_ref_list = np.array(_dist_to_ref_list)

                # use VIEW to modify the _spots_sel_region and then the initial _spots_sel_cluster 
                # reset the picked id for all spots in this region
                _spots_sel_region [:,-1]= 0
                # use VIEW to re-pick for this region 
                if min(_dist_to_ref_list) < _local_dist_th:   # use local distance th for the closest spot
                    _spots_sel_region[np.argmin(_dist_to_ref_list),-1]= _isolated_chromosome_id 
     
                # use VIEW to re-pick for this region in the working _spots_sel_cluster
                _spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id] = _spots_sel_region

        _iter +=1
    
    # finally, assign the working _spots_sel_cluster back to the original parental array
    _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,-2]== _isolated_chromosome_id] = _spots_sel_cluster 
    _spots_hzxyidap_array = _spots_hzxyida_array_to_pick.copy()
    if _verbose:
        print (f'-- finish spot picking for chromosome cluster {_isolated_chromosome_id}.')

    return _spots_hzxyidap_array
                

                






def pick_spots_for_multi_chromosomes ():

    return