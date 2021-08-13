
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
                                                    _region_ids = None, 
                                                    _distance = 2000,
                                                    _report_quality = False,
                                                    _isolated_chr_only = True,
                                                    _verbose = True):
    
    """Function to assign spots to given chromosome cluster by distance
    
        Input:
           _chromosome_cluster: selected chromsome cluster, which should be a subset of _chrom_azyxiuc_array with the same cluster id
           _spots_hzxyid_array: a preprocessed spot array after closet chr is found (see function directly above);
                                to enable the use of function during loop, this can also be a _spots_hzxyida_array, where spots may have been assigned
           _region_ids: the list/np.ndarray for all the combo/region ids
           _distance: the given distance to find all spots within the distance radius; use the same distance as the one used for chromosome clustering to avoid missing/repeating spots
           _report_quality: if reporting, the function returns additonal output for region quality
           _isolated_chr_only: if True, only isolated chromosme will be used to report quality; in this case, only resulting array will be returned
        
        Output:
           _spots_hzxyida_array: a (assigned chr cluster id) is appended or modified as the 7th element in addition to hzxyid'
           if reporting quality, quality result is also returned"""
    
    if not isinstance (_chromosome_cluster, np.ndarray) or not isinstance (_spots_hzxyid_array, np.ndarray):
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
    
    # for spot array that has never been assigned
    if _spots_hzxyid_array.shape[-1] == 6: 
        # add empty column to store results for spot's chromosome assignment 
        _result_cols = np.zeros([len(_spots_hzxyid_array), 1])
        _spots_hzxyid_array_to_assign = np.hstack((_spots_hzxyid_array.copy(),_result_cols))
    # for spot array that has been assigned, -> hzxyida array
    # where non-assigned spots should still remain zero as their 'a (assign cluster id)' column   
    if _spots_hzxyid_array.shape[-1] == 7: 
        _spots_hzxyid_array_to_assign = _spots_hzxyid_array.copy()
    
    ### 1. assing spots to chromosome cluster by given distance
    # loop through all chr within the cluster
    _cluster_id = _chromosome_cluster [0,-1]
    if _verbose:
        print (f'-- assigning spots to the chromosome cluster {int(_cluster_id)}.')
    
    for _chr in  _chromosome_cluster:  # _azyxiuc array
        
        for _spot in _spots_hzxyid_array_to_assign:
            if np.linalg.norm (_spot[1:4] - _chr [1:4]) < _distance:
                # append the cluster id;
                # and this would not be overwritten once generated since all chr here has the same cluster id
                # additionaly, spots outside cluster would remain zero as each cluster is independent as long as the same distance used
                _spot[-1] = _cluster_id
    
    _spots_hzxyida_array = _spots_hzxyid_array_to_assign
    
    ### 2. if reporting region quality 
    if _report_quality:
        if _verbose:
            print ('-- calculating region quality', end=' ')
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
            
            if _verbose:
                print ('using all region ids from the assigned spots.')
                print ("-- note if one region is completely missed, it cannot be reported.")
            _region_ids = np.unique(_spots_sel_cluster[:,4])      # i in hzxyida
           
            
        for _region_id in _region_ids:
                # slice spots in selected regions
            _spots_sel_region =_spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id]   # i in spot hzxyida
            # record number for each category 
            if len(_spots_sel_region) == len(_chromosome_cluster):
                _confident_region +=1
            elif len(_spots_sel_region) <= 2* len(_chromosome_cluster):
                _ambiguous_region +=1
            elif len(_spots_sel_region) > 2* len(_chromosome_cluster):
                _bad_region +=1
            elif len(_spots_sel_region) < len(_chromosome_cluster):
                _loss_region +=1
        
        if _isolated_chr_only and len(_chromosome_cluster) !=1:
            if _verbose:
                print ('-- not an isolated chromosome, region reporting skipped')
            return _spots_hzxyida_array
        else:
            return _spots_hzxyida_array, np.array([_confident_region,_ambiguous_region,_bad_region ,_loss_region])
        
    else:
        return _spots_hzxyida_array



def pick_spots_for_isolated_chromosome ():




    return





def pick_spots_for_multi_chromosomes ():

    return