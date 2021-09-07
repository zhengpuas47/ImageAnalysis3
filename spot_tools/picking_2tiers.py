
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



### Functioin to generate a hzxyi (for spot) or azyxi (for chrom) array 
def convert_spots_array_to_hzxyi_array(spot_array, pix_size=_distance_zxy, normalize_spot_background=False):
    """Function to convert spot array whose 1:4 elements are zxy into hzxyi array where i is added as the 5th element
       h/a (the first element) is kept as it is, zxy is converted from pixel to nm; 
       i is appended as the 5th element;
       For hzxy-11 element spot array, i is the region id for the hzxyi output;
       For azxyi chrom array, i is the gene id for the azxyi ouput"""
    
    _hzxys_list = []
    pix_size = np.array(pix_size)
    for _spots in spot_array:
        if len(_spots) == 0:
            _hzxys_list.append([])
        else:
            _spots = np.array(_spots).copy()
            if len(np.shape(_spots)) == 1:
                _hzxy = _spots[:1+len(pix_size)]
                _hzxy[1:1+len(pix_size)] = _hzxy[1:1+len(pix_size)] * pix_size
                if normalize_spot_background:
                    #print(_spots[2+len(pix_size)])
                    _hzxy[0] = _hzxy[0] / _spots[1+len(pix_size)]
                _hzxys_list.append(_hzxy)
            elif len(np.shape(_spots)) == 2:
                _hzxys = _spots[:, :1+len(pix_size)]
                _hzxys[:,1:1+len(pix_size)] = _hzxys[:,1:1+len(pix_size)] * pix_size
                if normalize_spot_background:
                    _hzxys[:,0] = _hzxys[:,0] / _spots[:,1+len(pix_size)]
                _hzxys_list.append(_hzxys)
            else:
                raise IndexError(f"_spots should be 1d or 2d array.")
                
    _hzxys_array = np.array(_hzxys_list) 
     # array of 4 elements (hzxy) for each row above
    _hzxyi_array = np.hstack((_hzxys_array, spot_array[:,-1][:, np.newaxis])) 
    # array of 5 elements (hzxyi) for each row above
     
    return _hzxyi_array



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
### If distance here is set to be >= 1/2 of the distance used above for finding chromosome cluster, some spots may be assigned to multiple cluster;
### We can re-assess these spots later after picking since most of them are likely to be quite far away from the chromosme center, thus unlikely to get picked
def assign_spots_to_chromosome_cluster_by_distance (_chromosome_cluster, 
                                                    _spots_hzxyid_array, 
                                                    #_region_ids = None, 
                                                    _distance = 2000,
                                                    _verbose = True, 
                                                    _batch = False):
    
    """Function to assign spots to given chromosome cluster by distance
    
        Input:
           _chromosome_cluster: selected chromsome cluster, which should be a subset of _chrom_azyxiuc_array with the same cluster id
           _spots_hzxyid_array: a preprocessed spot array after closet chr is found (see function directly above);
                                each row is a spot object, with h(intensity), z (nm), y (nm), x (nm), i (region_id) as elements in each column;
                                to enable the use of function during loop, this can also be a _spots_hzxyida_array, where spots may have been assigned (a)
           _region_ids: the list/np.ndarray for all the combo/region ids
           _distance: the given distance to find all spots within the distance radius; 
           _NOTE: if use the same distance as the one used for chromosome clustering, it will include a small number of repeating/duplicating spots (in the batch function), 
                the repeating spots can be re-assessed after picking if one spot has been picked multiple times; 
                thus it is okay to set a distance where majority of spots are seeminly exlcusive located within one cluster 
        
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
                _spot[6] = _cluster_id
    
    _spots_hzxyida_array = _spots_hzxyid_array_to_assign.copy()

    if _batch:
        _sel_spots_cluster = _spots_hzxyida_array[_spots_hzxyida_array[:,6]==_cluster_id]
        return _sel_spots_cluster
    else:
        return _spots_hzxyida_array



### BATCH Function to assign spots for all chromosomes cluster from the parental chrom azyxiuc array
def batch_assign_spots_to_chromosome_cluster_by_distance (_chrom_azyxiuc_array, 
                                                          _spots_hzxyid_array, 
                                                          _distance = 2000,
                                                          _verbose = True,
                                                          _num_threads =20):
    '''Batch function to assign spots;
        
        Use the function above but process individual chrom cluster parrallelly'''

    if not isinstance (_chrom_azyxiuc_array, np.ndarray) or not isinstance (_spots_hzxyid_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        # chrom cluster with single chromosome
        if _chrom_azyxiuc_array.shape [0] > 1 and _chrom_azyxiuc_array.shape [1] == 7:
            _chromosome_cluster_ids = np.unique(_chrom_azyxiuc_array[:,-1])
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
    
    # generate args for starmap 
    _batch = True
    _assign_spot_args = []
    for _chromosome_cluster_id in _chromosome_cluster_ids:
        _assign_spot_args.append((_chrom_azyxiuc_array[_chrom_azyxiuc_array[:,-1]==_chromosome_cluster_id], _spots_hzxyid_array_to_assign, _distance, _verbose, _batch))
    
    import multiprocessing as mp
    with mp.Pool(_num_threads,) as _spot_pool:
        if _verbose:
            print (f'-- start multiprocessing assign spots to chromosome clusters with {_num_threads} threads', end=' ')
            _multi_time = time.time()
    # Multi-proessing!
        _spot_pool_result = _spot_pool.starmap(assign_spots_to_chromosome_cluster_by_distance, _assign_spot_args)
        # close multiprocessing
        _spot_pool.close()
        _spot_pool.join()
        _spot_pool.terminate()
        if _verbose:
            print(f"in {time.time()-_multi_time:.3f}s.")
    
    # combine all results; note that spots within radius of multiple clusters will be kept
    _spots_hzxyida = np.vstack(_spot_pool_result)

    return _spots_hzxyida





### Function to estimate the distance radius setting and staining quality using the assigned spots for a given chromosome cluster.
def calculate_region_quality (_chromosome_cluster, 
                              _spots_hzxyida_array, 
                              _gene_region_ids_dict = {},
                              _region_ids = None,                       
                              _isolated_chr_only = False,
                              _verbose = True):

    """Function to_report_basic quality of staining etc
        Input:
           _chromosome_cluster: selected chromsome cluster, which should be a subset of _chrom_azyxiuc_array with the same cluster id
           _spots_hzxyida(p)_array: a preprocessed and assigned spot array after spot assignment (see function directly above);
                                 each row is a spot object, with h(intensity), z (nm), y (nm), x (nm), i (region_id), a (assigned chr cluster id) as elements in each column
                                 it can also be hzxyidap array if the interest of cluster has not been picked
           _region_ids: the list/np.ndarray for all the combo/region ids
           _isolated_chr_only: assess only isolated chromosome or not
           _gene_region_ids_dict: dict.keys as gene id; dict.values are lists of valid regiond ids for corresponding gene
           _batch: different format of output for batch function below
        Output: 
            if _isolated_chr_only: True, only isolated chromosme will be used to report quality; mutlti-chrom cluster will return None"""

    if not isinstance (_chromosome_cluster, np.ndarray) or not isinstance (_spots_hzxyida_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        # chrom cluster with single chromosome
        if _chromosome_cluster.shape == (7,) or _chromosome_cluster.shape == (7,): 
            _chromosome_clusters = np.array([_chromosome_cluster])
        # chrom cluster with more than one single chromosome
        elif len(_chromosome_cluster) >= 1 and _chromosome_cluster.shape[-1] == 7:
            _chromosome_clusters = _chromosome_cluster
        else:
            if _verbose:
                print ('-- chrom inputs are not valid.')
            return None
    
    if _spots_hzxyida_array.shape[-1] < 7:   # hzyxidap array is also supported [note the relevant regions should have not been picked for the purppose of assessing quality]
        if _verbose:
            print ('-- spot inputs are not valid.')
        return None


    # use _gene_region_ids_dict to refine _region_ids 
    # _shared_gene_region_ids_dict for chrom in the cluster and specified region ids
    
    import copy
    _default_region_ids = copy.deepcopy(_region_ids)
    _shared_gene_region_ids_dict = {}
    for _chr in _chromosome_cluster:
        _gene_id = _chr [4]
        if _gene_id in _gene_region_ids_dict.keys():
            _gene_region_ds = _gene_region_ids_dict[_gene_id]
            if len(np.intersect1d(_gene_region_ds, _default_region_ids)) >0:
                _shared_region_ids = np.intersect1d(_gene_region_ds, _default_region_ids)
            else:
                if _verbose:
                    print ('-- no region ids avalible to be analyzed.')
                return None
        else:
            _shared_region_ids = _default_region_ids

        _shared_gene_region_ids_dict[_chr[5]] = _shared_region_ids



    ### 1. process the spots assigned to the specified chromosome cluster
    _cluster_id = _chromosome_clusters [0,-1]
    
    if _verbose:
        print (f'-- calculating region quality for chromosome cluster {_cluster_id}', end=' ')

        # slice only spots belong to the processed cluster
    _spots_sel_cluster = _spots_hzxyida_array[_spots_hzxyida_array[:,6] == _cluster_id]   # a in spot hzxyida  or hzxyidap
        
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
        
        # if different chr has different region ids/length to evaluate
        _filtered_chromosome_cluster =  []
        for _chr in _chromosome_cluster:
            if _region_id in _shared_gene_region_ids_dict[_chr[5]]:
                _filtered_chromosome_cluster.append(_chr)
        _filtered_chromosome_cluster = np.array(_filtered_chromosome_cluster)

            # record number for each category 
            # (temp?) do not count negative/empty region a this point
        if len(_filtered_chromosome_cluster) > 0:
            if len(_spots_sel_region) < len(_filtered_chromosome_cluster):  # if this is the majority, suggesting a false postive chromosome center or the distance radius was given too small
                _loss_region +=1
            elif len(_spots_sel_region) == len(_filtered_chromosome_cluster): # if distance radius was given well, this number should plateau and decrease if keeping increasing the distance
                _confident_region +=1
            elif len(_spots_sel_region) <= 2* len(_filtered_chromosome_cluster):
                _ambiguous_region +=1
            elif len(_spots_sel_region) > 2* len(_filtered_chromosome_cluster):  # if this is the majority, suggesting a false negative (undetected) chromosome center in this cluster
                _bad_region +=1
       
        
    if _isolated_chr_only and len(_chromosome_cluster) !=1:
        if _verbose:
            print ('-- not an isolated chromosome, region reporting skipped')
        return np.array([_cluster_id, np.nan,np.nan,np.nan,np.nan])

    else:
        return np.array([_cluster_id, _confident_region,_ambiguous_region,_bad_region,_loss_region])




def batch_calculate_region_quality (_chrom_azyxiuc_array, 
                                   _spots_hzxyida_array, 
                                   _gene_region_ids_dict ={},
                                   _region_ids = None,                       
                                   _isolated_chr_only = False,
                                   _verbose = True,
                                   _num_threads = 20):
    '''Batch function to calculate region quality;
        
        Use the function above but process individual chrom cluster parrallelly'''

    if not isinstance (_chrom_azyxiuc_array, np.ndarray) or not isinstance (_spots_hzxyida_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        # chrom cluster with single chromosome
        if _chrom_azyxiuc_array.shape [0] > 1 and _chrom_azyxiuc_array.shape [1] == 7:
            _chromosome_cluster_ids = np.unique(_chrom_azyxiuc_array[:,-1])
        else:
            if _verbose:
                print ('-- chrom inputs are not valid.')
            return None
    
    if _spots_hzxyida_array.shape[-1] < 7:   # hzyxidap array is also supported [note the relevant regions should have not been picked for the purppose of assessing quality]
        if _verbose:
            print ('-- spot inputs are not valid.')
        return None
    
    # generate args for starmap 
    _calculate_region_args = []
    for _chromosome_cluster_id in _chromosome_cluster_ids:
        _calculate_region_args.append((_chrom_azyxiuc_array[_chrom_azyxiuc_array[:,-1]==_chromosome_cluster_id], _spots_hzxyida_array, _gene_region_ids_dict, _region_ids, _isolated_chr_only, _verbose))
    
    import multiprocessing as mp
    with mp.Pool(_num_threads,) as _spot_pool:
        if _verbose:
            print (f'-- start multiprocessing assign spots to chromosome clusters with {_num_threads} threads', end=' ')
            _multi_time = time.time()
    # Multi-proessing!
        _spot_pool_result = _spot_pool.starmap(calculate_region_quality, _calculate_region_args)
        # close multiprocessing
        _spot_pool.close()
        _spot_pool.join()
        _spot_pool.terminate()
        if _verbose:
            print(f"in {time.time()-_multi_time:.3f}s.")
    
    # combine all results
    _region_quality = np.vstack(_spot_pool_result)

    return _region_quality

    
                


### Function to pick assigned spots to the given isolated chromosome
def pick_spots_for_isolated_chromosome (_isolated_chromosome, 
                                        _spots_hzxyida_array, 
                                        _gene_region_ids_dict = {},
                                        _batch = False,
                                        _region_ids = None, 
                                        _neighbor_len = None, 
                                        _iter_num = 5, 
                                        #_dist_ratio = 0.8, 
                                        #_h_ratio = 0.8,
                                        _local_dist_th = 1500,
                                        _verbose = True):
    """Function to pick spots for isolated chromosome
       
          Input: 
              _isolated_chromosome: selected chromsome (cluster) which should be in a format of _chrom_azyxiuc_array 
              _spots_hzxyida_array: preprocessed spot array, each row is a spot object;
                                    with h(intensity), z (nm), y (nm), x (nm), i (region_id), d (chr id by closest distance), a (assigned chr cluster id) as elements in each column
                                alternatively, this array can also be hzyxidap that is picked.
              _region_ids: the list/np.ndarray for all the combo/region ids
              _neighbor_len: the number of neighboring region on each side for the region being picked
              _gene_region_ids_dict: dict.keys as gene id; dict.values are lists of valid regiond ids for corresponding gene
              _batch: different format of output for batch function below
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
                print ('-- using specified region ids.')
            _region_ids = _region_ids
    else:
    # if not specified, use all regions that has at least one spot
        if _verbose:
            print ('using all region ids from the assigned spots.')
            print ("-- note if one region is completely missed, it cannot be reported.")
        _region_ids = np.unique(_spots_sel_cluster[:,4])      # i in hzxyida

    if _neighbor_len is None:
        _neighbor_len = round(len(_region_ids/10))   # use 10 + 10 % of region length for local reference
    

    # if gene_with_different_length is specified:
    if not isinstance(_gene_region_ids_dict, dict):
        if _verbose:
                print ('-- the _gene_region_ids_dict should be a dict.')
        return None

    # use _gene_region_ids_dict to refine _region_ids
    _gene_id = _isolated_chromosome [0,4]
    if _gene_id in _gene_region_ids_dict.keys():
        _gene_region_ds = _gene_region_ids_dict[_gene_id]

        if len(np.intersect1d(_gene_region_ds, _region_ids)) >0:
            _region_ids = np.intersect1d(_gene_region_ds, _region_ids)
        else:
            if _verbose:
                print ('-- no available regions to be analyzed.')
            return None


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
            # if there are more than 5 good/confident regions around, use the mean as the ref point
            if len(_shared_local_ids) >= 5:
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

    ### 2. secondary picking using local neighbor to find the closest spot as well as excluding spot outside the distance radius of the local center (despite there is one spot for this region)
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
                if len(_shared_local_ids) >= 5:
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
    
    if _batch:  # return only  the spots for cluster so results can be combined in batch
        return _spots_sel_cluster 
    else:
        return _spots_hzxyidap_array
                

                





### Function to pick assigned spots to the given chromosome cluster
### Time consuming for the permutation step if too many chr within the cluster 
def pick_spots_for_multi_chromosomes (_chromosome_cluster, 
                                      _spots_hzxyida_array, 
                                      _gene_region_ids_dict = {},
                                      _batch = False,
                                      _region_ids = None, 
                                      _neighbor_len = None,
                                      _proximity_ratio = 0.2, 
                                      _iter_num = 5, #_iter_num_swapping = 2,
                                      _local_dist_th =1500, 
                                      _verbose = True, 
                                      _debug = False):

    """Function to pick spots for chromosome cluster containing more than one chr
       
          Input: 
              _chromosome cluster: selected chromsome (cluster) which should be in a format of _chrom_azyxiuc_array 
              _spots_hzxyida_array: preprocessed spot array, each row is a spot object;
                                    with h(intensity), z (nm), y (nm), x (nm), i (region_id), d (chr id by closest distance), a (assigned chr cluster id) as elements in each column
                                alternatively, this array can also be hzyxidap that is picked.
              _region_ids: the list/np.ndarray for all the combo/region ids
              _neighbor_len: the number of neighboring region on each side for the region being picked
              _gene_region_ids_dict: dict.keys as gene id; dict.values are lists of valid regiond ids for corresponding gene
              _batch: different format of output for batch function below
              _iter_num: number of iteration for picking and exchanging spots for multiple chrom
              _proximity_ratio: the ratio to find the exceptionally close spot
              #_h_ratio: the ratio used to compare intensity when multiple candidates are very close 
              _local_dist_th: the biggest distance between a spot and its local neighbor center
          
          Output:
              _spots_hzxyidap_array: the spots array where the picked chr id is added as the 8th element (p)"""


    if not isinstance (_chromosome_cluster, np.ndarray) or not isinstance (_spots_hzxyida_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        # chrom cluster should contain one single chromosome;
        if _chromosome_cluster.shape[0] >1 and _chromosome_cluster.shape[1] ==7:
            _chromosome_cluster_id = _chromosome_cluster [0,-1]
           # azyxiuc  c for cluster id for chromosome
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
    _spots_sel_cluster = _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,6]==_chromosome_cluster_id]   # d in hzxyidap array

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


    # if gene_with_different_length is specified:
    if not isinstance(_gene_region_ids_dict, dict):
        if _verbose:
                print ('-- the _gene_region_ids_dict should be a dict.')
        return None

    # use _gene_region_ids_dict to refine _region_ids 
    # _shared_gene_region_ids_dict for chrom in the cluster and specified region ids
    
    import copy
    _default_region_ids = copy.deepcopy(_region_ids)
    _shared_gene_region_ids_dict = {}
    for _chr in _chromosome_cluster:
        _gene_id = _chr [4]
        if _gene_id in _gene_region_ids_dict.keys():
            _gene_region_ds = _gene_region_ids_dict[_gene_id]
            if len(np.intersect1d(_gene_region_ds, _default_region_ids)) >0:
                _shared_region_ids = np.intersect1d(_gene_region_ds, _default_region_ids)
            else:
                if _verbose:
                    print ('-- no region ids avalible to be analyzed.')
                return None
        else:
            _shared_region_ids = _default_region_ids

        _shared_gene_region_ids_dict[_chr[5]] = _shared_region_ids


    ### 1. initial picking for uniquely closest spot
    # 1.1 pick spots are significantly closer to on chr than other chrs
    for _region_id in _region_ids:   
        _spots_sel_region =_spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id]   # i in spot hzxyidap
        
        # only analyze chro that has this region
        _filtered_chromosome_cluster =  []
        for _chr in _chromosome_cluster:
            if _region_id in _shared_gene_region_ids_dict[_chr[5]]:
                _filtered_chromosome_cluster.append(_chr)
        _filtered_chromosome_cluster = np.array(_filtered_chromosome_cluster)
    
        from scipy.spatial.distance import cdist
        
        # there could be no chr after chr filtering step 
        if len(_filtered_chromosome_cluster) >0:
            _dist_matrix =  cdist(_spots_sel_region[:,1:4],_filtered_chromosome_cluster[:,1:4])  # num_spot by num_chr matrix
    
        
            for _spot_index, _dist in enumerate(_dist_matrix):  # for each spot
            
            # initial pick for spot 
            # if only one chr after filtering in the _dist_matrix
                if len(_dist) ==1:
                    if _dist[0] < _local_dist_th:
                        _spots_sel_region[_spot_index, 7] = _filtered_chromosome_cluster [0,5]
                    else:
                        _spots_sel_region[_spot_index, 7] = 0
        # spot fall within the radius of any one chrom center
                elif len(_dist[_dist < _local_dist_th]) ==1:
                    _spots_sel_region[_spot_index, 7] = _filtered_chromosome_cluster [np.argmin(_dist),5]  #p in spot hzxyidap assigned from u in chrom azyxiuc  
        # the closeset chrom center is 5 times shorted than th second closeset chrom center
                elif np.sort(_dist)[0] < _proximity_ratio * np.sort(_dist)[1] and np.min(_dist) < _local_dist_th:
                    _spots_sel_region[_spot_index, 7] = _filtered_chromosome_cluster [np.argmin(_dist),5]
        # assign the picked result back to the initial _spots_sel_cluster       
            _spots_sel_cluster[_spots_sel_cluster[:,4] == _region_id] =  _spots_sel_region 
        if _verbose:         
            print(f'-- initiate picking in chromosome cluster {int(_chromosome_cluster_id)} with {len(_chromosome_cluster)} chromosomes')
        
    # 1.2 remove spots for regions where more than one significantly close spots are picked
    _picked_spots = _spots_sel_cluster[_spots_sel_cluster [:,-1] > 0]   #picked spot should be a chr unique id > 0

    for _chr in _chromosome_cluster:
        if _debug: 
            print(f'-- refine initial picking in chromosome {int(_chr[5])}')
        _picked_spots_chr = _picked_spots[_picked_spots[:,7]== _chr[5]]  #p in spot hzxyidap <--> u in azyxiuc
        # loop through regions
        for _region_id in _region_ids:  ######
            if len(_picked_spots_chr[_picked_spots_chr[:,4]==_region_id])>1:  # reset for region where there are more than two spots 
                _picked_spots_chr[_picked_spots_chr[:,4]==_region_id,-1] = 0
        # assign back to the _picked_spots
        _picked_spots[_picked_spots[:,7]==_chr[5]] =_picked_spots_chr
    # assign back to the _spots_sel_cluster
    _spots_sel_cluster[_spots_sel_cluster [:,-1] > 0] = _picked_spots
    
    
    
    ## for debug
    _first_picked_spots = _spots_sel_cluster[_spots_sel_cluster [:,-1] > 0]
    if _verbose:
        print(f'+ there are {len(_first_picked_spots)} spots picked after initial picking')
    
    
    ### 2. secondary picking to iteratively assign (all possible) unpicked spots based on previous picking; _unpicked_spots remain static for this round 
    _iter = 0
    while _iter <_iter_num:
        _iter +=1
        
        if _debug:
            print(f'--iteration round {_iter} to finish secondary picking in chromosome cluster {int(_chromosome_cluster_id)}')
    
        _unpicked_spots = _spots_sel_cluster[_spots_sel_cluster [:,-1] == 0] 
    
    #_unpicked_spots_to_modify = _unpicked_spots.copy() 
    
    # 2.1 pick closest spots (exclusively) for each chr for each region using local distance from picked neighboring regions;   
        for _region_id in _region_ids:
        # if there are any unpicked spots for this region
            if len(_unpicked_spots[_unpicked_spots[:,4]==_region_id]) > 0:    #hzyxi  i-->region_id


            # only analyze chro that has this region
                _filtered_chromosome_cluster =  []
                for _chr in _chromosome_cluster:
                    if _region_id in _shared_gene_region_ids_dict[_chr[5]]:
                        _filtered_chromosome_cluster.append(_chr)
                _filtered_chromosome_cluster = np.array(_filtered_chromosome_cluster)

                if len(_filtered_chromosome_cluster) >0:

            # store all local ref center for each chrom
                    _ref_center_list = []
                    for _chr in _filtered_chromosome_cluster:
                
                
                # this would remain the same for each chr throught the loop since it is directly sliced from _spots_sel_cluster
                        _picked_spots_chr = _spots_sel_cluster[_spots_sel_cluster [:,-1] == _chr[5]] #p in spot hzxyidap <--> u in azyxiuc
                        _ref_ids_from_picked = np.unique(_picked_spots_chr[:,4])
                #print(f'ref_ids for {_chr[5]} is {_ref_ids_from_picked}')
                # the above wont change during the loop since it is directly sliced from the _spots_sel_cluster 
            
                        _start, _end = max(_region_id - _neighbor_len, min (_region_ids)), min(_region_id + _neighbor_len, max(_region_ids)) 
                        _sel_local_ids = np.concatenate([np.arange(_start, _region_id), np.arange(_region_id +1, _end+1)])
                        _shared_local_ids = np.intersect1d(_sel_local_ids, _ref_ids_from_picked)
                # if chr has a picked spot, append a pseudo coord which enable this chr do not be picked
                        if _region_id in _ref_ids_from_picked:
                            _ref_center = np.array([0,-204800,-204800])   # keep the number of chr center but this center wont be picked 
                            _ref_center_list.append(_ref_center)    
                # for unpicked chrom, if more than 5 picked regions nearby
                        elif len(_shared_local_ids) >= 5:
                            _neighbor_spots = []
                            for _shared_local_id in _shared_local_ids:
                                _spot_ref_region =_picked_spots_chr[_picked_spots_chr[:,4] == _shared_local_id]
                                _neighbor_spots.append(_spot_ref_region[0])
                            _neighbor_spots= np.array(_neighbor_spots)
                            _ref_center = np.nanmean(_neighbor_spots [:,1:4],axis=0)
                            _ref_center_list.append(_ref_center)
                # use picked spots if more than 33% of the total regions
                        elif len(_ref_ids_from_picked) > round(len(_region_ids)/3):
                            _ref_center = np.nanmean(_picked_spots_chr[:,1:4] ,axis=0) 
                            _ref_center_list.append(_ref_center)    # zyx only 
                # use chrom center
                        else:
                            _ref_center = _chr[1:4].copy()
                            _ref_center_list.append(_ref_center)
            # all local ref center for this region from each chrom       
                    _ref_centers = np.array(_ref_center_list)
            # all _unpicked_spots for this region
                    _unpicked_spots_region = _unpicked_spots [_unpicked_spots[:,4] == _region_id] 
            
                    from scipy.spatial.distance import cdist
                    _dist_matrix =  cdist(_unpicked_spots_region[:,1:4],_ref_centers)
            # loop for each spot
                    for _spot_index, _dist_spot in enumerate(_dist_matrix):

                # if only one chr after filtering in the _dist_matrix
                        if len(_dist_spot) ==1:
                            if _dist_spot[0] < _local_dist_th:
                                _unpicked_spots_region[_spot_index, 7] = _filtered_chromosome_cluster [0,5]
                   
                            else:
                                _unpicked_spots_region[_spot_index, 7] = 0
                # find the closest spot-chr pair and append only when this spot is also closer to the chr compared to other spot 
                        elif np.min(_dist_spot) == np.min (_dist_matrix[:,np.argmin(_dist_spot)]) and np.min(_dist_spot) < _local_dist_th:
                            _unpicked_spots_region [_spot_index,-1] =  _filtered_chromosome_cluster[np.argmin(_dist_spot),5]  # picked chr id by index
            # assing back; other unpicked regions would remain unpicked
                    _unpicked_spots [_unpicked_spots [:,4] == _region_id] = _unpicked_spots_region
                
    # after finishing all regions, assign back

        _spots_sel_cluster[_spots_sel_cluster [:,-1] == 0] = _unpicked_spots
        # if the number does not increase after enough rounds of iteration, all other unpicked spots are very unlikely due to various reaons
        #_second_picked_spots_test = _spots_sel_cluster[_spots_sel_cluster [:,-1] > 0]
        _second_picked_spots = _spots_sel_cluster[_spots_sel_cluster [:,-1] > 0]
    if _verbose:  
        print(f'+ there are {len(_second_picked_spots)} spots picked after secondary picking')
    
    ### 3. finalize by swapping spot selection to minimize sum of distance for each chrom-spot pairs

    if len(_chromosome_cluster) > 6:
        _iter_num = 0
        if _verbose:
            print ('-- too many chromosomes, skip swapping')


    _iter = 0
    _num_swap = 0
    while _iter <_iter_num:
        _iter +=1
        if _debug:
            print(f'--iteration round {_iter} to minimize pairwise distance sum in chromosome cluster {int(_chromosome_cluster_id)}')
        #_second_picked_spots = _spots_sel_cluster[_spots_sel_cluster [:,-1] > 0]
        # swapping spots for each region to minimize spot-chr distance sum
        for _region_id in _region_ids:

            # only analyze chro that has this region
            _filtered_chromosome_cluster =  []
            for _chr in _chromosome_cluster:
                if _region_id in _shared_gene_region_ids_dict[_chr[5]]:
                    _filtered_chromosome_cluster.append(_chr)
            _filtered_chromosome_cluster = np.array(_filtered_chromosome_cluster)
            
            _picked_spots_sel_region = _second_picked_spots[_second_picked_spots[:,4]== _region_id]
            _ref_center_list = []

            if len(_filtered_chromosome_cluster) >0:
                for _chr in _filtered_chromosome_cluster:
                # slice from _spots_sel_cluster remain unchanged for each iteration
                    _picked_spots_chr = _spots_sel_cluster[_spots_sel_cluster [:,-1] == _chr[5]] #p in spot hzxyidap <--> u in azyxiuc
                    _ref_ids_from_picked = np.unique(_picked_spots_chr[:,4])
                    
                    _start, _end = max(_region_id - _neighbor_len, min (_region_ids)), min(_region_id + _neighbor_len, max(_region_ids)) 
                    _sel_local_ids = np.concatenate([np.arange(_start, _region_id), np.arange(_region_id +1, _end+1)])
                    _shared_local_ids = np.intersect1d(_sel_local_ids, _ref_ids_from_picked) 
                # if more than 5 picked regions nearby
                    if len(_shared_local_ids) >= 5:
                        _neighbor_spots = []
                        for _shared_local_id in _shared_local_ids:
                            _spot_ref_region =_picked_spots_chr[_picked_spots_chr[:,4] == _shared_local_id]
                            _neighbor_spots.append(_spot_ref_region[0])
                        _neighbor_spots= np.array(_neighbor_spots)
                        _ref_center = np.nanmean(_neighbor_spots [:,1:4],axis=0)
                        _ref_center_list.append(_ref_center)
                # use picked spots if more than 33% of the total regions
                    elif len(_ref_ids_from_picked) > round(len(_region_ids)/3):
                        _ref_center = np.nanmean(_picked_spots_chr,axis=0) [1:4] 
                        _ref_center_list.append(_ref_center)    # zyx only 
                # use chrom center
                    else:
                        _ref_center = _chr[1:4].copy()
                        _ref_center_list.append(_ref_center)
                
                _ref_centers = np.array(_ref_center_list)
          
            # skip region where only one chrom (because of filtering) and region where exists any invalid local ref center/  also skip region if there is only one spot picked in the cluster
                if len(_ref_centers) >1 and np.count_nonzero(_ref_centers) == len(_ref_centers) * 3 and len(_picked_spots_sel_region) > 1 :
                    from scipy.spatial.distance import cdist
                    _dist_matrix =  cdist(_picked_spots_sel_region[:,1:4],_ref_centers)
                # current chr index (relative to the chrom cluster) for the picked spots for this region; eg spot1 assigned to the 2 chr --> 0 (spot index)- 1 (chr index)
                    _orignal_index = []
                    for _index, _picked_id in enumerate(_picked_spots_sel_region[:,-1]):
                        if _picked_id in _filtered_chromosome_cluster[:,5]:
                            _orignal_index.append (int(np.where(_filtered_chromosome_cluster[:,5] == _picked_id)[0]))
                    _orignal_index= tuple(_orignal_index)
                # permutation for all possible spot-chr index scnarios; e.g., 2 picked spots for 4 candidate chr: (0,1)(0,2)(0,3)(1,0)(1,2)...
                    from itertools import permutations
                    _l = list(permutations(range(0, len(_ref_centers)),len(_dist_matrix)))
                    _sum_dist_list = []
                    for _i in _l:
                       #print(_i)
                    # distance sum for this permutation
                        _dist_sum = 0
                        for _spot_index in range(len(_dist_matrix)):
                            if _dist_matrix [_spot_index,_i[_spot_index]] > _local_dist_th: # if one of the spot-chr distance beyond the th, add a random large value to add penalty to this choice
                                _dist_sum += 10000 
                            else:        
                                _dist_sum += _dist_matrix [_spot_index,_i[_spot_index]] 
                        _sum_dist_list.append(_dist_sum)
                # find the spot-chr index permutation that has the smalles distance 
                    _sum_dist_list = np.array(_sum_dist_list) 
                    if len(_sum_dist_list) > 0:   ### TEMP bug fix ###  skip empty sequence
                        _exchanged_index = _l[np.argmin(_sum_dist_list)]
                    else: 
                        _exchanged_index = _orignal_index
                # count if there is swapping for this region (could be more than 2 spots that have been swapped)
                    if _orignal_index!=_exchanged_index:
                        _num_swap +=1
                # exchange the picked chr id based on the  spot-chr index
                        for _spot,_index in zip(_picked_spots_sel_region, _exchanged_index):
                            _spot[-1] = _chromosome_cluster[:,5][_index]
                # assign back to the picked spots
                    _second_picked_spots[_second_picked_spots[:,4]== _region_id] = _picked_spots_sel_region
        
        #print(f'{_num_swap} swapping in total after {_iter}')
        # assign back to the cluster after each iteration
        _spots_sel_cluster[_spots_sel_cluster [:,-1] > 0] = _second_picked_spots
    
    _finally_picked_spots = _spots_sel_cluster[_spots_sel_cluster [:,-1] > 0]
    if _verbose and _iter_num > 0:
        print(f'+ there are {_num_swap} swapping(s) in {len(_finally_picked_spots)} spots that are finally picked')
    # assign back to the input array
    _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,6]==_chromosome_cluster_id] = _spots_sel_cluster
    _spots_hzxyidap_array = _spots_hzxyida_array_to_pick.copy()

    if _batch:  # return only  the spots for cluster so results can be combined in batch
        return _spots_sel_cluster 
    else:
        return _spots_hzxyidap_array
        





def batch_pick_spots_for_all_chromosomes (_chrom_azyxiuc_array, 
                                        _spots_hzxyida_array, 
                                        _gene_region_ids_dict = {},
                                        _region_ids = None, 
                                        _neighbor_len = None, 
                                        _iter_num = 5, 
                                        #_iter_num_swapping =2,
                                        _proximity_ratio = 0.2,
                                        #_dist_ratio = 0.8, 
                                        #_h_ratio = 0.8,
                                        _local_dist_th = 1500,
                                        _verbose = True,
                                        #_debug = False,
                                        _num_threads = 20, 
                                        _keep_repeated_spots = False
                                        ):
    '''BATCH function to pick spots using functions above '''

    if not isinstance (_chrom_azyxiuc_array, np.ndarray) or not isinstance (_spots_hzxyida_array, np.ndarray):
        if _verbose:
            print ('-- chrom and spot inputs should both be np.ndarray.')
        return None
    else:
        # chrom cluster should contain one single chromosome;
        if _chrom_azyxiuc_array.shape[0] >1 and _chrom_azyxiuc_array.shape[1] ==7:
            _chromosome_cluster_ids = np.unique(_chrom_azyxiuc_array[:,-1])
           # azyxiuc  c for cluster id for chromosome
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

   # if regions that are specified 
    if isinstance(_region_ids, list) or isinstance (_region_ids, np.ndarray):
        if len(_region_ids) > 0: 
            if _verbose:
                print ('-- using specified region ids.')
            _region_ids = _region_ids
    else:
    # if not specified, use all regions that has at least one spot
        if _verbose:
            print ('-- region ids is required for batch function.')
        return None


    if _neighbor_len is None:
        _neighbor_len = round(len(_region_ids/10))   # use 10 + 10 % of region length for local reference
   
   ### 1. multiprocessing to pick spots
   # generate args for starmap 
    _batch = True
    _pick_spot_args_single = []
    _pick_spot_args_multi= []
    for _chromosome_cluster_id in _chromosome_cluster_ids:
        _chromosome_cluster = _chrom_azyxiuc_array[_chrom_azyxiuc_array[:,-1]==_chromosome_cluster_id]
        if len(_chromosome_cluster) == 1:
            _pick_spot_args_single.append((_chromosome_cluster, _spots_hzxyida_array_to_pick, _gene_region_ids_dict, _batch, _region_ids, _neighbor_len, _iter_num, _local_dist_th))
        if len(_chromosome_cluster) >1:
            _pick_spot_args_multi.append((_chromosome_cluster, _spots_hzxyida_array_to_pick, _gene_region_ids_dict, _batch, _region_ids, _neighbor_len, _proximity_ratio, _iter_num, _local_dist_th))
    
    import multiprocessing as mp
    with mp.Pool(_num_threads) as _spot_pool:
        if _verbose:
            print (f'-- start multiprocessing pick spots to isolated chromosome clusters with {_num_threads} threads', end=' ')
            _multi_time = time.time()
    # Multi-proessing!
        _spot_pool_result = _spot_pool.starmap(pick_spots_for_isolated_chromosome, _pick_spot_args_single)
        # close multiprocessing
        _spot_pool.close()
        _spot_pool.join()
        _spot_pool.terminate()
        if _verbose:
            print(f"in {time.time()-_multi_time:.3f}s.")
    # combine results for isolated chromosome
    _spots_hzxyida_single = np.vstack(_spot_pool_result)


    with mp.Pool(_num_threads) as _spot_pool:
        if _verbose:
            print (f'-- start multiprocessing pick spots to multi-chromosome clusters with {_num_threads} threads', end=' ')
            _multi_time = time.time()
    # Multi-proessing!
        _spot_pool_result2 = _spot_pool.starmap(pick_spots_for_multi_chromosomes, _pick_spot_args_multi)
        # close multiprocessing
        _spot_pool.close()
        _spot_pool.join()
        _spot_pool.terminate()
        if _verbose:
            print(f"in {time.time()-_multi_time:.3f}s.")
    # combine results for multi chromosomes
    _spots_hzxyida_multi = np.vstack(_spot_pool_result2)

    # combine all results; note that spots within radius of multiple clusters will be kept
    _spots_hzxyida = np.vstack((_spots_hzxyida_single, _spots_hzxyida_multi))

     ### 3. re-assess spots that are picked multiple times for different chromosome clusters

        # find all spots picked
    _all_picked_spots = _spots_hzxyida[_spots_hzxyida[:,7]>0]
    # make a copy that do not change throught the process
    _all_picked_spots_copy_for_ref =_all_picked_spots.copy()
    # find repeated picked spots and their index
    _unique_spot, _count = np.unique(_all_picked_spots[:,2:4], return_counts=True, axis=0)    # use xy to identify same spot
    _repeated_spot_list = _unique_spot[_count>1] 
    _repeated_spot_index_list = []
    for _repeated_spot in _repeated_spot_list:
        _repeated_spot_index = np.argwhere(np.all(_all_picked_spots[:,2:4] == _repeated_spot, axis=1))
        _repeated_spot_index_list.append(_repeated_spot_index.ravel())  # ravel 
        
   # process each repeated spot
    if len(_repeated_spot_index_list) > 0:
        _spot_index_to_reset_list = []

        for _repeated_spot_index in _repeated_spot_index_list:
        # same spot that share xy in each repeated spot group
            _dist_to_chr_list = []
            for _same_spot_index in _repeated_spot_index:     #e.g., 100in [100,101,562]
                _same_spot = _all_picked_spots_copy_for_ref[_same_spot_index]
                _picked_chr_id = _same_spot[7]
                _region_id = _same_spot[4]
                _picked_spots_chr = _all_picked_spots_copy_for_ref[_all_picked_spots_copy_for_ref[:,7]==_picked_chr_id]
                _ref_ids_from_picked = np.unique(_picked_spots_chr[:,4])
                _start, _end = max(_region_id - _neighbor_len, min (_region_ids)), min(_region_id + _neighbor_len, max(_region_ids))
                _sel_local_ids = np.concatenate([np.arange(_start, _region_id), np.arange(_region_id +1, _end+1)])
                _shared_local_ids = np.intersect1d(_sel_local_ids, _ref_ids_from_picked) 
               # use local center
                if len(_shared_local_ids) >= 5:
                    _neighbor_spots = []
                    for _shared_local_id in _shared_local_ids:
                        _spot_ref_region =_picked_spots_chr[_picked_spots_chr[:,4] == _shared_local_id]
                        _neighbor_spots.append(_spot_ref_region[0])
                    _neighbor_spots= np.array(_neighbor_spots)
                    _ref_center = np.nanmean(_neighbor_spots [:,1:4],axis=0) 
                # use center from all picked 
                elif len(_ref_ids_from_picked) > round(len(_region_ids)/3):
                    _ref_center = np.nanmean(_picked_spots_chr[:,1:4] ,axis=0) # zyx only 
               # use chrom center
                else:
                    _ref_center = _chrom_azyxiuc_array[_chrom_azyxiuc_array[:,5]==_picked_chr_id][0,1:4]
                
                _dist_to_chr = np.linalg.norm(_same_spot[1:4]-_ref_center)
                _dist_to_chr_list.append(_dist_to_chr)
            # get the spot index that have larger spot-chr distance 
            _dist_to_chr_list = np.array(_dist_to_chr_list)
           # pop the index that has the smallest distance
            _index_closest = np.argmin(_dist_to_chr_list)

            if np.min(_dist_to_chr_list) < _local_dist_th:
                _spot_index_to_reset = np.delete(_repeated_spot_index, _index_closest)
            # if all picked spots larger than _local_dist_th, reset all
            else:
                _spot_index_to_reset = _repeated_spot_index
           # reset the picked id for these spots to zero in the _all_picked_spots
            for _index in _spot_index_to_reset:
                _all_picked_spots[_index, -1] = 0
                _spot_index_to_reset_list.append(_index)
                
            
    
    # only reset so each spot has only one pick; repeated spots are kept so the return length should be eqaul to assigned spot array
    if _keep_repeated_spots:
        if _verbose:  ### each repeat count as 1 repeated spot (so a same spot can have many repeats)
            print (f'-- reset picking for {len(_spot_index_to_reset_list)} repeated spots; repeated spots are kept in the output')
        _spots_hzxyida[_spots_hzxyida[:,7]>0] = _all_picked_spots
        _spots_hzxyidap = _spots_hzxyida.copy()
    

    
    # completely remove repeated spots that are not finally picked and repeated spots that are never picked; 
    else:       
        if _verbose:
            print (f'-- reset and removed {len(_spot_index_to_reset_list)} repeated picked spots,', end ='')
        # assign back the results
        _spots_hzxyida[_spots_hzxyida[:,7]>0] = _all_picked_spots

        # find all repeated spots and their index
        _unique_spot, _count = np.unique(_spots_hzxyida[:,2:4], return_counts=True, axis=0)    # use xy to identify same spot
        _repeated_spot_list = _unique_spot[_count>1] 
        _repeated_spot_index_list = []
        for _repeated_spot in _repeated_spot_list:
            _repeated_spot_index = np.argwhere(np.all(_spots_hzxyida[:,2:4] == _repeated_spot, axis=1))
            _repeated_spot_index_list.append(_repeated_spot_index.ravel())  # ravel 

        if len(_repeated_spot_index_list) > 0:

            _spot_index_to_remove_list = [] # original index for removing all spots from the original array 
            for _repeated_spot_index in _repeated_spot_index_list:
                _picked_chr_ids = []
                _closest_chr_in_cluster_list = []
                for _same_spot_index in _repeated_spot_index: 
                    _same_spot = _spots_hzxyida[_same_spot_index]
                    _picked_chr_id = _same_spot[7]
                    _closest_chr_id= _same_spot[5]    # d in hzxyid
                    _assigned_cluster_id = _same_spot[6]
                    # picked_chr_id to find if any is picked
                    _picked_chr_ids.append(_picked_chr_id)
                    # assess if the closest chr is within the assigned cluster
                    if _chrom_azyxiuc_array[_chrom_azyxiuc_array[:,5]== _closest_chr_id][0, 6] == _assigned_cluster_id:
                        _closest_chr_in_cluster_list.append(1)
                    else:
                        _closest_chr_in_cluster_list.append(0)
                
                _picked_chr_ids =np.array(_picked_chr_ids)
                _closest_chr_in_cluster_list = np.array(_closest_chr_in_cluster_list)
                # if there is none that is picked
                if np.max(_picked_chr_ids) <1:
                    # if none are picked, keep the one that is assigned to the cluster which contains the closest chr of the spot
                    _index_to_keep = _closest_chr_in_cluster_list[_closest_chr_in_cluster_list>0][0]
                    _index_to_remove = np.delete(_repeated_spot_index, _index_to_keep)
                    for _index in _index_to_remove:
                        _spot_index_to_remove_list.append(_index)
                # keep the picked one, removing others
                else:
                    _index_to_keep = np.argwhere(_picked_chr_ids>0).ravel()[0]
                    _index_to_remove = np.delete(_repeated_spot_index, _index_to_keep)
                    for _index in _index_to_remove:
                        _spot_index_to_remove_list.append(_index)
        # delete the repeated spots from the original array
        if _verbose:
            print (f'and {len(_spot_index_to_remove_list)} repeated spots in total')     ### each repeat count as 1 repeated spot
        _spots_hzxyida = np.delete(_spots_hzxyida, _spot_index_to_remove_list, axis = 0)
        _spots_hzxyidap = _spots_hzxyida.copy()

    return _spots_hzxyidap




def convert_dict_to_array (chrom_coords_dict):
    '''Function to conver chrom_coord_dict or spot_hzxy_dict (with arrays) to one single array

            Output: the key (spot region id or chrom gene id is appened as the last element) to each row of elements''' 

    _all_chrom_coords = []
    for _chr_key, _chr_coord in chrom_coords_dict.items():
           # new column for gene id
        _chr_key_col = np.ones((len(_chr_coord),1)) * int(_chr_key)
        _new_chr_coord = np.hstack((_chr_coord,_chr_key_col))
    # append each chr 
        for _chr in _new_chr_coord:
            _all_chrom_coords.append(_chr)

    _all_chrom_coords = np.array(_all_chrom_coords )  
    
    return _all_chrom_coords





def batch_2tier_pick_spots_for_preprocessed_fov (_fov,
                                           _chr_size_filter_dict ={}, 
                                           _chr_size_filter = 100, 
                                           _distance_zxy = [200, 108, 108],
                                           _adjust_ratio = 0.9, 
                                           _gene_region_ids_dict = {},
                                           _chr_distance = 2000,
                                           _local_dist_th = 1500,
                                           _neighbor_len = 6,
                                           _save_results = True,
                                           _alt_save_folder = None,
                                           _verbose=True):


    '''Function to pick spots for preprocessed FOV with chrom_coords and combo_spots_list
    
       Utilize most of functions above; also save the analyzed spots and chrom data if choose to save'''    

    # convert fov.combo_spots_list type data to spot dict
    _all_combo_spots_dict = {}
    for _index, _spots in enumerate(_fov.combo_spots_list):
        _all_combo_spots_dict [str(_index+1)] = _spots

    # convert the spot dict to array with region id as the last element
    _all_combo_spots = convert_dict_to_array (_all_combo_spots_dict)
    # convert the chrom dict to array with gene id as the last element
    _all_chrom_coords = convert_dict_to_array (_fov.chrom_coords)

    # define chromosome size filter dict or use universal size filter for different genes
    if isinstance(_chr_size_filter_dict, dict):
        # if use the same size for different gene chrom
        if len(_chr_size_filter_dict) == 0:
            for _chrom_key in _fov.chrom_coords.keys():
                _chr_size_filter_dict[_chrom_key] = _chr_size_filter
    

    # filter chromsome by size
    _filtered_chrom_coords = [_chr for _chr in _all_chrom_coords if _chr[3]>_chr_size_filter_dict[str(int(_chr[-1]))]]
    _filtered_chrom_coords = np.array(_filtered_chrom_coords)
    # move chr area col to the first to match the hzxy pattern of spots for the subsequent operations
    _move_indexes = [3,0,1,2,4]
    _filtered_chrom_coords_azxyi = _filtered_chrom_coords[:,_move_indexes]
    if _verbose:
        print(f'-- there are {len(_filtered_chrom_coords_azxyi)} chrom centers in this fov.')
   
    # filter spot by intensity
    _filtered_combo_spots = [_spot for _spot in _all_combo_spots if _spot[0] > _adjust_ratio*_fov.spot_intensity_th[str(int(_spot[-1]))]]
    _filtered_combo_spots = np.array(_filtered_combo_spots)
    if _verbose:
        print(f'-- there are {len(_filtered_combo_spots)} candidate spots in this fov')

     # convert zyx to nm and append gene id
    _chrom_azxyi = convert_spots_array_to_hzxyi_array(_filtered_chrom_coords_azxyi,_distance_zxy)
     # convert zyx to nm and append region id
    _spots_hzxyi = convert_spots_array_to_hzxyi_array(_filtered_combo_spots,_distance_zxy)


    # label and cluster chromosomes using function above
    _labeled_chrom_azxyiuc = label_and_cluster_chromosome_by_distance (_chrom_azxyi,_distance =_chr_distance,_verbose=_verbose)
    # label closest chromosome for each spot using function above
    _spots_hzxyid = find_closest_chromosome_for_spots (_labeled_chrom_azxyiuc, _spots_hzxyi)
    # batch assign spots using funcion above
    _spots_hzxyida = batch_assign_spots_to_chromosome_cluster_by_distance(_labeled_chrom_azxyiuc, _spots_hzxyid,_distance =_chr_distance,_verbose=_verbose)
    # pick spots by 2tier method
    _spots_hzxyidap = batch_pick_spots_for_all_chromosomes (_labeled_chrom_azxyiuc, _spots_hzxyida, 
                                                             _gene_region_ids_dict=_gene_region_ids_dict,
                                                             _region_ids =_fov.combo_ids, _local_dist_th = _local_dist_th,
                                                             _neighbor_len=_neighbor_len)

    if _save_results:
        if _alt_save_folder is not None:
            if not os.path.exists(_alt_save_folder):
                os.mkdir(_alt_save_folder)

            _save_folder = _alt_save_folder + os.sep + 'Spots'
            if not os.path.exists(_save_folder):
                os.mkdir(_save_folder)
            _spot_savename = f'_spots_hzxyidap_{_adjust_ratio}.npy'
            _chrom_savename =f'_chrom_azxyiuc_{_adjust_ratio}.npy'


        else:
            _save_folder = _fov.save_folder + os.sep + 'Spots'
            if not os.path.exists(_save_folder):
                os.mkdir(_save_folder)
            _spot_savename = _fov.fov_name[:-4] + f'_spots_hzxyidap_{_adjust_ratio}.npy'
            _chrom_savename = _fov.fov_name[:-4] + f'_chrom_azxyiuc_{_adjust_ratio}.npy'
        
        if _verbose:
            print ('-- saving chromosome and spot results.')
        np.save (_save_folder + os.sep + _spot_savename, _spots_hzxyidap)
        np.save (_save_folder + os.sep + _chrom_savename, _labeled_chrom_azxyiuc)


    return _spots_hzxyidap, _labeled_chrom_azxyiuc







### structure is highly similar between EM and 2tier; 
### currently, one difference is EM appears to better at removing very bad staining hyb results (by using scores);
### another difference is EM appears to have higher false-positive rate (e.g, picking gene with shorter ids without telling it) 
def batch_EM_pick_spots_for_preprocessed_fov (_fov,
                                           _chr_size_filter_dict ={}, 
                                           _chr_size_filter = 100, 
                                           _distance_zxy = [200, 108, 108],
                                           _adjust_ratio = 0.9, 
                                           _gene_region_ids_dict = {},
                                           _chr_distance = 2000,
                                           _local_dist_th = 1500,
                                           _neighbor_len = 6,
                                           _save_results = True,
                                           _alt_save_folder = None,
                                           _verbose=True,
                                           _num_threads = 24,
                                           niter= 10,
                                           _score_th = -5,
                                           _int_th = 200,
                                           ref_chr_cts = None):


    '''Function to EM pick spots for preprocessed FOV with chrom_coords and combo_spots_list; 
    
       Utilize EM pick of assigned spots, which are assigned by the function above; also save the analyzed spots and chrom data if choose to save'''    

    # convert fov.combo_spots_list type data to spot dict
    _all_combo_spots_dict = {}
    for _index, _spots in enumerate(_fov.combo_spots_list):
        _all_combo_spots_dict [str(_index+1)] = _spots

    # convert the spot dict to array with region id as the last element
    _all_combo_spots = convert_dict_to_array (_all_combo_spots_dict)
    # convert the chrom dict to array with gene id as the last element
    _all_chrom_coords = convert_dict_to_array (_fov.chrom_coords)

    # define chromosome size filter dict or use universal size filter for different genes
    if isinstance(_chr_size_filter_dict, dict):
        # if use the same size for different gene chrom
        if len(_chr_size_filter_dict) == 0:
            for _chrom_key in _fov.chrom_coords.keys():
                _chr_size_filter_dict[_chrom_key] = _chr_size_filter


    # filter chromsome by size
    _filtered_chrom_coords = [_chr for _chr in _all_chrom_coords if _chr[3]>_chr_size_filter_dict[str(int(_chr[-1]))]]
    _filtered_chrom_coords = np.array(_filtered_chrom_coords)
    # move chr area col to the first to match the hzxy pattern of spots for the subsequent operations
    _move_indexes = [3,0,1,2,4]
    _filtered_chrom_coords_azxyi = _filtered_chrom_coords[:,_move_indexes]
    if _verbose:
        print(f'--there are {len(_filtered_chrom_coords_azxyi)} chrom centers in this fov.')
   
    # filter spot by intensity
    _filtered_combo_spots = [_spot for _spot in _all_combo_spots if _spot[0] > _adjust_ratio*_fov.spot_intensity_th[str(int(_spot[-1]))]]
    _filtered_combo_spots = np.array(_filtered_combo_spots)
    if _verbose:
        print(f'--there are {len(_filtered_combo_spots)} candidate spots in this fov')

     # convert zyx to nm and append gene id
    _chrom_azxyi = convert_spots_array_to_hzxyi_array(_filtered_chrom_coords_azxyi,_distance_zxy)
     # convert zyx to nm and append region id
    _spots_hzxyi = convert_spots_array_to_hzxyi_array(_filtered_combo_spots,_distance_zxy)


    # label and cluster chromosomes using function above
    _labeled_chrom_azxyiuc = label_and_cluster_chromosome_by_distance (_chrom_azxyi,_distance =_chr_distance,_verbose=_verbose)
    # label closest chromosome for each spot using function above
    _spots_hzxyid = find_closest_chromosome_for_spots (_labeled_chrom_azxyiuc, _spots_hzxyi)
    # batch assign spots using funcion above
    _spots_hzxyida = batch_assign_spots_to_chromosome_cluster_by_distance(_labeled_chrom_azxyiuc, _spots_hzxyid,_distance =_chr_distance,_verbose=_verbose)

    _result_cols = np.zeros([len(_spots_hzxyida), 1])
    _spots_hzxyida_array_to_pick = np.hstack((_spots_hzxyida.copy(),_result_cols))

    
    # define region ids for different genes
    import copy
    _default_region_ids = copy.deepcopy(_fov.combo_ids)
    _shared_gene_region_ids_dict = {}
    for _chrom_key in _fov.chrom_coords.keys():
        _gene_id = int(_chrom_key)
        if _gene_id in _gene_region_ids_dict.keys():
            _gene_region_ds = _gene_region_ids_dict[_gene_id]
            if len(np.intersect1d(_gene_region_ds, _default_region_ids)) >0:
                _shared_region_ids = np.intersect1d(_gene_region_ds, _default_region_ids)
            else:
                if _verbose:
                    print ('-- no region ids avalible to be analyzed.')
                return None
        else:
            _shared_region_ids = _default_region_ids

        _shared_gene_region_ids_dict[int(_chrom_key)] = _shared_region_ids


    # EM pick for each gene
    from .picking import _maximize_score_spot_picking_of_chr, pick_spots_by_intensities,pick_spots_by_scores, generate_reference_from_population, evaluate_differences
    from scipy.spatial.distance import pdist, squareform
    


    for _chrom_key in _fov.chrom_coords.keys():

        

        ### 2.1 extract spot and chrom info into sel_dna_cand_hzxys_list
        gid = int(_chrom_key)

        #print(f'TEST starting spots for gene {gid} is {len(_spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,-1]>0])}')
        
        # refine region selection for each gene
        _region_ids= _shared_gene_region_ids_dict[int(_chrom_key)]


        _all_chr_gene =_labeled_chrom_azxyiuc[_labeled_chrom_azxyiuc[:,4]==gid]
        _all_chr_id_gene = np.unique(_all_chr_gene[:,5])
        #chrom_coords = _fov.chrom_coords[str(gid)][:,:3]

        sel_dna_cand_hzxys_list = []

        for _chr_id in _all_chr_id_gene[:]:
    
            _chr_spot_list = []
            _chrom = _all_chr_gene[_all_chr_gene[:,5]==_chr_id].ravel()
            _cluster_id = _chrom[6]
            
    
            _spot_chr_cluster_assigned = _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,6]==_cluster_id]
    
            _spot_chr_selected_from_assigned = []

            for _spot in _spot_chr_cluster_assigned:
                if np.linalg.norm(_spot[1:4] - _chrom[1:4]) < _chr_distance:   # repeated spots in the assigned array will be kept; remove them at the end
                 _spot_chr_selected_from_assigned.append(_spot)
    
            _spot_chr_selected_from_assigned = np.array(_spot_chr_selected_from_assigned)
    
            for _region_id in _region_ids:
        
                _spot_sel_region = _spot_chr_selected_from_assigned[_spot_chr_selected_from_assigned[:,4]==_region_id]
        
                if len(_spot_sel_region) > 0:
                    _spot_sel_region = _spot_sel_region [:,0:4]
                else:
                    _spot_sel_region = []
                _chr_spot_list.append(_spot_sel_region)
    
            sel_dna_cand_hzxys_list.append(_chr_spot_list)



        ### 2.2 perform EM pick (for each gene)
        # initialize
        init_dna_hzxys = pick_spots_by_intensities(sel_dna_cand_hzxys_list)
        # set save list
        sel_dna_hzxys_list, sel_dna_scores_list, all_dna_scores_list = [init_dna_hzxys], [], []
        
        # use subset of region ids and channel info if necessary
        dna_reg_ids = _region_ids

        _used_channel = []
        for _index, _channel in enumerate(_fov.combo_channels):
            if _index+1 in dna_reg_ids:
                _used_channel.append(_channel)
           
        dna_reg_channels = _used_channel 



        for _iter in range(niter):
            print(f"+ iter:{_iter}")
         # E: generate reference
            ref_ct_dists, ref_local_dists, ref_ints = generate_reference_from_population(
                                                  sel_dna_hzxys_list[-1], dna_reg_ids, 
                                                  sel_dna_hzxys_list[-1], dna_reg_ids,
                                                  ref_channels=dna_reg_channels,
                                                  ref_chr_cts=ref_chr_cts,
                                                  num_threads=_num_threads,
                                                  collapse_regions=True,
                                                  split_channels=True,
                                                  verbose=True,)
    
          # M: pick based on scores
            sel_hzxys_list, sel_scores_list, all_scores_list, other_scores_list = \
                                     pick_spots_by_scores(
                                          sel_dna_cand_hzxys_list, dna_reg_ids,
                                         ref_hzxys_list=sel_dna_hzxys_list[-1], ref_ids=dna_reg_ids, ref_channels=dna_reg_channels,
                                          ref_ct_dists=ref_ct_dists, ref_local_dists=ref_local_dists, ref_ints=ref_ints, 
                                                               ref_chr_cts=ref_chr_cts,
                                                  num_threads=_num_threads,
                                                              collapse_regions=True,
                                                  split_intensity_channels=True,
                                                     return_other_scores=True,
                                                                   verbose=True,)

            update_rate = evaluate_differences(sel_hzxys_list, sel_dna_hzxys_list[-1])
            sel_dna_hzxys_list.append(sel_hzxys_list)
            sel_dna_scores_list.append(sel_scores_list)
            all_dna_scores_list.append(all_scores_list)

            if update_rate > 0.998:
                break



        ### 2.3 filter bad spots and assign back the picked chr id
        sel_iter = -1
        #final_dna_hzxys_list = []
        #kept_chr_ids = []
        score_th = _score_th
        int_th = _int_th
        # add chr id to assign back to the array

        if _verbose:  ### each repeat count as 1 repeated spot (so a same spot can have many repeats)
            print (f'--- assigning spot pick results for gene {gid} to the spot array')

        for _hzxys, _scores, _chr_id in zip(sel_dna_hzxys_list[sel_iter], sel_dna_scores_list[sel_iter], _all_chr_id_gene):

            _chrom = _all_chr_gene[_all_chr_gene[:,5]==_chr_id].ravel()
            _cluster_id = _chrom[6]
            
            # modify the cluster array
            _spot_chr_cluster_assigned = _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,6]==_cluster_id]

            _kept_hzxys = np.array(_hzxys).copy()
                # remove spots by intensity
            _bad_inds = _kept_hzxys[:,0] < int_th
                # remove spots by scores
            _bad_inds += _scores < score_th
                #print(np.mean(_bad_inds))
            _kept_hzxys[_bad_inds] = np.nan

            #_picked_spot_zxy_index_list = []
            # find index for remaining good spots in the cluster array (which should contain the picked spot); 
            # some cluster would be modified multiple times, but each time should be independent 
            # # repeated spots (dist < th for more than two chro) in the assigned array will be kept; remove them at the end
            for _spot_zxy in _kept_hzxys:
                if np.isnan(_spot_zxy).sum()==0:  # non np.nan for valid spot
                    _picked_spot_zxy_index = np.argwhere(np.all(_spot_chr_cluster_assigned[:,2:4] == _spot_zxy[2:4], axis=1))
                    if len(_picked_spot_zxy_index) == 1:
                        _spot_chr_cluster_assigned[_picked_spot_zxy_index.ravel(),-1]=  _chr_id
            # assign back to the input original array
            _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,6]==_cluster_id] = _spot_chr_cluster_assigned


            #print(f'TEST pick spots for gene {gid} is {len(_spot_chr_cluster_assigned[_spot_chr_cluster_assigned[:,-1]>0])}')

            #print(f'TEST saved spots for gene {gid} is {len(_spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,-1]>0])}')


     ### 3. re-assess and re-set (if) repeated spots (from different assigned cluster) that are picked multiple times for different chr 
     # note that re-setted spots are not removed from the array

        # find all spots picked
    _all_picked_spots = _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,7]>0]


    #print(f'TEST all pick spots {len(_all_picked_spots)}')


    # make a copy that do not change throught the process
    _all_picked_spots_copy_for_ref =_all_picked_spots.copy()
    # find repeated picked spots and their index
    _unique_spot, _count = np.unique(_all_picked_spots[:,2:4], return_counts=True, axis=0)    # use xy to identify same spot
    _repeated_spot_list = _unique_spot[_count>1] 
    _repeated_spot_index_list = []
    for _repeated_spot in _repeated_spot_list:
        _repeated_spot_index = np.argwhere(np.all(_all_picked_spots[:,2:4] == _repeated_spot, axis=1))
        _repeated_spot_index_list.append(_repeated_spot_index.ravel())  # ravel 
        
   # process each repeated spot
    if len(_repeated_spot_index_list) > 0:
        _spot_index_to_reset_list = []

        for _repeated_spot_index in _repeated_spot_index_list:
        # same spot that share xy in each repeated spot group
            _dist_to_chr_list = []
            for _same_spot_index in _repeated_spot_index:     #e.g., 100in [100,101,562]
                _same_spot = _all_picked_spots_copy_for_ref[_same_spot_index]
                _picked_chr_id = _same_spot[7]
                _region_id = _same_spot[4]
                _picked_spots_chr = _all_picked_spots_copy_for_ref[_all_picked_spots_copy_for_ref[:,7]==_picked_chr_id]
                _ref_ids_from_picked = np.unique(_picked_spots_chr[:,4])
                _start, _end = max(_region_id - _neighbor_len, min (_region_ids)), min(_region_id + _neighbor_len, max(_region_ids))
                _sel_local_ids = np.concatenate([np.arange(_start, _region_id), np.arange(_region_id +1, _end+1)])
                _shared_local_ids = np.intersect1d(_sel_local_ids, _ref_ids_from_picked) 
               # use local center
                if len(_shared_local_ids) >= 5:
                    _neighbor_spots = []
                    for _shared_local_id in _shared_local_ids:
                        _spot_ref_region =_picked_spots_chr[_picked_spots_chr[:,4] == _shared_local_id]
                        _neighbor_spots.append(_spot_ref_region[0])
                    _neighbor_spots= np.array(_neighbor_spots)
                    _ref_center = np.nanmean(_neighbor_spots [:,1:4],axis=0) 
                # use center from all picked 
                elif len(_ref_ids_from_picked) > round(len(_region_ids)/3):
                    _ref_center = np.nanmean(_picked_spots_chr[:,1:4] ,axis=0) # zyx only 
               # use chrom center
                else:
                    _ref_center = _labeled_chrom_azxyiuc[_labeled_chrom_azxyiuc[:,5]==_picked_chr_id][0,1:4]
                
                _dist_to_chr = np.linalg.norm(_same_spot[1:4]-_ref_center)
                _dist_to_chr_list.append(_dist_to_chr)
            # get the spot index that have larger spot-chr distance 
            _dist_to_chr_list = np.array(_dist_to_chr_list)
           # pop the index that has the smallest distance
            _index_closest = np.argmin(_dist_to_chr_list)

            if np.min(_dist_to_chr_list) < _local_dist_th:
                _spot_index_to_reset = np.delete(_repeated_spot_index, _index_closest)
            # if all picked spots larger than _local_dist_th, reset all
            else:
                _spot_index_to_reset = _repeated_spot_index
           # reset the picked id for these spots to zero in the _all_picked_spots
            for _index in _spot_index_to_reset:
                _all_picked_spots[_index, -1] = 0
                _spot_index_to_reset_list.append(_index)
    else:
        _spot_index_to_reset_list = _repeated_spot_index_list

    # assign back
    if _verbose:  ### each repeat count as 1 repeated spot (so a same spot can have many repeats)
        print (f'--- reset picking for {len(_spot_index_to_reset_list)} repeated spots; repeated spots are kept in the output')
    _spots_hzxyida_array_to_pick[_spots_hzxyida_array_to_pick[:,7]>0] = _all_picked_spots
        
   ### save and return result
                    
    _spots_hzxyidap = _spots_hzxyida_array_to_pick.copy()

    if _save_results:
        if _alt_save_folder is not None:
            if not os.path.exists(_alt_save_folder):
                os.mkdir(_alt_save_folder)

            _save_folder = _alt_save_folder + os.sep + 'Spots'
            if not os.path.exists(_save_folder):
                os.mkdir(_save_folder)
            _spot_savename = f'_EM_spots_hzxyidap_{_adjust_ratio}.npy'
            _chrom_savename =f'_EM_chrom_azxyiuc_{_adjust_ratio}.npy'


        else:
            _save_folder = _fov.save_folder + os.sep + 'Spots'
            if not os.path.exists(_save_folder):
                os.mkdir(_save_folder)
            _spot_savename = _fov.fov_name[:-4] + f'_EM_spots_hzxyidap_{_adjust_ratio}.npy'
            _chrom_savename = _fov.fov_name[:-4] + f'_EM_chrom_azxyiuc_{_adjust_ratio}.npy'
        
        if _verbose:
            print ('--- saving chromosome and spot results.')
        np.save (_save_folder + os.sep + _spot_savename, _spots_hzxyidap)
        np.save (_save_folder + os.sep + _chrom_savename, _labeled_chrom_azxyiuc)


    return _spots_hzxyidap,  _labeled_chrom_azxyiuc








# Function to assign chromosome (chr array) to labeled cells from for example RNA experiment (with drift)
# Output is a chrom array with original measurement (azxyiuc) plus aligned xy and assigned cell id
def assign_chromosome_to_labeled_cells (_chrom_azxyiuc_filename, _cell_segmentation_filename, _transpose_segmentation = True,
                                        _align_transformation = None, _inverse_transformation = True,
                                        _distance_xyz = [200,108, 108], _ims_shape = [60,2048,2048],
                                        _num_threads = 4, _save_results = True, _alt_save_folder = None,
                                        _verbose=True):
    '''Function to assign chromosome from processed and saved chromosome azxyiuc array to labeled cells
       
       Input: 
          _chrom_azxyiuc_filename: savename for the processed chromosome azxyiuc array
          _cell_segmentation_filename: savename for the segmentation file
          _transpose_segmentation: True if the xy of the sementation file is fliped compared to the xy of the chrom/spot images
          _align_transformation: if there is drift and rotation between segmentation image and chrom/spot image, provide the transmation matrix here
                               the transformation matrix is typically saved as attr and in the hdf5 file, which is the matrix that transform the segmentation image to the chrom image
          _inverse_transformation: if True, the inverse transformation matrix can be calculated from the input above; 
                                use this inverse matrix to transform chrom coord (map) back to the segmentation image, which is faster than the opposite way
          _other_settings: include image shape and xyz distance coversion from nm to pixel
        
        Output:
            _chrom_azxyiuc_xyl_array: the transformed (matching segmentation coord system) x and y in pixel saved as the 7th and 8th element; 
                                      the assigned cell id (from labels in the segmentation file) saved as the 9th element
                            if _save_results: the output is saved into the same folder where the input chrom array is located'''
    
    ### 1. preprocess input files

    # preprocess chrom array
    _labeled_chrom_azxyiuc = np.load(_chrom_azxyiuc_filename)
    # append cols to store results
    _result_cols = np.zeros([len(_labeled_chrom_azxyiuc), 3])
    _labeled_chrom_azxyiuc_xyl = np.hstack((_labeled_chrom_azxyiuc.copy(),_result_cols)) 
    # get save folder path if save results into the same folder where input chrom array is
    _chrom_file_name = _chrom_azxyiuc_filename.split('\\')[-1]
    _chrom_file_basename = _chrom_file_name.split('chrom')[0]
    _chrom_save_folder = _chrom_azxyiuc_filename.split(_chrom_file_name)[0] [:-1]  


   # preprocess segmentation file
    _cell_feature = np.load(_cell_segmentation_filename)
    #_segmentation_frames = _cell_feature.shape[0]
    _segmentation_shape =  _cell_feature.shape[1:]

    if list(_segmentation_shape) != _ims_shape[1:]:
        print ('-- invalid segmentation file shape, exit')
        return None
    
    # if segmentation xy needs to be transposed
    if _transpose_segmentation:
        _cell_feature = np.array([_lyr.T for _lyr in _cell_feature])
    
    ### 2. process transformation matrix 
    if _align_transformation is not None:
        tmat = _align_transformation
        if tmat.shape != (3,3):
            print ('-- the transformation matrix needs to be (3,3) homogeneous transformation matrix')  #[which is typically np.dot(Translation,Rotation)]'
            return None
        else:
            # if the input matrix is for transforming segmentation into chrom ims, do inverse transformation of the chrom coords
            if _inverse_transformation:
                if _verbose:
                    print('--  generate and use inverse transformation matrix') 
                _inv_matrix = np.zeros([3,3])
                _inv_matrix[:2,:2]= tmat[:2,:2].T
                _inv_matrix[2,2]=1
                H = tmat[0,2]
                K = tmat[1,2]
                _inv_matrix[0,2] = (-H * tmat[0,0]) - (K*tmat[1,0])
                _inv_matrix[1,2] = (H * tmat[1,0]) - (K*tmat[0,0])
                tmat = _inv_matrix
    else:
        tmat = None
    
    ### 3. sub-function to process chrs for each gene see the function below this
    
    ### 4. multiprocessing chrs from the input array using the sub-function above

    _assign_chr_args = []

    for _gene_id in np.unique(_labeled_chrom_azxyiuc_xyl[:,4]):

        _chr_gene = _labeled_chrom_azxyiuc_xyl[_labeled_chrom_azxyiuc_xyl[:,4]==_gene_id]
        _assign_chr_args.append((_chr_gene, _cell_feature, tmat, _distance_xyz, _ims_shape, _verbose))
    
    import multiprocessing as mp
    with mp.Pool(_num_threads) as _chr_pool:
        if _verbose:
            print (f'-- start multiprocessing assign chromosomes of each gene to labeled cells with {_num_threads} threads', end=' ')
            _multi_time = time.time()

    # Multi-proessing!
        _chrom_pool_result =  _chr_pool.starmap(assign_chromosome_to_labeled_cells_for_gene, _assign_chr_args)
        # close multiprocessing
        _chr_pool.close()
        _chr_pool.join()
        _chr_pool.terminate()
        if _verbose:
            print(f"in {time.time()-_multi_time:.3f}s.")
    # combine results for chromosomes from individual genes
    _labeled_chrom_azxyiucxyl = np.vstack(_chrom_pool_result)


    ### 5. save and return results

    if _save_results:
        if _alt_save_folder is not None:
            if not os.path.exists(_alt_save_folder):
                os.mkdir(_alt_save_folder)

            _save_folder = _alt_save_folder + os.sep + 'Spots'
            if not os.path.exists(_save_folder):
                os.mkdir(_save_folder)
            _chrom_savename =_chrom_file_basename + f'chrom_azxyiuc_xyl.npy'

        else:
            _save_folder =_chrom_save_folder 
            if not os.path.exists(_save_folder):
                os.mkdir(_save_folder)
            _chrom_savename =_chrom_file_basename + f'chrom_azxyiuc_xyl.npy'
        
        if _verbose:
            print ('--- saving chromosome assignment results.')
        
        np.save (_save_folder + os.sep + _chrom_savename, _labeled_chrom_azxyiucxyl)

    return _labeled_chrom_azxyiucxyl




    




    ### 3. a sub-function for process chrs for individual gene (a non-overlapping subset of the chrom array) for the batch function above
def assign_chromosome_to_labeled_cells_for_gene (_chr_gene, _segmentation, tmat =  None,  
                                                    _distance_xyz = [200,108, 108], _ims_shape = [60,2048,2048], _verbose = True):
    '''_chr_gene is a subset of _labeled_chrom_azxyiuc_xyl defined above outside this sub-function
        
        _segmentation is the cell_feature segmentation mask file, tmat is the homogeneous transformation matrix if supplied'''

    _batch = True # default setting 
    if _batch:

        ## 3.1 transform chrom xy coord 
        if tmat is not None:

            import skimage

             # generate chr map mask for each coord 
            _chr_map = np.zeros(_ims_shape[1:], dtype=np.uint16)
        
            # use the chr map mask to perform transformation;
            # the tmat appears to not work for directly transforming coords themselves; 
            # may have to do with the tamt is originally derived from transforming (DAPI) images instead of coords
            for _chr in _chr_gene:
                _chr_xy = _chr[2:4]/np.array(_distance_xyz[1:])
                # xy in pixel
                _chr_y, _chr_x= int(round(_chr_xy[0])),int(round(_chr_xy[1]))
                # label assigned by the unique chr id
                if _chr_map [_chr_y, _chr_x] == 0:
                    _chr_map [_chr_y, _chr_x] = _chr[5]  
                else: # though unlikely, if xy happend to be the same, shift the new coord by 1 pixel
                    _chr_map [_chr_y+1, _chr_x+1] = _chr[5]
            
            # convert tmat to skimage format
            tmat_skimage = skimage.transform.AffineTransform(tmat)
            # generate transformed chr map mask for each coord 
            _chr_map_tran = np.zeros([2048,2048], dtype=np.uint16)
            # preserves the label integrity by transforming each chr label at a time
            if _verbose:
                print ('-- generate transformed chromosome coord map')
            for _label in np.unique(_chr_map):
                    _labeled_chr = _chr_map == _label
                    _labeled_chr = _labeled_chr.astype(float)
                    _labeled_chr_tran = skimage.transform.warp(_labeled_chr,tmat_skimage, order=1)  # bi-linear interpolation which increase the coord siz ; but somehow neareast mode leads to loss of coords
                    _labeled_chr_tran[_labeled_chr_tran>0]=_label
                    _labeled_chr_tran = _labeled_chr_tran.astype(np.uint16)
                    _chr_map_tran += _labeled_chr_tran
            # retrive transformed xy (in pixel) from the transformed image
            for _label in np.unique(_chr_map_tran):
                if _label > 0  and _label in np.unique(_chr_map): # transformation can still merge two adjacent spots' label if there were within (~) two-three pixels; eliminates the merged label(s)
                    _find_coord = np.where(_chr_map_tran == _label)  
                    if len(_find_coord) >0 :  # if transformed chrom remain within the map range
                        # add transformed xy to the input array (7 and 8th element)       
                        _chr_gene[np.where(_chr_gene[:,5]==_label)[0][0],7] = _find_coord[0][0]   # transformation convert one coord pix into 4 neighboring pixels; use the first one as approximate
                        _chr_gene[np.where(_chr_gene[:,5]==_label)[0][0],8] = _find_coord[1][0]
                    else:
                        _chr_gene[np.where(_chr_gene[:,5]==_label)[0][0],7] = -1 # negative pseudo coord 
                        _chr_gene[np.where(_chr_gene[:,5]==_label)[0][0],8] = -1 # negative pseudo coord 


        # use original xy if no transformtation
        else:
            for _chr in _chr_gene:
                _chr_xy = _chr[2:4]/np.array(_distance_xyz[1:])
                _chr[7], _chr[8] = int(round(_chr_xy[0])),int(round(_chr_xy[1]))

        ## 3.2 assing cells using segmentation info
        _ims_frames = _ims_shape[0]
        _segmentation_frames = _segmentation.shape[0]
        _step_size = _ims_frames/_segmentation_frames

        if _verbose:
            print ('-- assigning transformed chromosome coord to labeled cells')

        for _chr in _chr_gene:
            # retrieve the pixel pos for z
            _chr_z = int(round(_chr[1]/np.array(_distance_xyz[0])))
            # initiate z recording
            _zmid = -1 
            _zbottom = -1
            _ztop = -1
            # loop through all segmentation to find the most relevant z(s)
            for _curr_z in np.arange(0,_ims_frames,_step_size):
            # identify the relevant z layer for segmentation info 
            # currently use a strigent criteria to only assign chr to cells whose masks exclusively cover the chr (while allow skipping) or the chr is very close to one specifc available z
            # considering the cell height and current z-thickness, typically most chr would be exclusively assign to one cell; 
            # but when top and bottom are different cells, it is difficult to determine which cell the chr belongs to; thus skip this chr seems better
                _dist_z = _chr_z- _curr_z
                if abs(_dist_z) <=1:
                    _zmid = _curr_z/_step_size
                elif _dist_z <_step_size and _dist_z >0:
                    _zbottom = _curr_z/_step_size
                elif _dist_z<0 and _dist_z > -_step_size:
                    _ztop = _curr_z/_step_size
      
             # if the chr is close to one specific z-layer (the transfromed xy must be in range of the segmentation image)
            if _zmid > -1 and int(round(_chr[7])) >0 and int(round(_chr[8])) >0 and int(round(_chr[7])) < _segmentation.shape[1] and int(round(_chr[8])) < _segmentation.shape[2]:
                _cell_feature_2d = _segmentation[int(_zmid)]
                # use z_mid segmentation to assign chromosome
                if _cell_feature_2d[int(round(_chr[7])),int(round(_chr[8]))]>0:
                    _chr[9]= _cell_feature_2d[int(round(_chr[7])),int(round(_chr[8]))]
                # if the specific z-mid happen to have no assigned cells, use top and bottom layer only if both give same result  
                # thus, strigent criteria is used here
                else:
                    if int(_zmid)*_step_size <  min(_ims_frames-1, (_segmentation_frames-1)*_step_size):
                        _cell_feature_2d_top = _segmentation[int(_zmid)+1]
                    else:
                        _cell_feature_2d_top = None
                    if int(_zmid)*_step_size >  0:
                        _cell_feature_2d_bottom = _segmentation[int(_zmid)-1]
                    else:
                        _cell_feature_2d_bottom = None
                    # if both have valid segmentation
                    if _cell_feature_2d_top is not None and _cell_feature_2d_bottom is not None:
                        _top_label = _cell_feature_2d_top[int(round(_chr[7])),int(round(_chr[8]))]
                        _bottom_label = _cell_feature_2d_bottom[int(round(_chr[7])),int(round(_chr[8]))]
                       # even if both have no assigned cells (0), the assignment result remain 0
                        if _top_label == _bottom_label:
                            _chr[9]= _top_label
                        else:
                            _chr[9]= 0

            # if the chr is near the middle of two z-layers
            elif int(round(_chr[7])) >0 and int(round(_chr[8])) >0 and int(round(_chr[7])) < _segmentation.shape[1] and int(round(_chr[8])) < _segmentation.shape[2]:
                _cell_feature_2d_top = _segmentation[int(_ztop)]
                _cell_feature_2d_bottom = _segmentation[int(_zbottom)]
                _top_label = _cell_feature_2d_top[int(round(_chr[7])),int(round(_chr[8]))]
                _bottom_label = _cell_feature_2d_bottom[int(round(_chr[7])),int(round(_chr[8]))]
               # only when both give the same result, assign the chromosome
                if _top_label == _bottom_label:
                   _chr[9]= _top_label
                else:
                   _chr[9]= 0

            # if transformed chr xy is out of range       
            else:
                _chr[9]= 0

        return _chr_gene

    
