import numpy as np
import time
import multiprocessing as mp 
def _calculate_binary_center(_binary_label):
    """Function to calculate center of mass for a binary image"""
    _pixel_inds = np.indices(np.shape(_binary_label)).astype(np.uint16)
    _coord = []
    for _l in (_pixel_inds * _binary_label):
        _coord.append(np.mean(_l[_l>0]))
    return np.array(_coord)


### add procedures to generate and return the area for a binary image/label @ Shiwei Liu
### this allow chr candidates to be filtered by their label area after coords are generated in the function [find_candidate_chromosomes_in_nucleus] below
def _calculate_binary_center_and_return_label_area (_binary_label):
    """Function to calculate center of mass for a binary image; additionaly, the area is returned for the label"""
    _pixel_inds = np.indices(np.shape(_binary_label)).astype(np.uint16)
    _coord = []
    for _l in (_pixel_inds * _binary_label):
        _coord.append(np.mean(_l[_l>0]))
    
    _coord = np.array(_coord)

    # two value binary image
    _binary_label_copy = _binary_label.copy()
    if np.max(_binary_label_copy) >1:
        _binary_label_copy[_binary_label_copy>0]=1
    _label_area = np.sum(_binary_label_copy)
    _coord_info = np.hstack ((_coord, _label_area))

    return _coord_info





# alternative method to find candidate chromosome @ Shiwei Liu
# including main features as below:
# 1. dna/dapi rough mask is used to filtering out the non-cell/non-nucleus region when calculating the intensity distribution of chr signal;
# 2. initial chr labels are seperated by their voxel size, which are subjected to subsequent binary operations using different parameters.
# 3. the label area for each cand chr is returned as the 4th element in the output (chrom_coords) in addition to the zyx, which can be used to filter chr coords posthoc
# (deprecated) 3. the specified chr/gene id is returned as the 4th element in the output (chrom_coords) in addition to the zyx, which would be used for simutaneous spots assigment to multiple genes.


# Some notes for adjusting the parameters:
# Test _chr_seed_size first so majority of single chromosome foci can be detected;
# Next, increase or decrease _percent_th_3chr and _percent_th_2chr if too much over-splitting or merging, respectively happened for chr seeds
# If oversplitting happens (especially on relatively condensed small foci) while merging is also frequent, try increase the _min_label_size. 
# Increase of _min_label_size decrease overspliting though it may lead to some detection loss for small chromosome seeds.

def find_candidate_chromosomes_in_nucleus (_chrom_im, _dna_im, _dna_mask = None,
                                           _chr_seed_size = 200,
                                           _filt_size =3, 
                                           _num_of_iter = 10,
                                           _size_pre_filtering = False,   
                                           _pre_min_label_size = 200,      
                                           _percent_th_3chr = 97.5,
                                           _percent_th_2chr = 85, 
                                           _use_percent_chr_area = False,
                                           _fold_3chr = 6,
                                           _fold_2chr = 4,
                                           _std_ratio = 3,
                                           _morphology_size=1, 
                                           _min_label_size=30, 
                                           _random_walk_beta=15, 
                                           _num_threads=4, 
                                           _verbose=True):
    '''Function to find candidate chromosome centers given
    Inputs:
        _chrom_im: image where chromosomes are lighted,
            it could be directly from experimental result, or assembled by stacking over images,
            np.ndarray(shape equal to single_im_size)_
        _dna_im: DNA/nuclei/cell image that are used for filtering out the non-cell/non-nucleus region
        _dna_mask: if use DNA/cell mask provided from elsewhere, define such mask here
        # (deprecated) _chr_id: the id number for the chr (gene) selected; default is 0
        _chr_seed_size: the rough min seed size for the initial seed
        _filt_size =3: filter size for max and min filters as well as edge exclusion 
        _num_of_iter: numer of iteration to adjust the initial binary seeds to match the min chr seed size
        _percent_th_3chr: the percentile th (for voxel size) as indicated for grouping large chr seeds that are likely formed by more than two chr
        _percent_th_3chr: the percentile th (for voxel size) as indicated for grouping large chr seeds that are likely formed by two chr
        _use_percent_chr_area: use percentile or the fold number to estimate large multi-chr foci
        _fold_3chr: the fold of median (single) chr size to be considered as large multi-chr seed
        _fold_2chr: the fold of median (single) chr size to be considered as large dual-chr seed
        _std_ratio: the number of std to be used to find the lighted chromosome
        _morphology_size: the size for erosion/dilation for single chr candidate; 
            this size is adjusted further for erosion/dilation for larger chr seeds that are likely formed by multiple chr candidate
        _min_label_size: size for removal of small objects after binary operation; note that this is typically smaller than what is used in the [find_candidate_chromosomes] function below
        _random_walk_beta: the higher the beta, the more difficult the diffusion is.
        _verbose: say something!, bool (default: True)
    Output:
         cand_chrom_coords: list of chrom coordinates in zxy pixels + the chr/gene id as the fourth element'''

    from skimage import morphology
    #from scipy.stats import scoreatpercentile
    from scipy.ndimage.filters import maximum_filter, minimum_filter
    from scipy import ndimage
    from skimage import measure
    from skimage import filters


     ## 1. process image to find chr seeds with edge exclusion

    _dna_im_zmax = np.max(_dna_im,axis=0)
    _dna_rough_mask =  _dna_im_zmax > filters.threshold_otsu (_dna_im_zmax)

    # if use provided dna mask
    if _dna_mask is not None and isinstance(_dna_mask, np.ndarray):
        if _dna_mask.shape == _chrom_im[0].shape:
            _dna_rough_mask = _dna_mask
            if _verbose:
                print (f"-- use provided DNA/cell mask.")
    
    _dna_rough_mask = morphology.dilation (_dna_rough_mask, morphology.disk(10))  # to include foci on the edge of nuclei
    

     # generate initial seed image after applying filters
    _filt_size =3 
    adj_chrom_im = np.array([_lyr/np.median(_lyr) for _lyr in _chrom_im])
    _max_ft = maximum_filter(adj_chrom_im, _filt_size, mode='nearest')
    _min_ft = minimum_filter(adj_chrom_im, _filt_size, mode='nearest')
    _seed_im = _max_ft - _min_ft

    # extract the seed signal within the cell/nuclei
    _seed_im_filtered =  np.max(_seed_im,axis=0) * _dna_rough_mask
    _seed_im_filtered = np.array([_i for _i in _seed_im_filtered.flatten() if _i >0])
     # use mean + std * 3 (or other ratio factor) to find the th for image binarization
    _seed_im_th =  np.mean(_seed_im_filtered) + np.std(_seed_im_filtered) * _std_ratio  # (default is mean + 3*std)
    if _verbose:
        print(f"-- binarize image with threshold: {_seed_im_th}")
    _binary_im = _seed_im * _dna_rough_mask > _seed_im_th   # also exclude the non-cell (xy max projection) area


    if _size_pre_filtering:
        if _verbose:
            print ('-- prefiltering the birnary im with _pre_min_label_size as {_pre_min_label_size}')
        _binary_im = morphology.remove_small_objects(_binary_im, _pre_min_label_size).astype(np.uint16)

    
     # exclude edges
    _filt_size=3
    _edge = int(np.ceil(_filt_size/2))
    _binary_im[:_edge] = 0
    _binary_im[-_edge:] = 0
    _binary_im[:,:_edge] = 0
    _binary_im[:,-_edge:] = 0
    _binary_im[:,:,:_edge] = 0
    _binary_im[:,:,-_edge:] = 0

     ## 2. binary operations for chr seeds with different voxel size
    
     # calculate the rough size for single chr seed
    _label_chr,_num_chr = ndimage.label(_binary_im)
    _rois_chr = measure.regionprops(_label_chr)
    _chr_size = []
    for _roi in _rois_chr:
        _chr_size.append(_roi.area)  # append the voxel area for each seed 

    # test if majority of generated seeds are smaller than expected
    if np.median(_chr_size) < _chr_seed_size:
        # this is typically caused by decrease of signal within expected seeds; connect them by dilation first with iteration to reach expected size
        # in addition, if seed size is < ~100-200, such seeds will be eliminated by 3d erosion using ball size of 1 below
        for _iter in range(_num_of_iter):
            if _verbose == True:
                print ('-- iterative dilation to increase initial binary seed size')
            _binary_im = morphology.dilation(_binary_im,morphology.ball(1))
            _label_chr,_num_chr = ndimage.label(_binary_im)
            _rois_chr = measure.regionprops(_label_chr)
            _chr_size = []
            for _roi in _rois_chr:
                _chr_size.append(_roi.area)
            if np.median(_chr_size) >= _chr_seed_size:
                break

    if _verbose:
        print(f"-- there are {_num_chr} objects after initial segmentation")
    
    # define size filter for multi-chr seeds
    if _use_percent_chr_area:
        _size_th_2chr = np.percentile(np.array(_chr_size),_percent_th_2chr)  # size th for large (or intermediate) seed
        _size_th_3chr = np.percentile(np.array(_chr_size),_percent_th_3chr)  # size th for larger seed
    else:
        _size_th_2chr = np.median(_chr_size) * _fold_2chr
        _size_th_3chr = np.median(_chr_size) * _fold_3chr

    # separate chr seeds based on their rough size
    _3chr_binary_im = morphology.remove_small_objects(_binary_im, _size_th_3chr).astype(np.uint16)   # larger seed  [convert bool to 1 so '-' operation can be done later]; 
    # note that astype(uint16) should be done after remove_small_objects; otherwise the small object size is somehow scaled differently and no longer comparable to the _chr_size (voxel) above
    _minus_binary_im = _binary_im - _3chr_binary_im
    _2chr_binary_im = morphology.remove_small_objects(_minus_binary_im, _size_th_2chr).astype(np.uint16)   # large (or intermediate) seed  
    _1chr_binary_im = _minus_binary_im - _2chr_binary_im  # small seed  
    # (3d) erosion/dilation and closing/opening for different groups of chr seeds
    if _verbose:
        print(f"-- process small chromosome seeds with size={_morphology_size}.")
    _binary_label = _1chr_binary_im.copy()
    _binary_label = ndimage.binary_erosion(_binary_label, morphology.ball(_morphology_size))
    _binary_label = ndimage.binary_dilation(_binary_label, morphology.ball(_morphology_size))
    _binary_label = ndimage.binary_fill_holes(_binary_label, structure=morphology.ball(_morphology_size))
    _open_objects = morphology.opening(_binary_label, morphology.ball(0))
    _1chr_close_objects = morphology.closing(_open_objects, morphology.ball(1))
    if _verbose:
        print(f"-- process large/intermediate chromosome seeds with size={_morphology_size+1}.")
    _binary_label = _2chr_binary_im.copy()
    _binary_label = ndimage.binary_erosion(_binary_label, morphology.ball(_morphology_size+1))
    _binary_label = ndimage.binary_dilation(_binary_label, morphology.ball(_morphology_size))
    _binary_label = ndimage.binary_fill_holes(_binary_label, structure=morphology.ball(_morphology_size+1))
    _open_objects = morphology.opening(_binary_label, morphology.ball(1))
    _2chr_close_objects = morphology.closing(_open_objects, morphology.ball(1))
    if _verbose:
        print(f"-- process larger chromosome seeds with size={_morphology_size+1}. closing seeds with size=2")
    _binary_label = _3chr_binary_im.copy()
    _binary_label = ndimage.binary_erosion(_binary_label, morphology.ball(_morphology_size+1))
    _binary_label = ndimage.binary_dilation(_binary_label, morphology.ball(_morphology_size))
    _binary_label = ndimage.binary_fill_holes(_binary_label, structure=morphology.ball(_morphology_size+1))
    _open_objects = morphology.opening(_binary_label, morphology.ball(2))
    _3chr_close_objects = morphology.closing(_open_objects, morphology.ball(1))
    # combine all masks above
    _close_objects = _1chr_close_objects + _2chr_close_objects + _3chr_close_objects
    # remove small objects
    _close_objects= morphology.remove_small_objects(_close_objects, _min_label_size).astype(np.uint16)  # use smaller_label_size since seeds are eroded in size.

    ## 3. segmentation
    _label, _num = ndimage.label(_close_objects)
    _label[_label==0] = -1
    if _verbose:
        print(f"-- random walk segmentation, beta={_random_walk_beta}.")
    from skimage.segmentation import random_walker
    _seg_label = random_walker(_chrom_im, _label, beta=_random_walk_beta, mode='cg_mg')
    _seg_label[_seg_label < 0] = 0
    _kept_label = _seg_label.astype(np.uint16)
    _label_ids = np.unique(_kept_label)
    _label_ids = _label_ids[_label_ids > 0]
    if _verbose:
        print(f"-- {len(_label_ids)} objects are found by segmentation.")
    

    ## 4. calculate chr centers    
    _chrom_args = [(_kept_label==_id,) for _id in _label_ids]  # mask for individual chr

    with mp.Pool(_num_threads,) as _chrom_pool:
        if _verbose:
            print(f"- Start multiprocessing caluclate chromosome coordinates with {_num_threads} threads", end=' ')
            _multi_time = time.time()
        # Multi-proessing!
        _chrom_coords = _chrom_pool.starmap(_calculate_binary_center_and_return_label_area, _chrom_args, chunksize=1)
        # close multiprocessing
        _chrom_pool.close()
        _chrom_pool.join()
        _chrom_pool.terminate()
        if _verbose:
            print(f"in {time.time()-_multi_time:.3f}s.")
            
    _chrom_coords = np.array(_chrom_coords)
    
    # remove empty celement from somehow failed [_calculate_binary_center_and_return_label_area] function
    _chrom_coords_kept = []
    for _chr in _chrom_coords:
        if isinstance(_chr, np.ndarray):
            if len(_chr) ==4:     # zxy + area for chr label
                _chrom_coords_kept.append(_chr)
    
    # add chr/gene id to the output as dict
    _chrom_coords = np.array(_chrom_coords_kept)  

    #_chr_id_col = np.ones([len(_chrom_coords),1]) * int(_chr_id)

    #_chrom_coords = np.hstack ((_chrom_coords, _chr_id_col))
    
    return _chrom_coords
    



def find_candidate_chromosomes(_chrom_im, 
                               _adjust_layers=False, 
                               _filt_size=3, 
                               _binary_per_th=99.5,
                               _morphology_size=1, 
                               _min_label_size=100,
                               _random_walk_beta=10,
                               _num_threads=12,
                               _verbose=True):
    """Function to find candidate chromsome centers given
    Inputs:
        _chrom_im: image that chromosomes are lighted, 
            it could be directly from experimental result, or assembled by stacking over images,
            np.ndarray(shape equal to single_im_size)
        _adjust_layers: whether adjust intensity for layers, bool (default: False)
        _filt_size: filter size for maximum and minimum filters used to find local maximum, int (default: 3)
        _binary_per_th: percentile threshold to binarilize the image to get peaks, float (default: 99.5)
        _morphology_size=1, 
        _min_label_size=100,
        _random_walk_beta=10,
        _verbose: say something!, bool (default: True)
    Output:
        _cand_chrom_coords: list of chrom coordinates in zxy pixels"""
    
    from scipy.ndimage.filters import maximum_filter, minimum_filter
    from skimage import morphology
    from scipy.stats import scoreatpercentile
    from scipy import ndimage

    ## 1.  adjust image to find seeds
    if _verbose:
        print(f"-- adjust seed image with filter size={_filt_size}")
    adj_chrom_im = np.array([_lyr/np.median(_lyr) for _lyr in _chrom_im])
    _max_ft = maximum_filter(adj_chrom_im, _filt_size, mode='nearest')
    _min_ft = minimum_filter(adj_chrom_im, _filt_size, mode='nearest')
    _seed_im = _max_ft - _min_ft

    ## 2. binarize image and exclude edge points
    if _verbose:
        print(f"-- binarize image with threshold: {_binary_per_th}%")
    _binary_im = _seed_im > scoreatpercentile(_seed_im, _binary_per_th)
    _edge = int(np.ceil(_filt_size/2))
    _binary_im[:_edge] = 0
    _binary_im[-_edge:] = 0
    _binary_im[:,:_edge] = 0
    _binary_im[:,-_edge:] = 0
    _binary_im[:,:,:_edge] = 0
    _binary_im[:,:,-_edge:] = 0
    
    ## 3. erosion and dialation
    if _verbose:
        print(f"-- erosion and dialation with size={_morphology_size}.")
    _binary_label = _binary_im.copy()
    _binary_label = ndimage.binary_erosion(_binary_label, morphology.ball(_morphology_size))
    _binary_label = ndimage.binary_dilation(_binary_label, morphology.ball(_morphology_size))
    _binary_label = ndimage.binary_fill_holes(_binary_label, structure=morphology.ball(_morphology_size))
    ## 4. find object
    if _verbose:
        print(f"-- find close objects.")
    _open_objects = morphology.opening(_binary_label, morphology.ball(0))
    _close_objects = morphology.closing(_open_objects, morphology.ball(1))
    _label, _num = ndimage.label(_close_objects)
    _label[_label==0] = -1
    ## 5. segmentation
    if _verbose:
        print(f"-- random walk segmentation, beta={_random_walk_beta}.")
    from skimage.segmentation import random_walker
    #_seg_label = random_walker(adj_chrom_im, _label, beta=10, mode='cg_mg')   #parameter input was not used for the actual function. correction see below -- Shiwei
    _seg_label = random_walker(adj_chrom_im, _label, beta=_random_walk_beta, mode='cg_mg')
    _seg_label[_seg_label < 0] = 0
    ## 6. keep objects
    if _verbose:
        print(f"-- find objects larger than size={_min_label_size}")
    _kept_label = morphology.remove_small_objects(_seg_label, _min_label_size).astype(np.uint16)
    _label_ids = np.unique(_kept_label)
    _label_ids = _label_ids[_label_ids > 0]
    if _verbose:
        print(f"-- {len(_label_ids)} objects are found by segmentation.")
        
    ## 7. calculate chr centers    
    _chrom_args = [(_kept_label==_id,) for _id in _label_ids]

    with mp.Pool(_num_threads,) as _chrom_pool:
        if _verbose:
            print(f"- Start multiprocessing caluclate chromosome coordinates with {_num_threads} threads", end=' ')
            _multi_time = time.time()
        # Multi-proessing!
        _chrom_coords = _chrom_pool.starmap(_calculate_binary_center, _chrom_args, chunksize=1)
        # close multiprocessing
        _chrom_pool.close()
        _chrom_pool.join()
        _chrom_pool.terminate()
        if _verbose:
            print(f"in {time.time()-_multi_time:.3f}s.")
            
    _chrom_coords = np.array(_chrom_coords)
    
    return _chrom_coords

def select_candidate_chromosomes(_cand_chrom_coords,
                                 _spots_list, 
                                 _cand_spot_intensity_th=0.5,
                                 _good_chr_loss_th=0.4,
                                 _verbose=True,
                                 ):
    _cand_chrom_coords = list(_cand_chrom_coords)
    from ..spot_tools.picking import assign_spots_to_chromosomes
    ## start finding bad candidate chromosomes
    if _verbose:
        print(f"- start select from {len(_cand_chrom_coords)} chromosomes with loss threshold={_good_chr_loss_th}")
        _start_time = time.time()
    _good_chr_flags = [1 for _chr in _cand_chrom_coords]

    while max(_good_chr_flags) > _good_chr_loss_th:
        # for currently existing chromosomes, assign candidate spots
        _cand_chr_spots = [[] for _ct in _cand_chrom_coords]
        for _spots in _spots_list:
            _spots = np.array(_spots)
            _sel_spots = _spots[_spots[:,0]>=_cand_spot_intensity_th]
            _cands_list = assign_spots_to_chromosomes(_sel_spots, _cand_chrom_coords)
            for _i, _cands in enumerate(_cands_list):
                _cand_chr_spots[_i].append(_cands)
                
        # update _good_chr_flags
        for _i, (_chr_spots, _chrom_coord) in enumerate(zip(_cand_chr_spots, _cand_chrom_coords)):
            _flg = np.mean([len(_spots)==0 for _spots in _chr_spots])
            _good_chr_flags[_i] = _flg

        # remove the worst chromosome
        if np.max(_good_chr_flags) > _good_chr_loss_th:
            _remove_ind = np.argmax(_good_chr_flags)
            # remove this chr
            _removed_flg = _good_chr_flags.pop(_remove_ind)
            _removed_chr = _cand_chrom_coords.pop(_remove_ind)
            if _verbose:
                print(f"-- remove chr id {_remove_ind}, percentage of lost rounds:{_removed_flg:.3f}.")

    # finalize chrom_coords        
    _chrom_coords = np.array(_cand_chrom_coords).copy()
    if _verbose:
        print(f"-- {len(_chrom_coords)} chromosomes are kept.")

    return _chrom_coords


def identify_chromosomes(chrom_im, dapi_im=None, 
                         seed_gfilt_size=0.75, background_gfilt_size=7.5, 
                         chrom_snr_th=1.5, dapi_snr_th=2,
                         morphology_size=1, min_label_size=25,
                         num_threads=12,
                         return_seed_im=False,
                         verbose=True):
    """Function to identify chromosomes"""
    from ..io_tools.load import find_image_background
    from ..segmentation_tools.chromosome import _calculate_binary_center
    from scipy.ndimage.filters import gaussian_filter
    from scipy import ndimage
    from skimage import morphology
    from skimage.segmentation import random_walker
    import multiprocessing as mp
    # 1. generate seeding image
    if verbose:
        print(f"-- generate seeding image.")
    _chrom_im = chrom_im.copy()
    _signal_im = np.array(gaussian_filter(_chrom_im, seed_gfilt_size), dtype=np.float)
    _background_im = np.array(gaussian_filter(_chrom_im, background_gfilt_size), dtype=np.float)
    _seed_im = _signal_im - _background_im
    
    # 2. binarize image
    if verbose:
        print(f"-- binarize image with chromosome SNR: {chrom_snr_th}")
    _binary_im = (_seed_im >= np.abs(chrom_snr_th-1) * find_image_background(chrom_im))
    if isinstance(dapi_im, np.ndarray):
        _binary_im *= (dapi_im > dapi_snr_th * find_image_background(dapi_im) )
    ## 3. erosion and dialation
    if verbose:
        print(f"-- erosion and dialation with size={morphology_size}.")
    _binary_label = _binary_im.copy()
    _binary_label = ndimage.binary_erosion(_binary_label, morphology.ball(morphology_size))
    _binary_label = ndimage.binary_dilation(_binary_label, morphology.ball(morphology_size))
    _binary_label = ndimage.binary_fill_holes(_binary_label, structure=morphology.ball(morphology_size))
    ## 4. find object
    if verbose:
        print(f"-- find close objects.")
    _open_objects = morphology.opening(_binary_label, morphology.ball(0))
    _close_objects = morphology.closing(_open_objects, morphology.ball(1))
    _label, _num = ndimage.label(_close_objects)
    _label[_label==0] = -1
    ## 5. segmentation
    _random_walk_beta = 10
    if verbose:
        print(f"-- random walk segmentation, beta={_random_walk_beta}.")
    _seg_label = random_walker(_seed_im, _label, beta=_random_walk_beta, mode='cg_mg')
    _seg_label[_seg_label < 0] = 0
    ## 6. keep objects
    if verbose:
        print(f"-- find objects larger than size={min_label_size}")
    _kept_label = morphology.remove_small_objects(_seg_label, min_label_size).astype(np.uint16)
    _label_ids = np.unique(_kept_label)
    _label_ids = _label_ids[_label_ids > 0]
    ## 7. calculate chr centers    
    _chrom_args = [(_kept_label==_id,) for _id in _label_ids]

    with mp.Pool(num_threads,) as _chrom_pool:
        if verbose:
            print(f"-- start multiprocessing caluclate chromosome coordinates with {num_threads} threads", end=' ')
            _multi_time = time.time()
        # Multi-proessing!
        _chrom_coords = _chrom_pool.starmap(_calculate_binary_center, _chrom_args, chunksize=1)
        # close multiprocessing
        _chrom_pool.close()
        _chrom_pool.join()
        _chrom_pool.terminate()
        if verbose:
            print(f"in {time.time()-_multi_time:.3f}s.")

    _chrom_coords = np.array(_chrom_coords)
    if verbose:
        print(f"-- {len(_chrom_coords)} chromosomes identified")

    if return_seed_im:
        return _chrom_coords, _seed_im
    else:
        return _chrom_coords