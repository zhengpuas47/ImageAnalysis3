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
    _seg_label = random_walker(adj_chrom_im, _label, beta=10, mode='cg_mg')
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
