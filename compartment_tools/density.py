import numpy as np
import multiprocessing as mp

def calculate_gaussian_density(centers, ref_center, sigma, 
                               intensity=1, background=0):
    sigma = np.array(sigma, dtype=float)
    g_pdf = np.exp(-0.5 * np.sum((centers - ref_center)**2 / sigma**2, axis=-1))
    g_pdf = float(intensity) * g_pdf + float(background)
    return g_pdf
## For Genome-wide DNA-MERFISH, do the calculation as follows:
def calculate_compartment_densities(chr_2_zxys, chr_2_AB_dict, 
                                    gaussian_radius,
                                    normalize_by_reg_num=False, 
                                    exclude_self=True,
                                    use_cis=False, use_trans=True,
                                    ):
    """Function to calculate comaprtment densities based on single-cell chr-2-zxys"""
    if not use_cis and not use_trans:
        raise ValueError(f"One of use_cis or use_trans should be true!")
    chr_2_densities = {}
    for _chr in chr_2_zxys:
        # extract info
        _zxys_list = chr_2_zxys[_chr]
        _AB_dict = chr_2_AB_dict[_chr]
        # init scores
        _A_scores_list = np.zeros(np.shape(_zxys_list)[:-1])
        _B_scores_list = np.zeros(np.shape(_zxys_list)[:-1])
        # collect all contributing coordinates:
        for _ihomo, _zxys in enumerate(_zxys_list):
            for _ireg, _zxy in enumerate(_zxys):
                # init ref_zxys for given zxy
                _A_zxys = []
                _B_zxys = []
                # if nan exists, skip calculation
                if np.isnan(_zxy).any():
                    _A_scores_list[_ihomo, _ireg] = np.nan
                    _B_scores_list[_ihomo, _ireg] = np.nan
                    continue
                # alternatively calculate
                if use_cis:
                    _cis_inds = np.arange(len(_zxys))
                    if exclude_self:
                        _cis_inds = np.setdiff1d(_cis_inds, _ireg)
                    # append cis AB
                    _A_zxys.append(_zxys[np.intersect1d(_AB_dict['A'], _cis_inds)])
                    _B_zxys.append(_zxys[np.intersect1d(_AB_dict['B'], _cis_inds)])    
                if use_trans:
                    for _ref_chr, _ref_zxys_list in chr_2_zxys.items():
                        _ref_AB_dict = chr_2_AB_dict[_ref_chr]
                        #print(_ref_zxys_list.shape, _ref_AB_dict['A'].dtype)
                        if _ref_chr != _chr:
                            _A_zxys.extend([_ref_zxys[_ref_AB_dict['A']] for _ref_zxys in _ref_zxys_list])
                            _B_zxys.extend([_ref_zxys[_ref_AB_dict['B']] for _ref_zxys in _ref_zxys_list])
                        else:
                            _A_zxys.extend([_ref_zxys[_ref_AB_dict['A']] 
                                            for _iref, _ref_zxys in enumerate(_ref_zxys_list) if _iref != _ihomo])
                            _B_zxys.extend([_ref_zxys[_ref_AB_dict['B']] 
                                            for _iref, _ref_zxys in enumerate(_ref_zxys_list) if _iref != _ihomo])
                # summarize
                #print(_chr, _ihomo, _ireg, len(_A_zxys), len(_B_zxys))
                if len(_A_zxys) == 0:
                    _A_scores_list[_ihomo, _ireg] = np.nan
                else:
                    _A_zxys = np.concatenate(_A_zxys)
                    _A_zxys = _A_zxys[np.isfinite(_A_zxys).all(1)]
                    # Calculate density
                    if normalize_by_reg_num:
                        _A_scores_list[_ihomo, _ireg] = calculate_gaussian_density(_A_zxys, _zxy, gaussian_radius).mean()
                    else:
                        _A_scores_list[_ihomo, _ireg] = calculate_gaussian_density(_A_zxys, _zxy, gaussian_radius).sum()
                if len(_B_zxys) == 0:
                    _B_scores_list[_ihomo, _ireg] = np.nan
                else:
                    _B_zxys = np.concatenate(_B_zxys)
                    _B_zxys = _B_zxys[np.isfinite(_B_zxys).all(1)]
                    # Calculate density
                    if normalize_by_reg_num:
                        _B_scores_list[_ihomo, _ireg] = calculate_gaussian_density(_B_zxys, _zxy, gaussian_radius).mean()
                    else:
                        _B_scores_list[_ihomo, _ireg] = calculate_gaussian_density(_B_zxys, _zxy, gaussian_radius).sum()
        # append to dict
        chr_2_densities[_chr] = {
            'A': _A_scores_list,
            'B': _B_scores_list,
        }
        
    return chr_2_densities
        
    
def BatchCompartmentDensities(chr_2_zxys_dicts, chr_2_AB_dict, gaussian_radius, 
                              num_threads=12,
                              normalize_by_reg_num=True, exclude_self=True, 
                              use_cis=False, use_trans=True):
    _args_list = [(_zxysDict, chr_2_AB_dict, gaussian_radius, normalize_by_reg_num, exclude_self, use_cis, use_trans)
                  for _zxysDict in chr_2_zxys_dicts]
    # pool
    with mp.Pool(num_threads) as density_pool:
        _chr_2_scores_dicts = density_pool.starmap(calculate_compartment_densities, _args_list, chunksize=1)
        density_pool.close()
        density_pool.join()
        density_pool.terminate()
    return _chr_2_scores_dicts