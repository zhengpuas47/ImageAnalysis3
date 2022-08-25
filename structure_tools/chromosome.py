
import numpy as np


def convert_chr2Zxys_2_Cloud(chr2Zxys,
                             pixel_size=0.1, im_radius=5,
                             gaussian_sigma=0.5,
                             allowed_homolog_num=[1,2], 
                             min_valid_spots=20, min_valid_per=0.25,
                             normalize_counts=False, normalize_pdf=False,
                             return_empty=False,
                             ):
    from ..figure_tools.plot_decode import Centering_Chr2ZxysListDict
    from ..visual_tools import add_source

    _centered_chr2Zxys = Centering_Chr2ZxysListDict(chr2Zxys)
    _chr2densityArrs = {
        _chr:np.zeros([len(_homologs)]+[int(im_radius*2/pixel_size)]*3, dtype=np.float32) 
        for _chr, _homologs in _centered_chr2Zxys.items()
        if len(_homologs) in allowed_homolog_num
    }
    _effective_sigma = np.array([gaussian_sigma/pixel_size]*3)
    for _chr in _chr2densityArrs.keys():
        
        _homologs = _centered_chr2Zxys[_chr]
        for _ihomo, _zxys in enumerate(_homologs):
            _valid_flags = np.isfinite(_zxys).all(1)
            # skip bad chromosomes
            if np.sum(_valid_flags) <= min_valid_spots or np.mean(_valid_flags) < min_valid_per:
                continue
            #print(_chr, _ihomo)
            for _zxy in _zxys:
                #print((im_radius+_zxy)/pixel_size, _chr2densityArrs[_chr][_ihomo].shape )
                if np.isfinite(_zxy).all():
                    _chr2densityArrs[_chr][_ihomo] = add_source(
                        _chr2densityArrs[_chr][_ihomo], 
                        pos=(im_radius+_zxy)/pixel_size,
                        h=1, sig=_effective_sigma, 
                        size_fold=im_radius*3)
            # normalize by counts
            if normalize_counts:
                _chr2densityArrs[_chr][_ihomo] = _chr2densityArrs[_chr][_ihomo] / np.sum(_valid_flags)
            # normalize by pdf
            if normalize_pdf:
                _chr2densityArrs[_chr][_ihomo] = _chr2densityArrs[_chr][_ihomo] / np.sum(_chr2densityArrs[_chr][_ihomo])
    
    kept_chrs = list(_chr2densityArrs.keys())
    for _chr in kept_chrs:
        if not return_empty:
                _homolog_kepts = _chr2densityArrs[_chr].any((1,2,3))
                #print(_homolog_kepts)
                if _homolog_kepts.any():
                    _chr2densityArrs[_chr] = _chr2densityArrs[_chr][_homolog_kepts]
                else:
                    del(_chr2densityArrs[_chr])

    return _chr2densityArrs
    