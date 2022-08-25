import numpy as np

def spAligner_2_chr2homologList(cell_data_df, codebook, 
                               info_names=['rna_experiment','fov_id','cell_id','subclass','uid'],
                               fill_blank=True):
    """Function to load spAligner result"""
    # init
    chr_2_homologList = {}
    # loop through chrs 
    for _chr_name in np.unique(cell_data_df['chr']):
        if 'chr' in _chr_name:
            _chr = _chr_name.split('chr')[1]
        else:
            _chr = _chr_name
        # sel chr codebook
        _chr_codebook = codebook[codebook['chr']==_chr]
        # search chr
        _chr_df = cell_data_df[cell_data_df['chr']==_chr_name]
        # init homologs
        _homologs = []
        for _i_fbr in np.unique(_chr_df['fiberidx']):
            _fbr_df = _chr_df[_chr_df['fiberidx']==_i_fbr].copy().sort_values('hyb')
            _inds = _fbr_df['hyb'].values
            _coords = _fbr_df[['z_um', 'x_um', 'y_um']].values 
            if fill_blank:
                _full_coords = np.ones([len(_chr_codebook),3]) * np.nan
                _full_coords[_inds] = _coords
                _homologs.append(_full_coords)
            else:
                _homologs.append(_coords)
        # append
        chr_2_homologList[_chr] = _homologs
    # summary info_dict
    info_dict = {}
    for _n in info_names:
        info_dict[_n] = np.unique(cell_data_df[_n])[0]
        
    return chr_2_homologList, info_dict

