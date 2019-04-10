import sys,glob,os, time
import numpy as np
sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')
import pickle as pickle
import matplotlib.pyplot as plt
import multiprocessing
from . import get_img_info, corrections, visual_tools
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy
from scipy.spatial.distance import pdist, squareform

# function to do segmentation
def Segmentation_All(analysis_folder, folders, fovs, type='small',
                     ref_name='H0R0', num_channel=5, dapi_channel=-1,
                     num_threads=5,
                     illumination_corr=True, correction_folder=_correction_folder, corr_channel='405',
                     denoise_window=5, max_ft_size=25, gl_ft_size=35,
                     conv_th=-5e-5, boundary_th=0.55, signal_cap_ratio=0.20,
                     max_cell_size=30000, min_cell_size=5000, min_shape_ratio=0.038,
                     max_iter=3, shrink_percent=14, dialation_dim=10,
                     segmentation_path='segmentation',
                     save=True, force=False, verbose=True):
    '''wrapped function to do DAPI segmentation
    Inputs:
        analysis_folder: directory of this data, string
        folders: list of sub-folder names, list of string
        fovs: list of field of view names, list of string
        type: type of algorithm used, small or large (for gaussian-laplacian window)
        ref_name: name of reference folder with DAPI in it, string (default: 'H0R0')
        num_channel: total color channel for ref images, int (default: 5)
        dapi_channel: index of channel having dapi, int (default: -1)
        num_threads ** Number of threads used in parallel computing, int (default: 4);
        illumination_corr: whether do illumination correction, bool (default: True)
        correction_folder: full directory for illumination correction profiles, string (default: '')
        denoise_window: window size of denoise filter used in segmentation, int (default: 5)
        max_ft_size: size of max-min filters to get cell boundaries, int (default: 25)
        gl_ft_size: window size for laplacian-gaussian filter, int (default: 35)
        conv_th: maximal convolution threshold, float(default: -5e-5)
        boundary_th: minimal boundary im threshold, float(default: 0.55)
        signal_cap_ratio: ratio to maximum for setting threshold considered as signal, float(default: 0.15)
        max_cell_size: upper limit for object otherwise undergoes extra screening, int(default: 30000)
        min_cell_size: smallest object size allowed as nucleus, int (default:5000 for 2D)
        min_shape_ratio: lower bound of A(x)/I(x)^2 for each nucleus, float (default: 0.041 for IMR90)
        max_iter: maximum iterations allowed in splitting shapes, int (default:3)
        shrink_percent: percentage of label areas removed during splitting, float (0-100, default: 13)
        dialation_dim: dimension for dialation after splitting objects, int (default:10)
        segmentation_path: subfolder of segmentation result, string (default: 'Analysis/segmentation')
        save: whether save segmentation result, bool (default: True)
        force: whether do segmentation despite of existing file, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
    _segmentation_labels: list of images with segmentation results, list of images
    _dapi_ims: original images for DAPI, list of 3D images
    '''

    # check inputs
    if type not in ['small', 'large']:
        raise ValueError(f"type keyword should be 'small' or 'large', but {type} is given!")
    # path to store segmentation result
    _savefolder = analysis_folder+os.sep+segmentation_path;
    # check dir and savefile
    if not os.path.isdir(_savefolder): # if save folder doesnt exist, create
        if verbose:
            print("- create segmentation saving folder", _savefolder)
        os.makedirs(_savefolder)

    # Load reference images
    for _i, _folder in enumerate(folders):
        if os.path.basename(_folder) == ref_name:
            _ref_ims, _ref_names = get_img_info.get_img_hyb(folders, fovs, hyb_id=_i)
    # split image into dictionary
    _ref_im_dic = get_img_info.split_channels_by_image(_ref_ims, _ref_names, num_channel=num_channel, DAPI=False)

    # record dapi_names and dapi_ims
    _process_markers = []
    _process_names, _process_ims = [], []
    for _id, _fov in enumerate(fovs):
        _savefile = _savefolder + os.sep + _fov.replace('.dax', '_segmentation.pkl')
        # if file already exists and not force, just
        if os.path.isfile(_savefile) and not force:
            _process_markers.append(False)
        else:
            _process_markers.append(True)
            _dapi_name = ref_name+os.sep+_fov
            _dapi_im = _ref_im_dic[_dapi_name][-1]
            # do some corrections
            # correct for z axis shift
            _dapi_im = corrections.Z_Shift_Correction(_dapi_im)
            # correct for hot pixels
            _dapi_im = corrections.Remove_Hot_Pixels(_dapi_im)
            # record
            _process_names.append(_dapi_name)
            _process_ims.append(_dapi_im)
    ## segmentation in parallel
    if type =='small':
        _args = [(_im,_nm, 0.5, illumination_corr, 405, correction_folder, 11, denoise_window, 13, signal_cap_ratio, min_cell_size, min_shape_ratio) for _im,_nm in zip(_process_ims, _process_names)];
    elif type=='large':
        _args = [(_im, _nm, 0.5,
                  illumination_corr, 405, correction_folder,
                  13, denoise_window, max_ft_size, gl_ft_size,
                  conv_th, boundary_th, signal_cap_ratio,
                  max_cell_size, min_cell_size, min_shape_ratio,
                  max_iter, shrink_percent,
                  dialation_dim, 0.1, 50, False, verbose) for _im,_nm in zip(_process_ims, _process_names)];
    if verbose:
        print(f"--- {len(_args)} of fovs are being processed by {num_threads} threads, chunk_size=1");
    # start parallel computing
    start_time = time.time();
    pool = multiprocessing.Pool(num_threads);
    if type =='small':
        _process_labels = pool.starmap(visual_tools.DAPI_segmentation, _args, chunksize=1);
    elif type=='large':
        _process_labels = pool.starmap(visual_tools.DAPI_convoluted_segmentation, _args, chunksize=1);
    pool.close()
    pool.join()
    pool.terminate()
    if verbose:
        print("--- used time is ", time.time() - start_time)
    # revert format
    _process_dic = {_dapi_name:_l[0] for _dapi_name, _l in zip(_process_names, _process_labels)}
    # merge with labels loaded from files and save
    if verbose:
        print("-- Merge result with stored results and Save");
    _dapi_ims, _segmentation_labels = [], []
    for _id, (_fov, _mk) in enumerate(zip(fovs, _process_markers)):
        # save file
        _savefile = _savefolder + os.sep + _fov.replace('.dax', '_segmentation.pkl')
        _dapi_name = ref_name+os.sep+_fov;
        _dapi_im = _ref_im_dic[_dapi_name][-1];
        if _mk: # did calculation previously
            if verbose:
                print(f"--- fov:{fovs[_id]} has been processed.");
            # store image
            _dapi_ims.append(_dapi_im);
            # store segmentation result:
            _segmentation_labels.append(_process_dic[_dapi_name]);
            if save:
                pickle.dump([_process_dic[_dapi_name], _dapi_im], open(_savefile, 'wb'))
        else:
            if verbose:
                print(f"--- fov-{fovs[_id]} has been loaded from file.")
            _segmentation_label, _dapi_im = pickle.load(open(_savefile, 'rb'))
            # store
            _dapi_ims.append(_dapi_im)
            _segmentation_labels.append(_segmentation_label)


    return _segmentation_labels, _dapi_ims

def Segmentation_Fov(analysis_folder, folders, fovs, fov_id, type='small',
                     ref_name='H0R0', num_channel=5, dapi_channel=-1,
                     illumination_corr=True, correction_folder=_correction_folder, corr_channel='405',
                     denoise_window=5, max_ft_size=25, gl_ft_size=35,
                     conv_th=-5e-5, boundary_th=0.55, signal_cap_ratio=0.20,
                     max_cell_size=30000, min_cell_size=5000, min_shape_ratio=0.038,
                     max_iter=3, shrink_percent=14, dialation_dim=10,
                     segmentation_path='segmentation',
                     make_plot=False, save=True, force=False, verbose=True):
    '''wrapped function to do DAPI segmentation
    Inputs:
        analysis_folder: directory of this data, string
        folders: list of sub-folder names, list of string
        fovs: list of field of view names, list of string
        type: type of algorithm used, small or large (for gaussian-laplacian window)
        ref_name: name of reference folder with DAPI in it, string (default: 'H0R0')
        num_channel: total color channel for ref images, int (default: 5)
        dapi_channel: index of channel having dapi, int (default: -1)
        num_threads ** Number of threads used in parallel computing, int (default: 4);
        illumination_corr: whether do illumination correction, bool (default: True)
        correction_folder: full directory for illumination correction profiles, string (default: '')
        denoise_window: window size of denoise filter used in segmentation, int (default: 5)
        max_ft_size: size of max-min filters to get cell boundaries, int (default: 25)
        gl_ft_size: window size for laplacian-gaussian filter, int (default: 35)
        conv_th: maximal convolution threshold, float(default: -5e-5)
        boundary_th: minimal boundary im threshold, float(default: 0.55)
        signal_cap_ratio: ratio to maximum for setting threshold considered as signal, float(default: 0.15)
        max_cell_size: upper limit for object otherwise undergoes extra screening, int(default: 30000)
        min_cell_size: smallest object size allowed as nucleus, int (default:5000 for 2D)
        min_shape_ratio: lower bound of A(x)/I(x)^2 for each nucleus, float (default: 0.041 for IMR90)
        max_iter: maximum iterations allowed in splitting shapes, int (default:3)
        shrink_percent: percentage of label areas removed during splitting, float (0-100, default: 13)
        dialation_dim: dimension for dialation after splitting objects, int (default:10)
        segmentation_path: subfolder of segmentation result, string (default: 'Analysis/segmentation')
        save: whether save segmentation result, bool (default: True)
        force: whether do segmentation despite of existing file, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
    _segmentation_label: list of images with segmentation results, list of images
    _dapi_im: original images for DAPI, list of 3D images
    '''
    # path to store segmentation result
    _savefolder = analysis_folder+os.sep+segmentation_path;
    _savefile = _savefolder + os.sep + fovs[fov_id].replace('.dax', '_segmentation.pkl');
    # check dir and savefile
    if not os.path.isdir(_savefolder): # if save folder doesnt exist, create
        if verbose:
            print("-- create segmentation saving folder", _savefolder)
        os.makedirs(_savefolder);
    if os.path.isfile(_savefile) and not force: # load existing file
        if verbose:
            print("-- load segmentation result from filename:", _savefile);
        _segmentation_label, _dapi_im = pickle.load(open(_savefile, 'rb'))
        return _segmentation_label, _dapi_im
    else: # do segmentation
        # Load reference images
        for _i, _folder in enumerate(folders):
            if os.path.basename(_folder) == ref_name:
                _ref_ims, _ref_names = get_img_info.get_img_hyb(folders, fovs, hyb_id=_i)
        # split channels
        _ref_im_dic = get_img_info.split_channels_by_image(_ref_ims, _ref_names, num_channel=num_channel, DAPI=False)
        # acquire dapi name and image
        _dapi_name = ref_name+os.sep+fovs[fov_id];
        _dapi_im = _ref_im_dic[_dapi_name][-1];
        # do some corrections
        # correct for z axis shift
        _dapi_im = corrections.Z_Shift_Correction(_dapi_im)
        # correct for hot pixels
        _dapi_im = corrections.Remove_Hot_Pixels(_dapi_im)
        # segmentation!
        if type == 'small':
            _segmentation_label = visual_tools.DAPI_segmentation(_dapi_im, _dapi_name,
                                                    illumination_correction=illumination_corr,
                                                    correction_folder=correction_folder,
                                                    shape_ratio_threshold=min_shape_ratio,
                                                    signal_cap_ratio=signal_cap_ratio,
                                                    denoise_window=denoise_window,
                                                    make_plot=make_plot, verbose=verbose)[0]
        elif type == 'large':
            _segmentation_label = visual_tools.DAPI_convoluted_segmentation(_dapi_im, _dapi_name,
                      illumination_correction=illumination_corr, correction_folder=correction_folder,
                      denoise_window=denoise_window, mft_size=max_ft_size, glft_size=gl_ft_size,
                      max_conv_th=conv_th, min_boundary_th=boundary_th, signal_cap_ratio=signal_cap_ratio,
                      max_cell_size=max_cell_size, min_cell_size=min_cell_size, min_shape_ratio=min_shape_ratio,
                      max_iter=max_iter, shrink_percent=shrink_percent,
                      dialation_dim=dialation_dim, make_plot=make_plot, verbose=verbose)[0]

        # save
        if save:
            if verbose:
                print("- saving segmentation result to file:",_savefile)
            pickle.dump([_segmentation_label, _dapi_im], open(_savefile, 'wb'))
        return _segmentation_label, _dapi_im

# a relative memory efficient way to load Images
def load_image_fov(folders, fovs, fov_id, channels, color_dic,
                   num_threads=12, loading_type='raw', type_key=None,
                   max_chunk_size=5,
                   z_shift_corr=True, hot_pixel_remove=True, illumination_corr=True, chromatic_corr=True,
                   correction_folder=_correction_folder,
                   temp_folder=_temp_folder, overwrite_temp=True,
                   return_type='filename', verbose=False):
    """Wrapped function to do batch image loading and processing.
    Inputs:
        folders: name of hyb-folders used in this experiment, list of strings
        fovs: name of field-of-views in this experiment, list of strings
        fov_id: index of chosen fov in fovs, int (smaller than len(fovs))
        channels: channels used in this experiment, list
        color_dic: color usage of this experiment, loaded from Color_Usage.csv, dic
        num_threads: number of threads used in this loading process, int (default:5)
        loading_type: type of images you want to load, raw/beads/dapi/unique/combo
        type_key: special key used for certain type, None/'beads'/'DAPI'/'u'/'c'
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        correction_folder: where to find correction folders, string (default: system correction folder)
        temp_folder: directory for storing temp files, string (default: somewhere in SSD)
        return_type: type return expected, 'filename'/'mmap'/'image'
        verbose: whether say something during the process, bool (default: False)
    Outputs:
        v1:
        splitted_ims
        v2:
        out_ims: list of images
        out_names: corresponding folder/fov_name
        out_channels: which channel this dic is from
        """
    # default settings
    loading_keys = {'raw':'',
                    'all':'',
                    'beads':'beads',
                    'dapi':'DAPI',
                    'unique':'u',
                    'combo':'c'}
    chromatic_corr_channels = ['750','647','561'];
    if not type_key:
        type_key = loading_keys[loading_type.lower()];
    # check return type
    return_type = return_type.lower()
    if return_type not in ['filename','mmap','image','filedic']:
        raise ValueError('Wrong kwd return_type given!');

    # Load image
    _ims, _names = get_img_info.get_img_fov(folders, fovs, fov_id, verbose=verbose)
    _channels = [str(ch) for ch in channels]
    # number of channels and whether use dapi
    if '405' in _channels:
        _num_ch = len(_channels) - 1
        _use_dapi = True
    else:
        _num_ch = len(_channels)
        _use_dapi = False
    # use dapi
    if _use_dapi:
        if loading_type.lower() == 'dapi':
            _dapi_index = get_img_info.find_dapi_channel(color_dic, type_key)
    # bead_index
    if loading_type.lower() == 'beads':
        _bead_index = get_img_info.find_bead_channel(color_dic, type_key)

    ## split images
    _splitted_ims = get_img_info.split_channels_by_image(_ims, _names, num_channel=_num_ch, DAPI=_use_dapi,verbose=verbose)
    _ims = None # release images

    # for beads
    if loading_type.lower() == 'beads':
        if verbose:
            print("-- Loading bead images.")
        _cand_ims, _cand_names, _cand_channels = [],[],[]
        for _n, _ims in _splitted_ims.items():
            _cand_ims.append(_ims[_bead_index])
            _cand_names.append(_n)
            _cand_channels.append(_channels[_bead_index])
    # for dapi
    elif loading_type.lower() == 'dapi':
        if verbose:
            print("-- Loading dapi images")
        _cand_ims, _cand_names, _cand_channels = [],[],[]
        for _n, _ims in _splitted_ims.items():
            if len(_ims) == len(channels):
                _cand_ims.append(_ims[_dapi_index])
                _cand_names.append(_n)
                _cand_channels.append(_channels[_dapi_index])
    # for raw, unique and combo:
    elif loading_type.lower() in ['all', 'raw', 'unique', 'combo']:
        if verbose:
            print("-- Loading all images for this fov")
        _cand_ims, _cand_names, _cand_channels, _cand_ch_id = [],[],[],[]
        for _n, _ims in _splitted_ims.items():
            if _n.split(os.sep)[0] in color_dic: # if this hyb is given in color_dic
                for _ch_id, _im in enumerate(_ims):
                    if loading_type.lower() == 'unique' and type_key in color_dic[_n.split(os.sep)[0]][_ch_id]:
                        _cand_ims.append(_im)
                        _cand_names.append(_n)
                        _cand_channels.append(_channels[_ch_id])
                        _cand_ch_id.append(_ch_id)
                    elif loading_type.lower() == 'combo' and type_key in color_dic[_n.split(os.sep)[0]][_ch_id]:
                        _cand_ims.append(_im)
                        _cand_names.append(_n)
                        _cand_channels.append(_channels[_ch_id])
                        _cand_ch_id.append(_ch_id)
                    elif loading_type.lower() == 'raw' or loading_type.lower() == 'all':
                        _cand_ims.append(_im)
                        _cand_names.append(_n)
                        _cand_channels.append(_channels[_ch_id])
                        _cand_ch_id.append(_ch_id)
    # release!
    del(_splitted_ims)

    ## multi-processing
    # corrections
    _temp_fls = [temp_folder+os.sep+_nm.replace(os.sep, '-').replace('.dax', '_'+str(_ch)+'_corrected') for _nm, _ch in zip(_cand_names, _cand_channels)]
    _args = [(_im, _channel, correction_folder, \
              z_shift_corr, hot_pixel_remove, illumination_corr, \
              ((_channel in chromatic_corr_channels) and chromatic_corr), \
              temp_folder, _temp_fl, overwrite_temp, return_type, verbose) \
             for _im,_channel, _temp_fl in zip(_cand_ims,_cand_channels, _temp_fls)] # full args
    _chunk_size = min([int(np.ceil(len(_args)/num_threads)),int(max_chunk_size)])
    if verbose:
        print(f"--- start image correction with {num_threads} threads, chunk_size={_chunk_size}")
    start_time = time.time();
    pool = multiprocessing.Pool(num_threads);
    if return_type == 'filename': # only return filenames, therefore they doesnt need to be synced
        pool.starmap_async(corrections.correction_wrapper, _args, chunksize=_chunk_size);
    else: # return mmap or directly image
        _corrected_ims = pool.starmap(corrections.correction_wrapper, _args, chunksize=_chunk_size);
    pool.close()
    pool.join()
    if verbose:
        print("--- time cost for this correction:", time.time()-start_time)
    pool.terminate()
    # release and terminate
    del(_cand_ims, _args, pool)
    if return_type == 'image' or return_type == 'mmap':
        ## return and save
        # re-compile into a dic
        if loading_type.lower() in ['all', 'raw', 'unique', 'combo']:
            _splitted_ims = {};
            for _hyb_fd, _info in color_dic.items():
                _splitted_ims[_hyb_fd+os.sep+fovs[fov_id]] = [[] for _i in range(len(_info))]
            for _im, _nm, _ch in zip(_corrected_ims, _cand_names, _cand_channels):
                _splitted_ims[_nm][_channels.index(_ch)] = _im;
            return _splitted_ims
        # directly return lists
        else:
            return _corrected_ims, _cand_names, _cand_channels
    elif return_type == 'filename':
        _cand_fls = [_tp+'.npy' for _tp in _temp_fls];
        return _cand_fls, _cand_names, _cand_channels
    elif return_type == 'filedic':
        pass
        



# Function
# load color usage
# load encoding scheme (if combo)
# do drift correction
# crop image based on segmentation labels
def Crop_Images_Field_of_View(master_folder, folders, fovs, fov_id,
                              segmentation_labels, dapi_ims, encoding_type,
                              analysis_path='Analysis',
                              drift_corr_size=125,
                              num_channel=None,
                              th_seed=300,dynamic=False,
                              color_filename='',
                              encoding_filename='',
                              correction_folder=_correction_folder,
                              save=False, verbose=True):
    '''Crop images for a certain field of view
    Inputs:
        master_folder: directory of this data, string
        folders: list of sub-folder names, list of string
        fovs: list of field of view names, list of string'''

    # check segmentation label and dapi image input
    if isinstance(segmentation_labels, list):
        _segmentation_label = segmentation_labels[fov_id];
    else:
        _segmentation_label = segmentation_labels
    if isinstance(dapi_ims, list):
        _dapi_im = dapi_ims[fov_id];
    else:
        _dapi_im = dapi_ims
    if correction_folder == '':
        correction_folder = master_folder;
    # load color usage
    if verbose:
        print("Loading color usage");
    if color_filename:
        _color_dic, _use_dapi, _colors = get_img_info.Load_Color_Usage(master_folder, color_filename=color_filename, return_color=True)
    else:
        _color_dic, _use_dapi, _colors = get_img_info.Load_Color_Usage(master_folder, return_color=True)

    # load multi-channel images in this field of view
    if verbose:
        print("Loading images for field of view", fov_id)
    _total_ims, _full_names = get_img_info.get_img_fov(folders, fovs, fov_id=fov_id)

    # split into different channels, return a dictionary
    if verbose:
        print("Splitting total images into different channels")
    if not num_channel:
        num_channel = len(_colors) - 1
    _im_dic = get_img_info.split_channels_by_image(_total_ims, _full_names, num_channel=num_channel, DAPI=_use_dapi)

    # drift correction:
    if verbose:
        print("Correcting drifts")
    _bead_channel = get_img_info.find_bead_channel(_color_dic);
    _bead_ims = [v[_bead_channel] for k,v in sorted(list(_im_dic.items()), key=lambda k_v: int(k_v[0].split('H')[1].split('R')[0]))]
    _bead_names = [k for k,v in sorted(list(_im_dic.items()), key=lambda k_v1: int(k_v1[0].split('H')[1].split('R')[0]))]
    _total_drift, _fail_count = corrections.STD_beaddrift_sequential(_bead_ims,_bead_names,
                                                    drift_folder=master_folder+os.sep+analysis_path,
                                                    fovs=fovs,
                                                    fov_id=fov_id,
                                                    drift_size=drift_corr_size,
                                                    overwrite=False, save=True,
                                                    correction_folder=correction_folder,
                                                    th_seed=th_seed, dynamic=dynamic, verbose=True)

    # take care of combo type:
    if encoding_type.lower() == 'combo':
        if verbose:
            print("Processing combo images")
        # load encoding scheme
        if encoding_filename:
            _encoding_scheme = get_img_info.Load_Encoding_Scheme(master_folder,
                                                            encoding_filename=encoding_filename,
                                                            return_info=False)
        else:
            _encoding_scheme = get_img_info.Load_Encoding_Scheme(master_folder, return_info=False)

        # initialize cell list
        _cell_num = np.max(_segmentation_label) # number of cells in this field of view
        print("- Number of cells in fov:", _cell_num)
        _cell_list = [{_color:{'names':[],'ims':[],'matrices':[]} for _color, _encoding_dic in sorted(_encoding_scheme.items())} for _i in range(_cell_num)];

        # append information for 405 channel (DAPI)
        if _use_dapi:
            _cell_dapi = visual_tools.crop_cell(_dapi_im, _segmentation_label);
            for _dict, _dapi in zip(_cell_list, _cell_dapi):
                _dict['405'] = {'ims':[_dapi], 'names':[folders[0]+os.sep+fovs[fov_id]]}
        # crop images!
        if verbose:
            print("Start corpping images in all channels")

        for _channel, _dic in sorted(_encoding_scheme.items()):
            if verbose:
                print("- processing channel:", _channel)
            for _folder_list, _matrices in zip(_dic['names'], _dic['matrices']):
                if verbose:
                    print("-- processing folders:", _folder_list)
                _group_cropped_ims = []
                for _fd in _folder_list:
                    _name = _fd+os.sep+fovs[fov_id];
                    _channel_index = _colors.index(int(_channel))
                    _im = _im_dic[_name][_channel_index];
                    # convert to local
                    _corr_im = _im;
                    # correct for z axis shift
                    _corr_im = corrections.Z_Shift_Correction(_corr_im, verbose=verbose)
                    # correct for hot pixels
                    _corr_im = corrections.Remove_Hot_Pixels(_corr_im, verbose=verbose)
                    # illumination correction
                    _corr_im = corrections.Illumination_correction(_im,
                    correction_channel=_channel, correction_folder=correction_folder, verbose=False)
                    # chromatic abbrevation
                    _corr_im = corrections.Chromatic_abbrevation_correction(_corr_im[0],
                    correction_channel=_channel, correction_folder=correction_folder, verbose=False)
                    # crop
                    _group_cropped_ims.append(visual_tools.crop_cell(_corr_im[0], _segmentation_label, _total_drift[_name]))
                # save cropped ims
                for _cell_id in range(_cell_num):
                    _cell_list[_cell_id][_channel]['names'].append(_folder_list);
                    _cell_list[_cell_id][_channel]['matrices'].append(_matrices);
                    _cell_list[_cell_id][_channel]['ims'].append([_cims[_cell_id] for _cims in _group_cropped_ims])
        # save
        if save:
            for _cell_id in range(_cell_num):
                _save_filename = master_folder + os.sep + analysis_path + os.sep + fovs[fov_id].replace('.dax','_cell_'+str(_cell_id)+'_'+str(encoding_type)+'.pkl')
                pickle.dump(_cell_list[_cell_id], open(_save_filename, 'wb'))

        return _cell_list, _color_dic, _colors, _encoding_scheme
    # process "unique"
    elif encoding_type.lower() == 'unique':
        if verbose:
            print("Processing unique images");

        # initialize cell list
        _cell_num = np.max(_segmentation_label) # number of cells in this field of view
        print("- Number of cells in fov:", _cell_num)
        _cell_list = [];
        for _cell_id in range(_cell_num):
            _sample_cell = {}
            for _index, _color in enumerate(_colors):
                if _index != _bead_channel:
                    _sample_cell[str(_color)] = {'names':[],'ims':[], 'u_names':[]};
            _cell_list.append(_sample_cell);

        # append information for 405 channel (DAPI)
        if _use_dapi:
            _cell_dapi = visual_tools.crop_cell(_dapi_im, _segmentation_label);
            for _dict, _dapi in zip(_cell_list, _cell_dapi):
                _dict['405'] = {'ims':[_dapi], 'names':[folders[0]+os.sep+fovs[fov_id]]}

        # crop images!
        if verbose:
            print("Start corpping images in all channels")
        # loop through all available unique hybes
        for _folder_name, _infos in sorted(list(_color_dic.items()), key=lambda k_v2:int(k_v2[0].split('H')[1].split('R')[0])):
            for _channel_index, (_channel, _info) in enumerate(zip(_colors[:len(_infos)], _infos)):
                if encoding_type[0] in _info:
                    if verbose:
                        print("- processing:", _folder_name, _channel, _info)
                    _name = _folder_name+os.sep+fovs[fov_id];
                    _im = _im_dic[_name][_channel_index];
                    # convert to local
                    _corr_im = _im;
                    # correct for z axis shift
                    _corr_im = corrections.Z_Shift_Correction(_corr_im, verbose=verbose)
                    # correct for hot pixels
                    _corr_im = corrections.Remove_Hot_Pixels(_corr_im, verbose=verbose)
                    # illumination correction
                    _corr_im = corrections.Illumination_correction(_im,
                    correction_channel=_channel, correction_folder=correction_folder, verbose=False)
                    # chromatic abbrevation
                    _corr_im = corrections.Chromatic_abbrevation_correction(_corr_im[0],
                    correction_channel=_channel, correction_folder=correction_folder, verbose=False)
                    # crop
                    _cropped_im = visual_tools.crop_cell(_corr_im[0], _segmentation_label, _total_drift[_name])
                    # save info
                    for _cell_id in range(_cell_num):
                        _cell_list[_cell_id][str(_channel)]['names'].append(_folder_name)
                        _cell_list[_cell_id][str(_channel)]['ims'].append(_cropped_im[_cell_id])
                        _cell_list[_cell_id][str(_channel)]['u_names'].append(_info)
                   # save
        if save:
            for _cell_id in range(_cell_num):
                _save_filename = master_folder + os.sep + analysis_path + os.sep + fovs[fov_id].replace('.dax','_cell_'+str(_cell_id)+'_'+str(encoding_type)+'.pkl')
                pickle.dump(_cell_list[_cell_id], open(_save_filename, 'wb'))

        return _cell_list, _color_dic, _colors

# update cell list from Crop_Images_Field_of_View
# append info from sparse reconstruction
def Update_Raw_Data(raw_file, decode_file, im_key='cs_ims', name_key='cs_names',
                    key_prefix=['cell-','channel-','group-'], allowed_channels=['750','647','561'],
                    verbose=True):
    '''Function to merge cropped raw data and decoded data
    Inputs:
        raw_file: raw cropped cell list, which could be filename or already loaded list
        decode_file: decoding result by sparse reconstruction, filename or already loaded dic
        im_key: key for decoded image refs, string (usually 'cs_ims' or 'sr_ims')
        name_key: key for decoded compatible name refs, string (usually 'cs_names' or 'sr_names')
        key_prefix: string prefix used in decode_files from Harry, list of strings
        allowed_channels: allowed allowed_channels for data images, list of string
        verbose: say something!, bool
    Output:
        _cell_list: updated cell list with decoded image merged into them
        '''
    # load raw data
    if verbose:
        print('- Loading raw cell list.');
    if isinstance(raw_file, list):
        _cell_list = raw_file
        if verbose:
            print('-- cell list directly given')
    elif isinstance(raw_file, str):
        if verbose:
            print('-- loading cell_list from '+str(raw_file))
        if '.zip' in raw_file:
            import gzip
            _cell_list = pickle.load(gzip.open(raw_file,'rb'))
        if '.gz' in raw_file:
            import gzip
            _cell_list = pickle.load(gzip.open(raw_file,'rb'))
        elif '.pkl' in raw_file:
            _cell_list = pickle.load(open(raw_file,'rb'))
        else:
            raise ValueError('wrong filetype for raw_file input, .gz or .pkl required!');
    else:
        raise ValueError('wrong data type of raw_file! not filename nor cell_list');

    # load decoded data
    if verbose:
        print('- Loading decoded images.');
    if isinstance(decode_file, dict):
        _decoded_ims = decode_file
        if verbose:
            print('-- decoded images directly given as dict')
    elif isinstance(decode_file, list):
        _decoded_ims = decode_file
        if verbose:
            print('-- decoded images directly given as list')
    elif isinstance(decode_file, str):
        if verbose:
            print('-- loading decoded_ims from '+str(decode_file))
        if '.gz' in decode_file:
            import gzip
            _decoded_ims = pickle.load(gzip.open(decode_file,'rb'))
        elif '.pkl' in decode_file:
            _decoded_ims = pickle.load(open(decode_file,'rb'))
        else:
            raise ValueError('wrong filetype for decode_file input, .gz or .pkl required!');
    else:
        raise ValueError('wrong data type of decode_file! not filename nor decoded_ims');
    ## convert decoded ims into list
    if isinstance(_decoded_ims, dict):
        _decoded_list = [None for _cell in _cell_list]
        for k,v in sorted(_decoded_ims.items()):
            if key_prefix[0] in k:
                _cell_id = int(k.split(key_prefix[0])[-1])
                _decoded_list[_cell_id] = v
        _decoded_ims = _decoded_list
    # match length
    if not _decoded_ims:
        raise ValueError('decoded image loading failed')
    elif len(_decoded_ims) != len(_cell_list):
        print('decoded images doesnot match raw cell list.')

    ## Loop through all cells and merge info
    if verbose:
        print("- Merging raw cell_list with decoded images")
    # a list to record successful cells
    _success_cells = [True for _cell in _cell_list]
    for _cell_id, _cell in enumerate(_cell_list):
        # load decoded result
        if _decoded_ims[_cell_id]:
            _d_ims = _decoded_ims[_cell_id];
        else:
            _success_cells[_cell_id] = False
            if verbose:
                print("--- no decoded info for cell:", _cell_id);
            continue

        if verbose:
            print('-- cell:', _cell_id);
        for _channel, _info in sorted(_cell.items()):
            if _channel not in allowed_channels:
                continue
            # initialize
            _cell_list[_cell_id][_channel][im_key] = []
            _cell_list[_cell_id][_channel][name_key] = []
            # loop through each group
            for _group, _matrix in enumerate(_info['matrices']):
                _n_hyb, _n_reg = np.shape(_matrix) # number of hybs and regions
                # check if this groups is decoded
                if key_prefix[1]+str(_channel) not in _d_ims:
                    _success_cells[_cell_id] = False
                    if verbose:
                        print("--- no decoded info for cell:", _cell_id, ', channel:', _channel);
                    continue;
                elif key_prefix[2]+str(_group) not in _d_ims[key_prefix[1]+str(_channel)]:
                    _success_cells[_cell_id] = False
                    if verbose:
                        print("--- no decoded info for cell:", _cell_id, ', channel:', _channel, ', group:',_group);
                    continue;
                # save decoded images
                _ims = list(_d_ims[key_prefix[1]+str(_channel)][key_prefix[2]+str(_group)])[:_n_reg];
                _cell_list[_cell_id][_channel][im_key].append(_ims)
                # save decoded names
                _names = [np.unique(_matrix[:,_col_id])[-1] for _col_id in range(_n_reg)]
                _cell_list[_cell_id][_channel][name_key].append(_names)
    # screen out failed cells
    _kept_cell_list = [_cell for _cell, _kept in zip(_cell_list, _success_cells) if _kept]

    return _kept_cell_list


# decoding by compressive sensing
def compressive_sensing_decomposition(master_folder, fov_name, cell_list, color_dic, encoding_scheme,
                                      analysis_path='Analysis',
                                      num_combination=2,
                                      signal_percentile=97., gaussian_sigma=0.5, zoom_ratio=2., residue_threshold = 2.,
                                      make_plot=False, save=False, force=False, verbose=True):
    '''Decode groups of images by compressive sensing based algorithm:
    Inputs:
        master_folder: directory for this dataset, string
        fov_name: name of this field of view, fovs[fov_id], string
        cell_list: list of all information for cropped images,
                generated by Crop_Images_Field_of_View, list of dics
        color_dic: dictionary of Color_Usage generated by Load_Color_Usage, dic
        encoding_scheme: dictionary of encoding, generated by Load_Encoding_Scheme, dic
        analysis_path: sub-folder for analysis files
        num_combination: number of regions contributing to each pixel, int (default: 2)
        signal_percentile: percentile of signal intensity for thresholding image, float (default: 95)
        gaussian_sigma: sigma for gaussian filter, float (if 0 or None, no gaussian filter applied)
        residue_threshold: threshold for residue/mean for non-negative least-square fitting, float (default: 1.)
        zoom_ratio: whether zoom before decomposition, float (zoom in will increase speed)
        save: whether save result into pickle file, bool (default: True)
        force: whether do the calculate despite of existing result file, bool (default: False)
        verbose: say something!, bool
    Outputs:
        _updated_cell_list: original cell list append with cs_ims and cs_names (cs: compressive sensing)
        '''
    import itertools
    from scipy import optimize
    from scipy import stats
    from scipy.ndimage import zoom
    from scipy.ndimage import gaussian_filter
    import pickle as pickle
    #pickle.dump(updated_cell_list, open(analysis_folder+os.sep+fovs[test_fov_id].replace('.dax','_decomposed_list.pkl'), 'wb'))
    # check if result pickle file already exist:

    _savefile = master_folder+os.sep+analysis_path+os.sep+fov_name.replace('.dax','_cs_list.pkl')
    if verbose:
        print("- check existence of savefile:", os.path.isfile(_savefile))
    if os.path.isfile(_savefile) and not force:
        if verbose:
            print("- loading savefile", _savefile)
        _updated_cell_list = pickle.load(open(_savefile, 'rb'), encoding='latin1');
        return _updated_cell_list

    # make a copy of local variables
    if verbose:
        print("- compressive sensing decomposition.")
    _updated_cell_list = [_cell for _cell in cell_list];
    _color_dic = color_dic;
    _encoding_scheme = encoding_scheme;

    # standardize used parameters
    _num_hyb, _num_reg = np.shape(_updated_cell_list[0][list(encoding_scheme.keys())[0]]['matrices'][0]);
    if verbose:
        print("-- number of hybs per group:", _num_hyb)
        print("-- number of regions per group:", _num_reg)
    _region_combinations = list(itertools.combinations(list(range(_num_reg)), num_combination)) # all possible combinations

    # loop through cells in cell_list:
    for _i, _cell in enumerate(_updated_cell_list):
        if verbose:
            print("-- processing cell:", _i)
        for _channel, _dic in sorted(_cell.items()):
            if _channel in list(_encoding_scheme.keys()):
                if verbose:
                    print("--- processing channel:", _channel)
                # initialize cs_names and cs_ims
                _updated_cell_list[_i][_channel]['cs_ims'] = [];
                _updated_cell_list[_i][_channel]['cs_names'] = [];

                # loop through groups
                for _im_group, _name_group, _matrix in zip(_dic['ims'], _dic['names'], _dic['matrices']):
                    if verbose:
                        print("---- processing group:", _name_group)
                    # check dimension
                    if len(_im_group) != len(_name_group) or len(_im_group) != _num_hyb:
                        raise ValueError('input dimension doesnt match!');
                    # remove background
                    _filtered_im_group = [_im for _im in _im_group];
                    if signal_percentile:
                        _limits = [stats.scoreatpercentile(_im, signal_percentile) for _im in _im_group];
                        for _fim, _limit in zip(_filtered_im_group, _limits):
                            _fim[_fim <= _limit] = 0;
                    # gaussian filter
                    if gaussian_sigma:
                        _filtered_im_group = [gaussian_filter(_im, gaussian_sigma) for _im in _im_group]

                    # zoom in
                    if zoom_ratio:
                        _filtered_im_group = [zoom(_im, 1/zoom_ratio) for _im in _filtered_im_group];
                    # transform into 2d
                    _data = np.array(_filtered_im_group).reshape(_num_hyb, -1);
                    _decoded = np.zeros([_num_reg, _data.shape[1]]); # decomposed data
                    # hyb matrix
                    _hyb_matrix = np.array(_matrix >=0, dtype=np.int)
                    # loop through pixels
                    for _pxl in range(_data.shape[1]):
                        # initialize region contribution to this pixel and their index
                        _reg_value, _reg_index = np.zeros(num_combination),np.zeros(num_combination)
                        # initialize min_residue for this pixel
                        _min_residue = np.inf;
                        # fluorescence intensity profile of this pixel:
                        _pxl_profile = _data[:, _pxl];
                        if _pxl_profile.any() == 0: # this pixel is always zero:
                            continue # skip this pixel, i.e., all zeros for all regions
                        else:
                            # loop through all combinations
                            for _c in range(len(_region_combinations)):
                                # extract info
                                _chosen_combo = _hyb_matrix[:, list(_region_combinations[_c])]
                                # non-negative least square fitting
                                _v, _rnorm = optimize.nnls(_chosen_combo, _pxl_profile);
                                if _rnorm < _min_residue: # if this combo has smaller residue, update
                                    _min_residue = _rnorm;
                                    _reg_value = _v;
                                    _reg_index = np.array(_region_combinations[_c]);
                            # choose the best combo
                            if _min_residue < np.mean(_pxl_profile) * residue_threshold:
                                for _j in range(num_combination):
                                    _decoded[_reg_index[_j], _pxl] = _reg_value[_j]
                    # reshape
                    _dims = np.shape(_filtered_im_group[0]);
                    _decoded = _decoded.reshape(_num_reg, _dims[0], _dims[1], _dims[2]);
                    # split into list of images
                    _cs_ims = [_decoded[_j] for _j in range(_num_reg)];
                    # zoom out
                    if zoom_ratio:
                        _cs_ims = [zoom(_cs_im, zoom_ratio) for _cs_im in _cs_ims];
                    _cs_names = [np.max(np.unique(_matrix[:,_col])) for _col in range(_num_reg)]
                    # save cs_ims and cs_names
                    _updated_cell_list[_i][_channel]['cs_ims'].append(_cs_ims);
                    _updated_cell_list[_i][_channel]['cs_names'].append(_cs_names);
    if save:
        if verbose:
            print("- Saving file:", _savefile)
        pickle.dump(_updated_cell_list, open(_savefile,'wb'));

    return _updated_cell_list


# Generate chromosome from cell_list
def generate_chromosome_from_cell_list(cell_list, encoding_type, merging_channel=['750','647','561'],
                                       signal_percentile=97., return_im=False, verbose=True):
    '''Generate chromosome based on cell_list, which has been cropped and corrected.
    Inputs:
        cell_list: list of cells with cropped and drift corrected images, list of dics
        encoding_type: encoding of regions in this list, 'combo' or 'unique'
        merging_channel: use which channel to merge as chromosome, list (default: ['750','647','561'])
        signal_percentile: percentile of signal intensity for thresholding image, float (default: 95)
        verbose: whether say something during processing, bool (default: True)
    Outputs:
        _chrom_cell_list: cell list appending with chromosome image, list of dics
    '''
    import numpy as np
    import os
    from scipy.stats import scoreatpercentile
    if verbose:
        print("- generating chromosome from cell_list.")
    # check merging_channel input
    for _channel in merging_channel:
        if str(_channel) not in list(cell_list[0].keys()):
            raise ValueError('merging_channel not compatible with cell list info.')
    # initialize _chrom_cell_list:
    _chrom_cell_list = cell_list;
    _total_im_list = [];
    if encoding_type.lower() == 'combo':
        # loop through all images
        for _cell_id, _cell in enumerate(_chrom_cell_list):
            if verbose:
                print("-- calculating mean profile of cell", _cell_id)
            _total_im = np.zeros(_cell[str(merging_channel[0])]['ims'][0][0].shape)
            for _channel in merging_channel:
                # extract all images
                _im_list = []
                for _group in _cell[str(_channel)]['ims']:
                    for _im in _group:
                        _im_list.append(_im)
                        _im_list[-1][_im < scoreatpercentile(_im, signal_percentile)] = 0;
                _total_im += np.array(sum(_im_list), dtype=np.float);

            _total_im_list.append(_total_im);
            _chrom_cell_list[_cell_id]['chrom'] = _total_im
        if return_im:
            return _chrom_cell_list, _total_im_list
        else:
            return _chrom_cell_list
    elif encoding_type.lower() == 'unique':
        # loop through cells
        for _cell_id, _cell in enumerate(_chrom_cell_list):
            if verbose:
                print("-- calculating mean profile of cell", _cell_id)
            _total_im = np.zeros(_cell[str(merging_channel[0])]['ims'][0].shape)
            for _channel in merging_channel:
                _im_list = [];
                for _im in _cell[str(_channel)]['ims']:
                    _im_list.append(_im);
                    _im_list[-1][_im < scoreatpercentile(_im, signal_percentile)] = 0;
                _total_im += np.array(sum(_im_list), dtype=np.float);

            _total_im_list.append(_total_im);
            _chrom_cell_list[_cell_id]['chrom'] = _total_im
        if return_im:
            return _chrom_cell_list, _total_im_list
        else:
            return _chrom_cell_list


# convolution kernel used in identify chromosome
def generate_kernel(_inner=2, _outer=2):
    '''generate a laplacian kernel for convolution, given inner dimension and outer dimension'''
    import numpy as np;
    _kernel = np.zeros([_inner+2*_outer]*3)
    _kernel[:,_outer:_outer+_inner,_outer:_outer+_inner] = -1;
    _kernel[_outer:_outer+_inner,:,_outer:_outer+_inner] = -1;
    _kernel[_outer:_outer+_inner,_outer:_outer+_inner,:] = -1;
    _kernel[_outer:_outer+_inner,_outer:_outer+_inner]= 3 * 2 * _outer / float(_inner)
    return _kernel


# identify chromosome, run this after generate_chromosome_from_cell_list
def identify_chromosome(chrom_cell_list, gfilt_size=0.75, zoom_ratio=2., conv_kernel=[2,2], thres_ratio=0.5,
                        keep_obj=2, return_im=False, verbose=True):
    '''Function to identify chromosome given cell_list with 'chrom' info
    Inputs:
        _chrom_cell_list: cell list with chromosome image, list of dics
        gfilt_size: sigma of gaussian filter,
        zoom_ratio: whether zoom in during segmentation of chromosomes, float (default: 2)
        conv_kernel: dimensions of laplacian kernel to convolute image, list of 2 (default: [2,2])
        thres_ratio: threshold to binarilize image, ratio to the max intensity, float (default: 0.5)
        keep_obj: number of largest objest kept in later processing, int (default: 2)
        return_im: whether return images for visualization, bool (default: False)
        verbose: say something during identifying chromosomes, bool (default: True)
    Output:
    _chrom_cell_list: chromosome cell list appended with chromosome coordinates, list of dics
    (optional) _seg_labels: list of images of chromosome segmentation result, list of 3d images
    '''
    from scipy import ndimage
    from skimage import morphology
    from skimage.segmentation import random_walker
    from scipy.ndimage.filters import gaussian_filter

    # initialize labels
    _chrom_cell_list = chrom_cell_list; # copy
    _seg_labels = [] # intialize
    # loop through cells
    if verbose:
        print("- identifying chromosomes")
    for _cell_id, _cell in enumerate(_chrom_cell_list):
        if verbose:
            print("-- processing cell", _cell_id)
        if 'chrom' not in list(_cell.keys()):
            raise ValueError('no chrom info for cell '+str(_cell_id));
        _chrom_im = np.zeros(np.shape(_cell['chrom']))
        # guassian filter
        if gfilt_size:
            _chrom_im += gaussian_filter(_cell['chrom'], gfilt_size)
        else:
            _chrom_im += _cell['chrom'];
        # zoom in
        if zoom_ratio:
            _chrom_im = ndimage.zoom(_chrom_im, 1/float(zoom_ratio));
        # convolution
        if conv_kernel:
            _kernel = generate_kernel(conv_kernel[0], conv_kernel[-1]);
            _chrom_im = ndimage.convolve(_chrom_im, _kernel);
            _chrom_im[0,:,:] = _chrom_im[1,:,:];
            _chrom_im[-1,:,:] = _chrom_im[-2,:,:];
            _chrom_im[:,0,:] = _chrom_im[:,1,:];
            _chrom_im[:,-1,:] = _chrom_im[:,-2,:];
            _chrom_im[:,:,0] = _chrom_im[:,:,1];
            _chrom_im[:,:,-1] = _chrom_im[:,:,-2];
        # binary
        _binary_chrom_im = (_chrom_im > np.max(_chrom_im) * thres_ratio).astype(np.int);
        # zoom out
        if zoom_ratio:
            _out_ratio = np.array(np.shape(_cell['chrom']),dtype=np.float) / np.array(np.shape(_binary_chrom_im),dtype=np.float)
            _binary_chrom_im = ndimage.zoom(_binary_chrom_im, _out_ratio);
        _binary_chrom_im = ndimage.binary_dilation(_binary_chrom_im, morphology.ball(1))
        # label
        _open_objects = morphology.opening(_binary_chrom_im, morphology.ball(0))
        _close_objects = morphology.closing(_open_objects, morphology.ball(1))
        _label, _num = ndimage.label(_close_objects);
        _label[_label==0] = -1;
        print("-- number of labels:", _num)
        # segmentation
        _seg_label = random_walker(_chrom_im, _label, beta=10000, mode='bf')
        # keep object
        _kept_label = -1 * np.ones(_seg_label.shape, dtype=np.int);
        if keep_obj > 0:
            _sizes = [np.sum(_seg_label==_j+1) for _j in range(np.max(_seg_label))]
            _size_order = np.flipud(np.argsort(_sizes));
            if keep_obj > len(_size_order):
                keep_obj = len(_size_order);
            for _k, _index in enumerate(_size_order[:keep_obj+1]):
                _kept_label[_seg_label == _index+1] = _k + 1;
        else:
            _kept_label = _seg_label

        # save
        _seg_labels.append(_kept_label);
        _chrom_cell_list[_cell_id]['chrom_segmentation'] = _kept_label;
        _chrom_cell_list[_cell_id]['chrom_coord'] = [ndimage.measurements.center_of_mass(_kept_label==_j+1) for _j in range(np.max(_kept_label))];
    if return_im:
        return _chrom_cell_list, _seg_labels
    else:
        return _chrom_cell_list

# fit candidate spots for each chromosome
def candidate_spots_in_chromosome(chrom_cell_list, encoding_type, encoding_scheme=None,
                                  intensity_thres = 500,
                                  im_source='cs_ims', name_source='cs_names',
                                  fit_radius=15, fit_width=[1.35,1.,1.], verbose=True):
    '''find candidate spots with segmented chromosomes in chromosomed-cell_list
    Inputs:
        chrom_cell_list: cell list with chromosome image and coordinate, list of dic
        encoding_type: type of data to be processed, 'combo' or 'unique'
        encoding_scheme: encoding scheme read from file, None for unique, dic for combo
        intensity_thres: intensitiy threshold to keep fitted image, dic when specifying all channels, int or float for general
        im_source: source of images to be fitted, string(key used in each channel)
        name_source: source of names to be fitted, string(key used in each channel)
        fit_radius: radius that has been searched around center of chromosome, float (default:15)
        fit_width: fixed width fitting for single gaussian, array-like, length=3
        verbose: say something during processing, bool (default: True)
    Output:
        _chrom_cell_list: chrom_cell_list appended with candidate points and candidate names
        '''
    # copy
    _chrom_cell_list = chrom_cell_list
    if verbose:
        print("- fit candidate spots given chromosomes");
    # check inputs
    for _cell_id, _cell in enumerate(_chrom_cell_list):
        if 'chrom' not in list(_cell.keys()):
            raise IndexError('no chromosome information in cell '+str(_cell_id))
        elif 'chrom_coord' not in list(_cell.keys()):
            raise IndexError('no chromosome coordinates in cell '+str(_cell_id))
    # check encoding
    if encoding_type.lower() == 'combo' and not encoding_scheme:
        raise ValueError('encoding type is combo, so encoding_scheme is required!')
    if encoding_type.lower() == 'unique':
        im_source, name_source = 'ims', 'u_names'; # default image and name sources
    # loop through cells and
    for _cell_id, _cell in enumerate(_chrom_cell_list):
        # number of chromosome in this cell
        _num_chrom = len(_cell['chrom_coord']);
        if verbose:
            print("-- processing cell: "+str(_cell_id), "with chromosome:", _num_chrom)
        # initialize candidate spots and candidate names
        _chrom_cell_list[_cell_id]['cand_points'] = [];
        _chrom_cell_list[_cell_id]['cand_names'] = [];
        # loop through all chromosomes
        for _chr_pt in _cell['chrom_coord']:
            _cand_pts = [];
            _cand_names = [];
            if encoding_type.lower() == 'combo':
                for _channel, _info in sorted(_cell.items()):
                    if _channel in list(encoding_scheme.keys()):
                        # intensity_threshold
                        if isinstance(intensity_thres, dict):
                            _channel_intensity = intensity_thres[str(_channel)]
                        else:
                            _channel_intensity = float(intensity_thres)
                        # extract info
                        _im_list = _info[im_source];
                        _name_list = _info[name_source];
                        for _i in range(len(_im_list)):
                            _ims = _im_list[_i]
                            _names = _name_list[_i]
                            for _j, (_im, _name) in enumerate(zip(_ims,_names)):
                                _seeds = visual_tools.get_seed_in_distance(_im, _chr_pt, num_seeds=1,seed_radius=fit_radius);
                                if _seeds.shape[1] > 0:
                                    _pt, _success = visual_tools.fitsinglegaussian_fixed_width(_im, _seeds[:,0],
                                                                                          radius=fit_radius,
                                                                                          width_zxy=fit_width)
                                else:
                                    _pt = np.ones(8)*np.inf
                                    _success = False;
                                if _success and _pt[0] > _channel_intensity:
                                    _cand_pts.append(_pt)
                                    _cand_names.append(_name)
                                else:
                                    _cand_pts.append(np.ones(_pt.shape)*np.inf) # failed ones receive zero vector
                                    _cand_names.append(_name)

            elif encoding_type.lower() == 'unique':
                for _channel, _info in sorted(_cell.items()):
                    if isinstance(_info, dict) and im_source in list(_info.keys()) and name_source in list(_info.keys()):
                        # intensity_threshold
                        if isinstance(intensity_thres, dict):
                            _channel_intensity = intensity_thres[str(_channel)]
                        else:
                            _channel_intensity = float(intensity_thres)
                        # extract info
                        _ims = _info[im_source]
                        _names = _info[name_source]
                        for _j, (_im, _name) in enumerate(zip(_ims,_names)):
                            _seeds = visual_tools.get_seed_in_distance(_im, _chr_pt, num_seeds=1,seed_radius=fit_radius, th_seed_percentile=80.);
                            if _seeds.shape[1] > 0:
                                _pt, _success = visual_tools.fit_single_gaussian(_im, _seeds[:,0],
                                                                                      radius=fit_radius,
                                                                                      width_zxy=fit_width)
                            else:
                                _success = False;
                            if _success and _pt[0] > _channel_intensity:
                                _cand_pts.append(_pt)
                                _cand_names.append(int(_name[1:]))
                            else:
                                _cand_pts.append(np.ones(_pt.shape)*np.inf) # failed ones receive zero vector
                                _cand_names.append(int(_name[1:]))

            _chrom_cell_list[_cell_id]['cand_points'].append(_cand_pts);
            _chrom_cell_list[_cell_id]['cand_names'].append(_cand_names);

    return _chrom_cell_list

# generate distance map given candidate spots
def generate_distance_map(cand_cell_list, master_folder, fov_name, encoding_type,
                          cand_point_key='cand_points', cand_name_key='cand_names',
                          zxy_index=[1,2,3], zxy_dimension=[200,150,150],
                          make_plot=False, plot_limits=[200, 1000], save_map=False,
                          map_subfolder = 'Analysis'+os.sep+'distmap',
                          verbose=True):
    '''generate standard distance map from candidate spots attached to each chromosome
    Inputs:
        cand_cell_list: cell list with candidate name and coordinates, list of dic
        master_folder: master folder directory for this dataset, string
        fov_name: name for this field of view, fovs[fov_id], string
        encoding_type: type of dataset, combo / unique
        cand_point_key: key to requrest cand_points, string
        cand_name_key: key to requrest cand_names, string
        zxy_index: index of z,x,y coordinate in given cel_list[i]['cand_points'], list of 3
        zxy_dimension: size in nm of z,x,y pixels, list of 3
        make_plot: whether draw distance map, bool (default: False)
        plot_limit: limits for colormap in distance map plotting, useless if not make_plot, list
        save_map: whether save single cell distance map, bool (default: False)
        _map_subfolder: sub-directory to save single-cell distmap, string (default:Analysis/distmap/*)
        verbose: say something during the process, bool
    Output:
        _cand_cell_list: updated cell list with sorted candidate info and distance map, list of dics
    '''
    from scipy.spatial.distance import pdist,squareform
    import numpy as np
    import matplotlib
    # copy
    _cand_cell_list = cand_cell_list
    if verbose:
        print("- calculate distance map");
    # check inputs
    for _cell_id, _cell in enumerate(_cand_cell_list):
        if cand_point_key not in list(_cell.keys()):
            raise IndexError('no cand_points in cell '+str(_cell_id))
        elif cand_name_key not in list(_cell.keys()):
            raise IndexError('no cand_names in cell '+str(_cell_id))
    if len(zxy_index) != 3 or len(zxy_dimension) != 3:
        raise ValueError('wrong dimension for zxy_index or dimension, length should be 3!');
    _allowed_encoding_type = ['combo', 'unique']
    if str(encoding_type) not in _allowed_encoding_type:
        raise ValueError('wrong given encoding type, should be combo or unique');
    # loop through cells, calculate distance map
    for _cell_id, _cell in enumerate(_cand_cell_list):
        # number of chromosome
        _num_chrom = len(_cell['chrom_coord'])
        # initialize
        _cand_cell_list[_cell_id]['sorted_cand_points'] = []
        _cand_cell_list[_cell_id]['sorted_cand_names'] = []
        _cand_cell_list[_cell_id]['distance_map'] = []
        for _i in range(_num_chrom):
            # extract info
            _cand_pts = _cell[cand_point_key][_i]
            _cand_names = _cell[cand_name_key][_i]
            # sort
            _sorted_cand_pts = [_p for _,_p in sorted(zip(_cand_names, _cand_pts))]
            _sorted_cand_names = [_n for _n,_ in sorted(zip(_cand_names, _cand_pts))]
            # convert pixel to nm in sorted_cand_pts
            _sorted_cand_pts = np.array(_sorted_cand_pts)
            for _index, _dim in zip(zxy_index, zxy_dimension):
                _sorted_cand_pts[:,_index] = _sorted_cand_pts[:,_index] * _dim
            # generate distance map
            _dist_map = squareform(pdist(np.array(_sorted_cand_pts)[:,list(zxy_index)]))
            _dist_map[_dist_map==np.nan] = np.inf # make failed distance to infinitive, so they won't be shown in distance map
            # save
            _cand_cell_list[_cell_id]['sorted_cand_points'].append(_sorted_cand_pts)
            _cand_cell_list[_cell_id]['sorted_cand_names'].append(_sorted_cand_names)
            _cand_cell_list[_cell_id]['distance_map'].append(_dist_map)
            # generate plot
            if make_plot:
                import matplotlib
                import matplotlib.pyplot as plt
                _invmap = matplotlib.colors.LinearSegmentedColormap('inv_seismic', matplotlib.cm.revcmap(matplotlib.cm.seismic._segmentdata))
                plt.figure()
                plt.title("cell: "+str(_cell_id)+', chrom: '+str(_i) )
                plt.imshow(_dist_map, interpolation='nearest',cmap=_invmap, vmax=max(plot_limits), vmin=min(plot_limits))
                plt.colorbar(ticks=range(0,2000,200), label='distance (nm)')
                # save
                if save_map:
                    _map_folder = master_folder + os.sep + map_subfolder + os.sep + fov_name.split('.dax')[0]
                    if not os.path.exists(_map_folder): # if folder not exist, Create
                        os.makedirs(_map_folder)
                    plt.savefig(_map_folder+os.sep+'dist-map_'+str(encoding_type)+'_'+str(_cell_id)+'_'+str(_i)+'.png' , transparent=True)
    return _cand_cell_list

# load distance map
def load_all_dist_maps(master_folder, encoding_type, analysis_dir='Analysis', post_fix='_result_noimage', verbose=True):
    '''Function to load all distance map given master_folder and encoding type
    Inputs:
        master_folder: full directory of the dataset, string
        encoding_type: the encoding type which should be recorded, combo or unique
        analysis_dir: subfolder of analysis files, string (default: Analysis)
        post_fix: all data with this post_fix will be loaded, string (default: _result_noimage)
        verbose: say something!, bool (default: True)
    Outputs
        _merged_map_list: map list that merged for all cells, list of dics '''
    import os
    import pickle as pickle
    # check inputs
    _analysis_folder = master_folder+os.sep+analysis_dir
    if not os.path.exists(_analysis_folder):
        raise IOError('no such analysis folder! '+_analysis_folder)
    if verbose:
        print("- loading", encoding_type, "from data folder:", _analysis_folder)
    _merged_map_list = []
    _file_list = os.listdir(_analysis_folder)
    for _file in _file_list:
        if str(encoding_type) in _file and post_fix in _file and '.pkl' in _file:
            _mp_lst = pickle.load(open(_analysis_folder+os.sep+_file, 'rb'));
            if isinstance(_mp_lst, list):
                if verbose:
                    print("-- loading file:", _file)
                _merged_map_list += _mp_lst
    return _merged_map_list

# screen distance map
def screen_dist_maps(map_list, min_chrom_num=1, max_chrom_num=3,
                     max_loss_per=0.15, max_zeros_per=0.10,
                     required_keys=['sorted_cand_points', 'sorted_cand_names', 'distance_map'],
                     verbose=True):
    '''Function to screen out bad distance maps from map list
    Inputs:
        map_list: list of cells containing all info to draw distance map, list of dics
        min_chrom_num: minimum chromosome number allowed, int (default: 1)
        max_chrom_num: maximum chromosome number allowed, int (default: 3)
        max_loss_per: maximum fitting loss percentage allowed, float (default: 0.15)
        max_zero_per: maximum very small distance allowed, float (default: 0.10)
        required_keys: these keys should exist for further computation, list (default: given by generate_map function)
        verbose: say something about results, bool (default: True)
    Outputs:
        _screen_tags: same structure as the first input, list of list'''
    _screen_tags = []
    _stats = [0,0,0,0];
    if verbose:
        print("- checking maps for list of cells")
    # loop through all cells
    for _cell in map_list:
        # check keys
        _exist_keys = [_k in _cell for _k in required_keys]
        if not np.array(_exist_keys).all():
            _stag = [False for _i in range(len(_cell['chrom']))]
            _screen_tags.append(_stag)
            _stats[0] += 1
            continue

        # check chromosome size
        if len(_cell['distance_map']) > max_chrom_num or len(_cell['distance_map']) < min_chrom_num:
            _stag = [False for _i in range(len(_cell['distance_map']))]
            _screen_tags.append(_stag)
            _stats[1] += 1
            continue

        _stag = [True for _i in range(len(_cell['distance_map']))]
        # check loss percentage
        for _i, _coords in enumerate(_cell['sorted_cand_points']):
            _fcount = 0
            for _coord in _coords:
                if np.inf in _coord:
                    _fcount += 1
            if _fcount / float(len(_coords)) > max_loss_per:
                _stag[_i] = False
                _stats[2] += 1

        # check zeros
        for _i, _map in enumerate(_cell['distance_map']):
            _small_per =np.sum(np.abs(_cell['distance_map'][0]) < 50.) / float(np.prod(np.shape(map_list[0]['distance_map'][0])));
            if _small_per > max_zeros_per:
                _stag[_i] = False
                _stats[3] += 1

        # append
        _screen_tags.append(_stag)
    if verbose:
        print(" screening stats:")
        print("-- cell without required keys", _stats[0])
        print("-- cell with wrong number of chromosomes", _stats[1])
        print("-- chromosome with too many losses", _stats[2])
        print("-- chromosome with too many close points", _stats[3])

    return _screen_tags

# calculate population distance map
def Calculate_population_map(map_list, master_folder, encoding_type,
                             _screen_tags=None, ignore_inf=True, stat_type='median',
                             make_plot=False, plot_limits=[300, 1300], save_map=False,
                             map_subfolder = 'Analysis'+os.sep+'distmap',
                             map_filename='', verbose=True):
    '''Calculate population distance map given map_list
    Inputs:
        map_list: list of cells with distance map, list of dic
        master_folder: full directory of this dataset, string
        encoding_type: type of dataset, combo / unique
        _screen_tags: screening tags generated by screen_dist_maps function, list (default: None)
        ignore_inf: whether compute ignoring inf in stats, bool (default: True)
        stat_type: generate median map or mean map, median / mean (default: median)
        make_plot: whether generate distance map plot, bool (default: False)
        plot_limits: lower and upper limits for distance colorbar, list (default: [300,1300])
        save_map: whether save this map, bool (default: False)
        map_subfolder: sub-directory to save map under master folder, string (default: Analysis/distmap)
        map_filename: filename for this distance map plot, string (default: '')
        verbose: say something!, bool (default: True)
    Output:
        _median_map: 2d matrix
        '''
    # check input
    _allowed_encoding_type = ['combo', 'unique']
    if str(encoding_type) not in _allowed_encoding_type:
        raise ValueError('wrong given encoding type, should be combo or unique');
    # initialize
    _maps = [];
    if verbose:
        print("- extracting all maps")
    # if screening tag applied
    if _screen_tags:
        for _tags, _cell in zip(_screen_tags, map_list):
            for _tag, _map in zip(_tags, _cell['distance_map']):
                if _tag:
                    _maps.append(_map)
    else:
        for _cell in map_list:
            for _map in _cell['distance_map']:
                _maps.append(_map)
    _num_kept = len(_maps)
    if verbose:
        print("-- number of chromosomes kept:", _num_kept)

    _total_map = np.array(_maps, dtype=np.float);
    if ignore_inf:
        _total_map[_total_map == np.inf] = np.nan
        if stat_type == 'median':
            _median_map = np.nanmedian(_total_map, axis=0)
        elif stat_type == 'mean':
            _median_map = np.nanmean(_total_map, axis=0)
        elif isinstance(stat_type) == int or isinstance(stat_type) == float:
            from scipy.stats import scoreatpercentile
            # havent finished yet
            _median_map = None
        else:
            _median_map = None
    else:
        if stat_type == 'median':
            _median_map = np.median(_total_map, axis=0)
        elif stat_type == 'mean':
             _total_map[_total_map == np.inf] = np.nan
             _median_map = np.nanmean(_total_map, axis=0)
        else:
            _median_map = None
    # generate plot if specified
    if make_plot:
        if verbose:
            print('- generating distance map');
        import matplotlib
        import matplotlib.pyplot as plt
        _invmap = matplotlib.colors.LinearSegmentedColormap('inv_seismic', matplotlib.cm.revcmap(matplotlib.cm.seismic._segmentdata))
        plt.figure()
        plt.title(stat_type + ' distance map, number of chromosomes='+str(_num_kept))
        plt.imshow(_median_map, interpolation='nearest',cmap=_invmap, vmax=max(plot_limits), vmin=min(plot_limits))
        plt.colorbar(ticks=range(0,2000,200), label='distance (nm)')
        # save plot
        if save_map:
            _map_folder = master_folder + os.sep + map_subfolder
            if not os.path.exists(_map_folder): # if folder not exist, Create
                os.mkdir(_map_folder);
            if  map_filename == '':
                map_filename += '_'+stat_type +'_'+str(encoding_type)+'_dist_map.png'
            else:
                if '.' in map_filename:
                    map_filename = map_filename.split('.')[0]
                map_filename += '_'+stat_type +'_'+str(encoding_type)+'_dist_map.png'
            plt.savefig(_map_folder+os.sep+map_filename, transparent=True)
    return _median_map ,_num_kept


# save a given list
def Save_Cell_List(cell_list, encoding_type, fov_name, master_folder,
                   save_subfolder='Analysis', postfix='_final_dist',
                   save_as_zip=False, force=False, verbose=True):
    import gzip
    '''Function to save a cell list'''
    _save_folder = master_folder + os.sep + save_subfolder;
    if not os.path.isdir(_save_folder): # if save folder doesnt exist, create
        if verbose:
            print("- create cell list saving folder", _save_folder)
        os.mkdir(_save_folder);
    if save_as_zip:
        _savefile = _save_folder +os.sep+fov_name.replace(".dax", "_"+str(encoding_type)+postfix+".zip")
        if os.path.isfile(_savefile):
            print(_savefile, "- already exist!");
            if not force:
                print("-- stop writting file, exit")
                return False
        else:
            pickle.dump(cell_list, gzip.open(_savefile, 'wb'));
    else:
        _savefile = _save_folder +os.sep+fov_name.replace(".dax", "_"+str(encoding_type)+postfix+".pkl")
        if os.path.isfile(_savefile):
            print(_savefile, "- already exist!");
            if not force:
                print("-- stop writting file, exit")
                return False
        else:
            pickle.dump(cell_list, open(_savefile, 'wb'));
    return True

# save only results
def Save_Result_List(full_cell_list, encoding_type, fov_name, master_folder,
                     save_subfolder='Analysis', postfix='_result_noimage',
                     save_as_zip=False, force=False):
    _kept_keys = ['sorted_cand_points', 'chrom_coord', 'sorted_cand_names','distance_map','chrom', 'chrom_segmentation']
    _result_list = []
    for _cell in full_cell_list:
        _cell_result = {_kk:_cell[_kk] for _kk in _kept_keys};
        _result_list.append(_cell_result);
    # save
    make_save = Save_Cell_List(_result_list, encoding_type, fov_name, master_folder, save_subfolder, postfix, save_as_zip, force)
    return make_save


def generate_normalization(genome_ref_file=r'E:\Users\puzheng\Documents\Libraries\CTP-04\chr21\regions.bed',
                           distance_map_file=r'Z:\20181022-IMR90_whole-chr21-unique\Analysis\distmap\distmaps.npz',
                           verbose=True):
    """Function to generate normalization matrix"""
    from scipy.stats import linregress

    ## generate linear fitting
    # load regions genomic positions
    def load_distance_info(distance_file):
        """Function to load color_usage file"""
        region_list = []
        with open(distance_file) as handle:
            regions = handle.read().splitlines()
            for _reg in regions:
                _chr = _reg.split(':')[0]
                _coords = _reg.split(':')[1]
                _start, _stop = _coords.split('-')

                _reg_dic = {'chromosome': _chr,
                            'start': int(_start),
                            'stop': int(_stop)}
                region_list.append(_reg_dic)
        return region_list
    region_list = load_distance_info(genome_ref_file)
    # load median distance map
    with np.load(distance_map_file) as handle:
        median_dist_map = handle['distance']
    # generate scaling
    dx, dy = np.shape(median_dist_map)
    if dx != dy or dx != len(region_list):
        raise ValueError(
            "Dimension doesn't match for genomic reference and median distance map!")
    physical_dist = []
    genomic_dist = []
    gd_matrix = np.ones([dx, dy])  # genomic distance matrix used for later
    for i in range(dx):
        for j in range(i+1, dy):
            physical_dist.append(median_dist_map[i, j])
            _gd = np.abs((region_list[i]['stop'] + region_list[i]['start']) -
                         (region_list[j]['stop'] + region_list[j]['start']))/2
            genomic_dist.append(_gd)
            gd_matrix[i, j] = _gd
            gd_matrix[j, i] = _gd
    # scaling
    lr = linregress(np.log(genomic_dist), np.log(physical_dist))
    if verbose:
        print(lr)
        print('pearson correlation:', np.sqrt(lr.rvalue))
    # generate normalization matrix
    norm_dist_matrix = np.exp(np.log(gd_matrix) * lr.slope + lr.intercept)

    return norm_dist_matrix

def get_AB_boundaries(im_cor, evec, sz_min=3, plt_val=False, plot_filename=None, verbose=True):
    """Given a correlation matrix im_cor and an eigenvector evec this returns the boundaries (and boudnary scores) for the on/off eigenvector with minimum size sz_min"""
    vec = np.dot(evec, im_cor)
    vec_ = np.array(vec)
    vec_[vec_ == 0] = 10.**(-6)

    def get_bds_sign(vec_s):
        val_prev = vec_s[0]
        bds_ = []
        for pos, val in enumerate(vec_s):
            if val != val_prev:
                bds_.append(pos)
                val_prev = val
        return np.array(bds_)

    vec_s = np.sign(vec_)
    bds_ = get_bds_sign(vec_s)
    vec_ss = vec_s.copy()
    bds_ext = np.concatenate([[0], bds_, [len(vec_s)]])
    for i in range(len(bds_ext)-1):
        if bds_ext[i+1]-bds_ext[i] < sz_min:
            vec_ss[bds_ext[i]:bds_ext[i+1]] = 0

    first_val = vec_ss[vec_ss != 0][0]
    vec_ss_ = []
    for vvec in vec_ss:
        if vvec == 0:
            if len(vec_ss_) > 0:
                vec_ss_.append(vec_ss_[-1])
            else:
                vec_ss_.append(first_val)
        else:
            vec_ss_.append(vvec)
    bds = get_bds_sign(vec_ss_)
    bds_score = []
    bds_ext = np.concatenate([[0], bds, [len(vec)]])
    if verbose:
        print("-- number of boundaries called:",len(bds))
    for i in range(len(bds)):
        lpca = np.median(vec[bds_ext[i]:bds_ext[i+1]])
        rpca = np.median(vec[bds_ext[i+1]:bds_ext[i+2]])
        #print lpca,rpca
        bds_score.append(np.abs(lpca-rpca))
    if plt_val:
        f1 = plt.figure()
        plt.title('A/B pca 1 projection')
        plt.plot(vec, 'ro-')
        if len(bds>0):
            plt.plot(bds, vec[bds], 'go')
        if isinstance(plot_filename, str):
            plot_folder = os.path.dirname(plot_filename)
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            plt.savefig(plot_filename, transparent=True)
        plt.close(f1)
    return bds, bds_score


def pca_components(im_cor):
    """returns the evals, evecs sorted by relevance"""
    from scipy import linalg as la
    data = im_cor.copy()
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    #evecs_red = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return evals, evecs  # , np.dot(evecs_red.T, data.T).T


def naive_pick_spots(cand_spots, region_ids, use_chrom_coord=True, chrom_id=None, 
                     return_indices=False, verbose=True):
    """Naive pick spots simply by intensity"""
    ## check inputs
    if len(cand_spots) != len(region_ids):
        raise ValueError(
            "cand_spots and region_ids should have the same length!")
    if chrom_id is None and use_chrom_coord:
        raise ValueError(
            f"chrom_id should be given if use_chrom_coord is True!")
    elif chrom_id is not None and not isinstance(chrom_id, int):
        chrom_id = int(chrom_id)

    ## For now only support use_chrom_coord mode.
    _selected_spots = []
    _selected_indices = []

    if use_chrom_coord:
        for _i, (_spots, _id) in enumerate(zip(cand_spots, region_ids)):
            # check chrom_id
            if len(_spots) <= chrom_id:
                raise IndexError(
                    f" spots:{_spots} for region:{_id} doesn't have spots for chromosome {chrom_id}")
            # extract points
            _pts = np.array(_spots[chrom_id])
            if len(_pts) == 0:
                _selected_spots.append(np.nan * np.ones(11))
                _selected_indices.append(-1)
            else:
                _selected_spots.append(_pts[np.argsort(_pts[:, 0])[-1]])
                _selected_indices.append(np.argsort(_pts[:, 0])[-1])
        
    ## for not use_chrom_coord
    else:
        for _i, (_spots, _id) in enumerate(zip(cand_spots, region_ids)):
            # extract points
            _pts = np.array(_spots)
            if len(_pts) == 0:
                _bad_pt = np.nan*np.ones(11)
                _bad_pt[0] = 0 # set bad_pt intensity=0
                _selected_spots.append(_bad_pt)
                _selected_indices.append(-1)
            else:
                _selected_spots.append(_pts[np.argsort(_pts[:, 0])[-1]])
                _selected_indices.append(np.argsort(_pts[:, 0])[-1])
    
    # return 
    if return_indices:
        return np.array(_selected_spots), np.array(_selected_indices, dtype=np.int)
    else:
        return np.array(_selected_spots)



def spot_score_in_chromosome(spots, reg_id, sel_spots, _chr_center=None,
                             _cc_dists=None, _lc_dists=None, _intensities=None,
                             distance_zxy=_distance_zxy, local_size=5, 
                             w_ccdist=1, w_lcdist=1, w_int=1, ignore_nan=True):
    """Function to calculate log-score for given spot in selected chr_pts from candidiate_points
    Inputs:
        spots: given fitted spots info, list of spots or one spot
        reg_id: region id for these given spots, int
        sel_spots: currently selected spots for chromosome tracing, list of spots / 2darray
        distance_zxy: transform from pixel to nm for z,x,y axes
        local_size: window size to calculate local distance, int (default: 5)
        w_ccdist: weight for distance to chr-center, float (default: 1)
        w_lcdist: weight for distance to local-center, float (default: 1)
        w_int: weight for intensity, float (default: 1)
    Output:
        _log_score: log score for this given spot, float 
    """
    # get chr coordinates
    _zxy = np.array(sel_spots)[:, 1:4]*np.array(distance_zxy)[np.newaxis, :]
    if _chr_center is None:
        _chr_center = np.nanmean(_zxy, axis=0)
    else:
        _chr_center = _chr_center.copy() * distance_zxy
    # get pt coordinates
    _pts = np.array(spots)
    if len(np.shape(_pts)) == 1:
        _pts = _pts[np.newaxis, :]
    _pt_zxy = _pts[:, 1:4] * np.array(distance_zxy)[np.newaxis, :]
    if isinstance(reg_id, int) or isinstance(reg_id, np.int32) or len(reg_id) == 1:
        _rids = reg_id * np.ones(len(_pts), dtype=np.int)
    elif len(reg_id) == len(_pts):
        _rids = np.array(reg_id, dtype=np.int)
    else:
        raise ValueError(f"Input reg_id should be either a int or list of ints aligned with spots!")# get chr statistics
    # if not given, generate from existing chrom_data
    if _cc_dists is None:
        _cc_dists = np.linalg.norm(_zxy - _chr_center, axis=1)
    if _lc_dists is None:
        _lc_dists = _local_distance(_zxy, _zxy, np.arange(len(_zxy)))
    if _intensities is None:
        _intensities = _pts[:, 0]
    # get pt statistics
    _pt_ct_dist = np.linalg.norm(_pt_zxy - _chr_center, axis=1)
    _pt_lc_dist = _local_distance(_pt_zxy, _zxy, _rids)
    _pt_intensity = _pts[:, 0]
    # get score
    _log_score = np.log(1-_cum_prob(_cc_dists, _pt_ct_dist))*w_ccdist \
        + np.log(1-_cum_prob(_lc_dists, _pt_lc_dist))*w_lcdist \
        + np.log(_cum_prob(_intensities, _pt_intensity))*w_int
    if ignore_nan:
        _nan_flags = np.isnan(_pts).sum(1)        
        _log_score[_nan_flags > 0] = np.nan 
    return _log_score


def distance_score_in_chromosome(dists, sel_spots=None, _nb_dists=None, 
                                 distance_zxy=_distance_zxy, distance_th=400, 
                                 w_dist=1):
    """Function to calculate log-score for given spot in selected chr_pts from candidiate_points
    Inputs:
        spots: given fitted spots info, list of spots or one spot
        sel_spots: currently selected spots for chromosome tracing, list of spots / 2darray
        distance_zxy: transform from pixel to nm for z,x,y axes
        w_dist: weight for distances, float (default: 1)
    Output:
        _log_score: log score for this given spot, float 
    """
    if _nb_dists is None:
        _zxy = np.array(sel_spots)[:, 1:4] * \
            np.array(distance_zxy)[np.newaxis, :]
        _nb_dists = np.linalg.norm(_zxy[1:]-_zxy[:-1], axis=1)
        _nb_dists = _nb_dists[np.isnan(_nb_dists)==False]
    _th_percentile = 1 - _cum_prob(_nb_dists, distance_th)[0]
    _max_percentile = 1 - _cum_prob(_nb_dists, 0)[0]
    _direct_scores = (1-_cum_prob(_nb_dists, dists)) / _th_percentile
    _direct_scores[_direct_scores > _max_percentile] = _max_percentile
    _scores = np.log( _direct_scores ) * w_dist

    return _scores

def generate_distance_score_pool(all_spots, distance_zxy=_distance_zxy):
    """Generate distance score pool from sel_spots"""
    if isinstance(all_spots, np.ndarray):
        _zxy = all_spots[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
    elif isinstance(all_spots[0], np.ndarray) or len(all_spots[0].shape)==1:
        _zxy =  np.stack(all_spots)[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
    elif  isinstance(all_spots[0], list) or len(all_spots[0].shape)==2:
        _spots = np.concatenate([np.array(_pts) for _pts in all_spots], axis=0)
        _zxy =  np.array(_spots)[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
    else:
        raise TypeError("Wrong input datatype for all_spots, should be list of spots or list of list of spots!")
    _nb_dists = np.linalg.norm(_zxy[1:]-_zxy[:-1], axis=1)
    _nb_dists = _nb_dists[np.isnan(_nb_dists) == False]
    return _nb_dists


def _local_distance(spot_zxys, chr_sel_zxy, pt_ids, size=5, minimal_dist=0.5):
    """Function to caluclate local distance"""
    chr_sel_zxy = np.array(chr_sel_zxy)
    _half_size = int((size-1)/2)
    _chr_len = len(chr_sel_zxy)
    _sizes = [min(_half_size, _id, _chr_len-_id-1) for _id in pt_ids]
    _inds = [np.delete(np.arange(_id-_sz,_id+_sz+1),_sz) for _id,_sz in zip(pt_ids, _sizes)]
    _local_dists = []
    for _spot, _ind in zip(spot_zxys,_inds):
        if len(_ind) == 0:
            _local_dists.append(minimal_dist)
        else:
            _local_dists.append(np.linalg.norm(
                np.nanmean(chr_sel_zxy[_ind], axis=0) - _spot))
    return np.array(_local_dists)

# accumulative prob.
def _cum_prob(data, target_value):
    """Function to calculate CDF from a dataset"""
    data = np.array(data, dtype=np.float)
    target_value = np.array(target_value, dtype=np.float)
    if len(target_value.shape) == 0:
        target_value = np.array([target_value], dtype=np.float)
    target_value[np.isnan(target_value)] = np.inf
    target_shape = np.shape(target_value)
    target_value = target_value.reshape(-1)
    cprob = np.array(
        [np.nansum(data[np.isnan(data)==False] < _v) / np.nansum(1-np.isnan(data)) for _v in target_value])
    cprob[cprob == 0] = 1 / np.nansum(1-np.isnan(data))
    cprob[cprob == 1] = 1 - 1 / np.nansum(1-np.isnan(data))
    cprob = cprob.reshape(target_shape)
    return cprob

# generate spot score pool 
def generate_spot_score_pool(all_spots, distance_zxy=_distance_zxy,
                             local_size=5, verbose=False):
    """Generate pool for spot_scores
    Inputs:
        all_spots: list of spots, or np.2drray, or list of list of spots
        distane_zxy: distance in nm for z,x,y pixels, array of 3 (defualt:[200,106,106])
        local_size: window size of local distance calculation, int (default:5)
        verbose: say something!, bool (default:False)
    """
    if isinstance(all_spots, np.ndarray):
        _zxy = all_spots[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
        _intensities = all_spots[:,0]
    elif isinstance(all_spots[0], np.ndarray) or len(all_spots[0].shape)==1:
        _spots = np.concatenate(all_spots)
        _zxy =  _spots[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
        _intensities = _spots[:, 0]
    elif  isinstance(all_spots[0], list) or len(all_spots[0].shape)==2:
        _spots = np.concatenate([np.array(_pts) for _pts in all_spots], axis=0)
        _zxy =  np.array(_spots)[:,1:4] * np.array(distance_zxy)[np.newaxis,:]
        _intensities = np.array(_spots)[:,0]
    else:
        raise TypeError("Wrong input datatype for all_spots, should be list of spots or list of list of spots!")
    _chr_center = np.nanmean(_zxy, axis=0)
    _cc_dists = np.linalg.norm(_zxy - _chr_center, axis=1)
    _lc_dists = _local_distance(_zxy, _zxy, np.arange(len(_zxy)))
    # remove bad points
    _cc_dists = _cc_dists[np.isnan(_cc_dists)==False]
    _lc_dists = _lc_dists[np.isnan(_lc_dists) == False]
    _intensities = _intensities[_intensities > 0]

    return _cc_dists, _lc_dists, _intensities


# Pick spots by dynamic-programming
def dynamic_pick_spots(chrom_cand_spots, unique_ids, cand_spot_scores, nb_dists,
                       w_nbdist=1, distance_zxy=_distance_zxy,
                       return_indices=False, verbose=True):
    """Function to dynamic-programming pick spots
    The idea is to use dynamic progamming to pick spots to get GLOBAL maximum
    for both spot_score (likelihood) and neighboring spot distance (continuity)
    ----------------------------------------------------------------------------
    Inputs:
        chrom_cand_spots: candidate spots for cenrtain chromosome, list of list of spots
        unique_ids: region uid for candidate spots, list/array of ints
        cand_spot_scores: scores for candidate spots corresponding to chrom_cand_spots, list of array of scores
        nb_dists: previous neighboring distance references, could come from different sources, 1darray
        w_nbdist: weight for neighboring distance score, float (default: 1)
        distance_zxy: translate pixel to nm, array of 3 (default: [200,106,106])
        return_indices: whether return indices for picked spots, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        _sel_spots: list of selected spots, list
    optional outputs:
        _sel_indices: list of indices for picked spots, list of ints        
    """
    from scipy.spatial.distance import cdist
    # extract zxy coordiates
    unique_ids = list(np.array(unique_ids, dtype=np.int))
    _zxy_list = [np.array(_spots)[:, 1:4]*np.array(distance_zxy)[np.newaxis, :]
                 for _spots in chrom_cand_spots if len(_spots) > 0]
    _ids = [_id for _id, _spots in zip(
        unique_ids, chrom_cand_spots) if len(_spots) > 0]
    # initialize dynamic score and pointers
    _dy_scores = [_scores for _scores, _spots in zip(
        cand_spot_scores, chrom_cand_spots) if len(_spots) > 0]
    _dy_pointers = [-np.ones(len(_spots), dtype=np.int)
                    for _spots in chrom_cand_spots if len(_spots) > 0]

    # if there are any spots:
    if len(_dy_scores) > 0:
        # forward
        for _i, (_zxys, _id) in enumerate(zip(_zxy_list[1:], _ids[1:])):
            # notice: i is actually 1 smaller than real indices
            # calculate min_distance and give score
            # real pair-wise distances
            _dists = cdist(_zxy_list[_i], _zxy_list[_i+1])
            # add distance score, which is normalized by how far exactly these two regions are
            _measure = distance_score_in_chromosome(
                _dists, _nb_dists=nb_dists, w_dist=w_nbdist) / (_ids[_i+1] - _ids[_i])
            _measure += _dy_scores[_i][:,np.newaxis]  # get previous maximum

            # update maximum values and pointers
            _dy_scores[_i+1] += np.max(_measure, axis=0)  # update maximum
            _dy_pointers[_i+1] = np.argmax(_measure, axis=0)  # update pointer

        # backward
        _dy_indices = [np.argmax(_dy_scores[-1])]
        _dy_spots = [
            chrom_cand_spots[unique_ids.index(_ids[-1])][_dy_indices[-1]]]
        for _id, _pointers in zip(_ids[:-1][::-1], _dy_pointers[1:][::-1]):
            _dy_indices.append(_pointers[_dy_indices[-1]])
            _dy_spots.append(
                chrom_cand_spots[unique_ids.index(_id)][_dy_indices[-1]])
        # inverse _sel_indices and _sel_spots
        _dy_indices.reverse()
        _dy_spots.reverse()

    _sel_spots, _sel_indices = [], []
    for _uid in unique_ids:
        if _uid in _ids:
            _sel_spots.append(_dy_spots[_ids.index(_uid)])
            _sel_indices.append(_dy_indices[_ids.index(_uid)])
        else:
            if len(_dy_spots) > 0:
                _bad_pt = np.nan*np.ones(len(_dy_spots[-1]))
                _bad_pt[0] = 0 # set bad_pt intensity=0
            else:
                _bad_pt = np.nan*np.ones(11)
                _bad_pt[0] = 0 # set bad_pt intensity=0
            _sel_spots.append(_bad_pt)
            _sel_indices.append(-1)
    if return_indices:
        return np.array(_sel_spots), np.array(_sel_indices, dtype=np.int)
    else:
        return np.array(_sel_spots)


# Pick spots by EM algorithm
def EM_pick_spots(chrom_cand_spots, unique_ids, _chrom_coord=None,
                  num_iters=np.inf, terminate_th=0.002, intensity_th=1,
                  distance_zxy=_distance_zxy, local_size=5, spot_num_th=200,
                  w_ccdist=1, w_lcdist=0.1, w_int=1, w_nbdist=3,
                  check_spots=True, check_th=-3., check_percentile=1., 
                  ignore_nan=True, make_plot=False, 
                  save_plot=False, save_path=None, save_filename='',
                  return_indices=False, return_scores=False, 
                  return_other_scores=False, verbose=True):
    """Function to achieve EM spot picking
    -------------------------------------------------------------------------------------
    E-step: calculate spot score based on:
        distance to chromosome center (consistency): w_ctdist
        distance to local center (smoothing): w_lcdist
        intensity (spot confidence): w_int
    M-step: pick spots from candidate spots to maximize spot score + neighboring distances
        distance to neighbor (continuity): w_nbdist
    Iterate till:
        a. iteration exceed num_iters
        b. picked point change percentage lower than terminate_th
        c. current result is stable and oscilliating around miminum
    -------------------------------------------------------------------------------------
    Inputs:
        chrom_cand_spots: candidate spots for cenrtain chromosome, list of list of spots
        unique_ids: region uid for candidate spots, list/array of ints
        _chrom_coord: specify 3d chromosome coordinate (in pixel) for reference,
            * otherwise chrom_center will be generated by sel_spots
        num_iters: maximum allowed number of iterations, int (default: np.inf, i.e. no limit)
        terminate_th: termination threshold for change percentage of spot-picking, float (default: 0.005)
        intensity_th: threshold for intensity that keep to try EM, float (default: 1)
            * threshold=1 means SNR=1, which is a pretty generous threshold
        distance_zxy: translate pixel to nm, array of 3 (default: [200,106,106])
        local_size: size to calculate local distance, int (default: 5)
        spot_num_th: minimum number of spots needed for calculate_spot_score, int (default:200)
        w_ccdist: weight for distance_to_chromosome_center, float (default: 1)
        w_lcdist: weight for distance_to_local_center, float (default: 1)
        w_int: weight for spot intensity, float (default: 2)
        w_nbdist:  weight for distance_to_neighbor_region, float (default: 1)
        check_spots: whether apply stringency check for selected spots, bool (default: True)
        check_th: the relative threshold for stringency check, 
            * which will multiply the sum of all weights to estimate threshold, bool (default: -3)
        check_percentile: another percentile threshold that may apply to data, float (default: 1.)
        make_plot: make plot for each iteration, bool (default: False)
        return_indices: whether return indices for picked spots, bool (default: False)
        return_scores: whether return scores for picked spots, bool (default: False)
        return_other_scores: whether return Other scores for cand_spots, bool (default: False)
        verbose: say something!, bool (default: True)
    Outputs:
        _sel_spots: list of selected spots, list
    optional outputs:
        _sel_indices: list of indices for picked spots, list of ints
    """
    ## check inputs
    # check candidate spot and unique id length
    if len(chrom_cand_spots) != len(unique_ids):
        raise ValueError(f"Length of chrom_cand_spots should match unique_ids, \
            while {len(chrom_cand_spots)} and {len(unique_ids)} received!")
    unique_ids = np.array(unique_ids, dtype=np.int)
    # check termination flags
    if num_iters == np.inf and terminate_th < 0:
        raise ValueError(f"At least one valid termination flag required!")
    # check other inputs
    local_size = int(local_size)

    ## initialize
    if verbose:
        print(f"- EM picking spots for {len(unique_ids)} regions.")
    # filter spots
    if verbose:
        print(f"-- filtering spots by intensity threshold = {intensity_th}.")
    for _i, _spots in enumerate(chrom_cand_spots):
        chrom_cand_spots[_i] = _spots[np.array(_spots)[:, 0] > intensity_th]
    # select spots by naive
    if verbose:
        print(f"-- initialize EM by naively picking spots!")
        
    _sel_spots, _sel_indices = naive_pick_spots(chrom_cand_spots, unique_ids,
                                                use_chrom_coord=False, return_indices=True)
    # make plot for initialized
    if make_plot:
        from scipy.spatial.distance import pdist, squareform
        _distmap_list = []
        _distmap = squareform(pdist(_sel_spots[:,1:4] * distance_zxy[np.newaxis,:]))
        _distmap[_distmap == np.inf] = np.nan
        _distmap_list.append(_distmap)

    # initialize flags to finish EM
    _iter = 0  # a counter for iteration
    _change_ratio = 1  # keep record of how much picked-points are changed
    _previous_ratios = []
    ## get into EM loops if
    # not exceeding num_iters and
    # picked point change percentage lower than terminate_th
    while(_iter < num_iters and _change_ratio >= terminate_th):
        if verbose:
            print(f"-- EM iter:{_iter}")

        ## E-step
        # distributions for spot-score
        _estart = time.time()
        if len(_sel_spots) < spot_num_th:
            _cc_dists, _lc_dists, _intensities = generate_spot_score_pool(chrom_cand_spots, distance_zxy=distance_zxy,
                                                                          local_size=local_size, verbose=verbose)
        else:
            _cc_dists, _lc_dists, _intensities = generate_spot_score_pool(_sel_spots, distance_zxy=distance_zxy,
                                                                          local_size=local_size, verbose=verbose)
        # distribution for neighbor distance
        _nb_dists = generate_distance_score_pool(_sel_spots)
        if verbose:
            print(f"--- E time: {np.round(time.time()-_estart, 4)} s,")

        ## M-step
        _mstart = time.time()
        # calcualte spot score
        _spot_scores = [spot_score_in_chromosome(_spots, _uid-1, _sel_spots, _chrom_coord, 
                        _cc_dists=_cc_dists, _lc_dists=_lc_dists, _intensities=_intensities, 
                        distance_zxy=distance_zxy, local_size=local_size, 
                        w_ccdist=w_ccdist, w_lcdist=w_lcdist, w_int=w_int, 
                        ignore_nan=ignore_nan) for _spots, _uid in zip(chrom_cand_spots, unique_ids)]
        # special modification for 
        _spot_scores[-1] += spot_score_in_chromosome(chrom_cand_spots[-1], len(unique_ids)-1, _sel_spots,
                                                     _chrom_coord,
                                                     _cc_dists=_cc_dists, _lc_dists=_lc_dists,
                                                     _intensities=_intensities, distance_zxy=distance_zxy,
                                                     local_size=local_size, w_ccdist=2, w_lcdist=0,
                                                     w_int=0, ignore_nan=ignore_nan)
        # pick spots by dynamic programming
        _sel_spots, _new_indices = dynamic_pick_spots(chrom_cand_spots, unique_ids, _spot_scores, _nb_dists,
                                                      w_nbdist=w_nbdist, distance_zxy=distance_zxy, return_indices=True, verbose=verbose)
        if verbose:
            print(f"--- M time: {np.round(time.time()-_mstart, 4)} s.")
        # make plot for initialized
        if make_plot:
            _distmap = squareform(pdist(_sel_spots[:,1:4] * distance_zxy[np.newaxis,:] ) )
            _distmap[_distmap == np.inf] = np.nan
            _distmap_list.append(_distmap)
        # update exit checking flags
        _iter += 1
        _change_ratio = sum(np.array(_new_indices, dtype=np.int) -
                            np.array(_sel_indices, dtype=np.int) != 0) / len(_sel_indices)
        _previous_ratios.append(_change_ratio)
        if verbose:
            print(f"--- change_ratio: {_change_ratio}")
        # update sel_indices
        _sel_indices = _new_indices

        # special exit for long term oscillation around minimum
        if len(_previous_ratios) > 5 and np.mean(_previous_ratios[-5:]) <= 2 * terminate_th:
            if verbose:
                print("- exit loop because of long oscillation around minimum.")
            break
    ## check spots if specified
    if check_spots:
        from scipy.stats import scoreatpercentile
        # weight for intensity now +1
        _sel_scores = spot_score_in_chromosome(_sel_spots, 
                            np.array(unique_ids, dtype=np.int)-min(unique_ids), 
                            _sel_spots, _chrom_coord, _cc_dists=_cc_dists, 
                            _lc_dists=_lc_dists, _intensities=_intensities,
                            distance_zxy=distance_zxy, local_size=local_size, 
                            w_ccdist=w_ccdist, w_lcdist=w_lcdist,
            w_int=w_int+1, ignore_nan=ignore_nan)
        _other_scores = []
        for _scs, _sel_i in zip(_spot_scores, _sel_indices):
            _other_cs = list(_scs)
            if len(_other_cs) > 0:
                _other_cs.pop(_sel_i)
                _other_scores += list(_other_cs)

        _th_sel = scoreatpercentile(_sel_scores, check_percentile)
        _th_other = scoreatpercentile(_other_scores, 100-check_percentile)
        _th_weight = check_th * (w_ccdist + w_lcdist + w_int + 1)
        if check_percentile > 0 and check_percentile < 100:
            _final_check_th = max(_th_sel, _th_other, _th_weight)
        else:
            _final_check_th = _th_weight
        if verbose:
            print(f"-- applying stringency cehck for spots, theshold={_final_check_th}")
        # remove bad spots
        if np.sum(_sel_scores < _final_check_th) > 0:
            _inds = np.where(_sel_scores < _final_check_th)[0]
            for _i in _inds:
                _sel_spots[_i] = np.nan
                _sel_spots[_i,0] = 0.
            if verbose:
                print(f"--- {len(_inds)} spots didn't pass stringent quality check.")
    ## make plot
    if make_plot:
        _num_im = len(_distmap_list)
        _plot_limits = [0,2000]
        _font_size = 14
        _dpi = 300
        _single_im_size = 5
        _fig,_axes = plt.subplots(1, _num_im, figsize=(_single_im_size*_num_im, _single_im_size*1.2), dpi=_dpi)
        _fig.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0.08, hspace=0)
        for _i, (ax,_distmap) in enumerate(zip(_axes.ravel(), _distmap_list)):
            # plot
            im = ax.imshow(_distmap, interpolation='nearest',  cmap='seismic_r', 
                           vmin=min(_plot_limits), vmax=max(_plot_limits))
            ax.tick_params(left=False, labelsize=_font_size, length=2)
            ax.yaxis.set_ticklabels([])
            # title
            if _i == 0:
                ax.set_title('Initialized by naive', fontsize=_font_size+2)
            else:
                ax.set_title(f"EM iter:{_i-1}", fontsize=_font_size+2)
            # add colorbar
            cb = plt.colorbar(im, ax=ax, ticks=np.arange(0,2200,200), shrink=0.6)
            cb.ax.tick_params(labelsize=_font_size, width=0.6, length=1)
        # save filename
        if save_plot and save_path is not None:
            if not os.path.exists(save_path):
                if verbose:
                    print(f"-- create folder for image: {save_path}")
                os.makedirs(save_path)
            if save_filename == '':
                save_filename = 'EM_iterations.png'
            else:
                save_filename = 'EM_iterations_'+save_filename
                if '.png' not in save_filename:
                    save_filename += '.png'
            _plot_filename = os.path.join(save_path, save_filename)
            if verbose:
                print(f"-- saving image to file: {_plot_filename}")
            _fig.savefig(_plot_filename, transparent=True)
        elif save_plot:
            print("Save path for plot is not given, skip!")
        # plot show if only in main stream
        if __name__ == '__main__':
            plt.show()

    # Return!
    # case 1: simple return selected spots
    if not return_indices and not return_scores and not return_other_scores:
        return np.array(_sel_spots)
    # return spots combined with other info
    else:
        _return_args = (np.array(_sel_spots),)
        if return_indices:
            _return_args += (np.array(_sel_indices, dtype=np.int),)
        if return_scores:
            # if not check_spots, generate a new score set
            _cc_dists, _lc_dists, _intensities = generate_spot_score_pool(_sel_spots, distance_zxy=distance_zxy,
                                                                        local_size=local_size, verbose=verbose)
            _sel_scores = spot_score_in_chromosome(_sel_spots, 
                                np.array(unique_ids, dtype=np.int)-min(unique_ids), 
                                _sel_spots, _chrom_coord, _cc_dists=_cc_dists, 
                                _lc_dists=_lc_dists, _intensities=_intensities,
                                distance_zxy=distance_zxy, local_size=local_size, 
                                w_ccdist=w_ccdist, w_lcdist=w_lcdist,
                                w_int=w_int+1)
            _sel_scores = np.array(_sel_scores)
            if ignore_nan:
                _sel_scores = _sel_scores[np.isnan(_sel_scores) == False]
            _return_args += (_sel_scores,)

        if return_other_scores:
            _other_scores = []
            for _scs, _sel_i in zip(_spot_scores, _sel_indices):
                _other_cs = list(_scs)
                if len(_other_cs) > 0:
                    _other_cs.pop(_sel_i)
                    _other_scores += list(_other_cs)
            _other_scores = np.array(_other_scores)
            if ignore_nan:
                _other_scores = _other_scores[np.isnan(_other_scores)==False]
            _return_args += (_other_scores,)
        # return!
        return _return_args
