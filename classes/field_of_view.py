import sys
import glob
import os
import time
import copy
import numpy as np
import pickle as pickle
# fix mp reducer 4GB limit
from ..required_files import pickle2reducer
import multiprocessing as mp
ctx = mp.get_context()
ctx.reducer = pickle2reducer.Pickle2Reducer()
#from ..required_files import pickle4reducer
#import multiprocessing as mp
#ctx = mp.get_context()
#ctx.reducer = pickle4reducer.Pickle4Reducer()

# saving
import h5py
import ast
# plotting
import matplotlib
import matplotlib.pyplot as plt

# import other sub-packages
# import package parameters
from .. import _correction_folder, _corr_channels, _temp_folder,_distance_zxy,\
    _sigma_zxy,_image_size, _allowed_colors, _num_buffer_frames, _num_empty_frames, _image_dtype
from . import _allowed_kwds, _max_num_seeds, _min_num_seeds, _spot_seeding_th

def __init__():
    print(f"Loading field of view class")
    pass

class Field_of_View():
    """Class of field-of-view of a certain sample, which includes all possible files across hybs and parameters
    Key features:
        1. load and save images
        2. pre-process images by warpping or generating spot-translating functions
        3. perform gaussian fitting, convert images into spot coordinates
        4. load and process chromosome image, DAPI image, reference(beads) image

    """
    def __init__(self, parameters, 
                 _fov_id=None, _fov_name=None, 
                 _parallel=True,
                 _color_info_kwargs={},
                 _save_filename=None, 
                 _savefile_kwargs={},
                 _segmentation_kwargs={},
                 _load_fov_info=True, 
                 _load_correction=True,
                 _load_segmentation=True, 
                 _prioritize_saved_attrs=True,
                 _force_initialize_savefile=False, 
                 _save_info_to_file=True, 
                 _debug=False, 
                 _verbose=True,
                 ):
        ## Initialize key attributes:
        self.verbose = _verbose
        self.debug = _debug
        #: attributes for unprocessed images:

        # correction profiles 
        self.correction_profiles = {'bleed':None,
                                    'chromatic':{},
                                    'illumination':{},}
        # parallel
        self.parallel = bool(_parallel)
        # drifts
        self.drift = {}
        # rotations
        self.rotation = {}
        # segmentation
        if 'segmentation_dim' not in _segmentation_kwargs:
            self.segmentation_dim = 2 # default is 2d segmentation
        else:
            self.segmentation_dim = int(_segmentation_kwargs['segmentation_dim'])

        #: attributes for processed images:
        # splitted processed images
        self.im_dict = {}
        # channel dict corresponding to im_dict
        self.channel_dict = {}

        ## check input datatype
        if not isinstance(parameters, dict):
            raise TypeError(f'wrong input type of parameters, should be dict containing essential info, but {type(parameters)} is given!')

        ## required parameters: 
        # data_folder: str of folder or list of str of folders
        if 'data_folder' not in parameters:
            raise KeyError(f"data_folder is required in parameters.")
        if isinstance(parameters['data_folder'], list):
            self.data_folder = [str(_fd) for _fd in parameters['data_folder']]
        else:
            self.data_folder = [str(parameters['data_folder'])]
        ## extract hybe folders and field-of-view names
        self.folders = []
        _all_fov_names = []
        for _fd in self.data_folder:
            from ..io_tools.data import get_folders
            try:
                _hyb_fds, _fovs = get_folders(_fd, feature='H', verbose=True)
                self.folders += _hyb_fds # here only extract folders not fovs
                _all_fov_names.append(_fovs)
            except:
                pass
        # select longest
        _fovs = _all_fov_names[np.argmax([len(_fs) for _fs in _all_fov_names])]
        
        if _fov_name is None and _fov_id is None:
            raise ValueError(f"either _fov_name or _fov_id should be given!")
        elif _fov_id is not None:
            _fov_id = int(_fov_id)
            # define fov_name
            _fov_name = _fovs[_fov_id]
        else:
            _fov_name = str(_fov_name)
            if _fov_name not in _fovs:
                raise ValueError(f"_fov_name:{_fov_name} should be within fovs:{_fovs}")
            _fov_id = _fovs.index(_fov_name)
        # append fov information 
        self.fov_id = _fov_id
        self.fov_name = _fov_name

        # find analysis folder
        if 'analysis_folder'  in parameters:
            self.analysis_folder = str(parameters['analysis_folder'])
        else:
            self.analysis_folder = os.path.join(self.data_folder[0], 'Analysis')
        
        ## load experimental info
        if '_color_filename' not in _color_info_kwargs:
            self.color_filename = 'Color_Usage'
            _color_info_kwargs['_color_filename'] = self.color_filename
        else:
            self.color_filename = _color_info_kwargs['_color_filename']
        if '_color_format' not in _color_info_kwargs:
            self.color_format = 'csv'
            _color_info_kwargs['_color_format'] = self.color_format
        else:
            self.color_format = _color_info_kwargs['_color_format']
        _color_dic = self._load_color_info(_annotate_folders=True, **_color_info_kwargs)

        # experiment_folder
        if 'experiment_folder'  in parameters:
            self.experiment_folder = parameters['experiment_folder']
        else:
            self.experiment_folder = os.path.join(self.data_folder[0], 'Experiment')
        ## analysis_folder, segmentation_folder, save_folder, correction_folder,map_folder

        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder']
        else:
            self.segmentation_folder = os.path.join(self.analysis_folder, 'segmentation')
        # save folder
        if 'save_folder' in parameters:
            self.save_folder = parameters['save_folder']
        else:
            self.save_folder = os.path.join(self.analysis_folder,'save')
        if not os.path.exists(self.save_folder):# create save folder if not exist
            if _verbose:
                print(f"+ creating save folder: {self.save_folder}")
            os.makedirs(self.save_folder)

        if 'correction_folder' in parameters:
            self.correction_folder = parameters['correction_folder']
        else:
            self.correction_folder = _correction_folder
        if 'drift_folder' in parameters:
            self.drift_folder = parameters['drift_folder']
        else:
            self.drift_folder =  os.path.join(self.analysis_folder, 'drift')
        if not os.path.exists(self.drift_folder):# create save folder if not exist
            if _verbose:
                print(f"+ creating drift folder: {self.drift_folder}")
            os.makedirs(self.drift_folder)

        if 'map_folder' in parameters:
            self.map_folder = parameters['map_folder']
        else:
            self.map_folder = os.path.join(self.analysis_folder, 'distmap')
        # number of num_threads
        if 'num_threads' in parameters:
            self.num_threads = parameters['num_threads']
        else:
            self.num_threads = int(os.cpu_count() / 4) # default: use one third of cpus.
        # ref_id
        if 'ref_id' in parameters:
            self.ref_id = int(parameters['ref_id'])
        else:
            self.ref_id = 0
        # data type
        if 'image_dtype' in parameters:
            self.image_dtype = parameters['image_dtype']
        else:
            self.image_dtype = _image_dtype
        ## shared_parameters
        # initialize
        if 'shared_parameters' in parameters:
            self.shared_parameters = parameters['shared_parameters']
        else:
            self.shared_parameters = {}
        # add parameter keys:      
        if 'distance_zxy' not in self.shared_parameters:    
            self.shared_parameters['distance_zxy'] = _distance_zxy
        if 'sigma_zxy' not in self.shared_parameters:
            self.shared_parameters['sigma_zxy'] = _sigma_zxy
        if 'single_im_size' not in self.shared_parameters:
            self.shared_parameters['single_im_size'] = _image_size
        if 'num_buffer_frames' not in self.shared_parameters:
            self.shared_parameters['num_buffer_frames'] = _num_buffer_frames
        if 'num_empty_frames' not in self.shared_parameters:
            self.shared_parameters['num_empty_frames'] = _num_empty_frames
        if 'normalization' not in self.shared_parameters:
            self.shared_parameters['normalization'] = False
        if 'corr_channels' not in self.shared_parameters:
            self.shared_parameters['corr_channels'] = _corr_channels
        # ref channel:
        if 'ref_channel' not in self.shared_parameters:
            if len(self.channels) >= 2:
                self.shared_parameters['ref_channel'] = self.channels[1]
            else:
                self.shared_parameters['ref_channel'] = self.channels[0]
        
        
        # adjust corr_channels
        _kept_corr_channels = []
        for _ch in self.shared_parameters['corr_channels']:
            if str(_ch) in self.channels:
                _kept_corr_channels.append(str(_ch))
        self.shared_parameters['corr_channels'] = _kept_corr_channels
        
        # parameter for corrections
        if 'corr_bleed' not in self.shared_parameters:
            self.shared_parameters['corr_bleed'] = True
        if 'corr_Z_shift' not in self.shared_parameters:
            self.shared_parameters['corr_Z_shift'] = True
        if 'corr_hot_pixel' not in self.shared_parameters:
            self.shared_parameters['corr_hot_pixel'] = True
        if 'corr_illumination' not in self.shared_parameters:
            self.shared_parameters['corr_illumination'] = True
        if 'corr_chromatic' not in self.shared_parameters:
            self.shared_parameters['corr_chromatic'] = True
        if 'corr_gaussian_highpass' not in self.shared_parameters:
            self.shared_parameters['corr_gaussian_highpass'] = False
        if 'allowed_kwds' not in self.shared_parameters:
            self.shared_parameters['allowed_data_types'] = _allowed_kwds
        # params for drift
        if 'max_num_seeds' not in self.shared_parameters:
            self.shared_parameters['max_num_seeds'] = _max_num_seeds
        if 'min_num_seeds' not in self.shared_parameters:
            self.shared_parameters['min_num_seeds'] = _min_num_seeds
        if 'drift_use_autocorr' not in self.shared_parameters:
            self.shared_parameters['drift_use_autocorr'] = True 
        if 'drift_sequential' not in self.shared_parameters:
            self.shared_parameters['drift_sequential'] = False 
        if 'good_drift_th' not in self.shared_parameters:
            self.shared_parameters['good_drift_th'] = 1. 
        if 'drift_args' not in self.shared_parameters:
            self.shared_parameters['drift_args'] = {
                'precision_fold': 100, # 0.01 pixel sampling
                'min_good_drifts': 3, # at least 3 drifts calculated
                'drift_diff_th': 1., # difference between drifts within 1 pixel
            } # use defaults for drift
            
        # param for spot_finding
        if 'spot_seeding_th' not in self.shared_parameters:
            self.shared_parameters['spot_seeding_th'] = _spot_seeding_th
        if 'normalize_intensity_background' not in self.shared_parameters:
            self.shared_parameters['normalize_intensity_background'] = False
        if 'normalize_intensity_local' not in self.shared_parameters:
            self.shared_parameters['normalize_intensity_local'] = True

        # parameter for saving
        if 'empty_value' not in self.shared_parameters:
            self.shared_parameters['empty_value'] = 0

        ## Drift
        # update ref_filename
        self.ref_filename = os.path.join(self.annotated_folders[self.ref_id], self.fov_name)
        # update drift filename
        _dft_fl_postfix = '_current_cor.pkl'
        if self.shared_parameters['drift_sequential']:
            _dft_fl_postfix = '_sequential'+_dft_fl_postfix
        self.drift_filename = os.path.join(self.drift_folder,
                                            self.fov_name.replace('.dax', _dft_fl_postfix))

        ## Create savefile
        # save filename
        if _save_filename is None:
            _save_filename = os.path.join(self.save_folder, self.fov_name.replace('.dax', '.hdf5'))
        # set save_filename attr
        self.save_filename = _save_filename
    
        # if savefile already exists, load attributes from it        
        if os.path.isfile(self.save_filename) and not _force_initialize_savefile:
            _new_savefile = False 
            # load attributes
            if _load_fov_info:
                self._load_from_file('fov_info', 
                                    _overwrite=_prioritize_saved_attrs,
                                    _verbose=_verbose,)
            if _load_correction:
                self._load_from_file('correction', 
                                    _overwrite=_prioritize_saved_attrs,
                                    _verbose=_verbose,)
            if _load_segmentation:
                self._load_from_file('segmentation', 
                                    _overwrite=_prioritize_saved_attrs,
                                    _verbose=_verbose,)                                            
        # otherwise, initialize save file if not exist
        else:
            _new_savefile = True 

            if _load_correction:
                self._load_correction_profiles(
                    _load_from_savefile_first=False,
                    _verbose=_verbose)
            if _load_segmentation:
                ### REQUIRES DEVELOPMENT###
                pass 
            # create savefile
            self._init_save_file(_save_filename=_save_filename, 
                                 **_savefile_kwargs)                        
        # if decided to overwrite savefile info with what this initiation gives, do it:
        if _save_info_to_file and not _new_savefile and not _prioritize_saved_attrs:
            self._save_to_file('fov_info', _overwrite=(not _prioritize_saved_attrs), _verbose=_verbose)


    ## Load basic info
    def _load_color_info(self, _color_filename=None, _color_format=None, 
                         _save_color_dic=True, _annotate_folders=False):
        """Function to load color usage representing experimental info"""
        ## check inputs
        if _color_filename is None:
            _color_filename = self.color_filename
        if _color_format is None:
            _color_format = self.color_format
        from ..get_img_info import Load_Color_Usage, find_bead_channel, find_dapi_channel
        _color_dic, _use_dapi, _channels = Load_Color_Usage(self.analysis_folder,
                                                            color_filename=_color_filename,
                                                            color_format=_color_format,
                                                            return_color=True)
        
        # need-based store color_dic
        if _save_color_dic:
            self.color_dic = _color_dic
        # store other info
        self.use_dapi = _use_dapi
        self.channels = [str(ch) for ch in _channels]
        # channel for beads
        _drift_channel = find_bead_channel(_color_dic)
        self.drift_channel = self.channels[_drift_channel]
        _dapi_channel = find_dapi_channel(_color_dic)
        self.dapi_channel = self.channels[_dapi_channel]
        self.dapi_channel_index = _dapi_channel

        # get annotated folders by color usage
        if _annotate_folders:
            self.annotated_folders = []
            for _hyb_fd, _info in self.color_dic.items():
                # only select the ones with existing image
                _matches = [_fd for _fd in self.folders if _hyb_fd == _fd.split(os.sep)[-1] and os.path.exists(os.path.join(_fd, self.fov_name)) ]
                if len(_matches)==1:
                    self.annotated_folders.append(_matches[0])
            print(f"- {len(self.annotated_folders)} folders are found according to color-usage annotation.")

        return _color_dic
    
    ### Here are some initialization functions
    def _init_save_file(self, _save_filename=None, 
                        _overwrite=False, _verbose=True):
        """Function to initialize save file for FOV object
        Inputs:
            _save_filename: full path for filename saving this dataset.
            _overwrite: whether overwrite existing info within save_file, bool (default: False)
            _verbose: say something!, bool (default: True)
        Outputs:
            save_file created, current info saved.
        """
        if _save_filename is None:
            _save_filename = getattr(self, 'save_filename')
        # set save_filename attr
        setattr(self, 'save_filename', _save_filename)
        if _verbose: 
            if not os.path.exists(_save_filename):
                print(f"- Creating save file for fov:{self.fov_name}: {_save_filename}.")
            else:
                print(f"- Initialize save file for fov:{self.fov_name}: {_save_filename}.")

        ## initialize fov_info, segmentation and correction
        for _type in ['fov_info', 'segmentation', 'correction']:
            self._save_to_file(_type, _overwrite=_overwrite, _verbose=_verbose)
        
        ## initialize image data types
        from .batch_functions import _color_dic_stat
        # acquire valid types
        _type_dic = _color_dic_stat(self.color_dic, 
                                    self.channels, 
                                    self.shared_parameters['allowed_data_types'],
                                   )
        # create
        for _type, _dict in _type_dic.items():
            print(f'save type: {_type}"')
            self._save_to_file(_type, _overwrite=_overwrite, _verbose=_verbose)
        
        return

    def _DAPI_segmentation(self):
        pass

    def _load_correction_profiles(self, _correction_folder=None, 
                                  _load_from_savefile_first=True,
                                  _corr_channels=['750','647','561'],
                                  _chromatic_target='647',
                                  _profile_postfix='.npy', 
                                  _empty_value=0, 
                                  _overwrite=False, 
                                  _verbose=True):
        """Function to load correction profiles in RAM"""
        from ..io_tools.load import load_correction_profile
        # determine correction folder
        if _correction_folder is None:
            _correction_folder = self.correction_folder
        # ref channel:
        _ref_channel = self.shared_parameters.get('ref_channel', _chromatic_target)
        print(f"Reference channel: {_ref_channel}")
        # loading bleedthrough
        if self.shared_parameters['corr_bleed']:
            if 'bleed' in self.correction_profiles and self.correction_profiles['bleed'] is not None and not _overwrite:
                print(f"++ bleed correction profile already exist, skip.")
            else:
                _savefile_load_bleed = False
                if _load_from_savefile_first:
                    # try to load from save_file
                    with h5py.File(self.save_filename, "a", libver='latest') as _f:
                        if 'corrections' in _f.keys():
                            _grp = _f['corrections']
                            _bleed_key = '_'.join(self.shared_parameters['corr_channels'])+'_bleed'
                            if _bleed_key in _grp:
                                _pf = _grp[_bleed_key][:]
                                if (np.array(_pf) != _empty_value).any():
                                    if _verbose:
                                        print(f"++ load bleed correction profile directly from savefile.")
                                    _savefile_load_bleed = True
                                    self.correction_profiles['bleed'] = _pf
                # if not successfully loaded or not savefile first, load from original file
                if not _savefile_load_bleed:
                    if _verbose:
                        print(f"++ load bleed correction profile from original file.")
                    self.correction_profiles['bleed'] = load_correction_profile('bleedthrough', self.shared_parameters['corr_channels'], 
                                                self.correction_folder, all_channels=self.channels, 
                                                im_size=self.shared_parameters['single_im_size'],
                                                ref_channel=_ref_channel,
                                                verbose=_verbose)
        # loading chromatic
        if self.shared_parameters['corr_chromatic']:
            ## chromatic profiles
            if 'chromatic' in self.correction_profiles and self.correction_profiles['chromatic'] is not None and len(self.correction_profiles['chromatic']) > 0 and not _overwrite:
                print(f"++ chromatic correction profile already exist, skip.")
            else:
                _savefile_load_chromatic = False
                if _load_from_savefile_first:
                    # try to load from save_file
                    with h5py.File(self.save_filename, "a", libver='latest') as _f:
                        if 'corrections' in _f.keys():
                            _grp = _f['corrections']
                            _pf_dict = {}
                            for _ch in self.shared_parameters['corr_channels']:
                                _chromatic_key = f'{_ch}_chromatic'
                                if _chromatic_key in _grp:
                                    _pf = _grp[_chromatic_key][:]
                                    if (np.array(_pf) != _empty_value).any():
                                        _pf_dict[str(_ch)] = _pf
                                    elif np.size(_pf) == 1:
                                        _pf_dict[str(_ch)] = None
                            if set(list(_pf_dict.keys())) == set(list(self.shared_parameters['corr_channels'])):
                                if _verbose:
                                    print(f"++ load chromatic correction profile directly from savefile.")
                                _savefile_load_chromatic = True
                                self.correction_profiles['chromatic'] = _pf_dict
                                
                # if not successfully loaded or not savefile first, load from original file
                if not _savefile_load_chromatic:
                    if _verbose:
                        print(f"++ load chromatic correction profile from original file.")
                    self.correction_profiles['chromatic']= load_correction_profile('chromatic', self.shared_parameters['corr_channels'], 
                                                self.correction_folder, all_channels=self.channels, 
                                                im_size=self.shared_parameters['single_im_size'],
                                                ref_channel=_ref_channel,
                                                verbose=_verbose)
            
            ## chromatic_constants
            if 'chromatic_constants' in self.correction_profiles and self.correction_profiles['chromatic_constants'] is not None and len(self.correction_profiles['chromatic_constants']) > 0 and not _overwrite:
                print(f"++ chromatic_constants correction profile already exist, skip.")
            else:
                _savefile_load_chromatic_constants = False
                if _load_from_savefile_first:
                    # try to load from save_file
                    with h5py.File(self.save_filename, "a", libver='latest') as _f:
                        if 'corrections' in _f.keys():
                            _grp = _f['corrections']
                            _pf_dict = {}
                            for _ch in self.shared_parameters['corr_channels']:
                                _chromatic_constants_key = f'{_ch}_chromatic_constants'
                                if _chromatic_constants_key in _grp:
                                    _info_dict= {}
                                    for _key in _grp[_chromatic_constants_key].keys():
                                        if _key == 'constants':
                                            _saved_consts = _grp[_chromatic_constants_key][_key][:]
                                            _consts = [_row[np.isnan(_row)==False] for _row in _saved_consts]
                                            _info_dict[_key] = _consts
                                        else:
                                            _info_dict[_key] = _grp[_chromatic_constants_key][_key][:]
                                    # append
                                    if len(_info_dict) > 0:
                                        _pf_dict[str(_ch)] = _info_dict
                                    else:
                                        _pf_dict[str(_ch)] = None
                            if set(list(_pf_dict.keys())) == set(list(self.shared_parameters['corr_channels'])):
                                if _verbose:
                                    print(f"++ load chromatic_constants correction profile directly from savefile.")
                                _savefile_load_chromatic_constants = True
                                self.correction_profiles['chromatic_constants'] = _pf_dict
                                
                # if not successfully loaded or not savefile first, load from original file
                if not _savefile_load_chromatic_constants:
                    if _verbose:
                        print(f"++ load chromatic_constants correction profile from original file.")
                    # chromatic constant dicts
                    self.correction_profiles['chromatic_constants']= load_correction_profile(
                        'chromatic_constants', 
                        self.shared_parameters['corr_channels'], 
                        self.correction_folder, all_channels=self.channels, 
                        im_size=self.shared_parameters['single_im_size'],
                        ref_channel=_ref_channel,
                        verbose=_verbose)
            
        # load illumination
        if self.shared_parameters['corr_illumination']:
            if 'illumination' in self.correction_profiles and self.correction_profiles['illumination'] is not None and len(self.correction_profiles['illumination']) > 0 and not _overwrite:
                print(f"++ illumination correction profile already exist, skip.")
            else:
                #drift_channel = str(self.drift_channel)
                #dapi_channel = str(self.dapi_channel)
                #self.shared_parameters['corr_channels']+[drift_channel, dapi_channel]
                # all channel needs illumination correction
                _illumination_channels = self.channels
                ## illumination profiles
                _savefile_load_illumination = False
                if _load_from_savefile_first:
                    # try to load from save_file
                    with h5py.File(self.save_filename, "a", libver='latest') as _f:
                        if 'corrections' in _f.keys():
                            _grp = _f['corrections']
                            _pf_dict = {}
                            for _ch in _illumination_channels:
                                _illumination_key = f'{_ch}_illumination'
                                if _illumination_key in _grp:
                                    _pf = _grp[_illumination_key][:]
                                    if (np.array(_pf) != _empty_value).any():
                                        _pf_dict[str(_ch)] = _pf
                                    elif np.size(_pf) == 1:
                                        _pf_dict[str(_ch)] = None
                            if set(list(_pf_dict.keys())) == set(list(_illumination_channels)):
                                if _verbose:
                                    print(f"++ load illumination correction profile directly from savefile.")
                                _savefile_load_illumination = True
                                self.correction_profiles['illumination'] = _pf_dict
                                
                # if not successfully loaded or not savefile first, load from original file
                if not _savefile_load_illumination:
                    if _verbose:
                        print(f"++ load illumination correction profile from original file.")
        

                    self.correction_profiles['illumination'] = load_correction_profile('illumination', 
                                                        _illumination_channels, 
                                                        self.correction_folder, all_channels=self.channels, 
                                                        im_size=self.shared_parameters['single_im_size'],
                                                        ref_channel=_ref_channel,
                                                        verbose=_verbose)
        return

    def _save_correction_profile(self, _correction_folder=None, _overwrite=False, _verbose=True):
        """Function to save loaded correction profiles into hdf5 save file"""
        
        with h5py.File(self.save_filename, "a", libver='latest') as _f:

            # create a correction_profile group if necessary
            if 'corrections' not in _f.keys():
                _grp = _f.create_group('corrections')
            else:
                _grp = _f['corrections']
            
            # save illumination
            if 'illumination' in self.correction_profiles:
                # loop through existing profiles
                for _ch, _pf in self.correction_profiles['illumination'].items():
                    _key = str(_ch)+"_illumination"
                    if _key not in _grp.keys():
                        _dat = _grp.create_dataset(_key, 
                                                    np.array(_pf.shape), 
                                                    dtype='f')
                    else:
                        _dat = _grp[_key]
                    # save
                    if (np.array(_dat[:]) != self.shared_parameters['empty_value']).any() and not _overwrite:
                        if _verbose:
                            print(f"-- {_key} profile already exist in save_file: {self.save_filename}, skip.")
                    else:
                        if _verbose:
                            print(f"-- saving {_key} profile to save_file: {self.save_filename}.")
                        if _pf is not None:
                            _dat[:] = _pf

            # save chromatic
            if 'chromatic' in self.correction_profiles:
                # loop through existing profiles
                for _ch, _pf in self.correction_profiles['chromatic'].items():
                    _key = str(_ch)+"_chromatic"
                    if _pf is None:
                        _shape = (1,)
                    else:
                        _shape = np.array(np.array(_pf).shape)
                    # create dataset
                    if _key not in _grp.keys():
                        _dat = _grp.create_dataset(_key, _shape, dtype='f')
                    else:
                        _dat = _grp[_key]
                    # save
                    if (np.array(_dat[:]) != self.shared_parameters['empty_value']).any() and not _overwrite:
                        if _verbose:
                            print(f"-- {_key} profile already exist in save_file: {self.save_filename}, skip.")
                    else:
                        if _verbose:
                            print(f"-- saving {_key} profile to save_file: {self.save_filename}.")
                        if _pf is not None:
                            _dat[:] = _pf

            # save chromatic_constants
            if 'chromatic_constants' in self.correction_profiles:
                # loop through existing profiles
                for _ch, _pf in self.correction_profiles['chromatic_constants'].items():
                    _key = str(_ch)+"_chromatic_constants"

                    # create dataset
                    if _key not in _grp.keys():
                        _dat = _grp.create_group(_key)
                    elif _overwrite:
                        del(_grp[_key])
                        _dat = _grp.create_group(_key)
                    else:
                        _dat = _grp[_key]
                    _updated_info = []
                    # save
                    if _pf is not None:
                        for _k, _v in _pf.items():
                            # write updating keys
                            if _k not in _dat.keys():
                                _updated_info.append(_k)
                                if np.array(_v).dtype == 'O':
                                    _size = max([len(_row) for _row in _v])
                                    _save_consts = np.ones([len(_v), _size]) * np.nan
                                    for _irow, _row in enumerate(_v):
                                        _save_consts[_irow, :len(_row)] = _row
                                    _info = _dat.create_dataset(_k, np.shape(_save_consts), dtype='f')
                                    _info[:] = _save_consts
                                else:
                                    _info = _dat.create_dataset(_k, np.shape(np.array(_v)), dtype='f')
                                    _info[:] = np.array(_v)
                    if _verbose:
                        if len(_updated_info) > 0:
                            print(f"-- saving {_key} profile with {_updated_info} to save_file: {self.save_filename}.")
                        else:
                            print(f"-- {_key} profile already exist in save_file: {self.save_filename}, skip.")
            # save chromatic
            if 'bleed' in self.correction_profiles:
                # loop through existing profiles
                _pf = self.correction_profiles['bleed']

                _key = ''
                for _ch in self.shared_parameters['corr_channels']:
                    _key += f'{_ch}_'
                _key += "bleed"

                if _pf is None:
                    _shape = (1,)
                else:
                    _shape = np.array(np.array(_pf).shape)
                # create dataset
                if _key not in _grp.keys():
                    _dat = _grp.create_dataset(_key, _shape, dtype='f')
                else:
                    _dat = _grp[_key]
                # save
                if (np.array(_dat[:]) != self.shared_parameters['empty_value']).any() and not _overwrite:
                    if _verbose:
                        print(f"-- {_key} profile already exist in save_file: {self.save_filename}, skip.")
                else:
                    if _verbose:
                        print(f"-- saving {_key} profile to save_file: {self.save_filename}.")
                    if _pf is not None:
                        _dat[:] = _pf


    ## load existing drift info
    def _load_drift_file(self , _drift_basename=None, _drift_postfix='_current_cor.pkl', 
                         _sequential_mode=False, _verbose=False):
        """Function to simply load drift file"""
        if _verbose:
            print(f"-- loading drift for fov: {self.fov_name}")
        if _drift_basename is None:
            _postfix = _drift_postfix
            if _sequential_mode:
                _postfix = '_sequential' + _postfix
            _drift_basename = self.fov_name.replace('.dax', _postfix)
        # get filename
        _drift_filename = os.path.join(self.drift_folder, _drift_basename)
        if os.path.isfile(_drift_filename):
            if _verbose:
                print(f"--- from file: {_drift_filename}")
            self.drift = pickle.load(open(_drift_filename, 'rb'))
            return True
        else:
            if _verbose:
                print(f"--- file {_drift_filename} not exist, exit.")
            return False

    ## generate reference images and reference centers
    def _load_reference_image(self, 
                              _data_type="",
                              _save=True, 
                              _overwrite=False, 
                              _verbose=True):
        """Function to load ref image for fov class"""
        # check ref_filename
        if _data_type == "":
            _ref_filename = getattr(self, f'ref_filename', "")
        else:
            _ref_filename = getattr(self, f'{_data_type}_ref_filename', "")
        if _ref_filename == "":
            if _data_type == "":
                _ref_id = getattr(self, f'ref_id', "")
            else:
                _ref_id = getattr(self, f'{_data_type}_ref_id', None)
            if _ref_id is None:
                raise AttributeError(f"data_type: {_data_type}_ref_filename or {_data_type}_ref_id should be given!")
            _ref_filename = os.path.join(self.annotated_folders[int(_ref_id)], self.fov_name)
        # define used_channels in this round
        _info = self.color_dic[os.path.basename(os.path.dirname(_ref_filename))]
        _used_channels = []
        for _mk, _ch in zip(_info, self.channels):
            if _mk.lower() == 'null':
                continue
            else:
                _used_channels.append(_ch)

        if _verbose:
            print(f"+ load reference image from file:{_ref_filename}")
        if 'correct_fov_image' not in locals():
            from ..io_tools.load import correct_fov_image
        
        if hasattr(self, f'{_data_type}_ref_im') and not _overwrite:
            if _verbose:
                print(f"++ directly return existing attribute.")
            _ref_im = getattr(self, f'{_data_type}_ref_im')
        else:
            #print("**", _used_channels)
            #print("**", _ref_filename)
            # load
            _ref_im = correct_fov_image(_ref_filename, 
                                        [self.drift_channel], 
                                        single_im_size=self.shared_parameters['single_im_size'],
                                        all_channels=_used_channels,
                                        num_buffer_frames=self.shared_parameters['num_buffer_frames'],
                                        num_empty_frames=self.shared_parameters['num_empty_frames'],
                                        drift=None, calculate_drift=False,
                                        drift_channel=self.drift_channel,
                                        correction_folder=self.correction_folder,
                                        warp_image=False,
                                        illumination_corr=True,
                                        bleed_corr=False, 
                                        chromatic_corr=False, 
                                        chromatic_ref_channel=self.shared_parameters['ref_channel'],
                                        z_shift_corr=self.shared_parameters['corr_Z_shift'],
                                        verbose=_verbose,
                                        )[0][0]

            setattr(self, f'{_data_type}_ref_im', _ref_im)

            # save new chromosome image
            if _save:
                self._save_to_file('fov_info', _save_attr_list=[f'{_data_type}_ref_im'],
                                   _overwrite=_overwrite,
                                   _verbose=_verbose)

        return _ref_im

    ## check drift info
    def _check_drift(self, _load_drift_kwargs={}, 
                     _load_info_kwargs={}, _verbose=False):
        """Check whether drift exists and whether all keys required for images exists"""
        ## try to load drift if not exist
        if not hasattr(self, 'drift') or len(self.drift) == 0:
            if _verbose:
                print("-- No drift attribute detected, try to load from file")
            _flag = self._load_drift_file(_verbose=_verbose, **_load_drift_kwargs)
            if not _flag:
                return False
        ## drift exist, do check
        # load color_dic as a reference
        if not hasattr(self, 'color_dic'):
            self._load_color_info(**_load_info_kwargs)
            # check every folder in color_dic whether exists in drift
            for _hyb_fd, _info in self.color_dic.items():
                _drift_query = os.path.join(_hyb_fd, self.fov_name)
                if _drift_query not in self.drift:
                    if _verbose:
                        print(f"-- drift info for {_drift_query} was not found")
                    return False
        # if everything is fine return True
        return True

    def _bead_drift(self, _sequential_mode=True, _load_annotated_only=True, 
                    _size=500, _ref_id=0, _drift_postfix='_current_cor.pkl', 
                    _num_threads=12, _coord_sel=None, _force=False, _dynamic=True, 
                    _stringent=True, _verbose=True):
        # num-threads
        if hasattr(self, 'num_threads'):
            _num_threads = min(_num_threads, self.num_threads)
        # if drift meets requirements:
        if self._check_drift(_verbose=False) and not _force:
            if _verbose:
                print(f"- drift already exists for fov:{self.fov_name}, skip")
            return getattr(self,'drift')
        else:
            # load color usage if not given
            if not hasattr(self, 'channels'):
                self._load_color_info()
            # check whether load annotated only
            if _load_annotated_only:
                _folders = self.annotated_folders
            else:
                _folders = self.folders
            # load existing drift file 
            _drift_filename = os.path.join(self.drift_folder, self.fov_name.replace('.dax', _drift_postfix))
            _sequential_drift_filename = os.path.join(self.drift_folder, self.fov_name.replace('.dax', '_sequential'+_drift_postfix))
            # check drift filename and sequential file name:
            # whether with sequential mode determines the order to load files
            if _sequential_mode:
                _check_dft_files = [_sequential_drift_filename, _drift_filename]
            else:
                _check_dft_files = [_drift_filename, _sequential_drift_filename]
            for _dft_filename in _check_dft_files:
                # check one drift file
                if os.path.isfile(_dft_filename):
                    _drift = pickle.load(open(_dft_filename, 'rb'))
                    _exist = [os.path.join(os.path.basename(_fd),self.fov_name) for _fd in _folders \
                            if os.path.join(os.path.basename(_fd),self.fov_name) in _drift]
                    if len(_exist) == len(_folders):
                        if _verbose:
                            print(f"- directly load drift from file:{_dft_filename}")
                        self.drift = _drift
                        return self.drift
            # if non-of existing files fulfills requirements, initialize
            if _verbose:
                print("- start a new drift correction!")

            ## proceed to amend drift correction
            from corrections import Calculate_Bead_Drift
            _drift, _failed_count = Calculate_Bead_Drift(_folders, [self.fov_name], 0, 
                                        num_threads=_num_threads,sequential_mode=_sequential_mode, 
                                        ref_id=_ref_id, drift_size=_size, coord_sel=_coord_sel,
                                        single_im_size=self.shared_parameters['single_im_size'], 
                                        all_channels=self.channels,
                                        num_buffer_frames=self.shared_parameters['num_buffer_frames'], 
                                        num_empty_frames=self.shared_parameters['num_empty_frames'], 
                                        illumination_corr=self.shared_parameters['corr_illumination'],
                                        save_postfix=_drift_postfix,
                                        save_folder=self.drift_folder, stringent=_stringent,
                                        overwrite=_force, verbose=_verbose)
            if _verbose:
                print(f"- drift correction for {len(_drift)} frames has been generated.")
            _exist = [os.path.join(os.path.basename(_fd),self.fov_name) for _fd in _folders \
                if os.path.join(os.path.basename(_fd),self.fov_name) in _drift]
            # print if some are failed
            if _failed_count > 0:
                print(f"-- failed number: {_failed_count}"
                )
            if len(_exist) == len(_folders):
                self.drift = _drift
                return self.drift
            else:
                raise ValueError("length of _drift doesn't match _folders!")


    def _process_image_to_spots(self, _data_type, _sel_folders=[], _sel_ids=[], 
                                _load_common_correction_profiles=True, 
                                _load_common_reference=True, 
                                _load_with_multiple=True, 
                                _use_exist_images=False, 
                                _warp_images=True, 
                                _save_images=True, 
                                _save_drift=True,
                                _save_fitted_spots=True, 
                                _correction_args={},
                                _drift_args={}, 
                                _fit_spots=True,
                                _fit_in_mask=False, 
                                _fitting_args={},
                                _overwrite_drift=False, 
                                _overwrite_image=False, 
                                _overwrite_spot=False,
                                _verbose=True):
        ## check inputs
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong input for _data_type:{_data_type}, should be within {self.shared_parameters['allowed_data_types'].keys()}")
        # extract datatype marker
        _dtype_mk = self.shared_parameters['allowed_data_types'][_data_type]
        # get color_dic data-type
        from .batch_functions import _color_dic_stat
        _type_dic = _color_dic_stat(self.color_dic, 
                                    self.channels, 
                                    self.shared_parameters['allowed_data_types'])
        ## select folders
        _input_fds = []
        if _sel_folders is not None and len(_sel_folders) > 0:
            # load folders
            for _fd in _sel_folders:
                if _fd in self.annotated_folders:
                    _input_fds.append(_fd)
            if _verbose:
                print(f"-- {len(_sel_folders)} folders given, {len(_input_fds)} folders selected.")
        else:
            _sel_folders = self.annotated_folders
            if _verbose:
                print(f"-- folders not selected, allow processing all {len(self.annotated_folders)} folders")
        # check selected ids
        if _sel_ids is not None and len(_sel_ids) > 0:
            _sel_ids = [int(_id) for _id in _sel_ids] # convert to list of ints
            for _id in _sel_ids:
                if _id not in _type_dic[_data_type]['ids']:
                    print(f"id: {_id} not allowed in color_dic!")
            _sel_ids = [_id for _id in _sel_ids if _id in _type_dic[_data_type]['ids']]
        else:
            # if not given, process all ids for this data_type
            _sel_ids = [int(_id) for _id in _type_dic[_data_type]['ids']]
        
        ## load correction profiles if necessary:
        if _load_common_correction_profiles and not hasattr(self, 'correction_profiles'):
            self._load_correction_profiles()

        ## load shared drift references
        if _load_common_reference and not hasattr(self, f'{_data_type}_ref_im'):
            self._load_reference_image(_data_type=_data_type, _verbose=_verbose)
        if hasattr(self, f'{_data_type}_ref_im'):
            _drift_reference = getattr(self, f'{_data_type}_ref_im')
        else:
            _drift_reference = getattr(self, f'{_data_type}_ref_filename')
            #self._prepare_dirft_references(_verbose=_verbose)

        ## multi-processing for correct_splice_images
        # prepare common params
        _correction_args.update(
            {
            'single_im_size': self.shared_parameters['single_im_size'],
            'all_channels':self.channels,
            'num_buffer_frames':self.shared_parameters['num_buffer_frames'],
            'num_empty_frames':self.shared_parameters['num_empty_frames'],
            'correction_folder':self.correction_folder,
            'corr_channels': self.shared_parameters['corr_channels'],
            'hot_pixel_corr':self.shared_parameters['corr_hot_pixel'], 
            'z_shift_corr':self.shared_parameters['corr_Z_shift'], 
            'bleed_corr':self.shared_parameters['corr_bleed'],
            'bleed_profile':self.correction_profiles['bleed'],
            'illumination_corr':self.shared_parameters['corr_illumination'],
            'illumination_profile':self.correction_profiles['illumination'],
            'chromatic_corr':self.shared_parameters['corr_chromatic'],
            'normalization':self.shared_parameters['normalization'],
            'gaussian_highpass': self.shared_parameters['corr_gaussian_highpass'],
            }
        )
        # specifically, chromatic profile loaded is based on whether warp image
        if _warp_images:
            _correction_args.update({
                'chromatic_profile':self.correction_profiles['chromatic'],})
        else:
            _correction_args.update({
                'chromatic_profile':self.correction_profiles['chromatic_constants'],})
        # required parameters for drift correction
        _drift_args.update({
            'drift_channel': self.drift_channel,
            'use_autocorr': self.shared_parameters['drift_use_autocorr'],
            'drift_args': self.shared_parameters['drift_args'],
        })
        # required parameters for fitting
        _fitting_args.update({
            'max_num_seeds' : self.shared_parameters['max_num_seeds'],
            'th_seed': self.shared_parameters['spot_seeding_th'],
            'min_dynamic_seeds': self.shared_parameters['min_num_seeds'],
            'remove_hot_pixel': self.shared_parameters['corr_hot_pixel'],
            'normalize_background': self.shared_parameters['normalize_intensity_background'],
            'normalize_local': self.shared_parameters['normalize_intensity_local'],
        })
        # check if seed_mask is given
        if _fit_in_mask:
            if 'seed_mask' not in _fitting_args or _fitting_args['seed_mask'] is None:
                raise KeyError(f"seed_mask should be given if _fit_in_mask specified.")

        # initiate locks
        _manager = mp.Manager()
        # recursive lock so each thread can acquire multiple times
        # lock to make sure only one thread is reading or writing from hard drive.
        _fov_savefile_lock = _manager.RLock() 
        # lock to write to drift file
        _drift_file_lock = _manager.RLock() 

        # prepare kwargs to be processed.
        _processing_arg_list = []
        _processing_id_list = []
        
        # loop through annotated folders
        for _fd in _sel_folders:
            # get the dax filename
            _dax_filename = os.path.join(_fd, self.fov_name)
            # get selected channels
            _info = self.color_dic[os.path.basename(_fd)]
            _used_channels = [] # all used channels in this round
            _sel_channels = [] # selected channels to be processed in this round
            _reg_ids = []
            # loop through color_dic to collect selected channels and ids
            for _mk, _ch in zip(_info, self.channels):
                if _mk.lower() == 'null' or _mk.lower() == 'empty':
                    continue
                else:
                    _used_channels.append(_ch)
                    if _dtype_mk in _mk:
                        _id = int(_mk.split(_dtype_mk)[1])
                        if _id in _sel_ids:
                            # append sel_channels and reg_ids for now
                            _sel_channels.append(_ch)
                            _reg_ids.append(_id)
            # check existence of these candidate selected ids
            if len(_sel_channels) > 0:
                # Case 1: if trying to use existing images 
                if _use_exist_images:
                    _exist_spots, _exist_drifts, _exist_flags, _exist_ims =  self._check_exist_data(_data_type, _reg_ids, _check_im=True, _verbose=_verbose)
                    # take overwrite into consideration
                    _exist_spots = _exist_spots & (not _overwrite_spot)
                    # if not required to fit spots, set flag to be exist
                    if not _fit_spots:
                        _exit_spots = True
                    _exist_ims = _exist_ims & (not _overwrite_image)
                    # select channels based on exist spots and ims
                    _sel_channels = [_ch for _ch, _es, _ei in zip(_sel_channels, _exist_spots, _exist_ims)
                                    if not _es or not _ei] # if spot or im not exist, process sel_channel
                    _reg_ids = [_id for _id, _es, _ei in zip(_reg_ids, _exist_spots, _exist_ims)
                                if not _es or not _ei] # if spot or im not exist, process reg_id
                # Case 2: image saving is not required
                else:
                    _exist_spots, _exist_drifts, _exist_flags =  self._check_exist_data(_data_type, _reg_ids, _check_im=False, _verbose=_verbose)
                    # take overwrite into consideration
                    _exist_spots = _exist_spots & (not _overwrite_spot)
                    # if not required to fit spots, set flag to be exist
                    if not _fit_spots:
                        _exit_spots = True
                    
                    # select channels based on exist spots and ims
                    _sel_channels = [_ch for _ch, _es in zip(_sel_channels, _exist_spots)
                                    if not _es] # if spot not exist, process sel_channel
                    _reg_ids = [_id for _id, _es in zip(_reg_ids, _exist_spots)
                                if not _es] # if spot not exist, process reg_id
            # update a correction_args with used_channel for this round
            #print(f"used_channels: {_used_channels}")
            _round_corr_channels = [_ch for _ch in _correction_args['corr_channels']
                if _ch in _used_channels]
            _round_correction_args = {_k:_v for _k,_v in _correction_args.items()}
            _round_correction_args.update({'all_channels':_used_channels})
            # if only one channel needs correction, don't do bleedthrough correction
            if len(_round_corr_channels) <= 1:
                _round_correction_args['bleed_corr'] = False
                _round_correction_args['bleed_profile'] = None
            # if length of round_correction_channels is smaller than corr_channel, subsample the correction profile
            elif len(_round_corr_channels) != len(_correction_args['corr_channels']):
                # get corr_ch_inds
                _round_corr_ch_inds = np.array([ _i for _i, _ch in enumerate(_correction_args['corr_channels']) if _ch in _used_channels], dtype=np.int32)
                # modify bleedthrough profile
                _round_correction_args['bleed_profile'] = _round_correction_args['bleed_profile'][_round_corr_ch_inds][:, _round_corr_ch_inds]


            # append if any channels selected
            if len(_sel_channels) > 0:
                _args = (_dax_filename, 
                         _sel_channels, 
                        self.save_filename, 
                        _data_type, 
                        _reg_ids, 
                        _drift_reference,
                        _fov_savefile_lock,
                        _warp_images, 
                        _round_correction_args, 
                        _save_images, 
                        self.shared_parameters['empty_value'],
                        _fov_savefile_lock, 
                        _overwrite_image, 
                        _drift_args, 
                        _save_drift, 
                        self.drift_filename,
                        _drift_file_lock, 
                        _overwrite_drift,
                        _fit_spots,
                        _fit_in_mask,
                        _fitting_args, 
                        _save_fitted_spots,
                        _fov_savefile_lock, 
                        _overwrite_spot, 
                        _verbose,)
                _processing_arg_list.append(_args)
                _processing_id_list.append(_reg_ids)

        if len(_processing_arg_list) > 0:
            from .batch_functions import batch_process_image_to_spots, killchild
            # multi-processing
            if self.parallel:
                with mp.Pool(self.num_threads) as _processing_pool:
                    if _verbose:
                        print(f"+ Start multi-processing of pre-processing for {len(_processing_arg_list)} images with {self.num_threads} threads")
                        print(f"++ processing {_data_type} ids: {np.sort(np.concatenate(_processing_id_list))}", end=' ')
                        _start_time = time.time()
                    # Multi-proessing!
                    _processing_pool.starmap(
                        batch_process_image_to_spots,
                        _processing_arg_list, 
                        chunksize=1)
                    # close multiprocessing
                    _processing_pool.close()
                    _processing_pool.join()
                    _processing_pool.terminate()
                # clear
                killchild()        
                if _verbose:
                    print(f", finish in {time.time()-_start_time:.2f}s.")
            else:
                if _verbose:
                    print(f"+ Start sequential pre-processing for {len(_processing_arg_list)} images")
                    print(f"++ processed {_data_type} ids: {np.sort(np.concatenate(_processing_id_list))}", end=' ')
                    _start_time = time.time()
                for _ids, _args in zip(_processing_id_list, _processing_arg_list):
                    batch_process_image_to_spots(*_args)
                if _verbose:
                    print(f"in {time.time()-_start_time:.2f}s.")
        else:
            if _verbose:
                print(f"- No {_data_type} images and spots requires processing, skip.")

    def _save_to_file(self, _type, _save_attr_list=[], 
                      _overwrite=False, _verbose=True):
        """Function to save attributes into standard HDF5 save_file of fov class.
        """
        ## check inputs:
        _type = str(_type).lower()
        # only allows saving the following types:
        _allowed_save_types = ['fov_info', 'segmentation', 'correction', 'signal'] \
                              + list(self.shared_parameters['allowed_data_types'].keys())
        if _type not in _allowed_save_types:
            raise ValueError(f"Wrong input for _type:{_type}, should be within:{_allowed_save_types}.")
        # exclude the following attributes:
        _excluded_attrs = ['im_dict', 'spot_dict', 'channel_dict', 'correction_profiles']
        # save file should exist
        if not os.path.isfile(self.save_filename):
            print(f"* create savefile: {self.save_filename}")
        
        ## start saving here:
        if _verbose:
            print(f"-- saving {_type} to file: {self.save_filename}")
            _save_start = time.time()
        
        with h5py.File(self.save_filename, "a", libver='latest') as _f:

            ### add signal as a temporary group to store chrom_coords and spot intensity th for each hyb @ Shiwei Liu
            ## signal 
            if _type == 'signal':
                # create signal group if not exist 
                if 'signal' not in _f.keys():
                    _grp = _f.create_group('signal') # create signal group
                else:
                    _grp = _f['signal']

                # create chrom_coords subgroup
                if 'chrom_coords' not in _grp:
                    _subgrp = _f.create_group('signal/chrom_coords')
                # save chrom_coords as dataset under the subgroup
                _subgrp = _f['signal']['chrom_coords']
                if hasattr(self,'chrom_coords'):
                    _chrom_coords = getattr(self, 'chrom_coords')
                    # check if dset exist
                    if len(_save_attr_list) > 0 and _save_attr_list is not None:
                        if 'chrom_coords' in _save_attr_list:
                            if not len(_subgrp.keys()) >0 and type(_chrom_coords) == dict:
                                # save new dataset
                                for _chr_key, _chr_coord in _chrom_coords.items():
                                    _dset = _subgrp.create_dataset (_chr_key, tuple(_chr_coord.shape))
                                    _dset [:] =  _chr_coord

                            elif len(_subgrp.keys()) >0 and type(_chrom_coords) == dict:
                                if _overwrite:
                                     # delete existing dataset first if overwrite
                                    for _dset in _subgrp.keys():
                                         del _subgrp[_dset]
                                    # save new dataset
                                    for _chr_key, _chr_coord in _chrom_coords.items():
                                        _dset = _subgrp.create_dataset (_chr_key, tuple(_chr_coord.shape))
                                        _dset [:] =  _chr_coord
                
                # create spot_intensity_th subgroup
                if 'spot_intensity_th' not in _grp:
                    _subgrp = _f.create_group('signal/spot_intensity_th')
                # save spot_intensity_th as dataset under the subgroup
                _subgrp = _f['signal']['spot_intensity_th']
                if hasattr(self,'spot_intensity_th'):
                    _spot_intensity_th = getattr(self, 'spot_intensity_th')
                    if len(_save_attr_list) > 0 and _save_attr_list is not None:
                        if 'spot_intensity_th' in _save_attr_list:
                            # check if dset exist
                            if not len(_subgrp.keys()) >0 and type(_spot_intensity_th) == dict:
                                # save new dataset
                                for _region_key, _spot_th in _spot_intensity_th.items():
                                    _spot_th = np.array([_spot_th])     
                                    _dset = _subgrp.create_dataset (_region_key, tuple(_spot_th.shape))
                                    _dset [:] =  _spot_th

                            elif len(_subgrp.keys()) >0 and type(_spot_intensity_th) == dict:
                                if _overwrite:
                                    # delete existing dataset first if overwrite
                                    for _dset in _subgrp.keys():
                                        del _subgrp[_dset]
                                     # save new dataset
                                    for _region_key, _spot_th in _spot_intensity_th.items():
                                        _spot_th = np.array([_spot_th])     
                                        _dset = _subgrp.create_dataset (_region_key, tuple(_spot_th.shape))
                                        _dset [:] =  _spot_th
               ### new signal group above 

            ## segmentation
            if _type == 'segmentation':
                # create segmentation group if not exist 
                if 'segmentation' not in _f.keys():
                    _grp = _f.create_group('segmentation') # create segmentation group
                else:
                    _grp = _f['segmentation']
                # directly create segmentation label dataset
                if 'segmentation_label' not in _grp:
                    _seg = _grp.create_dataset('segmentation_label', 
                                            self.shared_parameters['single_im_size'][-self.segmentation_dim:], 
                                            dtype='i8')
                if hasattr(self, 'segmentation_label'):
                    if len(_save_attr_list) > 0 and _save_attr_list is not None:
                        if 'segmentation_label' in _save_attr_list:
                            _grp['segmentation_label'] = getattr(self, 'segmentation_label')
                    else:
                        _grp['segmentation_label'] = getattr(self, 'segmentation_label')
                # create other segmentation related datasets
                for _attr_name in dir(self):
                    if _attr_name[0] != '_' and 'segmentation' in _attr_name and _attr_name not in _grp.keys():
                        if len(_save_attr_list) > 0 and _save_attr_list is not None:
                            # if save_attr_list is given validly and this attr not in it, skip.
                            if _attr_name not in _save_attr_list:
                                continue 
                        _grp[_attr_name] = getattr(self, _attr_name)

            elif _type == 'correction':
                self._save_correction_profile(self.correction_folder, _overwrite=_overwrite,
                                              _verbose=_verbose) 

            ## save basic attributes as info
            elif _type == 'fov_info':
                # initialize attributes as _info_attrs
                _info_attrs = []
                for _attr_name in dir(self):
                    # exclude all default attrs and functions
                    if _attr_name[0] != '_' and getattr(self, _attr_name) is not None and '<class ' not in str(getattr(self, _attr_name)):
                        # exclude some large files shouldnt save
                        if _attr_name in _excluded_attrs:
                            continue
                        # check within save_attr_list or not specified
                        if _save_attr_list is None or len(_save_attr_list) == 0 or _attr_name in _save_attr_list:
                            ## save
                            # extract the attribute
                            _attr = getattr(self, _attr_name)
                            # convert dict if necessary
                            if isinstance(_attr, dict):
                                _attr = str(_attr)
                            # save
                            if _attr_name not in _f.attrs or _overwrite:
                                _f.attrs[_attr_name] = _attr
                                _info_attrs.append(_attr_name)
                            
                if _verbose:
                    print(f"++ base attributes saved:{_info_attrs} in {time.time()-_save_start:.3f}s.")

            ## images and spots for a specific data type 
            elif _type in self.shared_parameters['allowed_data_types']:
                from .batch_functions import _color_dic_stat
                _type_dic = _color_dic_stat(self.color_dic, self.channels, self.shared_parameters['allowed_data_types'])
                if _type not in _type_dic:
                    print(f"--- given save type:{_type} doesn't exist in this dataset, skip.") 
                else:
                    # extract info dict for this data_type
                    _dict = _type_dic[_type]

                    # create data_type group if not exist 
                    if _type not in _f.keys():
                        _grp = _f.create_group(_type) 
                    else:
                        _grp = _f[_type]
                    # record updated data_type related attrs
                    _data_attrs = []
                    ## save images, ids, channels, save_flags
                    # calculate image shape and chunk shape
                    _im_shape = np.concatenate([np.array([len(_dict['ids'])]), 
                                                self.shared_parameters['single_im_size']])
                    _chunk_shape = np.concatenate([np.array([1]), 
                                                self.shared_parameters['single_im_size']])                              
                    # change size
                    _change_size_flag = []
                    # if missing any of these features, create new ones
                    # ids
                    if 'ids' not in _grp:
                        _ids = _grp.create_dataset('ids', (len(_dict['ids']),), dtype='i', data=_dict['ids'])
                        _ids = np.array(_dict['ids'], dtype=np.int) # save ids
                        _data_attrs.append('ids')
                    elif len(_dict['ids']) != len(_grp['ids']):
                        _change_size_flag.append('id')
                        _old_size=len(_grp['ids'])                   

                    # channels
                    if 'channels' not in _grp:
                        _channels = [str(_ch).encode('utf8') for _ch in _dict['channels']]
                        _chs = _grp.create_dataset('channels', (len(_dict['channels']),), dtype='S3', data=_channels)
                        _chs = np.array(_dict['channels'], dtype=str) # save ids
                        _data_attrs.append('channels')
                    elif len(_dict['channels']) != len(_grp['channels']):
                        _change_size_flag.append('channels')
                        _old_size=len(_grp['channels'])

                    # images
                    if 'ims' not in _grp:
                        _ims = _grp.create_dataset('ims', tuple(_im_shape), 
                                                   dtype='u2',  # uint16
                                                   chunks=tuple(_chunk_shape))
                        _data_attrs.append('ims')
                    elif len(_im_shape) != len(_grp['ims'].shape) or (_im_shape != (_grp['ims']).shape).any():
                        _change_size_flag.append('ims')
                        _old_size=len(_grp['ims'])

                    # spots
                    if 'spots' not in _grp:
                        if self.shared_parameters['max_num_seeds'] is None:
                            _spot_save_len = _max_num_seeds
                        else:
                            _spot_save_len = self.shared_parameters['max_num_seeds']
                        # create
                        _spots = _grp.create_dataset('spots', 
                                    (_im_shape[0], _spot_save_len, 11), 
                                    dtype='f', maxshape=(_im_shape[0],None,11), chunks=True)
                        _data_attrs.append('spots')
                    elif _im_shape[0] != len(_grp['spots']):
                        _change_size_flag.append('spots')
                        _old_size=len(_grp['spots'])
                    # raw spots (for debugging)
                    if 'raw_spots' not in _grp:
                        _spots = _grp.create_dataset('raw_spots', 
                                    (_im_shape[0], _spot_save_len, 11), 
                                    dtype='f', maxshape=(_im_shape[0],None,11), chunks=True)
                        _data_attrs.append('raw_spots')
                    elif _im_shape[0] != len(_grp['raw_spots']):
                        _change_size_flag.append('raw_spots')
                        _old_size=len(_grp['raw_spots'])

                    # drift
                    if 'drifts' not in _grp:
                        _drift = _grp.create_dataset('drifts', (_im_shape[0], 3), dtype='f')
                        _data_attrs.append('drifts')
                    elif _im_shape[0] != len(_grp['drifts']):
                        _change_size_flag.append('drifts')
                        _old_size=len(_grp['drifts'])

                    # flags for whether it's been written
                    if 'flags' not in _grp:
                        _filenames = _grp.create_dataset('flags', (_im_shape[0], ), dtype='u1')
                        _data_attrs.append('flags')
                    elif _im_shape[0] != len(_grp['flags']):
                        _change_size_flag.append('flags')
                        _old_size=len(_grp['flags'])

                    # Create other features
                    for _attr_name in dir(self):
                        if _attr_name[0] != '_' and _type+'_' in _attr_name:
                            _attr_feature = _attr_name.split(_type+'_')[1][1:]

                            if _attr_feature not in _grp.keys():
                                _grp[_attr_feature] = getattr(self, _attr_name)
                                _data_attrs.append(_attr_feature)
                    
                    # if change size, update these features:
                    if len(_change_size_flag) > 0:
                        print(f"* data size of {_type} is changing from {_old_size} to {len(_dict['ids'])} because of {_change_size_flag}")
                        ###UNDER CONSTRUCTION################
                        pass
                    # summarize
                    if _verbose:
                        print(f"--- {_type} attributes updated:{_data_attrs} in {time.time()-_save_start:.3f}s.")
                        _save_mid = time.time()

        ## save ims, spots, drifts, flags
        if _type in self.im_dict:
            _image_info = self.im_dict[_type]
            if 'ids' not in _image_info:
                print(f"--- ids for type:{_type} not given, skip.")
            else:
                _ids = _image_info['ids']
                # save images
                if 'ims' in _image_info and len(_image_info['ims']) == len(_ids):
                    from .batch_functions import save_image_to_fov_file
                    if 'drifts' in _image_info:
                        _drifts = _image_info['drifts']
                    else:
                        _drifts = None 
                    _ims_flag = save_image_to_fov_file(self.save_filename, 
                                                        _image_info['ims'],
                                                        _type, _ids, 
                                                        warp_image=self.shared_parameters['_warp_images'], 
                                                        drift=_drifts, 
                                                        overwrite=_overwrite,
                                                        verbose=_verbose)
                # save spots
                if 'spots' in _image_info and len(_image_info['spots']) == len(_ids):
                    from .batch_functions import save_spots_to_fov_file
                    _spots_flag = save_spots_to_fov_file(self.save_filename, 
                                                        _image_info['spots'],
                                                        _type, _ids, 
                                                        overwrite=_overwrite,
                                                        verbose=_verbose)
            if _verbose:
                print(f"--- save images and spots for {_type} in {time.time()-_save_mid:.3f}s.")

            

    def _check_exist_data(self, _data_type, _region_ids=None,
                          _check_im=False,
                          empty_value=0, _verbose=False):
        """function to check within a specific data_type, 
            a specific region_id, does image and spot exists
        Inputs:
            
        Outputs:
            _exist_im, 
            _exist_spots,
            _exist_drift,
            _exist_flag
        """
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong input data_type: {_data_type}, should be among:{self.shared_parameters['allowed_data_types']}")
        if _region_ids is None:
            # get color_dic data-type
            from .batch_functions import _color_dic_stat
            _type_dic = _color_dic_stat(self.color_dic, 
                                        self.channels, 
                                        self.shared_parameters['allowed_data_types'])
            _region_ids = _type_dic[_data_type]
        else:  
            if isinstance(_region_ids, int) or isinstance(_region_ids, np.int):
                _region_ids = [_region_ids]   
            elif not isinstance(_region_ids, list) and not isinstance(_region_ids, np.ndarray):
                raise TypeError(f"Wrong input type for region_ids:{_region_ids}")
            _region_ids = np.array([int(_i) for _i in _region_ids],dtype=np.int)
        # print
        if _verbose:
            _check_time = time.time()
            print(f"-- checking {_data_type}, region:{_region_ids}", end=' ')
            if _check_im:
                print("including images", end=' ')
        with h5py.File(self.save_filename, "a", libver='latest') as _f:
            if _data_type not in _f.keys():
                raise ValueError(f"input data type: {_data_type} doesn't exist in this save_file:{self.save_filename}")
            _grp = _f[_data_type]
            _ids = list(_grp['ids'][:])
            for _region_id in _region_ids:
                if _region_id not in _ids:
                    raise ValueError(f"region_id:{_region_id} not in {_data_type} ids.")

            # initialize
            _exist_im, _exist_spots, _exist_drift, _exist_flag = [],[],[],[]

            for _region_id in _region_ids:
                # get indices
                _ind = _ids.index(_region_id)

                if _check_im:
                    _exist_im.append((np.array(_grp['ims'][_ind]) != empty_value).any())

                _exist_spots.append((np.array(_grp['spots'][_ind]) != empty_value).any())
                _exist_drift.append((np.array(_grp['drifts'][_ind]) != empty_value).any())
                _exist_flag.append( np.array(_grp['flags'][_ind]))

        # convert to array
        _exist_spots = np.array(_exist_spots) 
        _exist_drift = np.array(_exist_drift) 
        _exist_flag = np.array(_exist_flag) 
        if _verbose:
            print(f"in {time.time()-_check_time:.3f}s.")
        if _check_im:
            _exist_im = np.array(_exist_im) 
            return _exist_spots, _exist_drift, _exist_flag, _exist_im
        else:
            return _exist_spots, _exist_drift, _exist_flag


    def _load_from_file(self, _type, _load_attr_list=[],
                        _load_processed=True, _load_image=False, 
                        _overwrite=False, _verbose=True):
        """Function to load from existing save_file"""
        ## check inputs:
        _type = str(_type).lower()
        # only allows loading the following types:
        _allowed_save_types = ['fov_info', 'segmentation', 'correction', 'signal'] \
                            + list(self.shared_parameters['allowed_data_types'].keys())
        if _type not in _allowed_save_types:
            raise ValueError(f"Wrong input for _type:{_type}, should be within:{_allowed_save_types}.")
        # save file should exist
        if not os.path.isfile(self.save_filename):
            raise FileExistsError(f"savefile: {self.save_filename} doesn't exist, exit.")
        
        ## start loading here:
        if _verbose:
            print(f"+ loading {_type} from file: {self.save_filename}")
            _load_start = time.time()
        
        _loaded_attrs = []
        
        with h5py.File(self.save_filename, "a", libver='latest') as _f:

            ### add loading for the new signal group, which is a temporary group to store chrom_coords and spot intensity th for each hyb @ Shiwei Liu

            ## signal
            if _type == 'signal':
                # check signal group exist or not
                if 'signal' not in _f.keys():
                    print(f"signal group doesn't exist in save_file:{self.save_filename}, exit.")
                else:
                    _grp = _f['signal']
                # check chrom_coords subgroup exist or not
                    if 'chrom_coords' not in _grp:
                        print(f"chrom_coords subgroup doesn't exist in save_file:{self.save_filename}, exit.")
                     # load chrom_coords as dataset under the subgroup
                    else:
                        if not hasattr(self, 'chrom_coords') or _overwrite:
                            if (len(_load_attr_list) > 0 and 'chrom_coords' in _load_attr_list):
                                _subgrp = _f['signal']['chrom_coords']
                                _chrom_coords = {}
                                if len(_subgrp.keys()) >0:
                                    # load dataset as a dict and set attr accordingly
                                    for _dset in _subgrp.keys():
                                        _chrom_coords[_dset] = _subgrp[_dset][:]
                                    setattr(self,'chrom_coords', _chrom_coords)
                                    _loaded_attrs.append('chrom_coords')

                                                    

                # check spot_intensity_th subgroup exist or not
                    if 'spot_intensity_th' not in _grp:
                        print(f"spot_intensity_th subgroup doesn't exist in save_file:{self.save_filename}, exit.")
                     # load spot_intensity_th as dataset under the subgroup
                    else:
                        if not hasattr(self,'spot_intensity_th') or _overwrite:
                            if (len(_load_attr_list) > 0 and 'spot_intensity_th' in _load_attr_list):
                                _subgrp = _f['signal']['spot_intensity_th']
                                _spot_intensity_th = {}
                                if len(_subgrp.keys()) >0:
                                    for _dset in _subgrp.keys():
                                        _spot_intensity_th[_dset] = float(_subgrp[_dset][:])
                                    setattr(self,'spot_intensity_th', _spot_intensity_th)
                                    _loaded_attrs.append('spot_intensity_th')

            ### loading for new signal group above 

  
            ## segmentation
            if _type == 'segmentation':
                # create segmentation group if not exist 
                if 'segmentation' not in _f.keys():
                    print(f"segmetation group doesn't exist in save_file:{self.save_filename}, exit.")
                else:
                    _grp = _f['segmentation']
                    # directly create segmentation label dataset
                    if 'segmentation_label' not in _grp:
                        _seg = _grp.create_dataset('segmentation_label', 
                                                self.shared_parameters['single_im_size'][-self.segmentation_dim:], 
                                                dtype='i8')
                    if hasattr(self, 'segmentation_label'):
                        if (len(_load_attr_list) > 0 and 'segmentation_label' in _load_attr_list) or len(_load_attr_list)==0:
                            if not hasattr(self, 'segmentation_label') or _overwrite:
                                setattr(self, 'segmentation_label', _grp['segmentation_label'])
                                _loaded_attrs.append('segmentation_label')
                        
                    # create other segmentation related datasets
                    for _attr_name in dir(self):
                        if _attr_name[0] != '_' and 'segmentation' in _attr_name and _attr_name in _grp.keys():
                            if (len(_load_attr_list) > 0 and _attr_name in _load_attr_list) or len(_load_attr_list)==0:
                                if not hasattr(self, _attr_name) or _overwrite:
                                    _value = np.array(_grp[_attr_name])
                                    # special treatment for strings
                                    if isinstance(_value.dtype, np.object):
                                        _value = str(_value)
                                    # save
                                    setattr(self, _attr_name, _value)
                                    _loaded_attrs.append(_attr_name)
                                    
            elif _type == 'correction':
                return self._load_correction_profiles(_overwrite=_overwrite,
                                                    _verbose=_verbose)
            
            ## save basic attributes as info     # load
            elif _type == 'fov_info':
                for _attr in _f.attrs:
                    if not hasattr(self, _attr) or _overwrite:
                        if _attr in _load_attr_list or len(_load_attr_list) == 0:
                            _value =  _f.attrs[_attr]

                            # There seems a SyntaxError, because continue not used properly in while/for loop @ Shiwei
                            # minor fix below using while loop
                            # str representation of dict can be loaded if the original dict is generated with str as keys
                            # convert dicts
                            if isinstance(_value, str) and _value[0] == '{' and _value[-1] == '}':
                                _convert_dict_bool = True
                                while _convert_dict_bool:
                                    try:
                                        print(f"try loading: {_attr}")
                                        _value = ast.literal_eval(str(_value))    # original dict should be str to be able to load
                                        _convert_dict_bool = False
                                    except:
                                        _convert_dict_bool = False
                                        continue
                                        

                            setattr(self, _attr, _value)
                            _loaded_attrs.append(_attr)

            elif _type in self.shared_parameters['allowed_data_types']:

                if hasattr(self, f"{_type}_ids") and \
                    hasattr(self, f"{_type}_drifts") and \
                    hasattr(self, f"{_type}_spots_list") and \
                    hasattr(self, f"{_type}_channels") and \
                    not _overwrite:
                    if _load_image and hasattr(self, f"{_type}_ims"):
                        return
                    elif not _load_image:
                        return
                _start_time = time.time()
                with h5py.File(self.save_filename, "a", libver='latest') as _f:
                    if _type in _f.keys():
                        _grp = _f[_type]
                        # load ids and spots
                        _flags = _grp['flags'][:]

                        # screen if specified
                        if _load_processed:
                            _ids = np.array([_id for _flg, _id in zip(_flags, _grp['ids'][:]) if _flg > 0])
                            _drifts = np.array([_dft for _flg, _dft in zip(_flags, _grp['drifts'][:]) if _flg > 0])
                            _spots_list = [_spots[_spots[:,0] > 0] for _flg, _spots in zip(_flags, _grp['spots'][:]) if _flg > 0]
                            _channels = [_ch.decode('utf8') for _flg, _ch in zip(_flags, _grp['channels'][:]) if _flg > 0]
                            if _load_image:
                                _ims = _grp['ims'][:]
                        else:
                            _ids = _grp['ids'][:]
                            _drifts = _grp['drifts'][:]
                            _spots_list = [_spots[_spots[:,0] > 0] for _spots in _grp['spots'][:]]
                            _channels = [str(_ch) for _ch in _grp['channels'][:]]
                            if _load_image:
                                _ims = _grp['ims'][:]
                        
                        # set attributes
                        setattr(self, f"{_type}_ids", _ids)
                        setattr(self, f"{_type}_drifts", _drifts)
                        setattr(self, f"{_type}_spots_list", _spots_list)
                        setattr(self, f"{_type}_channels", _channels)
                        setattr(self, f"{_type}_flags", _flags)
                        if _load_image:
                            setattr(self, f"{_type}_ims", _ims)
                        if _verbose:
                            print(f"++ finish loading {_type} in {time.time()-_start_time:.3f}s. ")
                    else:
                        raise KeyError(f"{_type} doesn't exist in savefile:{self.save_filename}")
                return

            
                        
        if _verbose:
            print(f"++ base attributes loaded:{_loaded_attrs} in {time.time()-_load_start:.3f}s.")

        return _loaded_attrs
    

    def _delete_save_file(self, _type):
        pass

    def _convert_to_cell_list(self):
        pass


    def _load_chromosome_image(self, _type='forward', 
                               _generate_data_type='unique', _generate_num_im=30,
                               _save=True, 
                               _overwrite=False, 
                               _verbose=True):
        """Function to load chrom image into fov class
        """
        
        if 'correct_fov_image' not in locals():
            from ..io_tools.load import correct_fov_image

        if hasattr(self, 'chrom_im') and not _overwrite:
            if _verbose:
                print(f"directly return existing attribute.")
            _chrom_im = getattr(self, 'chrom_im')
        else:
            # initialize default chromosome reference
            _select_chrom = False

            # find chrom im in color_usage
            for _fd, _infos in self.color_dic.items():
                
                for _ch, _info in zip(self.channels, _infos):
                    if 'chrom' in _info and _type in _info:
                        # define chromosome folder 
                        _chrom_fd = [_full_fd for _full_fd in self.annotated_folders 
                                    if os.path.basename(_full_fd) == _fd][0]
                        if _verbose:
                            print(f"-- choose chrom images from folder: {_chrom_fd}.")
                        _chrom_channel = _ch
                        _chrom_fd_ind = self.annotated_folders.index(_chrom_fd)
                        # load reference of chromosome image
                        if _chrom_fd_ind != self.ref_id:                                
                            if not hasattr(self, '_ref_im'):
                                self._load_reference_image(_verbose=_verbose)
                            _use_ref_im = True
                        else:
                            _use_ref_im = False 

                        _select_chrom = True  # successfully selected chrom
                        # decide used_channels
                        _used_channels = []
                        for _mk, _ch in zip(_infos, self.channels):
                            if _mk.lower() == 'null':
                                continue
                            else:
                                _used_channels.append(_ch)
                        break 

            if not _select_chrom:
                print(f'No {_type} chrom detected in color usage, generate chromosome with existing data')
                _use_ref_im = False 

                return self._generate_chrom_im_from_data(
                    _data_type=_generate_data_type, 
                    _num_loaded_image=_generate_num_im,
                    _save=_save, _overwrite=_overwrite, _verbose=_verbose
                )

            _chrom_filename = os.path.join(_chrom_fd, self.fov_name)
            
            # load from Dax file
            if hasattr(self, '_ref_im'):
                _drift_ref = getattr(self, '_ref_im')
            else:
                _drift_ref = getattr(self, 'ref_filename')
            print('**',  _infos, _used_channels)

            _chrom_im, _drift, _drift_flag = correct_fov_image(
                _chrom_filename, 
                [_chrom_channel],
                single_im_size=self.shared_parameters['single_im_size'],
                all_channels=_used_channels,
                num_buffer_frames=self.shared_parameters['num_buffer_frames'],
                num_empty_frames=self.shared_parameters['num_empty_frames'],
                drift=None, calculate_drift=_use_ref_im, 
                drift_channel=self.drift_channel,
                ref_filename=_drift_ref,
                correction_folder=self.correction_folder,
                corr_channels=[_ch for _ch in self.shared_parameters['corr_channels'] if _ch in _used_channels],
                warp_image=True,
                illumination_corr=self.shared_parameters['corr_illumination'],
                bleed_corr=False, 
                chromatic_corr=self.shared_parameters['corr_chromatic'], 
                chromatic_ref_channel=self.shared_parameters['ref_channel'],
                z_shift_corr=self.shared_parameters['corr_Z_shift'],
                return_drift=True, verbose=_verbose,
                )

            _chrom_im = _chrom_im[0]
            if _verbose:
                print(f"-- chromosome image has drift: {np.round(_drift, 2)}")
            
            # add chromosome image into attributes
            setattr(self, 'chrom_im', _chrom_im)
            
            # save new chromosome image
            if _save:
                self._save_to_file('fov_info', 
                    _save_attr_list=['chrom_im'],
                    _overwrite=_overwrite,        
                    _verbose=_verbose)
        
        return _chrom_im

    def _generate_chrom_im_from_data(self, _data_type, _num_loaded_image=10, 
                           _fast=True, _save=False,
                           _overwrite=False, _verbose=True):
        """Function to create chromosome image by stacking over all images from certain data_type
        _data_type: type of images to be used to generate chromosome image
        _num_loaded_image: image loaded per round, int
        _fast: whether do it in fast 
        """    
        from .batch_functions import load_image_from_fov_file
        from ..io_tools.load import find_image_background
        ## check inputs
        if _data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"Wrong input data_type: {_data_type}, should be among:{self.shared_parameters['allowed_data_types']}")
        _num_loaded_image = int(_num_loaded_image)
        
        if hasattr(self, 'chrom_im') and not _overwrite:
            if _verbose:
                print(f"- Directly load chrom_im from fov class.")
            _chrom_im = getattr(self, 'chrom_im')
        else:
            # load all IDs
            with h5py.File(self.save_filename, "a", libver='latest') as _f:
                # get the group
                _grp = _f[_data_type]

                _flags = _grp['flags'][:]
                _valid_ids = _grp['ids'][_flags > 0] # only load from processed ids
                _valid_flags = _flags[_flags > 0]
            if _verbose:
                print(f"- Generate chromosome image from {_data_type} images, {len(_valid_ids)} images planned to load.")
                _chrom_time = time.time()
            # start to load images
            _chrom_im = np.zeros(self.shared_parameters['single_im_size'])
            # load images in batch to avoid memory limitations
            for _batch_id in range(int(np.ceil(len(_valid_ids)/_num_loaded_image))):
                # load these iamges in this bat
                _load_ids = _valid_ids[_batch_id*_num_loaded_image:(_batch_id+1)*_num_loaded_image]
                _load_flags = _valid_flags[_batch_id*_num_loaded_image:(_batch_id+1)*_num_loaded_image]
                # load
                _ims, _flags, _drifts = load_image_from_fov_file(self.save_filename, _data_type, _load_ids,
                                                            image_dtype=self.image_dtype, 
                                                            load_drift=True, verbose=_verbose)
                # get only pixel level drift for computational efficiency
                #_shifted_ims = []
                if _verbose:
                    print(f"-- shifting images", end=' ')
                    _shift_time = time.time()
                
                if _fast:
                    _rough_drifts = np.round(_drifts).astype(np.int)
                    for _i, (_im, _flag, _drift) in enumerate(zip(_ims, _flags, _rough_drifts)):
                        # if warpped, directly add
                        if _flag == 2:
                            _chrom_im += _im
                        else:
                            # left limits
                            _llim = np.max([_drift, np.zeros(len(_drift), dtype=np.int)], axis=0)
                            _shift_llim = np.max([-_drift, np.zeros(len(_drift), dtype=np.int)], axis=0)
                            # corresponding right limits
                            _rlim = np.array(np.shape(_im)) - _shift_llim
                            _shift_rlim = _shift_llim + (_rlim - _llim)
                            # generate crops
                            _crops = tuple([slice(_l,_r) for _l,_r in zip(_llim, _rlim)])
                            _shift_crops = tuple([slice(_l,_r) for _l,_r in zip(_shift_llim, _shift_rlim)])
                            #_shift_im = np.ones(np.shape(_chrom_im)) * find_image_background(_im)
                            # crop image and add to shift_im
                            #_shift_im[_shift_crops] = _im[_crops]
                            # append
                            _background = np.median(_im)
                            _chrom_im += _background
                            _chrom_im[_shift_crops] += _im[_crops]-_background
                            #_shifted_ims.append(_shift_im)
                else:
                    from scipy.ndimage.interpolation import shift
                    for _i, (_im, _flag, _drift) in enumerate(zip(_ims, _flags, _drifts)):
                        # if warpped, directly add
                        if _flag == 2: # this image has been warpped
                            _shift_im = _im
                        else:
                            _shift_im = shift(_im, -_drift, order=1, mode='constant', cval=find_image_background(_im))
                        _chrom_im += _shift_im
                        #_shifted_ims.append(_shift_im)
                        
                if _verbose:
                    print(f"in {time.time()-_shift_time:.3f}s. ")

            # add attribute
            setattr(self, 'chrom_im', _chrom_im)

            if _verbose:
                print(f"-- finish generating chrom_im in {time.time()-_chrom_time:.3f}s. ")
            
            if _save:
                self._save_to_file('fov_info', _save_attr_list=['chrom_im'],
                                _verbose=_verbose)
            
        return _chrom_im #, _shifted_ims
    

    


    # alternative batch method to find candidate chromosome @ Shiwei Liu
    # including main features as below:
    # 1. dna/dapi rough mask is used to filtering out the non-cell/non-nucleus region when calculating the intensity distribution of chr signal;
    # 2. initial chr labels are seperated by their voxel size, which are subjected to subsequent binary operations using different parameters.
    # 3. the specified chr/gene id is returned as the dict.key in the output (chrom_coords) in addition to the zyx, which would be used for simutaneous spots assigment to multiple genes.


    # Some notes for adjusting the parameters:
    # Test _chr_seed_size first so majority of single chromosome foci were correctly labeled;
    # Next, increase or decrease _percent_th_3chr and _percent_th_2chr if too much over-splitting or merging of chr seeds
    # If oversplitting happens (especially on relatively condensed small foci) while merging is also frequent, try increase the _min_label_size. 
    # Increase of _min_label_size decrease overspliting though it may lead to some detection loss for small chromosome seeds.

    def _find_all_candidate_chromosomes_in_nucleus (self, 
                                                #_chrom_ims=None, 
                                                _dna_im=None, 
                                                _dna_mask=None,
                                                _chr_ids = [], 
                                                _chr_seed_size = 200,
                                                _percent_th_3chr = 97.5,
                                                _percent_th_2chr = 85, 
                                                _use_percent_chr_area = False,
                                                _fold_3chr = 7,
                                                _fold_2chr = 5,
                                                _std_ratio = 3,
                                                _morphology_size=1, 
                                                _min_label_size=30, 
                                                _random_walk_beta=15, 
                                                _num_threads=4,
                                                _save=True,
                                                _overwrite=False, 
                                                _verbose=True):
        '''Function to find all candidate chromosome centers for input genes given
        Inputs:
            # {(not defined as input anymore; defined by _chr_ids below) _chrom_ims: images where chromosomes are lighted for different genes [num_of_genes * (z,y,x)],
               it could be directly from experimental result, or assembled by stacking over images,
               np.ndarray(shape equal to single_im_size)_}
            _dna_im: DNA/nuclei/cell image that are used for filtering out the non-cell/non-nucleus region
            _dna_mask: if use DNA/cell mask provided from elsewhere, define such mask here
            _chr_id: the id number for the chr (gene) selected; default is empty, which will load all chr saved in gene_ids
            _chr_seed_size: the rough min seed size for the initial seed
            _percent_th_3chr: the percentile th (for voxel size) as indicated for grouping large chr seeds that are likely formed by more than one chr
            _percent_th_3chr: the percentile th (for voxel size) as indicated for grouping large chr seeds that are likely formed by two chr
            _use_percent_chr_area: use percentile or the fold number to estimate large multi-chr foci
            _fold_3chr: the fold of median (single) chr size to be considered as large multi-chr seed
            _fold_2chr: the fold of median (single) chr size to be considered as large dual-chr seed
            _std_ratio: the number of std to be used to find the lighted chromosome
            _morphology_size: the size for erosion/dilation for single chr candidate; 
               this size is adjusted further for erosion/dilation for larger chr seeds that are likely formed by multiple chr candidate
            _min_label_size: size for removal of small objects after binary operations; note that this is typically smaller than what is used in the [find_candidate_chromosomes] function below
            _random_walk_beta: the higher the beta, the more difficult the diffusion is.
            _verbose: say something!, bool (default: True)
        Output:
            cand_chrom_coords_all: dict of arrays of chrom coordinates in zxy pixels + the chr label area used for segmentation for all selected genes (as dict.key))'''

        from skimage import morphology
        #from scipy.stats import scoreatpercentile
        from scipy import ndimage
        from skimage import measure
        from skimage import filters
        if hasattr(self, 'chrom_coords') and not _overwrite:
            if _verbose:
                print(f"+ directly use current chromsome coordinates alternative.")
                return getattr(self, 'chrom_coords')
        elif not _overwrite:
            self._load_from_file('signal', _load_attr_list=['chrom_coords'],
                                _overwrite=_overwrite, _verbose=_verbose, )
            if hasattr(self, 'chrom_coords'):
                if _verbose:
                    print(f"+ use chromsome coordinates alternative from savefile: {os.path.basename(self.save_filename)}.")
                return getattr(self, 'chrom_coords')

        ## 1. assign and load _dna_image if not specified
        if _dna_im is None:
            if hasattr(self, 'dapi_im') and not _overwrite:
                if _verbose:
                    print(f"+ directly use current dapi image.")
                    _dna_im = getattr(self, 'dapi_im')
            else:
                _dna_im = self._load_dapi_image(_dapi_id=0, _overwrite=True, _save=_save)

        ## 2. process each spot images from saved HDF file for all hybs or selected hybs if not specified
        if isinstance (_chr_ids, list) or isinstance (_chr_ids, np.ndarray):
            if _chr_ids == []:
                if hasattr(self, 'gene_ids') and not _overwrite:
                    _chr_ids  = getattr(self, 'gene_ids')
                else:
                    self._load_from_file('gene')
                    _chr_ids  = getattr(self, 'gene_ids')

        # _chr_intensity_th dict to store all coordinates
        _chrom_coords_all = {}

        # load spot_im for one hyb at a time to save memory
        for _chr_id in _chr_ids:
            if _verbose:
                print (f'+ start analyzing the chr/gene {_chr_id}')
            with h5py.File(self.save_filename, "r", libver='latest') as _f:
                _grp = _f['gene']
                _chrom_im = _grp ['ims'][_chr_id-1]
                #_gene_id = _grp ['ids'][_chr_id-1]

                from ..segmentation_tools.chromosome import find_candidate_chromosomes_in_nucleus
                _chrom_coords = find_candidate_chromosomes_in_nucleus (
                _chrom_im, _dna_im = _dna_im, _dna_mask=_dna_mask, _chr_seed_size = _chr_seed_size, _percent_th_3chr =_percent_th_3chr, _percent_th_2chr=_percent_th_2chr, 
                _use_percent_chr_area= _use_percent_chr_area, _fold_3chr=_fold_3chr, _fold_2chr = _fold_2chr, _std_ratio=_std_ratio,_morphology_size=_morphology_size,
                _min_label_size=_min_label_size, _random_walk_beta=_random_walk_beta,_num_threads=_num_threads,_verbose=_verbose)
                _chrom_coords_all [str(_chr_id)] = _chrom_coords

        ## 3. set attributes
        setattr(self, 'chrom_coords', _chrom_coords_all)
        if _save:
            self._save_to_file('signal', _save_attr_list=['chrom_coords'],
                                _overwrite=_overwrite,
                                _verbose=_verbose)
        
        return _chrom_coords_all



    ### Short function to quickly convert the chrom_coord_dict above to a single array
    def _convert_all_chrom_coords_dict_to_array (self, chrom_coords_dict = None, _verbose = True):
    
        '''Function to conver chrom_coords_dict (with arrays) to one single array

            Output: for each chrom center, zxy are the first 3 elements 
              with the 4th element as the chr label
               and the 5th element as the gene id '''
        
        if hasattr(self, 'chrom_coords'):
            if _verbose:
                print(f"+ load and combine current chromsome coordinates alternative.")
            chrom_coords_dict = getattr(self, 'chrom_coords')

        elif not hasattr(self, 'chrom_coords') and isinstance (chrom_coords_dict, dict):
            if _verbose:
                print(f"+ use provided chromsome coordinates alternative.")
        
        else:
            if _verbose:
                print(f"+ no valid chromsome coordinates alternative. generate chrom coords first.")
            return None

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


    def _find_candidate_chromosomes_by_segmentation(self, 
                                                _chrom_im=None, 
                                                _adjust_layers=False, 
                                                _filt_size=4, 
                                                _binary_per_th=99.5,
                                                _morphology_size=1, 
                                                _min_label_size=50,
                                                _random_walk_beta=10,
                                                _save=True,
                                                _overwrite=False,
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
        if hasattr(self, 'cand_chrom_coords') and not _overwrite:
            if _verbose:
                print(f"+ directly use current chromsome coordinates.")
            return getattr(self, 'cand_chrom_coords')
        elif not _overwrite:
            self._load_from_file('fov_info', _load_attr_list=['cand_chrom_coords'],
                                _overwrite=_overwrite, _verbose=_verbose, )
            if hasattr(self, 'cand_chrom_coords'):
                if _verbose:
                    print(f"+ use chromsome coordinates from savefile: {os.path.basename(self.save_filename)}.")
                return getattr(self, 'cand_chrom_coords')
        
        ## 0. load or generate chromosome
        if _chrom_im is None:
            if hasattr(self, 'chrom_im'):
                _chrom_im = getattr(self, 'chrom_im')
            else:
                _chrom_im = self._load_chromosome_image(_verbose=_verbose)
        from ..segmentation_tools.chromosome import find_candidate_chromosomes
        _chrom_coords = find_candidate_chromosomes(
            _chrom_im,
            _adjust_layers=_adjust_layers, 
            _filt_size=_filt_size, 
            _binary_per_th=_binary_per_th,
            _morphology_size=_morphology_size, 
            _min_label_size=_min_label_size,
            _random_walk_beta=_random_walk_beta,
            _num_threads=self.num_threads,
            _verbose=_verbose,
        )
        # set attributes
        setattr(self, 'cand_chrom_coords', _chrom_coords)
        if _save:
            self._save_to_file('fov_info', _save_attr_list=['cand_chrom_coords'],
                                _overwrite=_overwrite,
                                _verbose=_verbose)
        
        return _chrom_coords




    ### Class function to estimate the spot_th for picking candidate spots from fitted spots, whose intensity are background substracted @ Shiwei Liu
    ### Use spot/region ids as prioritzied input to select image to process.

    ### Note: if region ids starts from 0, the corresponding image would be 0-1 = -1, which is the last hyb image. 
    def _find_itensity_th_for_selected_spots_in_nucleus(self, _region_ids = [], 
                               _dna_im = None, 
                               _dna_mask = None, 
                               _std_ratio = 3, 
                               _return_signal_and_background = False, 
                               _verbose = True, 
                               _parallel = False,  
                               _num_threads=4, 
                               _save=True, 
                               _overwrite=False):

        """Function to estimate spot intensity and spot background (excluding non-cell regions) given
           Input:
             _region_ids: the spot region ids used for selecting spot images, for example from each hyb
             _dna_im: dna image or cell boundary image to exlcude non-nuclear/non-cell area
             _dna_mask: if use provided dna mask [one z-slice]
             _std_ratio: the number_of_std to be applied to find the spot th
             _return_signal_and_background: if False, return the background substracted signal
             _verbose: bool; say sth
           Output:
             spot_intensity_th_dict: background substracted spot th for spot picking for selected regions/spots"""


        from skimage import morphology
        #from scipy.stats import scoreatpercentile
        #from scipy import ndimage
        from skimage import filters
        if hasattr(self, 'spot_intensity_th') and not _overwrite:
            if _verbose:
                print(f"+ directly use current spot intensity thresholds.")
                return getattr(self, 'spot_intensity_th')
        elif not _overwrite:
            self._load_from_file('signal', _load_attr_list=['spot_intensity_th'],
                                _overwrite=_overwrite, _verbose=_verbose, )
            if hasattr(self, 'spot_intensity_th'):
                if _verbose:
                    print(f"+ use spot intensity thresholds from savefile: {os.path.basename(self.save_filename)}.")
                return getattr(self, 'spot_intensity_th')

        ## 1. assign and load _dna_image if not specified
        if _dna_im is None:
            if hasattr(self, 'dapi_im') and not _overwrite:
                if _verbose:
                    print(f"+ directly use current dapi image.")
                    _dna_im = getattr(self, 'dapi_im')
            else:
                _dna_im = self._load_dapi_image(_dapi_id=0, _overwrite=True, _save=_save)

        ## 2. process each spot images from saved HDF file for all hybs or selected hybs if not specified
        if isinstance (_region_ids, list) or isinstance (_region_ids, np.ndarray):
            if _region_ids == []:
                if hasattr(self, 'combo_ids') and not _overwrite:
                    _region_ids = getattr(self, 'combo_ids')
                else:
                    self._load_from_file('combo')
                    _region_ids = getattr(self, 'combo_ids')
        
        # _spot_intensity_th dict to store all estimates
        _spot_intensity_th = {}
        
        # determining if multiprocessing or single processing
        if not self.parallel:  # prioritize shared parameter
            _parallel = False
        

        if _parallel == False:  # slow but less memory usage
                # load spot_im for one hyb at a time to save memory
            for _region_id in _region_ids:
                if _verbose:
                    print(f"+ estimate intensity threshold for region {_region_id}.")
                    if _region_id == 0:
                        print ("note that the spot image with index of '-1' is loaded for region_id == 0")
                with h5py.File(self.save_filename, "r", libver='latest') as _f:
                    _grp = _f['combo']
                    _spot_im = _grp ['ims'][_region_id-1]
                    #_combo_id = _grp ['ids'][_region_id-1]

                    from ..spot_tools.picking import find_spot_intensity_th_and_background_in_nucleus
                    _spot_intensity_th_each = find_spot_intensity_th_and_background_in_nucleus (_spot_im = _spot_im, _dna_im = _dna_im, _dna_mask =_dna_mask, _std_ratio = _std_ratio, 
                    _return_signal_and_background =_return_signal_and_background, _verbose = _verbose)

                    _spot_intensity_th [str(_region_id)] = _spot_intensity_th_each
        
        ########## NOT FINISHED below ########## 
        if _parallel == True:  # fast but more memory usage
            if len(_region_ids) > 0:

                import multiprocessing as mp
                from ..spot_tools.picking import find_spot_intensity_th_and_background_in_nucleus

                _spot_im_kwargs = [{'_spot_im_filename': self.save_filename,'_dna_im': _dna_im,'_spot_id': _region_id} for _region_id in _region_ids]

                with mp.Pool(_num_threads,) as _spot_ims_pool:
                    if _verbose:
                        print(f"- Start multiprocessing estimates spot intensity th with {_num_threads} threads", end=' ')
                        _multi_time = time.time()
                    # Multi-proessing!
                    _spot_intensity_th = _spot_ims_pool.starmap(find_spot_intensity_th_and_background_in_nucleus, _spot_im_kwargs, chunksize=1)
                    # close multiprocessing
                    _spot_ims_pool.close()
                    _spot_ims_pool.join()
                    _spot_ims_pool.terminate()
                    if _verbose:
                        print(f"in {time.time()-_multi_time:.3f}s.")
        ##########  NOT FINSIHED above ##########      


        ## 3. set attributes
        setattr(self, 'spot_intensity_th', _spot_intensity_th)
        if _save:
            self._save_to_file('signal', _save_attr_list=['spot_intensity_th'],
                                _overwrite=_overwrite,
                                _verbose=_verbose)
        
        return _spot_intensity_th



    def _select_chromosome_by_candidate_spots(self, 
                                            _spot_type='unique',
                                            _cand_chrom_coords=None,
                                            _good_chr_loss_th=0.4,
                                            _cand_spot_intensity_th=0.5, 
                                            _save=False, _overwrite=False,
                                            _verbose=True):
        """Function to select"""
        from ..segmentation_tools.chromosome import select_candidate_chromosomes
        
        # check if already exists
        if hasattr(self, 'chrom_coords') and not _overwrite:
            if _verbose:
                print(f"+ directly use current chromsome coordinates.")
            return getattr(self, 'chrom_coords')
        elif not _overwrite:
            self._load_from_file('fov_info', _load_attr_list=['chrom_coords'],
                                _overwrite=_overwrite, _verbose=_verbose, )
            if hasattr(self, 'chrom_coords'):
                if _verbose:
                    print(f"+ use chromsome coordinates from savefile: {os.path.basename(self.save_filename)}.")
                return getattr(self, 'chrom_coords')
                
        if _spot_type not in self.shared_parameters['allowed_data_types']:
            raise KeyError(f"Wrong input _spot_type:{_spot_type}, should be within {list(self.shared_parameters['allowed_data_types'].keys())}")
        # load spots
        _spot_attr = f"{_spot_type}_spots_list"
        # try to load
        if not hasattr(self, _spot_attr):
            self._load_from_file(_spot_type, 
                                _load_image=False, 
                                _load_processed=True, 
                                _verbose=_verbose)
            if not hasattr(self, _spot_attr):
                raise AttributeError(f"fov class doesn't have {_spot_attr}.")
        # extract
        _spots_list = getattr(self, _spot_attr)
        
        # chrom_coords
        if _cand_chrom_coords is None:
            if not hasattr(self, 'cand_chrom_coords'):
                self._load_from_file('fov_info', 
                                    _load_attr_list=['cand_chrom_coords'],
                                    _verbose=_verbose)
                if not hasattr(self, 'cand_chrom_coords'):
                    raise AttributeError(f"fov class doesn't have cand_chrom_coords.")
            # extract
            _cand_chrom_coords = list(getattr(self, 'cand_chrom_coords'))
        else:
            _cand_chrom_coords = list(_cand_chrom_coords)
        
        # calcualte
        _chrom_coords = select_candidate_chromosomes(_cand_chrom_coords, 
                                    _spots_list,
                                    _cand_spot_intensity_th=_cand_spot_intensity_th,
                                    _good_chr_loss_th=_good_chr_loss_th,
                                    _verbose=_verbose,
                                    )
        
        # set attributes
        setattr(self, 'chrom_coords', _chrom_coords)
        if _save:
            self._save_to_file('fov_info', _save_attr_list=['chrom_coords'],
                                _overwrite=_overwrite,
                                _verbose=_verbose)

        return _chrom_coords

    ## load DAPI image
    def _load_dapi_image(self, 
                         _dapi_id=None,
                         _save=True,
                         _overwrite=False, _verbose=True):
        """Function to load dapi image for fov class"""
        
        if 'correct_fov_image' not in locals():
            from ..io_tools.load import correct_fov_image
        
        if hasattr(self, 'dapi_im') and not _overwrite:
            if _verbose:
                print(f"directly return existing attribute.")
            _dapi_im = getattr(self, 'dapi_im')
        else:
            _use_ref_im = False

            # find DAPI in color_usage
            _dapi_fds = []
            for _fd, _infos in self.color_dic.items():
                if len(_infos) >= len(self.channels) and _infos[self.dapi_channel_index] == 'DAPI':
                    _full_fd = [_a_fd for _a_fd in self.annotated_folders if os.path.basename(_a_fd)==_fd]
                    if len(_full_fd) == 1:
                        _dapi_fds.append(_full_fd[0])
            
            if len(_dapi_fds) == 0:
                print('No DAPI detected in color usage, exit!')
                return None
            else:
                # choose dapi folder
                if len(_dapi_fds) == 1 or (len(_dapi_fds) > 1 and _dapi_id is None): 
                    _dapi_fd = _dapi_fds[0]
                elif isinstance(_dapi_id, int) or isinstance(_dapi_id, np.int):
                    _dapi_fd = _dapi_fds[_dapi_id]
                else:
                    raise TypeError(f"Wrong input type: {type(_dapi_id)} for _dapi_id.")
                if _verbose:
                    print(f"-- choose dapi images from folder: {_dapi_fd}.")
                # decide whether use extra reference id
                _dapi_fd_ind = list(self.annotated_folders).index((_dapi_fd))
                if _dapi_fd_ind != self.ref_id:                                
                    if not hasattr(self, '_ref_im'):
                        self._load_reference_image(_verbose=_verbose)
                    _use_ref_im = True
                else:
                    _use_ref_im = False 
                # get used_channels for this dapi folder:
                _info = self.color_dic[os.path.basename(_dapi_fd)]
                _used_channels = []
                for _mk, _ch in zip(_info, self.channels):
                    if _mk.lower() == 'null':
                        continue
                    else:
                        _used_channels.append(_ch)


            # assemble filename
            _dapi_filename = os.path.join(_dapi_fd, self.fov_name)
            _dapi_channel = self.dapi_channel

            # load from Dax file
            if hasattr(self, '_ref_im'):
                _drift_ref = getattr(self, '_ref_im', None)
            else:
                _drift_ref = getattr(self, 'ref_filename', None)

            # load
            _dapi_im = correct_fov_image(
                _dapi_filename, 
                [_dapi_channel],
                single_im_size=self.shared_parameters['single_im_size'],
                all_channels=_used_channels,
                num_buffer_frames=self.shared_parameters['num_buffer_frames'],
                num_empty_frames=self.shared_parameters['num_empty_frames'],
                drift=None, calculate_drift=_use_ref_im,
                drift_channel=self.drift_channel,
                ref_filename=_drift_ref,
                correction_folder=self.correction_folder,
                corr_channels=self.shared_parameters['corr_channels'],
                warp_image=True,
                illumination_corr=self.shared_parameters['corr_illumination'],
                bleed_corr=False, 
                chromatic_corr=False, 
                chromatic_ref_channel=self.shared_parameters['ref_channel'],
                z_shift_corr=self.shared_parameters['corr_Z_shift'],
                verbose=_verbose,
                )[0][0]
            setattr(self, 'dapi_im', _dapi_im)
            
            # save new chromosome image
            if _save:
                self._save_to_file('fov_info', _save_attr_list=['dapi_im'],
                    _overwrite=_overwrite,
                    _verbose=_verbose)
        
        return _dapi_im

    ## load bead image, for checking purposes
    def _load_bead_image(self, _bead_id, _drift=None,
                        _warp=True, 
                        _overwrite=False, _verbose=True):
        """Function to load bead image for fov class
        
        """

        if 'correct_fov_image' not in locals():
            from ..io_tools.load import correct_fov_image

        if isinstance(_bead_id, int) or isinstance(_bead_id, np.int):
            _ind = int(_bead_id)
        elif isinstance(_bead_id, str):
            for _i, _fd in enumerate(self.annotated_folders):
                if _bead_id in _fd:
                    _ind = _i
                    break
        _bead_folder = self.annotated_folders[_ind]
        _bead_filename = os.path.join(_bead_folder, self.fov_name)
        _drift_channel = self.drift_channel
        # get used_channels for this dapi folder:
        _info = self.color_dic[os.path.basename(_bead_folder)]
        _used_channels = []
        for _mk, _ch in zip(_info, self.channels):
            if _mk.lower() == 'null':
                continue
            else:
                _used_channels.append(_ch)

        # load from Dax file
        if hasattr(self, '_ref_im'):
            _drift_ref = getattr(self, '_ref_im')
        else:
            _drift_ref = getattr(self, 'ref_filename')
        # load this beads image
        _bead_im = correct_fov_image(
            _bead_filename, 
            [_drift_channel],
            single_im_size=self.shared_parameters['single_im_size'],
            all_channels=_used_channels,
            num_buffer_frames=self.shared_parameters['num_buffer_frames'],
            num_empty_frames=self.shared_parameters['num_empty_frames'],
            drift=_drift, calculate_drift=False,
            drift_channel=_drift_channel,
            ref_filename=_drift_ref,
            correction_folder=self.correction_folder,
            warp_image=_warp,
            illumination_corr=self.shared_parameters['corr_illumination'],
            bleed_corr=False, 
            chromatic_corr=False, 
            chromatic_ref_channel=self.shared_parameters['ref_channel'],
            z_shift_corr=self.shared_parameters['corr_Z_shift'],
            verbose=_verbose,
            )[0][0]

        return _bead_im

    def _load_bead_ims_for_bits(self, bit_ids, data_type, force_loading=True, parallel=True):
        """"""
        # check whether data_type is valid
        if data_type not in self.shared_parameters['allowed_data_types']:
            raise ValueError(f"data_type should be within {self.shared_parameters['allowed_data_types'].keys()}" )
        # load required data for this data_type
        self._load_from_file(data_type, _overwrite=force_loading)
        # load the reference for this data_type
        if hasattr(self, f'{data_type}_ref_im'):
            _drift_ref = getattr(self, f'{data_type}_ref_im')
        elif hasattr(self, f'{data_type}_ref_id'):
            _drift_ref = os.path.join(self.annotated_folders[getattr(self, f'{data_type}_ref_id')], 
                                    self.fov_name)

        if isinstance(bit_ids, int) or isinstance(bit_ids, np.int32):
            bit_ids = [bit_ids]
        else:
            bit_ids = list(bit_ids)
        
        _process_args = []
        
        for _bit_id in bit_ids:
            # check whether bit_id is valid
            if _bit_id not in getattr(self, f"{data_type}_ids"):
                raise ValueError(f"bit_id: {_bit_id} doesn't exist in {data_type}_ids" )
            elif getattr(self, f"{data_type}_flags")[np.where(getattr(self, f"{data_type}_ids")==_bit_id)[0][0]] == 0:
                raise ValueError(f"bit_id: {_bit_id} for data_type: {data_type} hasn't been processed." )

            _bit_ind = np.where(getattr(self, f"{data_type}_ids")==_bit_id)[0][0]
            _bit_drift = getattr(self, f"{data_type}_drifts")[_bit_ind]
            # find the corresponding folder
            _fd = find_bit_folder(_bit_id, 
                                  data_type, 
                                  self.color_dic, 
                                  self.shared_parameters['allowed_data_types'])
            #print(_bit_drift, _bit_ind, _fd)
            # get used channels for this bit
            _infos = self.color_dic[_fd]
            _used_channels, _used_corr_channel = [], []
            for _mk, _ch in zip(_infos, self.channels):
                if _mk.lower() == 'null':
                    continue
                else:
                    _used_channels.append(_ch)
                    if _ch in self.shared_parameters['corr_channels']:
                        _used_corr_channel.append(_ch)
                    
            _fd_full = [_f for _f in self.annotated_folders if os.path.basename(_f)==_fd]
            if len(_fd_full) != 1:
                raise ValueError(f"non-unique data folder detected for bit-{_bit_id} type-{data_type}")
            _fd_full = _fd_full[0]
            _filename = os.path.join(_fd_full, self.fov_name)
            
            # generate processing arg
            _arg = (
                _filename,
                [self.drift_channel],
                None,
                self.shared_parameters['single_im_size'],
                _used_channels,
                self.shared_parameters['num_buffer_frames'],
                self.shared_parameters['num_empty_frames'],
                _bit_drift,
                False,
                self.drift_channel,
                _drift_ref,
                True,
                {},
                _used_corr_channel,
                self.correction_folder,
                True,
                False,
                4,
                False,
                self.shared_parameters['corr_illumination'],
                self.correction_profiles['illumination'],
                False,
                None,
                self.shared_parameters['ref_channel'],
                False,
                None,
                False,
                5,
                2,
                False,
                self.image_dtype,
                False,
                True,  
            )
            _process_args.append(_arg)
        from ..io_tools.load import correct_fov_image
        if parallel:
            print(f"- Start multiprocessing process {len(_process_args)} bead_ims with {self.num_threads} threads", end=' ')
            _multi_time = time.time()
            with mp.Pool(self.num_threads,) as _bead_im_pool:
                _bead_ims = _bead_im_pool.starmap(correct_fov_image, _process_args, chunksize=1)
                _bead_ims = [_im[0][0] for _im in _bead_ims]
                # close multiprocessing
                _bead_im_pool.close()
                _bead_im_pool.join()
                _bead_im_pool.terminate()
            # final time
            print(f"in {time.time()-_multi_time:.3f}s.")
        else:
            print(f"- Start process {len(_process_args)} bead_ims")
            _seq_time = time.time()
            _bead_ims = [correct_fov_image(*_arg)[0][0] for _arg in _process_args]
            # final time
            print(f"- finish in {time.time()-_seq_time:.3f}s.")
            
        return _bead_ims


# add test comment
def find_bit_folder(bit_id, data_type, color_dict, allowed_type_dict):
    _feature_ref = f"{allowed_type_dict[data_type]}{bit_id}"
    #print(_feature_ref)
    for _fd, _infos in color_dict.items():
        for _info in _infos:
            if _info == _feature_ref:
                return _fd
    
    # return empty if not found
    return ""


            