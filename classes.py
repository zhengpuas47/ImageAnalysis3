import sys,glob,os,time
import numpy as np
sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')
import pickle as pickle
import matplotlib.pyplot as plt
from . import get_img_info, corrections, visual_tools, analysis

class Cell_List():
    """
    Class Cell_List:
    this is a typical data structure of cells within one chromosome with images in multiple independent color-channels and decoding-groups.

    """
    # initialize
    def __init__(self, data_folder):
        self.data_folder = data_folder;
        self.cells = [];
        self.index = len(self.cells);

        # segmentation_folder, save_folder, correction_folder,map_folder
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters[segmentation_folder];
        else:
            self.segmentation_folder = self.data_folder+os.sep+'Analysis'+os.sep+'segmentation'
        if 'save_folder' in parameters:
            self.save_folder = parameters[save_folder];
        else:
            self.save_folder = self.data_folder+os.sep+'Analysis'+os.sep+'raw'
        if 'correction_folder' in parameters:
            self.correction_folder = parameters[correction_folder];
        else:
            self.correction_folder = self.data_folder
        if 'map_folder' in parameters:
            self.map_folder = parameters[map_folder];
        else:
            self.map_folder = self.data_folder+os.sep+'Analysis'+os.sep+'distmap'

    # allow print info of Cell_List
    def __str__(self):
        if hasattr(self, 'data_folder'):
            print("Data folder:", self.data_folder);
        return 'test'

    # allow iteration of Cell_List
    def __iter__(self):
        return self

    def __next__(self):
        if not hasattr(self, 'cells') or not not hasattr(self, 'index'):
            raise StopIteration
        elif self.index == 0:
            raise StopIteration
        else:
            self.index -= 1;
        return self.cells[self.index]

    ## Load basic info
    def _load_color_info(self, _color_filename='Color_Usage', _color_format='csv', _save_color_dic=True):
        _color_dic, _use_dapi, _channels = get_img_info.Load_Color_Usage(self.data_folder,
                                                            color_filename=_color_filename,
                                                            color_format=_color_format,
                                                            return_color=True)
        # need-based store color_dic
        if _save_color_dic:
            self.color_dic = _color_dic;
        # store other info
        self.use_dapi = _use_dapi;
        self.channels = [str(ch) for ch in _channels]
        # channel for beads
        _bead_channel = get_img_info.find_bead_channel(_color_dic);
        self.bead_channel_index = _bead_channel
        _dapi_channel = get_img_info.find_dapi_channel(_color_dic);
        self.dapi_channel_index = _dapi_channel

        return _color_dic
    def _load_encoding_scheme(self, _encoding_filename='Encoding_Scheme', _encoding_format='csv', _save_encoding_scheme=True):
        _encoding_scheme, _num_hyb, _num_reg, _en_colors, _en_groups = get_img_info.Load_Encoding_Scheme(self.data_folder,
                                                                                                encoding_filename=_encoding_filename,
                                                                                                encoding_format=_encoding_format,
                                                                                                return_info=True)
        # need-based encoding scheme saving
        if _save_encoding_scheme:
            self.encoding_scheme = _encoding_scheme;
        # save other info
        self.hyb_per_group = _num_hyb;
        self.reg_per_group = _num_reg;
        self.encoding_colors = _en_colors;
        self.encoding_group_nums = _en_groups;

        return _encoding_scheme


class Cell_Data():
    """
    Class Cell_data:
    data structure of each cell with images in multiple independent color-channels and decoding-groups.
    initialization of cell_data requires:
    """
    # initialize
    def __init__(self, parameters, _load_all_attr=False):
        if not isinstance(parameters, dict):
            raise TypeError('wrong input type of parameters, should be a dictionary containing essential info.');
        # necessary parameters
        # data folder (list)
        if isinstance(parameters['data_folder'], list):
            self.data_folder = [str(_fd) for _fd in parameters['data_folder']]
        else:
            self.data_folder = [str(parameters['data_folder'])];
        # analysis folder
        if 'analysis_folder' not in parameters:
            self.analysis_folder = self.data_folder[0]+os.sep+'Analysis'
        else:
            self.analysis_folder = str(parameters['analysis_folder']);
        # extract hybe folders and field-of-view names
        self.folders = []
        for _fd in self.data_folder:
            _hyb_fds, self.fovs = get_img_info.get_folders(_fd, feature='H', verbose=True)
            self.folders += _hyb_fds;
        # fov id and cell id given
        self.fov_id = int(parameters['fov_id'])
        self.cell_id = int(parameters['cell_id'])
        # segmentation_folder, save_folder, correction_folder,map_folder
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder'];
        else:
            self.segmentation_folder = self.analysis_folder+os.sep+'segmentation'
        if 'save_folder' in parameters:
            self.save_folder = parameters['save_folder'];
        else:
            self.save_folder = self.analysis_folder+os.sep+'5x10'+os.sep+'fov-'+str(self.fov_id)+os.sep+'cell-'+str(self.cell_id);
        if 'correction_folder' in parameters:
            self.correction_folder = parameters['correction_folder'];
        else:
            self.correction_folder = self.data_folder
        if 'drift_folder' in parameters:
            self.drift_folder = parameters['drift_folder'];
        else:
            self.drift_folder =  self.analysis_folder+os.sep+'drift'
        if 'map_folder' in parameters:
            self.map_folder = parameters['map_folder'];
        else:
            self.map_folder = self.analysis_folder+os.sep+'distmap'
        # if loading all remaining attr in parameter
        if _load_all_attr:
            for _key, _value in parameters.items():
                if not hasattr(self, _key):
                    setattr(self, _key, _value);

    # allow print info of Cell_List
    def __str__(self):
        if hasattr(self, 'data_folder'):
            print("Cell Data from folder(s):", self.data_folder);
        if hasattr(self, 'analysis_folder'):
            print("\t path for anaylsis result:", self.analysis_folder);
        if hasattr(self, 'fov_id'):
            print("\t from field-of-view:", self.fov_id);
        if hasattr(self, 'cell_id'):
            print("\t with cell_id:", self.cell_id);

        return 'test'

    # allow iteration of Cell_List
    def __iter__(self):
        return self
    def __next__(self):
        return self

    ## Load basic info
    def _load_color_info(self, _color_filename='Color_Usage', _color_format='csv', _save_color_dic=True):
        _color_dic, _use_dapi, _channels = get_img_info.Load_Color_Usage(self.analysis_folder,
                                                            color_filename=_color_filename,
                                                            color_format=_color_format,
                                                            return_color=True)
        # need-based store color_dic
        if _save_color_dic:
            self.color_dic = _color_dic;
        # store other info
        self.use_dapi = _use_dapi;
        self.channels = [str(ch) for ch in _channels]
        # channel for beads
        _bead_channel = get_img_info.find_bead_channel(_color_dic);
        self.bead_channel_index = _bead_channel
        _dapi_channel = get_img_info.find_dapi_channel(_color_dic);
        self.dapi_channel_index = _dapi_channel

        return _color_dic
    def _load_encoding_scheme(self, _encoding_filename='Encoding_Scheme', _encoding_format='csv', _save_encoding_scheme=True):
        _encoding_scheme, self.hyb_per_group, self.reg_per_group, \
        self.encoding_colors, self.encoding_group_nums \
            = get_img_info.Load_Encoding_Scheme(self.analysis_folder,
                                                   encoding_filename=_encoding_filename,
                                                   encoding_format=_encoding_format,
                                                   return_info=True)
        # need-based encoding scheme saving
        if _save_encoding_scheme:
            self.encoding_scheme = _encoding_scheme;

        return _encoding_scheme

    ## load cell specific info
    def _load_segmentation(self, _shape_ratio_threshold=0.041, _signal_cap_ratio=0.2, _denoise_window=5,
                           _load_in_ram=True, _save=True, _force=False):
        # check attributes
        if not hasattr(self, 'channels'):
            self._load_color_info()

        # do segmentation if necessary, or just load existing segmentation file
        fov_segmentation_label, fov_dapi_im  = analysis.Segmentation_Fov(self.analysis_folder,
                                                self.folders, self.fovs, self.fov_id,
                                                shape_ratio_threshold=_shape_ratio_threshold,
                                                signal_cap_ratio=_signal_cap_ratio,
                                                denoise_window=_denoise_window,
                                                num_channel=len(self.channels),
                                                dapi_channel=self.dapi_channel_index,
                                                correction_folder=self.correction_folder,
                                                segmentation_path=os.path.basename(self.segmentation_folder),
                                                save=_save, force=_force)
        # exclude special cases
        if not hasattr(self, 'cell_id'):
            raise AttributeError('no cell_id attribute for this cell_data class object!');
        elif self.cell_id+1 not in np.unique(fov_segmentation_label):
            raise ValueError('segmentation label doesnot contain this cell id:', self.cell_id);
        # if everything works, keep segementation_albel and dapi_im for this cell
        else:
            _seg_label = - np.ones(fov_segmentation_label.shape);
            _seg_label[fov_segmentation_label==self.cell_id+1] = 1;
            _dapi_im = visual_tools.crop_cell(fov_dapi_im, _seg_label, drift=None)[0]
            # correct for z axis shift
            _dapi_im = corrections.Z_Shift_Correction(_dapi_im)
            # correct for hot pixels
            _dapi_im = corrections.Remove_Hot_Pixels(_dapi_im)
            if _load_in_ram:
                self.segmentation_label = _seg_label
                self.dapi_im = _dapi_im

        return _seg_label, _dapi_im

    def _load_drift(self, _size=450, _force=False, _dynamic=True, _verbose=True):
        # load color usage if not given
        if not hasattr(self, 'channels'):
            self._load_color_info();
        # scan for candidate drift correction files
        _dft_file_cand = glob.glob(self.drift_folder+os.sep+self.fovs[self.fov_id].replace(".dax", "*.pkl"))
        # if one unique drift file was found:
        if len(_dft_file_cand) == 1:
            _dft = pickle.load(open(_dft_file_cand[0], 'rb'));
            # check drift length
            if len(_dft) == len(self.color_dic):
                if _verbose:
                    print("- length matches, directly load from existing file:", _dft_file_cand[0]);
                self.drift = _dft;
                return self.drift
            else:
                if _verbose:
                    print("- length doesn't match, proceed to do drift correction.");

        if len(_dft_file_cand) == 0:
            if _verbose:
                print("- no drift result found in drift folder", self.drift_folder)
        # do drift correction from scratch
        if not hasattr(self, 'bead_ims'):
            if _verbose:
                print("- Loading beads images for drift correction")
            _bead_ims, _bead_names = self._load_images('beads', _load_in_ram=False);
        else:
             _bead_ims, _bead_names = self.bead_ims, self.bead_names
        # do drift correction
        self.drift, _, _failed_count = corrections.STD_beaddrift_sequential(_bead_ims, _bead_names,
                                                                    self.drift_folder,
                                                                    self.fovs, self.fov_id,
                                                                    sz_ex=_size, force=_force, dynamic=_dynamic)
        if _failed_count > 0:
            print('Failed drift noticed! total failure:', _failed_count);

        return self.drift

    def _load_images(self, _type, _extend_dim=20, _load_in_ram=True, _save_cropped_ims=False,
                     _load_annotated_only=True, _illumination_correction=True, _chromatic_correction=True,
                     _save=False, _overwrite=False, _verbose=False):
        """Core function to load images, support different types:"""
        if not hasattr(self, 'segmentation_label') and _type != 'beads':
            self._load_segmentation();
        if not hasattr(self, 'channels'):
            self._load_color_info();
        # load images if not pre_loaded:
        if not hasattr(self, 'splitted_ims'):
            if _load_annotated_only:
                _annotated_folders = [_fd for _fd in self.folders if os.path.basename(_fd) in self.color_dic];
                self.annotated_folders = _annotated_folders
                _ims, _names = get_img_info.get_img_fov(_annotated_folders, self.fovs, self.fov_id, verbose=False);
            else:
                _ims, _names = get_img_info.get_img_fov(self.folders, self.fovs, self.fov_id, verbose=False);

            if '405' in self.channels:
                _num_ch = len(self.channels) -1;
            else:
                _num_ch = len(self.channels);
            # split images
            _splitted_ims = get_img_info.split_channels_by_image(_ims, _names,
                                        num_channel=_num_ch, DAPI=self.use_dapi);
        else:
            _splitted_ims = self.splitted_ims;
        if str(_type).lower() == 'all':
            # Load all splitted images
            if _load_in_ram:
                self.splitted_ims = _splitted_ims;
            return _splitted_ims
        # if the purpose is to load bead images:
        if str(_type).lower() == "beads":
            # check color usage
            if not hasattr(self, 'color_dic'):
                self._load_color_info();
            _drift=False # no need to correct for beads because you are using it to correct
            _bead_ims = [];
            _bead_names = [];
            # load the images in the order of color_dic (which corresponding to experiment order)
            for _hyb_fd, _info in self.color_dic.items():
                _img_name = _hyb_fd + os.sep + self.fovs[self.fov_id];
                if _img_name in _splitted_ims:
                    _bead_im = _splitted_ims[_img_name][self.bead_channel_index]
                    # correct for z axis shift
                    _bead_im = corrections.Z_Shift_Correction(_bead_im)
                    # correct for hot pixels
                    _bead_im = corrections.Remove_Hot_Pixels(_bead_im)
                    # append
                    _bead_ims.append(_bead_im)
                    _bead_names.append(_img_name);
                else:
                    raise IOError('-- missing image:',_img_name);
            if _load_in_ram:
                self.bead_ims = _bead_ims
                self.bead_names = _bead_names
            return _bead_ims, _bead_names

        # load unique images
        elif str(_type).lower() == 'unique':
            # check attributes of color dic
            if not hasattr(self, 'color_dic'):
                self._load_color_info();
            # check drift info
            if not hasattr(self, 'drift'):
                 self._load_drift();
            # initialize
            _unique_ims = [];
            _unique_ids = [];
            _unique_channels = [];
            _unique_marker = 'u';
            # load the images in the order of color_dic (which corresponding to experiment order)
            for _hyb_fd, _info in self.color_dic.items():
                _img_name = _hyb_fd + os.sep + self.fovs[self.fov_id];
                if _img_name in _splitted_ims:
                    if len(_info) != len(_splitted_ims[_img_name]):
                        raise IndexError('information from color_usage doesnot match splitted images.')
                    for _i, (_channel_info, _channel_im) in enumerate(zip(_info, _splitted_ims[_img_name])):
                        if _unique_marker in _channel_info:
                            _channel = str(self.channels[_i]);
                            # correct for z axis shift
                            _corr_im = corrections.Z_Shift_Correction(_channel_im)
                            # correct for hot pixels
                            _corr_im = corrections.Remove_Hot_Pixels(_corr_im)
                            if _illumination_correction:
                                _corr_im = corrections.Illumination_correction(_corr_im, correction_channel=_channel,
                                            correction_folder=self.correction_folder, verbose=_verbose)[0]
                            if _chromatic_correction:
                                _corr_im = corrections.Chromatic_abbrevation_correction(_corr_im, correction_channel=_channel,
                                            correction_folder=self.correction_folder, verbose=_verbose)[0]
                            # cropping
                            _cropped_im = visual_tools.crop_cell(_corr_im, self.segmentation_label,
                                                                 drift=self.drift[_img_name])[0]

                            _unique_ims.append(_cropped_im)

                else:
                    raise IOError('-- missing image:',_img_name);

            if _load_in_ram:
                self.unique_ims = _unique_ims;
                self.unique_ids = _unique_ids;

            if _save:
                self._save_to_file('unique', _overwrite=_overwrite);

            return _unique_ims, _unique_ids

        elif str(_type).lower() == 'combo' or str(_source).lower() == 'sparse' :
            # check color usage
            if not hasattr(self, 'color_dic'):
                self._load_color_info();
            # check encoding scheme
            if not hasattr(self, 'encoding_scheme'):
                self._load_encoding_scheme();
            # check drift info
            if not hasattr(self, 'drift'):
                 self._load_drift();

            # initialize
            _combo_groups = []; # list to store encoding groups
            _combo_marker = 'c';
            # load the images in the order of color_dic (which corresponding to experiment order)
            for _channel, _encoding_info in self.encoding_scheme.items():
                if _verbose:
                    print("- Loading combo images in channel", _channel);
                # loop through groups in each color
                for _group_id, (_hyb_fds, _matrix) in enumerate(zip(_encoding_info['names'],_encoding_info['matrices'])):
                    if _verbose:
                        print("-- loading images for group:", _hyb_fds)
                    _combo_images = [];
                    for _hyb_fd in _hyb_fds:
                        # check whether this matches color_usage
                        if _hyb_fd not in self.color_dic:
                            raise ValueError('encoding scheme and color usage doesnot match, error in folder:', _hyb_fd)
                        # load images
                        _img_name = _hyb_fd + os.sep + self.fovs[self.fov_id];
                        if _img_name in _splitted_ims:
                            if len(self.color_dic[_hyb_fd]) != len(_splitted_ims[_img_name]):
                                raise IndexError('information from color_usage doesnot match splitted images in combo loading.')
                            # get index
                            _channel_idx = self.channels.index(_channel);
                            # check whether it is marked as combo
                            if _combo_marker not in self.color_dic[_hyb_fd][_channel_idx]:
                                raise ValueError('this', _hyb_fd, "does not correspond to combo in channel", _channel);
                            # get raw image
                            _raw_img = _splitted_ims[_img_name][_channel_idx];
                            # do all following processing:
                            # correct for z axis shift
                            _corr_im = corrections.Z_Shift_Correction(_raw_img, verbose=_verbose)
                            # correct for hot pixels
                            _corr_im = corrections.Remove_Hot_Pixels(_corr_im)
                            # illumination correction and chromatic correction
                            if _illumination_correction:
                                _corr_im = corrections.Illumination_correction(_corr_im, correction_channel=_channel,
                                            correction_folder=self.correction_folder, verbose=_verbose)[0]
                            if _chromatic_correction:
                                _corr_im = corrections.Chromatic_abbrevation_correction(_corr_im, correction_channel=_channel,
                                            correction_folder=self.correction_folder, verbose=_verbose)[0]
                            # cropping
                            _cropped_im = visual_tools.crop_cell(_corr_im, self.segmentation_label,
                                                                 drift=self.drift[_img_name])[0]
                            # store this image
                            _combo_images.append(_cropped_im);
                    # create a combo group
                    _group = Encoding_Group(_combo_images, _hyb_fds, _matrix, self.save_folder,
                                            self.fov_id, self.cell_id, _channel, _group_id);
                    _combo_groups.append(_group);
                    if _group_id >= 2:
                        break;

            if _load_in_ram:
                self.combo_groups = _combo_groups;

            if _save:
                self._save_to_file('combo', _overwrite=_overwrite)

            return _combo_groups;

        # exception: wrong _type key given
        else:
            raise ValueError('wrong image loading type given!, key given:', _type)
        return False

    # saving
    def _save_to_file(self, _type='all', _save_folder=None, _overwrite=False, _verbose=True):
        # special keys don't save in the 'cell_info' file. They are images!
        _special_keys = ['unique_ims', 'combo_groups','bead_ims', 'splitted_ims']
        if _save_folder == None:
            _save_folder = self.save_folder;
        if not os.path.exists(_save_folder):
            os.makedirs(_save_folder);
        # save file full name
        _savefile = _save_folder + os.sep + 'cell_info.pkl';
        if _verbose:
            print("- Save cell_info:")
        if os.path.isfile(_savefile) and not _overwrite:
            if _verbose:
                print("-- loading existing info from file:", _savefile);
            _save_dic = pickle.load(open(_savefile, 'rb'));
        else: # no existing file:
            _save_dic = {}; # create an empty dic

        if _type=='all' or type=='cell_info':
            _existing_attributes = [_attr for _attr in dir(self) if not _attr.startswith('_') and _attr not in _special_keys];
            # store updated information
            _updated_info = [];
            for _attr in _existing_attributes:
                if _attr not in _special_keys:
                    if _attr not in _save_dic:
                        _save_dic[_attr] = getattr(self, _attr);
                        _updated_info.append(_attr);

        if _verbose and len(_updated_info) > 0:
            print("-- information updated in cell_info.pkl:", _updated_info);
        # save info to all
        with open(_savefile, 'wb') as output_handle:
            if _verbose:
                print("- Writing cell data to file:", _savefile);
            pickle.dump(_save_dic, output_handle);
        # save combo
        if (_type =='all' or _type == 'combo') and hasattr(self, 'combo_groups'):
            for _group in self.combo_groups:
                _combo_savefolder = os.path.join(_save_folder,
                                                'group-'+str(_group.group_id),
                                                'channel-'+str(_group.color)
                                                )
                if not os.path.exists(_combo_savefolder):
                    os.makedirs(_combo_savefolder);
                _combo_savefile = _combo_savefolder+os.sep+'rounds.npz';
                # if file exists and not overwriting, skip
                if os.path.exists(_combo_savefile) and not _overwrite:
                    if _verbose:
                        print("file {:s} already exists, skip.".format(_combo_savefile));
                    continue
                # else, write
                _attrs = [_attr for _attr in dir(_group) if not _attr.startswith('_')];
                _combo_dic = {
                    'observation': np.concatenate([_im[np.newaxis,:] for _im in _group.ims]),
                    'encoding': _group.matrix,
                    'names': np.array(_group.names),
                }
                # save
                if _verbose:
                    print("-- saving combo to:", _combo_savefile)
                np.savez_compressed(_combo_savefile, **_combo_dic)
        # save unique
        if (_type == 'all' or _type == 'unique') and hasattr(self, 'unique_ims'):
            _unique_savefile = _save_folder + os.sep + 'unique_rounds.npz';
            _unique_dic = {
                'observation': np.concatenate([_im[np.newaxis,:] for _im in self.unique_ims]),
                'ids': np.array(self.unique_ids),
            }
            # save
            if _verbose:
                print("-- saving unique to:", _unique_savefile)
            np.savez_compressed(_unique_savefile, **_unique_dic)

    def _load_from_file(self, _type='all', _save_folder=None, _load_attrs=[],
                        _overwrite=False, _verbose=True):
        if not _save_folder and not hasattr(self, 'save_folder'):
            raise ValueError('Save folder info not given!');
        elif not _save_folder:
            _save_folder = self.save_folder;

        if _type == 'all' or _type == 'cell_info':
            _infofile = _save_folder + os.sep + 'cell_info.pkl';
            _info_dic = pickle.load(open(_infofile, 'rb'));
            #loading keys from info_dic
            for _key, _value in _info_dic.items():
                if not hasattr(self, _key) or _overwrite:
                    # if (load_attrs) specified:
                    if len(_load_attrs) > 0 and _key in _load_attrs:
                        setattr(self, _key, _value);
                    # no load_attr specified
                    elif len(_load_attrs) == 0:
                        setattr(self, _key, _value);

        if _type == 'all' or _type == 'combo':
            pass





class Encoding_Group():
    """defined class for each group of encoded images"""
    def __init__(self, ims, hybe_names, encoding_matrix, save_folder,
                 fov_id, cell_id, color, group_id, readout_list=None):
        # info for this cell
        self.ims = ims;
        self.names = hybe_names;
        self.matrix = encoding_matrix;
        self.save_folder = save_folder;
        # detailed info for this group
        self.fov_id = fov_id;
        self.cell_id = cell_id;
        self.color = color;
        self.group_id = group_id;
        if readout_list:
            self.readouts = readout_list;
    def _save_group(self, _overwrite=False):
        _combo_savefolder = os.path.join(self.save_folder,
                                        'group-'+str(self.group_id),
                                        'channel-'+str(self.color)
                                        )
        if not os.path.exists(_combo_savefolder):
            os.makedirs(_combo_savefolder);
        _combo_savefile = _combo_savefolder+os.sep+'rounds.npz';
        # if file exists and not overwriting, skip
        if os.path.exists(_combo_savefile) and not _overwrite:
            if _verbose:
                print("file {:s} already exists, skip.".format(_combo_savefile));
            return False
        # else, write
        _attrs = [_attr for _attr in dir(_group) if not _attr.startswith('_')];
        _combo_dic = {
            'observation': np.concatenate([_im[np.newaxis,:] for _im in self.ims]),
            'encoding': self.matrix,
            'names': np.array(self.names),
        }
        # save
        if _verbose:
            print("-- saving combo to:", _combo_savefile)
        np.savez_compressed(_combo_savefile, **_combo_dic)
        return True
