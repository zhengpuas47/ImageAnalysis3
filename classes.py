import sys,glob,os,time, copy
import numpy as np
sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')
import pickle as pickle
import matplotlib.pyplot as plt
from . import get_img_info, corrections, visual_tools, analysis

from scipy import ndimage
from scipy import stats
from skimage import morphology
from skimage.segmentation import random_walker

class Cell_List():
    """
    Class Cell_List:
    this is a typical data structure of cells within one chromosome with images in multiple independent color-channels and decoding-groups.

    """
    # initialize
    def __init__(self, parameters, _chosen_fovs=[], _exclude_fovs=[], _load_all_attr=False):
        if not isinstance(parameters, dict):
            raise TypeError('wrong input type of parameters, should be a dictionary containing essential info.');

        ## required parameters: data folder (list)
        if isinstance(parameters['data_folder'], list):
            self.data_folder = [str(_fd) for _fd in parameters['data_folder']]
        else:
            self.data_folder = [str(parameters['data_folder'])];

        ## extract hybe folders and field-of-view names
        self.folders = []
        for _fd in self.data_folder:
            _hyb_fds, _fovs = get_img_info.get_folders(_fd, feature='H', verbose=True)
            self.folders += _hyb_fds;
            self.fovs = _fovs;

        ## analysis_folder, segmentation_folder, save_folder, correction_folder,map_folder
        if 'analysis_folder' not in parameters:
            self.analysis_folder = self.data_folder[0]+os.sep+'Analysis'
        else:
            self.analysis_folder = str(parameters['analysis_folder']);
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder'];
        else:
            self.segmentation_folder = self.analysis_folder+os.sep+'segmentation'
        if 'save_folder' in parameters:
            self.save_folder = parameters['save_folder'];
        else:
            self.save_folder = self.analysis_folder+os.sep+'5x10';
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

        # number of num_threads
        if 'num_threads' in parameters:
            self.num_threads = parameters['num_threads'];
        else:
            self.num_threads = int(os.cpu_count() / 3); # default: use one third of cpus.

        ## if loading all remaining attr in parameter
        if _load_all_attr:
            for _key, _value in parameters.items():
                if not hasattr(self, _key):
                    setattr(self, _key, _value);

        ## list to store Cell_data
        self.cells = [];

        ## chosen field of views
        if len(_chosen_fovs) == 0: # no specification
            _chosen_fovs = np.arange(len(_fovs));
        if len(_chosen_fovs) > 0: # there are specifications
            _chosen_fovs = [_i for _i in _chosen_fovs if _i <= len(_fovs)];
            _chosen_fovs = list(np.array(np.unique(_chosen_fovs), dtype=np.int));
        # exclude fovs
        if len(_exclude_fovs) > 0: #exclude any fov:
            for _i in _exclude_fovs:
                if _i in _chosen_fovs:
                    _chosen_fovs.pop(_chosen_fovs.index(_i))
        # save values to the class
        self.fov_ids = _chosen_fovs;
        self.chosen_fovs = list(np.array(self.fovs)[np.array(self.fov_ids, dtype=np.int)]);

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

    ## Load segmentations info

    def _load_cell_segmentations(self, _num_threads=None, _allow_manual=True,
                            _shape_ratio_threshold=0.041, _signal_cap_ratio=0.2, _denoise_window=5,
                            _load_in_ram=True, _save=True, _force=False,
                            _cell_coord_fl='cell_coords.pkl', _verbose=True):
        ## load segmentation
        # check attributes
        if not hasattr(self, 'channels'):
            self._load_color_info()
        if not _num_threads:
            if not hasattr(self, 'num_threads'):
                raise AttributeError('No num_threads given in funtion kwds and class attributes');
            else:
                _num_threads = self.num_threads;
        if _verbose:
            print(f"{len(self.chosen_fovs)} of field-of-views are selected to load segmentation.");
        # do segmentation if necessary, or just load existing segmentation file
        _segmentation_labels, _dapi_ims  = analysis.Segmentation_All(self.analysis_folder,
                                        self.folders, self.chosen_fovs, ref_name='H0R0',
                                        num_threads=_num_threads,
                                        shape_ratio_threshold=_shape_ratio_threshold,
                                        signal_cap_ratio=_signal_cap_ratio,
                                        denoise_window=_denoise_window,
                                        num_channel=len(self.channels),
                                        dapi_channel=self.dapi_channel_index,
                                        correction_folder=self.correction_folder,
                                        segmentation_path=os.path.basename(self.segmentation_folder),
                                        save=_save, force=_force)
        _dapi_ims = [corrections.Remove_Hot_Pixels(_im) for _im in _dapi_ims];
        _dapi_ims = [corrections.Z_Shift_Correction(_im) for _im in _dapi_ims];

        ## pick(exclude) cells from previous result
        if _allow_manual:
            # generate coordinates
            _coord_list, _index_list = [],[];
            for _i, _label in enumerate(_segmentation_labels):
                for _j in range(np.max(_label)-1):
                    _center = np.round(ndimage.measurements.center_of_mass(_label==_j+1));
                    _center = list(np.flipud(_center));
                    _center.append(_dapi_ims[0].shape[0]/2)
                    _coord_list.append(_center)
                    _index_list.append(_i);
            # wrap into a dic
            _cell_coord_dic = {'coords': _coord_list,
                          'class_ids': _index_list,
                          'pfits':{},
                          'dec_text':{},
                          };
            self.cell_coord_dic = copy.deepcopy(_cell_coord_dic);
            # use visual tools to pick
            _cell_coord_savefile = self.save_folder + os.sep + _cell_coord_fl;

            _cell_viewer = visual_tools.imshow_mark_3d_v2(_dapi_ims, image_names=self.chosen_fovs,
                                                          save_file=_cell_coord_savefile,
                                                          given_dic=_cell_coord_dic);

            return _cell_viewer
        else:
            return _segmentation_labels, _dapi_ims

    def _update_cell_segmentations(self, _cell_coord_fl='cell_coords.pkl',
                                  _overwrite_segmentation=True,
                                  _marker_displace_th = 900,
                                  _verbose=True):
        """Function to update cell segmentation info from saved file,
            - usually do this after automatic segmentation"""
        _cell_coord_savefile = self.save_folder + os.sep + _cell_coord_fl;
        if not os.path.exists(_cell_coord_savefile):
            raise IOError(f'{_cell_coord_savefile} doesnot exist, exit')
            return False
        with open(_cell_coord_savefile, 'rb') as handle:
            _new_cell_coord_dic = pickle.load(handle);
        # parse
        _ccd = visual_tools.partition_map(self.cell_coord_dic['coords'], self.cell_coord_dic['class_ids'])
        _new_ccd = visual_tools.partition_map(_new_cell_coord_dic['coords'], _new_cell_coord_dic['class_ids'])

        # initialize
        _new_seg_labels, _dapi_ims = [], [];
        _remove_cts = [];
        for _i, (_cell_coords, _new_cell_coords) in enumerate(zip(_ccd, _new_ccd)):
            # now we are taking care of one specific field of view
            if _verbose:
                print(f"-- fov-{_i}, match manually picked cell with sgementation ")
            # load fov image
            _seg_file = self.segmentation_folder+os.sep+self.chosen_fovs[_i].replace('.dax', '_segmentation.pkl');
            _seg_label, _dapi_im = pickle.load(open(_seg_file, 'rb'));
            _remove = 0;
            if not _overwrite_segmentation:
                # save original seg label into another file
                _old_seg_file = _seg_file.replace('_segmentation.pkl', '_segmentation_old.pkl');
                pickle.dump([_seg_label, _dapi_im], open(_old_seg_file, 'wb'));

            for _l, _coord in enumerate(_cell_coords):
                _dist = [np.sum((_c-_coord)**2) for _c in _new_cell_coords];
                _match = [_d < _marker_displace_th for _d in _dist]
                if sum(_match) == 0:
                    _seg_label[_seg_label==_l+1-_remove] = -1;
                    _seg_label[_seg_label >_l+1-_remove] -= 1;
                    _remove += 1;
            if _verbose:
                print(f"--- {_remove} label(s) got removed!");
            _new_seg_labels.append(_seg_label)
            _dapi_ims.append(_dapi_im)
            _remove_cts.append(_remove);
            # save
            if _verbose:
                print(f"--- save updated segmentation to {os.path.basename(_seg_file)}");
            pickle.dump([_seg_label, _dapi_im], open(_seg_file, 'wb'))

        return _new_seg_labels, _dapi_ims, _remove_cts

    def _load_drifts(self, _num_threads=None, _size=500, _force=False, _dynamic=True, _verbose=True):

        # load color usage if not given
        if not hasattr(self, 'channels') or not hasattr(self, 'bead_channel_index'):
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



    def _create_cell(self, _fov_id, _cell_id, _load_in_list=True,
                     _load_info=True, _load_segmentation=True, _load_drift=True):
        _param = {'fov_id': _fov_id,
                  'cell_id': _cell_id,
                  'data_folder': self.data_folder,
                  'analysis_folder':self.analysis_folder,
                  'segmentation_folder': self.segmentation_folder,
                  'save_folder': self.save_folder,
                  'correction_folder': self.correction_folder,
                  'drift_folder': self.drift_folder,
                  'map_folder': self.map_folder,
                  }
        _cell = Cell_Data(_param, _load_all_attr=True)
        if _load_info:
            _cell._load_color_info();
            _cell._load_encoding_scheme();
        if _load_segmentation:
            _cell._load_segmentation();
        if _load_drift:
            _cell._load_drift();
        # whether store
        if _load_in_list:
            self.cells.append(_cell);
        return _cell

    def _create_cells_in_fov(self, _num_threads=None):
        if not _num_threads:
            _num_threads = int(self.num_threads);
        pass

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
    def _load_segmentation(self, _shape_ratio_threshold=0.030, _signal_cap_ratio=0.15, _denoise_window=5,
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
                                                illumination_corr=True,
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

    def _load_drift(self, _size=500, _force=False, _dynamic=True, _verbose=True):
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

    def _load_images(self, _type, _extend_dim=20, _load_in_ram=True, _load_annotated_only=True,
                     _illumination_correction=True, _chromatic_correction=True,
                     _save=False, _overwrite=False, _verbose=False):
        """Core function to load images, support different types:"""
        if not hasattr(self, 'segmentation_label') and _type != 'beads':
            self._load_segmentation();
        if not hasattr(self, 'channels'):
            self._load_color_info();
        # load images if not pre_loaded:
        if not hasattr(self, 'splitted_ims'):
            if _verbose:
                print("- Starting loading images")
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
            if _verbose:
                print(f"-- split into {_num_ch} channels")
            _splitted_ims = get_img_info.split_channels_by_image(_ims, _names,
                                        num_channel=_num_ch, DAPI=self.use_dapi);
            # do all corrections
            if _verbose:
                print(f"-- do all corrections")
            for _hyb_fd, _hyb_ims in _splitted_ims.items():
                for _i, (_channel, _hyb_im) in enumerate(zip(self.channels[:len(_hyb_ims)],_hyb_ims) ):
                    # correct for z axis shift
                    _hyb_im = corrections.Z_Shift_Correction(_hyb_im)
                    # correct for hot pixels
                    _hyb_im = corrections.Remove_Hot_Pixels(_hyb_im)
                    if _illumination_correction:
                        _hyb_im = corrections.Illumination_correction(_hyb_im, correction_channel=_channel,
                                    correction_folder=self.correction_folder, verbose=False)[0]
                    if _chromatic_correction:
                        _hyb_im = corrections.Chromatic_abbrevation_correction(_hyb_im, correction_channel=_channel,
                                    correction_folder=self.correction_folder, verbose=False)[0]
                    # replace original image
                    _splitted_ims[_hyb_fd][_i] = _hyb_im;
        else:
            _splitted_ims = self.splitted_ims;
        if str(_type).lower() == 'raw':
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
                            _uid = int(_channel_info.split(_unique_marker)[-1])
                            if _verbose:
                                print(f"-- loading unique region {_uid} at {_hyb_fd} and color {self.channels[_i]} ")
                            _channel = str(self.channels[_i]);
                            # cropping
                            _cropped_im = visual_tools.crop_cell(_corr_im, self.segmentation_label,
                                                                 drift=self.drift[_img_name])[0]

                            _unique_ims.append(_cropped_im)
                            _unique_ids.append(int(_channel_info.split(_unique_marker)[-1]))
                            _unique_channels.append(_channel);

                else:
                    raise IOError('-- missing image:',_img_name);

            if _load_in_ram:
                self.unique_ims = _unique_ims;
                self.unique_ids = _unique_ids;
                self.unique_channels = _unique_channels

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
                            _raw_im = _splitted_ims[_img_name][_channel_idx];
                            # cropping
                            _cropped_im = visual_tools.crop_cell(_raw_im, self.segmentation_label,
                                                                 drift=self.drift[_img_name])[0]
                            # store this image
                            _combo_images.append(_cropped_im);
                    # create a combo group
                    _group = Encoding_Group(_combo_images, _hyb_fds, _matrix, self.save_folder,
                                            self.fov_id, self.cell_id, _channel, _group_id);
                    _combo_groups.append(_group);


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

        if _type=='all' or type=='cell_info':
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
                if hasattr(_group, 'readouts'):
                    _combo_dic['readouts'] = np.array(_group.readouts);
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
                'channels': np.array(self.unique_channels)
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

        if _type == 'all' or _type == 'raw_combo':
            if not hasattr(self, 'combo_groups'):
                self.combo_groups = [];
            elif not isinstance(self.combo_groups, list):
                raise TypeError('Wrong datatype for combo_groups attribute for cell data.');

            # load existing combo files
            _raw_combo_fl = "rounds.npz"
            _combo_files = glob.glob(os.path.join(_save_folder, "group-*", "channel-*", _raw_combo_fl));
            for _combo_file in _combo_files:
                if _verbose:
                    print("-- loading combo from file:", _combo_file)
                with np.load(_combo_file) as handle:
                    _ims = list(handle['observation']);
                    _matrix = handle['encoding']
                    _names = handle['names'];
                    if 'readout' in handle:
                        _readouts = handle['readouts']
                _name_info = _combo_file.split(os.sep);
                _fov_id = [int(_i.split('-')[-1]) for _i in _name_info if "fov" in _i][0]
                _cell_id = [int(_i.split('-')[-1]) for _i in _name_info if "cell" in _i][0]
                _group_id = [int(_i.split('-')[-1]) for _i in _name_info if "group" in _i][0]
                _color = [_i.split('-')[-1] for _i in _name_info if "channel" in _i][0]
                # check duplication
                _check_duplicate = [(_g.fov_id==_fov_id) and (_g.cell_id==_cell_id) and (_g.group_id==_group_id) and (_g.color==_color) for _g in self.combo_groups];
                if sum(_check_duplicate) > 0: #duplicate found:
                    if not _overwrite:
                        if _verbose:
                            print("---", _combo_file.split(_save_folder)[-1], "already exists in combo_groups, skip")
                        continue;
                    else:
                        self.combo_groups.pop(_check_duplicate.index(True));
                # create new group
                if '_readouts' in locals():
                    _group = Encoding_Group(_ims, _names, _matrix, _save_folder,
                                            _fov_id, _cell_id, _color, _group_id, _readouts)
                else:
                    _group = Encoding_Group(_ims, _names, _matrix, _save_folder,
                                            _fov_id, _cell_id, _color, _group_id)
                # append
                self.combo_groups.append(_group);

        if _type == 'all' or _type == 'raw_unique':
            _unique_fl = 'unique_rounds.npz'
            _unique_savefile = _save_folder + os.sep + _unique_fl;
            if not os.path.exists(_unique_savefile):
                print(f"- savefile {_unique_savefile} not exists, exit.")
                return False
            if _verbose:
                print("- Loading unique from file:", _unique_savefile);
            with np.load(_unique_savefile) as handle:
                _unique_ims = list(handle['observation']);
                _unique_ids = list(handle['ids']);
                _unique_channels = list(handle['channels']);
            # save
            if not hasattr(self, 'unique_ids') or not hasattr(self, 'unique_ims'):
                self.unique_ids, self.unique_ims, self.unique_channels = [], [], [];
            for _uim, _uid, _channel in zip(_unique_ims, _unique_ids, _unique_channels):
                if int(_uid) not in self.unique_ids:
                    if _verbose:
                        print(f"loading image with unique_id: {_uid}")
                    self.unique_ids.append(_uid)
                    self.unique_ims.append(_uim)
                    self.unique_channels.append(_channel)
                elif int(_uid) in self.unique_ids and _overwrite:
                    if _verbose:
                        print(f"overwriting image with unique_id: {_uid}")
                    self.unique_ims[self.unique_ids.index(int(_uid))] = _uim;

    def _generate_chromosome_image(self, _source='combo', _max_count= 30, _verbose=False):
        """Generate chromosome from existing combo / unique images"""
        if _source.lower() != 'combo' and _source.lower() != 'unique':
            raise ValueError('wrong source key given, should be combo or unique. ');
        if _source == 'combo':
            if not hasattr(self, 'combo_groups'):
                raise AttributeError('cell_data doesnot have combo_groups.');
            # sum up existing Images
            _image_count = 0;
            _chrom_im = np.zeros(np.shape(self.combo_groups[0].ims[0]));
            for _group in self.combo_groups:
                _chrom_im += sum(_group.ims)
                _image_count += len(_group.ims)
                if _max_count > 0 and _image_count > _max_count:
                    break;
            _chrom_im = _chrom_im / _image_count

        elif _source == 'unique':
            if not hasattr(self, 'unique_ims'):
                raise AttributeError('cell_data doesnot have unique images');
            # sum up existing Images
            _image_count = 0;
            _chrom_im = np.zeros(np.shape(self.unique_ims[0]));
            for _im in self.unique_ims:
                _chrom_im += _im
                _image_count += 1
                if _max_count > 0 and _image_count > _max_count:
                    break;
            _chrom_im = _chrom_im / _image_count

        # final correction
        _chrom_im = corrections.Z_Shift_Correction(_chrom_im);
        _chrom_im = corrections.Remove_Hot_Pixels(_chrom_im);
        self.chrom_im = _chrom_im;

        return _chrom_im;

    def _identify_chromosomes(self, _gaussian_size=2, _cap_percentile=1, _seed_dim=3,
                              _th_percentile=99.5, _min_obj_size=125, _verbose=True):
        """Function to identify chromsome automatically first"""
        if not hasattr(self, 'chrom_im'):
            self._generate_chromosome_image();
        _chrom_im = np.zeros(np.shape(self.chrom_im), dtype=np.uint8) + self.chrom_im;
        if _gaussian_size:
            # gaussian filter
            _chrom_im = ndimage.filters.gaussian_filter(_chrom_im, _gaussian_size)
            # normalization
            _limit = stats.scoreatpercentile(_chrom_im, (_cap_percentile, 100.-_cap_percentile)).astype(np.float)
            _chrom_im = (_chrom_im-np.min(_limit))/(np.max(_limit)-np.min(_limit))
            # max filter - min filter
            _max_ft = ndimage.filters.maximum_filter(_chrom_im, _seed_dim)
            _min_ft = ndimage.filters.minimum_filter(_chrom_im, _seed_dim)
            _seed_im = 2*_max_ft - _min_ft
            # binarilize
            _binary_im = (_seed_im > stats.scoreatpercentile(_seed_im, _th_percentile))
            # dialation and erosion
            _binary_im = ndimage.binary_dilation(_binary_im, morphology.ball(1))
            _binary_im = ndimage.binary_erosion(_binary_im, morphology.ball(0))
            _binary_im = ndimage.binary_fill_holes(_binary_im, structure=morphology.ball(2))
            # find objects
            _open_objects = morphology.opening(_binary_im, morphology.ball(0))
            _close_objects = morphology.closing(_open_objects, morphology.ball(1))
            _label, _num = ndimage.label(_close_objects);
            _label[_label==0] = -1;
            # segmentation
            _seg_label = random_walker(_chrom_im, _label, beta=100, mode='bf')
            # keep object
            _kept_label = -1 * np.ones(_seg_label.shape, dtype=np.int);
            _sizes = [np.sum(_seg_label==_j+1) for _j in range(np.max(_seg_label))]
            # re-label
            _label_ct = 1;
            for _i, _size in enumerate(_sizes):
                if _size > _min_obj_size: # then save this label
                    _kept_label[_seg_label == _i+1] = _label_ct
                    _label_ct += 1;
            _chrom_coords = [ndimage.measurements.center_of_mass(_kept_label==_j+1) for _j in range(np.max(_kept_label))]
            # store
            self.chrom_segmentation = _kept_label;
            self.chrom_coords = _chrom_coords;

        return _kept_label, _chrom_coords

    def _pick_chromosome_manual(self, _save_folder=None, _save_fl='chrom_coord.pkl'):
        if not _save_folder:
            if hasattr(self, 'save_folder'):
                _save_folder = self.save_folder;
            else:
                raise ValueError('save_folder not given in keys and attributes.')
        _chrom_savefile = _save_folder + os.sep + _save_fl;
        if not hasattr(self, 'chrom_coords'):
            raise ValueError("chromosome coordinates doesnot exist in attributes.")
        _coord_dic = {'coords': [np.flipud(_coord) for _coord in self.chrom_coords],
                      'class_ids': list(np.zeros(len(self.chrom_coords),dtype=np.uint8)),
                      'pfits':{},
                      'dec_text':{},
                      };
        #pickle.dump(_coord_dic, open(_chrom_savefile, 'wb'));
        _viewer = visual_tools.imshow_mark_3d_v2([self.chrom_im], image_names=['chromosome'], save_file=_chrom_savefile, given_dic=_coord_dic)
        return _viewer

    def _update_chromosome_from_file(self, _save_folder=None, _save_fl='chrom_coord.pkl', _verbose=True):
        if not _save_folder:
            if hasattr(self, 'save_folder'):
                _save_folder = self.save_folder;
            else:
                raise ValueError('save_folder not given in keys and attributes.')
        _chrom_savefile = _save_folder + os.sep + _save_fl;
        _coord_dic = pickle.load(open(_chrom_savefile, 'rb'));
        _chrom_coords = [np.flipud(_coord) for _coord in _coord_dic['coords']];
        if _verbose:
            print(f"-- {len(_chrom_coords)} loaded")
        self.chrom_coords = _chrom_coords;

        return _chrom_coords

class Encoding_Group():
    """defined class for each group of encoded images"""
    def __init__(self, ims, hybe_names, encoding_matrix, save_folder,
                 fov_id, cell_id, color, group_id, readouts=None):
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
        if readouts:
            self.readouts = readouts;
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
        if hasattr(self, 'readouts'):
            _combo_dic['readouts'] = np.array(self.readouts);
        # save
        if _verbose:
            print("-- saving combo to:", _combo_savefile)
        np.savez_compressed(_combo_savefile, **_combo_dic)
        return True
