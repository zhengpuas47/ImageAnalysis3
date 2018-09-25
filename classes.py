import sys,glob,os
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

        # segmentation_folder, pickle_folder, correction_folder,map_folder
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters[segmentation_folder];
        else:
            self.segmentation_folder = self.data_folder+os.sep+'Analysis'+os.sep+'segmentation'
        if 'pickle_folder' in parameters:
            self.pickle_folder = parameters[pickle_folder];
        else:
            self.pickle_folder = self.data_folder+os.sep+'Analysis'+os.sep+'raw'
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
    # imports
    import os, sys, glob, time
    import ImageAnalysis3 as ia
    # initialize
    def __init__(self, parameters):
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
        # segmentation_folder, pickle_folder, correction_folder,map_folder
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder'];
        else:
            self.segmentation_folder = self.analysis_folder+os.sep+'segmentation'
        if 'pickle_folder' in parameters:
            self.pickle_folder = parameters['pickle_folder'];
        else:
            self.pickle_folder = self.analysis_folder+os.sep+'raw'
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
    def _load_segmentation(self, _shape_ratio_threshold=0.041, _signal_cap_ratio=0.2, _denoise_window=5):
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
                                                save=False, force=False)
        # exclude special cases
        if not hasattr(self, 'cell_id'):
            raise AttributeError('no cell_id attribute for this cell_data class object!');
        elif self.cell_id+1 not in np.unique(fov_segmentation_label):
            raise ValueError('segmentation label doesnot contain this cell id:', self.cell_id);
        # if everything works, keep segementation_albel and dapi_im for this cell
        else:
            _seg_label = - np.ones(fov_segmentation_label.shape);
            _seg_label[fov_segmentation_label==self.cell_id+1] = 1;
            self.segmentation_label = _seg_label
            self.dapi_im = visual_tools.crop_cell(fov_dapi_im, self.segmentation_label, drift=None)[0]

        return self.segmentation_label, self.dapi_im

    def _load_drift(self, _size=450, _force=False, _dynamic=True):
        if not hasattr(self, 'bead_ims'):
            print("images for beads not loaded yet!")
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

    def _load_images(self, _type, _extend_dim=20, _load_in_ram=True, _load_annotated_only=True):
        if not hasattr(self, 'segmentation_label'):
            self._load_segmentation();
        if not hasattr(self, 'channels'):
            self._load_color_info();
        # load images
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
                    _bead_ims.append(_splitted_ims[_img_name][self.bead_channel_index])
                    _bead_names.append(_img_name);
                else:
                    raise IOError('-- missing image:',_img_name);
            if _load_in_ram:
                self.bead_ims = _bead_ims
                self.bead_names = _bead_names

            return _bead_ims, _bead_names

        elif str(_type).lower() == 'unique':
            # check attributes of color dic
            if not hasattr(self, 'color_dic'):
                self._load_color_info();
            # initialize
            _unique_ims = [];
            _unique_ids = [];
            _unique_colors = [];
            _unique_marker = 'u';
            # load the images in the order of color_dic (which corresponding to experiment order)
            for _hyb_fd, _info in self.color_dic.items():
                _img_name = _hyb_fd + os.sep + self.fovs[self.fov_id];
                if _img_name in _splitted_ims:
                    if len(_info) != len(_splitted_ims[_img_name]):
                        raise IndexError('information from color_usage doesnot match splitted images.')
                    for _i, (_channel_info, _channel_im) in enumerate(zip(_info, _splitted_ims[_img_name])):
                        if _unique_marker in _channel_info:
                            _unique_ims.append()

                else:
                    raise IOError('-- missing image:',_img_name);
            if _load_in_ram:
                self.unique_ims = _unique_ims;
                self.unique_ids = _unique_ids;

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
            _cropped_ims = [];
            for _name, _sp_ims in _splitted_ims.items():
                _cropped_ims.append(visual_tools.crop_cell(_sp_ims[self.bead_channel_index],
                                        self.segmentation_label, extend_dim=_extend_dim)[0])
            self.bead_ims = _cropped_ims
        else:
            pass

class Encoding_Group():
    """defined class for each group of encoded images"""
    def __init__(self, ims, hybe_names, encoding_matrix, readout_list=None, save_folder=None, color=None):
        self.ims = ims;
        self.names = hybe_names;
        self.matrix = encoding_matrix;
        if readouts:
            self.readouts = readout_list;
        if save_folder:
            self.save_folder = save_folder;
        if color:
            self.color = color;
        return True
