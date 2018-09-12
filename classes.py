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
        self.data_folder = str(parameters['data_folder'])
        # extract hybe folders and field-of-view names
        self.folders, self.fovs = get_img_info.get_folders(self.data_folder, feature='H', verbose=True)
        # fov id and cell id given
        self.fov_id = int(parameters['fov_id'])
        self.cell_id = int(parameters['cell_id'])
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
            print("Cell Data from folder:", self.data_folder);
        if hasattr(self, 'fov_id'):
            print("\t from field-of-view:", self.fov_id);
        if hasattr(self, 'cell_id'):
            print("\t with cell_id:", self.cell_id);

        return 'test'

    # allow iteration of Cell_List
    def __iter__(self):
        return self

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
        _encoding_scheme, self.hyb_per_group, self.reg_per_group, \
        self.encoding_colors, self.encoding_group_nums \
            = get_img_info.Load_Encoding_Scheme(self.data_folder,
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
        fov_segmentation_label, fov_dapi_im  = analysis.Segmentation_Fov(self.data_folder,
                                                self.folders, self.fovs, self.fov_id,
                                                shape_ratio_threshold=_shape_ratio_threshold,
                                                signal_cap_ratio=_signal_cap_ratio,
                                                denoise_window=_denoise_window,
                                                num_channel=len(self.channels),
                                                dapi_channel=self.dapi_channel_index,
                                                correction_folder=self.correction_folder,
                                                segmentation_path=self.segmentation_folder.split(self.data_folder+os.sep)[1],
                                                save=False, force=False)
        # exclude special cases
        if not hasattr(self, 'cell_id'):
            raise AttributeError('no cell_id attribute for this cell_data class object!');
        elif self.cell_id+1 not in np.unique(fov_segmentation_label):
            raise ValueError('segmentation label doesnot contain this cell id:', self.cell_id);
        # if everything works, keep segementation_albel and dapi_im for this cell
        else:
            _seg_label = - np.ones(fov_segmentation_label.shape);
            _seg_label[fov_segmentation_label==self.cell_id+1] = self.cell_id+1;
            self.segmentation_label = _seg_label
            self.dapi_im = visual_tools.crop_cell(fov_dapi_im, self.segmentation_label, drift=None)[0]

        return self.segmentation_label, self.dapi_im

    def _load_drift(self,):
        pass



    def _load_images(self, _type, _drift=False ):
        if not hasattr(self, 'segmentation_label'):
            self._load_segmentation();
        if not hasattr(self, 'channels'):
            self._load_color_info();
        _ims, _names = get_img_info.get_img_fov(self.folders, self.fovs, self.fov_id, verbose=False);
        if '405' in self.channels:
            _num_ch = len(self.channels) -1;
        else:
            _num_ch = len(self.channels);
        _splitted_ims = get_img_info.split_channels_by_image(_ims, _names,
                                    num_channel=_num_ch, DAPI=self.use_dapi);
        if str(_type).lower() == "beads":
            _drift=False # no need to correct for beads because you are using it to correct
            return _splitted_ims
        elif str(_type).lower() == 'unique':
            # check attributes
            if not hasattr(self, 'channels'):
                self._load_color_info();
        elif str(_type).lower() == 'combo' or str(_source).lower() == 'sparse' :
            # check attributes
            if not hasattr(self, 'color_dic'):
                self._load_color_info();
            # check attributes
            if not hasattr(self, 'encoding_scheme'):
                self._load_encoding_scheme();
        else:
            pass
