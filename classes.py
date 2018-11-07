import sys,glob,os,time, copy
import numpy as np
import pickle as pickle
import multiprocessing

from . import get_img_info, corrections, visual_tools, analysis
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy
from scipy import ndimage, stats
from scipy.spatial.distance import pdist,cdist,squareform
from skimage import morphology
from skimage.segmentation import random_walker

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import seismic_r

def _do_cropping_for_cell(_cell, _cropping_args):
    _cell._load_images(*_cropping_args)

def _fit_single_image(_im, _id, _chrom_coords, _seeding_args, _fitting_args, _verbose):
    if _verbose:
        print(f"+++ fitting for region:{_id}")
    _spots_for_chrom = []
    for _chrom_coord in _chrom_coords:
        if _im is None:
            _spots_for_chrom.append(np.array([]))
        else:
            _seeds = visual_tools.get_seed_in_distance(_im, _chrom_coord, *_seeding_args)
            _fits = visual_tools.fit_multi_gaussian(_im, _seeds, *_fitting_args)
            _spots_for_chrom.append(_fits)
    return _spots_for_chrom

class Cell_List():
    """
    Class Cell_List:
    this is a typical data structure of cells within one chromosome with images in multiple independent color-channels and decoding-groups.

    """
    # initialize
    def __init__(self, parameters, _chosen_fovs=[], _exclude_fovs=[], _load_all_attr=False):
        if not isinstance(parameters, dict):
            raise TypeError('wrong input type of parameters, should be a dictionary containing essential info.')

        ## required parameters: data folder (list)
        if isinstance(parameters['data_folder'], list):
            self.data_folder = [str(_fd) for _fd in parameters['data_folder']]
        else:
            self.data_folder = [str(parameters['data_folder'])]

        ## extract hybe folders and field-of-view names
        self.folders = []
        for _fd in self.data_folder:
            _hyb_fds, _fovs = get_img_info.get_folders(_fd, feature='H', verbose=True)
            self.folders += _hyb_fds
            self.fovs = _fovs
        # temp_folder
        if 'temp_folder' in parameters:
            self.temp_folder = parameters['temp_folder']
        else:
            self.temp_folder = _temp_folder
        ## analysis_folder, segmentation_folder, save_folder, correction_folder,map_folder
        if 'analysis_folder'  in parameters:
            self.analysis_folder = str(parameters['analysis_folder'])
        else:
            self.analysis_folder = self.data_folder[0]+os.sep+'Analysis'
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder']
        else:
            self.segmentation_folder = self.analysis_folder+os.sep+'segmentation'
        if 'save_folder' in parameters:
            self.save_folder = parameters['save_folder']
        else:
            self.save_folder = self.analysis_folder+os.sep+'5x10'
        if 'correction_folder' in parameters:
            self.correction_folder = parameters['correction_folder']
        else:
            self.correction_folder = _correction_folder
        if 'drift_folder' in parameters:
            self.drift_folder = parameters['drift_folder']
        else:
            self.drift_folder =  self.analysis_folder+os.sep+'drift'
        if 'map_folder' in parameters:
            self.map_folder = parameters['map_folder']
        else:
            self.map_folder = self.analysis_folder+os.sep+'distmap'

        # number of num_threads
        if 'num_threads' in parameters:
            self.num_threads = parameters['num_threads']
        else:
            self.num_threads = int(os.cpu_count() / 3) # default: use one third of cpus.

        ## if loading all remaining attr in parameter
        if _load_all_attr:
            for _key, _value in parameters.items():
                if not hasattr(self, _key):
                    setattr(self, _key, _value)

        ## list to store Cell_data
        self.cells = []
        # distance from pixel to nm:
        self.distance_zxy = _distance_zxy
        self.sigma_zxy = _sigma_zxy

        ## chosen field of views
        if len(_chosen_fovs) == 0: # no specification
            _chosen_fovs = np.arange(len(_fovs))
        if len(_chosen_fovs) > 0: # there are specifications
            _chosen_fovs = [_i for _i in _chosen_fovs if _i <= len(_fovs)]
            _chosen_fovs = list(np.array(np.unique(_chosen_fovs), dtype=np.int))
        # exclude fovs
        if len(_exclude_fovs) > 0: #exclude any fov:
            for _i in _exclude_fovs:
                if _i in _chosen_fovs:
                    _chosen_fovs.pop(_chosen_fovs.index(_i))
        # save values to the class
        self.fov_ids = _chosen_fovs
        self.chosen_fovs = list(np.array(self.fovs)[np.array(self.fov_ids, dtype=np.int)])
        # read color-usage and encodding-scheme
        self._load_color_info()
        self._load_encoding_scheme()
        # get annotated folders by color usage
        self.annotated_folders = []
        for _hyb_fd, _info in self.color_dic.items():
            _matches = [_fd for _fd in self.folders if _hyb_fd == _fd.split(os.sep)[-1]]
            if len(_matches)==1:
                self.annotated_folders.append(_matches[0])
        print(f"{len(self.annotated_folders)} folders are found according to color-usage annotation.")
        # tool for iteration
        self.index = 0

    # allow print info of Cell_List
    def __str__(self):
        if hasattr(self, 'data_folder'):
            print("Data folder:", self.data_folder)
        if hasattr(self, 'cells'):
            print("Number of cells in this list:", len(self.cells))
        return 'test'
    # allow iteration of Cell_List
    def __iter__(self):
        return self.cells
    def __next__(self):
        if not hasattr(self, 'cells') or not not hasattr(self, 'index'):
            raise StopIteration
        elif self.index == 0:
            raise StopIteration
        else:
            self.index -= 1
        return self.cells[self.index]

    ## Load basic info
    def _load_color_info(self, _color_filename='Color_Usage', _color_format='csv', _save_color_dic=True):
        _color_dic, _use_dapi, _channels = get_img_info.Load_Color_Usage(self.analysis_folder,
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
        _bead_channel = get_img_info.find_bead_channel(_color_dic)
        self.bead_channel_index = _bead_channel
        _dapi_channel = get_img_info.find_dapi_channel(_color_dic)
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
            self.encoding_scheme = _encoding_scheme

        return _encoding_scheme

    ## Load segmentations info
    def _pick_cell_segmentations(self, _type='small', _num_threads=None, _allow_manual=True,
                            _min_shape_ratio=0.038, _signal_cap_ratio=0.2, _denoise_window=5,
                            _shrink_percent=14, _conv_th=-5e-5, _boundary_th=0.55,
                            _load_in_ram=True, _save=True, _force=False,
                            _cell_coord_fl='cell_coords.pkl', _verbose=True):
        ## load segmentation
        # check attributes
        if not hasattr(self, 'channels'):
            self._load_color_info()
        if not _num_threads:
            if not hasattr(self, 'num_threads'):
                raise AttributeError('No num_threads given in funtion kwds and class attributes')
            else:
                _num_threads = self.num_threads
        if _verbose:
            print(f"{len(self.chosen_fovs)} of field-of-views are selected to load segmentation.")
        # do segmentation if necessary, or just load existing segmentation file
        _segmentation_labels, _dapi_ims  = analysis.Segmentation_All(self.analysis_folder,
                    self.folders, self.chosen_fovs, _type, ref_name='H0R0',
                    num_threads=_num_threads,
                    min_shape_ratio=_min_shape_ratio, signal_cap_ratio=_signal_cap_ratio,
                    conv_th=_conv_th, boundary_th=_boundary_th,
                    denoise_window=_denoise_window,
                    num_channel=len(self.channels), dapi_channel=self.dapi_channel_index,
                    correction_folder=self.correction_folder,
                    segmentation_path=os.path.basename(self.segmentation_folder),
                    save=_save, force=_force, verbose=_verbose)
        _dapi_ims = [corrections.Remove_Hot_Pixels(_im) for _im in _dapi_ims]
        _dapi_ims = [corrections.Z_Shift_Correction(_im) for _im in _dapi_ims]

        ## pick(exclude) cells from previous result
        if _allow_manual:
            # generate coordinates
            _coord_list, _index_list = [],[]
            for _i, _label in enumerate(_segmentation_labels):
                for _j in range(np.max(_label)):
                    _center = np.round(ndimage.measurements.center_of_mass(_label==_j+1))
                    _center = list(np.flipud(_center))
                    _center.append(_dapi_ims[0].shape[0]/2)
                    _coord_list.append(_center)
                    _index_list.append(_i)
            # wrap into a dic
            _cell_coord_dic = {'coords': _coord_list,
                          'class_ids': _index_list,
                          'pfits':{},
                          'dec_text':{},
                          }
            self.cell_coord_dic = copy.deepcopy(_cell_coord_dic)
            # use visual tools to pick
            _cell_coord_savefile = self.save_folder + os.sep + _cell_coord_fl

            _cell_viewer = visual_tools.imshow_mark_3d_v2(_dapi_ims, image_names=self.chosen_fovs,
                                                          save_file=_cell_coord_savefile,
                                                          given_dic=_cell_coord_dic)

            return _cell_viewer
        else:
            return _segmentation_labels, _dapi_ims

    def _update_cell_segmentations(self, _cell_coord_fl='cell_coords.pkl',
                                  _overwrite_segmentation=True,
                                  _marker_displace_th = 2700,
                                  _append_new=True, _append_radius=90,
                                  _verbose=True):
        """Function to update cell segmentation info from saved file,
            - usually do this after automatic segmentation"""
        _cell_coord_savefile = self.save_folder + os.sep + _cell_coord_fl
        if not os.path.exists(_cell_coord_savefile):
            raise IOError(f'{_cell_coord_savefile} doesnot exist, exit')

        with open(_cell_coord_savefile, 'rb') as handle:
            _new_cell_coord_dic = pickle.load(handle)
        # parse
        _ccd = visual_tools.partition_map(self.cell_coord_dic['coords'], self.cell_coord_dic['class_ids'])
        _new_ccd = visual_tools.partition_map(_new_cell_coord_dic['coords'], _new_cell_coord_dic['class_ids'])

        # initialize
        _new_seg_labels, _dapi_ims = [], []
        _remove_cts, _append_cts = [], []
        for _i, (_cell_coords, _new_cell_coords) in enumerate(zip(_ccd, _new_ccd)):
            # now we are taking care of one specific field of view
            if _verbose:
                print(f"-- fov-{_i}, match manually picked cell with sgementation ")
            # load fov image
            _seg_file = self.segmentation_folder+os.sep+self.chosen_fovs[_i].replace('.dax', '_segmentation.pkl')
            _seg_label, _dapi_im = pickle.load(open(_seg_file, 'rb'))
            _remove = 0
            if not _overwrite_segmentation:
                # save original seg label into another file
                _old_seg_file = _seg_file.replace('_segmentation.pkl', '_segmentation_old.pkl')
                pickle.dump([_seg_label, _dapi_im], open(_old_seg_file, 'wb'))
            # keep cells in original segmentation with markers
            for _l, _coord in enumerate(_cell_coords):
                _dist = [np.sum((_c-_coord)**2) for _c in _new_cell_coords]
                _match = [_d < _marker_displace_th for _d in _dist]
                if sum(_match) == 0:
                    _seg_label[_seg_label==_l+1-_remove] = -1
                    _seg_label[_seg_label >_l+1-_remove] -= 1
                    _remove += 1
            if _append_new:
                _append = 0
                if _verbose:
                    print(f"-- Appending manually added markers with radius={_append_radius}")
                def _add_round_marker(_label, _center, _radius, overwrite_marker=False):
                    """Function to add round-marker with given center and radius"""
                    if len(_label.shape) != len(_center):
                        raise ValueError("Dimension of label and center doesn't match")
                    # convert format
                    _center = np.array(_center, dtype=np.int)
                    _radius = np.int(_radius)
                    # generate mask
                    _shape_lst = (list(range(_label.shape[i])) for i in range(len(_label.shape)))
                    _coord_lst = np.meshgrid(*_shape_lst, indexing='ij')
                    _dist = np.sqrt(np.sum(np.stack([(_coords - _ct)**2 for _coords, _ct in zip(_coord_lst, _center)]), axis=0))
                    _new_mask = np.array(_dist<=_radius, dtype=np.int)
                    if not overwrite_marker:
                        _new_mask *= np.array(_label<=0, dtype=np.int)
                    # create new label
                    _new_label = _label
                    _new_label[_new_mask > 0] = int(np.max(_label))+1
                    return _new_label
                for _l, _new_coord in enumerate(_new_cell_coords):
                    _dist = [np.sum((_c-_new_coord)**2) for _c in _cell_coords]
                    _match = [_d < _marker_displace_th for _d in _dist]
                    if sum(_match) == 0:
                        if _verbose:
                            print(f"--- adding manually picked new label in {_i}, label={np.max(_seg_label)+1} ")
                        _seg_label = _add_round_marker(_seg_label, np.flipud(_new_coord)[-len(_seg_label.shape):], _append_radius)
                        _append += 1
                _append_cts.append(_append)

            if _verbose:
                print(f"--- {_remove} label(s) got removed!")
            _new_seg_labels.append(_seg_label)
            _dapi_ims.append(_dapi_im)
            _remove_cts.append(_remove)
            # save
            if _verbose:
                print(f"--- save updated segmentation to {os.path.basename(_seg_file)}")
            pickle.dump([_seg_label, _dapi_im], open(_seg_file, 'wb'))

        return _new_seg_labels, _dapi_ims, _remove_cts, _append_cts

    def _create_cell(self, _parameter, _load_info=True,
                     _load_segmentation=True, _load_drift=True,
                     _load_file=True,
                     _save=False, _directly_append_cell_list=False, _verbose=True):
        """Function to create one cell_data object"""
        if _verbose:
            print(f"+ creating cell for fov:{_parameter['fov_id']}, cell:{_parameter['cell_id']}")
        _cell = Cell_Data(_parameter, _load_all_attr=True)
        if _load_info:
            if not hasattr(_cell, 'color_dic') or not hasattr(_cell, 'channels'):
                _cell._load_color_info()
            if not hasattr(_cell, 'encoding_scheme'):
                _cell._load_encoding_scheme()
        if _load_segmentation and not hasattr(_cell, 'segmentation_label'):
            _cell._load_segmentation()
        if _load_drift and not hasattr(_cell, 'drift'):
            _cell._load_drift()
        if _load_file:
            if os.path.isfile(os.path.join(_cell.save_folder, 'cell_info.pkl')):
                _cell._load_from_file('cell_info', _overwrite=False, _verbose=True)
        if _save:
            _cell._save_to_file('cell_info')
        # whether directly store
        if _directly_append_cell_list:
            self.cells.append(_cell)
        return _cell

    def _create_cells_fov(self, _fov_ids, _segmentation_type='small', _num_threads=None, _load_annotated_only=True, _overwrite_temp=True,
                          _drift_size=550, _drift_dynamic=True, _plot_segmentation=True, _verbose=True):
        """Create Cele_data objects for one field of view"""
        if not _num_threads:
            _num_threads = int(self.num_threads)
        if not isinstance(_fov_ids, list):
            _fov_ids = [_fov_ids]
        for _fov_id in _fov_ids:
            if _fov_id not in self.fov_ids:
                raise ValueError("Wrong fov_id kwd given! \
                    this should be real fov-number that allowed during intiation of class.")
        if _verbose:
            print(f"+ Create Cell_Data objects for field of view: {_fov_ids}")
            print("++ preparing variables")
        # check attributes
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info()
        # whether load annotated hybs only
        if _load_annotated_only:
            _folders = self.annotated_folders
        else:
            _folders = self.folders

        # load segmentation for this fov
        _args = []
        for _fov_id in _fov_ids:
            if _verbose:
                print("+ Load segmentation for fov", _fov_id)
            # do segmentation if necessary, or just load existing segmentation file
            _fov_segmentation_label, _fov_dapi_im  = analysis.Segmentation_Fov(self.analysis_folder,
                                                    _folders, self.fovs, _fov_id, _segmentation_type,
                                                    num_channel=len(self.channels),
                                                    dapi_channel=self.dapi_channel_index,
                                                    illumination_corr=True,
                                                    correction_folder=self.correction_folder,
                                                    segmentation_path=os.path.basename(self.segmentation_folder),
                                                    save=True, force=False, verbose=_verbose)
            if _plot_segmentation:
                plt.figure()
                plt.imshow(_fov_segmentation_label)
                plt.show()
            # do drift for this fov
            if _verbose:
                print("+ Drift correction for fov", _fov_id)
            _drift_fl = self.drift_folder+os.sep+self.fovs[_fov_id].replace('.dax', "*_cor.pkl")
            _drift_fl = glob.glob(_drift_fl)
            _do_drift_marker = False
            if len(_drift_fl) != 1 or not os.path.isfile(_drift_fl[0]):
                if _verbose:
                    print("+++ drift file not unique or not exist, proceed to do drift correction")
                _do_drift_marker = True
            else:
                _drift_fl = _drift_fl[0]
                if _verbose:
                    print("+++ try to load drift correction from file:", _drift_fl)
                _fov_drift = pickle.load(open(_drift_fl, 'rb'))
                for _fd in _folders:
                    _hyb_name = os.path.join(os.path.basename(_fd), self.fovs[_fov_id] )
                    if _hyb_name not in _fov_drift:
                        if _verbose:
                            print(f"+++ drift info for {_hyb_name} not exists, proceed to do drift correction")
                        _do_drift_marker = True

            if _do_drift_marker:
                if _verbose:
                    print("++ loading bead images for drift correction.")
                # load bead images
                _bead_temp_files, _,_ = analysis.load_image_fov(_folders, self.fovs, _fov_id, self.channels,  self.color_dic,
                                                    self.num_threads, loading_type='beads', max_chunk_size=6,
                                                    correction_folder=self.correction_folder, overwrite_temp=_overwrite_temp,
                                                    temp_folder=self.temp_folder, return_type='filename', verbose=_verbose)
                _bead_ims, _bead_names = analysis.reconstruct_from_temp(_bead_temp_files, _folders, self.fovs, _fov_id,
                                                           self.channels, self.color_dic, temp_folder=self.temp_folder,
                                                           find_all=True, num_threads=self.num_threads, loading_type='beads', verbose=_verbose)
                if _verbose:
                    print ("++ do drift correction!")
                # do drift corrections
                _fov_drift, _failed_count = corrections.STD_beaddrift_sequential(_bead_ims, _bead_names,
                                                                                self.drift_folder, self.fovs, _fov_id,
                                                                                drift_size=_drift_size, overwrite=False,
                                                                                dynamic=_drift_dynamic, verbose=_verbose)
                if _failed_count > 0:
                    print(f"++ {_failed_count} suspected failures detected in drift correction.")
                # release
                del(_bead_ims)

            # create cells in parallel
            _cell_ids = range(int(np.max(_fov_segmentation_label)-1))
            if _verbose:
                print(f"+ Create cell_data objects, num_of_cell:{len(_cell_ids)}")
            _params = [{'fov_id': _fov_id,
                      'cell_id': _cell_id,
                      'folders': self.folders,
                      'fovs': self.fovs,
                      'data_folder': self.data_folder,
                      'color_dic': self.color_dic,
                      'use_dapi': self.use_dapi,
                      'channels': self.channels,
                      'bead_channel_index': self.bead_channel_index,
                      'dapi_channel_index': self.dapi_channel_index,
                      'encoding_scheme': self.encoding_scheme,
                      'annotated_folders': self.annotated_folders,
                      'temp_folder': self.temp_folder,
                      'analysis_folder':self.analysis_folder,
                      'save_folder': self.save_folder,
                      'segmentation_folder': self.segmentation_folder,
                      'correction_folder': self.correction_folder,
                      'drift_folder': self.drift_folder,
                      'map_folder': self.map_folder,
                      'drift': _fov_drift,
                      'distance_zxy' : self.distance_zxy,
                      'sigma_zxy': self.sigma_zxy,
                      } for _cell_id in _cell_ids]
            _args += [(_p, True, True, True, True, False) for _p in _params]
            del(_fov_drift, _fov_segmentation_label, _fov_dapi_im, _params, _cell_ids)
        # do multi-processing to create cells!
        if _verbose:
            print(f"+ Creating cells with {_num_threads} threads.")
        _cell_pool = multiprocessing.Pool(_num_threads)
        _cells = _cell_pool.starmap(self._create_cell, _args, chunksize=1)
        _cell_pool.close()
        _cell_pool.join()
        _cell_pool.terminate()
        # load
        self.cells += _cells

    def _crop_image_for_cells(self, _type='all', _load_in_ram=True, _load_annotated_only=True,
                              _max_loading_chunk=5, _overwrite_temp=True, _overwrite_cell_info=False,
                              _save=True, _force=False, _verbose=True):
        """Load images for all cells in this cell_list
        Inputs:
            _type: loading type for this """
        # check whether cells and segmentation,drift info exists
        if _verbose:
            print ("+ Load images for cells in this cell list")
        if not hasattr(self, 'cells') or len(self.cells) ==0:
            raise ValueError("No cell information loaded in cell_list")
        # whether load annotated hybs only
        if _load_annotated_only:
            _folders = self.annotated_folders
        else:
            _folders = self.folders
        _fov_dic = {} # fov_id ->
        for _cell in self.cells:
            if not hasattr(_cell,'drift'):
                _cell._load_drift()
            if not hasattr(_cell, 'segmentation_label'):
                _cell._load_segmentation()
            # record fov list at the same time
            if _cell.fov_id not in _fov_dic:
                _fov_dic[_cell.fov_id] = {}
        if _verbose:
            print(f"++ Field of view loaded for this list: {list(_fov_dic.keys())}")
        # load images
        for _id in _fov_dic:
            if _verbose:
                print(f"++ loading image for fov:{_id}")
            _temp_files, _, _ = analysis.load_image_fov(_folders, self.fovs, _id,
                                self.channels,  self.color_dic, self.num_threads, loading_type=_type,
                                max_chunk_size=_max_loading_chunk, correction_folder=self.correction_folder, overwrite_temp=_overwrite_temp,
                                temp_folder=self.temp_folder, return_type='filename', verbose=_verbose)
            _fov_dic[_id] = analysis.reconstruct_from_temp(_temp_files, _folders, self.fovs, _id,
                                                       self.channels, self.color_dic, temp_folder=self.temp_folder,
                                                       find_all=True, num_threads=self.num_threads, loading_type=_type, verbose=_verbose)

        # Load for combo
        if _type == 'all' or _type == 'combo':
            if _verbose:
                print(f"++ cropping combo for cells")
            # record start time
            _start_time = time.time()
            # loop through all cells
            for _cell in self.cells:
                if _verbose:
                    print(f"+++ crop combo images for cell:{_cell.cell_id} in fov:{_cell.fov_id}")
                _cell._load_images('combo', _splitted_ims=_fov_dic[_cell.fov_id],
                                   _num_threads=self.num_threads, _extend_dim=20,
                                   _load_in_ram=_load_in_ram, _load_annotated_only=_load_annotated_only,
                                   _save=_save, _overwrite=_overwrite_cell_info, _verbose=_verbose)
                if _save:
                    _cell._save_to_file('cell_info', _overwrite=_overwrite_cell_info, _verbose=_verbose)
        # load for unique
        if _type == 'all' or _type == 'unique':
            if _verbose:
                print("++ cropping unique for cells")
            for _cell in self.cells:
                if _verbose:
                    print(f"+++ crop unique images for cell:{_cell.cell_id} in fov:{_cell.fov_id}")
                _cell._load_images('unique', _splitted_ims=_fov_dic[_cell.fov_id],
                                   _num_threads=self.num_threads, _extend_dim=20,
                                   _load_in_ram=_load_in_ram, _load_annotated_only=_load_annotated_only,
                                   _save=_save, _overwrite=_overwrite_cell_info, _verbose=_verbose)
                if _save:
                    _cell._save_to_file('cell_info', _overwrite=_overwrite_cell_info, _verbose=_verbose)

    def _load_cells_from_files(self, _type='all', _decoded_flag=None, _overwrite_cells=False, _verbose=True):
        """Function to load cells from existing files"""
        if _verbose:
            print("+ Load cells from existing files.")
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError('No cell information provided, should create cells first!')
        # check fov_id input
        for _cell in self.cells:
            if _verbose:
                print(f"++ loading info for fov:{_cell.fov_id}, cell:{_cell.cell_id}")
            if _type == 'decoded':
                _cell._load_from_file(_type=_type, _save_folder=None, _load_attrs=[], _decoded_flag=_decoded_flag,
                                        _overwrite=_overwrite_cells, _verbose=_verbose)
            else:
                _cell._load_from_file(_type=_type, _save_folder=None, _load_attrs=[],
                                        _overwrite=_overwrite_cells, _verbose=_verbose)

    def _get_chromosomes_for_cells(self, _source='combo', _max_count= 30,
                                   _gaussian_size=2, _cap_percentile=1, _seed_dim=3,
                                   _th_percentile=99.5, _min_obj_size=125,
                                   _coord_filename='chrom_coords.pkl', _overwrite=False, _verbose=True):
        """Function to generate chromosome and chromosome coordinates, open a picker to correct for it
        Inputs:
            _source: image source to generate chromosome image, combo requires "combo_gorups",
                unique requires 'unique_ims', 'combo'/'unique' (default: 'combo')
            _max_count: maximum image count to generate chromosome profile, int (default:30)
        Outputs:
            _chrom_viewer: chromosome viewer object, used to click"""
        # check attribute
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError('No cells are generated in this cell list!')
        if _verbose:
            print("+ Generate chromosomes for cells.")
        # chromsome savefile #
        _fov_ids = [_cell.fov_id for _cell in self.cells]
        _fov_ids = np.unique(_fov_ids)
        _chrom_savefile = os.path.join(self.temp_folder, _coord_filename.replace('.pkl', str(_fov_ids)+'.pkl'))
        # loop through cells to generate chromosome
        _chrom_ims = []
        _chrom_dims = []
        _coord_dic = {'coords': [],
                      'class_ids': [],
                      'pfits':{},
                      'dec_text':{},
                      } # initialize _coord_dic for picking
        for _i, _cell in enumerate(self.cells):
            if hasattr(_cell, 'chrom_im') and not _overwrite:
                _cim = _cell.chrom_im
            else:
                _cim = _cell._generate_chromosome_image(_source=_source, _max_count=_max_count, _verbose=_verbose)
                _cell.chrom_im = _cim
            _chrom_ims.append(_cim)
            _chrom_dims.append(np.array(np.shape(_cim)))
            if hasattr(_cell, 'chrom_coords') and not _overwrite:
                _chrom_coords = _cell.chrom_coords
            else:
                _, _chrom_coords = _cell._identify_chromosomes(_gaussian_size=_gaussian_size, _cap_percentile=_cap_percentile,
                                                               _seed_dim=_seed_dim, _th_percentile=_th_percentile,
                                                               _min_obj_size=_min_obj_size,_verbose=_verbose)
            # build chrom_coord_dic
            _coord_dic['coords'] += [np.flipud(_coord) for _coord in _chrom_coords]
            _coord_dic['class_ids'] += list(np.ones(len(_chrom_coords),dtype=np.uint8)*int(_i))
        # create existing coord_dic file
        if _verbose:
            print("++ dumping existing info to file:", _chrom_savefile)
        pickle.dump(_coord_dic, open(_chrom_savefile, 'wb'))
        # convert to the same dimension
        _max_dim = np.max(np.concatenate([_d[np.newaxis,:] for _d in _chrom_dims]), axis=0)
        if _verbose:
            print("Maximum dimension for these images:", _max_dim)
        _converted_ims = [np.ones(_max_dim) * np.min(_cim) for _cim in _chrom_ims]
        for _im, _d, _cim in zip(_converted_ims, _chrom_dims, _chrom_ims):
            _im[:_d[0], :_d[1],:_d[2]] = _cim

        _chrom_viewer = visual_tools.imshow_mark_3d_v2(_converted_ims, image_names=[f"fov:{_cell.fov_id}, cell:{_cell.cell_id}" for _cell in self.cells],
                                                       save_file=_chrom_savefile)
        _chrom_viewer.load_coords()

        return _chrom_viewer

    def _update_chromosomes_for_cells(self, _coord_filename='chrom_coords.pkl', _save=True, _verbose=True):
        # check attribute
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError('No cells are generated in this cell list!')
        if _verbose:
            print("+ Update manually picked chromosomes to cells")
        # chromsome savefile #
        _fov_ids = [_cell.fov_id for _cell in self.cells]
        _fov_ids = np.unique(_fov_ids)
        _chrom_savefile = os.path.join(self.temp_folder, _coord_filename.replace('.pkl', str(_fov_ids)+'.pkl'))
        # load from chrom-coord and partition it
        _coord_dic = pickle.load(open(_chrom_savefile, 'rb'))
        _coord_list = visual_tools.partition_map(_coord_dic['coords'], _coord_dic['class_ids'], enumerate_all=True)
        if len(_coord_list) > len(self.cells):
            raise ValueError(f'Number of cells doesnot match between cell-list and {_chrom_savefile}')
        elif len(_coord_list) < len(self.cells):
            print("++ fewer picked chromosome sets discovered than number of cells, append with empty lists.")
            for _i in range(len(self.cells) - len(_coord_list)):
                _coord_list.append([])
        # save to attribute first
        for _cell, _coords in zip(self.cells, _coord_list):
            _chrom_coords = [np.flipud(_coord) for _coord in _coords]
            _cell.chrom_coords = _chrom_coords
            if _verbose:
                print(f"++ matching {len(_chrom_coords)} chromosomes for fov:{_cell.fov_id}, cell:{_cell.cell_id}")
        # then update files if specified
            if _save:
                _cell._save_to_file('cell_info', _save_dic={'chrom_coords': _cell.chrom_coords},  _verbose=_verbose)
                if hasattr(_cell, 'combo_groups'):
                    _cell._save_to_file('combo', _overwrite=True, _verbose=_verbose)

    def _remove_temp_fov(self, _fov_id, _temp_marker='corrected.npy', _verbose=True):
        """Remove all temp files for given fov """
        _temp_fls = glob.glob(os.path.join(self.temp_folder, '*'))
        if _verbose:
            print(f"+ Remove temp file for fov:{_fov_id}")
        for _fl in _temp_fls:
            if os.path.isfile(_fl) and _temp_marker in _fl and self.fovs[_fov_id].replace('.dax', '') in _fl:
                print("++ removing temp file:", os.path.basename(_fl))
                os.remove(_fl)

    def _spot_finding_for_cells(self, _type='unique', _decoded_flag=None, _max_fitting_threads=5, _clear_image=False,
                                _use_chrom_coords=True, _seed_th_per=50, _max_filt_size=3,
                                _max_seed_count=6, _min_seed_count=3,
                                _expect_weight=1000, _min_height=100, _max_iter=10, _th_to_end=1e-5,
                                _save=True, _verbose=True):
        """Function to allow multi-fitting in cell_list"""
        ## Check attributes
        for _cell_id, _cell in enumerate(self.cells):
            _clear_image_for_cell = _clear_image # whether clear image for this cell
            if _type == 'unique':
                _result_attr='unique_spots'
                if not hasattr(_cell, 'unique_ims') or not hasattr(_cell, 'unique_ids'):
                    _clear_image_for_cell = True
                    try:
                        _cell._load_from_file('unique')
                    except:
                        raise IOError("Cannot load unique files")
            elif _type == 'decoded':
                _result_attr='decoded_spots'
                if not hasattr(_cell, 'decoded_ims') or not hasattr(_cell, 'decoded_ids'):
                    _clear_image_for_cell = True
                    try:
                        _cell._load_from_file('decoded',_decoded_flag=_decoded_flag)
                    except:
                        raise IOError("Cannot load decoded files")
            else:
                raise ValueError("Wrong _type keyword given!")
            # do multi_fitting
            _cell._multi_fitting(_type=_type, _use_chrom_coords=_use_chrom_coords, _num_threads=min(_max_fitting_threads, self.num_threads),
                                 _seed_th_per=_seed_th_per, _max_filt_size=_max_filt_size, _max_seed_count=_max_seed_count,
                                 _min_seed_count=_min_seed_count, _width_zxy=self.sigma_zxy, _fit_radius=10,
                                 _expect_weight=_expect_weight, _min_height=_min_height, _max_iter=_max_iter,
                                 _save=_save, _verbose=_verbose)
            if _clear_image_for_cell:
                if _verbose:
                    print(f"++ clear images for {_type} in fov:{_cell.fov_id}, cell:{_cell.cell_id}")
                if _type == 'unique':
                    _cell.unique_ims = None
                elif _type == 'decoded':
                    _cell.decoded_ims = None

    def _pick_spots_for_cells(self, _type='unique', _pick_type='dynamic', _use_chrom_coords=True, _distance_zxy=None,
                              _w_dist=2, _dist_ref=None, _penalty_type='trapezoidal', _penalty_factor=5,
                              _gen_distmap=True, _save_plot=True, _plot_limits=[200,1000],
                              _save=True, _verbose=True):
        """Function to pick spots given candidates."""
        ## Check attributes
        if _verbose:
            print("+ Pick spots and convert to distmap.")
        if _pick_type not in ['dynamic', 'naive']:
            raise ValueError(f"Wrong _pick_type kwd given ({_pick_type}), should be dynamic or naive.")
        for _cell in self.cells:
            if _type == 'unique':
                if not hasattr(_cell, 'unique_ids'):
                    try:
                        _cell._load_from_file('unique')
                    except:
                        _cell._load_images('unique', _load_in_ram=True)
                elif not hasattr(_cell, 'unique_spots'):
                    _cell._load_from_file('cell_info')
                    if not hasattr(_cell, 'unique_spots'):
                        raise ValueError(f"No unique spots info detected for cell:{_cell.cell_id}")
            elif _type == 'combo' or _type == 'decoded':
                if not hasattr(_cell, 'decoded_ids'):
                    try:
                        _cell._load_from_file('decoded')
                    except:
                        raise IOError("Cannot load decoded files!")
            else:
                raise ValueError(f"Wrong _type kwd given ({_type}), should be unique or decoded.")
        ## start pick chromosome
        for _cell in self.cells:
            if _verbose:
                print(f"++ picking spots for cell:{_cell.cell_id} by {_pick_type} method:")
            # pick spots
            if _pick_type == 'dynamic':
                _cell._dynamic_picking_spots(_type=_type, _use_chrom_coords=_use_chrom_coords,
                                             _distance_zxy=_distance_zxy, _w_int=1, _w_dist=_w_dist,
                                             _dist_ref=_dist_ref, _penalty_type=_penalty_type, _penalty_factor=_penalty_factor,
                                             _save=_save, _verbose=_verbose)
            elif _pick_type == 'naive':
                _cell._naive_picking_spots(_type=_type, _use_chrom_coords=_use_chrom_coords,
                                           _save=_save, _verbose=_verbose)
            # make map:
            if _gen_distmap:
                if _verbose:
                    print(f"+++ generating distance map for cell:{_cell.cell_id}")
                _cell._generate_distance_map(_type=_type, _distance_zxy=_distance_zxy, _save_info=_save,
                                             _save_plot=_save_plot, _limits=_plot_limits, _verbose=_verbose)

    def _calculate_population_map(self, _type='unique', _max_loss_prob=0.15,
                                  _ignore_inf=True, _stat_type='median',
                                  _make_plot=True, _save_plot=True, _save_name='distance_map',
                                  _plot_limits=[300,1500], _verbose=True):
        """Calculate 'averaged' map for all cells in this list
        Inputs:
            _type: unique or decoded
            _max_loss_prob: maximum """
        ## check inputs:
        if _type not in ['unique','decoded']:
            raise ValueError(f"Wrong _type kwd given, should be unique or decoded, {_type} is given!")
        elif _type == 'unique':
            _picked_spots_attr = 'picked_unique_spots'
            _distmap_attr = 'unique_distance_map'
        elif _type == 'decoded':
            _picked_spots_attr = 'picked_decoded_spots'
            _distmap_attr = 'decoded_distance_map'
        if _stat_type not in ['median', 'mean']:
            raise ValueError(f"Wrong _stat_type({_stat_type}) kwd is given!")
        # detect distmap shape
        _distmap_shape=[]
        for _cell in self.cells:
            for _distmap in getattr(_cell, _distmap_attr):
                if np.shape(_distmap)[0] not in _distmap_shape:
                    _distmap_shape.append(np.shape(_distmap)[0])
        if _verbose:
            print(f"+++ maximum distance-map size is {max(_distmap_shape)}")
        _cand_distmaps = []
        ## check and collect distance maps
        for _cell in self.cells:
            for _picked_spots, _distmap in zip(getattr(_cell, _picked_spots_attr), getattr(_cell, _distmap_attr)):

                _failed_count = sum([np.inf in _coord or len(_coord)==0 for _coord in _picked_spots])
                if _failed_count / len(_picked_spots) > _max_loss_prob:
                    if _verbose:
                        print(f"+++ chromosome filtered out by loss probability in fov:{_cell.fov_id}, cell:{_cell.cell_id}")
                    continue
                elif np.shape(_distmap)[0] != max(_distmap_shape):
                    if _verbose:
                        print(f"+++ chromosome filtered out by dist-map shape in fov:{_cell.fov_id}, cell:{_cell.cell_id}")
                    continue
                else:
                    _cand_distmaps.append(_distmap)
        if len(_cand_distmaps) == 0:
            print("No distant map loaded, return.")
            return None, 0

        ## calculate averaged map
        _total_map = np.array(_cand_distmaps, dtype=np.float)
        if _ignore_inf:
            _total_map[_total_map == np.inf] = np.nan
        if _stat_type == 'median':
            _averaged_map = np.nanmedian(_total_map, axis=0)
        elif _stat_type == 'mean':
            _averaged_map = np.nanmean(_total_map, axis=0)
        ## make plots
        if _make_plot:
            if _verbose:
                print(f"++ generating distance map for {len(_cand_distmaps)} chromosomes.")
            _used_fovs = []
            for _cell in self.cells:
                if _cell.fov_id not in _used_fovs:
                    _used_fovs.append(_cell.fov_id)
            _used_fovs = sorted(_used_fovs)
            plt.figure()
            plt.title(f"{_stat_type} distance map, num of chrom:{len(_cand_distmaps)}")
            plt.imshow(_averaged_map, interpolation='nearest', cmap=matplotlib.cm.seismic_r,
                       vmin=min(_plot_limits), vmax=max(_plot_limits))
            plt.colorbar(ticks=range(min(_plot_limits),max(_plot_limits),200), label='distance (nm)')
            if _save_plot:
                if _verbose:
                    print(f"++ saving {_stat_type} distance map.")
                _filename = os.path.join(self.map_folder, f"{_stat_type}_{_save_name}_fov{_used_fovs}.png")
                if not os.path.exists(self.map_folder):
                    os.makedirs(self.map_folder)
                plt.savefig(_filename, transparent=True)

        return _averaged_map, len(_cand_distmaps)


class Cell_Data():
    """
    Class Cell_data:
    data structure of each cell with images in multiple independent color-channels and decoding-groups.
    initialization of cell_data requires:
    """
    # initialize
    def __init__(self, parameters, _load_all_attr=False):
        if not isinstance(parameters, dict):
            raise TypeError('wrong input type of parameters, should be a dictionary containing essential info.')
        # necessary parameters
        # data folder (list)
        if isinstance(parameters['data_folder'], list):
            self.data_folder = [str(_fd) for _fd in parameters['data_folder']]
        else:
            self.data_folder = [str(parameters['data_folder'])]
        # analysis folder
        if 'analysis_folder'  in parameters:
            self.analysis_folder = str(parameters['analysis_folder'])
        else:
            self.analysis_folder = self.data_folder[0]+os.sep+'Analysis'
        # temp_folder
        if 'temp_folder'  in parameters:
            self.temp_folder = parameters['temp_folder']
        else:
            self.temp_folder = _temp_folder
        # extract hybe folders and field-of-view names
        if 'folders' in parameters and 'fovs' in parameters:
            self.folders = parameters['folders']
            self.fovs = parameters['fovs']
        else:
            self.folders = []
            for _fd in self.data_folder:
                _hyb_fds, self.fovs = get_img_info.get_folders(_fd, feature='H', verbose=True)
                self.folders += _hyb_fds
        # fov id and cell id given
        self.fov_id = int(parameters['fov_id'])
        self.cell_id = int(parameters['cell_id'])
        ## Constants
        # distance zxy
        self.distance_zxy = _distance_zxy
        self.sigma_zxy = _sigma_zxy
        if 'num_threads' in parameters:
            self.num_threads = parameters['num_threads']
        if 'distance_reference' in parameters:
            self.distance_reference = parameters['distance_reference']
        else:
            self.distance_reference = os.path.join(self.analysis_folder, 'distance_ref.npz')

        # segmentation_folder, save_folder, correction_folder,map_folder
        if 'segmentation_folder' in parameters:
            self.segmentation_folder = parameters['segmentation_folder']
        else:
            self.segmentation_folder = self.analysis_folder+os.sep+'segmentation'
        if 'save_folder' in parameters:
            self.save_folder = os.path.join(parameters['save_folder'],
                                            'fov-'+str(self.fov_id),
                                            'cell-'+str(self.cell_id))
        else:
            self.save_folder = os.path.join(self.analysis_folder,
                                            '5x10',
                                            'fov-'+str(self.fov_id),
                                            'cell-'+str(self.cell_id))
        if 'correction_folder' in parameters:
            self.correction_folder = parameters['correction_folder']
        else:
            self.correction_folder = _correction_folder
        if 'drift_folder' in parameters:
            self.drift_folder = parameters['drift_folder']
        else:
            self.drift_folder =  self.analysis_folder+os.sep+'drift'
        if 'map_folder' in parameters:
            self.map_folder = parameters['map_folder']
        else:
            self.map_folder = self.analysis_folder+os.sep+'distmap'
        # if loading all remaining attr in parameter
        if _load_all_attr:
            for _key, _value in parameters.items():
                if not hasattr(self, _key):
                    setattr(self, _key, _value)

        # load color info
        if not hasattr(self, 'color_dic') or not hasattr(self, 'channels'):
            self._load_color_info()
        # annotated folders
        if not hasattr(self, 'annotated_folders'):
            self.annotated_folders = []
            for _hyb_fd in self.color_dic:
                _matches = [_fd for _fd in self.folders if _hyb_fd == _fd.split(os.sep)[-1]]
                if len(_matches) == 1:
                    self.annotated_folders.append(_matches[0])
            print(f"-- {len(self.annotated_folders)} folders are found according to color-usage annotation.")



    # allow print info of Cell_List
    def __str__(self):
        if hasattr(self, 'data_folder'):
            print("Cell Data from folder(s):", self.data_folder)
        if hasattr(self, 'analysis_folder'):
            print("\t path for anaylsis result:", self.analysis_folder)
        if hasattr(self, 'fov_id'):
            print("\t from field-of-view:", self.fov_id)
        if hasattr(self, 'cell_id'):
            print("\t with cell_id:", self.cell_id)

        return 'test'

    # allow iteration of Cell_List
    def __iter__(self):
        return self
    def __next__(self):
        return self

    ## Load color_usage
    def _load_color_info(self, _color_filename='Color_Usage', _color_format='csv', _save_color_dic=True):
        _color_dic, _use_dapi, _channels = get_img_info.Load_Color_Usage(self.analysis_folder,
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
        _bead_channel = get_img_info.find_bead_channel(_color_dic)
        self.bead_channel_index = _bead_channel
        _dapi_channel = get_img_info.find_dapi_channel(_color_dic)
        self.dapi_channel_index = _dapi_channel

        return _color_dic
    ## Load encoding scheme
    def _load_encoding_scheme(self, _encoding_filename='Encoding_Scheme', _encoding_format='csv', _save_encoding_scheme=True):
        _encoding_scheme, self.hyb_per_group, self.reg_per_group, \
        self.encoding_colors, self.encoding_group_nums \
            = get_img_info.Load_Encoding_Scheme(self.analysis_folder,
                                                   encoding_filename=_encoding_filename,
                                                   encoding_format=_encoding_format,
                                                   return_info=True)
        # need-based encoding scheme saving
        if _save_encoding_scheme:
            self.encoding_scheme = _encoding_scheme

        return _encoding_scheme
    ## Load cell specific info
    def _load_segmentation(self, _type='small', _min_shape_ratio=0.038, _signal_cap_ratio=0.20, _denoise_window=5,
                           _shrink_percent=14, _conv_th=-5e-5, _boundary_th=0.55,
                           _load_in_ram=True, _save=True, _force=False, _verbose=False):
        # check attributes
        if not hasattr(self, 'channels'):
            self._load_color_info()

        # do segmentation if necessary, or just load existing segmentation file
        fov_segmentation_label, fov_dapi_im  = analysis.Segmentation_Fov(self.analysis_folder,
                                                self.folders, self.fovs, self.fov_id, type=_type,
                                                min_shape_ratio=_min_shape_ratio,
                                                signal_cap_ratio=_signal_cap_ratio,
                                                denoise_window=_denoise_window,
                                                shrink_percent=_shrink_percent,
                                                conv_th=_conv_th, boundary_th=_boundary_th,
                                                num_channel=len(self.channels),
                                                dapi_channel=self.dapi_channel_index,
                                                illumination_corr=True, correction_folder=self.correction_folder,
                                                segmentation_path=os.path.basename(self.segmentation_folder),
                                                save=_save, force=_force, verbose=_verbose)
        # exclude special cases
        if not hasattr(self, 'cell_id'):
            raise AttributeError('no cell_id attribute for this cell_data class object!')
        elif self.cell_id+1 not in np.unique(fov_segmentation_label):
            raise ValueError('segmentation label doesnot contain this cell id:', self.cell_id)
        # if everything works, keep segementation_albel and dapi_im for this cell
        else:
            _seg_label = - np.ones(fov_segmentation_label.shape)
            _seg_label[fov_segmentation_label==self.cell_id+1] = 1
            _dapi_im = visual_tools.crop_cell(fov_dapi_im, _seg_label, drift=None)[0]

            if _load_in_ram:
                self.segmentation_label = _seg_label
                self.dapi_im = _dapi_im

        return _seg_label, _dapi_im
    ## Load drift (better load, although de novo drift is allowed)
    def _load_drift(self, _size=500, _force=False, _dynamic=True, _verbose=True):
        # if exists:
        if hasattr(self, 'drift') and not _force:
            if _verbose:
                print(f"- drift already exists for cell:{self.cell_id}, skip")
            return self.drift
        # load color usage if not given
        if not hasattr(self, 'channels'):
            self._load_color_info()
        # scan for candidate drift correction files
        _dft_file_cand = glob.glob(self.drift_folder+os.sep+self.fovs[self.fov_id].replace(".dax", "*.pkl"))
        # if one unique drift file was found:
        if len(_dft_file_cand) == 1:
            _dft = pickle.load(open(_dft_file_cand[0], 'rb'))
            # check drift length
            if len(_dft) == len(self.color_dic):
                if _verbose:
                    print("- length matches, directly load from existing file:", _dft_file_cand[0])
                self.drift = _dft
                return self.drift
            else:
                if _verbose:
                    print("- length doesn't match, proceed to do drift correction.")

        if len(_dft_file_cand) == 0:
            if _verbose:
                print("- no drift result found in drift folder", self.drift_folder)
        # do drift correction from scratch
        if not hasattr(self, 'bead_ims'):
            if _verbose:
                print("- Loading beads images for drift correction")
            _bead_ims, _bead_names = self._load_images('beads', _chromatic_correction=False, _load_in_ram=False)
        else:
             _bead_ims, _bead_names = self.bead_ims, self.bead_names
        # do drift correction
        self.drift, _failed_count = corrections.STD_beaddrift_sequential(_bead_ims, _bead_names,
                                                            self.drift_folder, self.fovs, self.fov_id,
                                                            drift_size=_size, overwrite=_force, dynamic=_dynamic)
        if _failed_count > 0:
            print('Failed drift noticed! total failure:', _failed_count)

        return self.drift
    ## Crop images
    def _load_images(self, _type, _splitted_ims=None,
                     _num_threads=5, _extend_dim=10,
                     _load_in_ram=False, _load_annotated_only=True,
                     _illumination_correction=True, _chromatic_correction=True,
                     _save=True, _overwrite=False, _verbose=False):
        """Core function to load images, support different types:"""
        if not hasattr(self, 'segmentation_label') and _type in ['unique', 'combo', 'sparse']:
            self._load_segmentation()
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info()

        # annotated folders
        if _load_annotated_only:
            _folders = self.annotated_folders
        else:
            _folders = self.folders
        # Case: beads
        if str(_type).lower() == "beads":
            if hasattr(self, 'bead_ims') and hasattr(self, 'bead_names'):
                return self.bead_ims, self.bead_names
            else:
                _bead_ims, _bead_names, _ = analysis.load_image_fov(_folders, self.fovs, self.fov_id,
                                                                    self.channels, self.color_dic,
                                                                    num_threads=_num_threads, loading_type=_type,
                                                                    correction_folder=self.correction_folder,
                                                                    temp_folder=self.temp_folder,
                                                                    return_type='mmap', overwrite_temp=False, verbose=True)
                if _load_in_ram:
                    self.bead_ims = _bead_ims
                    self.bead_names = _bead_names
                return _bead_ims, _bead_names

        # Case: raw
        elif str(_type).lower() == 'raw':
            if hasattr(self, 'splitted_ims'):
                return self.splitted_ims
            elif _splitted_ims:
                if _load_in_ram:
                    self.splitted_ims = _splitted_ims
                return _splitted_ims
            else:
                # Load all splitted images
                _splitted_ims = analysis.load_image_fov(_folders, self.fovs, self.fov_id,
                                                        self.channels, self.color_dic,
                                                        num_threads=_num_threads, loading_type=_type,
                                                        correction_folder=self.correction_folder,
                                                        temp_folder=self.temp_folder,
                                                        return_type='mmap', overwrite_temp=False, verbose=True)
                if _load_in_ram:
                    self.splitted_ims = _splitted_ims
                return _splitted_ims

        # Case: unique images
        elif str(_type).lower() == 'unique':
            if _verbose:
                print(f"Loading unique images for cell:{self.cell_id} in fov:{self.fov_id}")
            # check if images are properly loaded
            if hasattr(self, 'unique_ims') and hasattr(self, 'unique_ids') and hasattr(self, 'unique_channels') and not _overwrite:
                return self.unique_ims, self.unique_ids, self.unique_channels
            elif hasattr(self, 'splitted_ims'):
                _splitted_ims = self.splitted_ims
            elif not _splitted_ims:
                # Load all splitted images
                _splitted_ims = analysis.load_image_fov(_folders, self.fovs, self.fov_id,
                                                        self.channels, self.color_dic,
                                                        num_threads=_num_threads, loading_type=_type,
                                                        correction_folder=self.correction_folder,
                                                        temp_folder=self.temp_folder,
                                                        return_type='mmap', overwrite_temp=False, verbose=True)

            # check drift info
            if not hasattr(self, 'drift'):
                 self._load_drift()

            # initialize
            _unique_ims = []
            _unique_ids = []
            _unique_channels = []
            _unique_marker = 'u'
            # load the images in the order of color_dic (which corresponding to experiment order)
            for _hyb_fd, _info in self.color_dic.items():
                _img_name = _hyb_fd + os.sep + self.fovs[self.fov_id]
                if _img_name in _splitted_ims:
                    if len(_info) != len(_splitted_ims[_img_name]):
                        raise IndexError('information from color_usage doesnot match splitted images.')
                    for _i, (_channel_info, _channel_im) in enumerate(zip(_info, _splitted_ims[_img_name])):
                        if _unique_marker in _channel_info:
                            _uid = int(_channel_info.split(_unique_marker)[-1])
                            if _verbose:
                                print(f"-- loading unique region {_uid} at {_hyb_fd} and color {self.channels[_i]} ")
                            _channel = str(self.channels[_i])
                            # cropping
                            _cropped_im = visual_tools.crop_cell(_channel_im, self.segmentation_label,
                                                                 drift=self.drift[_img_name],
                                                                 extend_dim=_extend_dim)[0]
                            _unique_ims.append(_cropped_im)
                            _unique_ids.append(int(_channel_info.split(_unique_marker)[-1]))
                            _unique_channels.append(_channel)
                else:
                    raise IOError('-- missing image:',_img_name)
            # sort
            _tp = [(_id,_im,_ch) for _id, _im, _ch in sorted(zip(_unique_ids, _unique_ims, _unique_channels))]
            _unique_ims = [_t[1] for _t in _tp]
            _unique_ids = [_t[0] for _t in _tp]
            _unique_channels = [_t[2] for _t in _tp]
            # release:
            if _load_in_ram:
                self.unique_ims = _unique_ims
                self.unique_ids = _unique_ids
                self.unique_channels = _unique_channels
                if _save:
                    _dc = {'unique_ims':_unique_ims,
                           'unique_ids':_unique_ids,
                           'unique_channels': _unique_channels}
                    self._save_to_file('unique', _save_dic=_dc, _overwrite=_overwrite)
            else:
                if _save:
                    _dc = {'unique_ims':_unique_ims,
                           'unique_ids':_unique_ids,
                           'unique_channels': _unique_channels}
                    self._save_to_file('unique', _save_dic=_dc, _overwrite=_overwrite)
            return _unique_ims, _unique_ids, _unique_channels

        elif str(_type).lower() == 'combo' or str(_type).lower() == 'sparse' :
            # check if images are properly loaded
            if hasattr(self, 'splitted_ims'):
                _splitted_ims = self.splitted_ims
            elif not _splitted_ims:
                # Load all splitted images
                _splitted_ims = analysis.load_image_fov(_folders, self.fovs, self.fov_id,
                                                        self.channels, self.color_dic,
                                                        num_threads=_num_threads, loading_type='combo',
                                                        correction_folder=self.correction_folder,
                                                        temp_folder=self.temp_folder,
                                                        return_type='mmap', verbose=True)

            # check drift info
            if not hasattr(self, 'drift'):
                 self._load_drift()
            # check encoding scheme
            if not hasattr(self, 'encoding_scheme'):
                self._load_encoding_scheme()
            # check drift info
            if not hasattr(self, 'drift'):
                 self._load_drift()

            # initialize
            _combo_groups = [] # list to store encoding groups
            _combo_marker = 'c'
            # load the images in the order of color_dic (which corresponding to experiment order)
            for _channel, _encoding_info in self.encoding_scheme.items():
                if _verbose:
                    print("- Loading combo images in channel", _channel)
                # loop through groups in each color
                for _group_id, (_hyb_fds, _matrix) in enumerate(zip(_encoding_info['names'],_encoding_info['matrices'])):
                    if _verbose:
                        print("-- cropping images for group:", _hyb_fds)
                    _combo_images = []
                    for _hyb_fd in _hyb_fds:
                        # check whether this matches color_usage
                        if _hyb_fd not in self.color_dic:
                            raise ValueError('encoding scheme and color usage doesnot match, error in folder:', _hyb_fd)
                        # load images
                        _img_name = _hyb_fd + os.sep + self.fovs[self.fov_id]
                        if _img_name in _splitted_ims:
                            if len(self.color_dic[_hyb_fd]) != len(_splitted_ims[_img_name]):
                                raise IndexError('information from color_usage doesnot match splitted images in combo loading.')
                            # get index
                            _channel_idx = self.channels.index(_channel)
                            # check whether it is marked as combo
                            if _combo_marker not in self.color_dic[_hyb_fd][_channel_idx]:
                                raise ValueError('this', _hyb_fd, "does not correspond to combo in channel", _channel)
                            # get raw image
                            _raw_im = _splitted_ims[_img_name][_channel_idx]
                            # cropping
                            _cropped_im = visual_tools.crop_cell(_raw_im, self.segmentation_label,
                                                                 drift=self.drift[_img_name])[0]
                            _cropped_im = corrections.Z_Shift_Correction(_cropped_im, verbose=False)
                            # store this image
                            _combo_images.append(_cropped_im)
                            _cropped_im=None
                    # create a combo group
                    # special treatment to combo_images:

                    _group = Encoding_Group(_combo_images, _hyb_fds, _matrix, self.save_folder,
                                            self.fov_id, self.cell_id, _channel, _group_id)
                    _combo_groups.append(_group)


            if _load_in_ram:
                self.combo_groups = _combo_groups
                if _save:
                    self._save_to_file('combo', _overwrite=_overwrite)
            else: # not in RAM
                if _save:
                    self._save_to_file('combo', _save_dic={'combo_groups':_combo_groups}, _overwrite=_overwrite)

            return _combo_groups

        # exception: wrong _type key given
        else:
            raise ValueError('wrong image loading type given!, key given:', _type)
        return False
    # saving
    def _save_to_file(self, _type='all', _save_dic={}, _save_folder=None, _overwrite=False, _verbose=True):
        # special keys don't save in the 'cell_info' file. They are images!
        _special_keys = ['unique_ims', 'combo_groups','bead_ims', 'splitted_ims', 'decoded_ims']
        if _save_folder == None:
            _save_folder = self.save_folder
        if not os.path.exists(_save_folder):
            os.makedirs(_save_folder)

        if _type=='all' or _type =='cell_info':
            # save file full name
            _savefile = _save_folder + os.sep + 'cell_info.pkl'
            if _verbose:
                print("- Save cell_info to:", _savefile)
            if os.path.isfile(_savefile) and not _overwrite:
                if _verbose:
                    print("-- loading existing info from file:", _savefile)
                _file_dic = pickle.load(open(_savefile, 'rb'))
            else: # no existing file:
                _file_dic = {} # create an empty dic

            _existing_attributes = [_attr for _attr in dir(self) if not _attr.startswith('_') and _attr not in _special_keys]
            # store updated information
            _updated_info = []
            for _attr in _existing_attributes:
                if _attr not in _special_keys:
                    if _attr not in _file_dic:
                        _file_dic[_attr] = getattr(self, _attr)
                        _updated_info.append(_attr)

            # if specified in save_dic, overwrite
            for _k,_v in _save_dic.items():
                if _k not in _updated_info:
                    _updated_info.append(_k)
                _file_dic[_k] = _v

            if _verbose and len(_updated_info) > 0:
                print("-- information updated in cell_info.pkl:", _updated_info)
            # save info to all
            with open(_savefile, 'wb') as output_handle:
                if _verbose:
                    print("- Writing cell data to file:", _savefile)
                pickle.dump(_file_dic, output_handle)

        # save combo
        if _type =='all' or _type == 'combo':
            if hasattr(self, 'combo_groups'):
                _combo_groups = self.combo_groups
            elif 'combo_groups' in _save_dic:
                _combo_groups = _save_dic['combo_groups']
            else:
                raise ValueError(f'No combo-groups information given in fov:{self.fov_id}, cell:{self.cell_id}')
            for _group in _combo_groups:
                _combo_savefolder = os.path.join(_save_folder,
                                                'group-'+str(_group.group_id),
                                                'channel-'+str(_group.color)
                                                )
                if not os.path.exists(_combo_savefolder):
                    os.makedirs(_combo_savefolder)
                _combo_savefile = _combo_savefolder+os.sep+'rounds.npz'
                # if file exists and not overwriting, skip
                if os.path.exists(_combo_savefile) and not _overwrite:
                    if _verbose:
                        print("file {:s} already exists, skip.".format(_combo_savefile))
                    continue
                # else, write
                _attrs = [_attr for _attr in dir(_group) if not _attr.startswith('_')]
                _combo_dic = {
                    'observation': np.concatenate([_im[np.newaxis,:] for _im in _group.ims]),
                    'encoding': _group.matrix,
                    'names': np.array(_group.names),
                }
                if hasattr(_group, 'readouts'):
                    _combo_dic['readouts'] = np.array(_group.readouts)
                # append chromosome info if exists
                if hasattr(self, 'chrom_coords'):
                    _combo_dic['chrom_coords'] = self.chrom_coords
                # save
                if _verbose:
                    print("-- saving combo to:", _combo_savefile, end="\t")
                    if 'chrom_coords' in _combo_dic:
                        print("with chrom_coords info.")
                    else:
                        print("")
                np.savez_compressed(_combo_savefile, **_combo_dic)
        # save unique
        if _type == 'all' or _type == 'unique':
            _unique_savefile = _save_folder + os.sep + 'unique_rounds.npz'
            # check unique_ims
            if hasattr(self, 'unique_ims'):
                _unique_ims = self.unique_ims
            elif 'unique_ims' in _save_dic:
                _unique_ims = _save_dic['unique_ims']
            else:
                raise ValueError(f'No unique_ims information given in fov:{self.fov_id}, cell:{self.cell_id}')
            # check unique_ids
            if hasattr(self, 'unique_ids'):
                _unique_ids = self.unique_ids
            elif 'unique_ids' in _save_dic:
                _unique_ids = _save_dic['unique_ids']
            else:
                raise ValueError(f'No unique_ids information given in fov:{self.fov_id}, cell:{self.cell_id}')
            # check unique_channels
            if hasattr(self, 'unique_channels'):
                _unique_channels = self.unique_channels
            elif 'unique_channels' in _save_dic:
                _unique_channels = _save_dic['unique_channels']
            else:
                raise ValueError(f'No unique_channels information given in fov:{self.fov_id}, cell:{self.cell_id}')

            _unique_dic = {
                'observation': np.concatenate([_im[np.newaxis,:] for _im in _unique_ims]),
                'ids': np.array(_unique_ids),
                'channels': np.array(_unique_channels)
            }
            # save
            if _verbose:
                print("-- saving unique to:", _unique_savefile)
            np.savez_compressed(_unique_savefile, **_unique_dic)

    def _load_from_file(self, _type='all', _save_folder=None, _decoded_flag=None, _load_attrs=[],
                        _overwrite=False, _verbose=True):
        """ Function to load cell_data from existing npz and pickle files
        Inputs:
            _type: 'all'/'combo'/'unique'/'decoded', string
            _save_folder: where did the files save, None or path string (default: None)
            _load_attrs: list of additional attributes that want to load to RAM, list of string (default: [])
            _overwrite: whether overwrite existing attributes in class, bool (default: false)
            _verbose: say something!, bool (default: True)
        (everything will be loaded into class attributes, so no return )
        """
        # check input
        _type=str(_type).lower()
        if _type not in ['all', 'cell_info', 'unique', 'combo', 'decoded']:
            raise ValueError("Wrong _type kwd given!")
        if not _save_folder and not hasattr(self, 'save_folder'):
            raise ValueError('Save folder info not given!')
        elif not _save_folder:
            _save_folder = self.save_folder

        if _type == 'all' or _type == 'cell_info':
            _infofile = _save_folder + os.sep + 'cell_info.pkl'
            _info_dic = pickle.load(open(_infofile, 'rb'))
            #loading keys from info_dic
            for _key, _value in _info_dic.items():
                if not hasattr(self, _key) or _overwrite:
                    # if (load_attrs) specified:
                    if len(_load_attrs) > 0 and _key in _load_attrs:
                        setattr(self, _key, _value)
                    # no load_attr specified
                    elif len(_load_attrs) == 0:
                        setattr(self, _key, _value)

        if _type == 'all' or _type == 'combo':
            if not hasattr(self, 'combo_groups'):
                self.combo_groups = []
            elif not isinstance(self.combo_groups, list):
                raise TypeError('Wrong datatype for combo_groups attribute for cell data.')

            # load existing combo files
            _raw_combo_fl = "rounds.npz"
            _combo_files = glob.glob(os.path.join(_save_folder, "group-*", "channel-*", _raw_combo_fl))
            for _combo_file in _combo_files:
                if _verbose:
                    print("-- loading combo from file:", _combo_file)
                with np.load(_combo_file) as handle:
                    _ims = list(handle['observation'])
                    _matrix = handle['encoding']
                    _names = handle['names']
                    if 'readout' in handle.keys():
                        _readouts = handle['readouts']
                    if 'chrom_coords' in handle.keys():
                        _chrom_coords = handle['chrom_coords']
                        if not hasattr(self, 'chrom_coords'):
                            self.chrom_coords = _chrom_coords
                _name_info = _combo_file.split(os.sep)
                _fov_id = [int(_i.split('-')[-1]) for _i in _name_info if "fov" in _i][0]
                _cell_id = [int(_i.split('-')[-1]) for _i in _name_info if "cell" in _i][0]
                _group_id = [int(_i.split('-')[-1]) for _i in _name_info if "group" in _i][0]
                _color = [_i.split('-')[-1] for _i in _name_info if "channel" in _i][0]
                # check duplication
                _check_duplicate = [(_g.fov_id==_fov_id) and (_g.cell_id==_cell_id) and (_g.group_id==_group_id) and (_g.color==_color) for _g in self.combo_groups]
                if sum(_check_duplicate) > 0: #duplicate found:
                    if not _overwrite:
                        if _verbose:
                            print("---", _combo_file.split(_save_folder)[-1], "already exists in combo_groups, skip")
                        continue
                    else:
                        self.combo_groups.pop(_check_duplicate.index(True))
                # create new group
                if '_readouts' in locals():
                    _group = Encoding_Group(_ims, _names, _matrix, _save_folder,
                                            _fov_id, _cell_id, _color, _group_id, _readouts)
                else:
                    _group = Encoding_Group(_ims, _names, _matrix, _save_folder,
                                            _fov_id, _cell_id, _color, _group_id)
                # append
                self.combo_groups.append(_group)

        if _type == 'all' or _type == 'unique':
            _unique_fl = 'unique_rounds.npz'
            _unique_savefile = _save_folder + os.sep + _unique_fl
            if not os.path.exists(_unique_savefile):
                print(f"- savefile {_unique_savefile} not exists, exit.")
                return False
            if _verbose:
                print("- Loading unique from file:", _unique_savefile)
            with np.load(_unique_savefile) as handle:
                _unique_ims = list(handle['observation'])
                _unique_ids = list(handle['ids'])
                _unique_channels = list(handle['channels'])
            # save
            if not hasattr(self, 'unique_ids') or not hasattr(self, 'unique_ims'):
                self.unique_ids, self.unique_ims, self.unique_channels = [], [], []
            elif len(self.unique_ims) == 0 or len(self.unique_ids) == 0:
                self.unique_ids, self.unique_ims, self.unique_channels = [], [], []
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
                    self.unique_ims[self.unique_ids.index(int(_uid))] = _uim

        if _type == 'all' or _type == 'decoded':
            if _type == 'decoded' and not _decoded_flag:
                raise ValueError("Kwd _decoded_flag not given, exit!")
            elif not _decoded_flag:
                print("Kwd _decoded_flag not given, skip this step.")
            else:
                # check combo_groups
                # look for decoded _results
                if not hasattr(self, 'combo_groups'):
                    _temp_flag = True
                    if _verbose:
                        print(f"No combo groups loaded in fov:{self.fov_id}, cell:{self.cell_id}, start loading combo!")
                    self._load_from_file('combo', _overwrite=_overwrite, _verbose=_verbose)
                else:
                    _temp_flag = False
                # scan existing combo files
                _raw_decoded_fl = "regions.npz"
                _decoded_files = glob.glob(os.path.join(_save_folder, "group-*", "channel-*", _decoded_flag, _raw_decoded_fl))
                # intialize list to store images
                _decoded_ims, _decoded_ids = [],[]
                # loop through files and
                for _decoded_file in _decoded_files:
                    if _verbose:
                        print("-- loading decoded result from file:", _decoded_file)
                    with np.load(_decoded_file) as handle:
                        _ims = handle['observation']
                        _ims = _ims.swapaxes(0,3).swapaxes(1,2)
                    _name_info = _decoded_file.split(os.sep)
                    _fov_id = [int(_i.split('-')[-1]) for _i in _name_info if "fov" in _i][0]
                    _cell_id = [int(_i.split('-')[-1]) for _i in _name_info if "cell" in _i][0]
                    _group_id = [int(_i.split('-')[-1]) for _i in _name_info if "group" in _i][0]
                    _color = [_i.split('-')[-1] for _i in _name_info if "channel" in _i][0]
                    # check duplication
                    _matches = [(_g.fov_id==_fov_id) and (_g.cell_id==_cell_id) and (_g.group_id==_group_id) and (_g.color==_color) for _g in self.combo_groups]
                    if sum(_matches) == 1: #duplicate found:
                        if _verbose:
                            print(f"--- decoded result matched for group:{_group_id}, color:{_color}")
                        _matched_group = self.combo_groups[_matches.index(True)]
                        _matrix = _matched_group.matrix
                        _n_hyb, _n_reg = np.shape(_matrix)
                        _ims = _ims[:_n_reg]
                        _ids = [np.unique(_matrix[:,_col_id])[-1] for _col_id in range(_n_reg)]
                        # filter by reg name
                        _kept_ims = [_im for _im,_id in zip(_ims, _ids) if _id >=0]
                        _kept_ids = [_id for _im,_id in zip(_ims, _ids) if _id >=0]
                        # append
                        _decoded_ids += _kept_ids
                        _decoded_ims += _kept_ims
                        print("--- kept ids:",_kept_ids)
                # check with combo groups
                for _group in self.combo_groups:
                    _ids = [np.unique(_group.matrix[:,_col_id])[-1] for _col_id in range(_group.matrix.shape[1])]
                    for _id in _ids:
                        if _id not in _decoded_ids and _id > 0:
                            if _verbose:
                                print(f"--- filling up not decoded region:{_id}")
                            _decoded_ids.append(_id)
                            _decoded_ims.append(None)
                # sort
                self.decoded_ims = [_im for _id,_im in sorted(zip(_decoded_ids, _decoded_ims))]
                self.decoded_ids = [_id for _id in sorted(_decoded_ids)]
                if _temp_flag:
                    delattr(self, 'combo_groups')

    # Generate pooled image representing chromosomes
    def _generate_chromosome_image(self, _source='combo', _max_count=40, _verbose=False):
        """Generate chromosome from existing combo / unique images"""
        _source = _source.lower()
        if _source != 'combo' and _source != 'unique':
            raise ValueError('wrong source key given, should be combo or unique. ')
        if _source == 'combo':
            if not hasattr(self, 'combo_groups'):
                _temp_flag = True # this means the combo groups images are temporarily loaded
                print('-- cell_data doesnot have combo images, trying to load now.')
                self._load_from_file('combo', _verbose=False)
            else:
                _temp_flag = False
            # sum up existing Images
            _image_count = 0
            _chrom_im = np.zeros(np.shape(self.combo_groups[0].ims[0]))
            for _group in self.combo_groups:
                _chrom_im += sum(_group.ims)
                _image_count += len(_group.ims)
                if _max_count > 0 and _image_count > _max_count:
                    break
            _chrom_im = _chrom_im / _image_count

        elif _source == 'unique':
            if not hasattr(self, 'unique_ims') or not hasattr(self, 'unique_ids'):
                _temp_flag = True # this means the unique images are temporarily loaded
                print('-- cell_data doesnot have unique images, trying to load now.')
                self._load_from_file('unique', _verbose=False)
            else:
                _temp_flag = False                
            # sum up existing Images
            _picking_freq = int(np.ceil(len(self.unique_ims)/_max_count))
            _selected_ims = self.unique_ims[::_picking_freq]
            _chrom_im = np.mean(np.stack(_selected_ims), axis=0)

        # final correction
        _chrom_im = corrections.Z_Shift_Correction(_chrom_im)
        _chrom_im = corrections.Remove_Hot_Pixels(_chrom_im)
        self.chrom_im = _chrom_im
        if _temp_flag: # if temp loaded, release
            if _source == 'combo':
                delattr(self, 'combo_groups')
            elif _source == 'unique':
                delattr(self, 'unique_ims')
        return _chrom_im
    # Identify chromosome(generated by _generate_chromosome_image)
    def _identify_chromosomes(self, _gaussian_size=2, _cap_percentile=1, _seed_dim=3,
                              _th_percentile=99.5, _min_obj_size=125, _verbose=True):
        """Function to identify chromsome automatically first"""
        if not hasattr(self, 'chrom_im'):
            self._generate_chromosome_image()
        _chrom_im = np.zeros(np.shape(self.chrom_im), dtype=np.uint8) + self.chrom_im
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
            _label, _num = ndimage.label(_close_objects)
            _label[_label==0] = -1
            # segmentation
            _seg_label = random_walker(_chrom_im, _label, beta=100, mode='bf')
            # keep object
            _kept_label = -1 * np.ones(_seg_label.shape, dtype=np.int)
            _sizes = [np.sum(_seg_label==_j+1) for _j in range(np.max(_seg_label))]
            # re-label
            _label_ct = 1
            for _i, _size in enumerate(_sizes):
                if _size > _min_obj_size: # then save this label
                    _kept_label[_seg_label == _i+1] = _label_ct
                    _label_ct += 1
            _chrom_coords = [ndimage.measurements.center_of_mass(_kept_label==_j+1) for _j in range(np.max(_kept_label))]
            # store
            self.chrom_segmentation = _kept_label
            self.chrom_coords = _chrom_coords

        return _kept_label, _chrom_coords

    def _pick_chromosome_manual(self, _save_folder=None, _save_fl='chrom_coord.pkl'):
        if not _save_folder:
            if hasattr(self, 'save_folder'):
                _save_folder = self.save_folder
            else:
                raise ValueError('save_folder not given in keys and attributes.')

        _chrom_savefile = os.path.join(_save_folder, _save_fl.replace('.pkl', '_'+str(self.fov_id)+'_'+str(self.cell_id)+'.pkl'))
        if not hasattr(self, 'chrom_coords'):
            raise ValueError("chromosome coordinates doesnot exist in attributes.")
        _coord_dic = {'coords': [np.flipud(_coord) for _coord in self.chrom_coords],
                      'class_ids': list(np.zeros(len(self.chrom_coords),dtype=np.uint8)),
                      'pfits':{},
                      'dec_text':{},
                      }
        #pickle.dump(_coord_dic, open(_chrom_savefile, 'wb'))
        _viewer = visual_tools.imshow_mark_3d_v2([self.chrom_im], image_names=['chromosome'],
                                                 save_file=_chrom_savefile, given_dic=_coord_dic)
        return _viewer

    def _update_chromosome_from_file(self, _save_folder=None, _save_fl='chrom_coord.pkl', _verbose=True):
        if not _save_folder:
            if hasattr(self, 'save_folder'):
                _save_folder = self.save_folder
            else:
                raise ValueError('save_folder not given in keys and attributes.')
        _chrom_savefile = os.path.join(_save_folder, _save_fl.replace('.pkl', '_'+str(self.fov_id)+'_'+str(self.cell_id)+'.pkl'))
        _coord_dic = pickle.load(open(_chrom_savefile, 'rb'))
        _chrom_coords = [np.flipud(_coord) for _coord in _coord_dic['coords']]
        if _verbose:
            print(f"-- {len(_chrom_coords)} loaded")
        self.chrom_coords = _chrom_coords

        return _chrom_coords

    def _multi_fitting(self, _type='unique',_use_chrom_coords=True, _num_threads=5,
                       _seed_th_per=50., _max_filt_size=3, _max_seed_count=0, _min_seed_count=1,
                       _width_zxy=None, _fit_radius=10, _expect_weight=500, _min_height=100, _max_iter=10, _th_to_end=1e-5,
                       _save=True, _verbose=True):
        # first check Inputs
        _allowed_types = ['unique', 'decoded']
        _type = _type.lower()
        if _type not in _allowed_types:
            raise KeyError(f"Wrong input key for _type:{_type}")
        if _width_zxy is None:
            _width_zxy = self.sigma_zxy
        if _use_chrom_coords:
            if not hasattr(self, 'chrom_coords'):
                self._load_from_file('cell_info')
                if not hasattr(self, 'chrom_coords'):
                    raise AttributeError("No chrom-coords info found in cell-data and saved cell_info.")
            if _verbose:
                print(f"+ Start multi-fitting for {_type} images")
            # check specific attributes
            if _type == 'unique':
                # check attributes
                if not hasattr(self, 'unique_ims') or not hasattr(self, 'unique_ids'):
                    _temp_flag = True # this means the unique images are temporarily loaded
                    print("++ no unique image info loaded to this cell, try loading:")
                    self._load_from_file('unique', _overwrite=False, _verbose=_verbose)
                else:
                    _temp_flag = False # not temporarily loaded
                _ims = self.unique_ims
                _ids = self.unique_ids
            elif _type == 'decoded':
                # check attributes
                if not hasattr(self, 'decoded_ims') or not hasattr(self, 'decoded_ids'):
                    _temp_flag = True  # this means the unique images are temporarily loaded
                    print("++ no decoded image info loaded to this cell, try loading:")
                    self._load_from_file(
                        'decoded', _overwrite=False, _verbose=_verbose)
                else:
                    _temp_flag = False  # not temporarily loaded
                _ims = self.decoded_ims
                _ids = self.decoded_ids

            ## Do the multi-fitting
            if _type == 'unique':
                _seeding_args = (_max_seed_count, 20, 0, _max_filt_size, _seed_th_per, True, 10, _min_seed_count, 0, False)
                _fitting_args = (_width_zxy, _fit_radius, 100, 500, _expect_weight, _th_to_end, _max_iter, 0.25, _min_height, False, _verbose)
            elif _type == 'decoded':
                _seeding_args = (_max_seed_count, 20, 0, _max_filt_size, _seed_th_per, True, 10, _min_seed_count, 0, False)
                _fitting_args = (_width_zxy, _fit_radius, 0.1, 0.5, _expect_weight/1000, _th_to_end, _max_iter, 0.25, 0.1, False, _verbose)

            _args = [(_im, _id, self.chrom_coords, _seeding_args, _fitting_args, _verbose) for _im, _id in zip(_ims, _ids)]
            # multi-processing for multi-Fitting
            if _verbose:
                print(f"++ start fitting {_type} for fov:{self.fov_id}, cell:{self.cell_id} with {_num_threads} threads")
            _start_time = time.time()
            _fitting_pool = multiprocessing.Pool(_num_threads)
            _spots = _fitting_pool.starmap(_fit_single_image, _args, chunksize=1)
            _fitting_pool.close()
            _fitting_pool.join()
            _fitting_pool.terminate()
            # release
            _ims, _ids = None, None
            if _temp_flag:
                if _type == 'unique':
                    delattr(self, 'unique_ims')
                elif _type == 'decoded':
                    delattr(self, 'decoded_ims')
            if _verbose:
                print(f"++ total time in fitting {_type}: {time.time()-_start_time}")
            ## return and save
            if _type == 'unique':
                self.unique_spots = _spots
                if _save:
                    self._save_to_file('cell_info',_save_dic={'unique_spots':self.unique_spots})
                return self.unique_spots
            elif _type == 'decoded':
                self.decoded_spots = _spots
                if _save:
                    self._save_to_file('cell_info',_save_dic={'decoded_spots':self.decoded_spots})
                return self.decoded_spots

    def _dynamic_picking_spots(self, _type='unique', _use_chrom_coords=True,
                               _distance_zxy=None, _w_int=1, _w_dist=2,
                               _dist_ref = None, _penalty_type='trapezoidal', _penalty_factor=5,
                               _save=True, _verbose=True):
        """Given selected spots, do picking by dynamic programming
        Input:"""
        ## check inputs
        if _distance_zxy is None: # dimension for distance trasfromation
            _distance_zxy = self.distance_zxy
        if _dist_ref is None: # full filename for distance reference npz file
            _dist_ref = self.distance_reference
        # load distance_reference:
        with np.load(_dist_ref) as data:
            ref_matrix = data['distance_map']
        # penalty function for distance
        def distance_penalty(real_dist, exp_dist, __type='trapezoidal', _factor=5):
            """Penalty function for distance, given matrix of real_dist and float exp_dist, return [0,1] penalty funciton"""
            real_dist = np.array(real_dist)
            if __type == 'gaussian':
                return np.exp(-(real_dist-exp_dist)**2/2/(exp_dist/_factor)**2)
            elif __type == 'spike':
                return np.max(np.stack([(exp_dist-np.abs(real_dist-exp_dist)/_factor)/exp_dist, np.zeros(real_dist.shape)]),0)
            elif __type == 'trapezoidal':
                return np.max(np.stack([np.zeros(real_dist.shape)**(real_dist>exp_dist),(exp_dist-np.abs(real_dist-exp_dist)/_factor)/exp_dist, np.zeros(real_dist.shape)]),0)
            elif __type == 'triangle':
                return np.max(np.stack([1-real_dist/exp_dist/_factor, np.zeros(real_dist.shape)]), 0)
            else:
                raise KeyError("Wrong input __type kwd!")
        # if isolate chromosomes:
        if _use_chrom_coords:
            if not hasattr(self, 'chrom_coords'):
                self._load_from_file('cell_info')
                if not hasattr(self, 'chrom_coords'):
                    raise AttributeError("No chrom-coords info found in cell-data and saved cell_info.")
            # check specific attributes and initialize
            if _type == 'unique':
                # check attributes
                if not hasattr(self, 'unique_spots'):
                    self._load_from_file('cell_info')
                    if not hasattr(self, 'unique_spots'):
                        raise AttributeError("No unique_spots info found in cell-data and saved cell_info.")
                if _verbose:
                    print(f"+ Pick {_type} spots for by brightness in fov:{self.fov_id}, cell:{self.cell_id}")
                # spots for unique:
                _cand_spots = self.unique_spots
                _ids = self.unique_ids
                _picked_spots = [] # initialize
            elif _type == 'decoded':
                # check attributes
                if not hasattr(self, 'decoded_spots'):
                    self._load_from_file('cell_info')
                    if not hasattr(self, 'decoded_spots'):
                        raise AttributeError("No decoded_spots info found in cell-data and saved cell_info.")
                if _verbose:
                    print(f"+ Pick {_type} spots for by brightness in fov:{self.fov_id}, cell:{self.cell_id}")
                # spots for unique:
                _cand_spots = self.decoded_spots
                _ids = self.decoded_ids
                _picked_spots = [] # initialize

            ## dynamic progamming
            for _chrom_id, _coord in enumerate(self.chrom_coords):
                _ch_pts = [chrpts[_chrom_id][:,1:4]*_distance_zxy for chrpts in _cand_spots if len(chrpts[_chrom_id]>0)]
                _ch_ids = [_id for chrpts,_id in zip(_cand_spots, _ids) if len(chrpts[_chrom_id]>0)]
                # initialize two stucture:
                _dy_values = [np.log(chrpts[_chrom_id][:,0]*_w_int) for chrpts in _cand_spots if len(chrpts[_chrom_id]>0)] # store maximum values
                _dy_pointers = [-np.ones(len(pt), dtype=np.int) for pt in _ch_pts] # store pointer to previous level
                # Forward
                for _j, (_pts, _id) in enumerate(zip(_ch_pts[1:], _ch_ids[1:])):
                    _dists = cdist(_ch_pts[_j], _ch_pts[_j+1]) # real pair-wise distance
                    _ref_dist = ref_matrix[_ch_ids[_j]-1, _ch_ids[_j+1]-1] # distance inferred by Hi-C as prior
                    # two components in dynamic progamming: distance and intensity
                    _measure =  distance_penalty(_dists, _ref_dist, _penalty_type, _penalty_factor) * _w_dist + _dy_values[_j][:,np.newaxis]
                    # update maximum values and maximum pointers
                    _dy_values[_j+1] += np.max(_measure, axis=0)
                    _dy_pointers[_j+1] = np.argmax(_measure, axis=0)
                # backward
                if len(_dy_values) > 0:
                    _picked_ids = [np.argmax(_dy_values[-1])]
                else:
                    _picked_ids = []
                for _j in range(len(_ch_pts)-1):
                    _picked_ids.append(_dy_pointers[-(_j+1)][_picked_ids[-1]])
                _picked_ids = np.flip(_picked_ids, axis=0)
                # clean up and match candidate spots
                _picked_chrom_pts = []
                _counter = 0
                for _j, (_cand_list, _id) in enumerate(zip(_cand_spots, _ids)):
                    _cands = _cand_list[_chrom_id]
                    if len(_cands) > 0 and _id in _ch_ids:
                        _picked_chrom_pts.append(_cands[_picked_ids[_counter]])
                        _counter += 1
                    else:
                        _picked_chrom_pts.append(np.inf*np.ones(8))
                _picked_spots.append(_picked_chrom_pts)

            ## dump into attribute and save
            if _type == 'unique':
                self.picked_unique_spots = _picked_spots
                # save
                if _save:
                    self._save_to_file('cell_info', _save_dic={'picked_unique_spots':self.picked_unique_spots})
                # return
                return self.picked_unique_spots
            if _type == 'decoded':
                self.picked_decoded_spots = _picked_spots
                # save
                if _save:
                    self._save_to_file('cell_info', _save_dic={'picked_decoded_spots':self.picked_decoded_spots})
                # return
                return self.picked_decoded_spots

    def _naive_picking_spots(self, _type='unique', _use_chrom_coords=True,
                             _save=True, _verbose=True):
        """Given selected spots, do picking by the brightness
        Input:"""
        if _use_chrom_coords:
            if not hasattr(self, 'chrom_coords'):
                self._load_from_file('cell_info')
                if not hasattr(self, 'chrom_coords'):
                    raise AttributeError("No chrom-coords info found in cell-data and saved cell_info.")
            # check specific attributes and initialize
            if _type == 'unique':
                # check attributes
                if not hasattr(self, 'unique_spots'):
                    self._load_from_file('cell_info')
                    if not hasattr(self, 'unique_spots'):
                        raise AttributeError("No unique_spots info found in cell-data and saved cell_info.")
                if _verbose:
                    print(f"+ Pick {_type} spots for by brightness in fov:{self.fov_id}, cell:{self.cell_id}")
                # spots for unique:
                _cand_spots = self.unique_spots
                _ids = self.unique_ids
                _picked_spots = [] # initialize
            elif _type == 'decoded':
                # check attributes
                if not hasattr(self, 'decoded_spots'):
                    self._load_from_file('cell_info')
                    if not hasattr(self, 'decoded_spots'):
                        raise AttributeError("No decoded_spots info found in cell-data and saved cell_info.")
                if _verbose:
                    print(f"+ Pick {_type} spots for by brightness in fov:{self.fov_id}, cell:{self.cell_id}")
                # spots for unique:
                _cand_spots = self.decoded_spots
                _ids = self.decoded_ids
                _picked_spots = [] # initialize

            # picking spots
            for _chrom_id, _chrom_coord in enumerate(self.chrom_coords):
                _picked_in_chrom = []
                for _cand_lst, _id in zip(_cand_spots, _ids):
                    # extract candidate spots for this
                    _cands = _cand_lst[_chrom_id]
                    # case 1: no fit at all:
                    if len(_cands) == 0:
                        _picked_in_chrom.append(np.inf*np.ones(8))
                    else:
                        _intensity_order = np.argsort(_cands[:,0])
                        # PICK THE BRIGHTEST ONE
                        _pspt = _cands[_intensity_order[-1]]
                        _picked_in_chrom.append(_pspt)
                # append
                _picked_spots.append(_picked_in_chrom)

            ## dump into attribute and save
            if _type == 'unique':
                self.picked_unique_spots = _picked_spots
                # save
                if _save:
                    self._save_to_file('cell_info', _save_dic={'picked_unique_spots':self.picked_unique_spots})
                # return
                return self.picked_unique_spots
            if _type == 'decoded':
                self.picked_decoded_spots = _picked_spots
                # save
                if _save:
                    self._save_to_file('cell_info', _save_dic={'picked_decoded_spots':self.picked_decoded_spots})
                # return
                return self.picked_decoded_spots

    def _match_regions(self, _save=True, _save_map=True):
        """Function to match decoded and unique regions and generate matched ids, spots, distance maps etc.
        Inputs:
            _save: whether save matched info to cell_info, bool (default:True)
            _save_map: whether save matched distance map, bool (default:True)
        """
        if not hasattr(self, 'decoded_ids'):
            pass

    def _generate_distance_map(self, _type='unique', _distance_zxy=None, _save_info=True, _save_plot=True,
                               _limits=[200,1000], _verbose=True):
        """Function to generate distance map"""
        ## check inputs
        if _distance_zxy is None: # dimension for distance trasfromation
            _distance_zxy = self.distance_zxy
        if not hasattr(self, 'chrom_coords'):
            self._load_from_file('cell_info')
            if not hasattr(self, 'chrom_coords'):
                raise AttributeError("No chrom_coords info found in cell-data and saved cell_info.")
        ## check specific attributes and initialize
        if _type == 'unique':
            # check attributes
            if not hasattr(self, 'picked_unique_spots'):
                self._load_from_file('cell_info')
                if not hasattr(self, 'picked_unique_spots'):
                    raise AttributeError("No picked_unique_spots info found in cell-data and saved cell_info.")
            _picked_spots = self.picked_unique_spots
        elif _type == 'decoded' or _type == 'combo':
            # check attributes
            if not hasattr(self, 'picked_decoded_spots'):
                self._load_from_file('cell_info')
                if not hasattr(self, 'picked_decoded_spots'):
                    raise AttributeError("No picked_decoded_spots info found in cell-data and saved cell_info.")
            _picked_spots = self.picked_decoded_spots

        ## loop through chrom_coords and make distance map
        _distance_maps = []
        for _chrom_id, _coord in enumerate(self.chrom_coords):
            if _verbose:
                print(f"++ generate {_type} dist-map for fov:{self.fov_id}, cell:{self.cell_id}, chrom:{_chrom_id}")
            # get coordiates
            _coords_in_pxl = np.stack([s[1:4] for s in _picked_spots[_chrom_id]]) # extract only coordnates
            # convert to nm
            _coords_in_nm = _coords_in_pxl * _distance_zxy
            # calculate dist-map
            _distmap = squareform(pdist(_coords_in_nm))
            # append
            _distance_maps.append(_distmap)
            # make plot
            plt.figure()
            plt.title(f"{_type} dist-map for fov:{self.fov_id}, cell:{self.cell_id}, chrom:{_chrom_id}")
            plt.imshow(_distmap, interpolation='nearest', cmap=seismic_r, vmin=np.min(_limits), vmax=np.max(_limits))
            plt.colorbar(ticks=range(np.min(_limits),np.max(_limits),200), label='distance (nm)')
            if _save_plot:
                if not os.path.exists(os.path.join(self.map_folder, self.fovs[self.fov_id].replace('.dax','')) ):
                    if _verbose:
                        print(f"++ Make directory:",os.path.join(self.map_folder, self.fovs[self.fov_id].replace('.dax','')))
                    os.makedirs(os.path.join(self.map_folder, self.fovs[self.fov_id].replace('.dax','')))
                _save_fl = os.path.join(self.map_folder,
                                        self.fovs[self.fov_id].replace('.dax',''),
                                        f"dist-map_{_type}_{self.cell_id}_{_chrom_id}.png")
                plt.savefig(_save_fl, transparent=True)
            plt.show()

        ## dump into attribute and save
        if _type == 'unique':
            self.unique_distance_map = _distance_maps
            # save
            if _save_info:
                self._save_to_file('cell_info', _save_dic={'unique_distance_map':self.unique_distance_map})
            # return
            return self.unique_distance_map
        elif _type == 'decoded' or _type == 'combo':
            self.decoded_distance_map = _distance_maps
            # save
            if _save_info:
                self._save_to_file('cell_info', _save_dic={'decoded_distance_map':self.decoded_distance_map})
            # return
            return self.decoded_distance_map




class Encoding_Group():
    """defined class for each group of encoded images"""
    def __init__(self, ims, hybe_names, encoding_matrix, save_folder,
                 fov_id, cell_id, color, group_id, readouts=None):
        # info for this cell
        self.ims = ims
        self.names = hybe_names
        self.matrix = encoding_matrix
        self.save_folder = save_folder
        # detailed info for this group
        self.fov_id = fov_id
        self.cell_id = cell_id
        self.color = color
        self.group_id = group_id
        if readouts:
            self.readouts = readouts
    def _save_group(self, _overwrite=False, _verbose=True):
        _combo_savefolder = os.path.join(self.save_folder,
                                        'group-'+str(self.group_id),
                                        'channel-'+str(self.color)
                                        )
        if not os.path.exists(_combo_savefolder):
            os.makedirs(_combo_savefolder)
        _combo_savefile = _combo_savefolder+os.sep+'rounds.npz'
        # if file exists and not overwriting, skip
        if os.path.exists(_combo_savefile) and not _overwrite:
            if _verbose:
                print("file {:s} already exists, skip.".format(_combo_savefile))
            return False
        # else, write
        _attrs = [_attr for _attr in dir(self) if not _attr.startswith('_')]
        _combo_dic = {
            'observation': np.concatenate([_im[np.newaxis,:] for _im in self.ims]),
            'encoding': self.matrix,
            'names': np.array(self.names),
        }
        if hasattr(self, 'readouts'):
            _combo_dic['readouts'] = np.array(self.readouts)
        # save
        if _verbose:
            print("-- saving combo to:", _combo_savefile)
        np.savez_compressed(_combo_savefile, **_combo_dic)
        return True
