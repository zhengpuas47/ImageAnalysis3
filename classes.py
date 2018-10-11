import sys,glob,os,time, copy
import numpy as np
sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')
import pickle as pickle
import matplotlib.pyplot as plt
from . import get_img_info, corrections, visual_tools, analysis
import multiprocessing
from scipy import ndimage
from scipy import stats
from skimage import morphology
from skimage.segmentation import random_walker

# global variables
_correction_folder=r'E:\Users\puzheng\Documents\Corrections'
_temp_folder = r'I:\Pu_temp'

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
        # temp_folder
        if 'temp_folder' in parameters:
            self.temp_folder = parameters['temp_folder'];
        else:
            self.temp_folder = _temp_folder
        ## analysis_folder, segmentation_folder, save_folder, correction_folder,map_folder
        if 'analysis_folder'  in parameters:
            self.analysis_folder = str(parameters['analysis_folder']);
        else:
            self.analysis_folder = self.data_folder[0]+os.sep+'Analysis'
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
            self.correction_folder = _correction_folder
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
        # read color-usage and encodding-scheme
        self._load_color_info();
        self._load_encoding_scheme();
        # get annotated folders by color usage
        self.annotated_folders = [];
        for _hyb_fd, _info in self.color_dic.items():
            _matches = [_fd for _fd in self.folders if _hyb_fd in _fd];
            if len(_matches)==1:
                self.annotated_folders.append(_matches[0])
        print(f"{len(self.annotated_folders)} folders are found according to color-usage annotation.")


    # allow print info of Cell_List
    def __str__(self):
        if hasattr(self, 'data_folder'):
            print("Data folder:", self.data_folder);
        if hasattr(self, 'cells'):
            print("Number of cells in this list:", len(self.cells));
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

    def _pick_cell_segmentations(self, _num_threads=None, _allow_manual=True,
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

    def _create_cell(self, _parameter, _load_info=True,
                     _load_segmentation=True, _load_drift=True,
                     _load_file=True,
                     _save=False, _append_cell_list=False):
        """Function to create one cell_data object"""
        _cell = Cell_Data(_parameter, _load_all_attr=True)
        if _load_info:
            _cell._load_color_info();
            _cell._load_encoding_scheme();
        if _load_segmentation:
            _cell._load_segmentation();
        if _load_drift:
            _cell._load_drift();
        if _load_file:
            if os.path.isfile(os.path.join(_cell.save_folder, 'cell_info.pkl')):
                _cell._load_from_file('cell_info', _overwrite=False, _verbose=True)
        if _save:
            _cell._save_to_file('cell_info')

        # whether directly store
        if _append_cell_list:
            self.cells.append(_cell);

        return _cell

    def _create_cells_fov(self, _fov_id, _num_threads=None, _load_annotated_only=True, _overwrite_temp=True,
                          _drift_size=550, _drift_dynamic=True, _verbose=True):
        """Create Cele_data objects for one field of view"""
        if not _num_threads:
            _num_threads = int(self.num_threads);
        if _fov_id not in self.fov_ids:
            raise ValueError("Wrong fov_id kwd given! \
                this should be real fov-number that allowed during intiation of class.")
        if _verbose:
            print("+ Create Cell_Data objects for field of view:", self.fovs[_fov_id], _fov_id);
            print("++ preparing variables");
        # check attributes
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info()
        # whether load annotated hybs only
        if _load_annotated_only:
            _folders = self.annotated_folders;
        else:
            _folders = self.folders;
        # load segmentation for this fov
        if _verbose:
            print("+ Load segmentation for fov", _fov_id)
        # do segmentation if necessary, or just load existing segmentation file
        _fov_segmentation_label, _fov_dapi_im  = analysis.Segmentation_Fov(self.analysis_folder,
                                                _folders, self.fovs, _fov_id,
                                                num_channel=len(self.channels),
                                                dapi_channel=self.dapi_channel_index,
                                                illumination_corr=True,
                                                correction_folder=self.correction_folder,
                                                segmentation_path=os.path.basename(self.segmentation_folder),
                                                save=True, force=False, verbose=_verbose)
        plt.figure()
        plt.imshow(_fov_segmentation_label)
        # do drift for this fov
        if _verbose:
            print("+ Drift correction for fov", _fov_id);
        _drift_fl = self.drift_folder+os.sep+self.fovs[_fov_id].replace('.dax', "*_cor.pkl");
        _drift_fl = glob.glob(_drift_fl);
        if len(_drift_fl) == 1 and os.path.isfile(_drift_fl[0]):
            _drift_fl = _drift_fl[0];
            if _verbose:
                print("++ Directly load drift correction from file:", _drift_fl)
            _fov_drift = pickle.load(open(_drift_fl, 'rb'))
        else:
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
            _fov_drift, _, _failed_count = corrections.STD_beaddrift_sequential(_bead_ims, _bead_names,
                                                                                self.drift_folder,
                                                                                self.fovs, _fov_id,
                                                                                sz_ex=_drift_size,
                                                                                force=True, dynamic=_drift_dynamic, verbose=_verbose)
            if _failed_count > 0:
                print(f"++ {_failed_count} suspected failures detected in drift correction.");
            # release
            del(_bead_ims, _)
        # create cells in parallel
        if _verbose:
            print("+ Create cell_data objects!");
        _cell_ids = range(int(np.max(_fov_segmentation_label)-1));
        _params = [{'fov_id': _fov_id,
                  'cell_id': _cell_id,
                  'data_folder': self.data_folder,
                  'temp_folder': self.temp_folder,
                  'analysis_folder':self.analysis_folder,
                  'segmentation_folder': self.segmentation_folder,
                  'correction_folder': self.correction_folder,
                  'drift_folder': self.drift_folder,
                  'map_folder': self.map_folder,
                  'drift': _fov_drift,
                  } for _cell_id in _cell_ids];
        _args = [(_p, True, True, True, True, False) for _p in _params]
        _cell_pool = multiprocessing.Pool(_num_threads)
        _cells = _cell_pool.starmap(self._create_cell, _args, chunksize=1)
        _cell_pool.close();
        _cell_pool.join();
        _cell_pool.terminate();
        # load
        self.cells += _cells

    def _crop_image_for_cells(self, _type='all', _load_in_ram=True, _load_annotated_only=True,
                              _save=True, _force=False,
                              _overwrite_temp=True, _overwrite_cell_info=False,
                              _verbose=True):
        """Load images for all cells in this cell_list
        Inputs:
            _type: loading type for this """
        # check whether cells and segmentation,drift info exists
        if _verbose:
            print ("+ Load images for cells in this cell list")
        if not hasattr(self, 'cells') or len(self.cells) ==0:
            raise ValueError("No cell information loaded in cell_list");
        # whether load annotated hybs only
        if _load_annotated_only:
            _folders = self.annotated_folders;
        else:
            _folders = self.folders;
        _fov_dic = {} # fov_id ->
        for _cell in self.cells:
            if not hasattr(_cell,'drift'):
                _cell._load_drift();
            if not hasattr(_cell, 'segmentation_label'):
                _cell._load_segmentation();
            # record fov list at the same time
            if _cell.fov_id not in _fov_dic:
                _fov_dic[_cell.fov_id] = {};
        if _verbose:
            print(f"++ Field of view loaded for this list: {list(_fov_dic.keys())}")
        # load images
        for _id in _fov_dic:
            if _verbose:
                print(f"++ loading image for fov:{_id}");
            _temp_files, _, _ = analysis.load_image_fov(_folders, self.fovs, _id,
                                self.channels,  self.color_dic, self.num_threads, loading_type=_type,
                                max_chunk_size=6, correction_folder=self.correction_folder, overwrite_temp=_overwrite_temp,
                                temp_folder=self.temp_folder, return_type='filename', verbose=_verbose)
            _fov_dic[_id] = analysis.reconstruct_from_temp(_temp_files, _folders, self.fovs, _id,
                                                       self.channels, self.color_dic, temp_folder=self.temp_folder,
                                                       find_all=True, num_threads=self.num_threads, loading_type=_type, verbose=_verbose)

        # Load for combo
        if _type == 'all' or _type == 'combo':
            if _verbose:
                print("++ processing combo for cells")
            # loop through all cells
            for _cell in self.cells:
                if _verbose:
                    print(f"+++ crop images for cell:{_cell.cell_id} in fov:{_cell.fov_id}")
                _cell._load_images('combo', _splitted_ims=_fov_dic[_cell.fov_id],
                                   _num_threads=self.num_threads, _extend_dim=20,
                                   _load_in_ram=_load_in_ram, _load_annotated_only=_load_annotated_only,
                                   _save=_save, _overwrite=_overwrite_cell_info, _verbose=_verbose);

        # load for unique
        if _type == 'all' or _type == 'unique':
            if _verbose:
                print("++ processing unique for cells")
            for _cell in self.cells:
                if _verbose:
                    print(f"+++ crop images for cell:{_cell.cell_id} in fov:{_cell.fov_id}")
                _cell._load_images('unique', _splitted_ims=_fov_dic[_cell.fov_id],
                                   _num_threads=self.num_threads, _extend_dim=20,
                                   _load_in_ram=_load_in_ram, _load_annotated_only=_load_annotated_only,
                                   _save=_save, _overwrite=_overwrite_cell_info, _verbose=_verbose);
        # save extra cell_info
        if _save:
            for _cell in self.cells:
                _cell._save_to_file('cell_info', _overwrite=_overwrite_cell_info, _verbose=_verbose)

    def _load_cells_from_files(self, _type='all', _fov_id=None, _overwrite_cells=False, _verbose=True):
        """Function to load cells from existing files"""
        if _verbose:
            print("+ Load cells from existing files.");
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError('No cell information provided, should create cells first!')
        # check fov_id input
        for _cell in self.cells:
            if _verbose:
                print(f"++ loading info for fov:{_cell.fov_id}, cell:{_cell.cell_id}");
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
            raise ValueError('No cells are generated in this cell list!');
        if _verbose:
            print("+ Generate chromosomes for cells.")
        # chromsome savefile #
        _fov_ids = [_cell.fov_id for _cell in self.cells];
        _fov_ids = np.unique(_fov_ids);
        _chrom_savefile = os.path.join(self.temp_folder, _coord_filename.replace('.pkl', str(_fov_ids)+'.pkl'));
        # loop through cells to generate chromosome
        _chrom_ims = [];
        _chrom_dims = [];
        _coord_dic = {'coords': [],
                      'class_ids': [],
                      'pfits':{},
                      'dec_text':{},
                      }; # initialize _coord_dic for picking
        for _i, _cell in enumerate(self.cells):
            if hasattr(_cell, 'chrom_im') and not _overwrite:
                _cim = _cell.chrom_im;
            else:
                _cim = _cell._generate_chromosome_image(_source=_source, _max_count=_max_count, _verbose=_verbose)
                _cell.chrom_im = _cim
            _chrom_ims.append(_cim);
            _chrom_dims.append(np.array(np.shape(_cim)));
            if hasattr(_cell, 'chrom_coords') and not _overwrite:
                _chrom_coords = _cell.chrom_coords;
            else:
                _, _chrom_coords = _cell._identify_chromosomes(_gaussian_size=_gaussian_size, _cap_percentile=_cap_percentile,
                                                               _seed_dim=_seed_dim, _th_percentile=_th_percentile,
                                                               _min_obj_size=_min_obj_size,_verbose=_verbose)
            # build chrom_coord_dic
            _coord_dic['coords'] += [np.flipud(_coord) for _coord in _chrom_coords]
            _coord_dic['class_ids'] += list(np.ones(len(_chrom_coords),dtype=np.uint8)*int(_i))
        # create existing coord_dic file
        if _verbose:
            print("++ dumping existing info to file:", _chrom_savefile);
        pickle.dump(_coord_dic, open(_chrom_savefile, 'wb'));
        # convert to the same dimension
        _max_dim = np.max(np.concatenate([_d[np.newaxis,:] for _d in _chrom_dims]), axis=0)
        if _verbose:
            print("Maximum dimension for these images:", _max_dim)
        _converted_ims = [np.ones(_max_dim) * np.min(_cim) for _cim in _chrom_ims];
        for _im, _d, _cim in zip(_converted_ims, _chrom_dims, _chrom_ims):
            _im[:_d[0], :_d[1],:_d[2]] = _cim

        _chrom_viewer = visual_tools.imshow_mark_3d_v2(_converted_ims, image_names=[f"fov:{_cell.fov_id}, cell:{_cell.cell_id}" for _cell in self.cells],
                                                       save_file=_chrom_savefile)
        _chrom_viewer.load_coords();

        return _chrom_viewer

    def _update_chromosomes_for_cells(self, _coord_filename='chrom_coords.pkl', _save=True, _verbose=True):
        # check attribute
        if not hasattr(self, 'cells') or len(self.cells) == 0:
            raise ValueError('No cells are generated in this cell list!');
        if _verbose:
            print("+ Update manually picked chromosomes to cells");
        # chromsome savefile #
        _fov_ids = [_cell.fov_id for _cell in self.cells];
        _fov_ids = np.unique(_fov_ids);
        _chrom_savefile = os.path.join(self.temp_folder, _coord_filename.replace('.pkl', str(_fov_ids)+'.pkl'));
        # load from chrom-coord and partition it
        _coord_dic = pickle.load(open(_chrom_savefile, 'rb'));
        _coord_list = visual_tools.partition_map(_coord_dic['coords'], _coord_dic['class_ids'], enumerate_all=True);
        if len(_coord_list) > len(self.cells):
            raise ValueError(f'Number of cells doesnot match between cell-list and {_chrom_savefile}')
        elif len(_coord_list) < len(self.cells):
            print("++ fewer picked chromosome sets discovered than number of cells, append with empty lists.")
            for _i in range(len(self.cells) - len(_coord_list)):
                _coord_list.append([]);
        # save to attribute first
        for _cell, _coords in zip(self.cells, _coord_list):
            _chrom_coords = [np.flipud(_coord) for _coord in _coords];
            _cell.chrom_coords = _chrom_coords;
            if _verbose:
                print(f"++ matching {len(_chrom_coords)} chromosomes for fov:{_cell.fov_id}, cell:{_cell.cell_id}")
        # then update files if specified
            if _save:
                _cell._save_to_file('cell_info', _verbose=_verbose);
                if hasattr(_cell, 'combo_groups'):
                    _cell._save_to_file('combo', _overwrite=True, _verbose=_verbose);
    def _remove_temp_fov(self, _fov_id, _temp_marker='corrected.npy', _verbose=True):
        """Remove all temp files for given fov """
        _temp_fls = glob.glob(os.path.join(self.temp_folder, '*'))
        if _verbose:
            print(f"+ Remove temp file for fov:{_fov_id}")
        for _fl in _temp_fls:
            if os.path.isfile(_fl) and _temp_marker in _fl and self.fovs[_fov_id].replace('.dax', '') in _fl:
                print("++ removing temp file:", os.path.basename(_fl))
                os.remove(_fl)


    def _load_decoded_for_cells(self):
        pass

    def _spot_finding_for_cells(self):
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
        if 'analysis_folder'  in parameters:
            self.analysis_folder = str(parameters['analysis_folder']);
        else:
            self.analysis_folder = self.data_folder[0]+os.sep+'Analysis'
        # temp_folder
        if 'temp_folder'  in parameters:
            self.temp_folder = parameters['temp_folder'];
        else:
            self.temp_folder = _temp_folder
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
            self.correction_folder = _correction_folder
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

            if _load_in_ram:
                self.segmentation_label = _seg_label
                self.dapi_im = _dapi_im

        return _seg_label, _dapi_im

    def _load_drift(self, _size=500, _force=False, _dynamic=True, _verbose=True):
        # if exists:
        if hasattr(self, 'drift') and not _force:
            if _verbose:
                print(f"- drift already exists for cell:{self.cell_id}, skip");
            return self.drift;
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

    def _load_images(self, _type, _splitted_ims=None,
                     _num_threads=5, _extend_dim=10,
                     _load_in_ram=False, _load_annotated_only=True,
                     _illumination_correction=True, _chromatic_correction=True,
                     _save=True, _overwrite=False, _verbose=False):
        """Core function to load images, support different types:"""
        if not hasattr(self, 'segmentation_label') and _type in ['unique', 'combo', 'sparse']:
            self._load_segmentation();
        if not hasattr(self, 'channels') or not hasattr(self, 'color_dic'):
            self._load_color_info();

        # annotated folders
        if _load_annotated_only and not hasattr(self, 'annotated_folders'):
            self.annotated_folders = []
            if not hasattr(self, 'annotated_folders'):
                for _hyb_fd in self.color_dic:
                    _matches = [_fd for _fd in self.folders if _hyb_fd in _fd];
                    if len(_matches) == 1:
                        self.annotated_folders.append(_matches[0]);
        else:
            self.annotated_folders = self.folders

        # Case: beads
        if str(_type).lower() == "beads":
            if hasattr(self, 'bead_ims') and hasattr(self, 'bead_names'):
                return self.bead_ims, self.bead_names
            else:
                _bead_ims, _bead_names, _ = analysis.load_image_fov(self.annotated_folders, self.fovs, self.fov_id,
                                                                    self.channels, self.color_dic,
                                                                    num_threads=_num_threads, loading_type=_type,
                                                                    correction_folder=self.correction_folder,
                                                                    temp_folder=self.temp_folder,
                                                                    return_type='mmap', overwrite_temp=False, verbose=True);
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
                _splitted_ims = analysis.load_image_fov(self.annotated_folders, self.fovs, self.fov_id,
                                                        self.channels, self.color_dic,
                                                        num_threads=_num_threads, loading_type=_type,
                                                        correction_folder=self.correction_folder,
                                                        temp_folder=self.temp_folder,
                                                        return_type='mmap', overwrite_temp=False, verbose=True);
                if _load_in_ram:
                    self.splitted_ims = _splitted_ims;
                return _splitted_ims

        # Case: unique images
        elif str(_type).lower() == 'unique':
            if _verbose:
                print(f"Loading unique images for cell:{self.cell_id} in fov:{self.fov_id}")
            # check if images are properly loaded
            if hasattr(self, 'unique_ims') and hasattr(self, 'unique_ids') and hasattr(self, 'unique_channels') and not _overwrite:
                return self.unique_ims, self.unique_ids, self.unique_channels
            elif hasattr(self, 'splitted_ims'):
                _splitted_ims = self.splitted_ims;
            elif not _splitted_ims:
                # Load all splitted images
                _splitted_ims = analysis.load_image_fov(self.annotated_folders, self.fovs, self.fov_id,
                                                        self.channels, self.color_dic,
                                                        num_threads=_num_threads, loading_type=_type,
                                                        correction_folder=self.correction_folder,
                                                        temp_folder=self.temp_folder,
                                                        return_type='mmap', overwrite_temp=False, verbose=True);

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
                            _cropped_im = visual_tools.crop_cell(_channel_im, self.segmentation_label,
                                                                 drift=self.drift[_img_name],
                                                                 extend_dim=_extend_dim)[0]
                            _unique_ims.append(_cropped_im)
                            _unique_ids.append(int(_channel_info.split(_unique_marker)[-1]))
                            _unique_channels.append(_channel);
                else:
                    raise IOError('-- missing image:',_img_name);
            # sort
            _tp = [(_id,_im,_ch) for _id, _im, _ch in sorted(zip(_unique_ids, _unique_ims, _unique_channels))]
            _unique_ims = [_t[1] for _t in _tp];
            _unique_ids = [_t[0] for _t in _tp];
            _unique_channels = [_t[2] for _t in _tp];

            if _load_in_ram:
                self.unique_ims = _unique_ims;
                self.unique_ids = _unique_ids;
                self.unique_channels = _unique_channels
                if _save:
                    self._save_to_file('unique', _overwrite=_overwrite);
            else:
                if _save:
                    _dc = {'unique_ims':_unique_ims,
                           'unique_ids':_unique_ids,
                           'unique_channels': _unique_channels}
                    self._save_to_file('unique', _save_dic=_dc, _overwrite=_overwrite)
            return _unique_ims, _unique_ids, _unique_channels

        elif str(_type).lower() == 'combo' or str(_source).lower() == 'sparse' :
            # check if images are properly loaded
            if hasattr(self, 'splitted_ims'):
                _splitted_ims = self.splitted_ims;
            elif not _splitted_ims:
                # Load all splitted images
                _splitted_ims = analysis.load_image_fov(self.annotated_folders, self.fovs, self.fov_id,
                                                        self.channels, self.color_dic,
                                                        num_threads=_num_threads, loading_type='combo',
                                                        correction_folder=self.correction_folder,
                                                        temp_folder=self.temp_folder,
                                                        return_type='mmap', verbose=True);

            # check drift info
            if not hasattr(self, 'drift'):
                 self._load_drift();
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
                        print("-- cropping images for group:", _hyb_fds)
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
                            _cropped_im = corrections.Z_Shift_Correction(_cropped_im, verbose=False);
                            # store this image
                            _combo_images.append(_cropped_im);
                            _cropped_im=None
                    # create a combo group
                    # special treatment to combo_images:

                    _group = Encoding_Group(_combo_images, _hyb_fds, _matrix, self.save_folder,
                                            self.fov_id, self.cell_id, _channel, _group_id);
                    _combo_groups.append(_group);


            if _load_in_ram:
                self.combo_groups = _combo_groups;

                if _save:
                    self._save_to_file('combo', _overwrite=_overwrite)
            else: # not in RAM
                if _save:
                    self._save_to_file('combo', _save_dic={'combo_groups':_combo_groups}, _overwrite=_overwrite);

            return _combo_groups;

        # exception: wrong _type key given
        else:
            raise ValueError('wrong image loading type given!, key given:', _type)
        return False

    # saving
    def _save_to_file(self, _type='all', _save_dic={}, _save_folder=None, _overwrite=False, _verbose=True):
        # special keys don't save in the 'cell_info' file. They are images!
        _special_keys = ['unique_ims', 'combo_groups','bead_ims', 'splitted_ims']
        if _save_folder == None:
            _save_folder = self.save_folder;
        if not os.path.exists(_save_folder):
            os.makedirs(_save_folder);

        if _type=='all' or _type =='cell_info':
            # save file full name
            _savefile = _save_folder + os.sep + 'cell_info.pkl';
            if _verbose:
                print("- Save cell_info to:", _savefile)
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
        if _type =='all' or _type == 'combo':
            if hasattr(self, 'combo_groups'):
                _combo_groups = self.combo_groups;
            elif 'combo_groups' in _save_dic:
                _combo_groups = _save_dic['combo_groups'];
            else:
                raise ValueError(f'No combo-groups information given in fov:{self.fov_id}, cell:{self.cell_id}');
            for _group in _combo_groups:
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
                # append chromosome info if exists
                if hasattr(self, 'chrom_coords'):
                    _combo_dic['chrom_coords'] = self.chrom_coords
                # save
                if _verbose:
                    print("-- saving combo to:", _combo_savefile)
                np.savez_compressed(_combo_savefile, **_combo_dic)
        # save unique
        if _type == 'all' or _type == 'unique':
            _unique_savefile = _save_folder + os.sep + 'unique_rounds.npz';
            # check unique_ims
            if hasattr(self, 'unique_ims'):
                _unique_ims = self.unique_ims;
            elif 'unique_ims' in _save_dic:
                _unique_ims = _save_dic['unique_ims'];
            else:
                raise ValueError(f'No unique_ims information given in fov:{self.fov_id}, cell:{self.cell_id}');
            # check unique_ids
            if hasattr(self, 'unique_ids'):
                _unique_ids = self.unique_ids;
            elif 'unique_ids' in _save_dic:
                _unique_ids = _save_dic['unique_ids'];
            else:
                raise ValueError(f'No unique_ids information given in fov:{self.fov_id}, cell:{self.cell_id}');
            # check unique_channels
            if hasattr(self, 'unique_channels'):
                _unique_channels = self.unique_channels;
            elif 'unique_channels' in _save_dic:
                _unique_channels = _save_dic['unique_channels'];
            else:
                raise ValueError(f'No unique_channels information given in fov:{self.fov_id}, cell:{self.cell_id}');

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
        _type=str(_type).lower();
        if _type not in ['all', 'cell_info', 'unique', 'combo', 'decoded']:
            raise ValueError("Wrong _type kwd given!")
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
                    if 'readout' in handle.keys():
                        _readouts = handle['readouts']
                    if 'chrom_coords' in handle.keys():
                        _chrom_coords = handle['chrom_coords']
                        if not hasattr(self, 'chrom_coords'):
                            self.chrom_coords = _chrom_coords;
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

        if _type == 'all' or _type == 'unique':
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

        if _type == 'all' or _type == 'decoded':
            if not _decoded_flag and _type == 'decoded':
                raise ValueError("Kwd _decoded_flag not given, exit!");
            elif not _decoded_flag:
                print("Kwd _decoded_flag not given, skip this step.");
                # load existing combo files
                _raw_combo_fl = "rounds.npz"
                _combo_files = glob.glob(os.path.join(_save_folder, "group-*", "channel-*", _raw_combo_fl));

            # look for decoded _results

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
        _viewer = visual_tools.imshow_mark_3d_v2([self.chrom_im], image_names=['chromosome'],
                                                 save_file=_chrom_savefile, given_dic=_coord_dic)
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

    def _multi_fitting(self, _type='unique',_use_chrom_coords=True, _seed_th_per=30., _max_filt_size=3,
                       _width_zxy=[1.35,1.9,1.9], _expect_weight=1000, _min_height=100,
                       _save=True, _verbose=True):
        # first check Inputs
        _allowed_types = ['unique', 'decoded'];
        _type = _type.lower()
        if _type not in _allowed_types:
            raise KeyError(f"Wrong input key for _type:{_type}");
        if _use_chrom_coords:
            if not hasattr(self, 'chrom_coords'):
                self._load_from_file('cell_info');
                if not hasattr(self, 'chrom_coords'):

        if _verbose:
            print(f"+ Start multi-fitting for {_type} images")
        # TYPE unique
        if _type == 'unique':
            # check attributes
            if not hasattr(self, 'unique_ims') or not hasattr(self, 'unique_ids'):
                print("++ no unique image info loaded to this cell, try loading:")
                self._load_from_file('unique')
            





    def _dynamic_picking_spots(self):
        pass

    def _naive_picking_spots(self):
        pass

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
