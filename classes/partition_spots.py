# required packages
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
#import multiprocessing as mp
# required internal functions
from ..figure_tools.plot_partition import plot_cell_spot_counts
from ..io_tools.spots import FovCell2Spots_2_DataFrame,FovSpots3D_2_DataFrame
from ..io_tools.parameters import _read_microscope_json
from .preprocess import Spots3D

#from ..io_tools.crop import generate_neighboring_crop
import copy

default_search_radius = 4
default_pixel_sizes = [250,108,108]
default_num_threads = 12

######################################################################
# Notes:
# 1. gene_readout_file must have columns: 'Bit number' and 'Gene' (or any given query label)
#
######################################################################

class Spots_Partition():
    """"""
    def __init__(self,
                 segmentation_masks:np.ndarray, 
                 readout_filename:str,
                 fov_id=None,
                 search_radius=default_search_radius,
                 pixel_sizes=default_pixel_sizes,
                 save_filename=None,
                 make_copy=True,
                 ):
        print("- Partition spots")
        # localize segmentation_masks
        if make_copy:
            self.segmentation_masks = copy.copy(segmentation_masks)
        else:
            self.segmentation_masks = segmentation_masks
        self.image_size = np.shape(segmentation_masks)
        self.fov_id=fov_id
        self.search_radius = int(search_radius)
        self.pixel_sizes = pixel_sizes
        # filenames
        self.readout_filename = readout_filename
        self.save_filename = save_filename
        
    def run_RNA(self, spots_list, bits=None,
            query_label='Gene', 
            save=True,
            overwrite=False, verbose=True):
        from .preprocess import Spots3D
        if os.path.exists(self.save_filename) and not overwrite:
            print(f"-- directly load from file: {self.save_filename}")
            _count_df = pd.read_csv(self.save_filename, header=0)
            setattr(self, f"{query_label.lower()}_count_df", _count_df)
        else:
            if bits is None:
                _bits = np.arange(1, len(spots_list)+1)
            elif len(bits) != len(spots_list):
                raise IndexError(f"length of spots_list and bits don't match")
            else:
                _bits = np.array(bits, dtype=np.int32)
            # read gene df
            if verbose:
                print(f"-- read gene_list")
            self.readout_df = self.read_gene_list(self.readout_filename)
            # initialize 
            _cells = np.unique(self.segmentation_masks)[1:]
            _cell_spots_list = {_c:{_bit:[] for _bit in _bits} 
                                for _c in _cells}
            _labels_list = []
            # loop through each bit
            for spots, bit in zip(spots_list, _bits):
                _bit = int(bit)
                _spots = Spots3D(spots, _bit, self.pixel_sizes)
                _labels = self.spots_to_labels(self.segmentation_masks,
                    _spots, self.image_size, self.search_radius, 
                    verbose=verbose)
                _labels_list.append(_labels)
                for _l in np.unique(_labels):
                    if _l > 0:
                        _cell_spots_list[_l][_bit] = _spots[_labels==_l]
            # use information in _cell_spots_list, update the gene_count_df
            _count_df = pd.DataFrame()
            for _cell, _cell_spots in _cell_spots_list.items():
                _info_dict = {'cell_id': _cell}
                if hasattr(self, 'fov_id') or self.fov_id is not None:
                    _info_dict['fov_id'] = self.fov_id
                for _bit, _spots in _cell_spots.items():
                    if _bit in self.readout_df['Bit number'].values:
                        _gene = self.readout_df.loc[self.readout_df['Bit number']==_bit, query_label].values[0]
                        _info_dict[_gene] = len(_spots)
                # append
                _count_df = _count_df.append(_info_dict, ignore_index=True, )
            # add to attribute
            setattr(self, f"{query_label.lower()}_count_df", _count_df)
            setattr(self, 'cell_spots_list', _cell_spots_list)
            setattr(self, 'labels_list', _labels_list)
            # save
            if save:
                if verbose:
                    print(f"-- save {query_label.lower()}_count_df into file: {self.save_filename}")
                _count_df.to_csv(self.save_filename, index=False, header=True)
        # return
        return _count_df

    @staticmethod
    def spots_to_labels(segmentation_masks:np.ndarray, 
                        spots:Spots3D, 
                        search_radius:int=10,
                        verbose:bool=True,
                        ):
        #from ..io_tools.crop import batch_crop
        if verbose:
            print(f"-- partition barcodes for {len(spots)} spots")
        # initialize
        _spot_labels = []

        # calculate
        _signals = find_coordinate_intensities(segmentation_masks, spots, 
            search_radius=search_radius)
        # stats
        for _spot_signal in _signals:
            # get unique markers
            _mks, _counts = np.unique(_spot_signal, return_counts=True)
            # filter counts and mks
            _counts = _counts[_mks>0]
            _mks = _mks[_mks>0]
            # append the label
            if len(_mks) == 0:
                _spot_labels.append(-1)
            else:
                _spot_labels.append(_mks[np.argmax(_counts)])

        return np.array(_spot_labels, dtype=np.int32)

    @staticmethod
    def spots_to_DAPI(dapi_im:np.ndarray, 
                      spots:Spots3D, 
                      search_radius:int=5,
                      verbose:bool=True,
                      ):
        #from ..io_tools.crop import batch_crop
        if verbose:
            print(f"-- calculate local DAPI signal for {len(spots)} spots")
        # calculate
        _signals = find_coordinate_intensities(dapi_im, spots, 
            search_radius=search_radius)
        # stats
        _dapi_stats = np.max(_signals, axis=1)

        return _dapi_stats

    @staticmethod
    def read_gene_list(readout_filename):
        return pd.read_csv(readout_filename, header=0, )


def Merge_GeneCounts(gene_counts_list,
                     fov_ids,
                     save=True, save_filename=None,
                     overwrite=False, verbose=True,
                     ):
    """Merge cell-locations from multiple field-of-views"""
    _start_time = time.time()
    
    if save_filename is None or not os.path.exists(save_filename) or overwrite:
        if verbose:
            print(f"- Start merging {len(gene_counts_list)} cell locations")
        # initialize
        merged_gene_counts_df = pd.DataFrame()
        # loop through each cell-location file
        for _fov_id, _gene_counts in zip(fov_ids, gene_counts_list):
            if isinstance(_gene_counts, str):
                _gene_counts =  pd.read_csv(_gene_counts, header=0)
            elif isinstance(_gene_counts, pd.DataFrame):
                _gene_counts = _gene_counts.copy()
            else:
                raise TypeError(f"Wrong input type for _gene_counts")
            # add fov 
            if 'fov_id' not in _gene_counts.columns or np.isnan(_gene_counts['fov_id']).all():
                
                _gene_counts['fov_id'] = _fov_id * np.ones(len(_gene_counts), dtype=np.int32)
            # merge
            merged_gene_counts_df = pd.concat([merged_gene_counts_df, _gene_counts],
                                                ignore_index=True)

        if verbose:
            print(f"-- {len(merged_gene_counts_df)} cells converted into MetaData")

        if save and (save_filename is not None or overwrite):
            if verbose:
                print(f"-- save {len(merged_gene_counts_df)} cells into file:{save_filename}")
            merged_gene_counts_df.to_csv(save_filename, index=False, header=True)
    else:
        merged_gene_counts_df = pd.read_csv(save_filename, header=0)
        if verbose:
            print(f"- directly load {len(merged_gene_counts_df)} cells file: {save_filename}")

    if verbose:
        _execute_time = time.time() - _start_time
        if verbose:
            print(f"-- merge cell-locations in {_execute_time:.3f}s")
            
    return merged_gene_counts_df

def find_coordinate_intensities(image:np.ndarray,
                                spots:Spots3D,
                                search_radius=5,
                                ):
    """Get nearest coordinate intensity"""
    image_size = np.array(np.shape(image))
    # round up coordinates
    _coords = np.round(spots.to_coords()).astype(np.int32).transpose()
    # generate local searches
    _local_coords =np.meshgrid(np.arange(-search_radius,search_radius+1), 
                               np.arange(-search_radius,search_radius+1), 
                               np.arange(-search_radius,search_radius+1), 
                              )
    _local_coords = np.stack(_local_coords).transpose((2, 1, 3, 0)).reshape(-1,3)
    # modify_coords
    all_ints = []
    for _lc in _local_coords:
        _modified_coords = _coords + _lc[:,np.newaxis]
        for _ic, _size in enumerate(image_size):
            _modified_coords[_ic][_modified_coords[_ic] < 0] = 0
            _modified_coords[_ic][_modified_coords[_ic] >= _size] = _size-1
        # get intensity
        all_ints.append(image[tuple(_modified_coords)])
    all_ints = np.array(all_ints).transpose()
    return all_ints

def batch_partition_smFISH_spots(        
                        segmentation_masks:np.ndarray, 
                        readout_filename:str,
                        fov_id,
                        spots_list,
                        bits=None,
                        query_label='Gene', 
                        search_radius=default_search_radius,
                        pixel_sizes=default_pixel_sizes,
                        save_filename=None,
                        ):
    """
    """
    _partition_cls = Spots_Partition(segmentation_masks,
    readout_filename, fov_id=fov_id, search_radius=search_radius,
    pixel_sizes=pixel_sizes, save_filename=save_filename)
    # run
    _df = _partition_cls.run_RNA(spots_list, bits, query_label=query_label)
    # return
    return _df


def _batch_partition_spots(
    fov_id, seg_label, fovcell_2_uid,
    cand_spots_list, cand_bits, cand_channels,
    cand_spots_savefile, 
    microscope_param_file=None, 
    search_radius=3, pixel_sizes=default_pixel_sizes,
    make_plot=True,
    overwrite:bool=False,
    debug:bool=False,
    verbose:bool=True,
    )->pd.DataFrame:
    """Partition spots with given segmentation label in batch"""
    if os.path.isfile(cand_spots_savefile) and not overwrite:
        if verbose:
            print(f"- Directly load cand_spots DataFrame for fov:{fov_id} from: {cand_spots_savefile}")
        _spots_df = pd.read_csv(cand_spots_savefile)
    else:
        if verbose:
            print(f"- Partition cand_spots for fov:{fov_id}")
        ## partition
        _cell_2_spots = {_c:{} for _c in np.arange(1, np.max(seg_label)+1)}
        label_dict = {}

        for _bit, _ch, _pts in zip(cand_bits, cand_channels, cand_spots_list):
            # cast spot class
            _spots = Spots3D(_pts, bits=_bit, channels=_ch, pixel_sizes=pixel_sizes)
            if microscope_param_file is not None:
                from ..spot_tools.translating import MicroscopeTranslate_Spots
                _spots = MicroscopeTranslate_Spots(_spots, microscope_param_file)
            # calculate labels
            _labels = Spots_Partition.spots_to_labels(seg_label, _spots, 
                                                      search_radius=search_radius, verbose=verbose)
            label_dict[_bit] = _labels
            # parittion
            for _l in np.unique(_labels):
                if _l > 0:
                    _keep_flags = (_labels==_l)
                    # append
                    _cell_2_spots[_l][_bit] = _spots[_keep_flags]
                    _cell_2_spots[_l][_bit].bits = _spots.bits[_keep_flags]
                    _cell_2_spots[_l][_bit].channels = _spots.channels[_keep_flags]
        ## Convert into DataFrame
        _bit_2_channel = {_b:_ch for _b,_ch in zip(cand_bits, cand_channels)}
        _spots_df = FovCell2Spots_2_DataFrame(_cell_2_spots, fov_id, _bit_2_channel, fovcell_2_uid,
                                              save_filename=cand_spots_savefile, verbose=verbose)
        # make plot if applicable
        if make_plot:
            _cell_spots_counts = []
            for _cell, _spots_dict in _cell_2_spots.items():
                _spots_counts = []
                for _bit in cand_bits:
                    if _bit in _spots_dict:
                        _spots_counts.append(len(_spots_dict[_bit]))
                    else:
                        _spots_counts.append(0)
                _cell_spots_counts.append(np.array(_spots_counts))
            _cell_spots_counts = np.array(_cell_spots_counts)
            # plot
            SpotCount_SaveFile = os.path.join(os.path.dirname(cand_spots_savefile), 'Figures', 
                                        f'Fov-{fov_id}_SpotCountPerCell.png')
            if not os.path.exists(os.path.dirname(SpotCount_SaveFile)):
                os.makedirs(os.path.dirname(SpotCount_SaveFile))
            # Plot
            count_ax = plot_cell_spot_counts(_cell_spots_counts, save=True, save_filename=SpotCount_SaveFile)
        
    return _spots_df



def batch_partition_DNA_spots(
    fov_id:int, 
    spots:np.ndarray, spot_bits:np.ndarray, spot_channels:np.ndarray,
    seg_label:np.ndarray, fovcell_2_uid:dict={},
    microscope_param_file=None, 
    search_radius=3, pixel_sizes=default_pixel_sizes,
    ignore_spots_out_cell=True,
    save=True, save_filename=None,
    make_plot=True, expected_count=60,
    overwite=False,
    verbose=True,
    ):
    """Batch partition DNA spots"""
    if os.path.isfile(save_filename) and not overwite:
        if verbose:
            print(f"- Directly load cand_spots DataFrame for fov:{fov_id} from: {save_filename}")
        _fov_spots_df = pd.read_csv(save_filename)
        
    else:
        if verbose:
            print(f"- Partition cand_spots for fov:{fov_id}")
            _start_time = time.time()
        # Merge spots list into all_spots
        _all_spots = Spots3D(spots, bits=spot_bits, channels=spot_channels, pixel_sizes=pixel_sizes)
        # Translate based on microscope.json if specified
        if microscope_param_file is not None:
            from ..spot_tools.translating import MicroscopeTranslate_Spots
            _all_spots = MicroscopeTranslate_Spots(_all_spots, microscope_param_file)
        # Search for segmentation label
        _labels = Spots_Partition.spots_to_labels(
            seg_label, _all_spots, 
            search_radius=search_radius, 
            verbose=verbose,
        )
        # convert into DataFrame
        _fov_spots_df = FovSpots3D_2_DataFrame(
            _all_spots, fov_id, cell_ids=_labels, 
            fovcell_2_uid=fovcell_2_uid, pixel_sizes=pixel_sizes,
            ignore_spots_out_cell=ignore_spots_out_cell,
        )
        # save
        if save and save_filename is not None:
            if verbose:
                print(f"- Save {len(_fov_spots_df)} spots to file: {save_filename}")
            _fov_spots_df.to_csv(save_filename, index=False)
        else:
            if verbose:
                print(f"- Not saving DataFrame into file, return it.")
        # make plot qc
        if make_plot:
            cell_spot_counts = []
            all_bits = np.unique(_fov_spots_df['bit'])
            for _cell in np.unique(_fov_spots_df['cell_id']):
                _cell_spots_df = _fov_spots_df[_fov_spots_df['cell_id']==_cell]
                _spot_counts = [np.sum(_cell_spots_df['bit']==_bit) for _bit in all_bits]
                cell_spot_counts.append(_spot_counts)
            cell_spot_counts = np.array(cell_spot_counts)
            # save image
            SpotCount_SaveFile = os.path.join(os.path.dirname(save_filename), 'Figures', f'Fov-{fov_id}_SpotCountPerCell.png')
            if not os.path.exists(os.path.dirname(SpotCount_SaveFile)):
                os.makedirs(os.path.dirname(SpotCount_SaveFile))
            count_ax = plot_cell_spot_counts(cell_spot_counts, expected_count=expected_count, save_filename=SpotCount_SaveFile)
        if verbose:
            print(f"-- finish partition in {time.time()-_start_time:.3f}s. ")
    
    return _fov_spots_df