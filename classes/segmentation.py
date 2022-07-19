import numpy as np
import os, time
from cellpose.models import CellposeModel
from sqlalchemy import over

default_image_sampling = (1024,1024)


class CellPoseSegment_3D():
    """Function to process 3D cellpose Segmentation"""
    def __init__(
        self,
        dapi_dax_file:str, 
        polyt_dax_file:str=None,
        correction_folder:str=None,
        dapi_channel:str='405',
        polyt_channel:str='750',
        save_folder:str=None,
        overwrite:bool=False,
        verbose:bool=True,
        )->None:
        # input files
        if '.dax' in dapi_dax_file:
            self.dapiFile = dapi_dax_file
        else:
            raise FileNotFoundError(f"{dapi_dax_file} is not a dax file, exit")
        if '.dax' in dapi_dax_file:
            self.polyTFile = polyt_dax_file
        else:
            raise FileNotFoundError(f"{polyt_dax_file} is not a dax file, exit")
        # param folders
        self.correctionFolder = correction_folder
        self.saveFolder = save_folder
        self.saveCellposeFile = os.path.join(save_folder, 
            os.path.basename(dapi_dax_file).replace('.dax','_Segmentation_Cellpose.npy')
        )
        # parameters
        self.dapiChannel = dapi_channel
        self.polyTChannel = polyt_channel
        self.overwrite = overwrite
        self.verbose = verbose
        # return
        return
    # Prepare images
    def _load_images(self):
        pass
    # CellPose Segment
    def _run_cellpose(self):
        pass
    # Save step1
    def _save_cellpose_output(self):
        pass
    # Load step1
    def _load_cellpose_output(self):
        if hasattr(self, 'segmentation_labels_cellpose') and not self.overwrite:
            if self.verbose:
                print(f"segmentation_labels_cellpose already exists, skip loading.")
        elif os.path.exists(self.saveCellposeFile):
            if self.verbose:
                print(f"- Loading cellpose output from file: {self.saveCellposeFile}")
            self.segmentation_labels_cellpose = np.load(self.saveCellposeFile)
        else:
            print(f"- Save file: {self.saveCellposeFile} doesn't exist, skip.")
        return    
    # Run watershed in PolyT
    def _run_watershed_in_polyT(self):
        pass
    # Save step2
    def _save_watershed_output(self):
        pass
    