## global variables
import numpy as np
# correction_folder
_correction_folder=r'\\10.245.74.158\Chromatin_NAS_0\Corrections\Corrections_202008'
# temp folder
_temp_folder = r'\\mendel\Mendel_SSD3\Common_Temp'
# distance_zxy
_distance_zxy = [200, 108, 108]
# sigma_zxy
_sigma_zxy = [1.35, 1.9, 1.9]
# image_dim
_image_size = [30,2048,2048]
# allowed_colors
_allowed_colors = ['750', '647', '561', '488', '405']
_corr_channels = ['750', '647', '561']
# number of buffer frames and empty frames
_num_buffer_frames = 0
_num_empty_frames = 0
# image datatype
_image_dtype = np.uint16

# library design tools
#from . import library_tools
# function to process fitted spots
from . import spot_tools
# everything about gaussian fitting, imshow3d
from . import visual_tools
# everything about aligments
from . import alignment_tools

# functions to get hybe, folders
from . import get_img_info
# Drift and illumination correction
from . import corrections
from . import correction_tools
# Defined class
from . import classes
# functions for post analysis, including compartment, epigenomics
from . import postanalysis
# functions to generate figures
from . import figure_tools
# functions to read and save images
from . import io_tools
# everything about domain analysis
#from . import domain_tools
# function to call and evaluate compartments
#from . import compartment_tools
# function to analyze structural features
from . import structure_tools
# function to help segmentations
from . import segmentation_tools
## import exteral functions
from .External import Fitting_v4
#from .External import DomainTools
