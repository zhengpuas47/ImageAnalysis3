## global variables
import numpy as np
# correction_folder
_correction_folder=r'Z:\Corrections'
# temp folder
_temp_folder = r'I:\Pu_temp'
# distance_zxy
_distance_zxy = np.array([200, 106, 106])
# sigma_zxy
_sigma_zxy = np.array([1.35, 1.9, 1.9])
# image_dim
_image_size = np.array([30,2048,2048])
# allowed_colors
_allowed_colors = ['750', '647', '561', '488', '405']
_corr_channels = ['750', '647', '561']
# number of buffer frames and empty frames
_num_buffer_frames = 10
_num_empty_frames = 1


# library design tools
from . import library_tools
# function to process fitted spots
from . import spot_tools
# everything about gaussian fitting, imshow3d
from . import visual_tools
# everything about aligments
from . import alignment_tools
# everything about domain analysis
from . import domain_tools
# functions to get hybe, folders
from . import get_img_info
# Drift and illumination correction
from . import corrections
# Defined class
from . import classes
# functions for post analysis, including compartment, epigenomics
from . import postanalysis
# functions to generate figures
from . import figure_tools
# functions to read and save images
from . import io_tools
# function to load compartment tools
from . import compartment_tools
## import exteral functions
from .External import Fitting_v3
from .External import DomainTools
