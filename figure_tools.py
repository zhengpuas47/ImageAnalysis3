import sys,os,re,time,glob
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy
from scipy.signal import fftconvolve
from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter
from scipy import ndimage, stats
from skimage import morphology, restoration, measure
from skimage.segmentation import random_walker
from scipy.ndimage import gaussian_laplace
import cv2
import multiprocessing as mp
from sklearn.decomposition import PCA

from . import get_img_info, corrections, alignment_tools, classes
from .External import Fitting_v3
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy,_image_size, _allowed_colors

from scipy.stats import linregress
#from astropy.convolution import Gaussian2DKernel,convolve

## Define some global settings
_dpi = 300 # dpi required by figure
_single_col_width = 2.25 # figure width in inch if occupy 1 colomn
_double_col_width = 4.75 # figure width in inch if occupy 1 colomn
_single_row_height= 2 # comparable height to match single-colomn-width
