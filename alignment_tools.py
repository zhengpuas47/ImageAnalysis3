import sys
import glob
import os
import time
import copy
import numpy as np
import pickle as pickle
import multiprocessing as mp
import psutil

from . import get_img_info, corrections, visual_tools, analysis
from . import _correction_folder, _temp_folder, _distance_zxy, _sigma_zxy, _image_size
from .External import Fitting_v3
from scipy import ndimage, stats
from scipy.spatial.distance import pdist, cdist, squareform
from skimage import morphology
from skimage.segmentation import random_walker

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import seismic_r

## Function for alignment between consequtive experiments

# 1. alignment for manually picked points
def align_manual_points(pos_file_before, pos_file_after,
                        save=True, save_folder=None, save_filename='', verbose=True):
    """Function to align two manually picked position files, 
    they should follow exactly the same order and of same length.
    Inputs:
        pos_file_before: full filename for positions file before translation
        pos_file_after: full filename for positions file after translation
        save: whether save rotation and translation info, bool (default: True)
        save_folder: where to save rotation and translation info, None or string (default: same folder as pos_file_before)
        save_filename: filename specified to save rotation and translation points
        verbose: say something! bool (default: True)
    Outputs:
        R: rotation for positions, 2x2 array
        T: traslation of positions, array of 2
    Here's example for how to translate points
        translated_ps_before = np.dot(ps_before, R) + t
    """
    # load position_before
    if os.path.isfile(pos_file_before):
        ps_before = np.loadtxt(pos_file_before, delimiter=',')
    else:
        raise IOError(
            f"- Position file:{pos_file_before} file doesn't exist, exit!")
    # load position_after
    if os.path.isfile(pos_file_after):
        ps_after = np.loadtxt(pos_file_after, delimiter=',')
    else:
        raise IOError(
            f"- Position file:{pos_file_after} file doesn't exist, exit!")
    # do SVD decomposition to get best fit for rigid-translation
    c_before = np.mean(ps_before, axis=0)
    c_after = np.mean(ps_after, axis=0)
    H = np.dot((ps_before - c_before).T, (ps_after - c_after))
    U, S, V = np.linalg.svd(H)  # do SVD
    # calcluate rotation
    R = np.dot(V, U.T).T
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    # calculate translation
    t = - np.dot(c_before, R) + c_after
    # here's example for how to translate points
    # translated_ps_before = np.dot(ps_before, R) + t
    if verbose:
        print(
            f"- Manually picked points aligned, rotation:\n{R},\n translation:{t}")
    # save
    if save:
        if save_folder is None:
            save_folder = os.path.dirname(pos_file_before)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if len(save_filename) > 0:
            save_filename += '_'
        rotation_name = os.path.join(save_folder, save_filename+'rotation')
        translation_name = os.path.join(
            save_folder, save_filename+'translation')
        np.save(rotation_name, R)
        np.save(translation_name, t)
        if verbose:
            print(f'-- rotation matrix saved to file:{rotation_name}')
            print(f'-- translation matrix saved to file:{translation_name}')
    return R, t
