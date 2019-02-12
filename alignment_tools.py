from scipy.spatial.distance import cdist
import cv2
import sys
import glob
import os
import time
import copy
import numpy as np
import pickle as pickle
import multiprocessing as mp
import psutil

from . import get_img_info, corrections, visual_tools, analysis, classes
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

## Translate images given drift


def fast_translate(im, trans):
	shape_ = im.shape
	zmax = shape_[0]
	xmax = shape_[1]
	ymax = shape_[2]
	zmin, xmin, ymin = 0, 0, 0
	trans_ = np.array(np.round(trans), dtype=int)
	zmin -= trans_[0]
	zmax -= trans_[0]
	xmin -= trans_[1]
	xmax -= trans_[1]
	ymin -= trans_[2]
	ymax -= trans_[2]
	im_base_0 = np.zeros([zmax-zmin, xmax-xmin, ymax-ymin])
	im_zmin = min(max(zmin, 0), shape_[0])
	im_zmax = min(max(zmax, 0), shape_[0])
	im_xmin = min(max(xmin, 0), shape_[1])
	im_xmax = min(max(xmax, 0), shape_[1])
	im_ymin = min(max(ymin, 0), shape_[2])
	im_ymax = min(max(ymax, 0), shape_[2])
	im_base_0[(im_zmin-zmin):(im_zmax-zmin), (im_xmin-xmin):(im_xmax-xmin), (im_ymin-ymin):(im_ymax-ymin)] = im[im_zmin:im_zmax, im_xmin:im_xmax, im_ymin:im_ymax]
	return im_base_0

def translate_points(position_file, rotation=None, translation=None, profile_folder=None, profile_filename='',
                     save=True, save_folder=None, save_filename='', verbose=True):
    """Function to translate a position file """

    pass

# function to do 2d-gaussian blur
def blurnorm2d(im, gb):
    """Normalize an input 2d image <im> by dividing by a cv2 gaussian filter of the image"""
    im_ = im.astype(np.float32)
    blurred = cv2.blur(im_, (gb, gb))
    return im_/blurred

# calculate pixel-level drift for 2d by FFT
def fftalign_2d(im1, im2, center=[0, 0], max_disp=50, plt_val=False):
    """
    Inputs: 2 2D images <im1>, <im2>, the expected displacement <center>, the maximum displacement <max_disp> around the expected vector.
    This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
    """
    from scipy.signal import fftconvolve
    im2_ = np.array(im2[::-1, ::-1], dtype=float)
    im2_ -= np.mean(im2_)
    im2_ /= np.std(im2_)
    im1_ = np.array(im1, dtype=float)
    im1_ -= np.mean(im1_)
    im1_ /= np.std(im1_)
    im_cor = fftconvolve(im1_, im2_, mode='full')

    sx_cor, sy_cor = im_cor.shape
    center_ = np.array(center)+np.array([sx_cor, sy_cor])/2.

    x_min = int(min(max(center_[0]-max_disp, 0), sx_cor))
    x_max = int(min(max(center_[0]+max_disp, 0), sx_cor))
    y_min = int(min(max(center_[1]-max_disp, 0), sy_cor))
    y_max = int(min(max(center_[1]+max_disp, 0), sy_cor))

    im_cor0 = np.zeros_like(im_cor)
    im_cor0[x_min:x_max, y_min:y_max] = 1
    im_cor = im_cor*im_cor0

    y, x = np.unravel_index(np.argmax(im_cor), im_cor.shape)
    if np.sum(im_cor > 0) > 0:
        im_cor[im_cor == 0] = np.min(im_cor[im_cor > 0])
    else:
        im_cor[im_cor == 0] = 0
    if plt_val:
        plt.figure()
        plt.plot([x], [y], 'k+')
        plt.imshow(im_cor, interpolation='nearest')
        plt.show()
    xt, yt = np.round(-np.array(im_cor.shape)/2.+[y, x]).astype(int)
    return xt, yt

def fft3d_from2d(im1, im2, gb=5, max_disp=150):
    """Given a refence 3d image <im1> and a target image <im2> 
    this max-projects along the first (z) axis and finds the best tx,ty using fftalign_2d.
    Then it trims and max-projects along the last (y) axis and finds tz.
    Before applying fftalignment we normalize the images using blurnorm2d for stability."""
    im1_ = blurnorm2d(np.max(im1, 0), gb)
    im2_ = blurnorm2d(np.max(im2, 0), gb)
    tx, ty = fftalign_2d(im1_, im2_, center=[0, 0], max_disp=max_disp, plt_val=False)
    sx, sy = im1_.shape

    im1_t = blurnorm2d(
        np.max(im1[:, max(tx, 0):sx+tx, max(ty, 0):sy+ty], axis=-1), gb)
    im2_t = blurnorm2d(
        np.max(im2[:, max(-tx, 0):sx-tx, max(-ty, 0):sy-ty], axis=-1), gb)
    tz, _ = fftalign_2d(im1_t, im2_t, center=[
                        0, 0], max_disp=max_disp, plt_val=False)
    return np.array([tz, tx, ty])





#Tzxy = fft3d_from2d(im_ref_sm2, im_sm2, gb=5, max_disp=np.inf)


def sparse_centers(centersh, dist_th=0, brightness_th=0, max_num=np.inf):
    """assuming input = zxyh"""
    all_cents = np.array(centersh).T
    centers = [all_cents[0]]
    from scipy.spatial.distance import cdist
    counter = 0
    while True:
        counter += 1
        if counter > len(all_cents)-1:
            break
        if all_cents[counter][-1] < brightness_th:
            break
        dists = cdist([all_cents[counter][:3]], [c[:3] for c in centers])
        if np.all(dists > dist_th):
            centers.append(all_cents[counter])
        if len(centers) >= max_num:
            break
    return np.array(centers).T


def get_ref_pts(im_ref_sm, dist_th=5, nbeads=400):
    #fit reference
    z, x, y, h = get_seed_points_base(im_ref_sm, return_h=True, th_std=4)
    zk, xk, yk, hk = sparse_centers(
        (z, x, y, h), dist_th=dist_th, brightness_th=0, max_num=nbeads)
    cr1 = np.array([zk, xk, yk]).T
    pfits1 = fast_local_fit(im_ref_sm, cr1, radius=5, width_zxy=[1, 1, 1])
    cr1 = pfits1[:, 1:4]
    return cr1


def get_cand_pts(im_sm, cr1, tzxy, dist_th=5):
    #fit candidate
    z, x, y, h = get_seed_points_base(im_sm, return_h=True, th_std=4)
    cr2_ = np.array([z, x, y]).T
    cr2_cand = cr2_+tzxy

    M = cdist(cr1, cr2_cand)
    M_th = M <= dist_th
    pairs = [(_cr1, cr2_[m][np.argmax(h[m])])
             for _cr1, m in zip(cr1, M_th) if np.sum(m) > 0]
    cr1, cr2 = map(np.array, zip(*pairs))
    pfits2, keep = fast_local_fit(im_sm, cr2, radius=5, width_zxy=[
                                  1, 1, 1], return_good=True)
    cr2 = pfits2[:, 1:4]
    return np.array(cr1)[keep], cr2


def get_STD_beaddrift_v2(ims_beads, coord_sel=None, sz_ex=50, desired_nbeads=20,
                         desired_displ=0.4, hseed=150, nseed=100,
                         ref=None, force=False, save=True, save_file='temp.pkl'):
    """Given a list of bead images <ims_beads> this handles the fine bead drift correction.
    For each 3d image in <ims_beads> the beads for subimages of size <sz_ex>,
    centered at [center,center,center],[center,center,center]+[0,2*sz_ex,0] are fitted using #get_STD_centers with paramaters <hseed>,<nseed>.
    Beads for each of the two subimages are aligned with the corresponding beads for the reference image of index <ref> (default = len(ims_beads)/2) in ims_beads.
    """
    repeat = True
    txyzs_both = []
    txyzs_both_med = []
    bad_inds = []
    if save:
        save_cor = save_file
        if os.path.exists(save_cor):
            txyzs_both = pickle.load(open(save_cor, 'rb'))
            if len(txyzs_both) == len(ims_beads) and not force:
                repeat = False
                return txyzs_both, repeat
    if force:
        txyzs_both = []
        bad_inds = []
    if repeat:
        #get txyz
        if ref is None:
            ref = len(ims_beads)/2
        im_ref = ims_beads[ref]
        coord_sel = np.array(im_ref.shape)/2
        coord_sel1 = coord_sel
        im_ref_sm1, coords1 = grab_block(
            im_ref, coord_sel1, [sz_ex]*3, return_coords=True)
        # get_STD_centers(im_ref_sm,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the ref cube 1
        cents_ref1 = get_ref_pts(im_ref_sm1, dist_th=5, nbeads=nseed)

        coord_sel2 = np.array([0, -sz_ex, 0])+coord_sel
        im_ref_sm2, coords2 = grab_block(
            im_ref, coord_sel2, [sz_ex]*3, return_coords=True)
        # get_STD_centers(im_ref_sm2,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the ref cube 2
        cents_ref2 = get_ref_pts(im_ref_sm2, dist_th=5, nbeads=nseed)

        cutoff_ = 2
        xyz_res_ = 1
        for iim in bad_inds+range(len(txyzs_both), len(ims_beads)):
            print("Aligning "+str(iim+1))
            im = ims_beads[iim]
            txy_prev = np.array([0, 0, 0])
            th_good_bead = 2  # pixels
            #set1
            im_sm1, coords1 = grab_block(
                im, coord_sel1+txy_prev, [sz_ex]*3, return_coords=True)
            Tzxy = fft3d_from2d(im_ref_sm1, im_sm1, gb=5, max_disp=np.inf)
            cr1, cr2 = get_cand_pts(im_sm1, cents_ref1, Tzxy, dist_th=5)
            txyz1 = np.median(cr2-cr1, axis=0)
            c11 = np.sum(np.linalg.norm(cr2-cr1-txyz1, axis=-1) <
                         th_good_bead)  # sub pixel number of beads
            #set2
            im_sm2, coords2 = grab_block(
                im, coord_sel2+txy_prev, [sz_ex]*3, return_coords=True)
            Tzxy = fft3d_from2d(im_ref_sm2, im_sm2, gb=5, max_disp=np.inf)
            cr1, cr2 = get_cand_pts(im_sm2, cents_ref2, Tzxy, dist_th=5)
            txyz2 = np.median(cr2-cr1, axis=0)
            c21 = np.sum(np.linalg.norm(cr2-cr1-txyz2, axis=-1) <
                         th_good_bead)  # sub pixel number of beads

            txyz = (txyz1+txyz2)/2.
            print( txyz1, txyz2, c11, c21)
            displ = np.max(np.abs(txyz1-txyz2))
            if (displ > desired_displ) or (c11 < desired_nbeads) or (c21 < desired_nbeads):
                print( "Suspecting failure.")
                #set3
                coord_sel3 = np.array([0, sz_ex, 0])+coord_sel
                im_ref_sm3, coords3 = grab_block(
                    im_ref, coord_sel3, [sz_ex]*3, return_coords=True)
                cents_ref3 = get_ref_pts(im_ref_sm3, dist_th=5, nbeads=nseed)
                im_sm3, coords3 = grab_block(
                    im, coord_sel3+txy_prev, [sz_ex]*3, return_coords=True)
                Tzxy = fft3d_from2d(im_ref_sm3, im_sm3, gb=5, max_disp=np.inf)
                cr1, cr2 = get_cand_pts(im_sm3, cents_ref3, Tzxy, dist_th=5)
                txyz3 = np.median(cr2-cr1, axis=0)
                c31 = np.sum(np.linalg.norm(cr2-cr1-txyz3, axis=-1)
                             < th_good_bead)  # sub pixel number of beads

                measures = map(np.sum, map(
                    np.abs, [txyz3-txyz1, txyz3-txyz2, txyz1-txyz2]))
                imeasure = np.argmin(measures)
                nbds = [c11, c21, c31]
                nbds_variants = [[nbds[2], nbds[0]], [
                    nbds[2], nbds[1]], [nbds[0], nbds[1]]]
                variants = [[txyz3, txyz1], [txyz3, txyz2], [txyz1, txyz2]]
                best_pair = variants[imeasure]
                best_measure = measures[imeasure]

                if best_measure > 6*desired_displ or np.max(nbds_variants[imeasure]) < desired_nbeads:
                    best_pair = [[txyz1, txyz1], [txyz2, txyz2],
                                 [txyz3, txyz3]][np.argmax(nbds)]

                print( best_pair, measures[imeasure], nbds)

            else:
                best_pair = [txyz1, txyz2]
            #update txyzs_both
            if iim < len(txyzs_both):
                txyzs_both[iim] = best_pair
            else:
                txyzs_both.append(best_pair)
        if save:
            save_cor = save_file
            pickle.dump(txyzs_both, open(save_cor,'wb'))
    return txyzs_both, repeat
