# required packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, glob, sys, time

# load common parameters
from .. import visual_tools
from .color import transparent_gradient
from . import _dpi,_single_col_width,_double_col_width,_single_row_height,_ref_bar_length, _ticklabel_size,_ticklabel_width,_font_size


def visualize_2d_projection(im, kept_axes=(1,2), ax=None, 
                            crop=None, cmap='gray', color_limits=None, figure_alpha=1, 
                            add_colorbar=False, add_reference_bar=True, 
                            reference_bar_length=_ref_bar_length, reference_bar_color=[0,0,0], 
                            figure_width=_single_col_width, figure_dpi=_dpi, imshow_kwargs={},
                            ticklabel_size=_ticklabel_size, ticklabel_width=_ticklabel_width, font_size=_font_size,
                            save=False, save_folder='.', save_basename='2d-projection.png', verbose=True):
    """Function to draw a 2d-projection of a representatitve image
    Inputs:
        im: imput image, np.ndarray
        kept_axes: kept axies to project this image, tuple of 2 (default: (1,2))
        ax: input axis to plot image, if not given then create a new one, matplotlib.axis (default: None)
        crop: crop for 2d projected image, None or np.ndarray (defualt: None, no cropping)
        cmap: color-map used for the map, string or matplotlib.cm (default: 'gray')
        color_limits: color map limits for image, array-like of 2 (default: None,0-max)
        add_colorbar: whether add colorbar to image, bool (defualt: False)
        add_reference_bar: whether add refernce bar to image, bool (default: True)
        reference_bar_length: length of reference bar, float (default: bar length representing 1um)
        figure_width: figure width in inch, float (default: _single_col_width=2.25) 
        figure_dpi: figure dpi, float (default: _dpi=300)
        save: whether save image into file, bool (default: False)
        save_folder: where to save image, str of path (default: '.', local)
        save_basename: file basename to save, str (default: '2dprojection.png')
        verbose: say something during plotting, bool (default: True)
    Output:
        ax: matplotlib.axis handle containing 2d image
    """
    ## check inputs
    _im = np.array(im)
    if len(_im.shape) < 2:
        raise IndexError(f"Wrong shape of image:{_im.shape}, should have at least 2-axes")
    if len(kept_axes) != 2:
        raise IndexError(f"Wrong length of kept_axis:{len(kept_axes)}, should be 2")
    for _a in kept_axes:
        if _a >= len(_im.shape):
            raise ValueError(f"Wrong kept axis value {_a}, should be less than image dimension:{len(_im.shape)}")
    # project image
    _proj_axes = tuple([_i for _i in range(len(_im.shape)) if _i not in kept_axes])
    _im = np.mean(_im, axis=_proj_axes)
    # crops
    if crop is None:
        crop_slice = tuple([slice(0, _im.shape[0]), slice(0, _im.shape[1])])
    else:
        crop = np.array(crop, dtype=np.int)
        if crop.shape[0] != 2 or crop.shape[1] != 2:
            raise ValueError(f"crop should be 2x2 array telling 2d crop of the image")
        crop_slice = tuple([slice(max(0, crop[0,0]), min(_im.shape[0], crop[0,1])),
                           slice(max(0, crop[1,0]), min(_im.shape[1], crop[1,1])) ])
    _im = _im[crop_slice] # crop image
    # color_limits
    if color_limits is None:
        color_limits = [0, np.max(_im)]
    elif len(color_limits) < 2:
        raise IndexError(f"Wrong input length of color_limits:{color_limits}, should have at least 2 elements.")
    ## create axis if necessary
    if ax is None:
        fig, ax = plt.subplots(figsize=(_single_col_width, _single_col_width*_im.shape[0]/im.shape[1]*(5/6)**add_colorbar),
                               dpi=figure_dpi)
        #grid = plt.GridSpec(3, 1, height_ratios=[5,1,1], hspace=0., wspace=0.2)
        _im_obj = ax.imshow(_im, cmap=cmap, 
                            vmin=min(color_limits), vmax=max(color_limits), 
                            alpha=figure_alpha, **imshow_kwargs)
        ax.axis('off')
        #ax.tick_params('both', width=ticklabel_width, length=0, labelsize=ticklabel_size)
        #ax.get_xaxis().set_ticklabels([])
        #ax.get_yaxis().set_ticklabels([])
        if add_colorbar:
            cbar = plt.colorbar(_im_obj, ax=ax, shrink=0.9)
        if add_reference_bar:
            if isinstance(reference_bar_color, list) or isinstance(reference_bar_color, np.ndarray):
                _ref_color = reference_bar_color[:3]
            else:
                _ref_color = matplotlib.cm.get_cmap(cmap)(255)
            ax.hlines(y=_im.shape[0]-10, xmin=_im.shape[1]-10-reference_bar_length, 
                      xmax=_im.shape[1]-10, color=_ref_color, linewidth=2, visible=True)
    
    if save:
        save_basename = f"axis-{kept_axes}"+'_'+save_basename
        if '.png' not in save_basename:
            save_basename += '.png'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, save_basename), dpi=figure_dpi, transparent=True)
    
    return ax


def visualize_2d_gaussian(im, spot, color=[0,0,0], kept_axes=(1,2), ax=None, crop=None, 
                          spot_alpha=0.8, spot_sigma_scale=1, 
                          plot_im=True, im_cmap='gray', im_background='black', color_limits=None, figure_alpha=1, 
                          add_colorbar=False, add_reference_bar=True, 
                          reference_bar_length=_ref_bar_length,
                          figure_width=_single_col_width, figure_dpi=_dpi, projection_kwargs={},
                          ticklabel_size=_ticklabel_size, ticklabel_width=_ticklabel_width, font_size=_font_size,
                          save=False, save_folder='.', save_basename='2d-projection.png', verbose=True):
    """Function to visualize a 2d-gaussian in given spot object
    Inputs:
    
    Outputs:
        ax: matplotlib.axis including plotted gaussian feature"""

    ## check inputs
    _im = np.array(im)
    if len(_im.shape) < 2:
        raise IndexError(f"Wrong shape of image:{_im.shape}, should have at least 2-axes")
    if len(kept_axes) != 2:
        raise IndexError(f"Wrong length of kept_axis:{len(kept_axes)}, should be 2")
    for _a in kept_axes:
        if _a >= len(_im.shape):
            raise ValueError(f"Wrong kept axis value {_a}, should be less than image dimension:{len(_im.shape)}")
    # project image
    _proj_axes = tuple([_i for _i in range(len(_im.shape)) if _i not in kept_axes])
    _im = np.mean(_im, axis=_proj_axes)
    # crops
    if crop is None:
        crop_slice = tuple([slice(0, _im.shape[0]), slice(0, _im.shape[1])])
    else:
        crop = np.array(crop, dtype=np.int)
        if crop.shape[0] != 2 or crop.shape[1] != 2:
            raise ValueError(f"crop should be 2x2 array telling 2d crop of the image")
        crop_slice = tuple([slice(max(0, crop[0,0]), min(_im.shape[0], crop[0,1])),
                           slice(max(0, crop[1,0]), min(_im.shape[1], crop[1,1])) ])
    _im = _im[crop_slice] # crop image
    
    # generate image axis
    if plot_im:
        ax = visualize_2d_projection(im, kept_axes=kept_axes, ax=ax, crop=crop, cmap=im_cmap,
                                     color_limits=color_limits, figure_alpha=figure_alpha,
                                     add_colorbar=add_colorbar, add_reference_bar=add_reference_bar,
                                     reference_bar_length=reference_bar_length, figure_width=figure_width,
                                     figure_dpi=figure_dpi, imshow_kwargs=projection_kwargs, 
                                     ticklabel_size=ticklabel_size, ticklabel_width=ticklabel_width,
                                     save=False, save_folder=save_folder, save_basename=save_basename,
                                     verbose=verbose)
    elif ax is None:
        fig, ax = plt.subplots(figsize=(_single_col_width, _single_col_width*_im.shape[0]/im.shape[1]*(5/6)**add_colorbar),
                               dpi=figure_dpi)
    # extract spot parameters
    _spot_center = [spot[1:4][_i] for _i in kept_axes]
    _spot_sigma = [spot[5:8][_i] for _i in kept_axes]
    # get transparent profile
    _spot_im = visual_tools.add_source(np.zeros(np.shape(_im)), h=1, pos=_spot_center, sig=np.array(_spot_sigma)*spot_sigma_scale)
    # plot image
    ax.imshow(_spot_im, cmap=transparent_gradient(color, max_alpha=spot_alpha), vmin=0, vmax=1,)
    return ax