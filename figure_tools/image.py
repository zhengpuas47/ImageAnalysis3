# required packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, glob, sys, time
# 3d plotting
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# load common parameters
from .. import visual_tools
from .. import _distance_zxy
from .color import transparent_gradient
from . import _dpi,_single_col_width,_double_col_width,_single_row_height,_ref_bar_length, _ticklabel_size,_ticklabel_width,_font_size


def visualize_2d_projection(im, kept_axes=(1,2), ax=None, 
                            crop=None, cmap='gray', color_limits=None, figure_alpha=1, 
                            add_colorbar=False, add_reference_bar=True, 
                            reference_bar_length=_ref_bar_length, reference_bar_color=[1,1,1], 
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
        fig, ax = plt.subplots(figsize=(figure_width, figure_width*_im.shape[0]/im.shape[1]*(5/6)**add_colorbar),
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
        fig, ax = plt.subplots(figsize=(figure_width, figure_width*_im.shape[0]/im.shape[1]*(5/6)**add_colorbar),
                               dpi=figure_dpi)
    # extract spot parameters
    _spot_center = [spot[1:4][_i] for _i in kept_axes]
    _spot_sigma = [spot[5:8][_i] for _i in kept_axes]
    # get transparent profile
    _spot_im = visual_tools.add_source(np.zeros(np.shape(_im)), h=1, pos=_spot_center, sig=np.array(_spot_sigma)*spot_sigma_scale)
    # plot image
    ax.imshow(_spot_im, cmap=transparent_gradient(color, max_alpha=spot_alpha), vmin=0, vmax=1,)

    # save
    if save:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if '.png' not in save_basename:
            save_basename += '.png'
        save_filename = os.path.join(save_folder, save_basename)
        plt.savefig(save_filename, transparent=True)
    
    return ax


def chromosome_structure_3d_rendering(spots, ax3d=None, cmap='Spectral', 
                                      distance_zxy=_distance_zxy, image_radius=2000,
                                      marker_size=6, marker_alpha=1, 
                                      background_color=[0,0,0], 
                                      line_width=1, line_alpha=1, 
                                      view_elev_angle=90, view_azim_angle=0, 
                                      add_reference_bar=True, reference_bar_length=1000, 
                                      reference_bar_width=2, reference_bar_color=[1,1,1],
                                      tick_label_length=_ticklabel_size, tick_label_width=_ticklabel_width, 
                                      font_size=_font_size, add_colorbar=True,
                                      figure_width=_single_col_width, figure_dpi=_dpi, 
                                      save=False, save_folder='.', save_basename='3d-projection.png', verbose=True):
    """Function to visualize 3d rendering of chromosome structure
    Inputs:
    
    
    Outputs:
        ax: matplotlib axes object containing chromosome structures
    """
    ## check inputs
    # spots
    _spots = np.array(spots)
    if len(np.shape(_spots)) != 2:
        raise IndexError(f"Wrong _spots dimension, should be 2d array but {_spot.shape} is given")
    # prepare spots
    if _spots.shape[1] == 3:
        _zxy = _spots 
    else:
        _zxy = _spots[:,1:4] * distance_zxy[np.newaxis,:]
    _n_zxy = visual_tools.normalize_center_spots(_zxy, distance_zxy=distance_zxy, 
                                                 center=True, scale_variance=False, 
                                                 pca_align=True, scaling=1)
    _valid_inds = (np.isnan(_n_zxy).sum(1) == 0)
    
    # set dimension
    if image_radius is None:
        _radius = np.nanmax(np.abs(_n_zxy)) + reference_bar_length/2 
    else:
        _radius = image_radius
    # cmap
    if isinstance(cmap, str):
        _cmap = matplotlib.cm.get_cmap(cmap)
        _colors = np.array([np.array(_cmap(_i)[:3]) for _i in np.linspace(0,1,len(_spots))])
    elif isinstance(cmap, np.ndarray) or isinstance(cmap, list):
        if len(cmap) == len(_spots):
            # create cmap
            _cmap_mat = np.ones([len(cmap), 4])
            _cmap_mat[:,:len(cmap[0])] = np.array(cmap)
            _cmap = ListedColormap(_cmap_mat)
            # color
            _colors = np.array(cmap)
        else:
            raise IndexError(f"length of cmap doesnt match number of spots")
    elif isinstance(cmap, matplotlib.colors.LinearSegmentedColormap): 
        _cmap = cmap
        _colors = np.array([np.array(_cmap(_i)[:3]) for _i in np.linspace(0,1,len(_spots))])

    elif isinstance(cmap, matplotlib.colors.ListedColormap):
        _cmap = cmap
        _colors = np.array([np.array(_cmap(_i)[:3]) for _i in range(len(_spots))])
    else:
        raise TypeError(f"Wrong input type for cmap:{type(cmap)}, should be str or matplotlib.colors.LinearSegmentedColormap")
    # extract colors from cmaps
    
    # background and reference bar color
    _back_color = np.array(background_color[:3])
    if isinstance(reference_bar_color, np.ndarray) or isinstance(reference_bar_color, list):
        _ref_bar_color = np.array(reference_bar_color[:3])
    elif reference_bar_color is None:
        _ref_bar_color = 1 - _back_color
    else:
        raise TypeError(f"Wrong input type for reference_bar_color:{type(reference_bar_color)}")

    ## plot
    # initialize figure
    if ax3d is None:
        fig = plt.figure(figsize=(figure_width, figure_width), dpi=figure_dpi)
        ax3d = fig.gca(projection='3d')
    else:
        if not isinstance(ax3d, matplotlib.axes._subplots.Axes3DSubplot):
            raise TypeError(f"Wrong input type for ax3d:{type(ax3d)}, it should be Axec3DsSubplot object.")
    # background color
    ax3d.set_facecolor(_back_color)
   
    # scatter plot
    _sc = ax3d.scatter(_n_zxy[_valid_inds,0], _n_zxy[_valid_inds,1], _n_zxy[_valid_inds,2],
                       c=_colors[_valid_inds], s=marker_size, depthshade=False, 
                       edgecolors=[[0,0,0, _c[-1]] for _c in _colors[_valid_inds]], 
                       linewidth=0.05)
    # plot lines between spots
    for _i,_coord in enumerate(_n_zxy[:-1]):
        _n_coord = _n_zxy[_i+1]
        # if both coordinates are valid:
        if _valid_inds[_i] and _valid_inds[_i+1]:
            ax3d.plot([_coord[0],_n_coord[0]],
                      [_coord[1],_n_coord[1]],
                      [_coord[2],_n_coord[2]],
                      color = _colors[_i], alpha=_colors[_i][-1], linewidth=line_width)
    # plot reference bar
    if add_reference_bar:
        _bar_starts = np.array([np.sin(view_elev_angle/180*np.pi)*_radius, 
                                np.sin(view_elev_angle/180*np.pi)*_radius, 
                                - np.cos(view_elev_angle/180*np.pi)*_radius])
        _bar_ends = _bar_starts  -  reference_bar_length * \
                      np.array([np.sin(view_azim_angle/180*np.pi) * np.sin(view_elev_angle/180*np.pi),
                                np.cos(view_azim_angle/180*np.pi), 
                                np.sin(view_azim_angle/180*np.pi) * np.cos(view_elev_angle/180*np.pi),
                                ])
        
        _ref_line = ax3d.plot([_bar_starts[0], _bar_ends[0]],
                              [_bar_starts[1], _bar_ends[1]], 
                              [_bar_starts[2], _bar_ends[2]], 
                              color=_ref_bar_color, 
                              linewidth=reference_bar_width)
    # colorbar    
    if add_colorbar:
        import matplotlib.cm as cm
        _color_inds = np.where(_valid_inds)[0]
        norm = cm.colors.Normalize(vmax=_color_inds.max(), vmin=_color_inds.min())
        print(norm, _cmap)
        m = cm.ScalarMappable(cmap=_cmap, norm=norm)
        m.set_array(_color_inds)
        #divider = make_axes_locatable(ax3d)
        #cax = divider.append_axes('bottom', size='6%', pad="2%")
        cb = plt.colorbar(m, ax=ax3d, orientation='horizontal', pad=0.01)
        cb.ax.tick_params(labelsize=font_size, width=tick_label_width, length=tick_label_length-1,pad=1)
        # border
        cb.outline.set_linewidth(tick_label_width)

    # axis view angle
    ax3d.grid(False)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.axis('off')
    # view angle
    ax3d.view_init(elev=view_elev_angle, azim=view_azim_angle)
    # set limits
    ax3d.set_xlim([-_radius, _radius])
    ax3d.set_ylim([-_radius, _radius])
    ax3d.set_zlim([-_radius, _radius])
    # save
    if save:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if '.png' not in save_basename:
            save_basename += '.png'
        save_filename = os.path.join(save_folder, save_basename)
        print(save_filename)
        plt.savefig(save_filename, transparent=False)

    return ax3d