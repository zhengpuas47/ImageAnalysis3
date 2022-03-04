import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import os
import matplotlib.cm as cm
import copy
from scipy.spatial.distance import pdist, squareform

from . import _dpi,_single_col_width,_double_col_width,_single_row_height,_ref_bar_length, _ticklabel_size,_ticklabel_width,_font_size

# draw distance map
def plot_distance_map(distmap, ax=None, cmap='seismic_r', 
                      color_limits=[0,1500], color_norm=None, imshow_kwargs={},
                      ticks=None, tick_labels=None, 
                      tick_label_length=_ticklabel_size, tick_label_width=_ticklabel_width, 
                      font_size=_font_size, ax_label=None,
                      add_colorbar=True, colorbar_labels=None, colorbar_kwargs={},
                      adjust_kwargs={'left':0.15, 'right':0.85, 'bottom':0.15},
                      figure_width=_single_col_width, figure_dpi=_dpi, 
                      save=False, save_folder='.', save_basename='distmap.png', verbose=True):
    """Function to plot distance maps"""
    
    ## check inputs
    if np.shape(distmap)[0] != np.shape(distmap)[1]:
        raise IndexError(f"Wrong input dimension for distmap, should be nxn matrix but {distmap.shape} is given")

    
    ## create image
    if ax is None:
        fig, ax = plt.subplots(figsize=(figure_width, figure_width),
                               dpi=figure_dpi)
    
    _distmap = distmap.copy()
    _distmap[_distmap<min(color_limits)] = min(color_limits)
    # generate imshow object
    _im = ax.imshow(_distmap, cmap=cmap, interpolation='nearest', norm=color_norm,
                    vmin=min(color_limits), vmax=max(color_limits), **imshow_kwargs)
    # border
    [i[1].set_linewidth(tick_label_width) for i in ax.spines.items()]
    # ticks
    ax.tick_params('both', labelsize=font_size, 
                   width=tick_label_width, length=tick_label_length,
                   pad=1)
    if ticks is None:
        _used_ticks = np.arange(0, len(_distmap), 2*10**np.floor(np.log10(len(distmap))))
    else:
        _used_ticks = ticks
    ax.set_xticks(_used_ticks, minor=False)
    ax.set_yticks(_used_ticks, minor=False)
    # tick labels
    if tick_labels is not None:
        # apply tick labels 
        if len(tick_labels) == len(_distmap):
            _used_labels = [_l for _i, _l in enumerate(tick_labels) if _i in _used_ticks]
            ax.set_xticklabels(_used_labels, rotation=60)
            ax.set_yticklabels(_used_labels)
        elif len(tick_labels) == len(_used_ticks):
            ax.set_xticklabels(tick_labels, rotation=60)
            ax.set_yticklabels(tick_labels)
        else:
            print(f"tick_labels length:{len(tick_labels)} doesn't match distmap:{len(_distmap)}, skip!")
    # axis labels
    if ax_label is not None:
        ax.set_xlabel(ax_label, labelpad=2, fontsize=font_size)
        ax.set_ylabel(ax_label, labelpad=2, fontsize=font_size)
    # set limits
    ax.set_xlim([-0.5, len(_distmap)-0.5])
    ax.set_ylim([len(_distmap)-0.5, -0.5])
    # colorbar    
    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='6%', pad="2%")
        cb = plt.colorbar(_im, cax=cax, orientation='vertical', 
                          extend='neither', 
                          **colorbar_kwargs)
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=font_size, width=tick_label_width, length=tick_label_length-1,pad=1)
        [i[1].set_linewidth(_ticklabel_width) for i in cb.ax.spines.items()]
        # border
        cb.outline.set_linewidth(tick_label_width)
        if colorbar_labels is not None:
            cb.set_label(colorbar_labels, fontsize=_font_size, labelpad=5, rotation=270)


    # adjust size
    plt.gcf().subplots_adjust(bottom=0.15*bool(ax_label), 
                              left=0.2*bool(ax_label), 
                              right=1-0.15*bool(colorbar_labels))

    # save
    if save:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_filename = os.path.join(save_folder, save_basename)
        if '.png' not in save_filename and '.pdf' not in save_filename:
            save_filename += '.png'
        if '.pdf' in save_filename:
            matplotlib.rcParams['pdf.fonttype'] = 42

        plt.savefig(save_filename, transparent=True)

    # return
    return ax


def GenomeWide_DistMap(all_chr_zxys, 
                       all_chr_names, all_chr_boundaries,
                       color_limits=[0,5], cmap='seismic_r',
                       figsize=(6,5), dpi=_dpi,
                       save=True, save_filename=None, 
                       show_image=False, verbose=True):
    """Plot genomewide distance map, given list of chr_zxys, chr_names and chr_boundary_inds"""
    _cmap = copy.copy( getattr(cm, cmap ) )
    _cmap.set_bad([0.5,0.5,0.5])
    
    try:
        _distmap = batch_zxys_2_distmap( np.concatenate(all_chr_zxys) )
    except:
        return None

    fig, ax = plt.subplots(figsize=figsize,dpi=dpi)
    _pf = ax.imshow(_distmap, cmap=_cmap, vmin=min(color_limits), vmax=max(color_limits))
    plt.colorbar(_pf, label=f'Pairwise distance (\u03bcm)')

    ax.set_xticks((all_chr_boundaries[1:] + all_chr_boundaries[:-1])/2)
    ax.set_xticklabels(all_chr_names, fontsize=6, rotation=60,)
    ax.set_yticks((all_chr_boundaries[1:] + all_chr_boundaries[:-1])/2)
    ax.set_yticklabels(all_chr_names, fontsize=6,)
    ax.tick_params(axis='both', which='major', pad=1)

    ax.hlines(all_chr_boundaries-0.5, 0, len(_distmap), color='black', linewidth=0.5)
    ax.vlines(all_chr_boundaries-0.5, 0, len(_distmap), color='black', linewidth=0.5)
    ax.set_xlim([0, len(_distmap)])
    ax.set_ylim([len(_distmap), 0])
    ax.set_title(f"kept_spots: { np.sum(np.isnan(np.concatenate(all_chr_zxys)).any(1)==0 ) }")        
    if save:
        if save_filename is not None:
            if verbose:
                print(f"- Save distmap into file: {save_filename}")
            fig.savefig(save_filename, dpi=dpi, transparent=True)
        else:
            print("save_filename not given.")
    # return
    if show_image:
        return ax
    else:
        plt.close(fig)
        return None

def batch_zxys_2_distmap(_zxys):
    return squareform(pdist(_zxys))