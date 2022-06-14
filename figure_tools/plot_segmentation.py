import numpy as np
import matplotlib.pyplot as plt


def plot_segmentation(_mask, _figsize=(4,3), _dpi=150, _cmap='Spectral', 
                      show_image=False, save=True, save_filename=None, verbose=True):
    """Plot segmentation result"""                      
    ## plot segmentation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import cm
    import copy
    _cmap = copy.copy(getattr(cm, _cmap))
    _cmap.set_bad([0.8, 0.8, 0.8])
    # plot 2D
    if len(np.shape(_mask)) == 2:
        _plot_mask = copy.copy(_mask)
    elif len(np.shape(_mask)) == 3:
        _plot_mask = _mask.max(0).astype(np.float32)
    # set nan
    _plot_mask[_plot_mask <= 0] = np.nan
    # plot
    fig, ax = plt.subplots(figsize=_figsize, dpi=_dpi)
    _pf = ax.imshow(_plot_mask, cmap=_cmap, vmin=0)
    # set ticks
    ax.tick_params('both', labelsize=8, 
                    width=0.75, length=2,
                    pad=1, labelleft=True, labelbottom=True) # remove bottom ticklabels for ax
    [i[1].set_linewidth(0.75) for i in ax.spines.items()]
    # colorbar ax
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='7%', pad="5%")
    cbar = plt.colorbar(_pf,cax=cax, ax=ax, )

    cbar.ax.tick_params('both', labelsize=8, 
                    width=0.5, length=2,
                    pad=1, labelleft=False) # remove bottom ticklabels for ax
    cbar.outline.set_linewidth(0.5)
    cbar.set_label('Cell label', 
                   fontsize=8, labelpad=5, rotation=270)
    # save 
    if save:
        if save_filename is not None:
            if verbose:
                print(f"-- save iamage to file: {save_filename}")
            fig.savefig(save_filename)
    
    if show_image:
        fig.show()
        return ax
    else:
        return None
    