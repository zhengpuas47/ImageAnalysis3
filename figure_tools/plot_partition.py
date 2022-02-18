import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

def plot_cell_spot_counts(cell_spot_counts,
                          _figsize=(4,3), _dpi=150,
                          expected_count=60, 
                          save=True, save_filename=None,
                          show_image=True,
                          ):
    """Plot cell-spot_count matrix"""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=_figsize, dpi=_dpi, )

    _pf = ax.imshow(cell_spot_counts, cmap='Spectral_r', vmin=0, vmax=expected_count)

    ax.tick_params('both', labelsize=8, 
                    width=0.5, length=2,
                    pad=1, labelleft=True, labelbottom=True) # remove bottom ticklabels for ax
    [i[1].set_linewidth(0.5) for i in ax.spines.items()]

    ax.set_xlabel('Bit', fontsize=8, labelpad=1)

    ax.set_ylabel('Cell id', fontsize=8, labelpad=0)
    # locate ax
    divider = make_axes_locatable(ax)
    # colorbar ax
    cax = divider.append_axes('right', size='7%', pad="5%")
    cbar = plt.colorbar(_pf,cax=cax, ax=ax, )

    cbar.ax.tick_params('both', labelsize=8, 
                    width=0.5, length=2,
                    pad=1, labelleft=False) # remove bottom ticklabels for ax
    cbar.outline.set_linewidth(0.5)
    cbar.set_label('CandSpots count', 
                   fontsize=7.5, labelpad=6, rotation=270)
    # save 
    if save:
        if save_filename is not None:
            print(f"-- save iamage to file: {save_filename}")
            fig.savefig(save_filename)
    
    if show_image:
        fig.show()
        return ax
    else:
        return None