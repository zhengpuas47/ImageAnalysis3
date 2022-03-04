import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
from . import _dpi, _single_col_width, _double_col_width, _single_row_height, _ref_bar_length, _ticklabel_size, _ticklabel_width, _font_size

def plot_decoding_ims(combo_ids, cropped_ims, _sel_bit_2_coords=None,
                      _sel_bit_2_usage=None,
                      num_cols=11, single_fig_size=1., _dpi=150,
                      title=None, max_th=None,
                      save=True, save_filename=None,
                      show_image=True,
                      ):
    """Plot array of cropped images"""
    
    # create fig
    num_rows = int(np.ceil(len(combo_ids)/num_cols))
    fig, ax_list = plt.subplots(num_rows, num_cols, 
                                figsize=(num_cols*single_fig_size, num_rows*single_fig_size*1.05),
                                dpi=_dpi)
    # threshold
    if max_th is None:
        max_th = np.max(cropped_ims) * 0.99
    min_th = np.min(cropped_ims)
    # plot
    for _i, (_bit, _im) in enumerate(zip(combo_ids, cropped_ims)):
        _ax = ax_list[int(_i/num_cols)][_i%num_cols]
        _ax.imshow(_im.max(0), cmap='Greys_r', vmin=np.min(_im), vmax=max_th)
        _ax.set_title(_bit, pad=2, fontsize=8)
        _ax.set_axis_off()
        # plot spots if applicable
        if _bit in _sel_bit_2_coords:
            if len(_sel_bit_2_coords[_bit]) > 0:
                _coords = np.array(_sel_bit_2_coords[_bit])
                if _bit in _sel_bit_2_usage:
                    _ax.scatter(_coords[:,2], _coords[:,1], c=_sel_bit_2_usage[_bit], cmap='rainbow', vmin=0, vmax=2)
                else:
                    _ax.plot(_coords[:,2], _coords[:,1], 'r.', )
    fig.subplots_adjust(hspace=0.2, wspace=0.05, top=0.93, bottom=0.03)
    if title is None:
        title = ''
    else:
        title = str(title)
    # append info
    title += f", vmax={max_th:.0f}"
    
    fig.suptitle(title, fontsize=12, y=0.97)

    # save 
    if save:
        if save_filename is not None:
            print(f"-- save iamage to file: {save_filename}")
            fig.savefig(save_filename)

    if show_image:
        fig.show()
        return ax_list
    else:
        return None

def plot_spot_stats(_spot_groups, _spot_usage, _max_usage=5,
                    save=True, save_filename=None,
                    show_image=True, verbose=True,
                    ):
    """Plot spot_group stats in decoder groups"""
    fig, axes = plt.subplots(1,2, figsize=(4,2), dpi=150)
    # Plot spot usage
    axes[0].hist(_spot_usage, bins=np.arange(_max_usage), width=0.8, color='r')
    axes[0].set_title("Spot Usage", fontsize=8, pad=3)
    axes[0].set_xticks(np.arange(0.5, _max_usage+0.5))
    axes[0].set_xticklabels(np.arange(_max_usage))
    # Plot spot number in groups
    axes[1].hist([len(_g.spots) for _g in _spot_groups], bins=np.arange(5), width=0.8, color='y')
    axes[1].set_title("Spot Num. in Groups", fontsize=8, pad=3)
    axes[1].set_xticks(np.arange(0.5, _max_usage+0.5))
    axes[1].set_xticklabels(np.arange(_max_usage))
    fig.subplots_adjust(wspace=0.3, top=0.80, bottom=0.12)
    fig.suptitle(f"{len(_spot_usage)} spots, {len(_spot_groups)} groups", 
                 fontsize=10, y=0.97)
    # save 
    if save:
        if save_filename is not None:
            if verbose:
                print(f"- Save spot_stats iamage to file: {save_filename}")
            fig.savefig(save_filename)
    if show_image:
        fig.show()
        return axes
    else:
        plt.close(fig)
        return None