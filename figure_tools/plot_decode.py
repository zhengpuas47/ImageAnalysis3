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

def plot_spot_stats(_spot_groups, _spot_usage, _max_usage=5, codebook=None,
                    save=True, save_filename=None,
                    show_image=True, verbose=True,
                    ):
    """Plot spot_group stats in decoder groups"""
    fig, axes = plt.subplots(1,3, figsize=(6,2), dpi=150)
    # Plot spot usage
    axes[0].hist(_spot_usage, bins=np.arange(_max_usage), width=0.8, color='r')
    axes[0].set_title("Usage of each spot", fontsize=8, pad=3)
    axes[0].set_xticks(np.arange(0.5, _max_usage+0.5))
    axes[0].set_xticklabels(np.arange(_max_usage))
    # Plot spot number in groups
    axes[1].hist([len(_g.spots) for _g in _spot_groups], bins=np.arange(_max_usage), width=0.8, color='y')
    axes[1].set_title("Spot Num. in Groups", fontsize=8, pad=3)
    axes[1].set_xticks(np.arange(0.5, _max_usage+0.5))
    axes[1].set_xticklabels(np.arange(_max_usage))

    # Plot number of decoded spots for each region
    regions = codebook['id'].values
    _decoded_counts = np.zeros(len(regions))
    for _g in _spot_groups:
        _decoded_counts[regions==_g.tuple_id] += 1
    axes[2].hist(_decoded_counts, bins=np.arange(0,np.max(_decoded_counts)+1), width=1, color='b')
    axes[2].set_title("Decoded Num. in regions", fontsize=8, pad=3)
    axes[2].set_xticks(np.arange(0.5, np.max(_decoded_counts)+1,2))
    axes[2].set_xticklabels(np.arange(0,np.max(_decoded_counts)+1,2).astype(np.int32))
    # total figure
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


def Centering_Chr2ZxysListDict(chr_2_zxys_list):
    # center
    _all_zxys = []
    for _zxys_list in chr_2_zxys_list.values():
        _all_zxys.extend(list(_zxys_list))
    _center = np.nanmean(np.concatenate(_all_zxys), axis=0)
    _centered_dict = {}
    for _chr, _zxys_list in chr_2_zxys_list.items():
        _centered_dict[_chr] = _zxys_list - _center[np.newaxis,:]
        
    return _centered_dict

def summarize_chr2Zxys(chr_2_zxys_list, codebook_df, keep_valid=False):
    # generate an order and sort by chr
    from ..structure_tools.distance import Generate_PlotOrder
    _chr_2_indices, _ = Generate_PlotOrder(codebook_df, codebook_df, sort_by_region=False) 
    _merged_zxys = []
    _merged_region_ids = []
    
    for _chr_name, _chr_inds in _chr_2_indices.items():
        if _chr_name in chr_2_zxys_list:
            _zxys_list = chr_2_zxys_list[_chr_name]
            for _zxys in _zxys_list:
                if keep_valid:
                    if len(np.shape(_zxys)) == 2:
                        _valid_flags = np.isfinite(_zxys).all(1)
                    else:
                        _valid_flags = np.isfinite(_zxys)
                    _merged_zxys.append(_zxys[_valid_flags])
                    _merged_region_ids.append(_chr_inds[_valid_flags])
                else:
                    _merged_zxys.append(_zxys)
                    _merged_region_ids.append(_chr_inds)
    return np.concatenate(_merged_zxys), np.concatenate(_merged_region_ids)