import os, glob, sys, time
import numpy as np 
import pandas

Axis3D_infos = ['z', 'x', 'y']
Spot3D_infos = ['height', 'z', 'x', 'y', 'background', 'sigma_z', 'sigma_x', 'sigma_y', 'sin_t', 'sin_p', 'eps']

Pixel3D_infos = [f"pixel_{_ax}" for _ax in Axis3D_infos]

def CellSpotsDf_2_CandSpots(_cell_spots_df, 
    spot_infos=Spot3D_infos, 
    pixel_infos=Pixel3D_infos):
    """Convert spot_df of a cell into cand_spot format"""
    from ..classes.preprocess import Spots3D
    return Spots3D(_cell_spots_df[spot_infos], 
                   bits=_cell_spots_df['bit'].values,
                   channels=_cell_spots_df['channel'].values,
                   pixel_sizes=np.unique(_cell_spots_df[pixel_infos].values, axis=0)[0])
