import numpy as np
import pandas as pd

def Load_PositionFile(position_filename):
    """Load position.txt file generated from Steve"""
    positions = pd.read_table(position_filename, delimiter=',', header=None)
    positions.columns = ['x','y']
    return positions

    