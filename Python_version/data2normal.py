import numpy as np

def data2normal(data, axis_meta, is_y=False):
    """
    Converts data coordinates to normalized plot coordinates.

    Parameters:
    - data: array-like
    - axis_meta: dict with keys 'S' (offset, scale) and 'type' (0=linear, 1=log)
    - is_y: bool, if True uses y-axis metadata
    """
    scale = axis_meta['S'][1]
    offset = axis_meta['S'][0]
    dtype = axis_meta['type']
    d = np.log10(data) if dtype != 0 else data
    return scale * d + offset
