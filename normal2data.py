def normal2data(norm, axis_meta, is_y=False):
    """
    Converts normalized plot coordinates to data coordinates.

    Parameters:
    - norm: array-like
    - axis_meta: dict with keys 'S' (offset, scale) and 'type' (0=linear, 1=log)
    - is_y: bool, if True uses y-axis metadata
    """
    scale = axis_meta['S'][1]
    offset = axis_meta['S'][0]
    dtype = axis_meta['type']
    data = (norm - offset) / scale
    if dtype != 0:
        data = 10.0 ** data
    return data
