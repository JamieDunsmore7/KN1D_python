import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d


def path_interp_2d(P, PX, PY, X, Y):
    """
    Faithful translation of IDL's Path_Interp_2D.pro.

    Interpolates a 2D array P along a trajectory defined by coordinates X and Y.
    P is defined on the grid (PX, PY), and PX and PY must be monotonically increasing.

    Parameters
    ----------
    P : ndarray, shape (len(PX), len(PY))
        The input 2D array to interpolate from.

    PX : ndarray, shape (n,)
        Grid coordinates along the first dimension (axis 0) of P.

    PY : ndarray, shape (m,)
        Grid coordinates along the second dimension (axis 1) of P.

    X : ndarray, shape (N,)
        X coordinates of the trajectory.

    Y : ndarray, shape (N,)
        Y coordinates of the trajectory.

    Returns
    -------
    interpolated : ndarray, shape (N,)
        Values of P interpolated at points (X, Y).
    """

    PX = np.asarray(PX)
    PY = np.asarray(PY)
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Check monotonicity (increasing)
    if not np.all(np.diff(PX) > 0):
        raise ValueError("ERROR in PATH_INTERP_2D => PX is non-monotonic!")
    if not np.all(np.diff(PY) > 0):
        raise ValueError("ERROR in PATH_INTERP_2D => PY is non-monotonic!")

    # Create index grids
    iPX = np.arange(len(PX), dtype=float)
    iPY = np.arange(len(PY), dtype=float)

    # Map physical coordinates to index space
    interp_iX = interp1d(PX, iPX, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_iY = interp1d(PY, iPY, kind='linear', bounds_error=False, fill_value=np.nan)

    iX = interp_iX(X)
    iY = interp_iY(Y)

    # Interpolate in index space (order=1 → bilinear interpolation)
    coords = np.vstack([iX, iY])  # shape (2, N)
    result = map_coordinates(P, coords, order=1, mode='constant', cval=np.nan)

    return result
