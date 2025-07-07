#
# Interp_ScalarX.py
#
# Interpolates 'density' profiles used by Kinetic_Neutrals.pro,
# Kinetic_H2.pro, Kinetic_H2.pro, and other related procedures.

import numpy as np
from scipy.interpolate import interp1d


def interp_scalarx(fa, xa, xb, warn=None, debug=False):
    """
    Interpolates a scalar spatial profile from one spatial grid (Xa) to another (Xb).

    Parameters
    ----------
    fa : ndarray of shape (nXa,)
        Input function values defined on xa.

    xa : ndarray of shape (nXa,)
        Original spatial coordinate grid.

    xb : ndarray of shape (nXb,)
        Target spatial coordinate grid.

    warn : float, optional
        If specified, triggers a warning if interpolated values at the edges of xb
        are non-zero beyond a specified fraction of the maximum.

    debug : bool, optional
        If True, prints internal diagnostics.

    Returns
    -------
    fb : ndarray of shape (nXb,)
        Interpolated scalar function on the xb grid. Values outside the xa range are zero.
    """

    nxa = len(xa)
    if len(fa) != nxa:
        raise ValueError("Number of elements in fa and xa do not agree!")

    # Find indices in xb that lie within the range of xa
    okk = np.where((xb >= np.min(xa)) & (xb <= np.max(xa)))[0]
    nk = len(okk)
    if nk < 1:
        raise ValueError("No values of xb are within range of xa")

    k0, k1 = okk[0], okk[-1]
    nxb = len(xb)
    fb = np.zeros(nxb)

    # Perform interpolation only on the valid subset
    interp_fn = interp1d(xa, fa, kind='linear', bounds_error=False, fill_value=0.0)
    fb[okk] = interp_fn(xb[okk])

    # Warn if values at the boundaries are unexpectedly large
    if warn is not None:
        big = np.max(np.abs(fb))
        if k0 > 0 or k1 < nxb - 1:
            if k0 > 0 and np.abs(fb[k0]) > warn * big:
                print("Warning: Non-zero value of fb detected at min(Xa) boundary")
            if k1 < nxb - 1 and np.abs(fb[k1]) > warn * big:
                print("Warning: Non-zero value of fb detected at max(Xa) boundary")

    return fb
