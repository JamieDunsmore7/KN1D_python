# Evaluates the ionization rate coefficient S (m^-3 s^-1)
# from Johnson-Hinnov table 2 (MKS units) using bicubic spline interpolation on log-log data.
#

import numpy as np
from scipy.interpolate import bisplev
from create_jh_bscoef import Create_JH_BSCoef


def JHS_Coef(Density, Te, create=False, no_null=False):
    """
    Returns the ionization rate coefficient S (m^-3 s^-1) from Johnson-Hinnov table 2.

    Parameters
    ----------
    Density : ndarray
        Electron density in m^-3
    Te : ndarray
        Electron temperature in eV
    create : bool
        If True, (re)create the spline coefficients
    no_null : bool
        If True, clamp out-of-bounds values instead of masking them

    Returns
    -------
    Result : ndarray
        Ionization rate coefficient S (m^-3 s^-1)
    """

    # Ensure 1D arrays
    if Density.ndim != 1 or Te.ndim != 1:
        raise ValueError("Density and Te must be 1D arrays of the same shape.")

    if create:
        Create_JH_BSCoef()

    try:
        data = np.load("jh_bscoef.npz")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find jh_bscoef.npz. Run with create=True first.")


    S_tx = data["S_tx"]
    S_ty = data["S_ty"]
    S_c = data["S_c"]
    S_kx = int(data["S_kx"])
    S_ky = int(data["S_ky"])
    tck = (S_tx, S_ty, S_c, S_kx, S_ky)

    log_Density = np.log(np.asarray(Density, dtype=np.float64))
    log_Te = np.log(np.asarray(Te, dtype=np.float64))

    Result = np.full_like(Density, 1.0e32)

    if no_null:
        # NOTE: the upper limits of these clip values are NOT the same as in IDL
        # because the knot locations are in slightly different places
        log_Density = np.clip(log_Density, S_tx[0] + 0.001, S_tx[-1] - 0.001)
        log_Te = np.clip(log_Te, S_ty[0] + 0.001, S_ty[-1] - 0.001)
        ok = np.arange(log_Density.size)
    else:
        mask = (
            (log_Density >= S_tx[0]) & (log_Density <= S_tx[-1]) &
            (log_Te >= S_ty[0]) & (log_Te <= S_ty[-1])
        )
        ok = np.where(mask)[0]

    if ok.size > 0:
        vals = np.array([
            bisplev(ld, lt, tck)
            for ld, lt in zip(log_Density[ok], log_Te[ok])
        ])
        Result[ok] = np.exp(vals)

    return Result