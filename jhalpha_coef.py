# Evaluates the recombination rate coefficient α (m^3/s) from
# Johnson-Hinnov table 1, using bicubic spline interpolation on log-log data.
#

#import sys
#sys.path.append("/Users/jamiedunsmore/Documents/MIT/Research/KN1D_Python")
import numpy as np
from scipy.interpolate import bisplev
from create_jh_bscoef import Create_JH_BSCoef


def JHAlpha_Coef(Density, Te, create=False, no_null=False):
    """
    Returns the recombination rate coefficient α (m^3/s) from Johnson-Hinnov table 1.

    Parameters
    ----------
    Density : ndarray
        Electron density in m^-3 (must match shape of Te)
    Te : ndarray
        Electron temperature in eV (must match shape of Density)
    create : bool, optional
        If True, will call Create_JH_BSCoef() to generate B-spline data
    no_null : bool, optional
        If True, clamps values outside knot range to boundary;
        otherwise, values outside valid range are masked to 1e32

    Returns
    -------
    Result : ndarray
        Recombination rate coefficient in m^3/s
    """

    if Density.ndim != 1 or Te.ndim != 1:
        raise ValueError("Error added by Jamie: Density and Te must be 1D arrays of the same shape")

    # Load coefficients
    if create:
        Create_JH_BSCoef()

    try:
        data = np.load("jh_bscoef.npz")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find jh_bscoef.npz. Run with create=True first.")

    tx = data["alpha_tx"]
    ty = data["alpha_ty"]
    c = data["alpha_c"]
    kx = int(data["alpha_kx"])
    ky = int(data["alpha_ky"])
    tck = (tx, ty, c, kx, ky)

    Density = np.asarray(Density, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)

    if Density.shape != Te.shape:
        raise ValueError("Number of elements in Density and Te are different!")

    LDensity = np.log(Density)
    LTe = np.log(Te)

    Result = np.full_like(Density, 1.0e32)

    if no_null:
        # NOTE: the upper limits of these clip values are NOT the same as in IDL
        # because the knot locations are in slightly different places
        LDensity = np.clip(LDensity, tx[0] + 0.001, tx[-1] - 0.001)
        LTe = np.clip(LTe, ty[0] + 0.001, ty[-1] - 0.001)
        ok = np.arange(LDensity.size)
    else:
        mask = (
            (LDensity >= tx[0]) & (LDensity <= tx[-1]) &
            (LTe >= ty[0]) & (LTe <= ty[-1])
        )
        ok = np.where(mask)[0]

    if ok.size > 0:
        vals = np.array([
            bisplev(ld, lt, tck)
            for ld, lt in zip(LDensity[ok], LTe[ok])
        ])
        Result[ok] = np.exp(vals)

    return Result
