# Evaluates the R coefficient (m^3/s) from
# Johnson-Hinnov tables using bicubic spline interpolation on log-log data.
#

#import sys
#sys.path.append("/Users/jamiedunsmore/Documents/MIT/Research/KN1D_Python")
import numpy as np
from scipy.interpolate import bisplev
from create_jh_bscoef import Create_JH_BSCoef


def JHR_Coef(Density, Te, Ion, p, create=False, no_null=False):
    """
    Returns the r0(p) or r1(p) coefficients from Johnson-Hinnov tables.

    Parameters
    ----------
    Density : ndarray
        Electron density in m^-3
    Te : ndarray
        Electron temperature in eV
    Ion : int
        =0: recombination coefficient r0(p)
        =1: ionization coefficient r1(p)
    p : int
        Hydrogen energy level, must be in range 2 <= p <= 6
    create : bool
        If True, (re)create the spline coefficients
    no_null : bool
        If True, clamp out-of-bounds values instead of masking them

    Returns
    -------
    Result : ndarray
        Evaluated rate coefficient in m^3/s
    """

    if Density.ndim != 1 or Te.ndim != 1:
        raise ValueError("Density and Te must be 1D arrays of the same shape")
    if not np.isscalar(Ion) or Ion not in [0, 1]:
        raise ValueError('"Ion" must be a scalar 0 or 1')
    if not np.isscalar(p) or not (2 <= p <= 6):
        raise ValueError('"p" must be a scalar integer in range 2 <= p <= 6')

    if create:
        Create_JH_BSCoef()

    try:
        data = np.load("jh_bscoef.npz")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find jh_bscoef.npz. Run with create=True first.")

    key = f"R_coeffs_{Ion}_{p}"
    if key not in data:
        raise ValueError(f"Spline coefficient '{key}' not found in .npz file.")

    tx = data["R_tx"]
    ty = data["R_ty"]
    c = data[key]
    kx = int(data["R_kx"])
    ky = int(data["R_ky"])
    tck = (tx, ty, c, kx, ky)

    Density = np.asarray(Density, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)

    if Density.shape != Te.shape:
        raise ValueError("Density and Te must have the same shape.")

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
