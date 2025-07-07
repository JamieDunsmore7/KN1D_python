#+
# SigmaV_EL_HH_P.py
#
#  Evaluates the elastic scattering (momentum transfer) <sigma v> for
#  H2 molecules with energy E impacting on protons with temperature T.
#
#  Data generated from cross sections tablulated in:
#
# Janev, "Atomic and Molecular Processes in Fusion Edge Plasmas", Chapter 11 -
# Elastic and Related Cross Sections for Low-Energy Collisions among Hydrogen and
# Helium Ions, Neutrals, and Isotopes  by D.R. Schultz, S. Yu. Ovchinnikov, and S.V.
# Passovets, page 305.

import numpy as np
from scipy.interpolate import bisplev, RectBivariateSpline
from scipy.ndimage import map_coordinates # should match the idl interpolate function



def SigmaV_EL_HH_P(T, E, use_bspline=True):
    """
    Evaluates the elastic scattering <sigma*v> for H2 molecules (monoenergetic) colliding with protons (Maxwellian).
    
    Parameters
    ----------
    T : float or array-like
        Proton temperature in eV (can be scalar or array).
    E : float or array-like
        H2 molecule energy in eV (same shape as T).
    use_bspline : bool
        Whether to use B-spline interpolation (True) or bilinear interpolation (False).

    Returns
    -------
    result : float or ndarray
        Sigma*v in m^3/s (same shape as input)
        if T and/or E is outside this range, the value on the boundary is returned
    """

    # Convert to numpy arrays
    E = np.asarray(E, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    # Check shapes
    if E.shape != T.shape:
        raise ValueError("Number of elements in E and T must match")

    # Clamp E and T to valid range
    _E = np.clip(E, 1e-4, 1e5)
    _T = np.clip(T, 1e-4, 1e5)

    LEP = np.log(_E)
    LTH2 = np.log(_T)

    if use_bspline:
        # Load B-spline coefficient data
        data = np.load("sigmav_el_h2_p_bscoef.npz")
        tx = data["EKnot_EL_H2_P"]
        ty = data["TKnot_EL_H2_P"]
        c = data["LogSigmaV_EL_H2_P_BSCoef"]
        k = int(data["order_EL_H2_P"])


        # Clamp logE and logT to knot range
        LEP = np.clip(LEP, np.min(tx), np.max(tx))
        LTH2 = np.clip(LTH2, np.min(ty), np.max(ty))

        # Evaluate B-spline on 2D grid
        result = np.exp(bisplev(LEP, LTH2, (tx, ty, c, k, k)))

    else:
        # Load tabulated data
        data = np.load("sigmav_el_h2_p_data.npz")
        Ln_E_Particle = data["Ln_E_Particle"]
        Ln_T_Target = data["Ln_T_Target"]
        SigmaV = data["SigmaV"]
        nEP = len(Ln_E_Particle) - 1
        nT = len(Ln_T_Target) - 1

        LEP = np.clip(LEP, Ln_E_Particle[0], Ln_E_Particle[-1])
        LTH2 = np.clip(LTH2, Ln_T_Target[0], Ln_T_Target[-1])

        # Normalize to index range
        iE = (LEP - Ln_E_Particle[0]) * nEP / (Ln_E_Particle[nEP] - Ln_E_Particle[0])
        iT = (LTH2 - Ln_T_Target[0]) * nT / (Ln_T_Target[nT] - Ln_T_Target[0])

        coords = np.vstack([iE, iT])
        result = map_coordinates(SigmaV, coords, order=1, mode='nearest')

    return result
