#  Evaluates the elastic scattering (momentum transfer) <sigma v> for
#  H  atoms with energy E impacting on protons with temperature T.
#
#  Data generated from cross sections tablulated in:
#
# Janev, "Atomic and Molecular Processes in Fusion Edge Plasmas", Chapter 11 -
# Elastic and Related Cross Sections for Low-Energy Collisions among Hydrogen and
# Helium Ions, Neutrals, and Isotopes  by D.R. Schultz, S. Yu. Ovchinnikov, and S.V.
# Passovets, page 301.

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates # should match the idl interpolate function
from scipy.interpolate import bisplev


def SigmaV_EL_H_P(T, E, use_bspline=True):
    """
    Evaluates the elastic scattering <sigma*v> for H atoms (monoenergetic) colliding with protons (Maxwellian).
    
    Parameters
    ----------
    T : float or array-like
        Proton temperature in eV (can be scalar or array).
    E : float or array-like
        H atom energy in eV (same shape as T).
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
    LTH = np.log(_T)


    if use_bspline:

        try:
            spline = np.load("sigmav_el_h_p_bscoef.npz")
        except FileNotFoundError:
            raise FileNotFoundError("Could not find sigmav_el_h_p_bscoef.npz. Run create_sigmav_el_h_p_bscoef.py first.")
        
        tx = spline["tx"]
        ty = spline["ty"]
        c = spline["c"]
        kx = int(spline["kx"])
        ky = int(spline["ky"])
        tck = (tx, ty, c, kx, ky)

        # Ugly fudge because bisplev only evaluates pointwise
        pts = [bisplev(xi, yi, tck) for xi, yi in zip(LEP.flatten(), LTH.flatten())]
        result = np.exp(np.array(pts).reshape(LEP.shape) )


    else:
        # Load tabulated data
        data = np.load("sigmav_el_h_p_data.npz")
        Ln_E_Particle = data["Ln_E_Particle"]
        Ln_T_Target = data["Ln_T_Target"]
        SigmaV = data["SigmaV"]
        nEP = len(Ln_E_Particle) - 1
        nT = len(Ln_T_Target) - 1

        LEP = np.clip(LEP, Ln_E_Particle[0], Ln_E_Particle[-1])
        LTH = np.clip(LTH, Ln_T_Target[0], Ln_T_Target[-1])

        # Normalize to index range
        iE = (LEP - Ln_E_Particle[0]) * nEP / (Ln_E_Particle[nEP] - Ln_E_Particle[0])
        iT = (LTH - Ln_T_Target[0]) * nT / (Ln_T_Target[nT] - Ln_T_Target[0])

        coords = np.vstack([iE, iT])
        result = map_coordinates(SigmaV, coords, order=1, mode='nearest')

    return result

