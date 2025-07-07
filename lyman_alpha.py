#
# lyman_alpha.py
#
# Computes Lyman-alpha emissivity (watts m^-3) given the local
# electron density, electron temperature, and ground-state neutral density.
#
#  Method:
#     (1) Compute the local n=2 population density using the Johnson-Hinnov
#         rate equations and coefficients [L.C.Johnson and E. Hinnov, J. Quant. 
#         Spectrosc. Radiat. Transfer. vol. 13 pp.333–358]
#     (2) Multiply by the n=2→1 spontaneous emission coefficient
#     (3) Convert to watts/m^3
#

import numpy as np
from create_jh_bscoef import Create_JH_BSCoef
from jhr_coef import JHR_Coef
from nhsaha import NHSaha


def Lyman_Alpha(Density, Te, N0, photons=False, create=False, no_null=False):
    """
    Compute Lyman-alpha emissivity (W/m^3) or photon emissivity (photons/m^3/s).

    Parameters
    ----------
    Density : ndarray
        Electron (and proton) density in m^-3
    Te : ndarray
        Electron temperature in eV
    N0 : ndarray
        Ground state neutral hydrogen density in m^-3
    photons : bool
        If True, return emissivity in photons/m^3/s instead of watts/m^3
    create : bool
        If True, create the B-spline coefficient file if not present
    no_null : bool
        If True, clamp values outside valid range instead of nulling them

    Returns
    -------
    result : ndarray
        Emissivity in W/m^3 or photons/m^3/s
    """

    # Input validation
    Density = np.asarray(Density, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)
    N0 = np.asarray(N0, dtype=np.float64)

    if not (Density.shape == Te.shape == N0.shape):
        raise ValueError("Density, Te, and N0 must all have the same shape")

    if create:
        Create_JH_BSCoef()

    data = np.load("jh_bscoef.npz")
    A_Lyman = data["A_lyman"]

    result = np.full_like(Density, 1.0e32)
    photons_array = np.full_like(Density, 1.0e32)

    # R coefficients for level n=2
    r02 = JHR_Coef(Density, Te, Ion=0, p=2, no_null=no_null)
    print(f"r02: {r02}")
    r12 = JHR_Coef(Density, Te, Ion=1, p=2, no_null=no_null)

    # Saha equilibrium densities
    NHSaha1 = NHSaha(Density, Te, p=1)
    NHSaha2 = NHSaha(Density, Te, p=2)

    mask = (
        (N0 > 0) & (N0 < 1e32) &
        (r02 < 1e32) & (r12 < 1e32) &
        (NHSaha1 < 1e32) & (NHSaha2 < 1e32)
    )
    ok = np.where(mask)[0]

    if ok.size > 0:
        n2 = (r02[ok] + r12[ok] * N0[ok] / NHSaha1[ok]) * NHSaha2[ok]
        photons_array[ok] = A_Lyman[0] * n2
        result[ok] = 13.6057 * 0.75 * photons_array[ok] * 1.6e-19

    if photons:
        return photons_array
    else:
        return result