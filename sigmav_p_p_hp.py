#
# SigmaV_P_P_HP.py
#
# Returns Maxwellian-averaged <sigma*v> for electron impact
# dissociation of molecular hydrogen ions resulting in two protons.
# Coefficients are taken from Janev,
# "Elementary Processes in Hydrogen-Helium Plasmas", Springer-Verlag, 1987, p.260.
#

import numpy as np

def SigmaV_P_P_HP(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact dissociation
    of H₂⁺ into two protons, based on Janev (1987).

    Parameters
    ----------
    Te : float or ndarray
        Electron temperature in eV.

    Returns
    -------
    sigma_v : float or ndarray
        Maxwellian-averaged <σv> in m^3/s, valid for 0.1 < Te < 2e4.
    """

    Te = np.atleast_1d(Te).astype(np.float64)

    # Clamp to valid domain
    Te = np.clip(Te, 0.1, 2.01e4)

    # Coefficients from Janev (1987), p.260
    b = np.array([
        -3.746192301092e+1,
         1.559355031108e+1,
        -6.693238367093e+0,
         1.981700292134e+0,
        -4.044820889297e-1,
         5.352391623039e-2,
        -4.317451841436e-3,
         1.918499873454e-4,
        -3.591779705419e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6  # IDL's poly uses lowest degree first

    if Te.size == 1:
        result = result[0]

    return result
