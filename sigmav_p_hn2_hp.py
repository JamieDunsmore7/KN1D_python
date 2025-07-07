#
# SigmaV_P_Hn2_HP.py
#
# Returns Maxwellian-averaged <sigma*v> for electron impact
# dissociation of molecular hydrogen ions (H₂⁺), resulting in 
# one proton and one H(n=2) atom.
# Coefficients are taken from Janev,
# "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.260.
#
# Also returns average, minimum, and maximum energy of the proton and H(n=2) atom.
#

import numpy as np

def SigmaV_P_Hn2_HP(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact dissociation
    of H₂⁺ into one proton and one H(n=2) atom, based on Janev (1987).

    Parameters
    ----------
    Te : float or ndarray
        Electron temperature in eV. Values outside [0.1, 2e4] are clipped.

    Returns
    -------
    sigma_v : float or ndarray
        Maxwellian-averaged <σv> in m^3/s

    Additional Outputs
    ------------------
    E0_ave : float
        Average energy of the H(n=2) + proton (eV)
    E0_min : float
        Minimum energy of the H(n=2) + proton (eV)
    E0_max : float
        Maximum energy of the H(n=2) + proton (eV)
    """

    # Output energy constants
    E0_ave = 1.5  # eV
    E0_min = 1.5  # eV
    E0_max = 1.5  # eV

    Te = np.atleast_1d(Te).astype(np.float64)

    # Clamp temperature to valid range
    Te = np.clip(Te, 0.1, 2.01e4)

    # Polynomial coefficients (Janev, 1987)
    b = np.array([
        -3.408905929046e+1,
         1.573560727511e+1,
        -6.992177456733e+0,
         1.852216261706e+0,
        -3.130312806531e-1,
         3.383704123189e-2,
        -2.265770525273e-3,
         8.565603779673e-5,
        -1.398131377085e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6

    if Te.size == 1:
        result = result[0]

    return result, E0_ave, E0_min, E0_max
