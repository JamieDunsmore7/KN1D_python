# Returns maxwellian averaged <sigma*v> for electron impact
# dissociation of molecular hydrogen ions (H₂⁺), resulting in 
# one proton and one H(1s) atom.
# Coefficients are taken from Janev,
# "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.260.
#
# Also returns average, minimum, and maximum energy of the proton and H(1s) atom.
#

import numpy as np

def SigmaV_P_H1s_HP(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact dissociation
    of H₂⁺ into one proton and one H(1s) atom, based on Janev (1987).

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
        Average energy of the H(1s) + proton (eV)
    E0_min : float
        Minimum energy of the H(1s) + proton (eV)
    E0_max : float
        Maximum energy of the H(1s) + proton (eV)
    """

    # Output energy constants
    E0_ave = 4.3  # eV
    E0_min = 4.3  # eV
    E0_max = 4.3  # eV

    Te = np.atleast_1d(Te).astype(np.float64)

    # Clamp temperature to valid range
    Te = np.clip(Te, 0.1, 2.01e4)

    # Polynomial coefficients (Janev, 1987)
    b = np.array([
        -1.781416067709e+1,
         2.277799785711e+0,
        -1.266868411626e+0,
         4.296170447419e-1,
        -9.609908013189e-2,
         1.387958040699e-2,
        -1.231349039470e-3,
         6.042383126281e-5,
        -1.247521040900e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6

    if Te.size == 1:
        result = result[0]

    return result
