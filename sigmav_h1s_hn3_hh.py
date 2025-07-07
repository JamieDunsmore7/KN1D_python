#
# SigmaV_H1s_Hn3_HH.py
#
# Returns maxwellian averaged <sigma*v> for electron impact
# dissociation of molecular hydrogen resulting in one H atom in
# the 1s state and one H atom in the n=3 state.
# Coefficients are taken from Janev,
# "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.259.
#
# Also returns minimum, maximum, and average energy of the resultant H(1s), H(n=3) atoms.
#

import numpy as np

def SigmaV_H1s_Hn3_HH(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact dissociation of H₂
    into one H(1s) atom and one H(n=3) atom, based on Janev (1987).

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
        Average energy of the H(1s), H(n=3) atoms (eV)
    E0_min : float
        Minimum energy of the H(1s), H(n=3) atoms (eV)
    E0_max : float
        Maximum energy of the H(1s), H(n=3) atoms (eV)
    """

    # Output constants
    E0_ave = 2.5   # eV
    E0_max = 3.75  # eV
    E0_min = 1.25  # eV

    Te = np.atleast_1d(Te).astype(np.float64)

    # Clamp to valid temperature range
    Te = np.clip(Te, 0.1, 2.01e4)

    # Polynomial coefficients (Janev, 1987)
    b = np.array([
        -3.884976142596e+1,
         1.520368281111e+1,
        -6.078494762845e+0,
         1.535455119900e+0,
        -2.628667482712e-1,
         2.994456451213e-2,
        -2.156175515382e-3,
         8.826547202670e-5,
        -1.558890013181e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6  # convert to m³/s

    if Te.size == 1:
        result = result[0]

    return result, E0_ave, E0_min, E0_max
