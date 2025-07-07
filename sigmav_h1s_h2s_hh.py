#
# SigmaV_H1s_H2s_HH.py
# Returns maxwellian averaged <sigma V) for electron impact
# dissociation of molecular hydrogen resulting in two H atoms in
# the 1s state. Coefficients are taken from Janev, 
# "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.259.
#
# Also returns minimum, maximum, and average energy of the resultant H(1s) atoms.
#

import numpy as np

def SigmaV_H1s_H2s_HH(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact dissociation of H₂
    into one H(1s) atom and one H(2s) atom, based on Janev (1987).

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
        Average energy of the H(1s) atoms (eV)
    E0_min : float
        Minimum energy of the H(1s) atoms (eV)
    E0_max : float
        Maximum energy of the H(1s) atoms (eV)
    """

    # Output constants
    E0_ave = 0.3   # eV
    E0_max = 0.55  # eV
    E0_min = 0.0   # eV

    Te = np.atleast_1d(Te).astype(np.float64)

    # Clamp to [0.1, 2e4] eV
    Te = np.clip(Te, 0.1, 2.01e4)

    # Polynomial coefficients (Janev, 1987)
    b = np.array([
        -3.454175591367e+1,
         1.412655911280e+1,
        -6.004466156761e+0,
         1.589476697488e+0,
        -2.775796909649e-1,
         3.152736888124e-2,
        -2.229578042005e-3,
         8.890114963166e-5,
        -1.523912962346e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6  # poly uses highest degree first

    if Te.size == 1:
        result = result[0]

    return result, E0_ave, E0_min, E0_max
