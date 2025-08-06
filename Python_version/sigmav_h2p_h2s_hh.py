# Returns maxwellian averaged <sigma*v> for electron impact
# dissociation of molecular hydrogen resulting in one H atom in
# the 2p state and one H atom in the 2s state.
# Coefficients are taken from Janev,
# "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.259.
#
# Also returns minimum, maximum, and average energy of the resultant H(2p), H(2s) atoms.
#

import numpy as np

def SigmaV_H2p_H2s_HH(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact dissociation of H₂
    into one H(2p) atom and one H(2s) atom, based on Janev (1987).

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
        Average energy of the H(2p), H(2s) atoms (eV)
    E0_min : float
        Minimum energy of the H(2p), H(2s) atoms (eV)
    E0_max : float
        Maximum energy of the H(2p), H(2s) atoms (eV)
    """

    # Output constants
    E0_ave = 4.85  # eV
    E0_max = 5.85  # eV
    E0_min = 2.85  # eV

    Te = np.atleast_1d(Te).astype(np.float64)

    # Clamp to valid temperature range
    Te = np.clip(Te, 0.1, 2.01e4)

    # Polynomial coefficients (Janev, 1987)
    b = np.array([
        -4.794288960529e+1,
         2.629649351119e+1,
        -1.151117702256e+1,
         2.991954880790e+0,
        -4.949305181578e-1,
         5.236320848415e-2,
        -3.433774290547e-3,
         1.272097387363e-4,
        -2.036079507592e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6

    if Te.size == 1:
        result = result[0]

    return result
