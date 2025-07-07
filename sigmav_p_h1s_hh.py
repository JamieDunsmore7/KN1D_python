#
# SigmaV_P_H1s_HH.py
#
# Returns maxwellian averaged <sigma*v> for electron impact
# ionization and dissociation of molecular hydrogen (H₂),
# resulting in one proton and one H(1s) atom.
# Coefficients are taken from Janev,
# "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.260.
#

import numpy as np

def SigmaV_P_H1s_HH(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact ionization +
    dissociation of H₂ resulting in one proton and one H(1s), based on Janev (1987).

    Parameters
    ----------
    Te : float or ndarray
        Electron temperature in eV. Values outside [0.1, 2e4] are clipped.

    Returns
    -------
    sigma_v : float or ndarray
        Maxwellian-averaged <σv> in m^3/s
    """

    Te = np.atleast_1d(Te).astype(np.float64)

    # Clamp to valid range
    Te = np.clip(Te, 0.1, 2.01e4)

    # Polynomial coefficients (Janev, 1987)
    b = np.array([
        -3.834597006782e+1,
         1.426322356722e+1,
        -5.826468569506e+0,
         1.727940947913e+0,
        -3.598120866343e-1,
         4.822199350494e-2,
        -3.909402993006e-3,
         1.738776657690e-4,
        -3.252844486351e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6

    if Te.size == 1:
        result = result[0]

    return result
