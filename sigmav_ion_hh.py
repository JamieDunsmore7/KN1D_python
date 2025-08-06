# Returns maxwellian averaged <sigma*v> for electron impact
# ionization of molecular hydrogen (H₂).
# Coefficients are taken from Janev,
# "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.259.
#

import numpy as np

def SigmaV_Ion_HH(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact ionization of H₂
    based on Janev (1987).

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

    # Clamp to valid temperature range
    Te = np.clip(Te, 0.1, 2.01e4)

    # Polynomial coefficients (Janev, 1987)
    b = np.array([
        -3.568640293666e+1,
         1.733468989961e+1,
        -7.767469363538e+0,
         2.211579405415e+0,
        -4.169840174384e-1,
         5.088289820867e-2,
        -3.832737518325e-3,
         1.612863120371e-4,
        -2.893391904431e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6

    if Te.size == 1:
        result = result[0]

    return result
