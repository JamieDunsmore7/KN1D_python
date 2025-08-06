# Returns maxwellian averaged <sigma*v> for electron impact
# ionization of atomic hydrogen.
# Coefficients are taken from Janev,
# "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.258.
#

import numpy as np

def SigmaV_Ion_H0(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for electron impact ionization of H₀
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
        -3.271396786375e+1,
         1.353655609057e+1,
        -5.739328757388e+0,
         1.563154982022e+0,
        -2.877056004391e-1,
         3.482559773737e-2,
        -2.631976175590e-3,
         1.119543953861e-4,
        -2.039149852002e-6
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6

    if Te.size == 1:
        result = result[0]

    return result
