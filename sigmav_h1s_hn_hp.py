#
# SigmaV_H1s_Hn_HP.py
#
# Returns maxwellian averaged <sigma V) for electron impact
# dissociative recombination of molecular hydrogen ions resulting in 
# one H atom in the 1s state and one H atom in state n > or = 2. Coefficients 
# are taken from Janev, "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.260.
#

import numpy as np

def SigmaV_H1s_Hn_HP(Te):
    """
    Computes the Maxwellian-averaged <sigma*v> for dissociative recombination
    of H₂⁺ ions, producing one H(1s) atom and one H(n≥2) atom.
    Coefficients are from Janev (1987), p.260.

    Parameters
    ----------
    Te : float or ndarray
        Electron temperature in eV.

    Returns
    -------
    sigma_v : float or ndarray
        Maxwellian-averaged <σv> in m^3/s for 0.1 < Te < 2e4

    """

    Te = np.atleast_1d(Te).astype(np.float64)

    # Clamp to [0.1, 2e4] eV
    Te = np.clip(Te, 0.1, 2.01e4)
    # Polynomial coefficients (Janev, 1987)
    b = np.array([
        -1.670435653561e+1,
        -6.035644995682e-1,
        -1.942745783445e-8,
        -2.005952284492e-7,
         2.962996104431e-8,
         2.134293274971e-8,
        -6.353973401838e-9,
         6.152557460831e-10,
        -2.025361858319e-11
    ])

    logTe = np.log(Te)
    result = np.exp(np.polyval(b[::-1], logTe)) * 1e-6  # poly uses highest degree first

    if Te.size == 1:
        result = result[0]

    return result