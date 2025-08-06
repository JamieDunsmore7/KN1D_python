# Returns charge exchange cross section for molecular hydrogen. Data are taken
# the polynomial fit in
#
#     Janev, "Elementary Processes in Hydrogen-Helium Plasmas", Springer-Verlag, 1987, p.253.
#
#________________________________________________________________________________

import numpy as np

def Sigma_CX_HH(E):
    '''
    Input:
    E	- fltarr(*) or float, energy of molecule corresponding to the
                relative velocity between molecule and molecular ion. (eV)
    Output:
	returns sigma_CX for 0.1 < E < 2e4
	units: m^-2
    '''

    E = np.atleast_1d(E).astype(np.float64) # turns into an array if not already
    _E = np.clip(E, 0.1, 2.01e4)            # clamp to [0.1, 2.01e4]

    alpha = np.array([
        -3.427958758517e+01, -7.121484125189e-02,  4.690466187943e-02,
        -8.033946660540e-03, -2.265090924593e-03, -2.102414848737e-04,
         1.948869487515e-04, -2.208124950005e-05,  7.262446915488e-07
    ])

    result = np.exp(np.polyval(alpha[::-1], np.log(_E))) * 1e-4 #np.polyval takes the coefficients in reverse order

    if E.size == 1:
        result = result[0]  # if scalar input, return scalar output
    
    return result