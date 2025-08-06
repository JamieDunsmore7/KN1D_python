# Returns momentum transfer cross section for elastic collisions of H onto H 
# for specified energy of H. Data are taken from 
# 
# Janev, "Atomic and Molecular Processes in Fusion Edge Plasmas", Chapter 11 - 
# Elastic and Related Cross Sections for Low-Energy Collisions among Hydrogen and 
# Helium Ions, Neutrals, and Isotopes  by D.R. Sdhultz, S. Yu. Ovchinnikov, and S.V.
# Passovets, page 305.
#
#________________________________________________________________________________

import numpy as np

def Sigma_EL_H_H(E,vis=False):
    '''
    Input:
    E	- fltarr(*) or float, energy of H atom (target H atom is at rest)
    
    KEYWORD input:
    VIS	- if set, then return viscosity cross section instead of momentum
    		  transfer cross section
    
    Output:
    Returns Sigma for 0.03 < E < 1e4. For E outside this range, 
        the value of Sigma at the 0.03 or 1e4 eV boundary is returned.
    
    units: m^-2
    '''
    E = np.atleast_1d(E).astype(np.float64) #turns into array if not already
    _E = np.clip(E, 0.03, 1.01e4) # clamp to [0.03, 1.01e4]

    if vis == True:
        a = np.array([
            -3.344860e+01, -4.238982e-01, -7.477873e-02,
            -7.915053e-03, -2.686129e-04
        ])

    else:
        a = np.array([
            -3.330843e+01, -5.738374e-01, -1.028610e-01,
            -3.920980e-03,  5.964135e-04
        ])

    result = np.exp(np.polyval(a[::-1], np.log(_E))) * 1e-4 #np.polyval takes the coefficients in reverse order

    if result.size == 1:
        result = result[0]  # if scalar input, return scalar output

    return result