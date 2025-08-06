# Returns momentum transfer cross section for elastic collisions of H+ onto H 
# for specified energy of H+. Data are taken from 
# 
# Janev, "Atomic and Molecular Processes in Fusion Edge Plasmas", Chapter 11 - 
# Elastic and Related Cross Sections for Low-Energy Collisions among Hydrogen and 
# Helium Ions, Neutrals, and Isotopes  by D.R. Sdhultz, S. Yu. Ovchinnikov, and S.V.
# Passovets, page 298.

import numpy as np

def Sigma_EL_P_H(E):
    '''
    Input:
    E	- fltarr(*) or float, energy of H+ ion (target H atom is at rest)
    
    Output:
    Returns Sigma for 0.001 < E < 1e5. For E outside this range, 
       the value of Sigma at the 0.001 or 1e5 eV boundary is returned.
       
    units: m^-2
    '''

    E = np.atleast_1d(E).astype(np.float64) #turns into array if not already

    _E = np.clip(E, 0.001, 1.01e5) # clamp to [0.001, 1.01e5]

    # Use a different fit depending on whether the energy is higher or lower than 10eV

    result = np.zeros_like(_E) # initialize result with zeros

    mask_low = _E <= 10.0

    if np.any(mask_low):
        a = np.array([
            -3.233966e1, -1.126918e-1, 5.287706e-3,
            -2.445017e-3, -1.044156e-3, 8.419691e-5,
            3.824773e-5
        ])
        result[mask_low] = np.exp(np.polyval(a[::-1], np.log(_E[mask_low]))) * 1e-4

    mask_high = _E > 10.0

    if np.any(mask_high):
        a = np.array([
            -3.231141e1, -1.386002e-1
        ])
        result[mask_high] = np.exp(np.polyval(a[::-1], np.log(_E[mask_high]))) * 1e-4

    if result.size == 1:
        result = result[0]
    
    return result