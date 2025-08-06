# Returns momentum transfer cross section for elastic collisions of H+ onto H2 
# for specified energy of H+. Data are taken from 
# 
# Janev, "Atomic and Molecular Processes in Fusion Edge Plasmas", Chapter 11 - 
# Elastic and Related Cross Sections for Low-Energy Collisions among Hydrogen and 
# Helium Ions, Neutrals, and Isotopes  by D.R. Sdhultz, S. Yu. Ovchinnikov, and S.V.
# Passovets, page 305.

import numpy as np

def Sigma_EL_P_HH(E):
    '''
    Input:
    E	- fltarr(*) or float, energy of H+ ion (target H2 molecule is at rest)

    Output:
    Returns Sigma for 0.03 < E < 1e4. For E outside this range, 
       the value of Sigma at the 0.03 or 1e4 eV boundary is returned.

    units: m^-2
    '''

    E = np.atleast_1d(E).astype(np.float64) #turns into array if not already
    _E = np.clip(E, 0.03, 1.01e4) # clamp to [0.03, 1.01e4]
    a = np.array([
        -3.355719e1, -5.696568e-1, -4.089556e-2,
        -1.143513e-2, 5.926596e-4
    ])
    result = np.exp(np.polyval(a[::-1], np.log(_E))) * 1e-4 #np.polyval takes the coefficients in reverse order

    if result.size == 1:
        result = result[0]
    
    return result


