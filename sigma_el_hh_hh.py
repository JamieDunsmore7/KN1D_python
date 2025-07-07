# Sigma_EL_HH_HH.pro
#
#
# Returns momentum transfer cross section for elastic collisions of H2 onto H2 
# for specified energy of H2. Data are taken from 
# 
# Janev, "Atomic and Molecular Processes in Fusion Edge Plasmas", Chapter 11 - 
# Elastic and Related Cross Sections for Low-Energy Collisions among Hydrogen and 
# Helium Ions, Neutrals, and Isotopes  by D.R. Sdhultz, S. Yu. Ovchinnikov, and S.V.
# Passovets, page 305.


import numpy as np

def Sigma_EL_HH_HH(E,vis=False):
    '''
    Input:
    E	- fltarr(*) or float, energy of H atom (target H2 molecule is at rest)
    
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
        print('WARNING in SIGMA_EL_HH_HH => using momentum transfer as viscosity cross-section')
        a = np.array([-3.430345e1, -2.960406e-1, -6.382532e-2,
                      -7.557519e-3, 2.606259e-4])
        
    else:
        a = np.array([-3.430345e1, -2.960406e-1, -6.382532e-2,
                      -7.557519e-3, 2.606259e-4])
        
    result = np.exp(np.polyval(a[::-1], np.log(_E))) * 1e-4 #np.polyval takes the coefficients in reverse order

    if result.size == 1:
        result = result[0]
    
    return result