# Returns charge exchange cross section for atomic hydrogen. Data are taken
# either from the polynomial fit in
#
#     Janev, "Elementary Processes in Hydrogen-Helium Plasmas", Springer-Verlag, 1987, p.250.
#
# from Freeman and Jone's analytic fit tabulated in 
#
#     Freeman, E.L., Jones, E.M., "Atomic Collision Processes in Plasma Physics
#     Experiments", UKAEA Report No. CLM-R137 (Culham Laboratory, Abington, England 1974)
#
#________________________________________________________________________________

import numpy as np

def Sigma_CX_H0(E,freeman=False):
    '''
    Input:
        E	- fltarr(*) or float, energy of proton corresponding to the
                    relative velocity between proton and hydrogen atom. (eV)

    Keywords:

        Freeman - if set, then return CX based on Freeman and Jones' analytic fit in
            Freeman, E.L., Jones, E.M., "Atomic Collision Processes in Plasma Physics
                    Experiments", UKAEA Report No. CLM-R137 (Culham Laboratory, Abington, England 1974)

                    otherwise return CX based on polynomial fit in
            Janev, "Elementary Processes in Hydrogen-Helium Plasmas", 
                Springer-Verlag, 1987, p.250, other
                    
                    Default is Freeman=0

    Output:
        returns sigma_CX for 0.1 < E < 2e4
        units: m^-2
    '''
    E = np.atleast_1d(E).astype(np.float64) #turns into array if not already
    
    if freeman == True:
        _E = np.clip(E, 0.1, 1.0e5)

        result = (
            1e-4 * 0.6937e-14 *
            (1.0 - 0.155 * np.log10(_E)) ** 2 /
            (1.0 + 0.1112e-14 * _E ** 3.3)
        )

    else:
        _E = np.clip(E, 0.1, 2.01e4)
        alpha = np.array([
            -3.274123792568e+01, -8.916456579806e-02, -3.016990732025e-02,
            9.205482406462e-03,  2.400266568315e-03, -1.927122311323e-03,
            3.654750340106e-04, -2.788866460622e-05,  7.422296363524e-07
        ])
        result = np.exp(np.polyval(alpha[::-1], np.log(_E))) * 1e-4 #np.polyval takes the coefficients in reverse order

    if result.size == 1:
        result = result[0]

    return result


