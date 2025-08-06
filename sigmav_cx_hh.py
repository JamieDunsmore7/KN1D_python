# Returns maxwellian averaged <sigma V> for charge exchange of molecular
# hydrogen. Coefficients are taken
# from Janev, "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.292.

# Jamie comments: the neutral has a single energy and we assume the ion to be maxwellian.
# So we are averaging the cross section over the ion velocity distribution.

import numpy as np

def SigmaV_CX_HH(T, E):
    '''
    Input:
        T	- fltarr(*) or float, molecular ion [neutral] temperature (eV)
        E	- fltarr(*) or float, molecular neutral [ion] mono-energy (eV)

    Output:
        returns <sigma V> for 0.1 < Te < 2e4 and 0.1 < E < 2e4 
        units: m^3/s
    '''

    T = np.atleast_1d(T).astype(np.float64) #turns into array if not already
    E = np.atleast_1d(E).astype(np.float64) #turns into array if not already

    if T.shape != E.shape:
        raise ValueError('number of elements of T and E are different!')
    
    _E = np.clip(E, 0.1, 2.01e4) # clamp to [0.1, 2.01e4]
    _T = np.clip(T, 0.1, 2.01e4) # clamp to [0.1, 2.01e4]

    logT = np.log(_T)
    logE = np.log(_E)

    alpha = np.zeros((9, 9))

    # E Index      0			1			2
    block1 = np.array([
       [-2.013143517466e+01,	1.875197914224e-01,	6.865479604288e-02],
       [ 2.643458086299e-01,    -1.177247941077e-01,    -6.758032286178e-03],
       [ 7.295645990688e-02,	6.053418575149e-03,    -1.068656224307e-02],
       [-1.022454343675e-02,     7.350954380641e-03,     6.814213275702e-04],
       [-4.801198168030e-03,    -1.111612877392e-03,     8.373319888351e-04],
       [ 1.141613586234e-03,    -1.371389288760e-04,    -1.733761953296e-04],
       [-3.388853048483e-05,     4.426148343648e-05,     9.992317920676e-06],
       [-6.418225985394e-06,    -3.652063962019e-06,     1.351312819077e-07],
       [ 3.555592819527e-07,     1.012701361110e-07,    -1.993091213299e-08]
    ])

    alpha[0:3,:] = block1.T

    # E Index      3			4			5
    block2 = np.array([
       [ 6.246595384100e-03,    -5.017891372102e-03,    -3.907644829287e-04],
       [ 8.585003721992e-03,    -3.261863407467e-04,    -3.322528542186e-04],
       [-9.371235639464e-04,     9.735708783528e-04,    -9.933049259228e-05],
       [-8.156435157073e-04,     2.903991825737e-05,     3.223596225946e-05],
       [ 1.392977576749e-04,    -9.316910697276e-05,     8.814981236658e-06],
       [ 1.602610140599e-05,     1.464235749797e-05,    -2.944711701791e-06],
       [-5.333970870280e-06,    -2.999105886511e-07,     2.275612517364e-07],
       [ 4.285396408056e-07,    -7.184302986068e-08,    -3.265552364687e-10],
       [-1.131561847140e-08,     3.678869095972e-09,    -3.639982258214e-10]
    ])

    alpha[3:6,:] = block2.T

    # E Index      6			7			8
    block3 = np.array([
       [ 2.786239030986e-04,    -2.942576591004e-05,     9.352275354690e-07],
       [ 6.015471216449e-05,    -4.039435357369e-06,     9.730479674748e-08],
       [-6.786246802840e-06,     1.438327767305e-06,    -5.530742535057e-08],
       [-5.199055182831e-06,     2.852443990256e-07,    -4.825480212106e-09],
       [ 6.675626166047e-07,    -1.325441927019e-07,     5.012529587757e-09],
       [ 6.365231650682e-08,     1.872976659964e-08,    -1.014883015867e-09],
       [-1.173422836715e-08,    -1.364602870139e-09,     9.566404348683e-11],
       [ 1.585228996542e-10,     6.431866226702e-11,    -4.507074278992e-12],
       [ 2.056662091085e-11,    -1.804254277469e-12,     9.042973335167e-14]
    ])

    alpha[6:9,:] = block3.T

    result = np.zeros_like(_E, dtype=_E.dtype)

    for n in range(9):
        for m in range(9):
            result += alpha[n, m] * (logE ** n) * (logT ** m)

    result = np.exp(result) * 1e-6

    if result.size == 1:
        result = result[0]

    return result