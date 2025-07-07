#+
# SigmaV_rec_H1s.py
#
# Returns maxwellian averaged <sigma V) for electron-ion radiative
# recombination to the atomic hydrogen in the 1s state.
# Coefficients are taken from Janev, "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer-Verlag, 1987, p.32.
#

import numpy as np

def SigmaV_rec_H1s(Te):
   '''
   Input:
   Te - fltarr(*) or float, electron temperature (eV)
   
   Output:
   returns <sigma V> for 0.1 < Te < 2e4.
   units: m^3/s
   '''

   Te = np.atleast_1d(Te).astype(np.float64)  # Ensure Te is a float array
   Te = np.clip(Te, 0.1, 2.01e4)  # Clamp to valid range

   # Data for nl = 1s

   n=1
   Ry=13.58
   Eion_n=Ry/n
   Anl=3.92
   Xnl=0.35

   Bn=Eion_n/Te
   result=Anl*1e-14*np.sqrt(Eion_n/Ry)*((Bn**1.5)/(Bn+Xnl))*1e-6
   if Te.size == 1:
       result = result[0]
       
   return result
