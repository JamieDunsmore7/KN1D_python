# Tests and plots the Johnson-Hinnov R coefficients r₀(p) and r₁(p) (m^3/s).

import numpy as np
import matplotlib.pyplot as plt
from jhr_coef import JHR_Coef

# Electron densities (cm^-3)
dens_cm3 = np.array([1e12, 1e13, 1e14])
dens_m3 = dens_cm3 * 1e6

# Electron temperature range (eV)
temp = np.linspace(1, 700, 1000)

# Test for both r0(p) and r1(p), for p = 2, 4, 6
for Ion in [0, 1]:
    for p in [2, 4, 6]:
        plt.figure(figsize=(10, 7))
        for d in dens_m3:
            D_array = np.full_like(temp, d)
            R_vals = JHR_Coef(D_array, temp, Ion, p)
            plt.plot(temp, R_vals, label=f'n = {d/1e6:.0e} cm⁻³')

        ion_label = "r₀(p)" if Ion == 0 else "r₁(p)"
        plt.xlabel('Temperature (eV)')
        plt.ylabel(f'{ion_label} (m³/s)')
        plt.title(f'Johnson-Hinnov {ion_label} for p = {p}')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1, 1000])
        plt.grid(True)
        plt.legend()
        plt.show()
