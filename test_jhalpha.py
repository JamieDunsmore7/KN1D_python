# Tests and plots the Johnson-Hinnov recombination coefficient α (m^3/s).

import numpy as np
import matplotlib.pyplot as plt
from jhalpha_coef import JHAlpha_Coef

# Electron densities (cm^-3)
dens_cm3 = np.array([1e12, 1e13, 1e14])
dens_m3 = dens_cm3 * 1e6  # Convert to m^-3

# Electron temperature range (eV)
temp = np.linspace(1, 700, 1000)

plt.figure(figsize=(10, 7))
for d in dens_m3:
    D_array = np.full_like(temp, d)
    alpha_vals = JHAlpha_Coef(D_array, temp)
    plt.plot(temp, alpha_vals * 1e6, label=f'n = {d/1e6:.0e} cm⁻³')

plt.xlabel('Temperature (eV)')
plt.ylabel('Recombination α (cm³/s)')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1, 1000])
plt.ylim([1e-17, 1e-12])
plt.grid(True)
plt.legend()
plt.title('Johnson-Hinnov Recombination Coefficient α')
plt.show()
