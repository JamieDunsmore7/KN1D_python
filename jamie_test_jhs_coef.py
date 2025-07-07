#
# test_jhs.py
#
# Test and plot Johnson-Hinnov S coefficients by calling JHS_Coef.
#

import numpy as np
import matplotlib.pyplot as plt
from jhs_coef import JHS_Coef

# Define densities in cm⁻³
dens_cm3 = np.array([1e12, 1e13, 1e14])
dens_m3 = dens_cm3 * 1e6  # Convert to m⁻³

# Temperature range in eV
temp = np.linspace(1, 700, 1000)
colours = ['red', 'green', 'blue']
# Evaluate for each density
plt.figure(figsize=(10, 7))
for i, d in enumerate(dens_m3):
    D_array = np.full_like(temp, d)
    S_vals = JHS_Coef(D_array, temp)
    print('density:', d)
    print('S_vals:', S_vals)
    plt.plot(temp, S_vals * 1e6, label=f'n = {d/1e6:.0e} cm⁻³', color=colours[i])

plt.xlabel('Temperature (eV)')
plt.ylabel('S (cm³/s)')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1, 1000])
plt.ylim([1e-11, 1e-7])
plt.grid(True)
plt.legend()
plt.title('JHS Coefficient: Ionisation')
plt.show()
