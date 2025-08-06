# Test and plot Johnson–Hinnov recombination α coefficients using JHAlpha_Coef.

import numpy as np
import matplotlib.pyplot as plt
from jhalpha_coef import JHAlpha_Coef

# 1) Define densities in cm⁻³ and convert to m⁻³
dens_cm3 = np.array([1e12, 1e13, 1e14])
dens_m3  = dens_cm3 * 1e6

# 2) Temperature range
temp = np.linspace(1, 700, 1000)
colours = ['red', 'green', 'blue']

# 3) Compute, print, and plot
plt.figure(figsize=(10, 7))
for i, d in enumerate(dens_m3):
    D_array = np.full_like(temp, d)
    alpha_vals = JHAlpha_Coef(D_array, temp, no_null=True)
    print(f'density: {d/1e6:.0e} cm⁻³')
    print('Alpha_vals (m^3/s):', alpha_vals)

    # convert to cm^3/s for plotting
    plt.plot(temp, alpha_vals * 1e6, label=f'n = {d/1e6:.0e} cm⁻³', color=colours[i])

# 4) Finalize plot
plt.xlabel('Temperature (eV)')
plt.ylabel('α (cm³/s)')
plt.xscale('log')
plt.yscale('log')
plt.xlim([0.1, 1000])
plt.ylim([1e-16, 1e-11])
plt.grid(True)
plt.legend()
plt.title('JH Coefficient: Recombination')
plt.show()
