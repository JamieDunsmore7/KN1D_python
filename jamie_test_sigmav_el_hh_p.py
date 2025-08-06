# Test and plot elastic‐collision reaction rates for H₂ on p,
# translated from plot_sigmav_EL_HH_P.pro

import numpy as np
import matplotlib.pyplot as plt
from sigmav_el_hh_p import SigmaV_EL_HH_P

# 1) Energy grid from 0.1 → 2×10^4 eV (101 log‐spaced points)
Emin, Emax = 0.1, 2.0e4
Ti = np.logspace(np.log10(Emin), np.log10(Emax), 101)

# 2) Loop over six monoenergetic beam energies E0 = 10^(i−1) eV
e0s = 10.0 ** (np.arange(6) - 1)
colors = ['red', 'blue', 'lime', 'yellow', 'orange', 'cyan']

# 3) Set up the figure
fig, ax = plt.subplots(figsize=(10,7))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 1e3)
ax.set_ylim(1e-12, 1e-8)
ax.set_title('H$_2$ + p → H$_2$ + p elastic')
ax.set_xlabel('Ti (eV)')
ax.set_ylabel(r'$\langle\sigma v\rangle$ (cm$^3$ s$^{-1}$)')
ax.grid(which='both', linestyle='-', linewidth=0.5)

# 4) Compute and plot each curve
for i, e0 in enumerate(e0s):
    E = np.full_like(Ti, e0)
    sigv = SigmaV_EL_HH_P(Ti, E)  # returns m^3/s
    sigv *= 1e6                    # convert to cm^3/s

    ax.plot(Ti, sigv,
            color=colors[i],
            linewidth=3,
            label=f'E0 = {e0:.0g} eV')

# 5) Add legend
ax.legend(loc='lower right', frameon=False)

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()
