# Test and plot elastic‐collision cross sections,
# translated from test_sigma_EL_HH.pro

import numpy as np
import matplotlib.pyplot as plt
from sigma_el_p_hh import Sigma_EL_P_HH
from sigma_el_hh_hh import Sigma_EL_HH_HH
from sigma_el_h_hh import Sigma_EL_H_HH

# 1) Energy grid from 0.1 → 1e3 eV (101 points, log‐spaced)
E = np.logspace(np.log10(0.1), np.log10(1e3), 101)

# 2) Evaluate all four cross‐sections (returned in m^2)
sig_P_HH       = Sigma_EL_P_HH(E)
sig_HH_HH      = Sigma_EL_HH_HH(E)          # mt branch
sig_H_HH       = Sigma_EL_H_HH(E)

# 3) Convert to cm^2
sig_P_HH      *= 1e4
sig_HH_HH     *= 1e4
sig_H_HH      *= 1e4

# 3) Set up the figure
fig, ax = plt.subplots(figsize=(10,7))
ax.loglog(E, sig_P_HH,      color='r', linewidth=3, label=r'H$^+$ → H$_2$')
ax.loglog(E, sig_HH_HH,     color='yellow', linewidth=3, label=r'H$_2$ → H$_2$')
ax.loglog(E, sig_H_HH,      color='lime', linewidth=3, label=r'H → H$_2$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 1e3)
ax.set_ylim(1e-19, 1e-14)
ax.set_title('Cross Sections for Elastic Collisions')
ax.set_xlabel('E (eV)')
ax.set_ylabel('Sigma (cm$^{-2}$)')
ax.grid(which='both', linestyle='-', linewidth=0.5)

# 4) Legend
ax.legend(loc='upper right', frameon=False)


plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()
