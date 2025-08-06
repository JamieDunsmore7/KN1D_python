# Test and plot elastic‐collision cross sections H⁺→H and H→H,
# translated from test_sigma_EL_H.pro

import numpy as np
import matplotlib.pyplot as plt
from sigma_el_p_h import Sigma_EL_P_H
from sigma_el_h_h import Sigma_EL_H_H

# 1) Energy grid from 0.1 → 1e3 eV (101 log‐spaced points)
E = np.logspace(np.log10(0.1), np.log10(1e3), 101)

# 2) Evaluate cross‐sections (returned in m^2)
sig_P_H     = Sigma_EL_P_H(E)
sig_H_H     = Sigma_EL_H_H(E)
sig_H_H_vis = Sigma_EL_H_H(E, vis=True)

# 3) Convert to cm^2
sig_P_H     *= 1e4
sig_H_H     *= 1e4
sig_H_H_vis *= 1e4

# 4) Plot
fig, ax = plt.subplots(figsize=(10,7))
ax.loglog(E, sig_P_H,      color='red', linewidth=3, label=r'H$^+$ → H (momentum)')
ax.loglog(E, sig_H_H,      color='lime', linewidth=3, label=r'H → H (momentum)')
ax.loglog(E, sig_H_H_vis,  color='blue', linewidth=3, label=r'H → H (viscosity)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 1e3)
ax.set_ylim(1e-19, 3e-14)
ax.set_title('Cross Sections for Elastic Collisions')
ax.set_xlabel('E (eV)')
ax.set_ylabel('Sigma (cm$^{-2}$)')
ax.grid(which='both', linestyle='-', linewidth=0.5)

# 5) Legend
ax.legend(loc='upper right', frameon=False)


plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()
