# Test and plot Lyman-alpha emissivity by calling Lyman_Alpha.

import numpy as np
import matplotlib.pyplot as plt
from lyman_alpha import Lyman_Alpha

# 1) Density scan at fixed Te=20 eV, N0=3.3e16 m⁻³
densities = 10.0**(16.0 + (21.0 - 16.0) * np.arange(101) / 100.0)
Te_const   = np.full_like(densities, 20.0)
N0_const   = np.full_like(densities, 3.3e16)

# Compute and print
ly_dens = Lyman_Alpha(densities, Te_const, N0_const)
print("=== Density scan (Te=20 eV, N0=3.3e16 m⁻³) ===")
print("densities (m^-3):", densities)
print("Lyman-α emissivity (W/m^3):", ly_dens)

# Plot
plt.figure(figsize=(8,6))
plt.loglog(densities, ly_dens)
plt.title('Lyman-α Emissivity vs Density')
plt.xlabel('Density (m⁻³)')
plt.ylabel('Lyman-α emissivity (W m⁻³)')
plt.grid(True, which='both', ls='--')
plt.xlim([1e16, 1e21])
plt.ylim([10, 1e6])
plt.tight_layout()
plt.show()


# 2) Te scan at fixed density=1e19 m⁻³, N0=3.3e16 m⁻³
Te_vals   = 10.0**( np.log10(0.35) + (np.log10(700.0) - np.log10(0.35)) * np.arange(101) / 100.0 )
dens_const = np.full_like(Te_vals, 1.0e19)
N0_const2 = np.full_like(Te_vals, 3.3e16)

# Compute and print
ly_Te = Lyman_Alpha(dens_const, Te_vals, N0_const2)
print("\n=== Te scan (density=1e19 m⁻³, N0=3.3e16 m⁻³) ===")
print("Te (eV):", Te_vals)
print("Lyman-α emissivity (W/m^3):", ly_Te)

# Plot
plt.figure(figsize=(8,6))
plt.loglog(Te_vals, ly_Te)
plt.title('Lyman-α Emissivity vs Electron Temperature')
plt.xlabel('Te (eV)')
plt.ylabel('Lyman-α emissivity (W m⁻³)')
plt.grid(True, which='both', ls='--')
plt.xlim([0.1, 1e3])
plt.ylim([10, 1e5])
plt.tight_layout()
plt.show()
