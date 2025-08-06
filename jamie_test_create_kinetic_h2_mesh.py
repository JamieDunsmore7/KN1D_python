# Stand‐alone test of create_kinetic_h2_mesh.py

import numpy as np
from scipy.interpolate import interp1d
from create_kinetic_h2_mesh import Create_Kinetic_H2_Mesh

# 1) Specified inputs
nv      = 12
mu      = 2.0
nX      = 8
x       = np.arange(nX) * 0.15         
Ti      = 5.0 + 0.2 * np.arange(nX)
Te      = 6.0 + 0.3 * np.arange(nX)
n       = 1e19 * (1 + 0.05 * np.arange(nX))
PipeDia = 0.2 + 0.01 * np.arange(nX)

res = Create_Kinetic_H2_Mesh(nv, mu, x, Ti, Te, n, PipeDia, fctr=1.0, E0_in=[0.0])

xH2       = res["xH2"]
TiH2      = res["TiH2"]
TeH2      = res["TeH2"]
neH2      = res["neH2"]
PipeDiaH2 = res["PipeDiaH2"]
vx        = res["vx"]
vr        = res["vr"]
Tnorm     = res["Tnorm"]
E0        = res["E0"]
ixE0      = res["ixE0"]
irE0      = res["irE0"]

# 3) Print out sizes and first/last entries
print("PYTHON → Create_Kinetic_H2_Mesh results:")
print(f" xH2  (npts)= {len(xH2):2d}  [{xH2[0]:.6f}, {xH2[-1]:.6f}]")
print(f" TiH2 (npts)= {len(TiH2):2d}  [{TiH2[0]:.6f}, {TiH2[-1]:.6f}]")
print(f" TeH2       =           [{TeH2[0]:.6f}, {TeH2[-1]:.6f}]")
print(f" neH2       = [{neH2[0]:.3e}, {neH2[-1]:.3e}]")
print(f" PipeDiaH2  = [{PipeDiaH2[0]:.6f}, {PipeDiaH2[-1]:.6f}]")
print(f" vx (len)= {vx.size:2d}, vr (len)= {vr.size:2d}")
print(f" Tnorm      = {Tnorm:.8f}")
print(f" E0         = {E0}, ixE0={ixE0}, irE0={irE0}")
