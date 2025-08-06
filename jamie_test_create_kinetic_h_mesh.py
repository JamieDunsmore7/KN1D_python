# Stand‐alone test of create_kinetic_h_mesh.py vs IDL.

import numpy as np
from create_kinetic_h_mesh import Create_Kinetic_H_Mesh

# 1) specified inputs
nv      = 12
mu      = 2.0
nX      = 8
x       = np.arange(nX) * 0.15 
Ti      = 5.0 + 0.2 * np.arange(nX)
Te      = 6.0 + 0.3 * np.arange(nX)
n       = 1e19 * (1 + 0.05 * np.arange(nX))
PipeDia = 0.2 + 0.01 * np.arange(nX)

res = Create_Kinetic_H_Mesh(nv, mu, x, Ti, Te, n, PipeDia, fctr=1.0, E0_in=[0.0])

xH       = res["xH"]
TiH      = res["TiH"]
TeH      = res["TeH"]
neH      = res["neH"]
PipeDiaH = res["PipeDiaH"]
vx       = res["vx"]
vr       = res["vr"]
Tnorm    = res["Tnorm"]
E0        = res["E0"]
ixE0     = res["ixE0"]
irE0     = res["irE0"]

# 3) Print out sizes and first/last entries
print("PYTHON → Create_Kinetic_H_Mesh results:")
print(f" xH  (npts)= {len(xH):2d}  [{xH[0]:.6f}, {xH[-1]:.6f}]")
print(f" TiH (npts)= {len(TiH):2d}  [{TiH[0]:.6f}, {TiH[-1]:.6f}]")
print(f" TeH        =           [{TeH[0]:.6f}, {TeH[-1]:.6f}]")
print(f" neH        = [{neH[0]:.3e}, {neH[-1]:.3e}]")
print(f" PipeDiaH   = [{PipeDiaH[0]:.6f}, {PipeDiaH[-1]:.6f}]")
print(f" vx (len)= {vx.size:2d}, vr (len)= {vr.size:2d}")
print(f" Tnorm      = {Tnorm:.8f}")
print(f" E0         = {E0}, ixE0={ixE0}, irE0={irE0}")
