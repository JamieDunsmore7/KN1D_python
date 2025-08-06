# Python port of IDL Test_INTERP_FVRVXX.PRO
import numpy as np
import matplotlib.pyplot as plt

from create_vrvxmesh import Create_VrVxMesh
from make_dvr_dvx import make_dvr_dvx
from interp_scalarx import interp_scalarx
from interp_fvrvxx import Interp_fVrVxX

# 1) “a” grid in X and Ti
nx_a = 100
xa_a, xb_a = 0.0, 0.02
x_a = xa_a + (xb_a - xa_a) * np.arange(nx_a) / (nx_a - 1)
Ti_a = 1.0 * np.exp(x_a / 0.025)

# 2) Build the a‐mesh in (vx, vr)
nv_a = 15
vx_a, vr_a, Tnorm_a, *_ = Create_VrVxMesh(nv_a, Ti_a)
nvr_a, nvx_a = vr_a.size, vx_a.size

# 3) Fill fa(vr,vx,x)
fa = np.zeros((nvr_a, nvx_a, nx_a))
Tneut_a = 5.0 + np.arange(nx_a) / (nx_a - 1)
mH, q, mu = 1.6726231e-27, 1.602177e-19, 1.0
Vtha = np.sqrt(2 * q * Tnorm_a / (mu * mH))
Uxa  = 0.5 * Vtha


for j in range(nvx_a):
    for k in range(nx_a):
        exponent = -(vr_a**2 + (vx_a[j] - Uxa/Vtha)**2) / (Tneut_a[k] / Tnorm_a)
        fa[:, j, k] = np.exp(- x_a[k]/0.01) * np.exp(exponent)



# 4) Define the b‐grid
nx_b = 100
xa_b, xb_b = 0.0, 0.05
x_b = xa_b + (xb_b - xa_b) * np.arange(nx_b) / (nx_b - 1)
Ti_b = 10.0 * np.exp(x_b / 0.025)

nv_b = 10
vx_b, vr_b, Tnorm_b, *_ = Create_VrVxMesh(nv_b, Ti_b)
nvr_b, nvx_b = vr_b.size, vx_b.size

_Tneut_a = interp_scalarx(Tneut_a, x_a, x_b)

# 5) Interpolate from fa → fb
fb = Interp_fVrVxX(
    fa, vr_a, vx_a, x_a, Tnorm_a,
    vr_b, vx_b, x_b, Tnorm_b,
    debug=True,
    correct=True
)

# 6) Overlay contours of k=0 slice
levels = 10.0 ** (-np.arange(10))[::-1]

fig, ax = plt.subplots(figsize=(6,6))
# original (fa)
ax.contour(
    np.sqrt(Tnorm_a)*vr_a,
    np.sqrt(Tnorm_a)*vx_a,
    fa[:, :, 0].T,
    levels=levels
)
# interpolated (fb), in a different color
ax.contour(
    np.sqrt(Tnorm_b)*vr_b,
    np.sqrt(Tnorm_b)*vx_b,
    fb[:, :, 0].T,
    levels=levels,
    colors='C3'
)

ax.set_title('Interpolated f(vr,vx) at k=0')
ax.set_xlabel(r'$v_r\,\sqrt{T_{\mathrm{norm}}}$')
ax.set_ylabel(r'$v_x\,\sqrt{T_{\mathrm{norm}}}$')
ax.set_xlim([0, 5])
ax.set_ylim([-6, 6])
plt.tight_layout()
plt.show()
