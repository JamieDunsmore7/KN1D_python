import numpy as np
from create_shifted_maxwellian import create_shifted_maxwellian
from make_dvr_dvx import make_dvr_dvx
from scipy.interpolate import interp1d

#— physical constants —
mH = 1.6726231e-27    # kg
q  = 1.602177e-19     # C

#— 1) Define some simple test meshes —
nvrM, nvxM = 20, 31
vrM = np.linspace(0.0, 5.0, nvrM)          # 0 → 5 m/s
vxM = np.linspace(-5.0, 5.0, nvxM)         # –5 → +5 m/s

#— 2) Positive-vx indices —
ipM = np.where(vxM > 0.0)[0]

#— 3) Gas parameters for H₂ beam —
GaugeH2    = 0.1
v0_bar     = 1.0e5    # m/s
DensM      = 3.537e19 * GaugeH2
GammaxH2BC = 0.25 * DensM * v0_bar

#— 4) Shifted Maxwellian inputs —
mu        = 2.0       # reduced-mass factor
mol       = 2         # diatomic
Twall     = 2.0e3     # eV
Tmaxwell  = np.array([Twall])
vx_shift  = np.array([0.0])
TnormM    = Tmaxwell.mean()

#— 5) Build and call —
Maxwell = create_shifted_maxwellian(
    vr=vrM, vx=vxM,
    Tmaxwell=Tmaxwell,
    Vx_shift=vx_shift,
    mu=mu, mol=mol,
    Tnorm=TnormM,
    debug=False
)  # shape (nvrM, nvxM, 1)

#— 6) Beam slice —
fH2BC = np.zeros((nvrM, nvxM), float)
fH2BC[:, ipM] = Maxwell[:, ipM, 0]

#— 7) Toy plasma → NuLoss (unchanged) —
nX = 50
x  = np.linspace(0.0, 0.1*(nX-1), nX)
Ti = 1.0 + 0.1 * np.arange(nX)
Te = 2.0 + 0.1 * np.arange(nX)
LC = 0.2 + 0.01 * np.arange(nX)
xH2 = x[x <= 2.0]

Cs_LC = np.zeros(nX)
mask = LC > 0.0
Cs_LC[mask] = np.sqrt(q*(Ti[mask] + Te[mask])/(mu*mH)) / LC[mask]

interp_Cs = interp1d(x, Cs_LC, kind='linear', bounds_error=False, fill_value='extrapolate')
NuLoss = interp_Cs(xH2)

#— 8) Extra moments of the shifted-Maxwellian —
#    get the velocity-space weights
(Vr2pidVr, VrVr4pidVr, dVx, vrL, vrR, vxL, vxR,
    Vol, Vth_DVx, Vx_DVx, Vr_DVr, vr2vx2_2d,
    jpa, jpb, jna, jnb) = make_dvr_dvx(vrM, vxM)

Vth = np.sqrt(2*q*TnormM/(mu*mH))

dens = np.sum(Vr2pidVr[:,None] * Maxwell[:,:,0] * dVx[None,:])
ux_out = Vth * np.sum(Vr2pidVr[:,None] * Maxwell[:,:,0] * (vxM[None,:]*dVx[None,:]))
T_out = (mol*mu*mH) * Vth**2 * np.sum(
    Vr2pidVr[:,None] * (vr2vx2_2d * Maxwell[:,:,0]) * dVx[None,:]
) / (3.0*q)

#— 9) Print everything —
print('--- Test Results ---')
print(f'DensM            = {DensM:.6e}')
print(f'GammaxH2BC       = {GammaxH2BC:.6e}')
print(f'NuLoss ({len(xH2)} points) =', NuLoss)
print('--- shifted-Maxwellian moments:')
print(f' target  vx shift = {vx_shift[0]:.6e}   actual ux_out = {ux_out:.6e}')
print(f' target Tmaxwell   = {Twall:.6e}   actual T_out  = {T_out:.6e}')
print(f' density norm      = {dens:.6e}')


