import numpy as np
from make_dvr_dvx import make_dvr_dvx


# 1) define a simple vr, vx
vr = np.array([0.0, 1.0, 2.0])
vx = np.array([-1.0, 0.0, 1.0])

# 2) call the Python version
(Vr2pidVr, VrVr4pidVr, dVx,
    vrL, vrR, vxL, vxR,
    Vol, Vth_DeltaVx, Vx_DeltaVx, Vr_DeltaVr,
    vr2vx2, jpa, jpb, jna, jnb) = make_dvr_dvx(vr, vx)

# 3) print everything in the same order
print("Vr2pidVr      =", Vr2pidVr)
print("VrVr4pidVr    =", VrVr4pidVr)
print("dVx           =", dVx)
print("vrL, vrR      =", vrL, vrR)
print("vxL, vxR      =", vxL, vxR)
print("Vol           =")
print(Vol)
# pick the central slice [i=1,*] to mimic IDL's Vth_DeltaVx[1,*]
print("Vth_DeltaVx[1,:] =", Vth_DeltaVx[1, :])
print("Vx_DeltaVx[1,:] =", Vx_DeltaVx[1, :])
# mimic IDL Vr_DeltaVr[*,1]
print("Vr_DeltaVr[:,1] =", Vr_DeltaVr[:, 1])
print("vr2vx2        =")
print(vr2vx2)
print("jpa, jpb, jna, jnb =", jpa, jpb, jna, jnb)
