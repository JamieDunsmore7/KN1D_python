#Make_dVr_dVx.py
#
#   Constructs velocity space differentials for distribution functions
# used by Kinetic_Neutrals.pro, Kinetic_H2.pro, Kinetic_H2.pro, and other 
# related procedures.
#
#   03/25/2004 Bug in computation of Vr2pidVr and VrVr4pidVr found by Jerry Hughes and corrected
          
import numpy as np

def make_dvr_dvx(vr, vx):
    """
    Determine velocity space differentials.
    """
    
    vr = np.atleast_1d(vr).astype(float)
    vx = np.atleast_1d(vx).astype(float)
    nvr = vr.size
    nvx = vx.size

    # — 1) Build extended radial grid _vr by appending one extra edge:
    #      _vr[-1] = 2*vr[-1] - vr[-2]
    _vr = np.concatenate((vr, [2*vr[-1] - vr[-2]]))

    # — 2) Compute mid‐points (vrL, vrR) exactly as IDL:
    #      vr_mid[0] = 0
    #      vr_mid[i+1] = 0.5*(_vr[i] + _vr[i+1]) for i=0..nvr
    _vr_shifted_left = np.concatenate((_vr[1:], [0.0]))
    vr_mid = np.empty(nvr+2, dtype=float)
    vr_mid[0] = 0.0
    vr_mid[1:] = 0.5 * (_vr + _vr_shifted_left)

    vrL = vr_mid[0  : nvr]    # vr_mid[0] .. vr_mid[nvr-1]
    vrR = vr_mid[1  : nvr+1]  # vr_mid[1] .. vr_mid[nvr]


    # — 3) Compute the two radial volume‐weights:
    Vr2pidVr   = np.pi * (vrR**2   - vrL**2)
    VrVr4pidVr = (4.0/3.0) * np.pi * (vrR**3 - vrL**3)

    # — 4) Build extended axial grid _vx (one extra point at each end):
    #      _vx[0] = 2*vx[0] - vx[1],  _vx[-1] = 2*vx[-1] - vx[-2]
    _vx = np.concatenate((
        [2*vx[0] - vx[1]],
        vx,
        [2*vx[-1] - vx[-2]]
    ))

    # — 5) Compute axial left/right edges exactly as IDL:
    _vx_shifted_left = np.concatenate((_vx[1:], [0.0]))
    _vx_shifted_right = np.concatenate(([0.0], _vx[:-1]))

    vxL = 0.5 * (_vx + _vx_shifted_right)
    vxR = 0.5 * (_vx + _vx_shifted_left)

    # then select only the "real" nvx points (IDL uses indices 1..nvx)
    vxL = vxL[1:nvx+1]
    vxR = vxR[1:nvx+1]

    # — 6) dVx is the width of each vx bin:
    dVx = vxR - vxL


    # --- volume elements vol[i,j] = Vr2pidVr[i] * dVx[j] ---
    vol = Vr2pidVr[:, None] * dVx[None, :] # TODO: check that this is correct translation of the idl code

    # --- delta arrays ---
    Deltavx = vxR - vxL
    Deltavr = vrR - vrL

    # CHECK THE FOLLOWING SECTION: I AM NOT SURE THAT IT IS CORRCECT!!

    # padded arrays of shape (nvr+2, nvx+2)
    vth_DeltaVx = np.zeros((nvr+2, nvx+2), float)
    vx_DeltaVx  = np.zeros((nvr+2, nvx+2), float)
    vr_DeltaVr  = np.zeros((nvr+2, nvx+2), float)

    # fill rows i=1..nvr, cols 1..nvx
    vth_DeltaVx[1:nvr+1, 1:nvx+1] = 1.0 / Deltavx[None, :]
    vx_DeltaVx [1:nvr+1, 1:nvx+1] = vx[None, :] / Deltavx[None, :]

    # the fix: broadcast vr/Deltavr down each column
    ratio = vr / Deltavr         # shape (nvr,)
    vr_DeltaVr[1:nvr+1, 1:nvx+1] = ratio[:, None]

    # --- v^2 array ---
    vr2vx2 = vr[:, None]**2 + vx[None, :]**2

    # --- indices of positive/negative vx ---
    pos = np.where(vx > 0)[0]
    jpa = int(pos[0]) if pos.size>0 else None
    jpb = int(pos[-1]) if pos.size>0 else None
    neg = np.where(vx < 0)[0]
    jna = int(neg[0]) if neg.size>0 else None
    jnb = int(neg[-1]) if neg.size>0 else None

    return (
        Vr2pidVr, VrVr4pidVr, dVx,
        vrL, vrR, vxL, vxR,
        vol, vth_DeltaVx, vx_DeltaVx, vr_DeltaVr,
        vr2vx2, jpa, jpb, jna, jnb
    )