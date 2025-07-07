import numpy as np
from create_shifted_maxwellian_core import create_shifted_maxwellian_core  # Assuming you've saved it in this module
from make_dvr_dvx import make_dVr_dVx

def create_shifted_maxwellian(vr, vx, Tmaxwell, Vx_shift, mu, mol, Tnorm, debug=False):
    """
    Python equivalent of CREATE_SHIFTED_MAXWELLIAN.PRO

    Parameters:
        vr : ndarray (nvr,)
        vx : ndarray (nvx,)
        Tmaxwell : ndarray (nx,)
        Vx_shift : ndarray (nx,)
        mu : float
        mol : int (1 for atom, 2 for diatomic molecule)
        Tnorm : float (eV)
        debug : bool, if True, print diagnostics

    Returns:
        Maxwell : ndarray (nvr, nvx, nx)
    """

    mH = 1.6726231e-27  # kg
    q = 1.602177e-19    # C
    
    vth = np.sqrt(2 * q * Tnorm / (mu * mH))

    # Compute required geometric and integration factors
    (Vr2pidVr, VrVr4pidVr, dVx, vrL, vrR, vxL, vxR,
        vol, vth_DeltaVx, vx_DeltaVx, vr_DeltaVr,
        vr2vx2, jpa, jpb, jna, jnb) = make_dVr_dVx(
        vr=vr, vx=vx, vth=vth
    )

    Maxwell = create_shifted_maxwellian_core(
        vr=vr, vx=vx,
        Vx_shift=Vx_shift, Tmaxwell=Tmaxwell,
        vth=vth, Tnorm=Tnorm,
        Vr2pidVr=Vr2pidVr,
        dVx=dVx,
        vol=vol,
        vth_Dvx=vth_DeltaVx,
        vx_Dvx=vx_DeltaVx,
        vr_Dvr=vr_DeltaVr,
        vr2vx2=vr2vx2,
        jpa=jpa,
        jpb=jpb,
        jna=jna,
        jnb=jnb,
        mol=mol, mu=mu, mH=mH, q=q,
        debug=debug
    )

    return Maxwell