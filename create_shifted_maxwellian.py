# python translation of the IDL `create_shifted_maxwellian.pro`

import numpy as np
from create_shifted_maxwellian_core import create_shifted_maxwellian_core  # Assuming you've saved it in this module
from make_dvr_dvx import make_dvr_dvx

def create_shifted_maxwellian(vr, vx, Tmaxwell, Vx_shift, mu, mol, Tnorm, debug=False):
    """
    Inputs
    ---------
    vr - array, vr grid (m/s)
    vx - array, vx grid (m/s)
    Tmaxwell - array, desired temperature of the distribution function (eV)
    Vx_shift - array, desired mean velocity in the x-direction of the distribution function (m/s)
    mu - float, reduced mass ratio parameter (unitless)
    mol - int, molecular weight factor (1 for atomic, 2 for diatomic)
    Tnorm : float, reference temperature (eV) used to compute thermal speed
    debug : bool, optional
        If True, print diagnostics comparing requested vs. achieved moments.
    
        
    Outputs
    ---------
    Maxwell : ndarray, shape (nvr, nvx, nx)
        The corrected, shifted Maxwellian distribution evaluated on the (vr,vx) grid
        for each of the nx slices. Each k-slice integrates (numerically) to unit density,
        has mean axial velocity ≈ Vx_shift[k], and temperature ≈ Tmaxwell[k].
    """
    mH = 1.6726231e-27  # kg
    q = 1.602177e-19    # C
    
    vth = np.sqrt(2 * q * Tnorm / (mu * mH))

    # Compute required geometric and integration factors
    (Vr2pidVr, VrVr4pidVr, dVx, vrL, vrR, vxL, vxR,
        vol, vth_DeltaVx, vx_DeltaVx, vr_DeltaVr,
        vr2vx2, jpa, jpb, jna, jnb) = make_dvr_dvx(
        vr, vx)

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