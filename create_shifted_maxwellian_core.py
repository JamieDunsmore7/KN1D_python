#
# create_shifted_maxwellian_core.py
#
# NOTE: this is the python translation of create_shifted_maxwellian.include in IDL
#

import numpy as np

def create_shifted_maxwellian_core(
    vr, vx, Vx_shift, Tmaxwell,
    vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
    vx_Dvx, vr_Dvr, vr2vx2,
    jpa, jpb, jna, jnb,
    mol, mu, mH, q,
    debug=False):
    
    '''
    INPUTS:
    -----------
    vr - array, vr grid (m s^-1)
    vx - array, vx grid (m s^-1)
    Vx_shift - desired mean velocity in the x-direction of the distribution function (m s^-1)
    Tmaxwell - desired temperature of the distribution function (eV)
    vth - float, Thermal speed corresponding to the normalization temperature Tnorm:
                    vth = sqrt(2*q*Tnorm/(mu*mH)).
    Tnorm : float, Reference temperature (eV) used to compute vth.
    Vr2pidVr : ndarray, Precomputed 2π·vr·Δvr array for integrating over radial velocity.
    dVx : ndarray, Grid spacing Δvx for the  x-velocity dimension.
    vol : ndarray, Combined volume element  = Vr2pidVr[:,None] * dVx[None,:].
    vth_Dvx : ndarray, Padded array of vth·Δvx with leading/trailing zeros, for neighbor‐shift computations.
    vx_Dvx : ndarray, Padded array of vx·Δvx with leading/trailing zeros.
    vr_Dvr : ndarray, Padded array of vr·Δvr with leading/trailing zeros.
    vr2vx2 : ndarray, Precomputed (vr² + vx²) values for computing energy moments.
    jpa, jpb, jna, jnb : int
        Index bounds in the padded vx dimension defining the “interior” region
        where neighbor shifts are applied.
    mol : int, Molecular weight factor: 1 for atomic, 2 for diatomic.
    mu : float, Reduced‐mass ratio parameter (unitless).
    mH : float, Mass of the hydrogen atom (kg).
    q : float, Elementary charge (C).

    KEYWORD ARGUMENTS:
    -----------
    debug : bool, optional
        If True, print diagnostics comparing requested vs. achieved moments.

    Returns
    -------
    Maxwell : ndarray, shape (nvr, nvx, nx)
        The corrected, shifted Maxwellian distribution evaluated on the (vr,vx) grid
        for each of the nx slices.  Each k‐slice integrates (numerically) to unit density,
        has mean axial velocity ≈ Vx_shift[k], and temperature ≈ Tmaxwell[k].


    Notes on Algorithm
    -----------------
    One might think that Maxwell could be simply computed by a direct evaluation of the EXP function:

        for i=0,nvr-1 do begin
        arg=-(vr(i)^2+(vx-Vx_shift/vth)^2) * mol*Tnorm/Tmaxwell
        Maxwell(i,*,k)=exp(arg > (-80))
        endfor

    But owing to the discrete velocity space bins, this method does not necessarily lead to a digital representation 
    of a shifted Maxwellian (Maxwell) that when integrated numerically has the desired vx moment of Vx_shift
    and temperature, Tmaxwell.

    In order to insure that Maxwell has the desired vx and T moments when evaluated numerically, a compensation
    scheme is employed - similar to that used in Interp_fVrVxX.pro
    '''
    
    Terror, Verror = np.nan, np.nan

    nvr, nvx = vr.size, vx.size
    nx = Vx_shift.size

    # some checks
    assert Tmaxwell.shape[0] == nx
    assert Vx_shift.shape[0] == nx
    assert Vr2pidVr.shape[0] == nvr
    assert dVx.shape[0] == nvx
    assert vr2vx2.shape == (nvr, nvx)

    Maxwell = np.zeros((nvr, nvx, nx), dtype=np.float64)

    for k in range(nx):
        if Tmaxwell[k] <= 0.0:
            continue


        for i in range(nvr):
            arg = -( vr[i]**2 + (vx - Vx_shift[k]/vth)**2 ) * (mol*Tnorm/Tmaxwell[k])
            arg = np.minimum(arg, 0.0)
            arg = np.maximum(arg, -80.0)
            Maxwell[i,:,k] = np.exp(arg)

        Maxwell[:,:,k] /= np.sum(Vr2pidVr * np.dot(Maxwell[:,:,k], dVx))

        if debug:
            vx_out1 = vth * np.sum(Vr2pidVr[:, None] * Maxwell[:, :, k] * vx[None, :] * dVx[None, :])
            vr2vx2_ran2 = np.zeros_like(Maxwell[:, :, k])
            for i in range(nvr):
                vr2vx2_ran2[i, :] = vr[i]**2 + (vx - vx_out1 / vth)**2
            T_out1 = (mol * mu * mH * vth*vth * np.sum(Vr2pidVr[:, None] * vr2vx2_ran2 * Maxwell[:, :, k] * dVx[None, :])) / (3 * q)
            vth_local = 0.1 * np.sqrt(2 * Tmaxwell[k] * q / (mol * mu * mH))
            Terror = abs(Tmaxwell[k] - T_out1) / Tmaxwell[k]
            Verror = abs(vx_out1 - Vx_shift[k]) / vth_local


        WxD = Vx_shift[k]
        ED  = WxD**2 + 3*q*Tmaxwell[k]/(mol*mu*mH)
        WxMax = vth*np.sum(Vr2pidVr * np.dot(Maxwell[:,:,k], (vx*dVx)))
        EMax  = vth*vth*np.sum(Vr2pidVr*(np.dot(vr2vx2*Maxwell[:,:,k], dVx)))

        Nij = np.zeros((nvr+2, nvx+2), float)
        Nij[1:nvr+1,1:nvx+1] = Maxwell[:,:,k] * vol

        Nij_vx_Dvx = Nij * vx_Dvx      # shape (nvr+2, nvx+2)
        Nij_vr_Dvr = Nij * vr_Dvr      # shape (nvr+2, nvx+2)

        Nijp1_vx_Dvx = np.concatenate((Nij_vx_Dvx[:,1:], np.zeros((nvr+2,1))), axis=1)
        Nijm1_vx_Dvx = np.concatenate((np.zeros((nvr+2,1)), Nij_vx_Dvx[:,:-1]), axis=1)

        Nip1j_vr_Dvr = np.concatenate((Nij_vr_Dvr[1:,:], np.zeros((1,nvx+2))), axis=0)
        Nim1j_vr_Dvr = np.concatenate((np.zeros((1,nvx+2)), Nij_vr_Dvr[:-1,:]), axis=0)


        AN = np.zeros((nvr, nvx, 2), float)
        BN = np.zeros((nvr, nvx, 2), float)
        sgn = [1, -1]




        _AN = np.roll(Nij * vth_Dvx, 1, axis=1) - (Nij * vth_Dvx)
        AN[:,:,0] = _AN[1:nvr+1, 1:nvx+1]
        _AN = -np.roll(Nij * vth_Dvx, -1, axis=1) + (Nij * vth_Dvx)
        AN[:,:,1] = _AN[1:nvr+1, 1:nvx+1]

        BN[:, jpa+1 : jpb+1, 0] = (Nijm1_vx_Dvx[1:nvr+1, jpa+2 : jpb+2] - Nij_vx_Dvx[1:nvr+1, jpa+2 : jpb+2])
        BN[:, jpa, 0] = - Nij_vx_Dvx[1:nvr+1, jpa+1]
        BN[:, jnb, 0] = Nij_vx_Dvx[1:nvr+1, jnb+1]
        BN[:, jna : jnb, 0] = (-Nijp1_vx_Dvx[1:nvr+1, jna+1 : jnb+1] + Nij_vx_Dvx[1:nvr+1, jna+1 : jnb+1])
        BN[:, :, 0] += (Nim1j_vr_Dvr[1:nvr+1, 1:nvx+1] - Nij_vr_Dvr[1:nvr+1, 1:nvx+1])

        BN[:, jpa+1 : jpb+1, 1] = (-Nijp1_vx_Dvx[1:nvr+1, jpa+2 : jpb+2] + Nij_vx_Dvx[1:nvr+1, jpa+2 : jpb+2])
        BN[:, jpa, 1] = - Nijp1_vx_Dvx[1:nvr+1, jpa+1]
        BN[:, jnb, 1] =   Nijm1_vx_Dvx[1:nvr+1, jnb+1]
        BN[:, jna : jnb, 1] = (Nijm1_vx_Dvx[1:nvr+1, jna+1 : jnb+1] - Nij_vx_Dvx[1:nvr+1, jna+1 : jnb+1])
        BN[1:nvr, :, 1] -= (Nip1j_vr_Dvr[2 : nvr+1, 1 : nvx+1] - Nij_vr_Dvr[2 : nvr+1, 1 : nvx+1])
        BN[0, :, 1] -= Nip1j_vr_Dvr[1, 1 : nvx+1]

        Nij=Nij[1:nvr+1,1:nvx+1]

        TB1 = np.zeros(2, float)
        TB2 = np.zeros(2, float)
        for ia in range(2):
            TA1 = vth * np.sum(AN[:,:,ia] * vx)
            TA2 = vth * vth * np.sum(AN[:,:,ia] * vr2vx2)
            for ib in range(2):
                if TB1[ib] == 0.0:
                    TB1[ib] = vth * np.sum(BN[:,:,ib] * vx)
                if TB2[ib] == 0.0:
                    TB2[ib] = vth*vth * np.sum(BN[:,:,ib] * vr2vx2)
                denom_ab = TA2*TB1[ib] - TA1*TB2[ib]
                a_Max = 0.0
                b_Max = 0.0
                if denom_ab != 0.0 and TA1 != 0.0:
                    b_Max = (TA2*(WxD - WxMax) - TA1*(ED - EMax)) / denom_ab
                    a_Max = (WxD - WxMax - TB1[ib]*b_Max) / TA1

                if a_Max*sgn[ia] > 0.0 and b_Max*sgn[ib] > 0.0:
                    a_Max, b_Max = a_Max, b_Max
                    Maxwell[:,:,k] = (Nij + AN[:,:,ia]*a_Max + BN[:,:,ib]*b_Max) / vol
                    break
            else:
                continue
            break

        else:
            print('No valid a_Max, b_Max found for k={}'.format(k))
            # print('I had this problem before, and it turned out that my translation of the # logic from IDL')
            # print('was leading to sums to zero when in fact they were just very small')
            # print('For example, I had to rewrite this line like below using np.dot to represent the # logic')
            # print('WxMax = vth*np.sum(Vr2pidVr * np.dot(Maxwell[:,:,k], (vx*dVx)))')
            # input("Press <Enter> to continue...")


        integral = np.sum(Vr2pidVr[:, None] * Maxwell[:, :, k] * dVx[None, :])
        Maxwell[:, :, k] /= integral

        if debug:
            vx_out2 = vth * np.sum(Vr2pidVr[:, None] * Maxwell[:, :, k] * vx[None, :] * dVx[None, :])
            vr2vx2_ran2 = np.zeros_like(Maxwell[:, :, k])
            for i in range(nvr):
                vr2vx2_ran2[i, :] = vr[i]**2 + (vx - vx_out2 / vth)**2
            T_out2 = (mol * mu * mH * vth**2 * np.sum(Vr2pidVr[:, None] * vr2vx2_ran2 * Maxwell[:, :, k] * dVx[None, :])) / (3 * q)
            Terror2 = abs(Tmaxwell[k] - T_out2) / Tmaxwell[k]
            Verror2 = abs(Vx_shift[k] - vx_out2) / vth_local
            print(f"CREATE_SHIFTED_MAXWELLIAN => Terror: {Terror:.3e}->{Terror2:.3e}  Verror: {Verror:.3e}->{Verror2:.3e}")

    return Maxwell


