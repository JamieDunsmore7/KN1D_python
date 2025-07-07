#
# create_shifted_maxwellian_core.py
#
# This INCLUDE file is used by Kinetic_H2.pro and Kinetic_H.pro
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
    Input:
        Vx_shift  - dblarr(nx), (m s^-1)
        Tmaxwell  - dblarr(nx), (eV)
        Shifted_Maxwellian_Debug - if set, then print debugging information
        mol       - 1=atom, 2=diatomic molecule

    Output:
        Maxwell   - dblarr(nvr,nvx,nx) a shifted Maxwellian distribution function
                having numerically evaluated vx moment close to Vx_shift and
                temperature close to Tmaxwell

    Notes on Algorithm

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

    NOTE: This is a python translate of the IDL script create_shifted_maxwellian.include
    '''
    
    # this allows the second debug loop to run even if we don't enter the first loop
    Terror, Verror = np.nan, np.nan

    nvr, nvx = vr.size, vx.size
    nx = Vx_shift.size

    # some checks
    assert Tmaxwell.shape[0] == nx
    assert Vx_shift.shape[0] == nx
    assert Vr2pidVr.shape[0] == nvr
    assert dVx.shape[0] == nvx
    assert vr2vx2.shape == (nvr, nvx)

    Maxwell = np.zeros((nvr, nvx, nx), float)

    for k in range(nx):
        if Tmaxwell[k] <= 0.0:
            continue

        # 1) raw Maxwellian

        for i in range(nvr):
            arg = -( vr[i]**2 + (vx - Vx_shift[k]/vth)**2 ) * (mol*Tnorm/Tmaxwell[k])
            arg = np.minimum(arg, 0.0)
            arg = np.maximum(arg, -80.0)
            Maxwell[i,:,k] = np.exp(arg)

        denom = np.sum(Vr2pidVr[:,None] * Maxwell[:,:,k] * dVx[None,:])
        Maxwell[:,:,k] /= denom

        if debug:
            vx_out1 = vth * np.sum(Vr2pidVr[:, None] * Maxwell[:, :, k] * vx[None, :] * dVx[None, :])
            vr2vx2_ran2 = np.zeros_like(Maxwell[:, :, k])
            for i in range(nvr):
                vr2vx2_ran2[i, :] = vr[i]**2 + (vx - vx_out1 / vth)**2
            T_out1 = (mol * mu * mH * vth*vth * np.sum(Vr2pidVr[:, None] * vr2vx2_ran2 * Maxwell[:, :, k] * dVx[None, :])) / (3 * q)
            vth_local = 0.1 * np.sqrt(2 * Tmaxwell[k] * q / (mol * mu * mH))
            Terror = abs(Tmaxwell[k] - T_out1) / Tmaxwell[k]
            Verror = abs(vx_out1 - Vx_shift[k]) / vth_local


        # 3) desired moments
        WxD = Vx_shift[k]
        ED  = WxD**2 + 3*q*Tmaxwell[k]/(mol*mu*mH)

        # 4) current moments
        WxMax = vth*np.sum(Vr2pidVr[:,None]*(Maxwell[:,:,k]*(vx[None,:]*dVx[None,:])))
        EMax  = vth*vth*np.sum(Vr2pidVr[:,None]*((vr2vx2*Maxwell[:,:,k])*dVx[None,:])) 
        # vth2 is the variable used in create_shifted_maxwellian.include, but vth2 is defined as vth * vth in create_shifted_maxwellian.pro 

        # 5) padded Nij
        Nij = np.zeros((nvr+2, nvx+2), float)
        Nij[1:nvr+1,1:nvx+1] = Maxwell[:,:,k] * vol

        # 6) neighbor slices (vx-shifted and vr-shifted)


        # 1) pre‐compute the element‐wise products
        Nij_vx_Dvx = Nij * vx_Dvx      # shape (nvr+2, nvx+2)
        Nij_vr_Dvr = Nij * vr_Dvr      # shape (nvr+2, nvx+2)

        Nijp1_vx_Dvx = np.concatenate((Nij_vx_Dvx[:,1:], np.zeros((nvr+2,1))), axis=1)
        Nijm1_vx_Dvx = np.concatenate((np.zeros((nvr+2,1)), Nij_vx_Dvx[:,:-1]), axis=1)

        Nip1j_vr_Dvr = np.concatenate((Nij_vr_Dvr[1:,:], np.zeros((1,nvx+2))), axis=0)
        Nim1j_vr_Dvr = np.concatenate((np.zeros((1,nvx+2)), Nij_vr_Dvr[:-1,:]), axis=0)


        # Allocate AN, BN
        AN = np.zeros((nvr, nvx, 2), float)
        BN = np.zeros((nvr, nvx, 2), float)
        sgn = [1, -1]

        _AN = np.concatenate((np.zeros((nvr+2,1)), Nij * vth_Dvx), axis=1) - (Nij * vth_Dvx)
        AN[:,:,0] = _AN[1:nvr+1, 1:nvx+1]
        _AN = -np.concatenate((np.zeros((nvr+2,1)), Nij * vth_Dvx), axis=1) + (Nij * vth_Dvx)
        AN[:,:,1] = _AN[1:nvr+1, 1:nvx+1]


        BN[:,:,0][ :, jpa+1:jpb+1 ] = (Nijm1_vx_Dvx[1:nvr+1, jpa+2:jpb+2] - Nij_vx_Dvx[1:nvr+1, jpa+2:jpb+2])
        BN[:,:,0][:, jpa] = -Nij_vx_Dvx[1:nvr+1, jpa+1]
        BN[:,:,0][:, jpb] = Nij_vx_Dvx[1:nvr+1, jpb+1]
        BN[:,:,0][:, jna:jnb ] = (-Nijp1_vx_Dvx[1:nvr+1, jna+1:jnb+1] + Nij_vx_Dvx[1:nvr+1, jna+1:jnb+1])
        BN[:,:,0] += (Nim1j_vr_Dvr[1:nvr+1,1:nvx+1] - Nij_vr_Dvr[1:nvr+1,1:nvx+1])

        BN[:,:,1][:, jpa+1:jpb+1] = (-Nijp1_vx_Dvx[1:nvr+1, jpa+2:jpb+2] + Nij_vx_Dvx[1:nvr+1, jpa+2:jpb+2])
        BN[:,:,1][:, jpa] = -Nijp1_vx_Dvx[1:nvr+1, jpa+1]
        BN[:,:,1][:, jnb] = Nijm1_vx_Dvx[1:nvr+1, jnb+1]
        BN[:,:,1][:, jna:jnb ] = (Nijm1_vx_Dvx[1:nvr+1, jna+1:jnb+1] - Nij_vx_Dvx[1:nvr+1, jna+1:jnb+1])
        # subtract vr shifts
        BN[:nvr-1,:,1] -= (Nip1j_vr_Dvr[2:nvr+2,1:nvx+1] - Nij_vr_Dvr[2:nvr+2,1:nvx+1])
        BN[0,:,1] -= Nip1j_vr_Dvr[1,1:nvx+1]

        Nij=Nij[1:nvr+1,1:nvx+1]


        # 6) solve for a_Max, b_Max
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
                else:
                    a_Max = b_Max = 0.0
                if a_Max*sgn[ia] > 0.0 and b_Max*sgn[ib] > 0.0:
                    a_Max, b_Max = a_Max, b_Max
                    Maxwell[:,:,k] = (Nij + AN[:,:,ia]*a_Max + BN[:,:,ib]*b_Max) / vol
                    break
            else:
                continue
            break

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


