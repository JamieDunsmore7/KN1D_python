#
# Interp_fVrVxX.py
#
# Interpolates distribution functions used by Kinetic_H.py, Kinetic_H2.py, 
# and other procedures.
#

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from make_dvr_dvx import make_dvr_dvx



def Interp_fVrVxX(
    fa, vra, vxa, xa, Tnorma,
    vrb, vxb, xb, Tnormb, debug=False,
    correct=True
):
    """
    Interpolates a 3D velocity-space distribution function from one grid to another,
    conserving total particle density in velocity space at each spatial location.

    Parameters
    ----------
    fa : ndarray of shape (nVra, nVxa, nXa)
        Input distribution function defined on the original (a) phase space grid.

    vra : ndarray of shape (nVra,)
        Radial velocity grid for input distribution `fa`.

    vxa : ndarray of shape (nVxa,)
        Axial velocity grid for input distribution `fa`.

    xa : ndarray of shape (nXa,)
        Spatial coordinate grid for input distribution `fa`.

    Tnorma : float
        Normalization temperature associated with `fa`. Used to compute thermal velocity
        for scaling between grids.

    vrb : ndarray of shape (nVrb,)
        Target radial velocity grid for output distribution `fb`.

    vxb : ndarray of shape (nVxb,)
        Target axial velocity grid for output distribution `fb`.

    xb : ndarray of shape (nXb,)
        Target spatial coordinate grid for output distribution `fb`.

    Tnormb : float
        Normalization temperature associated with the target velocity grid. Used to compute
        thermal velocity scaling.

    make_dvr_dvx_fn : function
        A function that computes the velocity-space volume elements and bin widths.
        Expected to return: (Vr2pidVr, VrVr4pidVr, dVx, ...)

    fb_out : ndarray of shape (nVrb, nVxb, nXb), optional
        If provided, output will be stored in this array. Otherwise, a new array is allocated.

    debug : bool, optional
        If True, prints internal diagnostic information (not implemented in current version).

    Returns
    -------
    fb : ndarray of shape (nVrb, nVxb, nXb)
        Interpolated distribution function. Values outside the bounds of the original
        phase space are set to zero. At each spatial point, `fb` is scaled such that its
        velocity-space integral matches that of `fa`.
    """


    nvra, nvxa, nxa = len(vra), len(vxa), len(xa)
    nvrb, nvxb, nxb = len(vrb), len(vxb), len(xb)

    if fa.shape != (nvra, nvxa, nxa):
        raise ValueError("Shape of fa does not match (nvra, nvxa, nxa)")

    mH = 1.6726231e-27  # hydrogen mass in kg
    q = 1.602177e-19    # elementary charge in C
    mu = 1

    # Compute Vtha, Vthb
    vtha = np.sqrt(2 * q * Tnorma / (mu * mH))
    vthb = np.sqrt(2 * q * Tnormb / (mu * mH))
    fV = vthb / vtha

    # Rescale b-grid into a-grid units
    vrb_rescaled = fV * vrb
    vxb_rescaled = fV * vxb

    if len(fa[:, 0, 0]) != len(vra):
        raise ValueError("Number of elements in fa(:,0,0) and Vra do not agree!")
    if len(fa[0, :, 0]) != len(vxa):
        raise ValueError("Number of elements in fa(0,:,0) and Vxa do not agree!")
    if len(fa[0, 0, :]) != len(xa):
        raise ValueError("Number of elements in fa(0,0,:) and Xa do not agree!")
    



    # find all indices of vrb_rescaled within [vra.min(), vra.max()]
    ok_i = np.where((vrb_rescaled >= vra.min()) & (vrb_rescaled <= vra.max()))[0]
    if ok_i.size < 1:
        raise ValueError("No values of Vrb are within range of Vra")
    i0, i1 = int(ok_i[0]), int(ok_i[-1])

    # same for vxb_rescaled vs. vxa
    ok_j = np.where((vxb_rescaled >= vxa.min()) & (vxb_rescaled <= vxa.max()))[0]
    if ok_j.size < 1:
        raise ValueError("No values of Vxb are within range of Vxa")
    j0, j1 = int(ok_j[0]), int(ok_j[-1])

    # and for the spatial grid xb vs xa (no fV scaling here)
    ok_k = np.where((xb >= xa.min()) & (xb <= xa.max()))[0]
    if ok_k.size < 1:
        raise ValueError("No values of Xb are within range of Xa")
    k0, k1 = int(ok_k[0]), int(ok_k[-1])

    # Then we can still check for a minimal amount of overlap:
    if (i1 - i0) < 1 or (j1 - j0) < 1 or (k1 - k0) < 1:
        raise ValueError("Not enough overlap between source and target grids")


    fb = np.zeros((nvrb, nvxb, nxb), dtype=float)

    Vr2pidVra, VrVr4pidVra, dVxa, vraL, vraR, vxaL, vxaR, _, _, _, _, Vra2Vxa2, _, _, _, _ = make_dvr_dvx(vra, vxa)
    Vr2pidVrb, VrVr4pidVrb, dVxb, vrbL, vrbR, vxbL, vxbR, Vol, Vth_DVx, Vx_DVx, Vr_DVr, Vrb2Vxb2, jpa, jpb, jna, jnb = make_dvr_dvx(vrb, vxb)

    # Assume weight must be recomputed (full caching logic omitted for now)
    weight = None
    w_new = True
    if debug:
        print("INTERP_FVRVXX => computing new Weight")

    # Skip all the stuff from the IDL about checking for pre-computed weights.
    # Doesn't change anything, may just speed things up a bit.

    # Set area contributions to Weight array
    _weight = np.zeros((nvrb, nvxb, nvra, nvxa))
    weight = np.zeros((nvrb * nvxb, nvra * nvxa))

    for ib in range(nvrb):
        for jb in range(nvxb):
            for ia in range(nvra):
                vra_min = max(fV * vrbL[ib], vraL[ia])
                vra_max = min(fV * vrbR[ib], vraR[ia])
                for ja in range(nvxa):
                    vxa_min = max(fV * vxbL[jb], vxaL[ja])
                    vxa_max = min(fV * vxbR[jb], vxaR[ja])
                    if vra_max > vra_min and vxa_max > vxa_min:
                        _weight[ib, jb, ia, ja] = (
                            2 * np.pi
                            * (vra_max**2 - vra_min**2)
                            * (vxa_max - vxa_min)
                            / (Vr2pidVrb[ib] * dVxb[jb])
                        )

    weight[:] = _weight.reshape(nvrb * nvxb, nvra * nvxa, order= 'F')

    # --- Interpolate in X: from source fa -> intermediate fb_xa ---
    _fa = fa.reshape(nvra * nvxa, nxa, order = 'F')  # Flatten velocity dimensions for matrix multiplication
    fb_xa = weight @ _fa  # fb_xa has shape (nvrb * nvxb, nxa)

    # Compute _Wxa and _Ea - these are the desired moments of fb, but on the xa grid
    na = np.zeros(nxa)
    _Wxa = np.zeros(nxa)
    _Ea = np.zeros(nxa)
    for k in range(nxa):
        fa_k = fa[:, :, k]
        na[k] = np.sum(Vr2pidVra[:, None] * fa_k * dVxa[None, :])
        if na[k] > 0:
            _Wxa[k] = (
                np.sqrt(Tnorma)
                * np.sum(Vr2pidVra[:, None] * fa_k * (vxa[None, :] * dVxa[None, :]))
                / na[k]
            )

            # fa_k has shape (nvr, nvx)
            integrand = Vr2pidVra[:,None] * (Vra2Vxa2 * fa_k) * dVxa[None,:]
            _Ea[k] = Tnorma * integrand.sum() / na[k]

    # Interpolate in x to get fb from fb_xa and to get Wxa, Ea from _Wva, _Ea
    Wxa = np.zeros(nxb)
    Ea = np.zeros(nxb)

    for k in range(k0, k1 + 1):
        # Locate interval in xa for xb[k]
        kL = np.searchsorted(xa, xb[k]) - 1
        kL = np.clip(kL, 0, len(xa) - 2)
        kR = kL + 1
        f = (xb[k] - xa[kL]) / (xa[kR] - xa[kL])
        fb[:, :, k] = (
            fb_xa[:, kL].reshape(nvrb, nvxb, order='F')
            + f * (fb_xa[:, kR].reshape(nvrb, nvxb, order='F') - fb_xa[:, kL].reshape(nvrb, nvxb, order='F'))
        )
        Wxa[k] = _Wxa[kL] + f * (_Wxa[kR] - _Wxa[kL])
        Ea[k] = _Ea[kL] + f * (_Ea[kR] - _Ea[kL])

    # Correct fb so that it has the same Wx and E moments as fa
    if correct == True:
        AN = np.zeros((nvrb, nvxb, 2))
        BN = np.zeros((nvrb, nvxb, 2))
        sgn = [1, -1]

        for k in range(nxb):
            allow_neg = False

            nb = np.sum(Vr2pidVrb[:, None] * fb[:, :, k] * dVxb[None, :])
            if nb > 0:
                while True:
                    nb = np.sum(Vr2pidVrb[:, None] * fb[:, :, k] * dVxb[None, :])
                    Wxb = np.sqrt(Tnormb) * np.sum(Vr2pidVrb[:, None] * fb[:, :, k] * (vxb[None, :] * dVxb[None, :])) / nb
                    Eb = Tnormb * np.sum(Vr2pidVrb[:, None] * Vrb2Vxb2 * fb[:, :, k] * dVxb[None, :]) / nb

                    Nij = np.zeros((nvrb + 2, nvxb + 2))
                    Nij[1:nvrb + 1, 1:nvxb + 1] = fb[:, :, k] * Vol / nb

                    cutoff = 1.0e-6 * np.max(Nij)
                    mask = (np.abs(Nij) < cutoff) & (np.abs(Nij) > 0)
                    Nij[mask] = 0.0

                    if np.all(Nij[2, :] <= 0.0):
                        allow_neg = True

                    Nijp1_vx_Dvx = np.roll(Nij * Vx_DVx, -1, axis=1)
                    Nij_vx_Dvx = Nij * Vx_DVx
                    Nijm1_vx_Dvx = np.roll(Nij * Vx_DVx, 1, axis=1)

                    Nip1j_vr_Dvr = np.roll(Nij * Vr_DVr, -1, axis=0)
                    Nij_vr_Dvr = Nij * Vr_DVr
                    Nim1j_vr_Dvr = np.roll(Nij * Vr_DVr, 1, axis=0)

                    _AN = np.roll(Nij * Vth_DVx, 1, axis=1) - Nij * Vth_DVx
                    AN[:, :, 0] = _AN[1:nvrb + 1, 1:nvxb + 1]
                    _AN = -np.roll(Nij * Vth_DVx, -1, axis=1) + Nij * Vth_DVx
                    AN[:, :, 1] = _AN[1:nvrb + 1, 1:nvxb + 1]

                    BN[:, jpa+1:jpb+1, 0] = Nijm1_vx_Dvx[1:nvrb + 1, jpa+2:jpb+2] - Nij_vx_Dvx[1:nvrb + 1, jpa+2:jpb+2]
                    BN[:, jpa, 0] = -Nij_vx_Dvx[1:nvrb + 1, jpa+1]
                    BN[:, jnb, 0] = Nij_vx_Dvx[1:nvrb + 1, jnb+1]
                    BN[:, jna:jnb, 0] = -Nijp1_vx_Dvx[1:nvrb + 1, jna+1:jnb+1] + Nij_vx_Dvx[1:nvrb + 1, jna+1:jnb+1]
                    BN[:, :, 0] += Nim1j_vr_Dvr[1:nvrb + 1, 1:nvxb + 1] - Nij_vr_Dvr[1:nvrb + 1, 1:nvxb + 1]

                    BN[:, jpa+1:jpb+1, 1] = -Nijp1_vx_Dvx[1:nvrb + 1, jpa+2:jpb+2] + Nij_vx_Dvx[1:nvrb + 1, jpa+2:jpb+2]
                    BN[:, jpa, 1] = -Nijp1_vx_Dvx[1:nvrb + 1, jpa+1]
                    BN[:, jnb, 1] = Nijm1_vx_Dvx[1:nvrb + 1, jnb+1]
                    BN[:, jna:jnb, 1] = Nijm1_vx_Dvx[1:nvrb + 1, jna+1:jnb+1] - Nij_vx_Dvx[1:nvrb + 1, jna+1:jnb+1]
                    BN[1:nvrb, :, 1] -= Nip1j_vr_Dvr[2:nvrb + 1, 1:nvxb + 1] - Nij_vr_Dvr[2:nvrb + 1, 1:nvxb + 1]
                    BN[0, :, 1] -= Nip1j_vr_Dvr[1, 1:nvxb + 1]

                    if allow_neg:
                        BN[0, :, 1] -= Nij_vr_Dvr[1, 1:nvxb + 1]
                        BN[1, :, 1] += Nij_vr_Dvr[1, 1:nvxb + 1]

                    Nij = Nij[1:nvrb + 1, 1:nvxb + 1]

                    TB1 = np.zeros(2)
                    TB2 = np.zeros(2)


                    for ia in range(2):
                        TA1 = np.sqrt(Tnormb) * np.sum(AN[:, :, ia] * vxb[None, :])
                        TA2 = Tnormb * np.sum(Vrb2Vxb2 * AN[:, :, ia])
                        for ib in range(2):
                            if TB1[ib] == 0.0:
                                TB1[ib] = np.sqrt(Tnormb) * np.sum(BN[:, :, ib] * vxb[None, :])
                            if TB2[ib] == 0.0:
                                TB2[ib] = Tnormb * np.sum(Vrb2Vxb2 * BN[:, :, ib])

                            denom = TA2 * TB1[ib] - TA1 * TB2[ib]
                            alpha = beta = 0.0
                            if denom != 0.0 and TA1 != 0.0:
                                beta = (TA2 * (Wxa[k] - Wxb) - TA1 * (Ea[k] - Eb)) / denom
                                alpha = (Wxa[k] - Wxb - TB1[ib] * beta) / TA1

                            print('k = ', k)
                            if alpha * sgn[ia] > 0.0 and beta * sgn[ib] > 0.0:
                                RHS = AN[:, :, ia] * alpha + BN[:, :, ib] * beta
                                break
                        else:
                            continue

                        break

                    s = 1.0
                    if not allow_neg:
                        idx = np.where(Nij != 0.0)
                        if len(idx[0]) > 0:
                            s = min(1.0, 1.0 / np.max(-RHS[idx] / Nij[idx]))

                    fb[:, :, k] = nb * (Nij + s * RHS) / Vol

                    if s >= 1.0:
                        break
    
    # --- Rescale to conserve density ---
    tot_a = np.array([np.sum(Vr2pidVra[:, None] * fa[:, :, k] * dVxa[None, :]) for k in range(nxa)])
    interp_tot_a = interp1d(xa, tot_a, kind='linear', bounds_error=False, fill_value='extrapolate')
    tot_b = interp_tot_a(xb[k0:k1+1])

    ii = np.where(fb > 0.0)
    if ii[0].size > 0:
        min_tot = np.min(fb[ii])
        for k in range(k0, k1 + 1):
            tot = np.sum(Vr2pidVrb[:, None] * fb[:, :, k] * dVxb[None, :])
            if tot > min_tot:
                if debug:
                    print(f"Density renormalization factor = {tot_b[k - k0]/tot}")
                fb[:, :, k] *= tot_b[k - k0] / tot

    # --- Optional debug diagnostics ---
    if debug:
        na = np.array([np.sum(Vr2pidVra[:, None] * fa[:, :, k] * dVxa[None, :]) for k in range(nxa)])
        Uxa = np.array([
            vtha * np.sum(Vr2pidVra[:, None] * fa[:, :, k] * (vxa[None, :] * dVxa[None, :])) / na[k] if na[k] > 0 else 0.0
            for k in range(nxa)
        ])
        Ta = np.zeros(nxa)
        for k in range(nxa):
            if na[k] > 0:
                ran2 = np.array([[vra[i]**2 + (vxa - Uxa[k]/vtha)**2 for vxa in vxa] for i in range(nvra)])
                Ta[k] = (mu * mH * vtha**2 *
                        np.sum(Vr2pidVra[:, None] * np.array(ran2) * fa[:, :, k] * dVxa[None, :]) / (3 * q * na[k]))

        nb_arr = np.array([np.sum(Vr2pidVrb[:, None] * fb[:, :, k] * dVxb[None, :]) for k in range(nxb)])
        Uxb = np.array([
            vthb * np.sum(Vr2pidVrb[:, None] * fb[:, :, k] * (vxb[None, :] * dVxb[None, :])) / nb_arr[k] if nb_arr[k] > 0 else 0.0
            for k in range(nxb)
        ])
        Tb = np.zeros(nxb)
        for k in range(nxb):
            if nb_arr[k] > 0:
                ran2 = np.array([[vrb[i]**2 + (vxb - Uxb[k]/vthb)**2 for vxb in vxb] for i in range(nvrb)])
                Tb[k] = (mu * mH * vthb**2 *
                        np.sum(Vr2pidVrb[:, None] * np.array(ran2) * fb[:, :, k] * dVxb[None, :]) / (3 * q * nb_arr[k]))


        plt.figure()
        plt.plot(xa, na, label='na')
        plt.plot(xb, nb_arr, label='nb')
        plt.title('Density conserved')
        plt.xlim([0, 0.02])
        plt.ylim([0,41])
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(xa, Uxa, label='Uxa')
        plt.plot(xb, Uxb, label='Uxb')
        plt.title('Ux conserved')
        plt.xlim([0, 0.02])
        plt.ylim([0,10000])
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(xa, Ta, label='Ta')
        plt.plot(xb, Tb, label='Tb')
        plt.title('T conserved')
        plt.xlim([0, 0.02])
        plt.ylim([0, 6])
        plt.legend()
        plt.show()



    return fb