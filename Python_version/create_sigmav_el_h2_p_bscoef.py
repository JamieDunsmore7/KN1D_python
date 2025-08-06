#
# Create_SIGMAV_EL_H2_P_BSCOEF.py
#
#   Creates a .npz file storing Bi-cubic spline interpolation
#

import numpy as np
from scipy.interpolate import RectBivariateSpline
from sigma_el_p_hh import Sigma_EL_P_HH
from make_sigma_v import Make_SigmaV
import os


def create_SIGMAV_EL_H2_P_bscoef(output_file='sigmav_el_h2_p_bscoef.npz'):
    """
    Creates a .npz file storing B-spline coefficients for log10-interpolated
    <sigma*v> values for elastic H2 on P collisions.
    """
    # Log-spaced energy and temperature arrays
    mE = 5
    nT = 5
    Emin, Emax = 0.1, 1.0e3  # eV
    Tmin, Tmax = 0.1, 1.0e3  # eV

    E_particle = np.logspace(np.log10(Emin), np.log10(Emax), mE)
    T_target = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)

    mu_particle = 2.0
    mu_target = 1.0

    # Allocate array for <sigma*v>
    SigmaV = np.zeros((mE, nT))

    for iT, T in enumerate(T_target):
        SigmaV[:, iT] = Make_SigmaV(E_particle, mu_particle, T, mu_target, Sigma_EL_P_HH)

    # Convert axes to log scale
    logE_particle = np.log(E_particle)
    logT_target = np.log(T_target)
    LogSigmaV_EL_H2_P = np.log(SigmaV)

    # Compute B-spline coefficients
    order = 4

    spline = RectBivariateSpline(logE_particle, logT_target, LogSigmaV_EL_H2_P, kx=order - 1, ky=order - 1)
    spline_tx, spline_ty, spline_c = spline.tck
    spline_kx, spline_ky = spline.degrees

    np.savez(output_file,
             tx=spline_tx,
             ty=spline_ty,
             c=spline_c,
             kx=spline_kx,
             ky=spline_ky,
             )
    
    print(f"Data saved to {output_file}")

    return

