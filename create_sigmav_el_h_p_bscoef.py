#
# Create_SIGMAV_EL_H_P_BSCOEF.py
#
#   Creates a .npz file storing Bi-cubic spline interpolation
#

import numpy as np
from scipy.interpolate import RectBivariateSpline
from sigma_el_p_h import Sigma_EL_P_H
from make_sigma_v import Make_SigmaV
from spline_helper_functions import save_rbspline
import os



def create_SIGMAV_EL_H_P_bscoef(sigma_function = Sigma_EL_P_H, output_file='sigmav_el_h_p_bscoef.npz'):
    """
    Creates a .npz file storing B-spline coefficients for log10-interpolated
    <sigma*v> values for elastic H on P collisions.

    Parameters:
            Function to compute sigma(E) in m^2 from relative energy in eV.

        output_file : str
            Path to output file.
    """    # Log-spaced energy and temperature arrays
    mE = 6
    nT = 6
    Emin, Emax = 0.1, 2.0e4  # eV
    Tmin, Tmax = 0.1, 2.0e4  # eV

    E_particle = np.logspace(np.log10(Emin), np.log10(Emax), mE)
    T_target = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)

    mu_particle = 1.0
    mu_target = 1.0


    SigmaV = np.zeros((mE, nT))
    for iT, T in enumerate(T_target):
        print(f"Processing T = {T:.3g} eV")
        SigmaV[:, iT] = Make_SigmaV(E_particle, mu_particle, T, mu_target, sigma_function)

    # Convert axes to log scale
    logE_particle = np.log(E_particle)
    logT_target = np.log(T_target)
    LogSigmaV_EL_H_P = np.log(SigmaV)

    order = 4

    spline = RectBivariateSpline(logE_particle, logT_target, LogSigmaV_EL_H_P, kx=order - 1, ky=order - 1)
    save_rbspline(output_file, spline)
    print(f"Spline saved to {output_file}")



if __name__ == "__main__":
    create_SIGMAV_EL_H_P_bscoef()
