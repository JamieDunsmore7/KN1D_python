#
# Create_SIGMAV_EL_H_P_DATA.py
#
#   Creates a 2D SigmaV data table in particle energy and target temperature
#   and saves it as a .npz file.
#

import numpy as np
from make_sigma_v import Make_SigmaV
from sigma_el_p_h import Sigma_EL_P_H  # Replace if needed with other cross section functions

def create_SIGMAV_EL_H_P_data(output_file='sigmav_el_h_p_data.npz'):
    """
    Generate 2D <sigma*v> table (m^2/s) for elastic H-P collisions
    and save as a NumPy .npz file.
    """
    mE, nT = 50, 50
    Emin, Emax = 0.1, 2.0e4  # eV
    Tmin, Tmax = 0.1, 2.0e4  # eV

    E_particle = np.logspace(np.log10(Emin), np.log10(Emax), mE)
    T_target = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)

    mu_particle = 1.0
    mu_target = 1.0

    SigmaV = np.zeros((mE, nT))

    for iT, T in enumerate(T_target):
        SigmaV[:, iT] = Make_SigmaV(E_particle, mu_particle, T, mu_target, Sigma_EL_P_H)

    ln_E_particle = np.log(E_particle)
    ln_T_target = np.log(T_target)

    print(f"Saving results to {output_file}")
    np.savez(
        output_file,
        Ln_E_Particle=ln_E_particle,
        Ln_T_Target=ln_T_target,
        SigmaV=SigmaV
    )

    return

