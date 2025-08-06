import numpy as np
from create_vrvxmesh import Create_VrVxMesh
from make_dvr_dvx import make_dvr_dvx
from sigma_el_p_h import Sigma_EL_P_H # may need to add the rest of the sigma functions here later

def Make_SigmaV(E_particle, mu_particle, T_target, mu_target, sigma_function):
    """
    Compute <sigma*v> for elastic collisions using a Maxwellian target.

    Parameters:
        E_particle : ndarray, eV
        mu_particle : float
        T_target : float, eV
        mu_target : float
        sigma_function : callable, expects E in eV and returns sigma in m^2

    Returns:
        sigma_v : ndarray of shape (len(E_particle),), units of m^2/s

    Test particle has velocity Vxa
    Target particles are a maxwellian in Vrb and Vb
    """

    # Constants
    mH = 1.6726231e-27  # kg
    q = 1.602177e-19    # C

    Trange = np.sort(np.concatenate([E_particle, [T_target]]))
    # Velocity mesh
    nvb = 100
    vxb, vrb, Tnorm, _, _ = Create_VrVxMesh(nvb, Trange)
    Vr2pidVrb, VrVr4pidVrb, dVxb = make_dvr_dvx(vrb, vxb)[0:3] # only need the first 3 ouputs

    vth = np.sqrt(2 * q * Tnorm / (mu_target * mH))

    # Normalized particle velocities
    Vxa = np.sqrt(2 * q * E_particle / (mu_particle * mH)) / vth
    nvxa = Vxa.size
    nvrb = vrb.size
    nvxb = vxb.size

    # Maxwellian target distribution
    fi_hat = np.exp(-((vrb[:, None]**2 + vxb[None, :]**2) * Tnorm / T_target))
    norm = np.sum(Vr2pidVrb[:, None] * fi_hat * dVxb[None, :])
    fi_hat /= norm

    # Relative velocities at each mesh point
    Vrel = np.zeros((nvrb, nvxb, nvxa))
    for k in range(nvxa):
        Vrel[:, :, k] = np.sqrt(vrb[:, None]**2 + (vxb[None, :] - Vxa[k])**2)

    # Energy for sigma function (in eV)
    E_rel = 0.5 * (vth * vth * Vrel * Vrel) * mu_particle * mH / q

    # Get sigma(E_rel), same shape as Vrel
    sig = sigma_function(E_rel) # remember this sigma function is passed into the function

    # Compute sigma*v averaged over Maxwellian target
    sigma_v = np.zeros(nvxa)
    for k in range(nvxa):
        integrand = sig[:, :, k] * Vrel[:, :, k] * fi_hat
        sigma_v[k] = vth * np.sum(Vr2pidVrb[:, None] * integrand * dVxb[None, :])

    return sigma_v
