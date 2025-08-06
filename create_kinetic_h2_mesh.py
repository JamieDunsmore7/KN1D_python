#
# create_kinetic_h2_mesh.py
#
# Python translation of Create_Kinetic_H2_Mesh.pro
# Computes spatial mesh and related quantities for kinetic_h2.py
#

import numpy as np
from scipy.interpolate import interp1d
from sigmav_ion_hh import SigmaV_Ion_HH
from sigmav_h1s_h1s_hh import SigmaV_H1s_H1s_HH
from sigmav_h1s_h2s_hh import SigmaV_H1s_H2s_HH
from sigmav_cx_hh import SigmaV_CX_HH
from create_vrvxmesh import Create_VrVxMesh


def Create_Kinetic_H2_Mesh(nv, mu, x, Ti, Te, n, PipeDia, fctr=1.0, E0_in=None):
    '''
    INPUT:
    nv      - integer, number of desired elements in vr mesh
    mu      - float, 2 for deuterium, 1 for hydrogen
    x       - array, initial spatial grid
    Ti      - array, ion temperature profile (on the x grid)
    Te      - array, electron temperature profile (on the x grid)
    n       - array, density profile (on the x grid)
    PipeDia - array, pipe diameter profile (on the x grid)
    fctr    - float, factor to scale the maximum grid spacing
    E0_in   - array, energy where a velocity bin is desired (optional)

    OUTPUT:
    result  - dictionary containing:
        xH2       - array, spatial grid for H2
        TiH2      - array, ion temperature profile on xH2 grid
        TeH2      - array, electron temperature profile on xH2 grid
        neH2      - array, electron density profile on xH2 grid
        PipeDiaH2 - array, pipe diameter profile on xH2 grid
        vx       - array, vx grid
        vr       - array, vr grid
        Tnorm    - float, optimum normalization temperature
        E0       - energy where a velocity bin is desired (if provided)
        ixE0     - indices of vx corresponding to E0
        irE0     - index of vr corresponding to E0
    '''

    mH = 1.6726231e-27
    q = 1.602177e-19
    k_boltz = 1.380658e-23
    Twall = 293.0 * k_boltz / q  # eV
    v0_bar = np.sqrt(8.0 * Twall * q / (np.pi * 2 * mu * mH))

    nx = len(x)


    # Estimate total reaction rate for destruction of molecules and for interation with side walls
    RR = (
        n * SigmaV_Ion_HH(Te)
        + n * SigmaV_H1s_H1s_HH(Te)
        + n * SigmaV_H1s_H2s_HH(Te)
    )

    # Compute Y from RR and v0_bar
    Y = np.zeros(nx)
    for k in range(1, nx):
        Y[k] = Y[k - 1] - (x[k] - x[k - 1]) * 0.5 * (RR[k] + RR[k - 1]) / v0_bar

    # Find x location where Y = -10, i.e., where nH2 should be down by exp(-10)
    if Y.min() > -10:
        xmaxH2 = np.max(x)
    else:
        interp_Y = interp1d(Y, x, bounds_error=False, fill_value=np.max(x))
        xmaxH2 = float(interp_Y(-10.0))
    xminH2 = x[0]

    # Interpolate Ti and Te onto a fine mesh between xminH2 and xmaxH2
    xfine = np.linspace(xminH2, xmaxH2, 1001)
    Tifine = interp1d(x, Ti, bounds_error=False, fill_value='extrapolate')(xfine)
    Tefine = interp1d(x, Te, bounds_error=False, fill_value='extrapolate')(xfine)
    nfine = interp1d(x, n, bounds_error=False, fill_value='extrapolate')(xfine)
    PipeDiafine = interp1d(x, PipeDia, bounds_error=False, fill_value='extrapolate')(xfine)

    # Setup a vx,vr mesh based on raw data to get typical vx, vr values
    vx, vr, Tnorm, ixE0, irE0 = Create_VrVxMesh(nv, Tifine, E0=E0_in)
    vth = np.sqrt(2 * q * Tnorm / (mu * mH))

    # Estimate interaction rate with side walls
    gamma_wall = np.zeros_like(xfine)
    for k in range(len(xfine)):
        if PipeDiafine[k] > 0:
            gamma_wall[k] = 2 * np.max(vr) * vth / PipeDiafine[k]

    # Estimate total reaction rate, including charge exchange, elastic scattering, and interaction with side walls
    RR = (
        nfine * SigmaV_Ion_HH(Tefine)
        + nfine * SigmaV_H1s_H1s_HH(Tefine)
        + nfine * SigmaV_H1s_H2s_HH(Tefine)
        + 0.1 * nfine * SigmaV_CX_HH(Tifine, Tifine)
        + gamma_wall
    )
    # Compute local maximum grid spacing from dx_max = 2 min(vr) / RR
    big_dx = 0.02 * fctr
    dx_max = np.minimum(fctr * 0.8 * (2 * vth * np.min(vr) / RR), big_dx)

    pts = [xmaxH2]
    xpt = xmaxH2
    interp_dx = interp1d(xfine, dx_max, bounds_error=False, fill_value="extrapolate")
    while xpt > xminH2:
        # prepend
        pts = [xpt] + pts

        dx1 = interp_dx(xpt)
        test = xpt - dx1
        dx2 = interp_dx(test) if test > xminH2 else dx1
        xpt -= min(dx1, dx2)

    xH2 = np.array([xminH2] + pts[:-1])


    TiH2 = interp1d(xfine, Tifine, bounds_error=False, fill_value='extrapolate')(xH2)
    TeH2 = interp1d(xfine, Tefine, bounds_error=False, fill_value='extrapolate')(xH2)
    neH2 = interp1d(xfine, nfine, bounds_error=False, fill_value='extrapolate')(xH2)
    PipeDiaH2 = interp1d(xfine, PipeDiafine, bounds_error=False, fill_value='extrapolate')(xH2)

    vx, vr, Tnorm, ixE0, irE0 = Create_VrVxMesh(nv, TiH2, E0=E0_in)

    result = {
        "xH2": xH2,
        "TiH2": TiH2,
        "TeH2": TeH2,
        "neH2": neH2,
        "PipeDiaH2": PipeDiaH2,
        "vx": vx,
        "vr": vr,
        "Tnorm": Tnorm,
        "E0": E0_in,
        "ixE0": ixE0,
        "irE0": irE0,
    }

    return result
