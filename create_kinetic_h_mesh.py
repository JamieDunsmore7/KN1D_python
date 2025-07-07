#
# create_kinetic_h_mesh.py
#
# Python translation of Create_Kinetic_H_Mesh.pro
# Computes spatial mesh and related quantities for kinetic neutral modeling.
#

import numpy as np
from dataclasses import dataclass
from scipy.interpolate import interp1d
from create_vrvxmesh import Create_VrVxMesh
from jhs_coef import JHS_Coef
from sigmav_cx_h0 import SigmaV_CX_H0


def Create_Kinetic_H_Mesh(nv, mu, x, Ti, Te, n, PipeDia, fctr=1.0, E0_in=None):
    
    mH = 1.6726231e-27
    q = 1.602177e-19
    nx = len(x)

    # Estimate total reaction rate for destruction of hydrogen atoms and for interation with side walls
    RR = n * JHS_Coef(n, Te)

    v0 = np.sqrt(2 * 10 * q / (mu * mH)) # Set v0 to thermal speed of 10 eV neutral

    # Compute Y from RR and v0
    Y = np.zeros(nx)
    for k in range(1, nx):
        Y[k] = Y[k-1] - (x[k] - x[k-1]) * 0.5 * (RR[k] + RR[k-1]) / v0

    # Find x location where Y = -5, i.e., where nH should be down by exp(-5)
    if Y.min() > -5:
        xmaxH = np.max(x)
    else:
        interp_Y = interp1d(Y, x, bounds_error=False, fill_value=np.max(x))
        xmaxH = float(interp_Y(-5.0))
    xminH = x[0]

    # Interpolate Ti and Te onto a fine mesh between xminH and xmaxH
    xfine = np.linspace(xminH, xmaxH, 1001)
    Tifine = interp1d(x, Ti, bounds_error=False, fill_value='extrapolate')(xfine)
    Tefine = interp1d(x, Te, bounds_error=False, fill_value='extrapolate')(xfine)
    nfine = interp1d(x, n, bounds_error=False, fill_value='extrapolate')(xfine)
    PipeDiafine = interp1d(x, PipeDia, bounds_error=False, fill_value='extrapolate')(xfine)

    # Setup a vx,vr mesh based on raw data to get typical vx, vr values
    vx, vr, Tnorm, E0, ixE0, irE0 = Create_VrVxMesh(nv, Tifine, E0=E0_in)
    vth = np.sqrt(2 * q * Tnorm / (mu * mH))
    minVr = np.min(vr)
    minE0 = 0.5 * mH * (minVr * vth)**2 / q

    # Estimate interaction rate with side walls
    gamma_wall = np.zeros_like(xfine)
    for k in range(len(xfine)):
        if PipeDiafine[k] > 0:
            gamma_wall[k] = 2 * np.max(vr) * vth / PipeDiafine[k]

    # Estimate total reaction rate, including charge exchange and elastic scattering, and interaction with side walls
    RR = nfine * JHS_Coef(nfine, Tefine) + nfine * SigmaV_CX_H0(Tifine, np.full_like(xfine, minE0)) + gamma_wall

    # Compute local maximum grid spacing from dx_max = 2 min(vr) / RR
    big_dx = 0.02 * fctr
    dx_max = np.minimum(fctr * 0.8 * (2 * vth * np.min(vr) / RR), big_dx)

    # Construct xH axis
    xH = [xmaxH]
    xpt = xmaxH
    while xpt > xminH:
        interpdx = interp1d(xfine, dx_max, bounds_error=False, fill_value='extrapolate')
        dx1 = interpdx(xpt)
        xpt_test = xpt - dx1
        dx2 = dx1
        if xpt_test > xminH:
            dx2 = interpdx(xpt_test)
        dx = min(dx1, dx2)
        xpt -= dx
        if xpt >= xminH:
            xH.append(xpt)
    xH = np.array([xminH] + xH[::-1][:-1])

    TiH = interp1d(xfine, Tifine, bounds_error=False, fill_value='extrapolate')(xH)
    TeH = interp1d(xfine, Tefine, bounds_error=False, fill_value='extrapolate')(xH)
    neH = interp1d(xfine, nfine, bounds_error=False, fill_value='extrapolate')(xH)
    PipeDiaH = interp1d(xfine, PipeDiafine, bounds_error=False, fill_value='extrapolate')(xH)

    vx, vr, Tnorm, E0, ixE0, irE0 = Create_VrVxMesh(nv, TiH, E0=E0_in)

    # return results as a dictionary

    result = {
        'xH': xH,
        'TiH': TiH,
        'TeH': TeH,
        'neH': neH,
        'PipeDiaH': PipeDiaH,
        'vx': vx,
        'vr': vr,
        'Tnorm': Tnorm,
        'E0': E0,
        'ixE0': ixE0,
        'irE0': irE0
    }

    return result
