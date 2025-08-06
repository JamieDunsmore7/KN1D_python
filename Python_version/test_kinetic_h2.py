"""
test_kinetic_h2.py

Standalone script to run the IDL-style Test_Kinetic_H2.pro
and verify basic sanity checks on the Python port.
"""

import numpy as np
from create_kinetic_h2_mesh import Create_Kinetic_H2_Mesh
from kinetic_h2 import kinetic_h2

def run_test_h2():
    #------------------------------------------------------------------------------
    # Common flags and parameters
    #------------------------------------------------------------------------------
    plot              = 1
    debug             = 0
    debrief           = 1
    pause             = False
    H2_H2_EL          = True
    H2_P_EL           = True
    H2_H_EL           = True
    H2_HP_CX          = True
    ni_correct        = True
    Simple_CX         = True
    Compute_H_Source  = True
    truncate          = 1e-4
    mu                = 2.0

    #------------------------------------------------------------------------------
    # Build the x‐grid and profiles (exactly as in Test_Kinetic_H2.pro)
    #------------------------------------------------------------------------------
    # spatial mesh: 8 points then 100 points
    nx1 = 8
    xw, xlim = 0.0, 0.2
    xa, xb = xlim - 0.02, 0.25

    x1 = xw + (xa - xw) * np.arange(nx1) / nx1
    nx2 = 100
    x2 = xa + (xb - xa) * np.arange(nx2) / (nx2 - 1)
    x  = np.concatenate([x1, x2])
    nx = x.size

    # temperature & density profiles
    Ti = np.maximum(1.0, 10.0 * np.exp((x - xlim) / 0.025))
    Te = Ti.copy()
    n  = np.maximum(1e17, 1e19 * np.exp((x - xlim) / 0.030))

    # uniform pipe diameter
    PipeDia = np.full(nx, 0.5)

    #------------------------------------------------------------------------------
    # Build the H2 velocity‐energy mesh
    #------------------------------------------------------------------------------
    nv    = 6
    # IDL uses 7 energies, but Create_Kinetic_H2_Mesh expects nv=6 (so 7 knots)
    Eneut = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])  # length nv+1
    fctr  = 0.8

    mesh2 = Create_Kinetic_H2_Mesh(
        nv=nv, mu=mu,
        x=x, Ti=Ti, Te=Te, n=n, PipeDia=PipeDia,
        E0_in=Eneut, fctr=fctr
    )
    # unpack mesh outputs
    xH2, TiH2, TeH2, neH2, PipeDiaH2 = (
        mesh2['xH2'], mesh2['TiH2'], mesh2['TeH2'],
        mesh2['neH2'], mesh2['PipeDiaH2']
    )
    vx, vr, Tnorm = mesh2['vx'], mesh2['vr'], mesh2['Tnorm']
    ixE0, irE0   = mesh2['ixE0'], mesh2['irE0']


    #------------------------------------------------------------------------------
    # Build the H2–boundary‐condition fH2BC and NuLoss array
    #------------------------------------------------------------------------------
    nvr, nvx = vr.size, vx.size
    vxi      = np.zeros_like(xH2)
    NuLoss   = np.zeros_like(xH2)

    # “cold” molecule source at Tneut=1/40 eV, corrected by √2
    Tneut = 1.0/40.0
    ip     = np.where(vx > 0)[0]

    fH2BC  = np.zeros((nvr, nvx))
    for j in ip:
        arg = -(vr**2 + vx[j]**2) / (Tneut/Tnorm/2.0)
        fH2BC[:, j] = np.exp(np.maximum(arg, -80.0))

    GammaxH2BC = 1e23


    #------------------------------------------------------------------------------
    # Run Kinetic_H2
    #------------------------------------------------------------------------------
    # COULD ALSO INPUT SEEDS INTO THIS FUNCTION IF I WANTED
    results2, seeds2 = kinetic_h2(
        vx=vx, vr=vr, x=xH2, Tnorm=Tnorm, mu=mu,
        Ti=TiH2, Te=TeH2, n=neH2, vxi=vxi,
        fH2BC=fH2BC, GammaxH2BC=GammaxH2BC,
        NuLoss=NuLoss, PipeDia=PipeDiaH2,
        truncate=truncate, Simple_CX=Simple_CX, Max_Gen=100,
        Compute_H_Source=Compute_H_Source,
        H2_H2_EL=H2_H2_EL, H2_P_EL=H2_P_EL,
        H2_H_EL=H2_H_EL, H2_HP_CX=H2_HP_CX,
        ni_correct=ni_correct,
        plot=plot, debug=debug, debrief=debrief, pause=pause, compute_errors=True
    )

    #------------------------------------------------------------------------------
    # Basic sanity checks
    #------------------------------------------------------------------------------
    # fH2 and nH2 must be in the results
    assert 'fH2' in results2 and 'nH2' in results2, "Missing H2 outputs"
    # shapes
    assert results2['fH2'].shape == (nvr, nvx, xH2.size)
    assert results2['nH2'].shape == xH2.shape
    # non‐negative
    assert np.all(results2['nH2'] >= 0), "Negative H2 density!"
    assert np.all(results2['fH2'] >= 0), "Negative H2 distribution!"

    print("Test_Kinetic_H2: PASS")
    return results2

def main():
    print("Running kinetic_h2 test script…")
    run_test_h2()

if __name__ == "__main__":
    main()
