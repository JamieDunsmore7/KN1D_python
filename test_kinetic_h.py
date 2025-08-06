"""
test_kinetic_h.py

A standalone script to run the three IDL-style test cases
for the Python implementations of Create_Kinetic_H_Mesh
and kinetic_h(), and report basic sanity checks.
"""

import numpy as np
from create_kinetic_h_mesh import Create_Kinetic_H_Mesh
from create_vrvxmesh import Create_VrVxMesh
from kinetic_h import kinetic_h

def run_test(test_case):
    # Common flags and parameters
    truncate            = 1e-4
    Simple_CX           = True # True
    H_H_EL              = True # True
    H_P_EL              = True # True
    H_H2_EL             = True # True
    H_P_CX              = True # True
    ni_correct          = True # True
    No_Johnson_Hinnov   = False # False
    No_Recomb           = False # False
    plot                = 1
    debug               = 0
    debrief             = 1
    pause               = False
    mu                  = 2.0

    if test_case == 1:
        # Test #1 — 3 eV neutrals
        nx1, xw, xlim = 8, 0.0, 0.2
        xa, xb = xlim - 0.02, 0.28
        x1 = xw + (xa - xw) * np.arange(nx1) / nx1
        nx2 = 100
        x2 = xa + (xb - xa) * np.arange(nx2) / (nx2 - 1)
        x = np.concatenate([x1, x2])
        Ti = np.maximum(1.0, 10.0 * np.exp((x - xlim) / 0.025))
        Te = Ti.copy()
        n  = np.clip(1e19 * np.exp((x - xlim) / 0.03), 1e15, 2e20)
        PipeDia = np.full_like(x, 0.5)

        mesh = Create_Kinetic_H_Mesh(nv=10, mu=mu,
                                    x=x, Ti=Ti, Te=Te, n=n, PipeDia=PipeDia,
                                    fctr=1.0, E0_in=None)
        xH, TiH, TeH, neH, PipeDiaH = mesh['xH'], mesh['TiH'], mesh['TeH'], mesh['neH'], mesh['PipeDiaH']
        vx, vr, Tnorm = mesh['vx'], mesh['vr'], mesh['Tnorm']

        max_gen   = 50
        GammaxHBC = 1e23
        Tneut     = 3.0

    elif test_case == 2:
        # Test #2 — no ionization, Ti = T0
        nx = 70
        xa, xb = 0.0, 0.05
        x = np.linspace(xa, xb, nx)
        Ti = np.full(nx, 10.0)
        Te = np.full(nx, 0.1)
        n  = np.full(nx, 5e19)
        PipeDia = np.zeros(nx)

        vx, vr, Tnorm, ixE0, irE0 = Create_VrVxMesh(nv=20, Ti=Ti)         
        xH, TiH, TeH, neH, PipeDiaH = x.copy(), Ti.copy(), Te.copy(), n.copy(), PipeDia.copy()

        max_gen   = 100
        GammaxHBC = 1e23
        Tneut     = Ti[0]

    else:
        # Test #3 — large ionization/CX fraction, Ti = T0
        nx = 70
        xa, xb = 0.0, 0.05
        x = np.linspace(xa, xb, nx)
        Ti = np.full(nx, 10.0)
        Te = np.full(nx, 30.0)
        n  = np.full(nx, 5e19)
        PipeDia = np.zeros(nx)

        vx, vr, Tnorm, ixE0, irE0 = Create_VrVxMesh(nv=40, Ti=Ti, E0=None)
        xH, TiH, TeH, neH, PipeDiaH = x.copy(), Ti.copy(), Te.copy(), n.copy(), PipeDia.copy()

        max_gen   = 100
        GammaxHBC = 1e22
        Tneut     = Ti[0]

    # Prepare vxi
    vxi = np.zeros_like(xH)

    # Build fHBC
    ip = np.where(vx > 0)[0]
    nvr, nvx = len(vr), len(vx)
    fHBC = np.zeros((nvr, nvx))
    for j in ip:
        exponent = -(vr**2 + vx[j]**2) / (Tneut / Tnorm)
        fHBC[:, j] = np.exp(np.maximum(exponent, -80.0))

    print('vx:', vx)
    print('vr:', vr)

    # Run kinetic_h
    results, seeds = kinetic_h(
        vx=vx, vr=vr, x=xH, Tnorm=Tnorm, mu=mu,
        Ti=TiH, Te=TeH, n=neH, vxi=vxi,
        fHBC=fHBC, GammaxHBC=GammaxHBC, PipeDia=PipeDiaH,
        fH2=None, fSH=None, nHP=None, THP=None,
        truncate=truncate, Simple_CX=Simple_CX, Max_Gen=max_gen,
        No_Johnson_Hinnov=No_Johnson_Hinnov, No_Recomb=No_Recomb,
        H_H_EL=H_H_EL, H_P_EL=H_P_EL, H_H2_EL=H_H2_EL, H_P_CX=H_P_CX,
        ni_correct=ni_correct, compute_errors=True,
        plot=plot, debug=debug, debrief=debrief, pause=pause,
        kinetic_h_seeds=None
    )

    # Sanity checks
    assert 'nH' in results and 'fH' in results, "Missing outputs"
    assert results['nH'].shape == xH.shape, "nH has wrong shape"
    assert results['fH'].shape == (nvr, nvx, xH.size), "fH has wrong shape"
    assert np.all(results['nH'] >= 0), "Negative density encountered"

    print(f"Test case {test_case}: PASS")
    return results

def main():
    print("Running kinetic_h test script...")
    for tc in (1,2,3):
        run_test(tc)
    print("All tests passed.")

if __name__ == "__main__":
    main()
