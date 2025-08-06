import numpy as np
from kn1d import KN1D

def run_test_kn1d():

    pause          = True
    compute_errors = True
    debrief        = True
    Hdebrief       = True
    H2debrief      = True
    mu             = 2.0

    #-------------------------------------------------------------------
    # Build the nonuniform x grid exactly as in IDL
    #-------------------------------------------------------------------
    nx1, xw, xlim = 8, 0.0, 0.2
    xa, xb         = xlim - 0.02, 0.265

    x1 = xw + (xa - xw) * np.arange(nx1) / nx1

    nx2 = 100
    x2  = xa + (xb - xa) * np.arange(nx2) / (nx2 - 1)

    x  = np.concatenate([x1, x2])
    nx = x.size

    #-------------------------------------------------------------------
    # Profiles Ti, Te, n
    #-------------------------------------------------------------------
    Ti = 10.0 * np.exp((x - xlim) / 0.025)
    Ti = np.maximum(Ti, 1.5)           # Ti = Ti > 1.5 in IDL

    Te = Ti.copy()

    n  = 1.0e19 * np.exp((x - xlim) / 0.03)
    n  = np.clip(n, 1.0e15, 2.0e20)

    #-------------------------------------------------------------------
    # Other inputs
    #-------------------------------------------------------------------
    GaugeH2  = 0.1
    LC       = np.zeros(nx)
    LC[x <= xlim] = 1.1
    vxi      = np.zeros(nx)
    xlimiter = 0.2
    xsep     = 0.25

    File = 'test_kn1d'
    PipeDia  = np.zeros(nx)

    # IDL used common KN1D_collisions to turn off H–H and H2–H2 elastic:
    H_H_EL   = False
    H2_H2_EL = False

    # Leave all the other collision flags at their defaults (True)
    H_P_EL     = True
    H_H2_EL    = True
    H_P_CX     = True
    H2_P_EL    = True
    H2_H_EL    = True
    H2_HP_CX   = True
    Simple_CX  = True

    # Simulation control
    truncate = 1e-3
    refine   = True
    NewFile  = True
    ReadInput= False


    #-------------------------------------------------------------------
    # Call the core KN1D routine
    #-------------------------------------------------------------------
    outputs = KN1D(
        x          = x,
        xlimiter   = xlimiter,
        xsep       = xsep,
        GaugeH2    = GaugeH2,
        mu         = mu,
        Ti         = Ti,
        Te         = Te,
        n          = n,
        vxi        = vxi,
        LC         = LC,
        PipeDia    = PipeDia,

        truncate     = truncate,
        refine       = refine,
        File = File,
        NewFile      = NewFile,
        ReadInput    = ReadInput,
        plot = 1,
        Hplot = 0,
        H2plot = 0,

        compute_errors = compute_errors,
        debrief        = debrief,
        pause          = pause,

        Hdebrief   = Hdebrief,
        H2debrief  = H2debrief,

        H_H_EL     = H_H_EL,
        H2_H2_EL   = H2_H2_EL,
        H_P_EL     = H_P_EL,
        # H_H2_EL    = H_H2_EL, this is set to be the same as H2_H_EL in internals of KN1D.pro
        H_P_CX     = H_P_CX,
        H2_P_EL    = H2_P_EL,
        H2_H_EL    = H2_H_EL,
        H2_HP_CX   = H2_HP_CX,
        Simple_CX  = Simple_CX,    )

    return outputs

if __name__ == "__main__":
    results = run_test_kn1d()
    print("KN1D test complete.")
