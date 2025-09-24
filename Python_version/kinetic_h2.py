#+
# Kinetic_H2.py
#
# This subroutine is part of the "KN1D" atomic and molecular neutal transport code.
#
#   This subroutine solves a 1-D spatial, 2-D velocity kinetic neutral transport
# problem for molecular hydrogen or deuterium (H2) by computing successive generations of
# charge exchange and elastic scattered neutrals. The routine handles electron-impact
# ionization and dissociation, molecular ion charge exchange, and elastic
# collisions with hydrogenic ions, neutral atoms, and molecules.
#
#   The positive vx half of the atomic neutral distribution function is inputted at x(0)
# (with arbitrary normalization). The desired flux on molecules entering the slab geometry at
# x(0) is specified. Background profiles of plasma ions, (e.g., Ti(x), Te(x), n(x), vxi(x),...)
# and atomic distribution function (fH) is inputted. (fH can be computed by procedure 
# "Kinetic_H.pro".) Optionally, a molecular source profile (SH2(x)) is also inputted. 
# The code returns the molecular hydrogen distribution function, fH2(vr,vx,x) for all 
# vx, vr, and x of the specified vr,vx,x grid. The atomic (H) and ionic (P) hydrogen 
# source profiles and the atomic source velocity distribution functions 
# resulting from Franck-Condon reaction product energies of H are also returned.
#
#   Since the problem involves only the x spatial dimension, all distribution functions
# are assumed to have rotational symmetry about the vx axis. Consequently, the distributions
# only depend on x, vx and vr where vr =sqrt(vy^2+vz^2)
#
#  History:
#
#    B. LaBombard   First coding based on Kinetic_Neutrals.pro 22-Dec-2000
#
#    For more information, see write-up: "A 1-D Space, 2-D Velocity, Kinetic
#    Neutral Transport Algorithm for Hydrogen Molecules in an Ionizing Plasma", B. LaBombard
#
#   Translated to python by J. Dunsmore, 06-Aug-2025
#
# Note: Variable names contain characters to help designate species -
#       atomic neutral (H), molecular neutral (H2), molecular ion (HP), proton (i) or (P)
#

import numpy as np
import matplotlib.pyplot as plt
from make_dvr_dvx import make_dvr_dvx
from create_shifted_maxwellian_core import create_shifted_maxwellian_core
from create_shifted_maxwellian import create_shifted_maxwellian
from sigmav_ion_hh import SigmaV_Ion_HH
from sigmav_h1s_h1s_hh import SigmaV_H1s_H1s_HH
from path_interp_2d import path_interp_2d
from sigmav_h1s_h2s_hh import SigmaV_H1s_H2s_HH
from sigmav_p_h1s_hh import SigmaV_P_H1s_HH
from sigmav_h2p_h2s_hh import SigmaV_H2p_H2s_HH
from sigmav_h1s_hn3_hh import SigmaV_H1s_Hn3_HH
from sigmav_p_h1s_hp import SigmaV_P_H1s_HP
from sigmav_p_hn2_hp import SigmaV_P_Hn2_HP
from sigmav_p_p_hp import SigmaV_P_P_HP
from sigmav_h1s_hn_hp import SigmaV_H1s_Hn_HP
from sigma_cx_hh import Sigma_CX_HH
from sigma_el_h_hh import Sigma_EL_H_HH
from sigma_el_hh_hh import Sigma_EL_HH_HH
from sigma_el_p_hh import Sigma_EL_P_HH
from sigmav_cx_hh import SigmaV_CX_HH




def kinetic_h2(
    # Pure inputs
    vx, vr, x, Tnorm, mu, Ti, Te, n, vxi,
    fH2BC, GammaxH2BC, NuLoss,
    PipeDia=None, fH=None, SH2=None,

    # the next ones are a bit confusing. They are outputs, but 'seed values' can be specified as input
    fH2=None, nHP=None, THP=None,

    # keyword inputs
    truncate=1.0e-4,
    Simple_CX=True, Max_Gen=50, Compute_H_Source=False,
    No_Sawada=False, H2_H2_EL=False, H2_P_EL=False, H2_H_EL=False,
    H2_HP_CX=False, ni_correct=False,
    error=0, compute_errors=False,
    plot=0, debug=0, debrief=0, pause=False,

    # next ones are the common blocks that can be used to pass data between runs
    h2_seeds = None,
    h2_H_moments = None,
    h2_internal = None,
):
    """
    Solve a 1D spatial, 2D velocity kinetic neutral transport problem for molecular hydrogen or deuterium.

    Parameters
    ----------
    vx : ndarray
        Normalized x velocity coordinate (must be even-length, symmetric, no zeros).
    vr : ndarray
        Normalized radial velocity coordinate (positive, increasing, no zeros).
    x : ndarray
        Spatial coordinate (meters, positive, increasing).
    Tnorm : float
        Temperature corresponding to thermal speed (eV).
    mu : float
        Mass ratio (1=hydrogen, 2=deuterium).
    Ti, Te, n, vxi : ndarray
        Ion temperature (eV), electron temperature (eV), density (m^-3), flow speed (m/s) profiles.
    fH2BC : ndarray
        Boundary condition for molecular hydrogen distribution (nvr x nvx).
    GammaxH2BC : float
        Desired H2 flux density at x=0 (m^-2 s^-1).
    NuLoss : ndarray
        Characteristic molecular ion loss frequency profile (1/s).
    PipeDia : ndarray or None
        Pipe diameter profile (m); if None, treated as zero.
    fH : ndarray or None
        Atomic hydrogen distribution (nvr x nvx x nx); if None, no H–H2 collisions.
    SH2 : ndarray or None
        Molecular H2 source profile (m^-3 s^-1); default zero.

    Inputs and Outputs
    ----------
    fH2 : ndarray or None
        Initial molecular hydrogen distribution (nvr x nvx x nx); if None, zeros.
    nHP, THP : ndarray or None
        Initial molecular ion density and temperature profiles; if None, zeros or defaults.
    
    Keyword Arguments
    ----------
    truncate : float
        Convergence threshold for generation iterations (default 1e-4).
    Simple_CX : bool
        Use Maxwellian-weighted CX source if True; else compute full convolution (default True).
    Max_Gen : int
        Maximum number of generations (default 50).
    Compute_H_Source : bool
        If True, compute atomic source distributions (default False).
    No_Sawada : bool
        If True, disable Sawada correction (default False).
    H2_H2_EL, H2_P_EL, H2_H_EL, H2_HP_CX : bool
        Flags for H2–H2 elastic, H2–proton elastic, H2–H elastic, H2–HP CX (all default False).
    ni_correct : bool
        Quasineutral correction: ni = n - nHP (default False).
    ESH : ndarray or None
        Output normalized H source energy distribution (nvr x nx).
    Eaxis : ndarray or None
        Output energy axis for ESH (nvr).
    error : int
        Error flag (0=no error, 1=error) (default 0).
    compute_errors : bool
        If True, compute error diagnostics (default False).
    plot, debug, debrief, pause : int/bool
        Plotting and debug controls (default 0/False).


    Returns
    -------
    results : dict
        Dictionary containing the computed distribution function and moments.
    seeds : dict
        Dictionary containing internal variables for potential reuse.
    output : dict
        Dictionary containing extra output variables such as piH2_xx, piH2_yy, etc.
    errors : dict
        Dictionary containing error diagnostics if compute_errors is True.
    H_moments : dict
        Dictionary containing H moments if provided.
    internal : dict
        Dictionary containing internal variables for potential reuse.
    NOTE: seeds, H_moments and internal from the previous run can also be provided as an input.
    """


    prompt = 'Kinetic_H2 => '


    # get the seeds from previous run if they have been passed
    if h2_seeds is not None:
        vx_s = h2_seeds['vx_s']
        vr_s = h2_seeds['vr_s']
        x_s = h2_seeds['x_s']
        Tnorm_s = h2_seeds['Tnorm_s']
        mu_s = h2_seeds['mu_s']
        Ti_s = h2_seeds['Ti_s']
        Te_s = h2_seeds['Te_s']
        n_s = h2_seeds['n_s']
        vxi_s = h2_seeds['vxi_s']
        fH2BC_s = h2_seeds['fH2BC_s']
        GammaxH2BC_s = h2_seeds['GammaxH2BC_s']
        NuLoss_s = h2_seeds['NuLoss_s']
        PipeDia_s = h2_seeds['PipeDia_s']
        fH_s = h2_seeds['fH_s']
        SH2_s = h2_seeds['SH2_s']
        fH2_s = h2_seeds['fH2_s']
        nHP_s = h2_seeds['nHP_s']
        THP_s = h2_seeds['THP_s']
        Simple_CX_s = h2_seeds['Simple_CX_s']
        Sawada_s = h2_seeds['Sawada_s']
        H2_H2_EL_s = h2_seeds['H2_H2_EL_s']
        H2_P_EL_s = h2_seeds['H2_P_EL_s']
        H2_H_EL_s = h2_seeds['H2_H_EL_s']
        H2_HP_CX_s = h2_seeds['H2_HP_CX_s']
        ni_correct_s = h2_seeds['ni_correct_s']
        print('Using seeds from previous run')

    else:
        vx_s = None
        vr_s = None
        x_s = None
        Tnorm_s = None
        mu_s = None
        Ti_s = None
        Te_s = None
        n_s = None
        vxi_s = None
        fH2BC_s = None
        GammaxH2BC_s = None
        NuLoss_s = None
        PipeDia_s = None
        fH_s = None
        SH2_s = None
        fH2_s = None
        nHP_s = None
        THP_s = None
        Simple_CX_s = None
        Sawada_s = None
        H2_H2_EL_s = None
        H2_P_EL_s = None
        H2_H_EL_s = None
        H2_HP_CX_s = None
        ni_correct_s = None
        print('No seeds provided, starting fresh')


    # Get the H moments from previous run if they have been passed
    if h2_H_moments is not None:
        nH = h2_H_moments['nH']
        VxH = h2_H_moments['VxH']
        TH = h2_H_moments['TH']
        print('Using H moments from previous run')
    else:
        nH = None
        VxH = None
        TH = None
        print('No H moments provided, starting fresh')

    # Get the internal variables from previous run if they have been passed
    if h2_internal is not None:
        vr2vx2 = h2_internal['vr2vx2']
        vr2vx_vxi2 = h2_internal['vr2vx_vxi2']
        fw_hat = h2_internal['fw_hat']
        fi_hat = h2_internal['fi_hat']
        fHp_hat = h2_internal['fHp_hat']
        EH2_P = h2_internal['EH2_P']
        sigv = h2_internal['sigv']
        alpha_loss = h2_internal['alpha_Loss']
        v_v2 = h2_internal['v_v2']
        v_v = h2_internal['v_v']
        vr2_vx2 = h2_internal['vr2_vx2']
        vx_vx = h2_internal['vx_vx']
        Vr2pidVrdVx = h2_internal['Vr2pidVrdVx']
        SIG_CX = h2_internal['SIG_CX']
        SIG_H2_H2 = h2_internal['SIG_H2_H2']
        SIG_H2_H = h2_internal['SIG_H2_H']
        SIG_H2_P = h2_internal['SIG_H2_P']
        Alpha_CX = h2_internal['Alpha_CX']
        Alpha_H2_H = h2_internal['Alpha_H2_H']
        MH2_H2_sum = h2_internal['MH2_H2_sum']
        Delta_nH2s = h2_internal['Delta_nH2s']
        print('Using internal values from previous run')
    else:
        vr2vx2 = None
        vr2vx_vxi2 = None
        fw_hat = None
        fi_hat = None
        fHp_hat = None
        EH2_P = None
        sigv = None
        alpha_loss = None
        v_v2 = None
        v_v = None
        vr2_vx2 = None
        vx_vx = None
        Vr2pidVrdVx = None
        SIG_CX = None
        SIG_H2_H2 = None
        SIG_H2_H = None
        SIG_H2_P = None
        Alpha_CX = None
        Alpha_H2_H = None
        MH2_H2_sum = None
        Delta_nH2s = None
        print('No internal values provided, starting fresh')

    # NOTE: this is something to keep an eye on. It feels like it should live in the common block, but this isn't the case in the IDL version
    Alpha_H2_P = None

    # Internal debug switches and tolerances
    shifted_Maxwellian_debug = False
    CI_Test = True
    Do_Alpha_CX_Test = False

    DeltaVx_tol = 0.01
    Wpp_tol = 0.001

    # Not sure about this
    if debug > 0:
        plot = plot > 1
        debrief = debrief > 1
        pause = True

    if No_Sawada:
        Sawada = False
    else:
        Sawada = True

    error=0

    # Convert to numpy arrays
    vr = np.asarray(vr, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    x  = np.asarray(x, dtype=np.float64)

    # Dimensions
    nvr = len(vr)
    nvx = len(vx)
    nx  = len(x)

    # Check dx monotonicity
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError(f'{prompt}x(*) must be increasing with index!')

    # vx must be even
    if len(vx) % 2 != 0:
        raise ValueError(f'{prompt}Number of elements in vx must be even!')

    # Check array lengths
    if len(Ti) != nx:
        raise ValueError(f'{prompt}Number of elements in Ti and x do not agree!')

    if vxi is None:
        vxi = np.zeros(nx)
    if len(vxi) != nx:
        raise ValueError(f'{prompt}Number of elements in vxi and x do not agree!')
    

    if len(Te) != nx:
        raise ValueError(f'{prompt}Number of elements in Te and x do not agree!')

    if len(n) != nx:
        raise ValueError(f'{prompt}Number of elements in n and x do not agree!')
    
    if NuLoss is None:
        NuLoss = np.zeros(nx)
    
    if len(NuLoss) != nx:
        raise ValueError(f'{prompt}Number of elements in NuLoss and x do not agree!')

    if GammaxH2BC is None:
        raise ValueError(f'{prompt}GammaxH2BC is not defined!')
    
    if fH is None:
        fH = np.zeros((nvr, nvx, nx))
    elif fH.shape != (nvr, nvx, nx):
        raise ValueError(f'{prompt}Shape of fH does not match (nvr, nvx, nx)!')

    if PipeDia is None:
        PipeDia = np.zeros(nx)
    if len(PipeDia) != nx:
        raise ValueError(f'{prompt}Number of elements in PipeDia and x do not agree!')

    if fH2BC.shape != (nvr, nvx):
        raise ValueError(f'{prompt}Shape of fH2BC does not match (nvr, nvx)!')

    if fH2 is None:
        fH2 = np.zeros((nvr, nvx, nx))
    elif fH2.shape != (nvr, nvx, nx):
        raise ValueError(f'{prompt}Shape of fH2 does not match (nvr, nvx, nx)!')
    
    if SH2 is None:
        SH2 = np.zeros(nx)
    if len(SH2) != nx:
        raise ValueError(f'{prompt}Number of elements in SH2 and x do not agree!')

    if nHP is None:
        nHP = np.zeros(nx)
    elif len(nHP) != nx:
        raise ValueError(f'{prompt}Number of elements in nHP and x do not agree!')

    if THP is None:
        THP = np.full(nx, 3.0)
    elif len(THP) != nx:
        raise ValueError(f'{prompt}Number of elements in THP and x do not agree!')

    if np.all(np.abs(vr) <= 0.0):
        raise ValueError(f'{prompt}vr is all 0!')

    if np.any(vr <= 0.0):
        raise ValueError(f'{prompt}vr contains zero or negative element(s)!')

    if np.all(np.abs(vx) <= 0.0):
        raise ValueError(f'{prompt}vx is all 0!')

    if np.sum(x) <= 0.0:
        raise ValueError(f'{prompt}Total(x) is less than or equal to 0!')

    if Tnorm is None:
        raise ValueError(f'{prompt}Tnorm is not defined!')

    if mu is None:
        raise ValueError(f'{prompt}mu is not defined!')

    if mu not in [1, 2]:
        raise ValueError(f'{prompt}mu must be 1 or 2!')
    
    # These next few lines are all about getting the right labels for the plots
    # Might make more sense to stick all of this inside a function or just write out the labels explicitly
    _e = r"$e^-$"

    if mu == 1:
        _p   = r"$\mathrm{H}^+$"
        _H   = r"$\mathrm{H}_0$"
        _H1s = r"$\mathrm{H}(1s)$"
        _H2s  = r"$\mathrm{H}^*(2s)$"
        _H2p  = r"$\mathrm{H}^*(2p)$" # NOTE: I think there is an error in the original idl where this was called _Hp
        _Hn2 = r"$\mathrm{H}^*(n=2)$"
        _Hn3 = r"$\mathrm{H}^*(n=3)$"
        _Hn  = r"$\mathrm{H}^*(n\geq2)$"
        _HH  = r"$\mathrm{H}_2$"
        _Hp = r"$\mathrm{H}_2^+$"
    else:
        _p   = r"$\mathrm{D}^+$"
        _H   = r"$\mathrm{D}_0$"
        _H1s = r"$\mathrm{D}(1s)$"
        _H2s  = r"$\mathrm{D}^*(2s)$"
        _H2p  = r"$\mathrm{D}^*(2p)$" # NOTE: I think there is an error in the original idl where this was called _Hp
        _Hn2 = r"$\mathrm{D}^*(n=2)$"
        _Hn3 = r"$\mathrm{D}^*(n=3)$"
        _Hn  = r"$\mathrm{D}^*(n\geq2)$"
        _HH  = r"$\mathrm{D}_2$"
        _Hp = r"$\mathrm{D}_2^+$"

    plus = r" + "
    arrow = r" $\rightarrow$ "
    elastic = r" (elastic)"

    # Reaction strings for molecular hydrogen (H₂ or D₂)
    _R1  = _e + plus + _HH + arrow + _e + plus + _Hp + plus + _e
    _R2  = _e + plus + _HH + arrow + _e + plus + _H1s + plus + _H1s
    _R3  = _e + plus + _HH + arrow + _e + plus + _H1s + plus + _H2s
    _R4  = _e + plus + _HH + arrow + _e + plus + _p   + plus + _H1s + plus + _e
    _R5  = _e + plus + _HH + arrow + _e + plus + _H2p + plus + _H2s
    _R6  = _e + plus + _HH + arrow + _e + plus + _H1s + plus + _Hn3
    _R7  = _e + plus + _Hp + arrow + _e + plus + _p   + plus + _H1s
    _R8  = _e + plus + _Hp + arrow + _e + plus + _p   + plus + _Hn2
    _R9  = _e + plus + _Hp + arrow + _e + plus + _p   + plus + _p   + plus + _e
    _R10 = _e + plus + _Hp + arrow + _H1s + plus + _Hn
    _R11 = _HH + plus + _p   + arrow + _HH + plus + _p   + elastic
    _R12 = _HH + plus + _H   + arrow + _HH + plus + _H   + elastic
    _R13 = _HH + plus + _HH  + arrow + _HH + plus + _HH  + elastic
    _R14 = _HH + plus + _Hp  + arrow + _Hp + plus + _HH

    _Rn=["",_R1,_R2,_R3,_R4,_R5,_R6,_R7,_R8,_R9,_R10,_R11,_R12,_R13,_R14]

    # Check vx symmetry
    in_neg = np.where(vx < 0)[0]
    if len(in_neg) < 1:
        raise ValueError(f'{prompt}vx contains no negative elements!')
    
    ip_pos = np.where(vx > 0)[0]
    if len(ip_pos) < 1:
        raise ValueError(f'{prompt}vx contains no positive elements!')
    
    izero = np.where(vx == 0)[0]
    if len(izero) > 0:
        raise ValueError(f'{prompt}vx contains one or more zero elements!')

    if not np.allclose(vx[ip_pos], -vx[in_neg][::-1]):
        raise ValueError(f'{prompt}vx array elements are not symmetric about zero!')

    # Prepare a copy of fH2BC containing only the positive-vx half
    fH2BC_input = fH2BC.copy()
    fH2BC_input[:, :] = 0.0
    fH2BC_input[:, ip_pos] = fH2BC[:, ip_pos]
    test = np.sum(fH2BC_input)
    if test <= 0.0:
        raise ValueError(f'{prompt}Values for fH2BC(*,*) with vx > 0 are all zero!')

    # Output variables
    nH2 = np.zeros(nx)
    GammaxH2 = np.zeros(nx)
    VxH2 = np.zeros(nx)
    pH2 = np.zeros(nx)
    TH2 = np.zeros(nx)
    NuDis = np.zeros(nx)
    NuE = np.zeros(nx)

    qxH2 = np.zeros(nx)
    qxH2_total = np.zeros(nx)
    Sloss = np.zeros(nx)
    WallH2 = np.zeros(nx)
    QH2 = np.zeros(nx)
    RxH2 = np.zeros(nx)
    QH2_total = np.zeros(nx)
    piH2_xx = np.zeros(nx)
    piH2_yy = np.zeros(nx)
    piH2_zz = np.zeros(nx)
    RxH2CX = np.zeros(nx)
    RxH_H2 = np.zeros(nx)
    RxP_H2 = np.zeros(nx)
    RxW_H2 = np.zeros(nx)
    EH2CX = np.zeros(nx)
    EH_H2 = np.zeros(nx)
    EP_H2 = np.zeros(nx)
    EW_H2 = np.zeros(nx)
    Epara_PerpH2_H2 = np.zeros(nx)
    AlbedoH2 = 0.0

    fSH = np.zeros((nvr, nvx, nx))
    SH = np.zeros(nx)
    SP = np.zeros(nx)
    SHP = np.zeros(nx)
    ESH = np.zeros((nvr, nx))
    Eaxis = np.zeros(nx)

    # --- Internal variables ---
    mH        = 1.6726231e-27       # hydrogen mass (kg)
    q   = 1.602177e-19        # elementary charge (C)
    k_boltz   = 1.380658e-23        # Boltzmann constant (J/K)
    Twall     = 293.0 * k_boltz / q   # room temperature in eV

    Work            = np.zeros(nvr * nvx)
    fH2G            = np.zeros((nvr, nvx, nx))
    NH2G            = np.zeros((nx, Max_Gen + 1))
    vth             = np.sqrt(2 * q * Tnorm / (mu * mH))
    vth2            = vth**2
    vth3            = vth2 * vth
    fH2s            = np.zeros(nx)
    nH2s            = np.zeros(nx)
    THPs            = np.zeros(nx)
    nHPs            = np.zeros(nx)
    Alpha_H2_H2     = np.zeros((nvr, nvx))
    Omega_H2_P      = np.zeros(nx)
    Omega_H2_H      = np.zeros(nx)
    Omega_H2_H2     = np.zeros(nx)
    VxH2G           = np.zeros(nx)
    TH2G            = np.zeros(nx)
    Wperp_paraH2    = np.zeros(nx)
    vr2vx2_ran2     = np.zeros((nvr, nvx))
    vr2_2vx_ran2    = np.zeros((nvr, nvx))
    vr2_2vx2_2D     = np.zeros((nvr, nvx))
    RxCI_CX        = np.zeros(nx)
    RxCI_H_H2       = np.zeros(nx)
    RxCI_P_H2       = np.zeros(nx)
    Epara_Perp_CI   = np.zeros(nx)
    CI_CX_error     = np.zeros(nx)
    CI_H_H2_error   = np.zeros(nx)
    CI_P_H2_error   = np.zeros(nx)
    CI_H2_H2_error  = np.zeros(nx)
    Maxwell         = np.zeros((nvr, nvx, nx))

    # --- Velocity space weights and volume elements ---
    (Vr2pidVr, VrVr4pidVr, dVx, vrL, vrR, vxL, vxR, vol,
    vth_Dvx, vx_Dvx, vr_Dvr, Vr2Vx2_2D,
    jpa, jpb, jna, jnb) = make_dvr_dvx(vr, vx)

    # Compute vr^2 - 2*vx^2 on the 2D grid
    for i in range(nvr):
        vr2_2vx2_2D[i, :] = vr[i] ** 2 - 2 * vx**2

    # Theta-prime coordinate (for angle integrals)
    ntheta = 5
    dTheta = np.ones(ntheta) / ntheta
    theta = np.pi * (np.arange(ntheta) / ntheta + 0.5 / ntheta)
    cos_theta = np.cos(theta)

    # Determine energy-space differentials
    Eaxis = vth2 * 0.5 * mu * mH * vr**2 / q
    _Eaxis = np.concatenate([Eaxis, [2 * Eaxis[-1] - Eaxis[-2]]])
    Eaxis_mid = np.empty(nvr + 1)
    Eaxis_mid[0] = 0.0
    Eaxis_mid[1:] = 0.5 * (_Eaxis[:-1] + _Eaxis[1:])
    dEaxis = Eaxis_mid[1:] - Eaxis_mid[:-1]

    # Scale input molecular distribution to match desired flux

    gamma_input = 1.0
    if abs(GammaxH2BC) > 0.0:
        gamma_input = vth * np.sum(Vr2pidVr * np.dot(fH2BC_input, vx * dVx))

    ratio = abs(GammaxH2BC) / gamma_input
    fH2BC_input *= ratio

    if abs(ratio - 1.0) > 0.01 * truncate:
        fH2BC = fH2BC_input.copy()

    # Initialize fH2 at x=0 for positive vx
    fH2[:, ip_pos, 0] = fH2BC_input[:, ip_pos]

    # If fH is all zero, turn off H2 <-> H elastic collisions
    if np.sum(fH) <= 0.0:
        H2_H_EL = False

    # Set iteration scheme flags
    fH2_iterate = False
    if (H2_H2_EL or H2_HP_CX or H2_H_EL or H2_P_EL or ni_correct):
        fH2_iterate = True

    fH2_generations = False
    if fH2_iterate:
        fH2_generations = True

    # Check for reuse of previously computed parameters
    New_Grid = True
    if vx_s is not None:
        test = 0
        test += np.sum(vx_s != vx)
        test += np.sum(vr_s != vr)
        test += np.sum(x_s != x)
        test += np.sum(Tnorm_s != Tnorm)
        test += int(mu_s != mu)
        if test <= 0:
            New_Grid = False

    New_Protons = True
    if Ti_s is not None:
        test = 0
        test += np.sum(Ti_s != Ti)
        test += np.sum(n_s != n)
        test += np.sum(vxi_s != vxi)
        if test <= 0:
            New_Protons = False

    New_Electrons = True
    if Te_s is not None:
        test = 0
        test += np.sum(Te_s != Te)
        test += np.sum(n_s != n)
        if test <= 0:
            New_Electrons = False

    New_fH = True
    if fH_s is not None:
        if np.all(fH_s == fH):
            New_fH = False

    New_Simple_CX = True
    if Simple_CX_s is not None:
        if Simple_CX_s == Simple_CX:
            New_Simple_CX = False

    New_H2_Seed = True
    if fH2_s is not None:
        if np.all(fH2_s == fH2):
            New_H2_Seed = False

    New_HP_Seed = True
    if nHP_s is not None:
        test = 0
        test += np.sum(nHP_s != nHP)
        test += np.sum(THP_s != THP)
        if test == 0:
            New_HP_Seed = False

    New_ni_correct = True
    if ni_correct_s is not None:
        if ni_correct_s == ni_correct:
            New_ni_correct = False


    # Determine which parts of the computation need to be (re)computed
    Do_sigv       = New_Grid or New_Electrons
    Do_fH_moments = (New_Grid or New_fH) and (np.sum(fH) > 0.0)
    Do_Alpha_CX   = ((New_Grid or (Alpha_CX is None) or New_HP_Seed or New_Simple_CX) and H2_HP_CX)
    # Do_Alpha_CX is updated in fH2_iteration loop
    Do_SIG_CX     = ((New_Grid or (SIG_CX is None) or New_Simple_CX) and (not Simple_CX) and Do_Alpha_CX)
    Do_Alpha_H2_H = ((New_Grid or (Alpha_H2_H is None) or New_fH) and H2_H_EL)
    Do_SIG_H2_H   = ((New_Grid or (SIG_H2_H is None))and Do_Alpha_H2_H)
    Do_SIG_H2_H2  = ((New_Grid or (SIG_H2_H2 is None))and H2_H2_EL)
    Do_Alpha_H2_P = ((New_Grid or (Alpha_H2_P is None) or New_Protons or New_ni_correct) and H2_P_EL)
    # Do_Alpha_H2_P is updated in fH2_iteration loop
    Do_SIG_H2_P   = ((New_Grid or (SIG_H2_P is None)) and Do_Alpha_H2_P)
    Do_v_v2       = ((New_Grid or (v_v2 is None)) and (CI_Test or Do_SIG_CX or Do_SIG_H2_H or Do_SIG_H2_H2 or Do_SIG_H2_P))

    # Allocate atomic–moment arrays (these will be used if Do_fH_moments is True)
    nH  = np.zeros(nx)
    VxH = np.zeros(nx)
    TH  = np.ones(nx)  # initialize temperatures to 1.0

    if Do_fH_moments:
        if debrief > 1:
            print(prompt + 'Computing vx and T moments of fH')

        # Compute x flow velocity and temperature of atomic species
        for k in range(nx):
            nH[k] = np.sum(Vr2pidVr * np.sum(fH[:, :, k] * dVx[None, :], axis=1))
            if nH[k] > 0.0:
                VxH[k] = (vth * np.sum(Vr2pidVr * np.sum(fH[:, :, k] * (vx * dVx)[None, :], axis=1)) / nH[k])
                for i in range(nvr):
                    vr2vx2_ran2[i, :] = vr[i] ** 2 + (vx - VxH[k] / vth) ** 2

                TH[k] = ((mu * mH) * vth2 * np.sum(Vr2pidVr * np.sum(vr2vx2_ran2 * fH[:, :, k] * dVx[None, :], axis=1)) / (3 * q * nH[k]))

    if New_Grid:
        if debrief > 1:
            print(prompt + 'Computing vr2vx2, vr2vx_vxi2, EH2_P')

        # Magnitude of total normalized v^2 at each mesh point
        vr2vx2 = np.zeros((nvr, nvx, nx))
        for i in range(nvr):
            for k in range(nx):
                vr2vx2[i, :, k] = vr[i] ** 2 + vx ** 2

        # Magnitude of total normalized (v-vxi)^2 at each mesh point
        vr2vx_vxi2 = np.zeros((nvr, nvx, nx))
        for i in range(nvr):
            for k in range(nx):
                vr2vx_vxi2[i, :, k] = vr[i] ** 2 + (vx - vxi[k] / vth) ** 2

        # Molecular hydrogen ion energy in local rest frame of plasma at each mesh point
        EH2_P = (mH * vr2vx_vxi2 * vth2) / q
        EH2_P = np.clip(EH2_P, 0.1, 2.0e4)

        # Compute Maxwellian H2 distribution at the wall temperature
        fw_hat = np.zeros((nvr, nvx))

        # NOTE: Molecular ions have 'normalizing temperature' of 2 Tnorm, i.e., in order to
        # achieve the same thermal velocity^2, a molecular ion distribution has to have twice the temperature 
        # as an atomic ion distribution
        if (np.sum(SH2) > 0.0) or (np.sum(PipeDia) > 0.0):
            if debrief > 1:
                print(prompt + 'Computing fw_Hat')

            vx_shift = np.array([0.0])
            Tmaxwell = np.array([Twall])
            mol = 2  # molecular ion
            _Maxwell = create_shifted_maxwellian(vr, vx, Tmaxwell, vx_shift, mu, mol, Tnorm, debug=debug)
            fw_hat = _Maxwell[:, :, 0]  # extract the (nvr × nvx) slice

    if New_Protons:
        # Compute fi_hat (Maxwellian for protons)
        if debrief > 1:
            print(prompt + 'Computing fi_Hat')

        # create_shifted_maxwellian_core for ionic distribution
        vx_shift = vxi          # array of length nx
        Tmaxwell = Ti           # array of length nx
        mol      = 1            # proton
        # create_shifted_maxwellian_core returns a (nvr × nvx × nx) array
        fi_hat = create_shifted_maxwellian_core(
            vr, vx, vx_shift, Tmaxwell,
            vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
            vx_Dvx, vr_Dvr, Vr2Vx2_2D,
            jpa, jpb, jna, jnb,
            mol, mu, mH, q,
            debug=debug
        )


    if Do_sigv:
        if debrief > 1:
            print(prompt + 'Computing sigv')

        # Compute sigmav rates for each reaction and optionally apply
        # CR model corrections of Sawada        
        sigv = np.zeros((nx, 11))

        # Reaction R1: e + H2 → e + H2(+) + e
        sigv[:, 1] = SigmaV_Ion_HH(Te)
        if Sawada:
            sigv[:, 1] *= 3.7 / 2.0

        # Reaction R2: e + H2 → H(1s) + H(1s)
        sigv[:, 2] = SigmaV_H1s_H1s_HH(Te)
        if Sawada:
            # Construct correction lookup tables
            Te_table = np.log(np.array([5, 20, 100]))
            Ne_table = np.log(np.array([1e14, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22]))
            fctr_Table = np.zeros((7, 3))
            fctr_Table[:, 0] = np.array([2.2, 2.2, 2.1, 1.9, 1.2, 1.1, 1.05]) / 5.3
            fctr_Table[:, 1] = np.array([5.1, 5.1, 4.3, 3.1, 1.5, 1.25, 1.25]) / 10.05
            fctr_Table[:, 2] = np.array([1.3, 1.3, 1.1, 0.8, 0.38, 0.24, 0.22]) / 2.1

            _Te = np.clip(Te, 5.0, 100.0)
            _n  = np.clip(n,  1e14, 1e22)
            # Path_Interp_2D expects log-space inputs
            fctr = path_interp_2d(fctr_Table, Ne_table, Te_table, np.log(_n), np.log(_Te),)
            sigv[:, 2] *= (1.0 + fctr)

        # Reaction R3: e + H2 → e + H(1s) + H*(2s)
        sigv[:, 3] = SigmaV_H1s_H2s_HH(Te)

        # Reaction R4: e + H2 → e + p + H(1s)
        sigv[:, 4] = SigmaV_P_H1s_HH(Te)
        if Sawada:
            sigv[:, 4] *= (1.0 / 0.6)

        # Reaction R5: e + H2 → e + H*(2p) + H*(2s)
        sigv[:, 5] = SigmaV_H2p_H2s_HH(Te)

        # Reaction R6: e + H2 → e + H(1s) + H*(n=3)
        sigv[:, 6] = SigmaV_H1s_Hn3_HH(Te)

        # Reaction R7: e + H2(+) → e + p + H(1s)
        sigv[:, 7] = SigmaV_P_H1s_HP(Te)

        # Reaction R8: e + H2(+) → e + p + H*(n=2)
        sigv[:, 8] = SigmaV_P_Hn2_HP(Te)

        # Reaction R9: e + H2(+) → e + p + p + e
        sigv[:, 9] = SigmaV_P_P_HP(Te)

        # Reaction R10: e + H2(+) → e + H(1s) + H*(n>=2)
        sigv[:, 10] = SigmaV_H1s_Hn_HP(Te)



        # Total H2 destruction rate (normalized by vth) = sum of reactions 1-6
        alpha_loss = np.zeros(nx)
        alpha_loss[:] = n * np.sum(sigv[:, 1:7], axis=1) / vth # NOTE: IDL might have been axis 2. Not sure about the indexing procedures


    # ——— Set up arrays for charge‐exchange & elastic collisions ———
    if Do_v_v2:
        if debrief > 1:
            print(prompt + 'Computing v_v2, v_v, vr2_vx2, and vx_vx')

        # v_v2=(v-v_prime)^2 at each double velocity space mesh point, including theta angle
        v_v2 = np.zeros((nvr, nvx, nvr, nvx, ntheta))

        # vr2_vx2=(vr2 + vr2_prime - 2*vr*vr_prime*cos(theta) - 2*(vx-vx_prime)^2
        # at each double velocity space mesh point, including theta angle
        vr2_vx2 = np.zeros_like(v_v2)
        for m in range(ntheta):
            cos_m = cos_theta[m]
            for l in range(nvx):
                dvx = vx - vx[l]  # shape (nvx,)
                dvx2 = dvx**2
                for k in range(nvr):
                    for i in range(nvr):
                        vr_term = vr[i]**2 + vr[k]**2 - 2 * vr[i] * vr[k] * cos_m
                        # Broadcast along nvx dimension
                        v_v2[i, :, k, l, m] = vr_term + dvx2
                        vr2_vx2[i, :, k, l, m] = vr_term - 2 * dvx2

        # v_v=|v-v_prime| at each double velocity space mesh point, including theta angle
        v_v = np.sqrt(v_v2)



        # vx_vx=(vx-vx_prime) at each double velocity space mesh point
        vx_vx = np.zeros((nvr, nvx, nvr, nvx))
        for j in range(nvx):
            for l in range(nvx):
                vx_vx[:, j, :, l] = vx[j] - vx[l]
        
        # Set Vr'2pidVr'*dVx' for each double velocity space mesh point
        Vr2pidVrdVx = np.zeros((nvr, nvx, nvr, nvx))
        for k in range(nvr):
            Vr2pidVrdVx[:, :, k, :] = Vr2pidVr[k]
        for l in range(nvx):
            Vr2pidVrdVx[:, :, :, l] *= dVx[l]



    if Simple_CX == False and Do_SIG_CX == True:
        if debrief > 1:
            print(f'{prompt}Computing SIG_CX')

        # Option (A) was selected: Compute SigmaV_CX from sigma directly.
        # In preparation, compute SIG_CX for present velocity space grid, if it has not 
        # already been computed with the present input parameters
        
        # Compute sigma_cx * v_v at all possible relative velocities
        energy = v_v2 * (mH * vth**2 / q)
        _Sig = v_v * Sigma_CX_HH(energy)

        integral = np.tensordot(_Sig, dTheta, axes=([-1],[0]))
        SIG_CX_4d = Vr2pidVrdVx * integral
        SIG_CX = SIG_CX_4d.reshape((nvr * nvx, nvr * nvx), order='F')

        # SIG_CX is now vr' * sigma_cx(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])


    if Do_SIG_H2_H == True:
        if debrief > 1:
            print(f'{prompt}Computing SIG_H2_H')

        # Compute SIG_H2_H for present velocity space grid, if it is needed and has not 
        # already been computed with the present input parameters

        # Compute sigma_HH * v_v at all possible relative velocities
        energy = v_v2 * (0.5 * mH * vth**2 / q)
        _Sig = v_v * Sigma_EL_H_HH(energy)

        integral = np.tensordot(_Sig, dTheta, axes=([-1],[0]))
        SIG_H2_H_4d = Vr2pidVrdVx * vx_vx * integral
        SIG_H2_H = SIG_H2_H_4d.reshape((nvr * nvx, nvr * nvx), order='F')

        # SIG_H2_H is now vr' * vx_vx * sigma_H2_H(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])


    if Do_SIG_H2_P == True:
        if debrief > 1:
            print(f'{prompt}Computing SIG_H2_P')
        # Compute SIG_H2_P for present velocity space grid, if it is needed and has not
        # already been computed with the present input parameters

        # Compute sigma_H2_P * v_v at all possible relative velocities
        energy = v_v2 * (0.5 * mH * vth**2 / q)
        _Sig = v_v * Sigma_EL_P_HH(energy)

        integral = np.tensordot(_Sig, dTheta, axes=([-1],[0]))
        SIG_H2_P_4d = Vr2pidVrdVx * vx_vx * integral
        SIG_H2_P = SIG_H2_P_4d.reshape((nvr * nvx, nvr * nvx), order='F')

        # SIG_H2_P is now vr' * vx_vx * sigma_H2_P(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])


    if Do_SIG_H2_H2 == True:
        if debrief > 1:
            print(f'{prompt}Computing SIG_H2_H2')

        # Compute SIG_H2_H2 for present velocity space grid, if it is needed and has not 
        # already been computed with the present input parameters
        
        # Compute sigma_H2_H2 * vr2_vx2 * v_v at all possible relative velocities

        energy = v_v2 * (mH * mu * vth**2 / q)
        _Sig = (vr2_vx2 * v_v * Sigma_EL_HH_HH(energy, vis=True)) / 8.0

        integral = np.tensordot(_Sig, dTheta, axes=([-1],[0]))
        SIG_H2_H2_4d = Vr2pidVrdVx * integral
        SIG_H2_H2 = SIG_H2_H2_4d.reshape((nvr * nvx, nvr * nvx), order='F')

        # SIG_H2_H2 is now vr' * sigma_H2_H2(v_v) * vr2_vx2 * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

    # Compute Alpha_H2_H for inputted fH, if it is needed and has not
    # already been computed with the present input parameters
    if Do_Alpha_H2_H:
        if debrief > 1:
            print(f'{prompt}Computing Alpha_H2_H')

        Alpha_H2_H = np.zeros((nvr, nvx, nx))
        for k in range(nx):
            Work = fH[:, :, k].reshape(nvr * nvx, order='F')
            Alpha_H2_H_flat = np.dot(SIG_H2_H, Work)
            Alpha_H2_H[:, :, k] = Alpha_H2_H_flat.reshape(nvr, nvx, order='F')

            # NOTE: below is the old attempt that did not work
            #Work = fH[:,:,k]
            #Alpha_H2_H [:,:,k] = np.tensordot(SIG_H2_H, Work, axes=([1], [0]))

    # Compute nH2
    for k in range(nx):
        nH2[k] = np.sum(Vr2pidVr[:, None] * fH2[:, :, k] * dVx[None, :])

    if New_H2_Seed:
        MH2_H2_sum = np.zeros((nvr, nvx, nx))
        Delta_nH2s = 1.0

    # Compute Side-Wall collision rate
    gamma_wall = np.zeros((nvr, nvx, nx))
    for k in range(nx):
        if PipeDia[k] > 0.0:
            for j in range(nvx):
                gamma_wall[:, j, k] = 2 * vr / PipeDia[k]

    #breakpoint()

    while True: # this is the equivalent of the fH2_iterate. Instead of the IDL goto command, we can just use the 'continue' statement to return to the start of this loop

        fH2s = fH2.copy()
        nH2s = nH2.copy()
        THPs = THP.copy()
        nHPs = nHP.copy()

        # Compute Alpha_CX for present THP and nHP, if it is needed and has not
        # already been computed with the present parameters

        # ——— Compute Alpha_CX ———
        if Do_Alpha_CX:
            if debrief > 1:
                print(f'{prompt}Computing Alpha_CX')

            # Maxwellian molecular-ion distribution drifting with vxi
            vx_shift = vxi
            Tmaxwell = THP
            mol = 2
            # fHp_hat: shape (nvr, nvx, nx)
            #fHp_hat = create_shifted_maxwellian_core(vr, vx, Tmaxwell, vx_shift, mu, mol, Tnorm, debug=debug)

            fHp_hat = create_shifted_maxwellian_core(
                vr, vx, vx_shift, Tmaxwell,
                vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
                vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                jpa, jpb, jna, jnb,
                mol, mu, mH, q,
                debug=debug
            )

            if Simple_CX:
                # Option (B): Use maxwellian weighted <sigma v>
                # THp/mu at each mesh point
                THp_mu = np.empty((nvr, nvx, nx))
                for k in range(nx):
                    THp_mu[:, :, k] = THP[k] / mu

                # molecular charge-exchange sink rate
                alpha_cx = SigmaV_CX_HH(THp_mu, EH2_P) / vth
                for k in range(nx):
                    alpha_cx[:, :, k] *= nHP[k]

            else:
                # Option (A): Compute SigmaV_CX from sigma directly via SIG_CX
                alpha_cx = np.zeros((nvr, nvx, nx))
                for k in range(nx):
                    Work = (fHp_hat[:, :, k] * nHP[k]).reshape(nvr * nvx, order='F')
                    alpha_cx_flat = np.dot(SIG_CX, Work)
                    alpha_cx[:, :, k] = alpha_cx_flat.reshape(nvr, nvx, order='F')


                    # NOTE: below is the old attempt that did not work
                    #Work = fHp_hat[:, :, k] * nHP[k]
                    #alpha_cx[:, :, k] = np.tensordot(SIG_CX, Work, axes=([1], [0]))


                if Do_Alpha_CX_Test:
                    alpha_cx_test = SigmaV_CX_HH(THp_mu, EH2_P) / vth
                    for k in range(nx):
                        alpha_cx_test[:, :, k] *= nHP[k]
                    print('Compare alpha_cx and alpha_cx_test')

        # Compute Alpha_H2_P for present Ti and ni (optionally correcting for nHP), 
        # if it is needed and has not already been computed with the present parameters
        if Do_Alpha_H2_P:
            if debrief > 1:
                print(f'{prompt}Computing Alpha_H2_P')
            Alpha_H2_P = np.zeros((nvr, nvx, nx))
            ni = n.copy()
            if ni_correct:
                # corrected proton density: n - nHP, floored at zero
                ni = np.where(n - nHP > 0.0, n - nHP, 0.0)
            for k in range(nx):
                Work = (fi_hat[:, :, k] * ni[k]).reshape(nvr*nvx, order='F')
                Alpha_H2_P_flat = np.dot(SIG_H2_P, Work)
                Alpha_H2_P[:, :, k] = Alpha_H2_P_flat.reshape(nvr, nvx, order='F')

                # NOTE: old is below
                # Work = fi_hat[:, :, k] * ni[k]
                # Alpha_H2_P[:, :, k] = np.tensordot(SIG_H2_P, Work, axes=([1], [0]))

        # Compute Omega values if nH is non-zero

        ii = np.where(nH2 <= 0)[0]
        if ii.size == 0:  # Proceed only if all nH > 0
            # recompute VxH2 if any elastic‐collision flags are set
            if H2_P_EL or H2_H_EL or H2_H2_EL:
                for k in range(nx):
                    numerator = np.dot(Vr2pidVr, np.dot(fH2[:, :, k], vx * dVx))
                    VxH2[k] = vth * numerator / nH2[k]

                    # NOTE: below is my old attempt that did not work
                    # VxH2[k] = vth * np.sum(Vr2pidVr * fH2[:, :, k] * (vx * dVx)[None, :]) / nH2[k]

            # Compute Omega_H2_P for present fH2 and Alpha_H2_P if H2_P elastic collisions are included
            if H2_P_EL:
                if debrief > 1:
                    print(f'{prompt}Computing Omega_H2_P')
                for k in range(nx):
                    DeltaVx = (VxH2[k] - vxi[k]) / vth
                    MagDeltaVx = max(abs(DeltaVx), DeltaVx_tol)
                    DeltaVx = np.sign(DeltaVx) * MagDeltaVx
                    Omega_H2_P[k] = np.sum(Vr2pidVr[:, None] * Alpha_H2_P[:, :, k] * fH2[:, :, k] * dVx[None, :]) / (nH2[k] * DeltaVx)

                Omega_H2_P = np.maximum(Omega_H2_P, 0.0)

            # Compute Omega_H2_H for present fH2 and Alpha_H2_H if H2_H elastic collisions are included
            if H2_H_EL:
                if debrief > 1:
                    print(f'{prompt}Computing Omega_H2_H')
                for k in range(nx):
                    DeltaVx = (VxH2[k] - VxH[k]) / vth
                    MagDeltaVx = max(abs(DeltaVx), DeltaVx_tol)
                    DeltaVx = np.sign(DeltaVx) * MagDeltaVx
                    Omega_H2_H[k] = np.sum(Vr2pidVr[:, None] * Alpha_H2_H[:, :, k] * fH2[:, :, k] * dVx[None, :]) / (nH2[k] * DeltaVx)
                Omega_H2_H = np.maximum(Omega_H2_H, 0.0)

            # Compute Omega_H2_H2 for present fH2 if H2_H2 elastic collisions are included
            if H2_H2_EL:
                if debrief > 1:
                    print(f'{prompt}Computing Omega_H2_H2')
                # build Wperp_paraH2
                if np.sum(MH2_H2_sum) <= 0.0:
                    for k in range(nx):
                        vr2_2vx_ran2 = np.zeros((nvr, nvx))
                        for i in range(nvr):
                            vr2_2vx_ran2[i, :] = vr[i]**2 - 2 * (vx - VxH2[k]/vth)**2
                        Wperp_paraH2[k] = np.sum(Vr2pidVr[:, None] * vr2_2vx_ran2 * fH2[:, :, k] * dVx[None, :]) / nH2[k]
                else:
                    for k in range(nx):
                        M_fH2 = MH2_H2_sum[:, :, k] - fH2[:, :, k]
                        Wperp_paraH2[k] = -np.sum(Vr2pidVr[:, None] * vr2_2vx2_2D * M_fH2 * dVx[None, :]) / nH2[k]
                
                # now Omega_H2_H2
                for k in range(nx):
                    Work = fH2[:, :, k]
                    Work_flat = Work.reshape(nvr * nvx, order='F')
                    Alpha_H2_H2_flat = np.dot(SIG_H2_H2, Work_flat)
                    Alpha_H2_H2 = Alpha_H2_H2_flat.reshape(nvr, nvx, order='F')

                    # NOTE: below is the old attempt that did not work
                    # Work = fH2[:, :, k]
                    # Alpha_H2_H2 = np.tensordot(SIG_H2_H2, Work, axes=([1], [0]))


                    Wpp = max(abs(Wperp_paraH2[k]), Wpp_tol)
                    Wpp = np.sign(Wperp_paraH2[k]) * Wpp
                    numerator = np.sum(Vr2pidVr[:, None] * (Alpha_H2_H2 * Work) * dVx[None, :])
                    Omega_H2_H2[k] = numerator / (nH2[k] * Wpp)
                Omega_H2_H2 = np.maximum(Omega_H2_H2, 0.0)

        # Total Elastic scattering frequency
        Omega_EL = Omega_H2_P + Omega_H2_H + Omega_H2_H2

        # Total collision frequency
        alpha_c = np.zeros((nvr, nvx, nx))
        if H2_HP_CX:
            for k in range(nx):
                alpha_c[:, :, k] = (alpha_cx[:, :, k] + alpha_loss[k] + Omega_EL[k] + gamma_wall[:, :, k])
        else:
            for k in range(nx):
                alpha_c[:, :, k] = (alpha_loss[k] + Omega_EL[k] + gamma_wall[:, :, k])


        # Test x grid spacing based on Eq.(27) in notes
        if debrief > 1:
            print(f'{prompt}Testing x grid spacing')
        Max_dx = np.full(nx, 1.0e32)
        for k in range(nx):
            for j in range(ip_pos[0], nvx):
                with np.errstate(divide='ignore', invalid='ignore'):
                    local_dx = 2.0 * vx[j] / alpha_c[:, j, k]
                local_dx = local_dx[np.isfinite(local_dx) & (local_dx > 0)]
                if local_dx.size > 0:
                    Max_dx[k] = min(Max_dx[k], np.min(local_dx))

        dx = np.roll(x, -1) - x  # Compute mesh spacing
        Max_dxL = Max_dx[:-1]
        Max_dxR = Max_dx[1:]
        Max_dx_comp = np.minimum(Max_dxL, Max_dxR)

        ilarge = np.where(Max_dx_comp < dx[:-1])[0]
        if ilarge.size > 0:
            print(f'{prompt}x mesh spacing is too large!')
            debug = True
            print('   x(k+1)-x(k)  Max_dx(k)   x(k+1)-x(k)  Max_dx(k)   x(k+1)-x(k)  Max_dx(k)   x(k+1)-x(k)  Max_dx(k)   x(k+1)-x(k)  Max_dx(k)')
            out = ''
            for jj, ii in enumerate(ilarge, 1):
                out += f'({ii:3}) {x[ii+1] - x[ii]:9.2e} {Max_dx_comp[ii]:9.2e}   '
                if jj % 5 == 0:
                    print(out)
                    out = ''
            if out:
                print(out)
            error = True
            #print(' USUALLY THIS WOULD THROW AN ERROR, BUT THE SAME ERROR APPEARS IN THE IDL, SO I AM JUST IGNORING IT FOR NOW')
            #print('REMEMBER TO CHANGE BACK ONCE THE CODE HAS BEEN VALIDATED. ALSO FIX THE ORIGINAL IDL TOO!')
            #input("Press Enter to continue…")
            raise RuntimeError('Aborting due to x mesh spacing error')


        # ——— Define transport coefficients Ak…Gk ———
        Ak = np.zeros((nvr, nvx, nx))
        Bk = np.zeros((nvr, nvx, nx))
        Ck = np.zeros((nvr, nvx, nx))
        Dk = np.zeros((nvr, nvx, nx))
        Fk = np.zeros((nvr, nvx, nx))
        Gk = np.zeros((nvr, nvx, nx))

        # forward (vx>0) sweep
        for k in range(nx-1):
            dx = x[k+1] - x[k]
            for j in range(ip_pos[0], nvx):
                denom = 2*vx[j] + dx*alpha_c[:, j, k+1]
                Ak[:, j, k] = (2*vx[j] - dx*alpha_c[:, j, k]) / denom
                Bk[:, j, k] = dx / denom
                Fk[:, j, k] = dx * fw_hat[:, j] * (SH2[k+1] + SH2[k]) / (vth * denom)

        # backward (vx<0) sweep
        for k in range(1, nx):
            dx = x[k] - x[k-1]
            for j in range(0, ip_pos[0]):
                denom = -2*vx[j] + dx*alpha_c[:, j, k-1]
                Ck[:, j, k] = (-2*vx[j] - dx*alpha_c[:, j, k]) / denom
                Dk[:, j, k] = dx / denom
                Gk[:, j, k] = dx * fw_hat[:, j] * (SH2[k] + SH2[k-1]) / (vth * denom)


        # ——— First‐flight (0th generation) molecular neutral function ———
        Swall_sum    = np.zeros((nvr, nvx, nx))
        Beta_CX_sum  = np.zeros((nvr, nvx, nx))
        MH2_P_sum    = np.zeros((nvr, nvx, nx))
        MH2_H_sum    = np.zeros((nvr, nvx, nx))
        MH2_H2_sum   = np.zeros((nvr, nvx, nx))
        igen = 0
        if debrief > 0:
            print(f'{prompt}Computing molecular neutral generation#{igen}')

        # seed fH2G at x=0 for positive vx
        fH2G[:, ip_pos, 0] = fH2[:, ip_pos, 0]

        # sweep forward
        for k in range(nx-1):
            fH2G[:, ip_pos, k+1] = fH2G[:, ip_pos, k]*Ak[:, ip_pos, k] + Fk[:, ip_pos, k]
            #if k == 70:
                #breakpoint()
        # sweep backward
        for k in reversed(range(1, nx)):
            fH2G[:, in_neg, k-1] = fH2G[:, in_neg, k]*Ck[:, in_neg, k] + Gk[:, in_neg, k]


        # first‐flight density profile
        for k in range(nx):
            NH2G[k, igen] = np.sum(Vr2pidVr[:, None] * fH2G[:, :, k] * dVx[None, :])


        # optional plot
        if plot > 1:
            fH21d = np.zeros((nvx, nx))
            for k in range(nx):
                fH21d[:, k] = np.tensordot(Vr2pidVr, fH2G[:, :, k], axes=1)
            plt.figure()
            plt.title(f'First Generation {_HH}')
            plt.ylim(0, np.max(fH21d))
            for i in range(nx):
                plt.plot(vx, fH21d[:, i], color=f'C{(i%8)+2}')
            if debug > 0:
                input("Press Return to continue...")
            plt.show()

        # Set total molecular neutral distribution function to first flight generation
        fH2 = fH2G.copy()
        nH2 = NH2G[:, 0].copy()

        # if no further generations requested, skip to done
        if fH2_generations == False:
            pass   # this is the equivalent of the IDL goto,fH_done command. We will exit the loop and proceed to the fH_done section
            # NOTE: used to be break but I think this works
        else:

            # Now we enter the next_generation loop from the IDL (line 1178 in the original code).
            while True:
                if igen + 1 > Max_Gen:
                    if debrief > 0:
                        print(f'{prompt}Completed {Max_Gen} generations. Returning present solution...')
                    break
                igen += 1
                if debrief > 0:
                    print(f'{prompt}Computing molecular neutral generation#{igen}')

                # —— Swall from side‐wall collisions ——
                Swall = np.zeros((nvr, nvx, nx))
                if np.sum(gamma_wall) > 0:
                    if debrief > 1:
                        print(f'{prompt}Computing Swall')
                    for k in range(nx):
                        integrated = np.sum(Vr2pidVr[:, None] * (gamma_wall[:, :, k] * fH2G[:, :, k]) * dVx[None, :])
                        Swall[:, :, k] = fw_hat * integrated
                    Swall_sum += Swall

                # —— Beta_CX from charge‐exchange ——  
                Beta_CX = np.zeros((nvr, nvx, nx)) 
                if H2_HP_CX:
                    if debrief > 1:
                        print(f'{prompt}Computing Beta_CX')
                    if Simple_CX:
                        # Option (B): Compute charge exchange source with assumption that CX source neutrals have
                        # molecular ion distribution function
                        for k in range(nx):
                            integ = np.sum(Vr2pidVr[:, None] * (alpha_cx[:, :, k] * fH2G[:, :, k]) * dVx[None, :])
                            Beta_CX[:, :, k] = fHp_hat[:, :, k] * integ
                    else:
                        # Option (A): Compute charge exchange source using fH2 and vr x sigma x v_v at each velocity mesh point
                        for k in range(nx):
                            Work = fH2G[:, :, k].reshape(nvr * nvx, order='F')
                            Beta_CX_flat = np.dot(SIG_CX, Work)
                            Beta_CX[:,:,k] = nHP[k] * fHp_hat[:, :, k] * Beta_CX_flat.reshape(nvr, nvx, order='F')

                            # NOTE: below is the old attempt that did not work
                            # Work = fH2G[:, :, k]
                            # Beta_CX[:, :, k] = nHP[k] * fHp_hat[:, :, k] * np.tensordot(SIG_CX, Work, axes=1)
                    Beta_CX_sum += Beta_CX

                # ——— Compute MH2 from previous generation ———
                MH2_H2 = np.zeros((nvr, nvx, nx))
                MH2_P  = np.zeros_like(MH2_H2)
                MH2_H  = np.zeros_like(MH2_H2)
                OmegaM = np.zeros_like(MH2_H2)

                if H2_H2_EL or H2_P_EL or H2_H_EL:
                    # Compute VxH2G and TH2G
                    for k in range(nx):
                        VxH2G[k] = (
                            vth
                            * np.sum(Vr2pidVr[:, None] * fH2G[:, :, k] * (vx * dVx)[None, :])
                            / NH2G[k, igen - 1]
                        )
                        for i in range(nvr):
                            vr2vx2_ran2[i, :] = vr[i]**2 + (vx - VxH2G[k]/vth)**2
                        TH2G[k] = (
                            2 * mu * mH * vth2
                            * np.sum(Vr2pidVr[:, None] * (vr2vx2_ran2 * fH2G[:, :, k]) * dVx[None, :])
                            / (3 * q * NH2G[k, igen - 1])
                        )

                    if H2_H2_EL:
                        # Compute MH2_H2
                        if debrief > 1:
                            print(f'{prompt}Computing MH2_H2')
                        vx_shift = VxH2G.copy()
                        Tmaxwell = TH2G.copy()
                        mol = 2

                        Maxwell = create_shifted_maxwellian_core(
                            vr, vx, vx_shift, Tmaxwell,
                            vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
                            vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                            jpa, jpb, jna, jnb,
                            mol, mu, mH, q,
                            debug=debug
                        )

                        for k in range(nx):
                            MH2_H2[:, :, k] = Maxwell[:, :, k] * NH2G[k, igen - 1]
                            OmegaM[:, :, k] += Omega_H2_H2[k] * MH2_H2[:, :, k]
                        MH2_H2_sum += MH2_H2

                    if H2_P_EL:
                        # Compute MH2_P
                        if debrief > 1:
                            print(f'{prompt}Computing MH2_P')
                        vx_shift = (2*VxH2G + vxi) / 3
                        Tmaxwell = TH2G + (4./9.) * (Ti - TH2G + mu*mH*(vxi - VxH2G)**2 / (6*q))
                        mol = 2

                        Maxwell = create_shifted_maxwellian_core(
                            vr, vx, vx_shift, Tmaxwell,
                            vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
                            vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                            jpa, jpb, jna, jnb,
                            mol, mu, mH, q,
                            debug=debug
                        )

                        for k in range(nx):
                            MH2_P[:, :, k] = Maxwell[:, :, k] * NH2G[k, igen - 1]
                            OmegaM[:, :, k] += Omega_H2_P[k] * MH2_P[:, :, k]
                        MH2_P_sum += MH2_P

                    if H2_H_EL:
                        # Compute MH2_H
                        if debrief > 1:
                            print(f'{prompt}Computing MH2_H')
                        vx_shift = (2*VxH2G + VxH) / 3
                        Tmaxwell = TH2G + (4./9.) * (TH - TH2G + mu*mH*(VxH - VxH2G)**2 / (6*q))
                        mol = 2

                        Maxwell = create_shifted_maxwellian_core(
                            vr, vx, vx_shift, Tmaxwell,
                            vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
                            vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                            jpa, jpb, jna, jnb,
                            mol, mu, mH, q,
                            debug=debug
                        )

                        for k in range(nx):
                            MH2_H[:, :, k] = Maxwell[:, :, k] * NH2G[k, igen - 1]
                            OmegaM[:, :, k] += Omega_H2_H[k] * MH2_H[:, :, k]
                        MH2_H_sum += MH2_H

                # ——— Compute next-generation molecular distribution ———
                fH2G.fill(0.0)
                for k in range(nx - 1):
                    fH2G[:, ip_pos, k+1] = (
                        Ak[:, ip_pos, k] * fH2G[:, ip_pos, k]
                        + Bk[:, ip_pos, k] * (
                            Swall[:, ip_pos, k+1] + Beta_CX[:, ip_pos, k+1]
                            + OmegaM[:, ip_pos, k+1] + Swall[:, ip_pos, k]
                            + Beta_CX[:, ip_pos, k] + OmegaM[:, ip_pos, k]
                        )
                    )
                for k in range(nx - 1, 0, -1):
                    fH2G[:, in_neg, k-1] = (
                        Ck[:, in_neg, k] * fH2G[:, in_neg, k]
                        + Dk[:, in_neg, k] * (
                            Swall[:, in_neg, k-1] + Beta_CX[:, in_neg, k-1]
                            + OmegaM[:, in_neg, k-1] + Swall[:, in_neg, k]
                            + Beta_CX[:, in_neg, k] + OmegaM[:, in_neg, k]
                        )
                    )

                # Update density profile
                for k in range(nx):
                    NH2G[k, igen] = np.sum(Vr2pidVr[:, None] * fH2G[:, :, k] * dVx[None, :])

                # Optional plotting
                if plot > 1:
                    fH21d = np.zeros((nvx, nx))
                    for k in range(nx):
                        fH21d[:, k] = np.tensordot(Vr2pidVr, fH2G[:, :, k], axes=1)
                    plt.figure()
                    plt.title(f'{igen} Generation {_HH}')
                    plt.ylim(0, np.max(fH21d))
                    for i in range(nx):
                        plt.plot(vx, fH21d[:, i], color=f'C{(i%8)+2}')
                    if debug > 0:
                        input("Press Return to continue...")
                    plt.show()

                # Add result to total neutral distribution function
                fH2 += fH2G
                nH2 += NH2G[:, igen]

                # Compute 'generation error': Delta_nH2G=max(NH2G(*,igen)/max(nH2))
                # and decide if another generation should be computed
                Delta_nH2G = np.max(NH2G[:, igen] / np.max(nH2))
                if fH2_iterate:
                    #print('generation error breakpoint: ', igen)
                    #breakpoint()
                    if (Delta_nH2G < 0.003 * Delta_nH2s) or (Delta_nH2G < truncate):
                        print('generation error breakpoint: ', igen)
                        # If fH2 'seed' is being iterated, then do another generation until the 'generation error'
                        # is less than 0.003 times the 'seed error' or is less than TRUNCATE
                        break
                else:
                    if Delta_nH2G < truncate:
                        # If fH2 'seed' is NOT being iterated, then do another generation unitl the 'generation error'
                        # is less than parameter TRUNCATE
                        break

        # Now we are in the fH2_done part of the loop. This is outside the next_generation loop but inside the fH_iterate loop.
        # This means we have the option to either go back to the start of the fH_iterate loop or exit to the end of the function (just like the IDL version with the goto commands).  
        if plot > 0:
            plt.figure()
            plt.yscale('log')
            #plt.ylim(np.max(NH2G) * truncate, np.max(NH2G))
            plt.xlim(0, 0.25)
            plt.ylim(1e16, 1e21)
            plt.title(f'{_HH} Density by Generation')
            plt.xlabel('x (m)')
            plt.ylabel('Density (m⁻³)')
            for i in range(igen+1):
                plt.plot(x, NH2G[:, i], color=f'C{i%8}')
            if pause:
                input("Press Return to continue...")
            plt.show()
            #breakpoint()


        # Compute final density, fluxes, and moments
        for k in range(nx):
            nH2[k] = np.sum(Vr2pidVr[:, None] * fH2[:, :, k] * dVx[None, :])
            GammaxH2[k] = vth * np.sum(Vr2pidVr[:, None] * fH2[:, :, k] * (vx * dVx)[None, :])
        VxH2 = GammaxH2 / nH2
        _VxH2 = VxH2 / vth

        # magnitude of random velocity at each mesh point
        vr2vx2_ran = np.zeros((nvr, nvx, nx))
        for i in range(nvr):
            for k in range(nx):
                vr2vx2_ran[i, :, k] = vr[i]**2 + (vx - _VxH2[k])**2

        # Pressure and temperature
        for k in range(nx):
            pH2[k] = (
                2 * mu * mH * vth2
                * np.sum(Vr2pidVr[:, None] * (vr2vx2_ran[:, :, k] * fH2[:, :, k]) * dVx[None, :])
                / (3 * q)
            )
        TH2 = pH2 / nH2

        # Dissociation & equilibration frequencies
        NuDis = n * np.sum(sigv[:, 7:11], axis=1)
        NuE   = 7.7e-7 * n * 1e-6 / (np.sqrt(mu) * Ti**1.5) # Energy equilibration frequency H(+) <-> H2(+)

        # Molecular ion density and temperature
        nHP = nH2 * n * sigv[:, 1] / (NuDis + NuLoss)
        THP = Ti * NuE / (NuE + NuDis + NuLoss)

        if fH2_iterate:
            # Compute 'seed error': Delta_nH2s=(|nH2s-nH2|)/max(nH2) 
            # If Delta_nH2s is greater than 10*truncate then iterate fH2
            Delta_nH2s = np.max(np.abs(nH2s - nH2)) / np.max(nH2)
            if Delta_nH2s > 10 * truncate:
                continue # this continue command sends us back to the start of the fH2_iterate 'while True' loop.

        # Update Swall and Beta_CX sums from last generation
        Swall = np.zeros((nvr, nvx, nx))
        if np.sum(gamma_wall) > 0:
            for k in range(nx):
                integ = np.sum(Vr2pidVr[:, None] * (gamma_wall[:, :, k] * fH2G[:, :, k]) * dVx[None, :])
                Swall[:, :, k] = fw_hat * integ
            Swall_sum += Swall

        Beta_CX = np.zeros((nvr, nvx, nx))
        if H2_HP_CX:
            if debrief > 1:
                print(f'{prompt}Computing Beta_CX')
            if Simple_CX:
                # Option (B): Compute charge exchange source with assumption that CX source neutrals have
                # molecular ion distribution function
                for k in range(nx):
                    integ = np.sum(Vr2pidVr[:, None] * (alpha_cx[:, :, k] * fH2G[:, :, k]) * dVx[None, :])
                    Beta_CX[:, :, k] = fHp_hat[:, :, k] * integ
            else:
                # Option (A): Compute charge exchange source using fH and vr x sigma x v_v at each velocity mesh point
                for k in range(nx):
                    Work = fH2G[:, :, k].reshape(nvr * nvx, order='F')
                    signal_2D = (SIG_CX @ Work).reshape(nvr, nvx, order='F')
                    Beta_CX[:, :, k] = nHP[k] * fHp_hat[:, :, k] * signal_2D

                    # NOTE: below is the old attempt that did not work
                    #Work = fH2G[:, :, k]
                    #Beta_CX[:, :, k] = nHP[k] * fHp_hat[:, :, k] * np.tensordot(SIG_CX @ Work)

            Beta_CX_sum += Beta_CX

        # Update MH2_*_sum using last generation (same pattern as above)
        MH2_H2 = np.zeros((nvr, nvx, nx))
        MH2_P  = np.zeros_like(MH2_H2)
        MH2_H  = np.zeros_like(MH2_H2)
        OmegaM = np.zeros_like(MH2_H2)
        if H2_H2_EL or H2_P_EL or H2_H_EL:
            for k in range(nx):
                VxH2G[k] = (
                    vth
                    * np.sum(Vr2pidVr[:, None] * fH2G[:, :, k] * (vx * dVx)[None, :])
                    / NH2G[k, igen]
                )
                for i in range(nvr):
                    vr2vx2_ran2[i, :] = vr[i]**2 + (vx - VxH2G[k]/vth)**2
                TH2G[k] = (
                    2 * mu * mH * vth2
                    * np.sum(Vr2pidVr[:, None] * (vr2vx2_ran2 * fH2G[:, :, k]) * dVx[None, :])
                    / (3 * q * NH2G[k, igen])
                )

            if H2_H2_EL:
                # Compute MH2_H2
                if debrief > 1:
                    print(f'{prompt}Computing MH2_H2')
                vx_shift  = VxH2G.copy()
                Tmaxwell  = TH2G.copy()
                mol        = 2

                Maxwell = create_shifted_maxwellian_core(
                    vr, vx, vx_shift, Tmaxwell,
                    vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
                    vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                    jpa, jpb, jna, jnb,
                    mol, mu, mH, q,
                    debug=debug
                )
                for k in range(nx):
                    MH2_H2[:, :, k] = Maxwell[:, :, k] * NH2G[k, igen]
                    OmegaM[:, :, k] += Omega_H2_H2[k] * MH2_H2[:, :, k]
                MH2_H2_sum += MH2_H2

            if H2_P_EL:
                # Compute MH2_P
                if debrief > 1:
                    print(f'{prompt}Computing MH2_P')
                vx_shift  = (2*VxH2G + vxi) / 3
                Tmaxwell  = TH2G + (4./9.) * (Ti - TH2G + mu*mH*(vxi - VxH2G)**2/(6*q))
                mol        = 2

                Maxwell = create_shifted_maxwellian_core(
                    vr, vx, vx_shift, Tmaxwell,
                    vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
                    vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                    jpa, jpb, jna, jnb,
                    mol, mu, mH, q,
                    debug=debug
                )
                for k in range(nx):
                    MH2_P[:, :, k] = Maxwell[:, :, k] * NH2G[k, igen]
                    OmegaM[:, :, k] += Omega_H2_P[k] * MH2_P[:, :, k]
                MH2_P_sum += MH2_P

            if H2_H_EL:
                # Compute MH2_H
                if debrief > 1:
                    print(f'{prompt}Computing MH2_H')
                vx_shift  = (2*VxH2G + VxH) / 3
                Tmaxwell  = TH2G + (4./9.) * (TH - TH2G + mu*mH*(VxH - VxH2G)**2/(6*q))
                mol        = 2

                Maxwell = create_shifted_maxwellian_core(
                    vr, vx, vx_shift, Tmaxwell,
                    vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
                    vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                    jpa, jpb, jna, jnb,
                    mol, mu, mH, q,
                    debug=debug
                )
                for k in range(nx):
                    MH2_H[:, :, k] = Maxwell[:, :, k] * NH2G[k, igen]
                    OmegaM[:, :, k] += Omega_H2_H[k] * MH2_H[:, :, k]
                MH2_H_sum += MH2_H


        # Compute remaining moments
        for k in range(nx):
            piH2_xx[k] = (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * (fH2[:, :, k] @ (dVx * (vx - _VxH2[k])**2))) / q - pH2[k]

        for k in range(nx):
            piH2_yy[k] = (2 * mu * mH) * vth2 * 0.5 * np.sum((Vr2pidVr * vr**2)[:, None] * (fH2[:, :, k] @ dVx)) / q - pH2[k]

        piH2_zz = piH2_yy.copy()

        for k in range(nx):
            qxH2[k] = 0.5 * (2 * mu * mH) * vth**3 * np.sum(Vr2pidVr * ((vr2vx2_ran[:, :, k] * fH2[:, :, k]) @ (dVx * (vx - _VxH2[k]))))

        # C = RHS of Boltzman equation for total fH2
        for k in range(nx):
            C = vth * (
                fw_hat * SH2[k] / vth +
                Swall_sum[:, :, k] +
                Beta_CX_sum[:, :, k] -
                alpha_c[:, :, k] * fH2[:, :, k] +
                Omega_H2_P[k] * MH2_P_sum[:, :, k] +
                Omega_H2_H[k] * MH2_H_sum[:, :, k] +
                Omega_H2_H2[k] * MH2_H2_sum[:, :, k]
            )

            QH2[k] = 0.5 * (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2_ran[:, :, k] * C) @ dVx))
            RxH2[k] = (2 * mu * mH) * vth * np.sum(Vr2pidVr * (C @ (dVx * (vx - _VxH2[k]))))
            Sloss[k] = -np.sum(Vr2pidVr * (C @ dVx)) + SH2[k]
            WallH2[k] = np.sum(Vr2pidVr * ((gamma_wall[:, :, k] * fH2[:, :, k]) @ dVx))

            if H2_H_EL:
                CH2_H = vth * Omega_H2_H[k] * (MH2_H_sum[:, :, k] - fH2[:, :, k])
                RxH_H2[k] = (2 * mu * mH) * vth * np.sum(Vr2pidVr * (CH2_H @ (dVx * (vx - _VxH2[k]))))
                EH_H2[k] = 0.5 * (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * CH2_H) @ dVx))

            if H2_P_EL:
                CH2_P = vth * Omega_H2_P[k] * (MH2_P_sum[:, :, k] - fH2[:, :, k])
                RxP_H2[k] = (2 * mu * mH) * vth * np.sum(Vr2pidVr * (CH2_P @ (dVx * (vx - _VxH2[k]))))
                EP_H2[k] = 0.5 * (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * CH2_P) @ dVx))

            if H2_HP_CX:
                CH2_HP_CX = vth * (Beta_CX_sum[:, :, k] - alpha_cx[:, :, k] * fH2[:, :, k])
                RxH2CX[k] = (2 * mu * mH) * vth * np.sum(Vr2pidVr * (CH2_HP_CX @ (dVx * (vx - _VxH2[k]))))
                EH2CX[k] = 0.5 * (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * CH2_HP_CX) @ dVx))

            CW_H2 = vth * (Swall_sum[:, :, k] - gamma_wall[:, :, k] * fH2[:, :, k])
            RxW_H2[k] = (2 * mu * mH) * vth * np.sum(Vr2pidVr * (CW_H2 @ (dVx * (vx - _VxH2[k]))))
            EW_H2[k] = 0.5 * (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * CW_H2) @ dVx))

            if H2_H2_EL:
                CH2_H2 = vth * Omega_H2_H2[k] * (MH2_H2_sum[:, :, k] - fH2[:, :, k])
                for i in range(nvr):
                    vr2_2vx_ran2[i, :] = vr[i]**2 - 2 * (vx - _VxH2[k])**2
                Epara_PerpH2_H2[k] = -0.5 * (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2_2vx_ran2 * CH2_H2) @ dVx))

        # qxH2_total
        qxH2_total = (0.5 * nH2 * (2 * mu * mH) * VxH2**2 + 2.5 * pH2 * q) * VxH2 + q * piH2_xx * VxH2 + qxH2

        # QH2_total
        QH2_total = QH2 + RxH2 * VxH2 - 0.5 * (2 * mu * mH) * (Sloss - SH2) * VxH2**2

        # Albedo
        AlbedoH2 = 0.0
        gammax_plus = vth * np.sum(Vr2pidVr * (fH2[:, ip_pos, 0] @ (vx[ip_pos] * dVx[ip_pos])))
        gammax_minus = vth * np.sum(Vr2pidVr * (fH2[:, in_neg, 0] @ (vx[in_neg] * dVx[in_neg])))
        if abs(gammax_plus) > 0:
            AlbedoH2 = -gammax_minus / gammax_plus

        # Compute Mesh Errors
        mesh_error = np.zeros((nvr, nvx, nx))
        max_mesh_error = 0.0
        min_mesh_error = 0.0
        mtest = 5
        moment_error = np.zeros((nx, mtest))
        max_moment_error = np.zeros(mtest)
        C_error = np.zeros(nx)
        CX_error = np.zeros(nx)
        Wall_error = np.zeros(nx)
        H2_H2_error = np.zeros((nx, 3))
        H2_H_error = np.zeros((nx, 3))
        H2_P_error = np.zeros((nx, 3))
        max_H2_H2_error = np.zeros(3)
        max_H2_H_error = np.zeros(3)
        max_H2_P_error = np.zeros(3)

        if compute_errors:
            if debrief > 1:
                print(prompt + 'Computing Collision Operator, Mesh, and Moment Normalized Errors')

            Sloss2 = vth * alpha_loss * nH2
            C_error = np.abs(Sloss - Sloss2) / np.maximum(np.abs(Sloss), np.abs(Sloss2))

            # Test conservation of particles for charge exchange operator
            if H2_HP_CX:
                for k in range(nx):
                    CX_A = np.sum(Vr2pidVr[:, None] * (alpha_cx[:, :, k] * fH2[:,:,k]) * dVx[None, :])
                    CX_B = np.sum(Vr2pidVr[:, None] * (Beta_CX_sum[:, :, k] * dVx[None, :]))
                    CX_error[k] = abs(CX_A - CX_B) / max(abs(CX_A), abs(CX_B))

            # Test conservation of particles for wall collision operator
            if np.any(PipeDia > 0):
                for k in range(nx):
                    Wall_A = WallH2[k]
                    Wall_B = np.sum(Vr2pidVr * (Swall_sum[:, :, k] @ dVx))
                    denom = max(abs(Wall_A), abs(Wall_B))
                    if denom > 0:
                        Wall_error[k] = abs(Wall_A - Wall_B) / denom

            # Test conservation of particles, x momentum, and total energy of elastic collision operators
            for m in range(3):
                for k in range(nx):
                    if m < 2:
                        TfH2 = np.sum(Vr2pidVr * (fH2[:, :, k] @ (dVx * vx**m)))
                    else:
                        TfH2 = np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * fH2[:, :, k]) @ dVx))

                    if H2_H2_EL:
                        if m < 2:
                            TH2_H2 = np.sum(Vr2pidVr * (MH2_H2_sum[:, :, k] @ (dVx * vx**m)))
                        else:
                            TH2_H2 = np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * MH2_H2_sum[:, :, k]) @ dVx))
                        H2_H2_error[k, m] = abs(TfH2 - TH2_H2) / max(abs(TfH2), abs(TH2_H2))

                    if H2_H_EL:
                        if m < 2:
                            TH2_H = np.sum(Vr2pidVr * (MH2_H_sum[:, :, k] @ (dVx * vx**m)))
                        else:
                            TH2_H = np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * MH2_H_sum[:, :, k]) @ dVx))
                        H2_H_error[k, m] = abs(TfH2 - TH2_H) / max(abs(TfH2), abs(TH2_H))

                    if H2_P_EL:
                        if m < 2:
                            TH2_P = np.sum(Vr2pidVr * (MH2_P_sum[:, :, k] @ (dVx * vx**m)))
                        else:
                            TH2_P = np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * MH2_P_sum[:, :, k]) @ dVx))
                        H2_P_error[k, m] = abs(TfH2 - TH2_P) / max(abs(TfH2), abs(TH2_P))

                max_H2_H2_error[m] = np.max(H2_H2_error[:, m])
                max_H2_H_error[m] = np.max(H2_H_error[:, m])
                max_H2_P_error[m] = np.max(H2_P_error[:, m])

            if CI_Test:
                minRx = 1.0e-6
                minEpara_perp = 1.0e-6

                # Compute Momentum transfer rate via full collision integrals for charge exchange and mixed elastic scattering
                # Then compute error between this and actual momentum transfer resulting from CX and BKG (elastic) models
                if H2_HP_CX:
                    print(f"{prompt}Computing H2(+) -> H2 Charge Exchange Momentum Transfer")
                    _Sig = v_v * Sigma_CX_HH(v_v2 * (mH * vth2 / q))
                    sig_4D = np.dot(_Sig, dTheta).reshape((nvr, nvx, nvr, nvx), order='F')
                    SIG_VX_CX_4D = Vr2pidVrdVx * vx_vx * sig_4D
                    SIG_VX_CX = SIG_VX_CX_4D.reshape((nvr * nvx, nvr * nvx), order='F')

                    # NOTE: below is my old attempt that did not work
                    # SIG_VX_CX = Vr2pidVrdVx * (vx_vx @ (_Sig @ dTheta))

                    alpha_vx_cx = np.zeros((nvr, nvx, nx))
                    for k in range(nx):
                        Work = (nHP[k] * fHp_hat[:, :, k]).reshape(nvr*nvx, order='F')
                        alpha_vx_cx_flat = np.dot(SIG_VX_CX, Work)
                        alpha_vx_cx[:, :, k] = alpha_vx_cx_flat.reshape((nvr, nvx), order='F')
                        
                        # NOTE: below is the old attempt that did not work
                        #Work = nHP[k] * fHp_hat[:, :, k]
                        #alpha_vx_cx[:, :, k] = SIG_VX_CX @ Work

                    for k in range(nx):
                        RxCI_CX[k] = -(2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((alpha_vx_cx[:, :, k] * fH2[:, :, k]) @ dVx))

                    norm = np.max(np.abs(np.array([RxH2CX, RxCI_CX])))
                    CI_CX_error = np.abs(RxH2CX - RxCI_CX) / norm
                    print(f"{prompt}Maximum normalized momentum transfer error in CX collision operator: {np.max(CI_CX_error)}")

                # P -> H2 elastic BKG
                if H2_P_EL:
                    for k in range(nx):
                        RxCI_P_H2[k] = -(1.0 / 3.0) * (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((Alpha_H2_P[:, :, k] * fH2[:, :, k]) @ dVx))

                    norm = np.max(np.abs(np.array([RxP_H2, RxCI_P_H2])))
                    CI_P_H2_error = np.abs(RxP_H2 - RxCI_P_H2) / norm
                    print(f"{prompt}Maximum normalized momentum transfer error in P -> H2 elastic BKG collision operator: {np.max(CI_P_H2_error)}")

                # H -> H2 elastic BKG
                if H2_H_EL:
                    for k in range(nx):
                        RxCI_H_H2[k] = -(1.0 / 3.0) * (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * ((Alpha_H2_H[:, :, k] * fH2[:, :, k]) @ dVx))

                    norm = np.max(np.abs(np.array([RxH_H2, RxCI_H_H2])))
                    CI_H_H2_error = np.abs(RxH_H2 - RxCI_H_H2) / norm
                    print(f"{prompt}Maximum normalized momentum transfer error in H -> H2 elastic BKG collision operator: {np.max(CI_H_H2_error)}")

                # H2 -> H2 parallel/perpendicular energy exchange
                if H2_H2_EL:
                    for k in range(nx):
                        Work = fH2[:, :, k].reshape(nvr*nvx, order='F')
                        Alpha_H2_H2_flat = np.dot(SIG_H2_H2, Work)
                        Alpha_H2_H2 = Alpha_H2_H2_flat.reshape((nvr, nvx), order='F')
                        Epara_Perp_CI[k] = 0.5 * (2 * mu * mH) * vth3 * np.sum(Vr2pidVr * ((Alpha_H2_H2 * fH2[:, :, k]) @ dVx))
                        
                        # NOTE: below is the old code that did not work
                        #Work = fH2[:, :, k]
                        #Alpha_H2_H2 = SIG_H2_H2 @ Work
                        #Epara_Perp_CI[k] = 0.5 * (2 * mu * mH) * vth3 * np.sum(Vr2pidVr * ((Alpha_H2_H2 * fH2[:, :, k]) @ dVx))

                    norm = np.max(np.abs(np.array([Epara_PerpH2_H2, Epara_Perp_CI])))
                    CI_H2_H2_error = np.abs(Epara_PerpH2_H2 - Epara_Perp_CI) / norm
                    print(f"{prompt}Maximum normalized perp/parallel energy transfer error in H2 -> H2 elastic BKG collision operator: {np.max(CI_H2_H2_error)}")

            # Mesh Point Error based on fH2 satisfying Boltzmann equation
            T1 = np.zeros((nvr, nvx, nx))
            T2 = np.zeros_like(T1)
            T3 = np.zeros_like(T1)
            T4 = np.zeros_like(T1)
            T5 = np.zeros_like(T1)
            T6 = np.zeros_like(T1)

            for k in range(nx - 1):
                for j in range(nvx):
                    T1[:, j, k] = 2 * vx[j] * (fH2[:, j, k + 1] - fH2[:, j, k]) / (x[k + 1] - x[k])
                T2[:, :, k] = fw_hat * (SH2[k + 1] + SH2[k]) / vth
                T3[:, :, k] = Beta_CX_sum[:, :, k + 1] + Beta_CX_sum[:, :, k]
                T4[:, :, k] = alpha_c[:, :, k + 1] * fH2[:, :, k + 1] + alpha_c[:, :, k] * fH2[:, :, k]
                T5[:, :, k] = (
                    Omega_H2_P[k + 1] * MH2_P_sum[:, :, k + 1] +
                    Omega_H2_H[k + 1] * MH2_H_sum[:, :, k + 1] +
                    Omega_H2_H2[k + 1] * MH2_H2_sum[:, :, k + 1] +
                    Omega_H2_P[k] * MH2_P_sum[:, :, k] +
                    Omega_H2_H[k] * MH2_H_sum[:, :, k] +
                    Omega_H2_H2[k] * MH2_H2_sum[:, :, k]
                )
                T6[:, :, k] = Swall_sum[:, :, k + 1] + Swall_sum[:, :, k]

                max_terms = np.max(np.stack([np.abs(T1[:,:,k]), np.abs(T2[:,:,k]), np.abs(T3[:,:,k]), np.abs(T4[:,:,k]), np.abs(T5[:,:,k]), np.abs(T6[:,:,k])]))

                mesh_error[:, :, k] = np.abs(
                    T1[:, :, k] - T2[:, :, k] - T3[:, :, k] + T4[:, :, k] - T5[:, :, k] - T6[:, :, k]
                ) / max_terms

            ave_mesh_error = np.sum(mesh_error) / mesh_error.size
            max_mesh_error = np.max(mesh_error)
            min_mesh_error = np.min(mesh_error[:, :, :nx - 1])

            # Moment Error
            for m in range(mtest):
                for k in range(nx - 1):
                    MT1 = np.sum(Vr2pidVr * (T1[:, :, k] @ (dVx * vx**m)))
                    MT2 = np.sum(Vr2pidVr * (T2[:, :, k] @ (dVx * vx**m)))
                    MT3 = np.sum(Vr2pidVr * (T3[:, :, k] @ (dVx * vx**m)))
                    MT4 = np.sum(Vr2pidVr * (T4[:, :, k] @ (dVx * vx**m)))
                    MT5 = np.sum(Vr2pidVr * (T5[:, :, k] @ (dVx * vx**m)))
                    MT6 = np.sum(Vr2pidVr * (T6[:, :, k] @ (dVx * vx**m)))
                    moment_error[k, m] = np.abs(MT1 - MT2 - MT3 + MT4 - MT5 - MT6) / np.max(np.abs([MT1, MT2, MT3, MT4, MT5, MT6]))
                max_moment_error[m] = np.max(moment_error[:, m])


            # Compute error in qxH2_total
            #    qxH2_total2 total neutral heat flux profile (watts m^-2)
            #               This is the total heat flux transported by the neutrals
            #               computed in a different way from:
            #
            #               qxH2_total2(k)=vth3*total(Vr2pidVr*((vr2vx2(*,*,k)*fH2(*,*,k))#(Vx*dVx)))*0.5*(2*mu*mH)
            #               This should agree with qxH2_total if the definitions of nH2, pH2, piH2_xx,
            #               TH2, VxH2, and qxH2 are coded correctly.

            qxH2_total2 = np.zeros(nx)
            for k in range(nx):
                qxH2_total2[k] = 0.5 * (2 * mu * mH) * vth3 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * fH2[:, :, k]) @ (vx * dVx)))

            qxH2_total_error = np.abs(qxH2_total - qxH2_total2) / np.max(np.abs([qxH2_total, qxH2_total2]))

            Q1 = np.zeros(nx)
            Q2 = np.zeros(nx)
            for k in range(nx - 1):
                Q1[k] = (qxH2_total[k + 1] - qxH2_total[k]) / (x[k + 1] - x[k])
                Q2[k] = 0.5 * (QH2_total[k + 1] + QH2_total[k])
            QH2_total_error = np.abs(Q1 - Q2) / np.max(np.abs([Q1, Q2]))

            #breakpoint()
            if debrief > 0:
                print(f"{prompt}Maximum particle conservation error of total collision operator: {np.max(C_error)}")
                print(f"{prompt}Maximum H2_HP_CX particle conservation error: {np.max(CX_error)}")
                print(f"{prompt}Maximum H2_Wall particle conservation error: {np.max(Wall_error)}")
                print(f"{prompt}Maximum H2_H2_EL particle conservation error: {max_H2_H2_error[0]}")
                print(f"{prompt}Maximum H2_H2_EL x-momentum conservation error: {max_H2_H2_error[1]}")
                print(f"{prompt}Maximum H2_H2_EL total energy conservation error: {max_H2_H2_error[2]}")
                print(f"{prompt}Maximum H2_H_EL  particle conservation error: {max_H2_H_error[0]}")
                print(f"{prompt}Maximum H2_P_EL  particle conservation error: {max_H2_P_error[0]}")
                print(f"{prompt}Average mesh_error = {ave_mesh_error}")
                print(f"{prompt}Maximum mesh_error = {max_mesh_error}")
                for m in range(5):
                    print(f"{prompt}Maximum fH2 vx^{m} moment error: {max_moment_error[m]}")
                print(f"{prompt}Maximum qxH2_total error = {np.max(qxH2_total_error)}")
                print(f"{prompt}Maximum QH2_total error = {np.max(QH2_total_error)}")
                if debug > 0:
                    input("Press return to continue...")

        mid1 = np.argmin(np.abs(x - 0.7 * (np.max(x) + np.min(x)) / 2))
        mid2 = np.argmin(np.abs(x - 0.85 * (np.max(x) + np.min(x)) / 2))
        mid3 = np.argmin(np.abs(x - 0.5 * (np.max(x) + np.min(x))))
        mid4 = np.argmin(np.abs(x - 1.15 * (np.max(x) + np.min(x)) / 2))
        mid5 = np.argmin(np.abs(x - 1.3 * (np.max(x) + np.min(x)) / 2))
        mid6 = np.argmin(np.abs(x - 1.45 * (np.max(x) + np.min(x)) / 2))

        if plot > 1:
            fH21d = np.zeros((nvx, nx))
            for k in range(nx):
                fH21d[:, k] = Vr2pidVr @ fH2[:, :, k]

            ymin = np.min(fH21d)
            ymax = np.max(fH21d)

            plt.figure()
            plt.title(f'{_HH} Velocity Distribution Function: fH2(Vx)')
            plt.xlabel('Vx/Vth')
            plt.ylabel('fH2')
            plt.ylim([ymin, ymax])

            for i in range(nx):
                plt.plot(vx, fH21d[:, i], color=plt.cm.tab10((i % 6) + 2))

            plt.show()

        if plot > 0:
            data = np.stack([nH, n, nHP, nH2])
            jp = np.where(data > 0)
            yrange = [np.min(data[jp]), np.max(data[jp])]

            plt.figure()
            plt.title('Density Profiles')
            plt.xlabel('x (meters)')
            plt.ylabel('m$^{-3}$')
            plt.yscale('log')
            plt.ylim(yrange)

            plt.plot(x, nH, label=_H, color='tab:red')
            plt.text(x[mid1], 1.2 * nH[mid1], _H, color='tab:red')

            plt.plot(x, n, label='e-', color='tab:green')
            plt.text(x[mid2], 1.2 * n[mid2], 'e-', color='tab:green')

            plt.plot(x, nH2, label=_HH, color='tab:blue')
            plt.text(x[mid3], 1.2 * nH2[mid3], _HH, color='tab:blue')

            plt.plot(x, nHP, label=_Hp, color='tab:orange')
            plt.text(x[mid4], 1.2 * nHP[mid4], _Hp, color='tab:orange')

            plt.legend()
            plt.show()


        if plot > 0:
            data = np.stack([TH, Te, THP, TH2])
            jp = np.where(data > 0)
            yrange = [np.min(data[jp]), np.max(data[jp])]

            plt.figure()
            plt.title('Temperature Profiles')
            plt.xlabel('x (meters)')
            plt.ylabel('eV')
            plt.yscale('log')
            plt.ylim(yrange)

            plt.plot(x, TH, label=_H, color='tab:red')
            plt.text(x[mid1], 1.2 * TH[mid1], _H, color='tab:red')

            plt.plot(x, Te, label='e-', color='tab:green')
            plt.text(x[mid2], 1.2 * Te[mid2], 'e-', color='tab:green')

            plt.plot(x, TH2, label=_HH, color='tab:blue')
            plt.text(x[mid3], 1.2 * TH2[mid3], _HH, color='tab:blue')

            plt.plot(x, THP, label=_Hp, color='tab:orange')
            plt.text(x[mid4], 1.2 * THP[mid4], _Hp, color='tab:orange')

            plt.legend()
            plt.show()

        if Compute_H_Source:
            if debrief > 1:
                print(f"{prompt}Computing Velocity Distributions of H products...")

            # Set Normalized Franck-Condon Velocity Distributions for reactions R2, R3, R4, R5, R6, R7, R8, R10
            nFC = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 7])  # Make lookup table to select reaction Rn in SFCn

            SFCn = np.zeros((nvr, nvx, nx, 8))
            Eave = np.zeros((nx, 8))
            Emax = np.zeros((nx, 8))
            Emin = np.zeros((nx, 8))

            # === Reaction R2: e + H2 -> e + H(1s) + H(1s)
            ii = nFC[2]
            Eave[:, ii] = 3.0
            Emax[:, ii] = 4.25
            Emin[:, ii] = 2.0

            # === Reaction R3: e + H2 -> e + H(1s) + H*(2s)
            ii = nFC[3]
            Eave[:, ii] = 0.3
            Emax[:, ii] = 0.55
            Emin[:, ii] = 0.0

            # === Reaction R4: e + H2 -> e + H(+) + H(1s) + e
            ii = nFC[4]
            Ee = 3.0 * Te / 2.0  # Electron energy (3/2 kT)
            kk = np.where(Ee <= 26.0)[0]
            if kk.size > 0:
                Eave[kk, ii] = 0.25
            kk = np.where((Ee > 26.0) & (Ee <= 41.6))[0]
            if kk.size > 0:
                Eave[kk, ii] = 0.5 * (Ee[kk] - 26)
                Eave[kk, ii] = np.maximum(Eave[kk, ii], 0.25)
            kk = np.where(Ee > 41.6)[0]
            if kk.size > 0:
                Eave[kk, ii] = 7.8
            Emax[:, ii] = 1.5 * Eave[:, ii]  # Note the max/min values here are a guess
            Emin[:, ii] = 0.5 * Eave[:, ii]  # Note the max/min values here are a guess

            # === Reaction R5: e + H2 -> e + H*(2p) + H*(2s)
            ii = nFC[5]
            Eave[:, ii] = 4.85
            Emax[:, ii] = 5.85
            Emin[:, ii] = 2.85

            # === Reaction R6: e + H2 -> e + H(1s) + H*(n=3)
            ii = nFC[6]
            Eave[:, ii] = 2.5
            Emax[:, ii] = 3.75
            Emin[:, ii] = 1.25

            # === Reaction R7: e + H2(+) -> e + H(+) + H(1s)
            ii = nFC[7]
            Eave[:, ii] = 4.3
            Emax[:, ii] = 4.3 + 2.1
            Emin[:, ii] = 4.3 - 2.1

            # === Reaction R8: e + H2(+) -> e + H(+) + H*(n=2)
            ii = nFC[8]
            Eave[:, ii] = 1.5
            Emax[:, ii] = 1.5 + 0.75
            Emin[:, ii] = 1.5 - 0.75

            # === Reaction R10: e + H2(+) -> H(1s) + H*(n≥2)
            ii = nFC[10]

            # Compute relative cross-sections for populating a specific n level for reaction R10
            # (see page 62 in Janev, "Elementary Processes in Hydrogen-Helium Plasmas", Springer-Verlag, 1987)
            #         n=2   3    4    5    6            
            R10rel = np.array([0.1, 0.45, 0.22, 0.12, 0.069])
            for k in range(7, 11):  # Extend for n = 7,8,9,10
                R10rel = np.append(R10rel, 10.0 / k**3)
            En = 13.58 / (2 + np.arange(9))**2  # Energy levels in eV (for n=2 to 10)

            #breakpoint()

            # NOTE: in IDL, when Ee and En are different lengths, the longer one is simply cut to be the same length as the short one
            # That throws an error in python, so we need to manually set them to be the same length
            nmin = min(len(Ee), len(En), len(R10rel))
            Ee_t = Ee[:nmin]
            En_t = En[:nmin]
            R10rel_t = R10rel[:nmin]

            for k in range(nx):
                EHn = 0.5 * (Ee_t - En_t) * R10rel_t / np.sum(R10rel) # use the truncated versions except for the total in the denominator
                EHn = np.maximum(EHn, 0.0)
                Eave[k, ii] = max(np.sum(EHn), 0.25)
                Emax[k, ii] = 1.5 * Eave[k, ii]
                Emin[k, ii] = 0.5 * Eave[k, ii]

            # Set SFCn values for reactions R2, R3, R4, R5, R6, R7, R8, R10
            Vfc = np.zeros((nvr, nvx, nx))   # Placeholder: Franck-Condon velocity-space distributions
            Tfc = np.zeros_like(Vfc)
            magV = np.sqrt(vr2vx2)  # Magnitude of velocity at each mesh point

            _THP = np.zeros((nvr, nvx, nx))
            _TH2 = np.zeros_like(_THP)

            for k in range(nx):
                _THP[:, :, k] = THP[k] / Tnorm
                _TH2[:, :, k] = TH2[k] / Tnorm


            # The following function is choosen to represent the velocity distribution of the
            # hydrogen products for a given reaction, accounting for the Franck-Condon energy 
            # distribution and accounting for additional velocity spreading due to the finite 
            # temperature of the molcules (neutral and ionic) prior to breakup:
            #
            #     f(Vr,Vx) = exp( -0.5*mH*mu*(|v|-Vfc+0.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )
            #
            #       	|v|=sqrt(Vr^2+Vx^2)
            #	        Tfc= Franck-Condon 'temperature' = (Emax-Emin)/4
            #	        Vfc= Franck-Condon  velocity = sqrt(2 Eave/mH/mu)
            #		Tmol= temperature of H2 molecule (neutral or ionic)
            #
            #    This function is isotropic in velocity space and can be written in terms
            #  of a distribution in particle speed, |v|, 
            #
            #     f(|v|) = exp( -(|v|-Vfc+1.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )
            #
            # with velocities normalized by vth and T normalized by Tnorm.
            #
            #  Recognizing the the distribution in energy, f(E), and particle speed, f(|v|),
            #  are related by  f(E) dE = f(|v|) 2 pi v^2 dv, and that dE/dv = mH mu v,
            #  f(E) can be written as
            #
            #     f(E) = f(|v|) 2 pi |v|/(mH mu) = const. |v| exp( -(|v|-Vfc+1.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )
            #
            # The function f(Vr,Vx) was chosen because it has has the following characteristics:
            #
            # (1) For Tmol/2 << Tfc,  the peak in the v^2 times the energy distribution, can be found
            #    by finding the |v| where df(E)/d|v| =0
            #
            #    df(E)/d|v|= 0 = 3v^2 exp() - 2(|v|-Vfc+1.5*Tfc/Vfc)/Tfc v^3 exp() 
            #                    2(|v|-Vfc+1.5*Tfc/Vfc)/Tfc |v| = 3
            #
            #    which is satisfied when |v|=Vfc. Thus the energy-weighted energy distribution peaks
            #    at the velocity corresponding to the average Franck-Condon energy.
            #
            # (2) for Tmol/2 >> Tfc ~ Vfc^2, the velocity distribution becomes
            #
            #	f(|v|) = exp( -2(|v|-Vfc+1.5*Tfc/Vfc)^2/Tmol )
            #
            #    which leads to a velocity distribution that approaches the molecular velocity
            #    distribution with the magnitude of the average velocity divided by 2. This
            #    is the appropriate situation for when the Franck-Condon energies are negligible
            #    relative to the thermal speed of the molecules.

            Rn = [2, 3, 4, 5, 6, 7, 8, 10]
            for jRn, R in enumerate(Rn):
                ii = nFC[R]
                # Broadcast Franck-Condon 'mean velocity' and 'temperature'
                Tfc[0, 0, :] = 0.25 * (Emax[:, ii] - Emin[:, ii]) / Tnorm # Franck-Condon 'effective temperature'
                Vfc[0, 0, :] = np.sqrt(Eave[:, ii]) / np.sqrt(Tnorm) # Velocity corresponding to Franck-Condon 'mean evergy'

                for k in range(nx):
                    Vfc[:,:,k] = Vfc[0, 0, k]  # Broadcast Vfc to all Vr-Vx points
                    Tfc[:,:,k] = Tfc[0, 0, k]  # Broadcast Tfc to all Vr-Vx points

                # For R2-R6, the Franck-Condon 'mean energy' is taken equal to Eave
                # and the 'temperature' corresponds to the sum of the Franck-Condon 'temperature', Tfc,
                # and the temperature of the H2 molecules, TH2. (Note: directed neutral molecule velocity
                # is not included and assumed to be small)

                # For R7, R8 and R10, the Franck-Condon 'mean energy' is taken equal to Eave
                # and the 'temperature' corresponds to the sum of the Franck-Condon 'temperature', Tfc,
                # and the temperature of the H2(+) molecular ions, THP. (Note: directed molecular ion velocity
                # is not included and assumed to be small)
                if R <= 6:
                    Tmol = _TH2
                else:
                    Tmol = _THP

                # Compute Franck-Condon distribution function
                arg = -((magV - Vfc + 1.5 * Tfc / Vfc)**2) / (Tfc + 0.5 * Tmol)
                SFCn[:, :, :, ii] = np.exp(np.maximum(arg, -80.0))  # numerical cutoff at -80

                # Normalize over Vr-Vx for each spatial slice
                for k in range(nx):
                    integral = np.sum(Vr2pidVr[:, None] * SFCn[:, :, k, ii] * dVx[None, :])
                    #if integral > 0:
                    SFCn[:, :, k, ii] /= integral

            nm=3
            cmap = plt.get_cmap('tab10')
            if plot > 3:
                x_indices = np.round(np.linspace(0, nx - 1, nm)).astype(int)
                fig, axes = plt.subplots(nm, 1, figsize=(6, 3 * nm), constrained_layout=True)

                for mm, ax in enumerate(axes):
                    kx = x_indices[mm]
                    for jRn, R in enumerate(Rn):
                        ii = nFC[R]
                        c = cmap(jRn % 10)
                        CS = ax.contour(SFCn[:, :, kx, ii], levels=10, linewidths=1.0, colors=[c])
                        ax.clabel(CS, inline=True, fontsize=8)
                        ax.set_title(f"R{R}, Te = {Te[kx]:.2f} eV")
                    ax.set_xlabel('Vx index')
                    ax.set_ylabel('Vr index')

                fig.suptitle('SFCn - Franck-Condon Velocity Distributions')
                plt.show()

            if plot > 2:
                x_indices = np.round(np.linspace(0, nx - 1, nm)).astype(int)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.set_xscale('log')
                ax.set_ylim(0, 1)
                ax.set_title(f'Energy Distribution of {_H} Products')
                ax.set_xlabel('Energy (eV)')
                ax.set_ylabel('Normalized')

                for mm, kx in enumerate(x_indices):
                    ax.text(0.85, 0.92 - 0.04 * mm, f'Te = {Te[kx]:.1f}', transform=ax.transAxes, fontsize=8)
                    for jRn, R in enumerate(Rn):
                        ii = nFC[R]
                        EFC = Eaxis * SFCn[:, ip_pos[0], kx, ii] * VrVr4pidVr / dEaxis
                        EFC /= np.max(EFC) if np.max(EFC) > 0 else 1
                        ax.plot(Eaxis, EFC, label=f'R{R}', color=cmap(jRn % 10))

                ax.legend(fontsize=8)
                plt.show()

            if plot > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.set_yscale('log')
                ax.set_ylim(0.1, 100)
                ax.set_xlabel('x (meters)')
                ax.set_ylabel('Average Energy (eV)')
                ax.set_title(f'Average Energy of {_H}, {_p} Products')

                for jRn, R in enumerate(Rn):
                    ii = nFC[R]
                    Ebar = np.zeros(nx)

                    for k in range(nx):
                        integrand = vr2vx2[:, :, k] * SFCn[:, :, k, ii]
                        Ebar[k] = 0.5 * mu * mH * vth2 * np.sum(Vr2pidVr @ (integrand @ dVx)) / q

                    ax.plot(x, Ebar, label=f'R{R}: {_Rn[R]}', color=cmap(jRn % 10))
                    ax.text(0.4, 0.9 - 0.03 * jRn, f'R{R}: {_Rn[R]}',
                            transform=ax.transAxes, fontsize=8, color=cmap(jRn % 10))

                ax.legend(fontsize=8)
                plt.show()


            
            # here
            Vbar_Error = np.zeros(nx)
            if compute_errors:
                # Test: The average speed of a non-shifted maxwellian should be 2*Vth*sqrt(Ti(x)/Tnorm)/sqrt(!pi)
                TFC = np.linspace(np.min(Eave[0, :]), np.max(Eave[0, :]), nx)
                vx_shift = np.zeros(nx)
                Tmaxwell = TFC
                mol = 1  # indicates molecular Maxwellian

                Maxwell = create_shifted_maxwellian_core(
                    vr, vx, vx_shift, Tmaxwell,
                    vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
                    vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                    jpa, jpb, jna, jnb,
                    mol, mu, mH, q,
                    debug=debug
                )


                vbar_test = vth * np.sqrt(vr2vx2[:, :, 0])

                for k in range(nx):
                    integrand = vbar_test * Maxwell[:, :, k]
                    vbar = np.sum(Vr2pidVr @ (integrand @ dVx))
                    vbar_exact = 2 * vth * np.sqrt(TFC[k] / Tnorm) / np.sqrt(np.pi)
                    Vbar_Error[k] = abs(vbar - vbar_exact) / vbar_exact

                if debrief > 0:
                    print(f"{prompt}Maximum Vbar error over FC energy range =", np.max(Vbar_Error))

            # Compute atomic hydrogen source distribution function
            # using normalized FC source distributions SFCn
            for k in range(nx):
                fSH[:, :, k] = (
                    n[k] * nH2[k] * (
                        2 * sigv[k, 2] * SFCn[:, :, k, nFC[2]] +
                        2 * sigv[k, 3] * SFCn[:, :, k, nFC[3]] +
                        1 * sigv[k, 4] * SFCn[:, :, k, nFC[4]] +
                        2 * sigv[k, 5] * SFCn[:, :, k, nFC[5]] +
                        2 * sigv[k, 6] * SFCn[:, :, k, nFC[6]]
                    ) +
                    n[k] * nHP[k] * (
                        1 * sigv[k, 7] * SFCn[:, :, k, nFC[7]] +
                        1 * sigv[k, 8] * SFCn[:, :, k, nFC[8]] +
                        2 * sigv[k, 10] * SFCn[:, :, k, nFC[10]]
                    )
                )

            # Compute total H and H(+) sources
            for k in range(nx):
                SH[k] = np.sum(Vr2pidVr @ (fSH[:, :, k] @ dVx))
                SP[k] = (
                    n[k] * nH2[k] * sigv[k, 4] +
                    n[k] * nHP[k] * (sigv[k, 7] + sigv[k, 8] + 2 * sigv[k, 9])
                )

            # Compute total HP source
            SHP = n * nH2 * sigv[:, 1]

            # Compute energy distribution of H source
            for k in range(nx):
                ESH[:, k] = Eaxis * fSH[:, ip_pos[0], k] * VrVr4pidVr / dEaxis
                ESH[:, k] /= np.max(ESH[:, k]) if np.max(ESH[:, k]) > 0 else 1

            if plot > 2:
                fH21d = np.zeros((nvx, nx))
                for k in range(nx):
                    fH21d[:, k] = Vr2pidVr @ fSH[:, :, k]

                plt.figure()
                for i in range(nx):
                    plt.plot(vx, fH21d[:, i], color=cmap((i % 6) + 1))
                plt.title(f'{_H} Source Velocity Distribution Function: fSH(Vx)')
                plt.xlabel('Vx / Vth')
                plt.ylabel('m⁻³ s⁻¹ dVx⁻¹')
                plt.show()

            if plot > 2:
                plt.figure()
                for k in range(nx):
                    plt.plot(Eaxis, ESH[:, k], color=cmap((k % 6) + 1))
                plt.xscale('log')
                plt.title(f'{_H} Source Energy Distribution: ESH(E) = E fSH(E)')
                plt.xlabel('E (eV)')
                plt.ylabel('Normalized')
                plt.show()

            if plot > 1:
                Ebar = np.zeros(nx)
                for k in range(nx):
                    num = np.sum(Vr2pidVr @ ((vr2vx2[:, :, k] * fSH[:, :, k]) @ dVx))
                    den = np.sum(Vr2pidVr @ (fSH[:, :, k] @ dVx))
                    Ebar[k] = 0.5 * mu * mH * vth2 * num / (q * den if den > 0 else 1)

                plt.figure()
                plt.plot(x, Ebar, color='C2')
                plt.yscale('log')
                plt.title(f'Average Energy of {_H} Source Distribution')
                plt.xlabel('x (meters)')
                plt.ylabel('eV')
                plt.show()

            if plot > 0:
                data = np.stack([SH, SP, SHP, SH2, NuLoss * nHP, NuDis * nHP])
                valid = data[data < 1e32]
                yrange = [np.min(valid), np.max(valid)]

                plt.figure()
                plt.yscale('log')
                plt.ylim(*yrange)
                plt.title("Source and Sink Profiles")
                plt.xlabel("x (meters)")
                plt.ylabel("m⁻³ s⁻¹")

                plt.plot(x, SH, label=f'{_H} source', color='C2')
                plt.plot(x, SHP, label=f'{_Hp} source', color='C3')
                plt.plot(x, SP, label=f'{_p} source', color='C4')
                plt.plot(x, NuLoss * nHP, label=f'{_Hp} loss', color='C5')
                plt.plot(x, NuDis * nHP, label=f'{_Hp} dissoc.', color='C6')
                plt.plot(x, SH2, label=f'{_HH} source', color='C1')
                plt.legend()
                plt.show()

            if plot > 0:
                gammaxH2_plus = np.zeros(nx)
                gammaxH2_minus = np.zeros(nx)
                for k in range(nx):
                    gammaxH2_plus[k] = vth * np.sum(Vr2pidVr @ (fH2[:, ip_pos, k] @ (vx[ip_pos] * dVx[ip_pos])))
                    gammaxH2_minus[k] = vth * np.sum(Vr2pidVr @ (fH2[:, in_neg, k] @ (vx[in_neg] * dVx[in_neg])))

                data = np.concatenate([gammaxH2_plus, gammaxH2_minus, GammaxH2])
                valid = data[data < 1e32]
                yrange = [np.min(valid), np.max(valid)]

                plt.figure()
                plt.ylim(*yrange)
                plt.title(f"{_HH} Fluxes")
                plt.xlabel("x (meters)")
                plt.ylabel("m⁻² s⁻¹")
                plt.plot(x, GammaxH2, label="Γ", color='C2')
                plt.plot(x, gammaxH2_plus, label="Γ⁺", color='C3')
                plt.plot(x, gammaxH2_minus, label="Γ⁻", color='C4')
                plt.legend()
                plt.show()

            # Compute Source error
            if compute_errors:
                if debrief > 1:
                    print(f"{prompt}Computing Source Error")

                Source_Error = np.zeros(nx)

                dGammaxH2dx = np.diff(GammaxH2) / np.diff(x)
                SH_p = 0.5 * (
                    SH[1:] + SP[1:] + 2 * NuLoss[1:] * nHP[1:] - 2 * SH2[1:] +
                    SH[:-1] + SP[:-1] + 2 * NuLoss[:-1] * nHP[:-1] - 2 * SH2[:-1]
                )

                max_source = np.max(np.concatenate([SH, 2 * SH2]))
                for k in range(nx - 1):
                    numerator   = abs(2 * dGammaxH2dx[k] + SH_p[k])
                    denominator = max(abs(2 * dGammaxH2dx[k]), abs(SH_p[k]), max_source)
                    Source_Error[k] = numerator / denominator

                if debrief > 0:
                    print(f"{prompt}Maximum Normalized Source_error =", np.max(Source_Error))
        # here
        # Don't worry about saving for the moment
        # np.savez('kinetic_H2_results.npz',
        #     # Saved inputs
        #     vx=vx,
        #     vr=vr,
        #     x=x,
        #     Tnorm=Tnorm,
        #     mu=mu,
        #     Ti=Ti,
        #     vxi=vxi,
        #     Te=Te,
        #     n=n,
        #     fH2BC=fH2BC,
        #     GammaxH2BC=GammaxH2BC,
        #     NuLoss=NuLoss,
        #     PipeDia=PipeDia,
        #     fH=fH,
        #     SH2=SH2,
        #     fH2=fH2,
        #     nHP=nHP,
        #     THP=THP,

        #     # Saved outputs (cast to single precision)
        #     #fH2=fH2.astype(np.float32),
        #     nH2=nH2.astype(np.float32),
        #     GammaxH2=GammaxH2.astype(np.float32),
        #     VxH2=VxH2.astype(np.float32),
        #     pH2=pH2.astype(np.float32),
        #     TH2=TH2.astype(np.float32),
        #     qxH2=qxH2.astype(np.float32),
        #     qxH2_total=qxH2_total.astype(np.float32),
        #     Sloss=Sloss.astype(np.float32),
        #     QH2=QH2.astype(np.float32),
        #     RxH2=RxH2.astype(np.float32),
        #     QH2_total=QH2_total.astype(np.float32),
        #     AlbedoH2=np.float32(AlbedoH2),
        #     fSH=fSH.astype(np.float32),
        #     SH=SH.astype(np.float32),
        #     SP=SP.astype(np.float32),
        #     SHP=SHP.astype(np.float32),
        #     NuE=NuE.astype(np.float32),
        #     NuDis=NuDis.astype(np.float32),
        #     piH2_xx=piH2_xx.astype(np.float32),
        #     piH2_yy=piH2_yy.astype(np.float32),
        #     piH2_zz=piH2_zz.astype(np.float32),
        #     RxH_H2=RxH_H2.astype(np.float32),
        #     #RxHp_H2=RxHp_H2.astype(np.float32),
        #     RxP_H2=RxP_H2.astype(np.float32),
        #     #RxP_H2_CX=RxP_H2_CX.astype(np.float32),
        #     #EHp_H2=EHp_H2.astype(np.float32),
        #     EP_H2=EP_H2.astype(np.float32),
        #     #EP_H2_CX=EP_H2_CX.astype(np.float32),
        #     Epara_PerpH2_H2=Epara_PerpH2_H2.astype(np.float32),
        #     ESH=ESH.astype(np.float32),
        #     Eaxis=Eaxis.astype(np.float32)
        #     )
    

        if debug > 0:
            print(f"{prompt}Finished")


        result = {
            'fH2': fH2,
            'nH2': nH2,
            'GammaxH2': GammaxH2,
            'VxH2': VxH2,
            'pH2': pH2,
            'TH2': TH2,
            'qxH2': qxH2,
            'qxH2_total': qxH2_total,
            'Sloss': Sloss,
            'QH2': QH2,
            'RxH2': RxH2,
            'QH2_total': QH2_total,
            'AlbedoH2': AlbedoH2,
            'WallH2': WallH2,
            'nHP': nHP,
            'THP': THP,
            'fSH': fSH,
            'SH': SH,
            'SP': SP,
            'SHP': SHP,
            'NuE': NuE,
            'NuDis': NuDis,
            #keyword outputs
            'ESH': ESH,
            'Eaxis': Eaxis,
        }

        
        seeds = {
            'vx_s' : vx,
            'vr_s' : vr,
            'x_s' : x,
            'Tnorm_s' : Tnorm,
            'mu_s' : mu,
            'Ti_s' : Ti,
            'Te_s' : Te,
            'n_s' : n,
            'vxi_s' : vxi,
            'fH2BC_s' : fH2BC,
            'GammaxH2BC_s' : GammaxH2BC,
            'NuLoss_s' : NuLoss,
            'PipeDia_s' : PipeDia,
            'fH_s' : fH,
            'SH2_s' : SH2,
            'fH2_s' : fH2,
            'nHP_s' : nHP,
            'THP_s' : THP,
            'Simple_CX_s' : Simple_CX,
            'Sawada_s' : Sawada,
            'H2_H2_EL_s' : H2_H2_EL,
            'H2_P_EL_s' : H2_P_EL,
            'H2_H_EL_s' : H2_H_EL,
            'H2_HP_CX_s' : H2_HP_CX,
            'ni_correct_s' : ni_correct
        }

        #output
        output = {
            'piH2_xx': piH2_xx,
            'piH2_yy': piH2_yy,
            'piH2_zz': piH2_zz,
            'RxH2CX': RxH2CX,
            'RxH_H2': RxH_H2,
            'RxP_H2': RxP_H2,
            'RxW_H2': RxW_H2,
            'EH2CX': EH2CX,
            'EH_H2': EH_H2,
            'EP_H2': EP_H2,
            'EW_H2': EW_H2,
            'Epara_PerpH2_H2': Epara_PerpH2_H2,
        }

        errors = {
            'Max_dx': Max_dx,
            'Vbar_Error': Vbar_Error,
            'mesh_error': mesh_error,
            'moment_error': moment_error,
            'C_error': C_error,
            'CX_error': CX_error,
            'Wall_error': Wall_error,
            'H2_H2_error': H2_H2_error,
            'Source_Error': Source_Error,
            'qxH2_total_error': qxH2_total_error,
            'QH2_total_error': QH2_total_error,
        }

        H_moments = {
            'nH': nH,
            'VxH': VxH,
            'TH': TH,
        }

        #internal
        internal = {
            'vr2vx2': vr2vx2,
            'vr2vx_vxi2': vr2vx_vxi2,
            'fw_hat': fw_hat,
            'fi_hat': fi_hat,
            'fHp_hat': fHp_hat,
            'EH2_P': EH2_P,
            'sigv': sigv,
            'alpha_Loss': alpha_loss,
            'v_v2': v_v2,
            'v_v': v_v,
            'vr2_vx2': vr2_vx2,
            'vx_vx': vx_vx,
            'Vr2pidVrdVx': Vr2pidVrdVx,
            'SIG_CX': SIG_CX,
            'SIG_H2_H2': SIG_H2_H2,
            'SIG_H2_H': SIG_H2_H,
            'SIG_H2_P': SIG_H2_P,
            'Alpha_CX': Alpha_CX,
            'Alpha_H2_H': Alpha_H2_H,
            'MH2_H2_sum': MH2_H2_sum,
            'Delta_nH2s': Delta_nH2s,
        }

        
        # return the outputs and all the common blocks from the procedure
        return result, seeds, output, errors, H_moments, internal