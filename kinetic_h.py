#
# Kinetic_H.py
#
# This subroutine is part of the "KN1D" atomic and molecular neutral transport code.
#
#   This subroutine solves a 1-D spatial, 2-D velocity kinetic neutral transport 
# problem for atomic hydrogen (H) or deuterium by computing successive generations of 
# charge exchange and elastic scattered neutrals. The routine handles electron-impact 
# ionization, proton-atom charge exchange, radiative recombination, and elastic
# collisions with hydrogenic ions, neutral atoms, and molecules.
#
#   The positive vx half of the atomic neutral distribution function is inputted at x(0) 
# (with arbitrary normalization) and the desired flux of hydrogen atoms entering the slab,
# at x(0) is specified. Background profiles of plasma ions, (e.g., Ti(x), Te(x), n(x), vxi(x),...)
# molecular ions, (nHP(x), THP(x)), and molecular distribution function (fH) are inputted.
#
# Optionally, the hydrogen source velocity distribution function is also inputted.
# (The H source and fH2 distribution functions can be computed using procedure 
# "Kinetic_H2.py".) The code returns the atomic hydrogen distribution function, fH(vr,vx,x) 
# for all vx, vr, and x of the specified vr,vx,x grid.
#
#   Since the problem involves only the x spatial dimension, all distribution functions
# are assumed to have rotational symmetry about the vx axis. Consequently, the distributions
# only depend on x, vx and vr where vr =sqrt(vy^2+vz^2)
#
#  History:
#
#    B. LaBombard   First coding in IDL based on Kinetic_Neutrals.pro 		22-Dec-2000
#
#    For more information, see write-up: "A 1-D Space, 2-D Velocity, Kinetic 
#    Neutral Transport Algorithm for Hydrogen Atoms in an Ionizing Plasma", B. LaBombard
#    
#    Translated to Python by J. Dunsmore					                06-Aug-2025
#
# Note: Variable names contain characters to help designate species -
#	atomic neutral (H), molecular neutral (H2), molecular ion (HP), proton (i) or (P) 
#

import numpy as np
import matplotlib.pyplot as plt
from make_dvr_dvx import make_dvr_dvx
from create_shifted_maxwellian_core import create_shifted_maxwellian_core
from jhs_coef import JHS_Coef
from jhalpha_coef import JHAlpha_Coef
from sigmav_ion_h0 import SigmaV_Ion_H0
from sigmav_rec_h1s import SigmaV_rec_H1s
from sigma_cx_h0 import Sigma_CX_H0
from sigma_el_h_h import Sigma_EL_H_H
from sigma_el_h_hh import Sigma_EL_H_HH
from sigma_el_p_h import Sigma_EL_P_H
from sigmav_cx_h0 import SigmaV_CX_H0
from scipy.io import readsav

def kinetic_h(
    # standard inputs
    vx, vr, x, Tnorm, mu, Ti, Te, n, vxi,
    fHBC, GammaxHBC, PipeDia=None, fH2=None, fSH=None, nHP=None, THP=None,

    # input and output (i.e a seed value can be provided)
    fH=None,

    # keywords
    truncate=1.0e-4, Simple_CX=True, Max_Gen=50,
    No_Johnson_Hinnov=False, No_Recomb=False,
    H_H_EL=False, H_P_EL=False, H_H2_EL=False, H_P_CX=False, ni_correct=False,
    compute_errors=False, plot=0, debug=0, debrief=0, pause=False,

    # these are the common blocks that can be used to pass data between runs
    h_seeds=None,
    h_H2_moments=None,
    h_internal=None,
):
    """
    Solve a 1D spatial, 2D velocity kinetic neutral transport problem for atomic hydrogen or deuterium.

    Inputs
    ----------
    vx : ndarray
        Normalized x velocity coordinate. Monotonically increasing. nvx must be even and symmetric.
    vr : ndarray
        Normalized radial velocity coordinate. Positive values only.
    x : ndarray
        Spatial coordinate (m), positive and monotonically increasing.
    Tnorm : float
        Temperature corresponding to the thermal speed (eV).
    mu : float
        Mass ratio, 1=hydrogen, 2=deuterium.
    Ti, Te, n, vxi : ndarray
        Plasma profiles (eV, eV, m^-3, m/s).
    fHBC : ndarray
        Boundary condition distribution function at x=0. Arbitrary normalization.
    GammaxHBC : float
        Desired neutral atom flux density at x=0 (m^-2 s^-1).
    PipeDia : ndarray or None
        Pipe diameter (m). Zero treated as infinite. Default is zero.
    fH2, fSH : ndarray or None
        Molecular and source velocity distribution functions.
    nHP, THP : ndarray or None
        Molecular ion density and temperature profiles.
    
    Inputs/Outputs
    ----------
    fH : ndarray or None
        Atomic neutral distribution function. Initialized if None.

    Keywords
    ----------
    truncate : float
        Stopping criterion on fractional neutral density change.
    Simple_CX : bool
        If True, use simplified charge exchange.
    Max_Gen : int
        Maximum number of generations.
    No_Johnson_Hinnov : bool
        If True, use Janev's rates instead of Johnson-Hinnov model.
    No_Recomb : bool
        If True, exclude recombination.
    H_H_EL, H_P_EL, H_H2_EL, H_P_CX : bool
        Toggles for elastic and CX collision types.
    ni_correct : bool
        If True, apply quasineutral correction.
    compute_errors : bool
        If True, compute error diagnostics.
    plot, debug, debrief : int
        Flags for plotting and verbosity.
    pause : bool
        If True, pause between plots.

    Returns
    -------
    results : dict
        Dictionary containing the computed distribution function and moments.
    seeds : dict
        Dictionary containing internal variables for potential reuse.
    outputs : dict
        Dictionary containing extra output variables such as piH2_xx, piH2_yy, etc.
    errors : dict
        Dictionary containing error diagnostics if compute_errors is True.
    H_moments : dict
        Dictionary containing H moments if provided.
    internal : dict
        Dictionary containing internal variables for potential reuse.
    
    NOTE: seeds, H_moments and internal from the previous run can also be provided as an input.
    """

    prompt = 'Kinetic_H => '

    # if fed in, then read the seeds from previous run
    if h_seeds is not None:
        vx_s = h_seeds['vx_s']
        vr_s = h_seeds['vr_s']
        x_s = h_seeds['x_s']
        Tnorm_s = h_seeds['Tnorm_s']
        mu_s = h_seeds['mu_s']
        Ti_s = h_seeds['Ti_s']
        Te_s = h_seeds['Te_s']
        n_s = h_seeds['n_s']
        vxi_s = h_seeds['vxi_s']
        fHBC_s = h_seeds['fHBC_s']
        GammaxHBC_s = h_seeds['GammaxHBC_s']
        PipeDia_s = h_seeds['PipeDia_s']
        fH2_s = h_seeds['fH2_s']
        fSH_s = h_seeds['fSH_s']
        nHP_s = h_seeds['nHP_s']
        THP_s = h_seeds['THP_s']
        fH_s = h_seeds['fH_s']
        Simple_CX_s = h_seeds['Simple_CX_s']
        JH_s = h_seeds['JH_s']
        Recomb_s = h_seeds['Recomb_s']
        H_H_EL_s = h_seeds['H_H_EL_s']
        H_P_EL_s = h_seeds['H_P_EL_s']
        H_H2_EL_s = h_seeds['H_H2_EL_s']
        H_P_CX_s = h_seeds['H_P_CX_s']
        print('Using h seeds from previous run.')

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
        fHBC_s = None
        GammaxHBC_s = None
        PipeDia_s = None
        fH2_s = None
        fSH_s = None
        nHP_s = None
        THP_s = None
        fH_s = None
        Simple_CX_s = None
        JH_s = None
        Recomb_s = None
        H_H_EL_s = None
        H_P_EL_s = None
        H_H2_EL_s = None
        H_P_CX_s = None
        print('No h seeds provided, starting fresh')

    # if read in, then read the H2 moments from previous run
    if h_H2_moments is not None:
        nH2 = h_H2_moments['nH2']
        vxH2 = h_H2_moments['vxH2']
        TH2 = h_H2_moments['TH2']
        print('Using h2 moments from previous run.')
    else:
        nH2 = None
        vxH2 = None
        TH2 = None
        print('No h2 moments provided, starting fresh')

    # if read in, then read the internals from previous run
    if h_internal is not None:
        vr2vx2 = h_internal['vr2vx2']
        vr2vx_vxi2 = h_internal['vr2vx_vxi2']
        fi_hat = h_internal['fi_hat']
        ErelH_P = h_internal['ErelH_P']
        Ti_mu = h_internal['Ti_mu']
        ni = h_internal['ni']
        sigv = h_internal['sigv']
        alpha_ion = h_internal['alpha_ion']
        v_v2 = h_internal['v_v2']
        v_v = h_internal['v_v']
        vr2_vx2 = h_internal['vr2_vx2']
        vx_vx = h_internal['vx_vx']
        Vr2pidVrdVx = h_internal['Vr2pidVrdVx']
        SIG_CX = h_internal['SIG_CX']
        SIG_H_H = h_internal['SIG_H_H']
        SIG_H_H2 = h_internal['SIG_H_H2']
        SIG_H_P = h_internal['SIG_H_P']
        Alpha_CX = h_internal['Alpha_CX']
        Alpha_H_H2 = h_internal['Alpha_H_H2']
        Alpha_H_P = h_internal['Alpha_H_P']
        MH_H_sum = h_internal['MH_H_sum']
        Delta_nHs = h_internal['Delta_nHs']
        Sn = h_internal['Sn']
        Rec = h_internal['Rec']
        print('Using h internal data from previous run.')
    else:
        vr2vx2 = None
        vr2vx_vxi2 = None
        fi_hat = None
        ErelH_P = None
        Ti_mu = None
        ni = None
        sigv = None
        alpha_ion = None
        v_v2 = None
        v_v = None
        vr2_vx2 = None
        vx_vx = None
        Vr2pidVrdVx = None
        SIG_CX = None
        SIG_H_H = None
        SIG_H_H2 = None
        SIG_H_P = None
        Alpha_CX = None
        Alpha_H_H2 = None
        Alpha_H_P = None
        MH_H_sum = None
        Delta_nHs = None
        Sn = None
        Rec = None
        print('No h internal data provided, starting fresh')


    # Internal debug switches and tolerances
    shifted_Maxwellian_debug = False
    CI_Test = True
    Do_Alpha_CX_Test = False

    DeltaVx_tol = 0.01
    Wpp_tol = 0.001

    # Keyword processing logic
    JH = not No_Johnson_Hinnov
    Recomb = not No_Recomb

    # NOTE: Not sure about this bit
    if debug > 0:
        plot = plot > 1
        debrief = debrief > 1
        pause = True
    
    # Input dimension sizes
    nvr = len(vr)
    nvx = len(vx)
    nx = len(x)

    # Ensure correct data types
    vr = np.asarray(vr, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

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
        print('vxi is zero')
        vxi = np.zeros(nx)
    if len(vxi) != nx:
        raise ValueError(f'{prompt}Number of elements in vxi and x do not agree!')

    if len(Te) != nx:
        raise ValueError(f'{prompt}Number of elements in Te and x do not agree!')

    if len(n) != nx:
        raise ValueError(f'{prompt}Number of elements in n and x do not agree!')

    if GammaxHBC is None:
        raise ValueError(f'{prompt}GammaxHBC is not defined!')

    if PipeDia is None:
        PipeDia = np.zeros(nx)
    if len(PipeDia) != nx:
        raise ValueError(f'{prompt}Number of elements in PipeDia and x do not agree!')

    if fHBC.shape[0] != vr.shape[0]:
        raise ValueError("First dimension of fHBC and length of vr do not match.")
    if fHBC.shape[1] != vx.shape[0]:
        raise ValueError("Second dimension of fHBC and length of vx do not match.")


    if fH2 is None:
        fH2 = np.zeros((nvr, nvx, nx))
    elif fH2.shape != (nvr, nvx, nx):
        raise ValueError(f'{prompt}Shape of fH2 does not match (nvr, nvx, nx)!')

    if fSH is None:
        fSH = np.zeros((nvr, nvx, nx))
    elif fSH.shape != (nvr, nvx, nx):
        raise ValueError(f'{prompt}Shape of fSH does not match (nvr, nvx, nx)!')

    if nHP is None:
        nHP = np.zeros(nx)
    elif len(nHP) != nx:
        raise ValueError(f'{prompt}Number of elements in nHP and x do not agree!')

    if THP is None:
        THP = np.full(nx, 1.0)
    elif len(THP) != nx:
        raise ValueError(f'{prompt}Number of elements in THP and x do not agree!')

    if fH is None:
        fH = np.zeros((nvr, nvx, nx))
    elif fH.shape != (nvr, nvx, nx):
        raise ValueError(f'{prompt}Shape of fH does not match (nvr, nvx, nx)!')

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
    _hv = r"$hv$"

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

    _R1 = _e + plus + _H1s + arrow + _e + plus + _p + plus + _e
    _R2 = _e + plus + _p + arrow + _H1s + plus + _hv
    _R3 = _p + plus + _H1s + arrow + _H1s + plus + _p
    _R4 = _H + plus + _p + arrow + _H + plus + _p + elastic
    _R5 = _H + plus + _HH + arrow + _H + plus + _HH + elastic
    _R6 = _H + plus + _H + arrow + _H + plus + _H + elastic

    _Rn = ["", _R1, _R2, _R3, _R4, _R5, _R6]

    # Check vx symmetry
    in_neg = np.where(vx < 0)[0]
    if len(in_neg) < 1:
        raise ValueError(f'{prompt}vx contains no negative elements!')
    
    ip_pos = np.where(vx > 0)[0]
    if len(ip_pos) < 1:
        raise ValueError(f'{prompt}vx contains no positive elements!')
    
    iz_zero = np.where(vx == 0)[0]
    if len(iz_zero) > 0:
        raise ValueError(f'{prompt}vx contains one or more zero elements!')

    if not np.allclose(vx[ip_pos], -vx[in_neg][::-1]):
        raise ValueError(f'{prompt}vx array elements are not symmetric about zero!')

    # Validate fHBC has positive half values
    fHBC_input = np.zeros_like(fHBC)
    fHBC_input[:, ip_pos] = fHBC[:, ip_pos]
    if np.sum(fHBC_input) <= 0.0 and np.abs(GammaxHBC) > 0.0:
        raise ValueError(f'{prompt}Values for fHBC(*,*) with vx > 0 are all zero!')
    
    # Output variables
    nH = np.zeros(nx)
    GammaxH = np.zeros(nx)
    VxH = np.zeros(nx)
    pH = np.zeros(nx)
    TH = np.zeros(nx)
    qxH = np.zeros(nx)
    qxH_total = np.zeros(nx)
    NetHSource = np.zeros(nx)
    WallH = np.zeros(nx)
    Sion = np.zeros(nx)
    QH = np.zeros(nx)
    RxH = np.zeros(nx)
    QH_total = np.zeros(nx)
    piH_xx = np.zeros(nx)
    piH_yy = np.zeros(nx)
    piH_zz = np.zeros(nx)
    RxHCX = np.zeros(nx)
    RxH2_H = np.zeros(nx)
    RxP_H = np.zeros(nx)
    RxW_H = np.zeros(nx)
    EHCX = np.zeros(nx)
    EH2_H = np.zeros(nx)
    EP_H = np.zeros(nx)
    EW_H = np.zeros(nx)
    Epara_PerpH_H = np.zeros(nx)
    AlbedoH = 0.0
    SourceH = np.zeros(nx)
    SRecomb = np.zeros(nx)

    # --- Internal constants ---
    mH = 1.6726231e-27          # hydrogen mass (kg)
    q = 1.602177e-19            # elementary charge (C)
    k_boltz = 1.380658e-23      # Boltzmann constant (J/K)
    Twall = 293.0 * k_boltz / q # Room temperature in eV

    # --- Internal variables ---
    Work = np.zeros(nvr * nvx)
    fHG = np.zeros((nvr, nvx, nx))
    NHG = np.zeros((nx, Max_Gen + 1))
    vth = np.sqrt(2 * q * Tnorm / (mu * mH))
    vth2 = vth**2
    vth3 = vth**3
    fHs = np.zeros(nx)
    nHs = np.zeros(nx)
    Alpha_H_H = np.zeros((nvr, nvx))
    Omega_H_P = np.zeros(nx)
    Omega_H_H2 = np.zeros(nx)
    Omega_H_H = np.zeros(nx)
    VxHG = np.zeros(nx)
    THG = np.zeros(nx)
    Wperp_paraH = np.zeros(nx)
    vr2vx2_ran2 = np.zeros((nvr, nvx))
    vr2_2vx_ran2 = np.zeros((nvr, nvx))
    vr2_2vx2_2D = np.zeros((nvr, nvx))
    RxCI_CX = np.zeros(nx)
    RxCI_H2_H = np.zeros(nx)
    RxCI_P_H = np.zeros(nx)
    Epara_Perp_CI = np.zeros(nx)
    CI_CX_error = np.zeros(nx)
    CI_H2_H_error = np.zeros(nx)
    CI_P_H_error = np.zeros(nx)
    CI_H_H_error = np.zeros(nx)
    Maxwell = np.zeros((nvr, nvx, nx))

    # --- Velocity space weights and volume elements ---
    (Vr2pidVr, VrVr4pidVr, dVx, vrL, vrR, vxL, vxR, vol,
    vth_Dvx, vx_Dvx, vr_Dvr, Vr2Vx2_2D,
    jpa, jpb, jna, jnb) = make_dvr_dvx(vr, vx)



    # Vr^2 - 2 * Vx^2
    for i in range(nvr):
        vr2_2vx2_2D[i, :] = vr[i]**2 - 2 * vx**2

    # Theta-prime coordinate (for angle integrals)
    ntheta = 5
    dTheta = np.ones(ntheta) / ntheta
    theta = np.pi * (np.arange(ntheta) / ntheta + 0.5 / ntheta)
    cos_theta = np.cos(theta)

    # Scale input molecular distribution function to agree with desired flux
    gamma_input = 1.0
    if abs(GammaxHBC) > 0.0:
        gamma_input = vth * np.sum(Vr2pidVr * np.dot(fHBC_input, vx * dVx))

    ratio = abs(GammaxHBC) / gamma_input
    fHBC_input *= ratio

    if abs(ratio - 1.0) > 0.01 * truncate:
        fHBC = fHBC_input.copy()

    #breakpoint()

    if fHBC.ndim == 3:
        fH[:, ip_pos, 0] = fHBC_input[:, ip_pos, 0]
    elif fHBC.ndim == 2:
        fH[:, ip_pos, 0] = fHBC_input[:, ip_pos]
    else:
        raise ValueError(f'{prompt}fHBC is usually a 2D or 3D array and i have not accounted yet in the code above for any other possibilities!')

    # Disable elastic H2 <-> H collisions if fH2 is zero
    if np.sum(fH2) <= 0.0:
        H_H2_EL = False

    # Set iteration flags
    fH_iterate = False
    if (H_H_EL == True) or (H_P_EL == True) or (H_H2_EL == True):
        fH_iterate = True

    fH_generations = False
    if (fH_iterate == True) or (H_P_CX == True):
        fH_generations = True

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

    New_Molecular_Ions = True
    if nHP_s is not None:
        test = 0
        test += np.sum(nHP_s != nHP)
        test += np.sum(THP_s != THP)
        if test <= 0:
            New_Molecular_Ions = False
    
    New_Electrons = True
    if Te_s is not None:
        test = 0
        test += np.sum(Te_s != Te)
        test += np.sum(n_s != n)
        if test <= 0:
            New_Electrons = False
    
    New_fH2 = True
    if fH2_s is not None:
        if np.all(fH2_s == fH2):
            New_fH2 = False
    
    New_fSH = True
    if fSH_s is not None:
        if np.all(fSH_s == fSH):
            New_fSH = False

    New_Simple_CX = True
    if Simple_CX_s is not None:
        if np.all(Simple_CX_s == Simple_CX):
            New_Simple_CX = False
    
    New_H_Seed = True
    if fH_s is not None:
        if np.all(fH_s == fH):
            New_H_Seed = False

    # Dependency flags
    Do_sigv         = New_Grid or New_Electrons
    Do_ni           = New_Grid or New_Electrons or New_Protons or New_Molecular_Ions
    Do_fH2_moments  = (New_Grid or New_fH2) and np.sum(fH2) > 0.0
    Do_Alpha_CX     = ((New_Grid or (Alpha_CX is None)) or Do_ni or New_Simple_CX) and H_P_CX
    Do_SIG_CX       = ((New_Grid or (SIG_CX is None)) or New_Simple_CX) and (not Simple_CX) and Do_Alpha_CX
    Do_Alpha_H_H2   = ((New_Grid or (Alpha_H_H2 is None)) or New_fH2) and H_H2_EL
    Do_SIG_H_H2     = ((New_Grid or (SIG_H_H2 is None))) and Do_Alpha_H_H2
    Do_SIG_H_H      = ((New_Grid or (SIG_H_H is None))) and H_H_EL
    Do_Alpha_H_P    = ((New_Grid or (Alpha_H_P is None)) or Do_ni) and H_P_EL
    Do_SIG_H_P      = ((New_Grid or (SIG_H_P is None))) and Do_Alpha_H_P
    Do_v_v2         = ((New_Grid or (v_v2 is None))) and (CI_Test or Do_SIG_CX or Do_SIG_H_H2 or Do_SIG_H_H or Do_SIG_H_P)



    # Allocate molecular species quantities
    nH2 = np.zeros(nx)
    vxH2 = np.zeros(nx)
    TH2 = np.ones(nx)  # Initialized to 1.0

    if Do_fH2_moments:
        if debrief > 1:
            print(prompt + 'Computing vx and T moments of fH2')
        # Compute x flow velocity and temperature of molecular species
        for k in range(nx):
            nH2[k] = np.sum(Vr2pidVr * np.sum(fH2[:, :, k] * dVx, axis=1))
            if nH2[k] > 0.0:
                vxH2[k] = vth * np.sum(Vr2pidVr * np.sum(fH2[:, :, k] * (vx * dVx), axis=1)) / nH2[k]
                for i in range(nvr):
                    vr2vx2_ran2[i, :] = vr[i] ** 2 + (vx - vxH2[k] / vth) ** 2
                TH2[k] = (2 * mu * mH) * vth2 * np.sum(Vr2pidVr * np.sum(vr2vx2_ran2 * fH2[:, :, k] * dVx, axis=1)) / (3 * q * nH2[k])

    if New_Grid:
        if debrief > 1:
            print(prompt + 'Computing vr2vx2, vr2vx_vxi2, ErelH_P')
        # Magnitude of total normalized v^2 at each mesh point
        vr2vx2 = np.empty((nvr, nvx, nx))
        for i in range(nvr):
            for k in range(nx):
                vr2vx2[i, :, k] = vr[i] ** 2 + vx ** 2

        # Magnitude of total normalized (v-vxi)^2 at each mesh point
        vr2vx_vxi2 = np.empty((nvr, nvx, nx))
        for i in range(nvr):
            for k in range(nx):
                vr2vx_vxi2[i, :, k] = vr[i] ** 2 + (vx - vxi[k] / vth) ** 2

        # Atomic hydrogen ion energy in local rest frame of plasma at each mesh point
        ErelH_P = 0.5 * mH * vr2vx_vxi2 * vth2 / q
        ErelH_P = np.clip(ErelH_P, 0.1, 2.0e4)

    if New_Protons:
        if debrief > 1:
            print(prompt + 'Computing Ti/mu at each mesh point')
        # Ti/mu at each mesh point
        Ti_mu = np.empty((nvr, nvx, nx))
        for k in range(nx):
            Ti_mu[:, :, k] = Ti[k] / mu

        # Compute Fi_hat
        if debrief > 1:
            print(prompt + 'Computing fi_Hat')
        vx_shift = vxi
        Tmaxwell = Ti
        mol = 1
        fi_hat = create_shifted_maxwellian_core(
            vr, vx, vx_shift, Tmaxwell,
            vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
            vx_Dvx, vr_Dvr, Vr2Vx2_2D,
            jpa, jpb, jna, jnb,
            mol, mu, mH, q,
            debug=debug
    )

    if compute_errors:
        if debrief > 1:
            print(prompt + 'Computing Vbar_Error')

        # Test: The average speed of a non-shifted maxwellian should be 2*Vth*sqrt(Ti(x)/Tnorm)/sqrt(!pi)
        vx_shift = np.zeros(nx)
        Tmaxwell = Ti
        mol = 1
        Maxwell = create_shifted_maxwellian_core(
            vr, vx, vx_shift, Tmaxwell,
            vth, Tnorm, Vr2pidVr, dVx, vol, vth_Dvx,
            vx_Dvx, vr_Dvr, Vr2Vx2_2D,
            jpa, jpb, jna, jnb,
            mol, mu, mH, q,
            debug=debug)

        vbar_test = np.empty((nvr, nvx, ntheta))
        Vbar_Error = np.zeros(nx)
        for m in range(ntheta):
            vbar_test[:, :, m] = vr2vx2[:, :, 0]

        _vbar_test = np.empty((nvr * nvx, ntheta))
        _vbar_test[:] = vth * np.sqrt(vbar_test.reshape(nvr * nvx, ntheta, order='F'))  
        vbar_test = (_vbar_test @ dTheta).reshape(nvr, nvx, order='F')  # Matrix-vector product

        for k in range(nx):
            vbar = np.sum(Vr2pidVr * np.sum(vbar_test * Maxwell[:, :, k] * dVx, axis=1))
            vbar_exact = 2 * vth * np.sqrt(Ti[k] / Tnorm) / np.sqrt(np.pi)
            Vbar_Error[k] = abs(vbar - vbar_exact) / vbar_exact

        if debrief > 0:
            print(prompt + f'Maximum Vbar error = {np.max(Vbar_Error):.3e}')

    if Do_ni:
        if debrief > 1:
            print(prompt + 'Computing ni profile')
        if ni_correct:
            ni = n - nHP
        else:
            ni = n
        ni = np.maximum(ni, 0.01 * n)

    if Do_sigv:
        if debrief > 1:
            print(prompt + 'Computing sigv')
        # Compute sigmav rates for each reaction with option to use rates
        # from CR model of Johnson-Hinnov
        sigv = np.zeros((nx, 3))

        # Reaction R1:  e + H -> e + H(+) + e
        if JH:
            sigv[:, 1] = JHS_Coef(n, Te, no_null=True)
            sigv[:, 2] = JHAlpha_Coef(n, Te, no_null=True)
        else:
            sigv[:, 1] = SigmaV_Ion_H0(Te)
            sigv[:, 2] = SigmaV_rec_H1s(Te)

        # H ionization rate (normalized by vth) = reaction 1
        alpha_ion = n * sigv[:, 1] / vth
        # Recombination rate (normalized by vth) = reaction 2
        Rec = n * sigv[:, 2] / vth


    # Compute total atomic hydrogen source
    Sn = np.zeros((nvr, nvx, nx))

    # Add Recombination (optionally) and User-Supplied Hydrogen Source (velocity space distribution)
    for k in range(nx):
        Sn[:, :, k] = fSH[:, :, k] / vth
        if Recomb:
            Sn[:, :, k] += fi_hat[:, :, k] * ni[k] * Rec[k]

    # Set up arrays for charge exchange and elastic collision computations, if needed
    if Do_v_v2 is True:
        if debrief > 1:
            print(f'{prompt}Computing v_v2, v_v, vr2_vx2, and vx_vx')
        
        # v_v2=(v-v_prime)^2 at each double velocity space mesh point, including theta angle
        v_v2 = np.zeros((nvr, nvx, nvr, nvx, ntheta))

        # vr2_vx2=0.125* [ vr2 + vr2_prime - 2*vr*vr_prime*cos(theta) - 2*(vx-vx_prime)^2 ]
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
        
        energy = v_v2 * (0.5 * mH * vth**2 / q)
        _Sig =  v_v * Sigma_CX_H0(energy)

        integral = np.tensordot(_Sig, dTheta, axes=([-1],[0]))
        SIG_CX_4d = Vr2pidVrdVx * integral
        SIG_CX = SIG_CX_4d.reshape((nvr * nvx, nvr * nvx), order='F')
        
        # SIG_CX is now vr' * sigma_cx(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

    if Do_SIG_H_H == True:
        if debrief > 1:
            print(f'{prompt}Computing SIG_H_H')

        # Compute SIG_H_H for present velocity space grid, if it is needed and has not 
        # already been computed with the present input parameters
        energy   = v_v2 * (0.5 * mH * mu * vth**2 / q)
        _Sig = vr2_vx2 * v_v * Sigma_EL_H_H(energy, vis=True) / 8.0  # shape (nvr,nvx,nvr,nvx,ntheta) 
        
        integral = np.tensordot(_Sig, dTheta, axes=([-1],[0]))
        SIG_H_H_4d = Vr2pidVrdVx * integral
        SIG_H_H = SIG_H_H_4d.reshape((nvr * nvx, nvr * nvx), order='F')
        
        # SIG_H_H is now vr' * sigma_H_H(v_v) * vr2_vx2 * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

    # Compute SIG_H_H2
    if 1 == 1:  # This is a placeholder for the condition to compute SIG_H_H2
    #if Do_SIG_H_H2 == True:
        if debrief > 1:
            print(f'{prompt}Computing SIG_H_H2')

        # Compute SIG_H_H2 for present velocity space grid, if it is needed and has not 
        # already been computed with the present input parameters
        energy = v_v2 * (0.5 * mH * vth**2 / q)  # NOTE: using H energy here for cross-sections tabulated as H->H2
        _Sig = v_v * Sigma_EL_H_HH(energy)

        integral = np.tensordot(_Sig, dTheta, axes=([-1],[0]))
        SIG_H_H2_4d = Vr2pidVrdVx * vx_vx * integral
        SIG_H_H2 = SIG_H_H2_4d.reshape((nvr * nvx, nvr * nvx), order='F')

        # SIG_H_H2 is now vr' *vx_vx * sigma_H_H2(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

    # Compute SIG_H_P
    if Do_SIG_H_P == True:
        if debrief > 1:
            print(f'{prompt}Computing SIG_H_P')
        
        # Compute SIG_H_P for present velocity space grid, if it is needed and has not 
        # already been computed with the present input parameters

        energy   = v_v2 * (0.5 * mH * vth2 / q)          # shape (nvr,nvx,nvr,nvx,ntheta)
        _Sig     = v_v * Sigma_EL_P_H(energy)                           # same shape

        integral = np.tensordot(_Sig, dTheta, axes=([-1],[0]))
        SIG_H_P_4d = Vr2pidVrdVx * vx_vx * integral
        SIG_H_P = SIG_H_P_4d.reshape((nvr * nvx, nvr * nvx), order='F')

    # Compute Alpha_CX for present Ti and ni, if it is needed and has not
    # already been computed with the present parameters

    if Do_Alpha_CX == True:
        if debrief > 1:
            print(f'{prompt}Computing Alpha_CX')

        if Simple_CX == True:
            # Option (B): Use Maxwellian-weighted <σv>

            # Charge Exchange sink rate
            alpha_cx = SigmaV_CX_H0(Ti_mu, ErelH_P) / vth  # shape (nvr, nvx, nx)

            for k in range(nx):
                alpha_cx[:, :, k] *= ni[k]
        
        else:
            # Option (A): Compute SigmaV_CX from sigma directly via SIG_CX
            alpha_cx = np.zeros((nvr, nvx, nx))


            for k in range(nx):
                print(SIG_CX.shape, fi_hat.shape, ni[k].shape)
                Work = (fi_hat[:, :, k] * ni[k]).reshape(nvr*nvx, order='F')
                alpha_cx_flat = np.dot(SIG_CX, Work)
                alpha_cx[:, :, k] = alpha_cx_flat.reshape((nvr, nvx), order='F')

                # NOTE: This is my old script (that didn't work)
                # for k in range(nx):
                #     Work = fi_hat[:, :, k] * ni[k]
                #     alpha_cx[:, :, k] = np.tensordot(SIG_CX, Work, axes=([1], [0]))

            if Do_Alpha_CX_Test:
                alpha_cx_test = SigmaV_CX_H0(Ti_mu, ErelH_P) / vth
                for k in range(nx):
                    alpha_cx_test[:, :, k] *= ni[k]
                print("Compare alpha_cx and alpha_cx_test")

    # -------------------------------------------------------------------------
    # Compute Alpha_H_H2 for inputted fH, if it is needed and has not
    # already been computed with the present input parameters
    # -------------------------------------------------------------------------
        
    if Do_Alpha_H_H2:
        if debrief > 1:
            print(f'{prompt}Computing Alpha_H_H2')
        Alpha_H_H2 = np.zeros((nvr, nvx, nx))
        for k in range(nx):
            Work = fH2[:, :, k].reshape(nvr*nvx, order='F')
            Alpha_H_H2_flat = np.dot(SIG_H_H2, Work)
            Alpha_H_H2[:, :, k] = Alpha_H_H2_flat.reshape((nvr, nvx), order='F')
            
            # NOTE: This is the old attempt that didn't work
            #Work = fH2[:, :, k]
            #Alpha_H_H2[:, :, k] = np.tensordot(SIG_H_H2, Work, axes=([1], [0]))

    # -------------------------------------------------------------------------
    # Compute Alpha_H_P for present Ti and ni 
    # if it is needed and has not already been computed with the present parameters
    # -------------------------------------------------------------------------
    if Do_Alpha_H_P:
        if debrief > 1:
            print(f'{prompt}Computing Alpha_H_P')
        Alpha_H_P = np.zeros((nvr, nvx, nx))
        for k in range(nx):
            Work = (fi_hat[:, :, k] * ni[k]).reshape(nvr*nvx, order='F')
            Alpha_H_P_flat = np.dot(SIG_H_P, Work)
            Alpha_H_P[:, :, k] = Alpha_H_P_flat.reshape((nvr, nvx), order='F')

            # NOTE: This is the old attempt that didn't work
            # Work = fi_hat[:, :, k].reshape(nvr*nvx, order='F')
            # Work = fi_hat[:, :, k] * ni[k]
            # Alpha_H_P[:, :, k] = np.tensordot(SIG_H_P, Work, axes=([1], [0]))

    # Compute nH
    nH = np.zeros(nx)
    for k in range(nx):
        nH[k] = np.sum(Vr2pidVr[:, None] * fH[:, :, k] * dVx[None, :])
        #nH[k] = Vr2pidVr @ (fH[:, :, k] @ dVx)

    if New_H_Seed:
        MH_H_sum = np.zeros((nvr, nvx, nx))
        Delta_nHs = 1.0

    # Compute Side-Wall collision rate
    gamma_wall = np.zeros((nvr, nvx, nx))
    for k in range(nx):
        if PipeDia[k] > 0.0:
            for j in range(nvx):
                gamma_wall[:, j, k] = 2 * vr / PipeDia[k]

    while True: # this is the equivalent of the fH_iterate. Instead of the IDL goto command, we can just use the 'continue' statement to return to the start of this loop

        # Save seed values for iteration
        fHs = fH.copy()
        nHs = nH.copy()

        # Compute Omega values if nH is non-zero
        ii = np.where(nH <= 0)[0]

        if ii.size == 0:  # Proceed only if all nH > 0

            # Compute VxH
            if H_P_EL or H_H2_EL or H_H_EL:
                for k in range(nx):
                    VxH[k] = vth * np.sum(Vr2pidVr[:, None] * fH[:, :, k] * vx[None, :] * dVx[None, :]) / nH[k]

            # Compute Omega_H_P for present fH and Alpha_H_P if H_P elastic collisions are included
            if H_P_EL:
                if debrief > 1:
                    print(f'{prompt}Computing Omega_H_P')
                for k in range(nx):
                    DeltaVx = (VxH[k] - vxi[k]) / vth
                    MagDeltaVx = max(abs(DeltaVx), DeltaVx_tol)
                    DeltaVx = np.sign(DeltaVx) * MagDeltaVx
                    Omega_H_P[k] = np.sum(Vr2pidVr[:, None] * Alpha_H_P[:, :, k] * fH[:, :, k] * dVx[None, :]) / (nH[k] * DeltaVx)
                Omega_H_P = np.maximum(Omega_H_P, 0.0)

            # Compute Omega_H_H2 for present fH and Alpha_H_H2 if H_H2 elastic collisions are included
            if H_H2_EL:
                if debrief > 1:
                    print(f'{prompt}Computing Omega_H_H2')
                Omega_H_H2 = np.zeros(nx)
                for k in range(nx):
                    DeltaVx = (VxH[k] - vxH2[k]) / vth
                    MagDeltaVx = max(abs(DeltaVx), DeltaVx_tol)
                    DeltaVx = np.sign(DeltaVx) * MagDeltaVx
                    Omega_H_H2[k] = np.sum(Vr2pidVr[:, None] * Alpha_H_H2[:, :, k] * fH[:, :, k] * dVx[None, :]) / (nH[k] * DeltaVx)
                Omega_H_H2 = np.maximum(Omega_H_H2, 0.0)

            # Compute Omega_H_H for present fH if H_H elastic collisions are included
            if H_H_EL:
                if debrief > 1:
                    print(f'{prompt}Computing Omega_H_H')

                if np.sum(MH_H_sum) <= 0.0:
                    for k in range(nx):
                        vr2_2vx_ran2 = np.zeros((nvr, nvx))
                        for i in range(nvr):
                            vr2_2vx_ran2[i, :] = vr[i]**2 - 2 * (vx - VxH[k] / vth)**2
                        Wperp_paraH[k] = np.sum(Vr2pidVr[:, None] * vr2_2vx_ran2 * fH[:, :, k] * dVx[None, :]) / nH[k]
                else:
                    for k in range(nx):
                        M_fH = MH_H_sum[:, :, k] - fH[:, :, k]
                        Wperp_paraH[k] = -np.sum(Vr2pidVr[:, None] * vr2_2vx2_2D * M_fH * dVx[None, :]) / nH[k]

                for k in range(nx):
                    Work = fH[:, :, k]
                    Work_flat = Work.reshape((nvr*nvx), order='F')
                    Alpha_H_H_flat = np.dot(SIG_H_H, Work_flat)
                    Alpha_H_H = Alpha_H_H_flat.reshape((nvr, nvx), order='F')

                    # NOTE: This is the old attempt that didn't work
                    #Work = fH[:, :, k]
                    #Alpha_H_H = np.tensordot(SIG_H_H, Work, axes=([1], [0]))

                    Wpp = max(abs(Wperp_paraH[k]), Wpp_tol)
                    Wpp = np.sign(Wperp_paraH[k]) * Wpp
                    Omega_H_H[k] = np.sum(Vr2pidVr[:, None] * (Alpha_H_H * Work) * dVx[None, :]) / (nH[k] * Wpp)
                    
                    # NOTE: Below is my old attempt that didn't work
                    # Omega_H_H[k] = np.sum(Vr2pidVr[:, None] * Alpha_H_H * Work * dVx[None, :]) / (nH[k] * Wpp)
                Omega_H_H = np.maximum(Omega_H_H, 0.0)

        # Total Elastic scattering frequency
        Omega_EL = Omega_H_P + Omega_H_H2 + Omega_H_H

        # Total collision frequency
        alpha_c = np.zeros((nvr, nvx, nx))
        if H_P_CX:
            for k in range(nx):
                alpha_c[:, :, k] = alpha_cx[:, :, k] + alpha_ion[k] + Omega_EL[k] + gamma_wall[:, :, k]
        else:
            for k in range(nx):
                alpha_c[:, :, k] = alpha_ion[k] + Omega_EL[k] + gamma_wall[:, :, k]

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
            raise RuntimeError('Aborting due to x mesh spacing error')
        
        # Define parameters Ak, Bk, Ck, Dk, Fk, Gk
        Ak = np.zeros((nvr, nvx, nx))
        Bk = np.zeros((nvr, nvx, nx))
        Ck = np.zeros((nvr, nvx, nx))
        Dk = np.zeros((nvr, nvx, nx))
        Fk = np.zeros((nvr, nvx, nx))
        Gk = np.zeros((nvr, nvx, nx))

        for k in range(nx - 1):
            dx = x[k + 1] - x[k]
            for j in range(ip_pos[0], nvx):
                denom = 2 * vx[j] + dx * alpha_c[:, j, k + 1]
                Ak[:, j, k] = (2 * vx[j] - dx * alpha_c[:, j, k]) / denom
                Bk[:, j, k] = dx / denom
                Fk[:, j, k] = dx * (Sn[:, j, k + 1] + Sn[:, j, k]) / denom

        for k in range(1, nx):
            dx = x[k] - x[k - 1]
            for j in range(0, ip_pos[0]):
                denom = -2 * vx[j] + dx * alpha_c[:, j, k - 1]
                Ck[:, j, k] = (-2 * vx[j] - dx * alpha_c[:, j, k]) / denom
                Dk[:, j, k] = dx / denom
                Gk[:, j, k] = dx * (Sn[:, j, k] + Sn[:, j, k - 1]) / denom

        # Compute first-flight (0th generation) neutral distribution function
        Beta_CX_sum = np.zeros((nvr, nvx, nx))
        MH_P_sum = np.zeros((nvr, nvx, nx))
        MH_H2_sum = np.zeros((nvr, nvx, nx))
        MH_H_sum = np.zeros((nvr, nvx, nx))
        igen = 0

        if debrief > 0:
            print(f'{prompt}Computing atomic neutral generation#{igen}')
        fHG[:, ip_pos, 0] = fH[:, ip_pos, 0]
        for k in range(nx - 1):
            fHG[:, ip_pos, k + 1] = fHG[:, ip_pos, k] * Ak[:, ip_pos, k] + Fk[:, ip_pos, k]
        for k in reversed(range(1, nx)):
            fHG[:, in_neg, k - 1] = fHG[:, in_neg, k] * Ck[:, in_neg, k] + Gk[:, in_neg, k]

        # Compute first-flight neutral density profile
        for k in range(nx):
            NHG[k, igen] = np.sum(Vr2pidVr[:, None] * fHG[:, :, k] * dVx[None, :])
        
        # Plotting first generation if enabled
        if plot > 1:
            fH1d = np.zeros((nvx, nx))
            for k in range(nx):
                fH1d[:, k] = np.tensordot(Vr2pidVr, fHG[:, :, k], axes=1)
            plt.figure()
            plt.title('First Generation ' + _H)
            plt.ylim(0, np.max(fH1d))
            for i in range(nx):
                plt.plot(vx, fH1d[:, i], color=f'C{(i % 8) + 2}')
            if debug > 0:
                input("Press Return to continue...")
            plt.show()

        # Set total atomic neutral distribution function to first flight generation
        fH = fHG.copy()
        nH = NHG[:, 0].copy()


        if fH_generations == False:
            pass # this is the equivalent of the IDL goto,fH_done command. We will exit the loop and proceed to the fH_done section
            # NOTE: used to be 'break' not 'pass' but I think this is correct

        else:
            # Now we enter the next_generation loop from the IDL (about line 1112 in the original code).
            while True:
                if igen + 1 > Max_Gen:
                    if debrief > 0:
                        print(f'{prompt}Completed {Max_Gen} generations. Returning present solution...')
                    break

                igen += 1
                if debrief > 0:
                    print(f'{prompt}Computing atomic neutral generation#{igen}')

                # Compute Beta_CX from previous generation
                Beta_CX = np.zeros((nvr, nvx, nx))

                if H_P_CX:
                    if debrief > 1:
                        print(f'{prompt}Computing Beta_CX')

                    if Simple_CX:
                        # Option (B): Compute charge exchange source with assumption that CX source neutrals have
                        # ion distribution function
                        for k in range(nx):
                            integrated = np.sum(Vr2pidVr[:, None] * (alpha_cx[:, :, k] * fHG[:, :, k]) * dVx[None, :])
                            Beta_CX[:, :, k] = fi_hat[:, :, k] * integrated

                    else:
                        # Option (A): Compute charge exchange source using fH and vr x sigma x v_v at each velocity mesh point
                        for k in range(nx):
                            Work = fHG[:, :, k].reshape(nvr * nvx, order='F')
                            Beta_CX_flat = np.dot(SIG_CX, Work)
                            Beta_CX[:, :, k] = ni[k] * fi_hat[:, :, k] * Beta_CX_flat.reshape((nvr, nvx), order='F')

                            # NOTE: This is the old attempt that didn't work
                            #Work = fHG[:, :, k]
                            #Beta_CX[:, :, k] = ni[k] * fi_hat[:, :, k] * np.tensordot(SIG_CX, Work, axes=1)

                    # Sum charge exchange source over all generations
                    Beta_CX_sum += Beta_CX

                # Compute MH from previous generation
                MH_H = np.zeros((nvr, nvx, nx))
                MH_P = np.zeros((nvr, nvx, nx))
                MH_H2 = np.zeros((nvr, nvx, nx))
                OmegaM = np.zeros((nvr, nvx, nx))

                if H_H_EL or H_P_EL or H_H2_EL:
                    # Compute VxHG, THG
                    for k in range(nx):
                        VxHG[k] = vth * np.sum(Vr2pidVr[:, None] * fHG[:, :, k] * (vx * dVx)[None, :]) / NHG[k, igen - 1]
                        for i in range(nvr):
                            vr2vx2_ran2[i, :] = vr[i]**2 + (vx - VxHG[k] / vth)**2
                        THG[k] = (mu * mH * vth**2 * np.sum(Vr2pidVr[:, None] * vr2vx2_ran2 * fHG[:, :, k] * dVx[None, :])) / (3 * q * NHG[k, igen - 1])

                    if H_H_EL:
                        if debrief > 1:
                            print(f'{prompt}Computing MH_H')
                        # Compute MH_H
                        vx_shift = VxHG
                        Tmaxwell = THG
                        mol = 1
                        Maxwell = create_shifted_maxwellian_core(
                            vr, vx, vx_shift, Tmaxwell, vth, Tnorm,
                            Vr2pidVr, dVx, vol, vth_Dvx, vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                            jpa, jpb, jna, jnb, mol, mu, mH, q, debug
                        )
                        for k in range(nx):
                            MH_H[:, :, k] = Maxwell[:, :, k] * NHG[k, igen - 1]
                            OmegaM[:, :, k] += Omega_H_H[k] * MH_H[:, :, k]
                        MH_H_sum += MH_H

                    if H_P_EL:
                        if debrief > 1:
                            print(f'{prompt}Computing MH_P')
                        vx_shift = (VxHG + vxi) / 2
                        Tmaxwell = THG + (0.5) * (Ti - THG + mu * mH * (vxi - VxHG)**2 / (6 * q))
                        mol = 1
                        Maxwell = create_shifted_maxwellian_core(
                            vr, vx, vx_shift, Tmaxwell, vth, Tnorm,
                            Vr2pidVr, dVx, vol, vth_Dvx, vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                            jpa, jpb, jna, jnb, mol, mu, mH, q, debug
                        )                    
                        for k in range(nx):
                            MH_P[:, :, k] = Maxwell[:, :, k] * NHG[k, igen - 1]
                            OmegaM[:, :, k] += Omega_H_P[k] * MH_P[:, :, k]
                        MH_P_sum += MH_P

                    if H_H2_EL:
                        if debrief > 1:
                            print(f'{prompt}Computing MH_H2')
                        vx_shift = (VxHG + 2 * vxH2) / 3
                        Tmaxwell = THG + (4. / 9.) * (TH2 - THG + 2 * mu * mH * (vxH2 - VxHG)**2 / (6 * q))
                        mol = 1
                        Maxwell = create_shifted_maxwellian_core(
                            vr, vx, vx_shift, Tmaxwell, vth, Tnorm,
                            Vr2pidVr, dVx, vol, vth_Dvx, vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                            jpa, jpb, jna, jnb, mol, mu, mH, q, debug
                        )                    
                        for k in range(nx):
                            MH_H2[:, :, k] = Maxwell[:, :, k] * NHG[k, igen - 1]
                            OmegaM[:, :, k] += Omega_H_H2[k] * MH_H2[:, :, k]
                        MH_H2_sum += MH_H2


                # Compute next generation atomic distribution
                fHG.fill(0.0)

                for k in range(nx - 1):
                    fHG[:, ip_pos, k + 1] = (
                        Ak[:, ip_pos, k] * fHG[:, ip_pos, k]
                        + Bk[:, ip_pos, k] * (
                            Beta_CX[:, ip_pos, k + 1] + OmegaM[:, ip_pos, k + 1] +
                            Beta_CX[:, ip_pos, k] + OmegaM[:, ip_pos, k]
                        )
                    )

                for k in range(nx - 1, 0, -1):
                    fHG[:, in_neg, k - 1] = (
                        Ck[:, in_neg, k] * fHG[:, in_neg, k]
                        + Dk[:, in_neg, k] * (
                            Beta_CX[:, in_neg, k - 1] + OmegaM[:, in_neg, k - 1] +
                            Beta_CX[:, in_neg, k] + OmegaM[:, in_neg, k]
                        )
                    )

                for k in range(nx):
                    NHG[k, igen] = np.sum(Vr2pidVr[:, None] * fHG[:, :, k] * dVx[None, :])

                if plot > 1:
                    fH1d = np.zeros((nvx, nx), dtype=float)
                    for k in range(nx):
                        fH1d[:, k] = np.sum(Vr2pidVr[:, None] * fHG[:, :, k], axis=0)

                    plt.figure()
                    plt.title(f"{igen} Generation {_H}")
                    plt.plot(vx, fH1d[:, 0])
                    for i in range(nx):
                        plt.plot(vx, np.maximum(fH1d[:, i], 0.9), color=plt.cm.tab10(i % 8))
                    if debug > 0:
                        input("Press return to continue")
                    plt.show()

                # Add result to total neutral distribution function
                fH += fHG
                nH += NHG[:, igen]


                # Compute 'generation error': Delta_nHG=max(NHG(*,igen)/max(nH))
                # and decide if another generation should be computed
                Delta_nHG = np.max(NHG[:, igen] / np.max(nH))

                if fH_iterate:
                    print('fH iteration breakpoint')
                    # If fH 'seed' is being iterated, then do another generation until the 'generation error'
                    # is less than 0.003 times the 'seed error' or is less than TRUNCATE
                    if (Delta_nHG < 0.003 * Delta_nHs) or (Delta_nHG < truncate):
                        break # IDL euqivalent of goto,fH_done
                else:
                    if Delta_nHG < truncate:
                        break # IDL euqivalent of goto,fH_done
                
                # If convergence criteria are not met, then the loop will just continue


        # Now we are in the fH_done part of the loop. This is outside the next_generation loop but inside the fH_iterate loop.
        # This means we have the option to either go back to the start of the fH_iterate loop or exit to the end of the function (just like the IDL version with the goto commands).
        if plot > 0:
            plt.figure()
            plt.yscale('log')
            #plt.xlim([0, 0.3])
            plt.xlim([min(x), max(x)])
            plt.ylim([1e14, 1e19])
            plt.title(f"{_H} Density by Generation")
            plt.xlabel("x (m)")
            plt.ylabel("Density (m⁻³)")
            colours = ['red', 'blue', 'lime', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
            for i in range(igen + 1):
                plt.plot(x, NHG[:, i], color=colours[i % 8], label=f'Gen {i}')
            plt.legend()
            plt.show()

            print('end of iteration breakpoint')
            breakpoint()


        # Compute H density profile
        for k in range(nx):
            nH[k] = np.sum(Vr2pidVr[:, None] * fH[:, :, k] * dVx[None, :])

        if fH_iterate:
            # Compute 'seed error': Delta_nHs=(|nHs-nH|)/max(nH) 
            # If Delta_nHs is greater than 10*truncate then iterate fH
            Delta_nHs = np.max(np.abs(nHs - nH)) / np.max(nH)
            if Delta_nHs > 10 * truncate:
                continue # this continue command sends us back to the start of the fH_iterate 'while True' loop.

        # Now we've finished iterating so we update calculations using the final generate then exit the fH_iterate loop

        # Update Beta_CX_sum using last generation
        Beta_CX = np.zeros((nvr, nvx, nx), dtype=float)
        if H_P_CX:
            if debrief > 1:
                print(f"{prompt}Computing Beta_CX")

            if Simple_CX:
                # Option (B): Compute charge exchange source with assumption that CX source neutrals have
                # ion distribution function
                for k in range(nx):
                    Beta_CX[:, :, k] = fi_hat[:, :, k] * np.sum(
                        Vr2pidVr[:, None] * (alpha_cx[:, :, k] * fHG[:, :, k]) * dVx[None, :],
                        axis=(0, 1)
                    )

            else:
                # Option (A): Compute charge exchange source using fH and vr x sigma x v_v at each velocity mesh point
                for k in range(nx):
                    Work = fHG[:, :, k].reshape(nvr * nvx, order='F')
                    signal_2D = (SIG_CX @ Work).reshape((nvr, nvx), order='F')
                    Beta_CX[:, :, k] = ni[k] * fi_hat[:, :, k] * signal_2D

                    # NOTE: This is the old bit
                    # Work = fHG[:, :, k]
                    # Beta_CX[:, :, k] = ni[k] * fi_hat[:, :, k] * (SIG_CX @ Work)


            # Sum charge exchange source over all generations
            Beta_CX_sum += Beta_CX
        
        # Update MH_*_sum using last generation
        MH_H = np.zeros((nvr, nvx, nx))
        MH_P = np.zeros((nvr, nvx, nx))
        MH_H2 = np.zeros((nvr, nvx, nx))
        OmegaM = np.zeros((nvr, nvx, nx))

        if H_H_EL or H_P_EL or H_H2_EL:
            for k in range(nx):
                VxHG[k] = vth * np.sum(Vr2pidVr * (fHG[:, :, k] @ (vx * dVx))) / NHG[k, igen]
                for i in range(nvr):
                    vr2vx2_ran2[i, :] = vr[i]**2 + (vx - VxHG[k] / vth)**2
                THG[k] = (
                    mu * mH * vth2 *
                    np.sum(Vr2pidVr * (vr2vx2_ran2 * fHG[:, :, k] @ dVx)) /
                    (3 * q * NHG[k, igen])
                )

            if H_H_EL:
                if debrief > 1:
                    print(prompt + 'Computing MH_H')
                vx_shift = VxHG.copy()
                Tmaxwell = THG.copy()
                mol = 1
                Maxwell = create_shifted_maxwellian_core(
                        vr, vx, vx_shift, Tmaxwell, vth, Tnorm,
                        Vr2pidVr, dVx, vol, vth_Dvx, vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                        jpa, jpb, jna, jnb, mol, mu, mH, q, debug
                    )
                for k in range(nx):
                    MH_H[:, :, k] = Maxwell[:, :, k] * NHG[k, igen]
                    OmegaM[:, :, k] += Omega_H_H[k] * MH_H[:, :, k]
                MH_H_sum += MH_H

            if H_P_EL:
                if debrief > 1:
                    print(prompt + 'Computing MH_P')
                vx_shift = (VxHG + vxi) / 2
                Tmaxwell = THG + 0.5 * (Ti - THG + mu * mH * (vxi - VxHG)**2 / (6 * q))
                mol = 1
                Maxwell = create_shifted_maxwellian_core(
                        vr, vx, vx_shift, Tmaxwell, vth, Tnorm,
                        Vr2pidVr, dVx, vol, vth_Dvx, vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                        jpa, jpb, jna, jnb, mol, mu, mH, q, debug
                    )  
                for k in range(nx):
                    MH_P[:, :, k] = Maxwell[:, :, k] * NHG[k, igen]
                    OmegaM[:, :, k] += Omega_H_P[k] * MH_P[:, :, k]
                MH_P_sum += MH_P

            if H_H2_EL:
                if debrief > 1:
                    print(prompt + 'Computing MH_H2')
                vx_shift = (VxHG + 2 * vxH2) / 3
                Tmaxwell = THG + (4. / 9.) * (TH2 - THG + 2 * mu * mH * (vxH2 - VxHG)**2 / (6 * q))
                mol = 1
                Maxwell = create_shifted_maxwellian_core(
                        vr, vx, vx_shift, Tmaxwell, vth, Tnorm,
                        Vr2pidVr, dVx, vol, vth_Dvx, vx_Dvx, vr_Dvr, Vr2Vx2_2D,
                        jpa, jpb, jna, jnb, mol, mu, mH, q, debug
                    )   
                for k in range(nx):
                    MH_H2[:, :, k] = Maxwell[:, :, k] * NHG[k, igen]
                    OmegaM[:, :, k] += Omega_H_H2[k] * MH_H2[:, :, k]
            MH_H2_sum += MH_H2
        
        # Compute remaining moments

        # GammaxH - particle flux in x direction
        for k in range(nx):
            GammaxH[k] = vth * np.sum(Vr2pidVr * (fH[:, :, k] @ (vx * dVx)))

        VxH = GammaxH / nH
        _VxH = VxH / vth

        # magnitude of random velocity at each mesh point
        vr2vx2_ran = np.zeros((nvr, nvx, nx))
        for i in range(nvr):
            for k in range(nx):
                vr2vx2_ran[i, :, k] = vr[i]**2 + (vx - _VxH[k])**2

        # pH - pressure
        for k in range(nx):
            pH[k] = (mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2_ran[:, :, k] * fH[:, :, k]) @ dVx)) / (3 * q)

        # TH - temperature
        TH = pH / nH

        for k in range(nx):
            piH_xx[k] = (mu * mH) * vth2 * np.sum(Vr2pidVr * (fH[:, :, k] @ (dVx * (vx - _VxH[k])**2))) / q - pH[k]

        for k in range(nx):
            piH_yy[k] = (mu * mH) * vth2 * 0.5 * np.sum((Vr2pidVr * vr**2)[:, None] * (fH[:, :, k] @ dVx)) / q - pH[k]

        piH_zz = piH_yy.copy()

        for k in range(nx):
            qxH[k] = 0.5 * (mu * mH) * vth**3 * np.sum(Vr2pidVr * ((vr2vx2_ran[:, :, k] * fH[:, :, k]) @ (dVx * (vx - _VxH[k]))))

        # C = RHS of Boltzman equation for total fH
        for k in range(nx):
            C = vth * (
                Sn[:, :, k] + Beta_CX_sum[:, :, k] - alpha_c[:, :, k] * fH[:, :, k] +
                Omega_H_P[k] * MH_P_sum[:, :, k] +
                Omega_H_H2[k] * MH_H2_sum[:, :, k] +
                Omega_H_H[k] * MH_H_sum[:, :, k]
            )

            QH[k] = 0.5 * (mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2_ran[:, :, k] * C) @ dVx))
            RxH[k] = (mu * mH) * vth * np.sum(Vr2pidVr * (C @ (dVx * (vx - _VxH[k]))))
            NetHSource[k] = np.sum(Vr2pidVr * (C @ dVx))
            Sion[k] = vth * nH[k] * alpha_ion[k]
            SourceH[k] = np.sum(Vr2pidVr * (fSH[:, :, k] @ dVx))
            WallH[k] = vth * np.sum(Vr2pidVr * ((gamma_wall[:, :, k] * fH[:, :, k]) @ dVx))

            if Recomb:
                SRecomb[k] = vth * ni[k] * Rec[k]
            else:
                SRecomb[k] = 0.0

            if H_P_CX:
                CCX = vth * (Beta_CX_sum[:, :, k] - alpha_cx[:, :, k] * fH[:, :, k])
                RxHCX[k] = (mu * mH) * vth * np.sum(Vr2pidVr * (CCX @ (dVx * (vx - _VxH[k]))))
                EHCX[k] = 0.5 * (mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * CCX) @ dVx))

            if H_H2_EL:
                CH_H2 = vth * Omega_H_H2[k] * (MH_H2_sum[:, :, k] - fH[:, :, k])
                RxH2_H[k] = (mu * mH) * vth * np.sum(Vr2pidVr * (CH_H2 @ (dVx * (vx - _VxH[k]))))
                EH2_H[k] = 0.5 * (mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * CH_H2) @ dVx))

            if H_P_EL:
                CH_P = vth * Omega_H_P[k] * (MH_P_sum[:, :, k] - fH[:, :, k])
                RxP_H[k] = (mu * mH) * vth * np.sum(Vr2pidVr * (CH_P @ (dVx * (vx - _VxH[k]))))
                EP_H[k] = 0.5 * (mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * CH_P) @ dVx))

            CW_H = -vth * (gamma_wall[:, :, k] * fH[:, :, k])
            RxW_H[k] = (mu * mH) * vth * np.sum(Vr2pidVr * (CW_H @ (dVx * (vx - _VxH[k]))))
            EW_H[k] = 0.5 * (mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * CW_H) @ dVx))

            if H_H_EL:
                CH_H = vth * Omega_H_H[k] * (MH_H_sum[:, :, k] - fH[:, :, k])
                for i in range(nvr):
                    vr2_2vx_ran2[i, :] = vr[i]**2 - 2 * (vx - _VxH[k])**2
                Epara_PerpH_H[k] = -0.5 * (mu * mH) * vth2 * np.sum(Vr2pidVr * ((vr2_2vx_ran2 * CH_H) @ dVx))

        # qxH_total
        qxH_total = (0.5 * nH * mu * mH * VxH**2 + 2.5 * pH * q) * VxH + q * piH_xx * VxH + qxH

        # QH_total
        QH_total = QH + RxH * VxH + 0.5 * mu * mH * NetHSource * VxH**2

        # Albedo
        AlbedoH = 0.0

        gammax_plus = vth * np.sum(Vr2pidVr * (fH[:, ip_pos, 0] @ (vx[ip_pos] * dVx[ip_pos])))
        gammax_minus = vth * np.sum(Vr2pidVr * (fH[:, in_neg, 0] @ (vx[in_neg] * dVx[in_neg])))

        if abs(gammax_plus) > 0:
            AlbedoH = -gammax_minus / gammax_plus

        # Compute Mesh Errors
        mesh_error = np.zeros((nvr, nvx, nx))
        max_mesh_error = 0.0
        min_mesh_error = 0.0
        mtest = 5
        moment_error = np.zeros((nx, mtest))
        max_moment_error = np.zeros(mtest)
        C_error = np.zeros(nx)
        CX_error = np.zeros(nx)
        H_H_error = np.zeros((nx, 3))
        H_H2_error = np.zeros((nx, 3))
        H_P_error = np.zeros((nx, 3))
        max_H_H_error = np.zeros(3)
        max_H_H2_error = np.zeros(3)
        max_H_P_error = np.zeros(3)


        if compute_errors:
            if debrief > 1:
                print(prompt + 'Computing Collision Operator, Mesh, and Moment Normalized Errors')

            NetHSource2 = SourceH + SRecomb - Sion - WallH
            C_error = np.abs(NetHSource - NetHSource2) / np.maximum(np.abs(NetHSource), np.abs(NetHSource2))


            # Test conservation of particles for charge exchange operator
            if H_P_CX:
                for k in range(nx):
                    CX_A = np.sum(Vr2pidVr[:, None] * (alpha_cx[:, :, k] * fH[:,:,k]) * dVx[None, :])
                    CX_B = np.sum(Vr2pidVr[:, None] * (Beta_CX_sum[:, :, k] * dVx[None, :]))
                    CX_error[k] = abs(CX_A - CX_B) / max(abs(CX_A), abs(CX_B))

                    # NOTE: below is old attempt that didn't work
                    #CX_A = np.sum(Vr2pidVr * (alpha_cx[:, :, k] * fH[:, :, k]) @ dVx)
                    #CX_B = np.sum(Vr2pidVr * Beta_CX_sum[:, :, k] @ dVx)
                    #CX_error[k] = abs(CX_A - CX_B) / max(abs(CX_A), abs(CX_B))

            # Test conservation of particles, x momentum, and total energy of elastic collision operators
            for m in range(3):
                for k in range(nx):
                    if m < 2:
                        TfH = np.sum(Vr2pidVr * (fH[:, :, k] @ (dVx * vx**m)))
                    else:
                        TfH = np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * fH[:, :, k]) @ dVx))

                    if H_H_EL:
                        if m < 2:
                            TH_H = np.sum(Vr2pidVr * (MH_H_sum[:, :, k] @ (dVx * vx**m)))
                        else:
                            TH_H = np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * MH_H_sum[:, :, k]) @ dVx))
                        H_H_error[k, m] = abs(TfH - TH_H) / max(abs(TfH), abs(TH_H))

                    if H_H2_EL:
                        if m < 2:
                            TH_H2 = np.sum(Vr2pidVr * (MH_H2_sum[:, :, k] @ (dVx * vx**m)))
                        else:
                            TH_H2 = np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * MH_H2_sum[:, :, k]) @ dVx))
                        H_H2_error[k, m] = abs(TfH - TH_H2) / max(abs(TfH), abs(TH_H2))

                    if H_P_EL:
                        if m < 2:
                            TH_P = np.sum(Vr2pidVr * (MH_P_sum[:, :, k] @ (dVx * vx**m)))
                        else:
                            TH_P = np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * MH_P_sum[:, :, k]) @ dVx))
                        H_P_error[k, m] = abs(TfH - TH_P) / max(abs(TfH), abs(TH_P))

                max_H_H_error[m] = np.max(H_H_error[:, m])
                max_H_H2_error[m] = np.max(H_H2_error[:, m])
                max_H_P_error[m] = np.max(H_P_error[:, m])

            if CI_Test:
                # Compute Momentum transfer rate via full collision integrals for charge exchange and mixed elastic scattering
                # Then compute error between this and actual momentum transfer resulting from CX and BKG (elastic) models

                if H_P_CX:
                    print(f"{prompt}Computing P -> H Charge Exchange Momentum Transfer")
                    _Sig = v_v * Sigma_CX_H0(v_v2 * (0.5 * mH * vth2 / q))
                    sig_4D = np.dot(_Sig, dTheta).reshape((nvr, nvx, nvr, nvx), order='F')  # (40000,)
                    SIG_VX_CX_4D = Vr2pidVrdVx * vx_vx * sig_4D                 # (10,20,10,20)
                    SIG_VX_CX = SIG_VX_CX_4D.reshape((nvr*nvx, nvr*nvx), order='F') # (200,200)

                    # NOTE: below is the old attempt that didn't work
                    # SIG_VX_CX = Vr2pidVrdVx * vx_vx @ (_Sig @ dTheta)

                    alpha_vx_cx = np.zeros((nvr, nvx, nx))
                    for k in range(nx):
                        Work = (fi_hat[:, :, k] * ni[k]).reshape(nvr*nvx, order='F')
                        alpha_vx_cx_flat = np.dot(SIG_VX_CX, Work)
                        alpha_vx_cx[:, :, k] = alpha_vx_cx_flat.reshape((nvr, nvx), order='F')

                        # NOTE: below is my old attempt that didn't work
                        # Work = fi_hat[:, :, k] * ni[k]
                        # alpha_vx_cx[:, :, k] = SIG_VX_CX @ Work

                    for k in range(nx):
                        RxCI_CX[k] = -(mu * mH) * vth2 * np.sum(Vr2pidVr * ((alpha_vx_cx[:, :, k] * fH[:, :, k]) @ dVx))

                    norm = np.max(np.abs(np.array([RxHCX, RxCI_CX])))

                    for k in range(nx):
                        CI_CX_error[k] = np.abs(RxHCX[k] - RxCI_CX[k]) / norm

                    print(f"{prompt}Maximum normalized momentum transfer error in CX collision operator: {np.max(CI_CX_error)}")

                if H_P_EL:
                    for k in range(nx):
                        RxCI_P_H[k] = -0.5 * (mu * mH) * vth2 * np.sum(Vr2pidVr * ((Alpha_H_P[:, :, k] * fH[:, :, k]) @ dVx))

                    norm = np.max(np.abs(np.array([RxP_H, RxCI_P_H])))
                    CI_P_H_error = np.abs(RxP_H - RxCI_P_H) / norm
                    print(f"{prompt}Maximum normalized momentum transfer error in P -> H elastic BKG collision operator: {np.max(CI_P_H_error)}")

                if H_H2_EL:
                    for k in range(nx):
                        RxCI_H2_H[k] = -(2.0 / 3.0) * (mu * mH) * vth2 * np.sum(Vr2pidVr * ((Alpha_H_H2[:, :, k] * fH[:, :, k]) @ dVx))

                    norm = np.max(np.abs(np.array([RxH2_H, RxCI_H2_H])))
                    CI_H2_H_error = np.abs(RxH2_H - RxCI_H2_H) / norm
                    print(f"{prompt}Maximum normalized momentum transfer error in H2 -> H elastic BKG collision operator: {np.max(CI_H2_H_error)}")

                if H_H_EL:
                    for k in range(nx):
                        Work = fH[:, :, k].reshape(nvr*nvx, order='F')
                        Alpha_H_H_flat = np.dot(SIG_H_H, Work)
                        Alpha_H_H = Alpha_H_H_flat.reshape((nvr, nvx), order='F')
                        Epara_Perp_CI[k] = 0.5 * (mu * mH) * vth3 * np.sum(Vr2pidVr * ((Alpha_H_H * fH[:, :, k]) @ dVx))

                        # NOTE: below is old attempt that didn't work
                        # Work = fH[:, :, k]
                        # Alpha_H_H = SIG_H_H @ Work
                        # Epara_Perp_CI[k] = 0.5 * (mu * mH) * vth3 * np.sum(Vr2pidVr * ((Alpha_H_H * fH[:, :, k]) @ dVx))

                    norm = np.max(np.abs(np.array([Epara_PerpH_H, Epara_Perp_CI])))
                    CI_H_H_error = np.abs(Epara_PerpH_H - Epara_Perp_CI) / norm
                    print(f"{prompt}Maximum normalized perp/parallel energy transfer error in H -> H elastic BKG collision operator: {np.max(CI_H_H_error)}")
            

            # Mesh Point Error based on fH satisfying Boltzmann equation
            T1 = np.zeros((nvr, nvx, nx))
            T2 = np.zeros_like(T1)
            T3 = np.zeros_like(T1)
            T4 = np.zeros_like(T1)
            T5 = np.zeros_like(T1)

            for k in range(nx - 1):
                for j in range(nvx):
                    T1[:, j, k] = 2 * vx[j] * (fH[:, j, k + 1] - fH[:, j, k]) / (x[k + 1] - x[k])
                T2[:, :, k] = Sn[:, :, k + 1] + Sn[:, :, k]
                T3[:, :, k] = Beta_CX_sum[:, :, k + 1] + Beta_CX_sum[:, :, k]
                T4[:, :, k] = alpha_c[:, :, k + 1] * fH[:, :, k + 1] + alpha_c[:, :, k] * fH[:, :, k]
                T5[:, :, k] = (
                    Omega_H_P[k + 1] * MH_P_sum[:, :, k + 1] +
                    Omega_H_H2[k + 1] * MH_H2_sum[:, :, k + 1] +
                    Omega_H_H[k + 1] * MH_H_sum[:, :, k + 1] +
                    Omega_H_P[k] * MH_P_sum[:, :, k] +
                    Omega_H_H2[k] * MH_H2_sum[:, :, k] +
                    Omega_H_H[k] * MH_H_sum[:, :, k]
                )

                max_terms = np.max(np.stack([np.abs(T1[:,:,k]), np.abs(T2[:,:,k]), np.abs(T3[:,:,k]), np.abs(T4[:,:,k]), np.abs(T5[:,:,k])]))
                mesh_error[:, :, k] = np.abs(T1[:, :, k] - T2[:, :, k] - T3[:, :, k] + T4[:, :, k] - T5[:, :, k]) / max_terms

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
                    moment_error[k, m] = np.abs(MT1 - MT2 - MT3 + MT4 - MT5) / np.max(np.abs([MT1, MT2, MT3, MT4, MT5]))
                max_moment_error[m] = np.max(moment_error[:, m])

            # Compute error in qxH_total
            #
            #    qxH_total2 total neutral heat flux profile (watts m^-2)
            #               This is the total heat flux transported by the neutrals
            #               computed in a different way from:
            #
            #               qxH_total2(k)=vth3*total(Vr2pidVr*((vr2vx2(*,*,k)*fH(*,*,k))#(Vx*dVx)))*0.5*(mu*mH)
            #
            #               This should agree with qxH_total if the definitions of nH, pH, piH_xx,
            #               TH, VxH, and qxH are coded correctly.

            qxH_total2 = np.zeros(nx)
            for k in range(nx):
                qxH_total2[k] = 0.5 * (mu * mH) * vth3 * np.sum(Vr2pidVr * ((vr2vx2[:, :, k] * fH[:, :, k]) @ (vx * dVx)))

            qxH_total_error = np.abs(qxH_total - qxH_total2) / np.max(np.abs([qxH_total, qxH_total2]))

            # Compute error in QH_total
            Q1 = np.zeros(nx)
            Q2 = np.zeros(nx)
            for k in range(nx - 1):
                Q1[k] = (qxH_total[k + 1] - qxH_total[k]) / (x[k + 1] - x[k])
                Q2[k] = 0.5 * (QH_total[k + 1] + QH_total[k])
            QH_total_error = np.abs(Q1 - Q2) / np.max(np.abs([Q1, Q2]))

            if debrief > 0:
                print(f"{prompt}Maximum particle conservation error of total collision operator: {np.max(C_error)}")
                print(f"{prompt}Maximum H_P_CX  particle conservation error: {np.max(CX_error)}")
                print(f"{prompt}Maximum H_H_EL  particle conservation error: {max_H_H_error[0]}")
                print(f"{prompt}Maximum H_H_EL  x-momentum conservation error: {max_H_H_error[1]}")
                print(f"{prompt}Maximum H_H_EL  total energy conservation error: {max_H_H_error[2]}")
                print(f"{prompt}Maximum H_H2_EL particle conservation error: {max_H_H2_error[0]}")
                print(f"{prompt}Maximum H_P_EL  particle conservation error: {max_H_P_error[0]}")
                print(f"{prompt}Average mesh_error = {ave_mesh_error}")
                print(f"{prompt}Maximum mesh_error = {max_mesh_error}")
                for m in range(5):
                    print(f"{prompt}Maximum fH vx^{m} moment error: {max_moment_error[m]}")
                print(f"{prompt}Maximum qxH_total error = {np.max(qxH_total_error)}")
                print(f"{prompt}Maximum QH_total error = {np.max(QH_total_error)}")
                if debug > 0:
                    input("Press return to continue...")
        
        if plot > 1:
            fH1d = np.zeros((nvx, nx))
            for k in range(nx):
                fH1d[:, k] = Vr2pidVr @ fH[:, :, k]

            ymin = np.min(fH1d)
            ymax = np.max(fH1d)

            plt.figure()
            plt.title(f'{_H} Velocity Distribution Function: fH(Vx)')
            plt.xlabel('Vx/Vth')
            plt.ylabel('fH1d')
            plt.ylim([ymin, ymax])

            for i in range(nx):
                plt.plot(vx, fH1d[:, i], color=plt.cm.tab10((i % 6) + 2))

            plt.show()

        
        mid1 = np.argmin(np.abs(x - 0.7 * (np.max(x) + np.min(x)) / 2))
        mid2 = np.argmin(np.abs(x - 0.85 * (np.max(x) + np.min(x)) / 2))
        mid3 = np.argmin(np.abs(x - 0.5 * (np.max(x) + np.min(x))))
        mid4 = np.argmin(np.abs(x - 1.15 * (np.max(x) + np.min(x)) / 2))
        mid5 = np.argmin(np.abs(x - 1.3 * (np.max(x) + np.min(x)) / 2))

        # Density Profiles
        if plot > 0:
            data = np.stack([nH, n, nHP, nH2])
            jp = np.where(data > 0)
            yrange = [np.min(data[jp]), np.max(data[jp])]

            plt.figure()
            plt.title('Density Profiles')
            plt.xlabel('x (meters)')
            plt.ylabel('m$^{-3}$')
            plt.yscale('log')
            plt.xlim([np.min(x), np.max(x)])
            plt.ylim(yrange)

            plt.plot(x, nH, label=_H, color='tab:red')
            plt.text(x[mid1], 1.2 * nH[mid1], _H, color='tab:red')

            plt.plot(x, n, label='e-', color='tab:blue')
            plt.text(x[mid2], 1.2 * n[mid2], 'e-', color='tab:blue')

            plt.plot(x, nH2, label=_HH, color='tab:green')
            plt.text(x[mid3], 1.2 * nH2[mid3], _HH, color='tab:green')

            plt.plot(x, nHP, label=_Hp, color='tab:orange')
            plt.text(x[mid4], 1.2 * nHP[mid4], _Hp, color='tab:orange')

            plt.legend()
            plt.show()

        # Temperature Profiles
        if plot > 0:
            data = np.stack([TH, Te, Ti, THP, TH2])
            jp = np.where(data > 0)
            yrange = [np.min(data[jp]), np.max(data[jp])]

            plt.figure()
            plt.title('Temperature Profiles')
            plt.xlabel('x (meters)')
            plt.ylabel('eV')
            plt.yscale('log')
            plt.xlim([np.min(x), np.max(x)])
            plt.ylim(yrange)

            plt.plot(x, TH, label=_H, color='tab:red')
            plt.text(x[mid1], 1.2 * TH[mid1], _H, color='tab:red')

            plt.plot(x, Ti, label=_p, color='gold')
            plt.text(1.1 * x[mid2], 1.2 * Ti[mid2], _p, color='gold')

            plt.plot(x, Te, label='e-', color='tab:blue')
            plt.text(x[mid3], 1.2 * Te[mid3], 'e-', color='tab:blue')

            plt.plot(x, TH2, label=_HH, color='tab:green')
            plt.text(x[mid4], 1.2 * TH2[mid4], _HH, color='tab:green')

            plt.plot(x, THP, label=_Hp, color='tab:orange')
            plt.text(x[mid5], 1.2 * THP[mid5], _Hp, color='tab:orange')

            plt.legend()
            plt.show()


        # Source and Sink Profiles
        if plot > 0:
            data = np.stack([SourceH, SRecomb, Sion])
            jp = np.where(data > 0)
            yrange = [np.min(data[jp]), np.max(data[jp])]

            plt.figure()
            plt.title(f'{_H} Source and Sink Profiles')
            plt.xlabel('x (meters)')
            plt.ylabel('m$^{-3}$ s$^{-1}$')
            plt.yscale('log')
            plt.xlim([np.min(x), np.max(x)])
            plt.ylim(yrange)

            plt.plot(x, SourceH, label=f'{_HH} Dissociation', color='tab:red')
            plt.text(x[mid1], 1.2 * SourceH[mid1], f'{_HH} Dissociation', color='tab:red')

            plt.plot(x, SRecomb, label=f'{_p} Recombination', color='tab:blue')
            plt.text(x[mid2], 1.2 * SRecomb[mid2], f'{_p} Recombination', color='tab:blue')

            plt.plot(x, Sion, label=f'{_H} Ionization', color='tab:green')
            plt.text(x[mid3], 1.2 * Sion[mid3], f'{_H} Ionization', color='tab:green')

            plt.plot(x, WallH, label=f'{_H} Side-Wall Loss', color='tab:orange')
            plt.text(x[mid4], 1.2 * WallH[mid4], f'{_H} Side-Wall Loss', color='tab:orange')

            plt.legend()
            plt.show()
            if pause:
                input("Press Enter to continue...")

        # Fluxes
        if plot > 0:
            gammaxH_plus = np.zeros(nx)
            gammaxH_minus = np.zeros(nx)
            for k in range(nx):
                gammaxH_plus[k] = vth * np.sum(Vr2pidVr * (fH[:, ip_pos, k] @ (vx[ip_pos] * dVx[ip_pos])))
                gammaxH_minus[k] = vth * np.sum(Vr2pidVr * (fH[:, in_neg, k] @ (vx[in_neg] * dVx[in_neg])))

            data = np.stack([gammaxH_plus, gammaxH_minus, GammaxH])
            jp = np.where(data < 1.0e32)
            yrange = [np.min(data[jp]), np.max(data[jp])]

            plt.figure()
            plt.title(f'{_H} Fluxes')
            plt.xlabel('x (meters)')
            plt.ylabel('m$^{-2}$ s$^{-1}$')
            plt.xlim([np.min(x), np.max(x)])
            plt.ylim(yrange)

            plt.plot(x, GammaxH, label='Γ', color='tab:red')
            plt.text(x[mid1], GammaxH[mid1], 'Γ', color='tab:red')

            plt.plot(x, gammaxH_plus, label='Γ+', color='tab:blue')
            plt.text(x[mid2], gammaxH_plus[mid2], 'Γ+', color='tab:blue')

            plt.plot(x, gammaxH_minus, label='Γ−', color='tab:green')
            plt.text(x[mid3], gammaxH_minus[mid3], 'Γ−', color='tab:green')

            plt.legend()
            plt.show()

        np.savez('kinetic_H_results.npz',
            vx=vx,
            vr=vr,
            x=x,
            Tnorm=Tnorm,
            mu=mu,
            Ti=Ti,
            vxi=vxi,
            Te=Te,
            n=n,
            fHBC=fHBC,
            GammaxHBC=GammaxHBC,
            PipeDia=PipeDia,
            fH2=fH2,
            fSH=fSH,
            nHP=nHP,
            THP=THP,

            fH=fH,
            Simple_CX=Simple_CX,
            JH=JH,
            Recomb=Recomb,
            H_H_EL=H_H_EL,
            H_P_EL=H_P_EL,
            H_H2_EL=H_H2_EL,
            H_P_CX=H_P_CX,
            nH=nH.astype(np.float32),
            GammaxH=GammaxH.astype(np.float32),
            VxH=VxH.astype(np.float32),
            pH=pH.astype(np.float32),
            TH=TH.astype(np.float32),
            qxH=qxH.astype(np.float32),
            qxH_total=qxH_total.astype(np.float32),
            NetHSource=NetHSource.astype(np.float32),
            Sion=Sion.astype(np.float32),
            QH=QH.astype(np.float32),
            RxH=RxH.astype(np.float32),
            QH_total=QH_total.astype(np.float32),
            AlbedoH=np.float32(AlbedoH),
            piH_xx=piH_xx.astype(np.float32),
            piH_yy=piH_yy.astype(np.float32),
            piH_zz=piH_zz.astype(np.float32),
            RxH2_H=RxH2_H.astype(np.float32),
            RxP_H=RxP_H.astype(np.float32),
            RxHCX=RxHCX.astype(np.float32),
            EH2_H=EH2_H.astype(np.float32),
            EP_H=EP_H.astype(np.float32),
            EHCX=EHCX.astype(np.float32),
            Epara_PerpH_H=Epara_PerpH_H.astype(np.float32))
        
        if debug > 0:
            print(f"{prompt}Finished")

        seeds = {
            'vx_s': vx,
            'vr_s': vr,
            'x_s': x,
            'Tnorm_s': Tnorm,
            'mu_s': mu,
            'Ti_s': Ti,
            'Te_s': Te,
            'n_s': n,
            'vxi_s': vxi,
            'fHBC_s': fHBC,
            'GammaxHBC_s': GammaxHBC,
            'PipeDia_s': PipeDia,
            'fH2_s': fH2,
            'fSH_s': fSH,
            'nHP_s': nHP,
            'THP_s': THP,
            'fH_s': fH,
            'Simple_CX_s': Simple_CX,
            'JH_s': JH,
            'Recomb_s': Recomb,
            'H_H_EL_s': H_H_EL,
            'H_P_EL_s': H_P_EL,
            'H_H2_EL_s': H_H2_EL,
            'H_P_CX_s': H_P_CX,
        }


        results = {
            'fH': fH,
            'nH': nH,
            'GammaxH': GammaxH,
            'VxH': VxH,
            'pH': pH,
            'TH': TH,
            'qxH': qxH,
            'qxH_total': qxH_total,
            'NetHSource': NetHSource,
            'Sion': Sion,
            'QH': QH,
            'RxH': RxH,
            'QH_total': QH_total,
            'AlbedoH': AlbedoH,
            'WallH': WallH,
        }

        output = {
            'piH_xx': piH_xx,
            'piH_yy': piH_yy,
            'piH_zz': piH_zz,
            'RxHCX': RxHCX,
            'RxH2_H': RxH2_H,
            'RxP_H': RxP_H,
            'RxW_H': RxW_H,
            'EHCX': EHCX,
            'EH2_H': EH2_H,
            'EP_H': EP_H,
            'EW_H': EW_H,
            'Epara_PerpH_H': Epara_PerpH_H,
            'SourceH': SourceH,
            'SRecomb': SRecomb,
        }

        errors = {
            'Max_dx': Max_dx,
            'Vbar_Error': Vbar_Error,
            'mesh_error': mesh_error,
            'moment_error': moment_error,
            'C_Error': C_error,
            'CX_Error': CX_error,
            'H_H_error': H_H_error,
            'qxH_total_error': qxH_total_error,
            'QH_total_error': QH_total_error,
        }

        internal = {
            'vr2vx2': vr2vx2,
            'vr2vx_vxi2': vr2vx_vxi2,
            'fi_hat': fi_hat,
            'ErelH_P': ErelH_P,
            'Ti_mu': Ti_mu,
            'ni': ni,
            'sigv': sigv,
            'alpha_ion': alpha_ion,
            'v_v2': v_v2,
            'v_v': v_v,
            'vr2_vx2': vr2_vx2,
            'vx_vx': vx_vx,
            'Vr2pidVrdVx': Vr2pidVrdVx,
            'SIG_CX': SIG_CX,
            'SIG_H_H': SIG_H_H,
            'SIG_H_H2': SIG_H_H2,
            'SIG_H_P': SIG_H_P,
            'Alpha_CX': Alpha_CX,
            'Alpha_H_H2': Alpha_H_H2,
            'Alpha_H_P': Alpha_H_P,
            'MH_H_sum': MH_H_sum,
            'Delta_nHs': Delta_nHs,
            'Sn': Sn,
            'Rec': Rec
        }

        H2_moments = {
            'nH2': nH2,
            'vxH2': vxH2,
            'TH2': TH2,
        }

        return results, seeds, output, errors, H2_moments, internal