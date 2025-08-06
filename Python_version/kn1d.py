# NOTE: in IDL, there are some things stored in common memory blocks in IDL.
# This means that, if I have recently run a different KN1D run, there will be non-zero seeds
# for the following values in IDL:

# fH_s, fH2_s, nH2_s, nHP_s, THP_s, SpH2_s

# I have not implemented this in python, but if i get to a case where I am iterating KN1D runs then I will probably need to account for these seeds in some way.

#
# KN1D.py
#
# Computes the molecular and atomic neutral profiles for inputted profiles
# of Ti(x), Te(x), n(x), and molecular neutral pressure, GaugeH2, at the boundary using
# IDL routines Kinetic_H and Kinetic_H2. Molecular densities, ionization profiles,
# atomic densities and moments of the atomic distribution function, such as
# T0(x), Qin(x), qx0_total(x),... are returned. 
#
#     It is assumed that molecular neutrals with temperature equal to the wall temperature
# (~ 1/40 eV) are attacking the plasma at x=x(0).
#
# History: First coding 5/1/2001  -  B. LaBombard
#
# Translated into Python by J. Dunsmore, 06-Aug-2025
#________________________________________________________________________________
#

import os
import numpy as np
import matplotlib.pyplot as plt
from create_kinetic_h2_mesh import Create_Kinetic_H2_Mesh
from create_kinetic_h_mesh import Create_Kinetic_H_Mesh
from create_shifted_maxwellian import create_shifted_maxwellian
from scipy.interpolate import interp1d
from make_dvr_dvx import make_dvr_dvx
from interp_fvrvxx import Interp_fVrVxX
from kinetic_h2 import kinetic_h2
from interp_scalarx import interp_scalarx
from kinetic_h import kinetic_h
from lyman_alpha import Lyman_Alpha
from balmer_alpha import Balmer_Alpha
from kn1d_include import kn1d_include
from kn1d_limiter_include import kn1d_limiter_include

from scipy.io.idl import readsav

def KN1D(
    x, xlimiter, xsep, GaugeH2, mu, Ti, Te, n, vxi, LC, PipeDia,
    truncate=1e-3, refine=False, File=None, NewFile=False, ReadInput=False,

    # collisions
    H2_H2_EL=True, H2_P_EL=True, H2_H_EL=True, H2_HP_CX=True,
    H_H_EL=True, H_P_EL=True, H_P_CX=True, Simple_CX=True,

    # mesh inputs
    H2Gridfctr=0.3, HGridfctr=0.3,

    compute_errors=False,
    plot=False, debug=False, debrief=False, pause=False,
    Hplot=False, Hdebug=False, Hdebrief=False, Hpause=False,
    H2plot=False, H2debug=False, H2debrief=False, H2pause=False,

    # KN1D_internal inputs
    fH_s=None, fH2_s=None, nH2_s=None, nHP_s=None, THP_s=None,
    SpH2_s=None,
):
    """
    Input:
    -----------
    x	- fltarr(nx), cross-field coordinate (meters)
    xlimiter - float, cross-field coordinate of limiter edge (meters) (for graphic on plots)
    xsep	- float, cross-field coordinate separatrix (meters) (for graphic on plots)
    GaugeH2	- float, Molecular pressure (mtorr)
    mu	- float, 1=hydrogen, 2=deuterium
    Ti	- fltarr(nx), ion temperature profile (eV)
    Te	- fltarr(nx), electron temperature profile (eV)
    n	- fltarr(nx), density profile (m^-3)
    vxi	- fltarr(nx), plasma velocity profile [negative is towards 'wall' (m s^-1)]
    LC	- fltarr(nx), connection length (surface to surface) along field lines to nearest limiters (meters)
                Zero values of LC are treated as LC=infinity.
    PipeDia	- fltarr(nx), effective pipe diameter (meters)
        This variable allows collisions with the 'side-walls' to be simulated.
        If this variable is undefined, then PipeDia set set to zero. Zero values
        of PipeDia are ignored (i.e., treated as an infinite diameter).

    Keyword Input:
    -----------
    truncate	- float, this parameter is also passed to Kinetic_H and Kinetic_H2.
                fH and fH2 are refined by iteration via routines Kinetic_H2 and Kinetic_H
        until the maximum change in molecular neutral density (over its profile) normalized to 
        the maximum value of molecular density is less than this 
            value in a subsequent iteration. Default value is 1.0e-3

    refine  - if set, then use previously computed atomic and molecular distribution functions
        stored in internal common block (if any) or from FILE (see below) as the initial 
                'seed' value'

        file  - string, if not null, then read in 'file'.kn1d_mesh save set and compare contents
                to the present input parameters and computational mesh. If these are the same
        then read results from previous run in 'file'.kn1d_H2 and 'file'.kn1d_H.

    Newfile - if set, then do not generate an error and exit if 'file'.KN1D_mesh or 'file'.KN1D_H2
                or 'file'.KN1D_H do not exist or differ from the present input parameters. Instead, write 
                new mesh and output files on exiting.

    ReadInput - if set, then reset all input variables to that contained in 'file'.KN1D_input


    Collision inputs:
    -----------------
    NOTE: these are inputted using a common block in IDL. Perhaps should use the same in python?

    H2_H2_EL	- if set, then include H2 -> H2 elastic self collisions
    H2_P_EL	- if set, then include H2 -> H(+) elastic collisions 
    H2_H_EL	- if set, then include H2 <-> H elastic collisions 
    H2_HP_CX	- if set, then include H2 -> H2(+) charge exchange collisions
    H_H_EL	- if set, then include H -> H elastic self collisions
    H_P_CX	- if set, then include H -> H(+) charge exchange collisions 
    H_P_EL	- if set, then include H -> H(+) elastic collisions 
    Simple_CX	- if set, then use CX source option (B): Neutrals are born
                in velocity with a distribution proportional to the local
                ion distribution function. Simple_CX=1 is default.
    
    mesh inputs:
    -----------------
    H2Gridfctr - float, factor to scale the molecular grid size. Default is 0.3.
        This is used to determine the number of velocity bins in the molecular grid.
        If GaugeH2 > 15.0, then this factor is scaled by 15.0 / GaugeH2.

    HGridfctr  - float, factor to scale the atomic grid size. Default is 0.3.

    keyword inputs
    ------------------
    compute_errors - if set, then compute and return errors in the molecular and atomic
        distribution functions. This is used for debugging purposes and is not
        typically needed in normal operation.
    plot - if set, then plot the results of the simulation.
    debug - if set, then print debugging information to the console.
    debrief - if set, then print a summary of the simulation results to the console.
    pause - if set, then pause the simulation after each iteration.
    Hplot - if set, then plot the results of the atomic simulation.
    Hdebug - if set, then print debugging information for the atomic simulation.
    Hdebrief - if set, then print a summary of the atomic simulation results to the console.
    Hpause - if set, then pause the atomic simulation after each iteration.
    H2plot - if set, then plot the results of the molecular simulation.
    H2debug - if set, then print debugging information for the molecular simulation.
    H2debrief - if set, then print a summary of the molecular simulation results to the console.
    H2pause - if set, then pause the molecular simulation after each iteration.


    KN1D_internal inputs:
    -----------------------
    NOTE: this input is a common block in IDL (like the collisions inputs). May be worth changing at some point
    fH_s, fH2_s, nH2_s, nHP_s, THP_s, SpH2_s - optional inputs for the initial seeds

    ------------------------------------------------------------------------------
    Output:
    -----------
    Four files
    'file'.KN1D_input.npz - a numpy save file containing the input parameters
    'file'.KN1D_mesh.npz - a numpy save file containing the mesh parameters
    'file'.KN1D_H2.npz - a numpy save file containing the molecular distribution function and moments
    'file'.KN1D_H.npz - a numpy save file containing the atomic distribution function and moments
    """
    prompt = 'KN1D => '

    # Internal state defaults
    interp_debug = False
    warn = None
    max_gen = 100

    if ReadInput:
        KN1D_input_path = f"{File}.KN1D_input.npz"
        if os.path.exists(KN1D_input_path):
            if debrief:
                print(f"{prompt} Reading input variables stored in {KN1D_input_path!r}")
            data = np.load(KN1D_input_path, allow_pickle=True)

            # now unpack each variable exactly as you saved it
            x         = data['x']
            xlimiter  = data['xlimiter']
            xsep      = data['xsep']
            GaugeH2   = data['GaugeH2']
            mu        = data['mu']
            Ti        = data['Ti']
            Te        = data['Te']
            n         = data['n']
            vxi       = data['vxi']
            LC        = data['LC']
            PipeDia   = data['PipeDia']
            truncate  = data['truncate']

            xH2       = data['xH2']
            TiM       = data['TiM']
            TeM       = data['TeM']
            nM        = data['nM']
            PipeDiaM  = data['PipeDiaM']
            vxM       = data['vxM']
            vrM       = data['vrM']
            TnormM    = data['TnormM']

            xH        = data['xH']
            TiA       = data['TiA']
            TeA       = data['TeA']
            nA        = data['nA']
            PipeDiaA  = data['PipeDiaA']
            vxA       = data['vxA']
            vrA       = data['vrA']
            TnormA    = data['TnormA']

        else:
            raise FileNotFoundError(f"{prompt} Input file {KN1D_input_path} does not exist. Please provide valid input parameters.")

    else:
        # Determine optimized vr, vx, x grid for Kinetic_H2 (molecules, M)
        nv    = 6
        Eneut = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
        fctr  = H2Gridfctr
        if GaugeH2 > 15.0:
            fctr *= 15.0 / GaugeH2
            
        kinetic_h2_mesh = Create_Kinetic_H2_Mesh(nv, mu, x, Ti, Te, n, PipeDia, fctr=fctr, E0_in=Eneut)

        xH2 = kinetic_h2_mesh['xH2']
        TiM = kinetic_h2_mesh['TiH2']
        TeM = kinetic_h2_mesh['TeH2']
        nM = kinetic_h2_mesh['neH2']
        PipeDiaM = kinetic_h2_mesh['PipeDiaH2']
        vxM = kinetic_h2_mesh['vx']
        vrM = kinetic_h2_mesh['vr']
        TnormM = kinetic_h2_mesh['Tnorm']

        # Determine optimized vr, vx, x grid for Kinetic_H (atoms, A)
        nv = 10
        fctr = 0.3
        if GaugeH2 > 30.0:
            fctr *= 30.0 / GaugeH2


        kinetic_h_mesh = Create_Kinetic_H_Mesh(nv, mu, x, Ti, Te, n, PipeDia, fctr=fctr)


        xH = kinetic_h_mesh['xH']
        TiA = kinetic_h_mesh['TiH']
        TeA = kinetic_h_mesh['TeH']
        nA = kinetic_h_mesh['neH']
        PipeDiaA = kinetic_h_mesh['PipeDiaH']
        vxA = kinetic_h_mesh['vx']
        vrA = kinetic_h_mesh['vr']
        TnormA = kinetic_h_mesh['Tnorm']


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

    # Physical constants
    mH       = 1.6726231e-27       # hydrogen mass (kg)
    q        = 1.602177e-19        # elementary charge (C)
    k_boltz  = 1.380658e-23        # Boltzmann's constant (J/K)
    Twall    = 293.0 * k_boltz / q # room temperature in eV
    v0_bar   = np.sqrt(8.0 * Twall * q / (np.pi * 2 * mu * mH))



    # 1) Identify positive/negative velocity indices and sizes
    ipM   = np.where(vxM > 0)[0]
    inM   = np.where(vxM < 0)[0]
    nvrM  = vrM.size
    nvxM  = vxM.size
    nxH2  = xH2.size

    ipA   = np.where(vxA > 0)[0]
    inA   = np.where(vxA < 0)[0]
    nvrA  = vrA.size
    nvxA  = vxA.size
    nxH   = xH.size

    # 2) Initialize fH and fH2 (these may be overwritten by data from an old run below)
    if refine:
        fH = np.zeros((nvrA, nvxA, nxH), dtype=float) if fH_s is None else fH_s
        fH2 = np.zeros((nvrM, nvxM, nxH2), dtype=float) if fH2_s is None else fH2_s
        nH2 = np.zeros(nxH2, dtype=float) if nH2_s is None else nH2_s
        nHP = np.zeros(nxH2, dtype=float) if nHP_s is None else nHP_s
        THP = np.zeros(nxH2, dtype=float) if THP_s is None else THP_s
    else:
        fH = np.zeros((nvrA, nvxA, nxH), dtype=float)
        fH2 = np.zeros((nvrM, nvxM, nxH2), dtype=float)
        nH2 = np.zeros(nxH2, dtype=float)
        nHP = np.zeros(nxH2, dtype=float)
        THP = np.zeros(nxH2, dtype=float)


    # 3) Convert input pressure (mtorr) to molecular density and flux
    fH2BC = np.zeros((nvrM, nvxM), dtype=float)
    DensM = 3.537e19 * GaugeH2      # molecular number density (m^-3)
    GammaxH2BC = 0.25 * DensM * v0_bar   # molecular flux at boundary

    Tmaxwell = np.array([Twall])
    vx_shift = np.array([0.0])
    mol = 2

    Maxwell = create_shifted_maxwellian(vrM, vxM, Tmaxwell, vx_shift, mu, mol, TnormM)
    fH2BC[:, ipM] = Maxwell[:, ipM, 0]  # Copy only the outgoing half (index 0 in Maxwell’s third dim)


    # 4) Compute NuLoss = Cs/LC
    Cs_LC = np.zeros_like(LC, dtype=float)
    ii = np.where(LC > 0)[0]
    if ii.size > 0:
        Cs_LC[ii] = np.sqrt(q * (Ti[ii] + Te[ii]) / (mu * mH)) / LC[ii]

    # IDL is the following:
    # NuLoss = interpol(Cs_LC, x, xH2)
    NuLoss = interp1d(x, Cs_LC, kind='linear', bounds_error=False, fill_value='extrapolate')(xH2)

    # 5) Compute first guess SpH2
    #
    # If plasma recycling accounts for molecular source, then SpH2 = 1/2 n Cs/LC (1/2 accounts for H2 versus H)
    # But, allow for SpH2 to be proportional to this function:
    #   SpH2 = beta n Cs/LC 
    # with beta being an adjustable parameter, set by achieving a net H flux of zero onto the wall.
    # For first guess of beta, set the total molecular source according to 
    # the formula
    #
    # (See notes "Procedure to adjust the normalization of the molecular source at the 
    # limiters (SpH2) to attain a net zero atom/molecule flux from wall")
    #
    # Integral{SpH2}dx =  (2/3) GammaxH2BC = beta Integral{n Cs/LC}dx
    #

    nCs_LC = n * Cs_LC
    SpH2_hat = interp1d(x, nCs_LC, kind='linear', bounds_error=False, fill_value='extrapolate')(xH2)

    # IDL:  integ_bl(/value, xH2, SpH2_hat) returns ∫ SpH2_hat dx
    # Python: integral = integ_bl(xH2, SpH2_hat)
    SpH2_hat /= np.trapz(SpH2_hat, xH2)  # Normalize SpH2_hat to ensure integral is 1
    beta = (2.0/3.0) * GammaxH2BC

    if refine:
        SpH2 = beta * SpH2_hat if SpH2_s is None else SpH2_s
    else:
        SpH2 = beta * SpH2_hat

    SH2 = SpH2.copy()  # Initialize SH2 with SpH2

    # 6) Interpolate for vxiM and vxiA
    vxiM = interp1d(x, vxi, kind='linear', bounds_error=False, fill_value='extrapolate')(xH2)
    vxiA = interp1d(x, vxi, kind='linear', bounds_error=False, fill_value='extrapolate')(xH)

    # 7) Initialize iteration history
    n_iter   = 0
    EH_hist = [0.0]
    SI_hist = [0.0]

    oldrun = False

    # Option: Read results from previous run
    if File:
        # Check for old data present
        mesh_file = f"{File}.KN1D_mesh.npz"
        H2_file   = f"{File}.KN1D_H2.npz"
        H_file    = f"{File}.KN1D_H.npz"

        mp  = os.path.exists(mesh_file)
        H2p = os.path.exists(H2_file)
        Hp  = os.path.exists(H_file)
        old_data_present = mp and H2p and Hp

        if old_data_present:
            if debrief:
                print(f"{prompt} Reading mesh variables stored in {mesh_file}")
            mesh_data = np.load(mesh_file, allow_pickle=True)

            # Compare saved mesh inputs against current inputs
            print('breakpoint inside the file thing')
            test = 0
            if not np.array_equal(mesh_data['x_s'], x): test += 1
            if mesh_data['GaugeH2_s'].item() != GaugeH2: test += 1
            if mesh_data['mu_s'].item() != mu: test += 1
            if not np.array_equal(mesh_data['Ti_s'], Ti): test += 1
            if not np.array_equal(mesh_data['Te_s'], Te): test += 1
            if not np.array_equal(mesh_data['n_s'], n): test += 1
            if not np.array_equal(mesh_data['vxi_s'], vxi): test += 1
            if not np.array_equal(mesh_data['PipeDia_s'], PipeDia): test += 1
            if not np.array_equal(mesh_data['LC_s'], LC): test += 1
            if not np.array_equal(mesh_data['xH2_s'], xH2): test += 1
            if not np.array_equal(mesh_data['vxM_s'], vxM): test += 1
            if not np.array_equal(mesh_data['vrM_s'], vrM): test += 1
            if not np.array_equal(mesh_data['TnormM_s'], TnormM): test += 1
            if not np.array_equal(mesh_data['xH_s'], xH): test += 1
            if not np.array_equal(mesh_data['vxA_s'], vxA): test += 1
            if not np.array_equal(mesh_data['vrA_s'], vrA): test += 1
            if not np.array_equal(mesh_data['TnormA_s'], TnormA): test += 1

            if test == 0:
                oldrun = True # if all the checks are met, then restore the old data

            if oldrun:
                if debrief:
                    print(f"{prompt} Reading output variables stored in {H2_file}")

                h2_data = np.load(H2_file, allow_pickle=True)
                xH2             = h2_data['xH2']
                fH2             = h2_data['fH2']
                nH2             = h2_data['nH2']
                GammaxH2        = h2_data['GammaxH2']
                VxH2            = h2_data['VxH2']
                pH2             = h2_data['pH2']
                TH2             = h2_data['TH2']
                qxH2            = h2_data['qxH2']
                qxH2_total      = h2_data['qxH2_total']
                Sloss           = h2_data['Sloss']
                QH2             = h2_data['QH2']
                RxH2            = h2_data['RxH2']
                QH2_total       = h2_data['QH2_total']
                AlbedoH2        = h2_data['AlbedoH2']
                nHP             = h2_data['nHP']
                THP             = h2_data['THP']
                fSH             = h2_data['fSH']
                SH              = h2_data['SH']
                SP              = h2_data['SP']
                SHP             = h2_data['SHP']
                NuE             = h2_data['NuE']
                NuDis           = h2_data['NuDis']
                NuLoss          = h2_data['NuLoss']
                SpH2            = h2_data['SpH2']
                piH2_xx         = h2_data['piH2_xx']
                piH2_yy         = h2_data['piH2_yy']
                piH2_zz         = h2_data['piH2_zz']
                RxH2CX          = h2_data['RxH2CX']
                RxH_H2          = h2_data['RxH_H2']
                RxP_H2          = h2_data['RxP_H2']
                RxW_H2          = h2_data['RxW_H2']
                EH2CX           = h2_data['EH2CX']
                EH_H2           = h2_data['EH_H2']
                EP_H2           = h2_data['EP_H2']
                EW_H2           = h2_data['EW_H2']
                Epara_PerpH2_H2 = h2_data['Epara_PerpH2_H2']
                Gam             = h2_data['Gam']
                gammaxH2_plus   = h2_data['gammaxH2_plus']
                gammaxH2_minus  = h2_data['gammaxH2_minus']
    
                if debrief:
                    print(f"{prompt} Reading output variables stored in {H_file}")

                h_data = np.load(H_file, allow_pickle=True)
                xH             = h_data['xH']
                fH             = h_data['fH']
                nH             = h_data['nH']
                GammaxH        = h_data['GammaxH']
                VxH            = h_data['VxH']
                pH             = h_data['pH']
                TH             = h_data['TH']
                qxH            = h_data['qxH']
                qxH_total      = h_data['qxH_total']
                NetHSource     = h_data['NetHSource']
                Sion           = h_data['Sion']
                SideWallH      = h_data['SideWallH']
                QH             = h_data['QH']
                RxH            = h_data['RxH']
                QH_total       = h_data['QH_total']
                AlbedoH        = h_data['AlbedoH']
                GammaHLim      = h_data['GammaHLim']
                nDelta_nH2     = h_data['nDelta_nH2']
                piH_xx         = h_data['piH_xx']
                piH_yy         = h_data['piH_yy']
                piH_zz         = h_data['piH_zz']
                RxHCX          = h_data['RxHCX']
                RxH2_H         = h_data['RxH2_H']
                RxP_H          = h_data['RxP_H']
                RxW_H          = h_data['RxW_H']
                EHCX           = h_data['EHCX']
                EH2_H          = h_data['EH2_H']
                EP_H           = h_data['EP_H']
                EW_H           = h_data['EW_H']
                Epara_PerpH_H  = h_data['Epara_PerpH_H']
                SourceH        = h_data['SourceH']
                SRecomb        = h_data['SRecomb']
                EH_hist        = h_data['EH_hist']
                SI_hist        = h_data['SI_hist']
                gammaxH_plus   = h_data['gammaxH_plus']
                gammaxH_minus  = h_data['gammaxH_minus']
                Lyman          = h_data['Lyman']
                Balmer         = h_data['Balmer']

                fH_s   = fH
                fH2_s  = fH2
                nH2_s  = nH2
                SpH2_s = SpH2
                nHP_s  = nHP
                THP_s  = THP

            else:
                if not NewFile and debrief:
                    print(f"{prompt} Mesh variables from previous run are different! Computing new output...")

        else:
            if not NewFile:
                raise FileNotFoundError(f"{prompt} Cannot read old data: {mesh_file}, {H2_file}, {H_file}")

    else:
        if NewFile:
            raise FileNotFoundError(f"Error: No file name specified")
    
    # Test for v0_bar consistency in the numerics by computing it from a half maxwellian at the wall temperature
    vthM = np.sqrt(2 * q * TnormM / (mu * mH))
    Vr2pidVrM, VrVr4pidVrM, dVxM = make_dvr_dvx(vrM, vxM)[0:3]
    vthA = np.sqrt(2 * q * TnormA / (mu * mH))
    Vr2pidVrA, VrVr4pidVrA, dVxA = make_dvr_dvx(vrA, vxA)[0:3]

    # New is below
    nbarHMax = np.dot(Vr2pidVrM, np.dot(fH2BC, dVxM))
    temp = np.dot(fH2BC, vxM*dVxM)
    numerator = np.dot(Vr2pidVrM, temp)
    vbarM = 2 * vthM * numerator / nbarHMax
    vbarM_error = abs(vbarM - v0_bar) / max(vbarM, v0_bar)

    nvrm, nvxm = vrM.size, vxM.size
    vr2vx2_ran2 = np.empty((nvrm, nvxm))

    # Full Maxwell for TMax
    Max0     = Maxwell[:, :, 0]

    nbarMax  = np.dot(Vr2pidVrM, np.dot(Max0, dVxM))
    temp1 = Max0 @ (vxM * dVxM)  # This is equivalent to np.sum(Max0 * vxM * dVxM, axis=1)
    numerator = np.dot(Vr2pidVrM, temp1)
    UxMax = vthM * numerator / nbarMax

    for i in range(nvrm):
        vr2vx2_ran2[i, :] = vrM[i]**2 + (vxM - UxMax/vthM)**2
    
    TMax = (2*mu*mH) * vthM**2 * np.dot(
            Vr2pidVrM,
            (vr2vx2_ran2 * Max0) @ dVxM
       ) / (3*q*nbarMax)
    

    # Half‐Maxwell temperature THMax
    UxHMax = vthM * (Vr2pidVrM @ (fH2BC @ (vxM * dVxM))) / nbarHMax


    for i in range(nvrm):
        vr2vx2_ran2[i, :] = vrM[i]**2 + (vxM - UxHMax/vthM)**2

    THMax = (2*mu*mH) * vthM**2 * (
                Vr2pidVrM @ ((vr2vx2_ran2 * fH2BC) @ dVxM)
            ) / (3*q*nbarHMax)

    if compute_errors and debrief:
        print(f"{prompt} VbarM_error:       {vbarM_error}")
        print(f"{prompt} TWall Maxwellian:   {TMax} eV")
        print(f"{prompt} TWall Half Maxwellian: {THMax} eV")


    # ————————————————————————————————————————————————————————————————
    # 9) Option to view input profiles
    # ————————————————————————————————————————————————————————————————
    if plot > 0:
        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 12))
        fig.suptitle(f"Input profiles (Gauge H₂ pressure = {GaugeH2:.3f} mtorr)", y=0.96)

        # 1) density
        axes[0].plot(x, n,     color='C0')
        axes[0].set_ylabel(r'$n$ (m$^{-3}$)')
        axes[0].set_yscale('log')

        # 2) electron temperature
        axes[1].plot(x, Te,    color='C1')
        axes[1].set_ylabel(r'$T_e$ (eV)')
        axes[1].set_yscale('log')

        # 3) ion temperature
        axes[2].plot(x, Ti,    color='C2')
        axes[2].set_ylabel(r'$T_i$ (eV)')
        axes[2].set_yscale('log')

        # 4) connection length
        axes[3].plot(x, LC,    color='C3')
        axes[3].set_ylabel('Connection Length (m)')

        # 5) pipe diameter
        axes[4].plot(x, 1.001*PipeDia, color='C4')
        axes[4].set_ylabel('Pipe Diameter (m)')
        axes[4].set_xlabel('x (m)')

        # draw limiter & separatrix
        for ax in axes:
            ax.axvline(x=xlimiter, linestyle='--', color='k')
            ax.axvline(x=xsep,     linestyle='--', color='k')

        # annotate regions on the top panel
        ylim = axes[0].get_ylim()
        y_mid = 10**(np.log10(ylim[0]) * 0.3 + np.log10(ylim[1]) * 0.7)
        axes[0].text(0.5*(x.min()+xlimiter), y_mid, 'LIMITER',
                    ha='center', va='center', rotation=90, fontsize=10)
        axes[0].text(0.5*(xsep+xlimiter),   y_mid, 'SOL',
                    ha='center', va='center', rotation=90, fontsize=10)
        axes[0].text(0.5*(xsep+x.max()),     y_mid, 'CORE',
                    ha='center', va='center', rotation=90, fontsize=10)

        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()


    h_seeds_common = None
    h_output_common = None
    h_errors_common = None
    h_H2_moments_common = None
    h_internal_common = None

    h2_seeds_common = None
    h2_output_common = None
    h2_errors_common = None
    h2_H_moments_common = None
    h2_internal_common = None

    while True: # this replaces the fH_fH2 iterate loop
        if not oldrun: # if oldrun is True, skip straight to the checking stage
        # Initialize iteration tracking
            nH2_prev    = nH2.copy()
            nDelta_nH2  = np.inf

            n_iter += 1
            if debrief:
                print(f"{prompt} fH/fH2 Iteration: {n_iter}")

            # NOTE: I haven't yet incorporated the warn kwarg into Interp_fVrVxX
            warn = 5e-3
            fHM = Interp_fVrVxX(fH, vrA, vxA, xH, TnormA,vrM, vxM, xH2, TnormM,debug=interp_debug, correct=True)

            # 10.2) Compute fH2 on the H2 mesh via your Python Kinetic_H2
            ni_correct = True
            compute_h_source = True
            H2compute_errors = compute_errors and H2debrief

            # call the Python version of Kinetic_H2
            h2_results, h2_seeds, h2_output, h2_errors, h2_H_moments, h2_internal = kinetic_h2(
                vx=vxM,
                vr=vrM,
                x=xH2,
                Tnorm=TnormM,
                mu=mu,
                Ti=TiM,
                Te=TeM,
                n=nM,
                vxi=vxiM,
                fH2BC=fH2BC,
                GammaxH2BC=GammaxH2BC,
                NuLoss=NuLoss,
                PipeDia=PipeDiaM,
                fH=fHM,
                SH2=SH2,

                # the next three are outputs but 'seed' values can be provided at input
                fH2=fH2,
                nHP=nHP,
                THP=THP,
                
                # collision keywords
                Simple_CX=Simple_CX,
                H2_H2_EL=H2_H2_EL,
                H2_P_EL=H2_P_EL,
                H2_H_EL=H2_H_EL,
                H2_HP_CX=H2_HP_CX,
                ni_correct=ni_correct,
                # No_Sawada = No_Sawada # NOTE this one doesn't seem to be used at all in the IDL code

                # control & debug flags
                truncate=truncate,
                Max_Gen=max_gen,
                Compute_H_Source=compute_h_source,
                
                compute_errors=H2compute_errors,
                plot=H2plot,
                debug=H2debug,
                debrief=H2debrief,
                pause=H2pause,

                # the next ones are the equivalent of the IDL common blocks (i.e values carried over from previous runs)
                h2_seeds=h2_seeds_common,
                h2_H_moments=h2_H_moments_common,
                h2_internal=h2_internal_common
            )

            # unpack the main outputs
            fH2           = h2_results['fH2']
            nH2           = h2_results['nH2']
            GammaxH2      = h2_results['GammaxH2']
            VxH2          = h2_results['VxH2']
            pH2           = h2_results['pH2']
            TH2           = h2_results['TH2']
            qxH2          = h2_results['qxH2']
            qxH2_total    = h2_results['qxH2_total']
            Sloss         = h2_results['Sloss']
            QH2           = h2_results['QH2']
            RxH2          = h2_results['RxH2']
            QH2_total     = h2_results['QH2_total']
            AlbedoH2      = h2_results['AlbedoH2']
            WallH2        = h2_results['WallH2']
            nHP           = h2_results['nHP']
            THP           = h2_results['THP']
            fSH           = h2_results['fSH']
            SH            = h2_results['SH']
            SP            = h2_results['SP']
            SHP           = h2_results['SHP']
            NuE           = h2_results['NuE']
            NuDis         = h2_results['NuDis']
            ESH           = h2_results['ESH']
            Eaxis         = h2_results['Eaxis']

            # fH2BC is modified in kinetic_h2 to match the flux, so need to make sure I use the updated version in all future iterations
            fH2BC = h2_seeds['fH2BC_s']

            # set up all the common block variables to be read in during the next iteration
            h2_seeds_common = h2_seeds
            h2_output_common = h2_output
            h2_errors_common = h2_errors
            h2_H_moments_common = h2_H_moments
            h2_internal_common = h2_internal

            # Interpolate H2 data onto H mesh: fH2 -> fH2A, fSH -> fSHA, nHP -> nHPA, THP -> THPA
            warn = 5e-3
            fH2A = Interp_fVrVxX(fa=fH2, vra=vrM, vxa=vxM, xa=xH2, Tnorma=TnormM,vrb=vrA, vxb=vxA, xb=xH,   Tnormb=TnormA,debug=interp_debug, correct=True)
            fSHA = Interp_fVrVxX(fa=fSH, vra=vrM, vxa=vxM, xa=xH2, Tnorma=TnormM,vrb=vrA, vxb=vxA, xb=xH,   Tnormb=TnormA,debug=interp_debug, correct=True)

            # 3) scalar profiles nHP, THP → nHPA, THPA on the H‐grid
            nHPA  = interp_scalarx(nHP,  xH2, xH, warn=warn, debug=interp_debug)
            THPA  = interp_scalarx(THP,  xH2, xH, warn=warn, debug=interp_debug)

            # Compute fH using Kinetic_H
            GammaxHBC = 0.0
            fHBC = np.zeros((nvrA, nvxA, nxH))
            H_H2_EL = H2_H_EL
            ni_correct = True
            Hcompute_errors = compute_errors and Hdebrief

            breakpoint()


            # call Python version of Kinetic_H
            h_results, h_seeds, h_output, h_errors, h_H2_moments, h_internal = kinetic_h(
                # velocity & spatial grids
                vx=vxA,
                vr=vrA,
                x=xH,
                Tnorm=TnormA,
                mu=mu,
                Ti=TiA,
                Te=TeA,
                n=nA,
                vxi=vxiA,
                # boundary-condition & molecular inputs
                fHBC=fHBC,
                GammaxHBC=GammaxHBC,
                PipeDia=PipeDiaA,
                fH2=fH2A,
                fSH=fSHA,
                nHP=nHPA,
                THP=THPA,

                # “seed” in/out for fH
                fH=fH,

                # algorithmic control flags
                truncate=truncate,
                Simple_CX=Simple_CX,
                Max_Gen=max_gen,
                #No_Johnson_Hinnov=No_Johnson_Hinnov, # NOTE: just use the default here
                #No_Recomb=No_Recomb,
                H_H_EL=H_H_EL,
                H_P_EL=H_P_EL,
                H_H2_EL=H_H2_EL,
                H_P_CX=H_P_CX,
                ni_correct=ni_correct,

                # error-handling & diagnostics
                compute_errors=Hcompute_errors,
                plot=Hplot,
                debug=Hdebug,
                debrief=Hdebrief,
                pause=Hpause,

                h_seeds=h_seeds_common,
                h_H2_moments=h_H2_moments_common,
                h_internal=h_internal_common
            )

            breakpoint()

            # unpack the “real” outputs
            fH          = h_results['fH']
            nH          = h_results['nH']
            GammaxH     = h_results['GammaxH']
            VxH         = h_results['VxH']
            pH          = h_results['pH']
            TH          = h_results['TH']
            qxH         = h_results['qxH']
            qxH_total   = h_results['qxH_total']
            NetHSource  = h_results['NetHSource']
            Sion        = h_results['Sion']
            QH          = h_results['QH']
            RxH         = h_results['RxH']
            QH_total    = h_results['QH_total']
            AlbedoH     = h_results['AlbedoH']
            SideWallH   = h_results['WallH']


            # fHBC is modified in kinetic_h to match the flux, so need to make sure I use the updated version in all future iterations
            fHBC = h_seeds['fHBC_s']

            # set up all the common block variables to be read in during the next iteration
            h_seeds_common = h_seeds
            h_output_common = h_output
            h_errors_common = h_errors
            h_H2_moments_common = h_H2_moments
            h_internal_common = h_internal


            # — Interpolate SideWallH onto the molecular (H2) mesh —
            SideWallHM = interp_scalarx(
                SideWallH,    # fa
                xH,           # xa
                xH2,          # xb
                warn=warn,
                debug=interp_debug
            )

            # Adjust SpH2 to achieve net zero hydrogen atom/molecule flux from wall
            # (See notes "Procedure to adjust the normalization of the molecular source at the 
            #   limiters (SpH2) to attain a net zero atom/molecule flux from wall")
            # 
            # Compute SI, GammaH2Wall_minus, and GammaHWall_minus

            # — Compute the two “area” integrals —
            SI      = np.trapz(SpH2,    xH2)                     # ∫ SpH2 dx  
            SwallI  = np.trapz(0.5*SideWallHM, xH2)               # ∫ 0.5·SideWallHM dx

            # — Fluxes leaving the wall —
            GammaH2Wall_minus = AlbedoH2 * GammaxH2BC             # AlbedoH2·incoming H2 flux  
            GammaHWall_minus  = -GammaxH[0]                       # minus the H‐flux at xH[0]

            # Compute Epsilon and alphaplus1RH0Dis
            Epsilon               = 2.0 * GammaH2Wall_minus / (SI + SwallI)
            alpha_plus_1RH0Dis    = GammaHWall_minus / (
                (1.0 - 0.5*Epsilon)*(SI + SwallI) + GammaxH2BC
            )

            # — Compute the flux “error” EH and its derivative dEHdSI —
            EH     = 2.0 * GammaxH2[0] - GammaHWall_minus
            dEHdSI = -Epsilon - alpha_plus_1RH0Dis * (1.0 - 0.5*Epsilon)

            # Option: print normalized flux error
            if debrief and compute_errors:
                nEH = abs(EH) / max(abs(2.0*GammaxH2[0]), abs(GammaHWall_minus))
                print(f"{prompt} Normalized Hydrogen Flux Error: {nEH:.3e}")

            # — Adjust the total source integral SI to drive EH→0 —
            Delta_SI = -EH / dEHdSI
            SI      = SI + Delta_SI

            # — Rescale the molecular source to the new integral —
            SpH2     = SI * SpH2_hat
            EH_hist.append(EH)
            SI_hist.append(SI)

            # Set total H2 source
            SH2=SpH2+0.5*SideWallHM

            # 10.7) Optional momentum‐transfer error check
            if compute_errors:
                _RxH_H2 = interp_scalarx(h2_output['RxH_H2'], xH2, xH, warn=warn, debug=interp_debug)
                DRx     = _RxH_H2 + h_output['RxH2_H']
                nDRx    = np.max(np.abs(DRx)) / np.max(np.abs([_RxH_H2, h_output['RxH2_H']]))
                if debrief:
                    print(f"{prompt} Normalized H₂↔H Momentum Transfer Error: {nDRx:.3e}")

            # 10.8) Convergence metric
            Delta_nH2     = np.abs(nH2 - nH2_prev)
            nDelta_nH2    = np.max(Delta_nH2 / np.max(nH2))

        # Now test for convergence
        if debrief:
            print(f"{prompt} Maximum Normalized Change in nH2: {nDelta_nH2:.3e}")
        if nDelta_nH2 < truncate:
            print(f"{prompt} Converged after {n_iter} iterations.")
            break  # exit the fH/fH2 iteration loop
        else:
            print(f"{prompt} Not converged, continuing iteration...")
            oldrun = False  # reset oldrun to False for the next iteration
            continue  # loop back to the start of the fH/fH2 iteration
    

    # Compute total H flux through crossing limiter radius
    _GammaxH2 = interp_scalarx(GammaxH2, xH2, xH,
                            warn=warn, debug=interp_debug)
    Gam = 2.0 * _GammaxH2 + GammaxH
    interp_function = interp1d(
        xH, Gam,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    GammaHLim = interp_function(xlimiter)

    ipA = np.where(vxA > 0)[0]
    inA = np.where(vxA < 0)[0]
    gammaxH_plus  = np.zeros(nxH)
    gammaxH_minus = np.zeros(nxH)
    for k in range(nxH):
        fH_k = fH[:, :, k]
        gammaxH_plus[k]  = vthA * np.sum(
            Vr2pidVrA[:,None] * fH_k[:, ipA] * (vxA[ipA] * dVxA[ipA])[None,:]
        )
        gammaxH_minus[k] = vthA * np.sum(
            Vr2pidVrA[:,None] * fH_k[:, inA] * (vxA[inA] * dVxA[inA])[None,:]
        )

    ipM = np.where(vxM > 0)[0]
    inM = np.where(vxM < 0)[0]
    gammaxH2_plus  = np.zeros(nxH2)
    gammaxH2_minus = np.zeros(nxH2)
    for k in range(nxH2):
        fH2_k = fH2[:, :, k]
        gammaxH2_plus[k]  = vthM * np.sum(
            Vr2pidVrM[:,None] * fH2_k[:, ipM] * (vxM[ipM] * dVxM[ipM])[None,:]
        )
        gammaxH2_minus[k] = vthM * np.sum(
            Vr2pidVrM[:,None] * fH2_k[:, inM] * (vxM[inM] * dVxM[inM])[None,:]
        )

    # 4) Lyman‐ and Balmer‐α emissivities
    Lyman  = Lyman_Alpha(nA, TeA, nH, no_null=True)
    Balmer = Balmer_Alpha(nA, TeA, nH, no_null=True)

    # 5) Refresh all the “_s” seed variables for a potential restart
    fH_s    = fH.copy()
    fH2_s   = fH2.copy()
    nH2_s   = nH2.copy()
    SpH2_s  = SpH2.copy()
    nHP_s   = nHP.copy()
    THP_s   = THP.copy()

    x_s         = x.copy()
    GaugeH2_s   = GaugeH2
    mu_s        = mu
    Ti_s        = Ti.copy()
    Te_s        = Te.copy()
    n_s         = n.copy()
    vxi_s       = vxi.copy()
    PipeDia_s   = PipeDia.copy()
    LC_s        = LC.copy()
    xH2_s       = xH2.copy()
    vxM_s       = vxM.copy()
    vrM_s       = vrM.copy()
    TnormM_s    = TnormM
    xH_s        = xH.copy()
    vxA_s       = vxA.copy()
    vrA_s       = vrA.copy()
    TnormA_s    = TnormA

    EH_hist = np.array(EH_hist)
    SI_hist = np.array(SI_hist)

    if File is not None and n_iter > 0:
        # -- inputs --
        np.savez(f"{File}.KN1D_input.npz",
            x=x, xlimiter=xlimiter, xsep=xsep, GaugeH2=GaugeH2,
            mu=mu, Ti=Ti, Te=Te, n=n, vxi=vxi, LC=LC,
            PipeDia=PipeDia, truncate=truncate,
            xH2=xH2, TiM=TiM, TeM=TeM, nM=nM, PipeDiaM=PipeDiaM,
            vxM=vxM, vrM=vrM, TnormM=TnormM,
            xH=xH, TiA=TiA, TeA=TeA, nA=nA,
            PipeDiaA=PipeDiaA, vxA=vxA, vrA=vrA, TnormA=TnormA
        )

        # -- mesh state --
        np.savez(f"{File}.KN1D_mesh.npz",
            x_s=x_s, GaugeH2_s=GaugeH2_s, mu_s=mu_s,
            Ti_s=Ti_s, Te_s=Te_s, n_s=n_s, vxi_s=vxi_s,
            LC_s=LC_s, PipeDia_s=PipeDia_s,
            xH2_s=xH2_s, vxM_s=vxM_s, vrM_s=vrM_s,
            TnormM_s=TnormM_s, xH_s=xH_s, vxA_s=vxA_s,
            vrA_s=vrA_s, TnormA_s=TnormA_s
        )

        # -- molecular output --
        np.savez(f"{File}.KN1D_H2.npz",
            xH2=xH2,
            fH2=fH2, nH2=nH2, GammaxH2=GammaxH2,
            VxH2=VxH2, pH2=pH2, TH2=TH2,
            qxH2=qxH2, qxH2_total=qxH2_total,
            Sloss=Sloss, QH2=QH2, RxH2=RxH2,
            QH2_total=QH2_total, AlbedoH2=AlbedoH2,
            nHP=nHP, THP=THP, fSH=fSH, SH=SH,
            SP=SP, SHP=SHP,
            NuE=NuE, NuDis=NuDis, NuLoss=NuLoss, # it shouldn't have been changed since it was first defined
            SpH2=SpH2,
            # now for the common block outputs
            piH2_xx = h2_output['piH2_xx'], piH2_yy = h2_output['piH2_yy'], piH2_zz = h2_output['piH2_zz'],
            RxH2CX=h2_output['RxH2CX'], RxH_H2=h2_output['RxH_H2'], RxP_H2=h2_output['RxP_H2'],
            RxW_H2=h2_output['RxW_H2'], EH2CX=h2_output['EH2CX'], EH_H2=h2_output['EH_H2'],
            EP_H2=h2_output['EP_H2'], EW_H2=h2_output['EW_H2'],
            Epara_PerpH2_H2=h2_output['Epara_PerpH2_H2'],
            # post-calculations
            Gam=Gam,
            gammaxH2_plus=gammaxH2_plus,
            gammaxH2_minus=gammaxH2_minus
        )

        # -- atomic output --
        np.savez(f"{File}.KN1D_H.npz",
            xH=xH, fH=fH, nH=nH,
            GammaxH=GammaxH, VxH=VxH, pH=pH, TH=TH,
            qxH=qxH, qxH_total=qxH_total,
            NetHSource=NetHSource, Sion=Sion,
            SideWallH=SideWallH, QH=QH,
            RxH=RxH, QH_total=QH_total,
            AlbedoH=AlbedoH, GammaHLim=GammaHLim,
            nDelta_nH2=nDelta_nH2,
            # now for the common block outputs
            piH_xx=h_output['piH_xx'], piH_yy=h_output['piH_yy'], piH_zz=h_output['piH_zz'],
            RxHCX=h_output['RxHCX'], RxH2_H=h_output['RxH2_H'], RxP_H=h_output['RxP_H'],
            RxW_H=h_output['RxW_H'], EHCX=h_output['EHCX'],
            EH2_H=h_output['EH2_H'], EP_H=h_output['EP_H'], EW_H=h_output['EW_H'],
            Epara_PerpH_H=h_output['Epara_PerpH_H'],
            SourceH=h_output['SourceH'], SRecomb=h_output['SRecomb'],
            #post-calculations
            EH_hist=EH_hist, SI_hist=SI_hist,
            gammaxH_plus=gammaxH_plus,
            gammaxH_minus=gammaxH_minus,
            Lyman=Lyman, Balmer=Balmer
        )

        breakpoint()


    if plot > 0:
        # precompute all the “mid‐point” indices in one shot
        # kn1d_include will also draw the “Gauge pressure” and “FILE:” text
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        
        # ————————————————————————————————————————————————————————————————
        # 1) Density profiles
        # ————————————————————————————————————————————————————————————————
        ydata = np.hstack([n, nH2, nHP, nH])
        mask  = ydata > 0
        yrng  = (ydata[mask].min(), ydata[mask].max())
        
        ax.set_yscale('log')
        ax.set_ylim(yrng)
        ax.set_title("KN1D: Density Profiles")
        ax.set_xlabel("x (meters)")
        ax.set_ylabel(r"$n\ (\mathrm{m}^{-3})$")
        
        # plot in the same order as your IDL
        ax.plot(x,   n,   color='C0')
        ax.text(x[mid[4]], 1.1*n[mid[4]], _e,   color='C0')
        
        ax.plot(xH2, nH2, color='C1')
        ax.text(xH2[midH2[1]], 1.1*nH2[midH2[1]], _HH, color='C1')
        
        ax.plot(xH,  nH,  color='C2')
        ax.text(xH[midH[2]], 1.1*nH[midH[2]], _H,   color='C2')
        
        ax.plot(xH2, nHP, color='C3')
        ax.text(xH2[midH2[3]], 1.1*nHP[midH2[3]], _Hp, color='C3')
        
        kn1d_limiter_include(ax, xlimiter, xsep)
        plt.tight_layout()
        plt.show()
        
        
        # ————————————————————————————————————————————————————————————————
        # 2) Temperature profiles
        # ————————————————————————————————————————————————————————————————
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        
        ax.set_yscale('log')
        ax.set_ylim(0.02, 200.)
        ax.set_title("KN1D: Temperature Profiles")
        ax.set_xlabel("x (meters)")
        ax.set_ylabel("Temperature (eV)")
        
        ax.plot(   x,  Ti, color='C0'); ax.text(x[mid[5]], 1.1*Ti[mid[5]],   _p,  color='C0')
        ax.plot(   x,  Te, color='C1'); ax.text(x[mid[4]], 1.1*Te[mid[4]],   _e,  color='C1')
        ax.plot(  xH,  TH, color='C2'); ax.text(xH[midH[2]], 1.1*TH[midH[2]], _H,  color='C2')
        ax.plot( xH2, THP, color='C3'); ax.text(xH2[midH2[3]], 1.1*THP[midH2[3]], _Hp, color='C3')
        ax.plot( xH2, TH2, color='C4'); ax.text(xH2[midH2[1]], 1.1*TH2[midH2[1]], _HH, color='C4')
        
        kn1d_limiter_include(ax, xlimiter, xsep)
        plt.tight_layout()
        plt.show()
        
        
        # ————————————————————————————————————————————————————————————————
        # 3) Particle fluxes
        # ————————————————————————————————————————————————————————————————
        GammaxP = n * vxi
        ydata   = np.hstack([2*GammaxH2, GammaxH, GammaxP])
        fscale  = 1.0/1e21
        yrng    = (ydata.min()*fscale, 1.2*ydata.max()*fscale)
        
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        
        ax.set_ylim(yrng)
        ax.set_title("KN1D: Particle Fluxes")
        ax.set_xlabel("x (meters)")
        ax.set_ylabel(r"$10^{21}\,\mathrm{m^{-2}s^{-1}}$")
        
        ax.plot(   xH2, 2*GammaxH2*fscale, color='C1')
        ax.text(xH2[midH2[0]], 1.1*2*GammaxH2[midH2[0]]*fscale, f"2×{_HH}", color='C1')
        
        ax.plot(   x,   GammaxP*fscale,   color='C0')
        ax.text(x[mid[1]], 1.1*GammaxP[mid[1]]*fscale, _e, color='C0')
        
        ax.plot(  xH,  GammaxH*fscale,   color='C2')
        ax.text(xH[midH[2]], 1.1*GammaxH[midH[2]]*fscale, _H, color='C2')
        
        ax.plot(  xH,  Gam*fscale,  color='C3')
        ax.text(xH[midH[3]], 1.1*Gam[midH[3]]*fscale, f"2×{_HH}+{_H}", color='C3')
        
        kn1d_limiter_include(ax, xlimiter, xsep)
        ax.text(0.6, 0.9,
                f"Total {_H} Flux at limiter edge: {GammaHLim:.2e}",
                transform=ax.transAxes, va='center')
        
        plt.tight_layout()
        plt.show()
        
        
        # ————————————————————————————————————————————————————————————————
        # 4) Positive/Negative flux components
        # ————————————————————————————————————————————————————————————————
        if 'gammaxH_plus' not in locals():
            ip = np.where(vxA>0)[0]
            iv = np.where(vxA<0)[0]
            gammaxH_plus  = np.array([vthA * np.sum(Vr2pidVrA * (fH[:,ip,k]*(vxA[ip]*dVxA[ip])))  for k in range(nxH)])
            gammaxH_minus = np.array([vthA * np.sum(Vr2pidVrA * (fH[:,iv,k]*(vxA[iv]*dVxA[iv]))) for k in range(nxH)])
            ip = np.where(vxM>0)[0]
            iv = np.where(vxM<0)[0]
            gammaxH2_plus  = np.array([vthM * np.sum(Vr2pidVrM * (fH2[:,ip,k]*(vxM[ip]*dVxM[ip])))  for k in range(nxH2)])
            gammaxH2_minus = np.array([vthM * np.sum(Vr2pidVrM * (fH2[:,iv,k]*(vxM[iv]*dVxM[iv]))) for k in range(nxH2)])
        
        ydata = np.hstack([
            gammaxH_plus,
            gammaxH_minus,
            GammaxH,
            2*gammaxH2_plus,
            2*gammaxH2_minus,
            2*GammaxH2
        ])
        yrng = (ydata.min()*fscale, 1.2*ydata.max()*fscale)
        
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        
        ax.set_ylim(yrng)
        ax.set_title("KN1D: Particle Flux Components")
        ax.set_xlabel("x (meters)")
        ax.set_ylabel(r"$10^{21}\,\mathrm{m^{-2}s^{-1}}$")
        
        ax.plot(   xH2, 2*gammaxH2_plus*fscale,   color='C1'); ax.text(xH2[midH2[0]], 2*gammaxH2_plus[midH2[0]]*fscale, f"2×{_HH}(+)", color='C1')
        ax.plot(   xH2, 2*gammaxH2_minus*fscale,  color='C1'); ax.text(xH2[midH2[0]], 2*gammaxH2_minus[midH2[0]]*fscale, f"2×{_HH}(-)", color='C1')
        ax.plot(   xH2, 2*GammaxH2*fscale,        color='C2'); ax.text(xH2[midH2[0]], 2*GammaxH2[midH2[0]]*fscale,  f"2×{_HH}", color='C2')
        ax.plot(    xH,  gammaxH_plus*fscale,     color='C3'); ax.text(xH[midH[2]],  gammaxH_plus[midH[2]]*fscale,     f"{_H}(+)", color='C3')
        ax.plot(    xH,  gammaxH_minus*fscale,    color='C3'); ax.text(xH[midH[2]],  gammaxH_minus[midH[2]]*fscale,    f"{_H}(-)", color='C3')
        ax.plot(    xH,  GammaxH*fscale,          color='C4'); ax.text(xH[midH[2]],  GammaxH[midH[2]]*fscale,          _H, color='C4')
        
        kn1d_limiter_include(ax, xlimiter, xsep)
        ax.text(0.6, 0.9,
                f"Total {_H} Flux at limiter edge: {GammaHLim:.2e}",
                transform=ax.transAxes, va='center')
        
        plt.tight_layout()
        plt.show()



        # Sources/Sinks
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        ydata = np.hstack([SH, SP, SHP, SpH2, SideWallH,
                        NuLoss*nHP, NuDis*nHP, Sion, h_output['SRecomb']])
        ax.set_yscale('log')
        ax.set_ylim((ydata[ydata>0].min(), ydata[ydata>0].max()))
        ax.set_title("KN1D: Source(+) and Sink(−) Profiles")
        ax.set_xlabel("x (m)"); ax.set_ylabel("m^-3 s^-1")
        ax.plot(xH2, SH);   ax.text(xH2[midH2[4]], 2*SH[midH2[4]],    '+'+_H+'('+_HH+')')
        ax.plot(xH2, SHP);  ax.text(xH2[midH2[1]], 1.2*SHP[midH2[1]], '+'+_Hp+'('+_HH+')')
        ax.plot(xH2, SP);   ax.text(xH2[midH2[2]], 1.2*SP[midH2[2]],  '+'+_p+'('+_HH+')')
        ax.plot(xH2, NuLoss*nHP); ax.text(xH2[midH2[4]], 1.2*NuLoss[midH2[4]]*nHP[midH2[4]], '-'+_Hp+'(LIM)')
        ax.plot(xH2, NuDis*nHP);  ax.text(xH2[midH2[0]], 1.2*NuDis[midH2[0]]*nHP[midH2[0]], '-'+_Hp+'(Dis)')
        ax.plot(xH2, SpH2);        ax.text(xH2[midH2[2]], 0.8*SpH2[midH2[2]],     '+'+_HH+'(LIM)')
        ax.plot(xH, SideWallH);    ax.text(xH[midH[0]],    0.8*SideWallH[midH[0]], '+'+_HH+'(Side Wall)')
        ax.plot(xH, Sion);         ax.text(xH[midH[5]],    1.1*Sion[midH[5]],      '-'+_H+'(Ion)')
        ax.plot(xH, h_output['SRecomb']);      ax.text(xH[midH[1]],    1.1*h_output['SRecomb'][midH[1]],   '+'+_H+'(Rec)')
        kn1d_limiter_include(ax, xlimiter, xsep)
        plt.tight_layout()
        plt.show()

        # x-Momentum Transfer
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        ydata = np.hstack([h2_output['RxH2CX'], h2_output['RxH_H2'], h2_output['RxP_H2'], h2_output['RxW_H2'], h_output['RxHCX'], h_output['RxH2_H'], h_output['RxP_H'], h_output['RxW_H']])
        ax.set_ylim((ydata.min(), ydata.max()))
        ax.set_title("KN1D: x-Momentum Transfer Rates")
        ax.set_xlabel("x (m)"); ax.set_ylabel("N m^-3")
        ax.plot(xH2, h2_output['RxH2CX']);    ax.text(xH2[midH2[3]], h2_output['RxH2CX'][midH2[3]],   _p+'→'+_HH)
        ax.plot(xH2, h2_output['RxH_H2']);    ax.text(xH2[midH2[1]], 1.2*h2_output['RxH_H2'][midH2[1]],  _H+'→'+_HH)
        ax.plot(xH2, h2_output['RxP_H2']);    ax.text(xH2[midH2[2]], h2_output['RxP_H2'][midH2[2]],    _p+'→'+_HH)
        ax.plot(xH2,-h2_output['RxW_H2']);    ax.text(xH2[midH2[0]], -1.2*h2_output['RxW_H2'][midH2[0]], _HH+'→Side Wall')
        ax.plot(xH,  h_output['RxHCX']);     ax.text(xH[midH[4]],    h_output['RxHCX'][midH[4]],      _p+'→'+_H+'(CX)')
        ax.plot(xH,  h_output['RxH2_H']);    ax.text(xH[midH[1]], 1.2*h_output['RxH2_H'][midH[1]], _HH+'→'+_H)
        ax.plot(xH,  h_output['RxP_H']);     ax.text(xH[midH[5]],    h_output['RxP_H'][midH[5]],      _p+'→'+_H+'(EL)')
        ax.plot(xH, -h_output['RxW_H']);     ax.text(xH[midH[0]], -1.2*h_output['RxW_H'][midH[0]],   _H+'→Side Wall')
        kn1d_limiter_include(ax, xlimiter, xsep)
        plt.tight_layout()
        plt.show()

        # Energy Transfer
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        f = 1e-6
        ydata = np.hstack([h2_output['EH2CX'], h2_output['EH_H2'], h2_output['EP_H2'], h2_output['EW_H2'], h_output['EHCX'], h_output['EH2_H'], h_output['EP_H'], h_output['EW_H']]) * f
        ax.set_ylim((ydata.min(), ydata.max()))
        ax.set_title("KN1D: Energy Transfer Rates")
        ax.set_xlabel("x (m)"); ax.set_ylabel("MW m^-3")
        ax.plot(xH2, h2_output['EH2CX']*f);  ax.text(xH2[midH2[2]], h2_output['EH2CX'][midH2[2]]*f, _p+'→'+_HH)
        ax.plot(xH2, h2_output['EH_H2']*f);  ax.text(xH2[midH2[1]], h2_output['EH_H2'][midH2[1]]*f, _H+'→'+_HH)
        ax.plot(xH2, h2_output['EP_H2']*f);  ax.text(xH2[midH2[3]], h2_output['EP_H2'][midH2[3]]*f, _p+'→'+_HH)
        ax.plot(xH2,-h2_output['EW_H2']*f);  ax.text(xH2[midH2[0]],-h2_output['EW_H2'][midH2[0]]*f, _HH+'→Side Wall')
        ax.plot(xH,  h_output['EHCX']*f);   ax.text(xH[midH[5]],    h_output['EHCX'][midH[5]]*f,  _p+'→'+_H+'(CX)')
        ax.plot(xH,  h_output['EH2_H']*f);  ax.text(xH[midH[4]],    h_output['EH2_H'][midH[4]]*f, _HH+'→'+_H)
        ax.plot(xH,  h_output['EP_H']*f);   ax.text(xH[midH[4]],    h_output['EP_H'][midH[4]]*f,  _p+'→'+_H+'(EL)')
        ax.plot(xH,-h_output['EW_H']*f);    ax.text(xH[midH[0]],   -h_output['EW_H'][midH[0]]*f,  _H+'→Side Wall')
        kn1d_limiter_include(ax, xlimiter, xsep)
        plt.tight_layout()
        plt.show()

        # Temperature Isotropization
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        ydata = np.hstack([h2_output['Epara_PerpH2_H2'], h_output['Epara_PerpH_H']])
        ax.set_ylim((ydata.min(), ydata.max()))
        ax.set_title("KN1D: T⊥→T∥ Isotropization Rates")
        ax.set_xlabel("x (m)"); ax.set_ylabel("W m^-3")
        ax.plot(xH2, h2_output['Epara_PerpH2_H2']); ax.text(xH2[midH2[2]], h2_output['Epara_PerpH2_H2'][midH2[2]], _HH+'↔'+_HH+'(EL)')
        ax.plot(xH,  h_output['Epara_PerpH_H']);  ax.text(xH[midH[3]],  h_output['Epara_PerpH_H'][midH[3]],  _H+'↔'+_H+'(EL)')
        kn1d_limiter_include(ax, xlimiter, xsep)
        plt.tight_layout()
        plt.show()

        # Heat Fluxes
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        f = 1e-3
        ydata = np.hstack([qxH_total, qxH2_total]) * f
        ax.set_ylim((ydata.min(), ydata.max()))
        ax.set_title("KN1D: Heat Fluxes")
        ax.set_xlabel("x (m)"); ax.set_ylabel("kW m^-2")
        ax.plot(xH2, qxH2_total*f); ax.text(xH2[midH2[2]], qxH2_total[midH2[2]]*f, _HH)
        ax.plot(xH,  qxH_total*f);  ax.text(xH[midH[3]],  qxH_total[midH[3]]*f,  _H)
        kn1d_limiter_include(ax, xlimiter, xsep)
        plt.tight_layout()
        plt.show()

        # Emissivities
        fig, ax = plt.subplots()
        mid, midH, midH2 = kn1d_include(ax, x, xH, xH2, GaugeH2, File, HH_label=_HH)
        f = 1e-3
        ydata = np.hstack([100*Balmer, Lyman]) * f
        ax.set_ylim((ydata.min(), ydata.max()))
        ax.set_title("KN1D: Balmer-α×100 & Lyman-α Emissivities")
        ax.set_xlabel("x (m)"); ax.set_ylabel("kW m^-3")
        ax.plot(xH, 100*Balmer*f); ax.text(xH[midH[4]], 100*Balmer[midH[4]]*f, 'Balmerx100')
        ax.plot(xH, Lyman*f);       ax.text(xH[midH[4]], Lyman[midH[4]]*f,       'Lyman-α')
        kn1d_limiter_include(ax, xlimiter, xsep)
        plt.tight_layout()
        plt.show()
        
    return

        







                






