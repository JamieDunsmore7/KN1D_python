#
# Create_JH_BSCoef.pro
#
# Creates a .npz file storing Bi-cubic spline interpolation
# coefficients for parameters in Johnson-Hinov rate equations.

import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from scipy.interpolate import bisplev

def Create_JH_BSCoef(output_file='jh_bscoef.npz'):
    # Axes
    dens = np.array([1.e10, 1.e11, 1.e12, 1.e13, 1.e14, 1.e15, 1.e16])
    temp = np.array([0.345, 0.69, 1.38, 2.76, 5.52, 11.0, 22.1, 44.1, 88.0, 176.5, 706.])

    # R values (shape 7x11x2x5)
    R = np.zeros((7, 11, 2, 5))
    # Example:
    R[:, :, 0, 0] = np.array([
        7.6E-6, 1.1E-5, 1.9E-5, 4.9E-5, 2.4E-4, 2.2E-3, 1.8e-2,
        1.5E-3, 1.8E-3, 2.5E-3, 4.5E-3, 1.3E-2, 7.1E-2, 3.7e-1,
        2.6E-2, 2.9E-2, 3.5E-2, 4.9E-2, 9.6E-2, 3.2E-1, 7.8e-1,
        1.3E-1, 1.4E-1, 1.5E-1, 1.9E-1, 2.8E-1, 6.1E-1, 9.2e-1,
        3.6E-1, 3.7E-1, 3.8E-1, 4.2E-1, 5.2E-1, 8.0E-1, 9.6e-1,
        6.9E-1, 6.9E-1, 7.0E-1, 7.3E-1, 7.9E-1, 9.2E-1, 9.8e-1,
        1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0,
        1.5, 1.5, 1.5, 1.5, 1.4, 1.1, 1.0,
        2.0, 2.0, 1.9, 1.9, 1.7, 1.3, 1.0,
        2.4, 2.4, 2.4, 2.3, 2.1, 1.4, 1.1,
        3.4, 3.4, 3.3, 3.2, 2.9, 2.0, 1.2
    ]).reshape((7, 11), order='F')

    R[:, :, 1, 0] = np.array([
        2.5E-7, 2.5E-6, 2.5E-5, 2.5E-4, 2.5E-3, 2.4E-2, 2.0e-1,
        1.9E-7, 1.9E-6, 1.9E-5, 1.9E-4, 1.9E-3, 1.8E-2, 1.0e-1,
        1.6E-7, 1.6E-6, 1.6E-5, 1.6E-4, 1.5E-3, 1.1E-2, 3.2e-2,
        1.5E-7, 1.5E-6, 1.5E-5, 1.5E-4, 1.3E-3, 7.2E-3, 1.3e-2,
        1.6E-7, 1.6E-6, 1.6E-5, 1.5E-4, 1.3E-3, 5.4E-3, 8.0e-3,
        1.8E-7, 1.8E-6, 1.8E-5, 1.7E-4, 1.4E-3, 5.1E-3, 7.0e-3,
        2.1E-7, 2.1E-6, 2.1E-5, 2.0E-4, 1.6E-3, 5.6E-3, 7.5e-3,
        2.3E-7, 2.3E-6, 2.3E-5, 2.2E-4, 1.7E-3, 6.3E-3, 8.7e-3,
        2.3E-7, 2.3E-6, 2.3E-5, 2.2E-4, 1.8E-3, 7.0E-3, 1.0e-2,
        2.2e-7, 2.2e-6, 2.1e-5, 2.1e-4, 1.7e-3, 7.4e-3, 1.1e-2,
        1.6E-7, 1.6E-6, 1.6E-5, 1.6E-4, 1.4E-3, 7.2E-3, 1.3e-2
    ]).reshape((7, 11), order='F')

    R[:, :, 0, 1] = np.array([
        2.2E-3, 3.1E-3, 6.0E-3, 2.2E-2, 1.3E-1, 3.5E-1, 4.2e-1,
        2.6E-2, 3.3E-2, 5.0E-2, 1.2E-1, 4.3E-1, 7.2E-1, 8.5e-1,
        1.1E-1, 1.3E-1, 1.6E-1, 3.0E-1, 6.8E-1, 8.9E-1, 9.7e-1,
        2.7E-1, 2.9E-1, 3.4E-1, 5.0E-1, 8.2E-1, 9.5E-1, 9.9e-1,
        4.8E-1, 5.0E-1, 5.4E-1, 6.8E-1, 9.0E-1, 9.8E-1, 1.0,
        7.3E-1, 7.4E-1, 7.7E-1, 8.5E-1, 9.5E-1, 9.9E-1, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.3, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0,
        1.6, 1.6, 1.5, 1.4, 1.1, 1.0, 1.0,
        1.9, 1.9, 1.8, 1.6, 1.2, 1.1, 1.0,
        2.5, 2.4, 2.4, 2.1, 1.5, 1.1, 1.0,
    ]).reshape((7, 11), order='F')

    R[:, :, 1, 1] = np.array([
        1.0E-7,1.0E-6,1.0E-5,1.0E-4,1.1E-3,1.3E-2,1.1e-1,
        8.2E-8,8.1E-7,8.0E-6,7.7E-5,6.1E-4,4.5E-3,4.2e-2,
        7.1E-8,7.0E-7,6.8E-6,5.9E-5,3.3E-4,1.6E-3,4.1e-3,
        6.8E-8,6.7E-7,6.3E-6,4.9E-5,2.1E-4,7.3E-4,1.2e-3,
        7.2E-8,7.0E-7,6.5E-6,4.7E-5,1.8E-4,4.9E-4,6.8e-4,
        8.1E-8,7.8E-7,7.2E-6,5.1E-5,1.9E-4,4.5E-4,5.8e-4,
        9.1E-8,8.9E-7,8.2E-6,5.8E-5,2.1E-4,5.0E-4,6.4e-4,
        9.7E-8,9.5E-7,8.8E-6,6.5E-5,2.5E-4,6.0E-4,7.6e-4,
        9.7E-8,9.4E-7,8.8E-6,6.7E-5,2.7E-4,6.9E-4,9.1e-4,
        8.9e-8,8.7e-7,8.2e-6,6.5e-5,2.8e-4,7.6e-4,1.0e-3,
        6.5E-8,6.4E-7,6.1E-6,5.2E-5,2.7E-4,8.0E-4,1.2e-3
    ]).reshape((7, 11), order='F')

    R[:,:,0,2] = np.array([
        1.8E-2, 2.8E-2, 7.3E-2, 3.1E-1, 6.0E-1, 7.4E-1, 7.7e-1,
        8.2E-2, 1.1E-1, 2.2E-1, 5.6E-1, 8.3E-1, 9.3E-1, 9.6e-1,
        2.0E-1, 2.4E-1, 3.9E-1, 7.4E-1, 9.2E-1, .98, .99,
        3.7E-1, 4.1E-1, 5.7E-1, 8.4E-1, 9.6E-1, .99, 1.0,
        5.6E-1, 6.0E-1, 7.2E-1, 9.0E-1, 9.8E-1, 1.0, 1.0,
        7.7E-1, 7.9E-1, 8.5E-1, 9.5E-1, 9.9E-1, 1.0, 1.0,
        9.9E-1, 9.9E-1, 9.9E-1, 1.0, 1.0, 1.0, 1.0,
        1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 1.0,
        1.4, 1.4, 1.3, 1.1, 1.0, 1.0, 1.0,
        1.7, 1.6, 1.5, 1.2, 1.1, 1.0, 1.0,
        2.1, 2.1, 1.9, 1.5, 1.1, 1.0, 1.0
    ]).reshape((7,11), order='F')

    R[:,:,1,2] = np.array([
        7.2E-8, 7.1E-7, 6.9E-6, 5.7E-5, 4.8E-4, 5.3E-3, 4.5e-2,
        5.9E-8, 5.7E-7, 5.1E-6, 3.1E-5, 1.7E-4, 1.1E-3, 5.9e-3,
        5.1E-8, 4.9E-7, 4.0E-6, 1.9E-5, 7.1E-5, 3.0E-4, 7.8e-4,
        4.8E-8, 4.5E-7, 3.4E-6, 1.4E-5, 4.2E-5, 1.3E-4, 2.1e-4,
        5.0E-8, 4.7E-7, 3.4E-6, 1.3E-5, 3.5E-5, 8.6E-5, 1.2e-4,
        5.6E-8, 5.2E-7, 3.7E-6, 1.4E-5, 3.6E-5, 8.1E-5, 1.0e-4,
        6.3E-8, 5.9E-7, 4.3E-6, 1.6E-5, 4.3E-5, 9.3E-5, 1.2e-4,
        6.7E-8, 6.3E-7, 4.8E-6, 1.9E-5, 5.2E-5, 1.1E-4, 1.4e-4,
        6.6E-8, 6.3E-7, 4.9E-6, 2.1E-5, 5.9E-5, 1.3E-4, 1.7e-4,
        6.1e-8, 5.8e-7, 4.7e-6, 2.2e-5, 6.3e-5, 1.4e-4, 2.0e-4,
        4.4E-8, 4.2E-7, 3.7E-6, 2.0E-5, 6.4E-5, 1.6E-4, 2.5e-4
    ]).reshape((7, 11), order='F')

    R[:,:,0,3] = np.array([
        5.5E-2, 1.E-1, 3.3E-1, 6.8E-1, 8.5E-1, .9, .92,
        1.5E-1, 2.4E-1, 5.5E-1, 8.4E-1, 9.5E-1, .98, .99,
        2.9E-1, 4.0E-1, 7.0E-1, 9.1E-1, 9.8E-1, .99, 1.0,
        4.5E-1, 5.5E-1, 8.E-1, 9.5E-1, 9.9E-1, 1.0, 1.0,
        6.2E-1, 7.0E-1, 8.7E-1, 9.7E-1, 9.9E-1, 1.0, 1.0,
        8.0E-1, 8.4E-1, 9.3E-1, 9.8E-1, 1.0, 1.0, 1.0,
        9.8E-1, 9.8E-1, 9.9E-1, 1.0, 1.0, 1.0, 1.0,
        1.2, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0,
        1.4, 1.3, 1.2, 1.0, 1.0, 1.0, 1.0,
        1.5, 1.5, 1.3, 1.1, 1.0, 1.0, 1.0,
        1.9, 1.9, 1.6, 1.2, 1.0, 1.0, 1.0
    ]).reshape((7, 11), order='F')
    
    R[:,:,1,3] = np.array([
        6.0E-8, 5.7E-7, 4.4E-6, 2.5E-5, 1.8E-4, 2.0E-3, 1.6e-2,
        4.8E-8, 4.4E-7, 2.7E-6, 1.1E-5, 5.0E-5, 3.2E-4, 1.7e-3,
        4.1E-8, 3.5E-7, 1.8E-6, 5.9E-6, 1.9E-5, 8.1E-5, 2.0e-4,
        3.8E-8, 3.2E-7, 1.5E-6, 4.2E-6, 1.1E-5, 3.4E-5, 5.5e-5,
        4.0E-8, 3.2E-7, 1.4E-6, 3.8E-6, 9.4E-6, 2.3E-5, 3.1e-5,
        4.4E-8, 3.6E-7, 1.6E-6, 4.3E-6, 1.0E-5, 2.2E-5, 2.8e-5,
        5.0E-8, 4.1E-7, 1.9E-6, 5.2E-6, 1.2E-5, 2.5E-5, 3.2e-5,
        5.3E-8, 4.5E-7, 2.2E-6, 6.2E-6, 1.5E-5, 3.1E-5, 3.9e-5,
        5.2E-8, 4.5E-7, 2.4E-6, 7.1E-6, 1.7E-5, 3.7E-5, 4.8e-5,
        4.8e-8, 4.3e-7, 2.4e-6, 7.6e-6, 1.9e-5, 4.3e-5, 5.7e-5,
        3.5E-8, 3.2E-7, 2.1E-6, 7.5E-6, 2.0E-5, 4.8E-5, 7.2e-5
    ]).reshape((7, 11), order='F')

    R[:,:,0,4] = np.array([
        1.1E-1, 2.7E-1, 6.4E-1, 8.6E-1, 9.4E-1, .96, .97,
        2.4E-1, 4.5E-1, 7.9E-1, 9.4E-1, 9.8E-1, .99, 1.0,
        3.8E-1, 6.E-1, 8.7E-1, 9.7E-1, 9.9E-1, 1.0, 1.0,
        5.3E-1, 7.2E-1, 9.1E-1, 9.8E-1, 1.0, 1.0, 1.0,
        6.8E-1, 8.1E-1, 9.4E-1, 9.9E-1, 1.0, 1.0, 1.0,
        8.2E-1, 9.0E-1, 9.7E-1, 9.9E-1, 1.0, 1.0, 1.0,
        9.7E-1, 9.9E-1, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.3, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0,
        1.5, 1.3, 1.1, 1.0, 1.0, 1.0, 1.0,
        1.8, 1.7, 1.3, 1.1, 1.0, 1.0, 1.0
    ]).reshape((7, 11), order='F')

    R[:,:,1,4] = np.array([
        5.2E-8, 4.3E-7, 2.3E-6, 1.0E-5, 7.2E-5, 7.7E-4, 6.5e-3,
        4.1E-8, 3.0E-7, 1.2E-6, 4.0E-6, 1.7E-5, 1.1E-4, 5.9e-4,
        3.4E-8, 2.2E-7, 7.7E-7, 2.1E-6, 6.6E-6, 2.7E-5, 6.8e-5,
        3.1E-8, 1.9E-7, 6.0E-7, 1.5E-6, 3.8E-6, 1.1E-5, 1.8e-5,
        3.2E-8, 1.9E-7, 5.9E-7, 1.4E-6, 3.2E-6, 7.7E-6, 1.0e-5,
        3.6E-8, 2.1E-7, 6.7E-7, 1.6E-6, 3.5E-6, 7.5E-6, 9.5e-6,
        4.1E-8, 2.5E-7, 8.2E-7, 1.9E-6, 4.3E-6, 8.9E-6, 1.1e-5,
        4.4E-8, 2.9E-7, 9.8E-7, 2.4E-6, 5.3E-6, 1.1E-5, 1.4e-5,
        4.4E-8, 3.0E-7, 1.1E-6, 2.7E-6, 6.3E-6, 1.3E-5, 1.7e-5,
        4.0e-8, 3.0e-7, 1.2e-6, 3.0e-6, 7.0e-6, 1.5e-5, 2.1e-5,
        3.0E-8, 2.4E-7, 1.1E-6, 3.1E-6, 7.5E-6, 1.8E-5, 2.6e-5
    ]).reshape((7, 11), order='F')


    # S and alpha (shape 7x11)
    S = np.zeros((7, 11))
    alpha = np.zeros((7, 11))
    
    S[:,0] = [2.1e-26,3.2e-26,6.5e-26,2.1e-25,1.3e-24,1.4e-23,1.2e-22]
    S[:,1] = [1.0e-17,1.3e-17,2.0e-17,4.3e-17,1.5e-16,9.4e-16,5.0e-15]
    S[:,2] = [3.0e-13,3.4e-13,4.4e-13,7.1e-13,1.7e-12,6.1e-12,1.5e-11]
    S[:,3] = [6.7e-11,7.3e-11,8.6e-11,1.1e-10,2.0e-10,4.9e-10,7.6e-10]
    S[:,4] = [1.3e-9,1.4e-9,1.5e-9,1.9e-9,2.7e-9,5.0e-9,6.4e-9]
    S[:,5] = [6.9e-9,7.2e-9,7.7e-9,8.9e-9,1.2e-8,1.9e-8,2.2e-8]
    S[:,6] = [1.8e-8,1.8e-8,1.9e-8,2.1e-8,2.7e-8,4.0e-8,4.5e-8]
    S[:,7] = [2.8e-8,2.9e-8,3.0e-8,3.3e-8,4.1e-8,5.8e-8,6.7e-8]
    S[:,8] = [3.4e-8,3.5e-8,3.6e-8,3.9e-8,4.8e-8,6.7e-8,7.7e-8]
    S[:,9] = [3.4e-8,3.4e-8,3.6e-8,3.9e-8,4.7e-8,6.5e-8,7.7e-8]
    S[:,10] = [2.5e-8,2.6e-8,2.6e-8,2.8e-8,3.3e-8,4.6e-8,5.8e-8]

    S *= 1e-6

    alpha[:,0] = [1.2e-12,1.7e-12,2.9e-12,7.1e-12,2.7e-11,1.6e-10,1.4e-9]
    alpha[:,1] = [6.1e-13,7.3e-13,1.0e-12,1.7e-12,3.9e-12,1.4e-11,7.1e-11]
    alpha[:,2] = [3.3e-13,3.6e-13,4.3e-13,5.7e-13,9.2e-13,2.0e-12,4.8e-12]
    alpha[:,3] = [1.8e-13,1.9e-13,2.1e-13,2.4e-13,3.1e-13,4.8e-13,7.0e-13]
    alpha[:,4] = [1.0e-13,1.0e-13,1.1e-13,1.2e-13,1.3e-13,1.6e-13,1.9e-13]
    alpha[:,5] = [5.6e-14,5.7e-14,5.7e-14,5.9e-14,6.1e-14,6.5e-14,7.2e-14]
    alpha[:,6] = [3.0e-14,3.0e-14,3.0e-14,3.0e-14,3.0e-14,3.0e-14,3.2e-14]
    alpha[:,7] = [1.5e-14,1.5e-14,1.5e-14,1.5e-14,1.5e-14,1.4e-14,1.5e-14]
    alpha[:,8] = [7.3e-15,7.3e-15,7.2e-15,7.1e-15,6.9e-15,6.6e-15,6.7e-15]
    alpha[:,9] = [3.4e-15,3.4e-15,3.3e-15,3.3e-15,3.2e-15,3.0e-15,3.0e-15]
    alpha[:,10] = [6.5e-16,6.5e-16,6.4e-16,6.4e-16,6.2e-16,5.8e-16,5.7e-16]

    alpha *= 1e-6

    # A coefficients
    A_lyman = np.array([4.699E8, 5.575E7, 1.278E7, 4.125E6, 1.644E6, 7.568E5, 3.869E5,
                        2.143E5, 1.263E5, 7.834E4, 5.066E4, 3.393E4, 2.341E4, 1.657E4, 1.200E4])
    A_balmer = np.array([4.41E7, 8.42E6, 2.53E6, 9.732E5, 4.389e5, 2.215e5, 1.216e5,
                         7.122e4, 4.397e4, 2.83e4, 18288.8, 12249.1, 8451.26, 5981.95, 4332.13])
    

    # Convert R to log
    log_dens = np.log(dens * 1e6)  # Convert to m^-3
    log_temp = np.log(temp)



    # Compute and store B-spline coefficients for R
    print('Computing B-Spline coefficients for R values')
    order = 4
    kx = ky = order - 1

    R_coeffs_dict = {}
    R_tx_ref = R_ty_ref = None
    R_kx_ref = R_ky_ref = None

    for nIon in range(2):
        for ip in range(2, 7):  # ip from 2 to 6
            z = R[:, :, nIon, ip - 2]
            spline = RectBivariateSpline(log_dens, log_temp, np.log(z), kx=kx, ky=ky)
            R_tx, R_ty, R_coeffs = spline.tck
            R_deg_x, R_deg_y = spline.degrees

            # On first run, store reference
            if R_tx_ref is None:
                R_tx_ref, R_ty_ref = R_tx, R_ty
                R_kx_ref, R_ky_ref = R_deg_x, R_deg_y
            else:
                # Check that all subsequent splines match the reference
                assert np.allclose(R_tx, R_tx_ref), f"tx mismatch for nIon={nIon}, ip={ip}"
                assert np.allclose(R_ty, R_ty_ref), f"ty mismatch for nIon={nIon}, ip={ip}"
                assert R_deg_x == R_kx_ref, f"kx mismatch for nIon={nIon}, ip={ip}"
                assert R_deg_y == R_ky_ref, f"ky mismatch for nIon={nIon}, ip={ip}"

            key = f'R_coeffs_{nIon}_{ip}'
            R_coeffs_dict[key] = R_coeffs



    spline_S = RectBivariateSpline(log_dens, log_temp, np.log(S), kx=kx, ky=ky)
    spline_alpha = RectBivariateSpline(log_dens, log_temp, np.log(alpha), kx=kx, ky=ky)
    LogS_BSCoef = spline_S.tck[2].ravel()
    LogAlpha_BSCoef = spline_alpha.tck[2].ravel()

    S_tx, S_ty, S_c = spline_S.tck
    S_kx, S_ky   = spline_S.degrees   # these are attributes on the spline object
    S_full_tck = (S_tx, S_ty, S_c, S_kx, S_ky)

    alpha_tx, alpha_ty, alpha_c = spline_alpha.tck
    alpha_kx, alpha_ky = spline_alpha.degrees  # these are attributes on the spline object
    alpha_full_tck = (alpha_tx, alpha_ty, alpha_c, alpha_kx, alpha_ky)


    # save the S spline full tck
    np.savez(
        output_file,
        # S spline components
        S_tx=S_tx,
        S_ty=S_ty,
        S_c =S_c,
        S_kx=S_kx,
        S_ky=S_ky,
        # α spline components
        alpha_tx=alpha_tx,
        alpha_ty=alpha_ty,
        alpha_c =alpha_c,
        alpha_kx=alpha_kx,
        alpha_ky=alpha_ky,
        # B-spline coefficients for R
        R_tx =R_tx_ref,
        R_ty =R_ty_ref,
        R_kx =R_kx_ref,
        R_ky =R_ky_ref,
        **R_coeffs_dict, # unpack the R coefficients dictionary
        A_lyman=A_lyman,
        A_balmer=A_balmer,
    )

    print(f'Saved data to {output_file}')

    '''
    # now read in the same output file

    data = np.load(output_file, allow_pickle=True)

    S_tck     = (data['S_tx'],     data['S_ty'],     data['S_c'],     int(data['S_kx']),     int(data['S_ky']))
    alpha_tck = (data['alpha_tx'], data['alpha_ty'], data['alpha_c'], int(data['alpha_kx']), int(data['alpha_ky']))


    #S_tck = data['S_tck']
    #alpha_tck = data['alpha_tck']

    # example density values
    dens = np.array([10**12, 10**13, 10**14]) # in cm^-3
    temp = np.linspace(1, 700, 1000)

    log_dens = np.log(dens * 1e6)  # Convert to m^-3
    log_temp = np.log(temp)

    S_evaluated = np.exp(bisplev(log_dens, log_temp, S_tck))
    alpha_evaluated = np.exp(bisplev(log_dens, log_temp, alpha_tck))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(len(dens)):
        cx = S_evaluated[i, :]
        plt.plot(temp, cx*1e6, label=f'Density: {dens[i]} m^-3')
    plt.xlabel('Temperature')
    plt.ylabel('Sigma V')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1, 1000])
    plt.ylim([10e-12, 10e-8])
    plt.legend()
    plt.grid()
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(len(dens)):
        cx = alpha_evaluated[i, :]
        plt.plot(temp, cx*1e6, label=f'Density: {dens[i]} m^-3')
    plt.xlabel('Temperature')
    plt.ylabel('Sigma V')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1, 1000])
    plt.ylim([10e-17, 10e-12])
    plt.legend()
    plt.grid()
    plt.show()


    np.savez(
        output_file,
        DKnot=Dknot,
        TKnot=Tknot,
        order=order,
        LogR_BSCoef=LogR_BSCoef,
        LogS_BSCoef=LogS_BSCoef,
        LogAlpha_BSCoef=LogAlpha_BSCoef,
        A_lyman=A_lyman,
        A_balmer=A_balmer,
    )
    

    print(f'Saved data to {output_file}')
    '''

Create_JH_BSCoef()


