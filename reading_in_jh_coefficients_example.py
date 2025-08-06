# NOTE: not part of the core IDL code

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import bisplev

output_file = 'jh_spline_coefs_for_Steven.npz'

data = np.load(output_file, allow_pickle=True)

S_tck     = (data['S_tx'],     data['S_ty'],     data['S_c'],     int(data['S_kx']),     int(data['S_ky']))
alpha_tck = (data['alpha_tx'], data['alpha_ty'], data['alpha_c'], int(data['alpha_kx']), int(data['alpha_ky']))

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
plt.xlabel('Temperature (eV)')
plt.ylabel('Sigma V (cm^3/s)')
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
plt.xlabel('Temperature (eV)')
plt.ylabel('Sigma V (cm^3/s)')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1, 1000])
plt.ylim([10e-17, 10e-12])
plt.legend()
plt.grid()
plt.show()