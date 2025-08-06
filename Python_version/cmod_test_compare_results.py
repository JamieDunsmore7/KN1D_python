import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


h_data = np.load('test_kn1d_cmod.KN1D_H.npz')
xH = h_data['xH']
nH = h_data['nH']
Sion = h_data['Sion']
GammaxH = h_data['GammaxH']


cmod_results_file = 'cmod_test_out_for_comparison.pkl'
with open(cmod_results_file, 'rb') as f:
    cmod_results = pkl.load(f)

plt.plot(xH, nH, label='Python KN1D', marker='x')
plt.plot(cmod_results['xh'], cmod_results['nh'], label='Andres', linestyle='dashed', marker='x')
plt.yscale('log')
plt.xlabel('x (m)')
plt.ylabel('nH (m⁻³)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.show()

breakpoint()
