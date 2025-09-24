import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# --- Load data ---
h_data = np.load('test_kn1d_cmod.KN1D_H.npz')
xH = h_data['xH']
nH = h_data['nH']
Sion = h_data['Sion']
GammaxH = h_data['GammaxH']

cmod_results_file = 'cmod_test_out_for_comparison.pkl'
with open(cmod_results_file, 'rb') as f:
    cmod_results = pkl.load(f)

with open('cmod_test_in.pkl', 'rb') as f:
    input_data = pkl.load(f)

x_lim = input_data['x_lim']
x_sep = input_data['x_sep']

# --- Plot setup ---
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True, constrained_layout=True)

labels = {"Python": "tab:blue", "IDL": "tab:orange"}

# --- Top panel (log scale) ---
ax = axes[0]
ax.plot(xH, nH, label="Python", color=labels["Python"], marker="x", linestyle="-")
ax.plot(cmod_results['xh'], cmod_results['nh'], label="IDL", color=labels["IDL"], linestyle="--", marker="x")
ax.axvline(x=x_lim, color="k", linestyle="--", label="Limiter")
ax.axvline(x=x_sep, color="r", linestyle="--", label="Separatrix")

ax.set_yscale("log")
ax.set_ylabel("nH (m⁻³)", fontsize=16)
ax.grid(True, which="both", ls="--", alpha=0.6)
ax.legend(fontsize=14, frameon=False)

# --- Bottom panel (linear scale) ---
ax = axes[1]
ax.plot(xH, nH, label="Python", color=labels["Python"], marker="x", linestyle="-")
ax.plot(cmod_results['xh'], cmod_results['nh'], label="IDL", color=labels["IDL"], linestyle="--", marker="x")
ax.axvline(x=x_lim, color="k", linestyle="--", label="Limiter")
ax.axvline(x=x_sep, color="r", linestyle="--", label="Separatrix")

ax.set_ylabel("nH (m⁻³)", fontsize=16)
ax.set_xlabel("x (m)", fontsize=16)
ax.grid(True, ls="--", alpha=0.6)

for ax in axes:
    ax.tick_params(axis="both", which="major", labelsize=16)


# --- Save & show ---
plt.savefig("cmod_test_compare_nH.png", dpi=300)
plt.show()

