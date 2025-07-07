import numpy as np
import matplotlib.pyplot as plt
from sigmav_cx_hh import SigmaV_CX_HH

def main_test_sigmav_cx_hh():
    """
    Generates and plots reaction rates <sigma v> for
    H₂⁺ + H₂ → H₂ + H₂⁺ at various E₀ values.
    """
    # Define T (ion temperature in eV)
    Ti = 10.0 ** (-1 + (5 + np.log10(2)) * np.arange(101) / 100)
    
    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 2e4)
    ax.set_ylim(1.0e-10, 2.0e-7)
    
    ax.set_title(r'H$_2^+$ + H$_2$ $\rightarrow$ H$_2$ + H$_2^+$')
    ax.set_xlabel('T (eV)')
    ax.set_ylabel(r'<$\sigma$v> (cm$^3$/s)')
    
    ax.tick_params(axis='both', which='major', length=6)
    ax.tick_params(axis='both', which='minor', length=3)
    
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'cyan']
    
    # Loop over E₀ values and overplot
    for i in range(6):
        print('i', i)
        e0 = 10.0 ** (i - 1)
        E_array = np.full_like(Ti, e0)
        
        sigv = SigmaV_CX_HH(Ti, E_array)
        
        ax.plot(Ti, sigv * 1e6, color=colors[i], linewidth=3)
        ax.text(0.15, 0.7 + i * 0.04,
                f'E = {e0:.1e} eV',
                transform=ax.transAxes,
                fontsize=9,
                color=colors[i])
    
    # Global annotations
    ax.text(0.15, 0.18,
            'Output from python function which evaluates the reaction rates using:',
            transform=ax.transAxes,
            fontsize=9)
    ax.text(0.15, 0.15,
            'Data from Janev et al., "Elementary Processes in Hydrogen-Helium Plasmas", p 292.',
            transform=ax.transAxes,
            fontsize=9)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    main_test_sigmav_cx_hh()
