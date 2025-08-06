import numpy as np
import matplotlib.pyplot as plt
from sigmav_h1s_h1s_hh import SigmaV_H1s_H1s_HH

def main_test_sigmav_h1s_h1s_hh():
    """
    Generates and plots reaction rates <sigma v> for
    e + H2 -> e + H(1s) + H(1s) over a range of Te.
    """
    # Define Te (Electron Temperature in eV)
    Te = 10.0 ** (-1 + (5 + np.log10(2)) * np.arange(101) / 100)
    
    # Compute <sigma v> (returns in m^3/s)
    sigv = SigmaV_H1s_H1s_HH(Te)
    
    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 2e4)
    ax.set_ylim(1.0e-11, 1e-7)

    print(sigv)
    
    # Plot the curve (convert to cm^3/s)
    ax.plot(Te, sigv * 1e6, color='red', linewidth=3)
    
    # Titles and labels
    ax.set_title(r'e + H$_2$ $\rightarrow$ e + H(1s) + H(1s)')
    ax.set_xlabel('Te (eV)')
    ax.set_ylabel(r'<$\sigma$v> (cm$^3$/s)')
    
    # Annotations
    ax.text(0.15, 0.90,
            'Output from python function which evaluates the reaction rates using',
            transform=ax.transAxes,
            fontsize=9)
    ax.text(0.15, 0.87,
            'Data from Janev et al., "Elementary Processes in Hydrogen-Helium Plasmas", p 259.',
            transform=ax.transAxes,
            fontsize=9)
    
    # Grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.show()

if __name__ == '__main__':
    main_test_sigmav_h1s_h1s_hh()
