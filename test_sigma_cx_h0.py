import numpy as np
import matplotlib.pyplot as plt
from sigma_cx_h0 import Sigma_CX_H0


def main(plot_both=True):
    """
    Generates and plots the charge-exchange cross-section for p + H(1s).
    
    Args:
        plot_both (bool): If True, plots data from both Freeman and Janev.
                          If False, plots only the Janev data.
                          This replaces the IDL 'key_default, both, 0' logic.
    """
    
    # --- Data Generation ---
    # Equivalent of: E=10.0^(-1+(5+alog10(2))*findgen(101)/100)
    E = 10.0**(-1 + (5 + np.log10(2)) * np.arange(101) / 100)

    # --- Plot Setup ---
    # Equivalent of the IDL 'plot, /nodata, ...' command
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set scales and limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 2e4)
    ax.set_ylim(1.0e-16, 1.0e-14)
    
    # Set titles and labels
    ax.set_title('p + H(1s) -> H(1s) + p')
    ax.set_xlabel('E (eV)')
    ax.set_ylabel('sigma (cm$^2$)') # Use LaTeX for formatting
    
    # Set tick parameters (Matplotlib defaults are usually sufficient)
    ax.tick_params(axis='both', which='major', length=6)
    ax.tick_params(axis='both', which='minor', length=3)
    
    # --- Plotting Logic ---
    if plot_both:
        # --- Freeman and Jones data ---
        freeman_flag = True
        sig_freeman = Sigma_CX_H0(E, freeman=freeman_flag)
        # Oplot with color=2 (red), thick=3
        # The 1e4 factor converts from m^2 to cm^2
        ax.plot(E, sig_freeman * 1e4, color='red', linewidth=3, label='Freeman & Jones')

        # --- Janev et al. data ---
        freeman_flag = False
        sig_janev = Sigma_CX_H0(E, freeman=freeman_flag)
        # Oplot with color=4 (blue), thick=3
        ax.plot(E, sig_janev * 1e4, color='blue', linewidth=3, label='Janev et al.')
        
        # --- Text annotations (equivalent to xyouts) ---
        # The 'transform=ax.transAxes' makes coordinates relative to the axes (0,0 to 1,1)
        ax.text(0.15, 0.20, 'Output from python function which evaluates the CX Cross-Section using:',
                transform=ax.transAxes, fontsize=9)
        ax.text(0.15, 0.17, 'Freeman and Jones, "Atomic Collision Processes..." table 2.',
                transform=ax.transAxes, fontsize=9, color='red')
        ax.text(0.15, 0.14, 'Polynomial fit from Janev et al., "Elementary Processes..." p 250.',
                transform=ax.transAxes, fontsize=9, color='blue')
        
    else: # Corresponds to the 'else' block in IDL
        # --- Janev et al. data only ---
        freeman_flag = False
        sig_janev = Sigma_CX_H0(E, freeman=freeman_flag)
        # Oplot with color=4 (blue), thick=3
        ax.plot(E, sig_janev * 1e4, color='blue', linewidth=3, label='Janev et al.')
        
        # --- Text annotations ---
        ax.text(0.15, 0.20, 'Output from python function which evaluates the CX Cross-Section using:',
                transform=ax.transAxes, fontsize=9)
        ax.text(0.15, 0.17, 'Polynomial fit from Janev et al., "Elementary Processes..." p 250.',
                transform=ax.transAxes, fontsize=9, color='blue')

    # Add a grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Display the plot
    plt.show()


if __name__ == '__main__':
    # To replicate the 'if both then' part of the script, run:
    main(plot_both=True)
    
    # To replicate the 'else' part of the script, run:
    # main(plot_both=False)