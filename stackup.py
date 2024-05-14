import numpy as np
import matplotlib.pyplot as plt
from data_vis import data_vis

'''
stackup.py

This file helps with visualizing how different errors can stack onto each other. It can go through a range of delta and psi values multiple times, each time with a different set of errors, and plot them on a single plot.
'''

# Set both to True to see stackup plots for Delta and Psi.
delta_plot = True
psi_plot = True

# Set to True if Monte Carlo simulation desired, false if worst-case scenario desired.
mc_sim = True

### CONFIGURABLE INPUTS ###

# Inputs, can put in multiple configurations of retardance_err, polarizer_err, and analyzer_err. Other
# parameters do not have tested functionality for an array of values. Current upper limit is three for error length in the stackup.
retardance_err = [0, 0, 0.01]   # Peak retardance error
polarizer_err = [0, 1, 1]       # Maximum polarizer error [deg]
analyzer_err = [1, 1, 1]        # Maximum analyzer error [deg]
pol_mod_err = 0                 # Polarizer - modulator relative error [deg]
residual_birefringence_err = 1e-5   # Static retardation error [rad]
detector_noise = True           # Add detector noise equal to 1 / SNR if true
sampling_noise = True           # Limit sampling rate if true

n_iter = 100  # Number of iterations for Monte Carlo simulation
psi_inputs = np.arange(44.5, 45.5, 0.05)    # Range for Psi inputs
delta_inputs = np.arange(160, 161, 0.05)    # Range for Delta inputs

psi_locked = np.deg2rad(45)  # Where to keep Psi while Delta is varying
delta_locked = np.deg2rad(160)  # Where to keep Delta while Psi is varying

# Initialize important arrays
s_1_inputs = np.zeros((len(retardance_err), np.shape(delta_inputs)[0], n_iter))
s_2_inputs = np.zeros((len(retardance_err), np.shape(delta_inputs)[0], n_iter))
s_3_inputs = np.zeros((len(retardance_err), np.shape(delta_inputs)[0], n_iter))
s_1_outputs = np.zeros((len(retardance_err), np.shape(delta_inputs)[0], n_iter))
s_2_outputs = np.zeros((len(retardance_err), np.shape(delta_inputs)[0], n_iter))
s_3_outputs = np.zeros((len(retardance_err), np.shape(delta_inputs)[0], n_iter))
delta_outputs = np.zeros((len(retardance_err), np.shape(delta_inputs)[0], n_iter))
psi_outputs = np.zeros((len(retardance_err), np.shape(psi_inputs)[0], n_iter))

### MAIN LOOP ###
for error_num, error in enumerate(retardance_err):
    # Run data visualization code for a given set of input errors.
    s_1_inputs[error_num, :, :], s_1_outputs[error_num, :, :], s_2_inputs[error_num, :, :], s_2_outputs[error_num, :, :], s_3_inputs[error_num, :, :], s_3_outputs[error_num, :, :], delta_outputs[error_num, :, :], psi_outputs[error_num, :, :] = data_vis(error, polarizer_err[error_num], analyzer_err[error_num], pol_mod_err, residual_birefringence_err, detector_noise, sampling_noise, n_iter, psi_inputs, delta_inputs, psi_locked, delta_locked, mc_sim, delta_plots=False, psi_plots=False, stokes_plots=False)

# If true, plot Delta stackup
if delta_plot:
    plt.rc('font', size=18)
    plt.figure()
    colors = ['blue', 'green', 'red']
    # Iterate through error list to get delta outputs and error for each set of inputs.
    for i in range(len(retardance_err)):
        minimum_spread = []
        maximum_spread = []
        for j in range(len(delta_outputs[i])):
            # Calculate boundaries of Monte Carlo simulation
            minimum_spread.append(np.min(delta_outputs[i, j]))
            maximum_spread.append(np.max(delta_outputs[i, j]))
        plt.plot(delta_inputs, minimum_spread, color=colors[i], label='_nolegend_')
        plt.plot(delta_inputs, maximum_spread, color=colors[i], label='_nolegend_')
        label_text = f"Analyzer Error: {analyzer_err[i]}°, Polarizer Error: {polarizer_err[i]}°, Retardance Error: {retardance_err[i]}"
        # Shade area to show minimum to maximum range.
        plt.fill_between(delta_inputs, minimum_spread, maximum_spread, facecolor=colors[i], color=colors[i], alpha=0.2, label=label_text)
    plt.plot(delta_inputs, delta_inputs, color='black', label='Expected $\Delta$ Outputs')
    plt.title("$\Delta$ output ranges with different error combinations")
    plt.xlabel("$\Delta$ inputs [°]")
    plt.ylabel("$\Delta$ outputs [°]")
    plt.legend()
    plt.show()

# If true, plot Psi stackup, with same logic as Delta stackup.
if psi_plot:
    plt.rc('font', size=18)
    plt.figure()
    colors = ['blue', 'green', 'red']
    for i in range(len(retardance_err)):
        minimum_spread = []
        maximum_spread = []
        for j in range(len(psi_outputs[i])):
            minimum_spread.append(np.min(psi_outputs[i, j]))
            maximum_spread.append(np.max(psi_outputs[i, j]))
        plt.plot(psi_inputs, minimum_spread, color=colors[i], label='_nolegend_')
        plt.plot(psi_inputs, maximum_spread, color=colors[i], label='_nolegend_')
        label_text = f"Analyzer Error: {analyzer_err[i]}°, Polarizer Error: {polarizer_err[i]}°, Retardance Error: {retardance_err[i]}"
        plt.fill_between(psi_inputs, minimum_spread, maximum_spread, facecolor=colors[i], color=colors[i], alpha=0.2, label=label_text)
    plt.plot(psi_inputs, psi_inputs, color='black', label='Expected $\Psi$ Outputs')
    plt.title("$\Psi$ output ranges with different error combinations")
    plt.xlabel("$\Psi$ inputs [°]")
    plt.ylabel("$\Psi$ outputs [°]")
    plt.legend()
    plt.show()