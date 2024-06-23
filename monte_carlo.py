import numpy as np
import matplotlib.pyplot as plt
import random
import config as cfg
from polarimetry import polarimetry
from psi_delta import psi_delta, nsc
from tqdm import tqdm


'''
monte_carlo.py

This script runs a Monte Carlo simulation of the various perturbations that can be applied to the system.
This includes angular perturbations, retardance perturbations, and extinction ratio.
'''

def monte_carlo(n_iter, errors, inputs, mc_sim, config='delta', show_plots=False):
    '''
    This function runs a Monte Carlo simulation for a given set of errors at a given Psi and Delta input. It uses the psi_delta.py
    file to apply the lock-in amplifier to the signal and calculate Psi and Delta.

    Inputs:
        - n_iter [int]: number of iterations for the Monte Carlo simulation.
        - errors [arr]: list of errors to track for the Monte Carlo simulation, [polarizer error, analyzer error, polarizer-modulator relative error, retardance error, static retardation error, detector noise error, sampling error]
        - inputs [arr]: angular inputs for Monte Carlo simulation, [psi, delta]
        - mc_sim [bool]: true if a full Monte Carlo simulation should be performed, or false if a simulation that has the worst-case scenario is desired.
        - config [str]: set to 'psi' for a psi calculation, and 'delta' for a delta calculation.
        - psi_plot [bool]: if true, plots Monte Carlo simulation of Psi
        - delta_plot [bool]: if true, plots Monte Carlo simulation of Delta
    '''

    ### ORGANIZING INPUTS ###
    polarizer_err = errors[0]   # Error in polarizer angle
    analyzer_err = errors[1]    # Error in analyzer angle
    pol_mod_err = errors[2]     # Error between polarizer and modulator positions
    retardance_err = errors[3]  # Error in peak retardance
    residual_birefringence_err = errors[4] # Static retardation error
    detector_noise = errors[5]  # Detector noise due to SNR
    sampling_noise = errors[6]  # Sampling noise from pre-amplifier limitations.
    extinction_ratio = errors[7]  # Extinction ratio from polarizers
    noise = errors[8]

    psi_input = inputs[0] # Psi for the input signal
    delta_input = inputs[1] # Delta for the input signal
    
    configurations = [1, 2] # Both configurations needed for a full measurement

    # Initializing important arrays
    psi_list = []
    delta_list = []
    S_1_list = []
    S_2_list = []
    S_3_list = []
    dolp_list = []
    aolp_list = []
    docp_list = []

    ### MAIN MONTE CARLO LOOP ###
    for i in tqdm(range(int(n_iter))):
        # Perform simulation for each configuration to get both delta and psi.
        for configuration in configurations:
            if configuration == 1:
                # Add random error within maximum range if mc_sim set to True
                if mc_sim:
                    P_angle = np.deg2rad(45 + random.uniform(-polarizer_err, polarizer_err))
                    A_angle = np.deg2rad(-45 + random.uniform(-analyzer_err, analyzer_err))
                    M_angle = np.deg2rad(np.rad2deg(P_angle) - 45 + random.uniform(-pol_mod_err, pol_mod_err))
                    retardance_angle = np.deg2rad(138 + (random.uniform(-retardance_err*138, retardance_err*138)))
                # Add worst case scenario if mc_sim set to False
                else:
                    P_angle = np.deg2rad(45 + polarizer_err)
                    A_angle = np.deg2rad(-45 + analyzer_err)
                    M_angle = np.deg2rad(np.rad2deg(P_angle) - 45 + pol_mod_err)
                    retardance_angle = np.deg2rad(138 + (retardance_err*138))
            # Same logic applies to configuration 2
            elif configuration == 2:
                if mc_sim:
                    P_angle = np.deg2rad(0 + random.uniform(-polarizer_err, polarizer_err))
                    A_angle = np.deg2rad(-45 + random.uniform(-analyzer_err, analyzer_err))
                    if config == 'delta':
                        M_angle = np.deg2rad(np.rad2deg(P_angle) + random.uniform(-pol_mod_err, pol_mod_err))
                    elif config == 'psi':
                        M_angle = np.deg2rad(np.rad2deg(P_angle) - 45 + random.uniform(-pol_mod_err, pol_mod_err))
                    retardance_angle = np.deg2rad(138 + (random.uniform(-retardance_err*138, retardance_err*138)))
                else:
                    P_angle = np.deg2rad(0 + polarizer_err)
                    A_angle = np.deg2rad(-45 + analyzer_err)
                    if config == 'delta':
                        M_angle = np.deg2rad(np.rad2deg(P_angle))
                    elif config == 'psi':
                        M_angle = np.deg2rad(np.rad2deg(P_angle) - 45 + pol_mod_err)
                    retardance_angle = np.deg2rad(138 + (retardance_err*138))
        
            # First gather the input signal
            I, expected_coeffs, = polarimetry(A_angle, P_angle, M_angle, retardance_angle, psi_input, delta_input, detector_noise, sampling_noise, extinction_ratio, noise)
            # Calculating N, S, and C using both configurations.
            if configuration == 1:
                N, S, _, N_actual, S_actual, _ = nsc(I, A_angle, configuration, expected_coeffs, residual_birefringence_err, sampling_noise)
            elif configuration == 2:
                _, _, C, _, _, C_actual = nsc(I, A_angle, configuration, expected_coeffs, residual_birefringence_err, sampling_noise)
        # Calculating Psi, Delta, and the other parameters once N, S, and C have all been calculated
        psi, delta, stokes, other_params = psi_delta(N, S, C, [N_actual, S_actual, C_actual], config, delta_input, psi_input)

        # Update output lists
        psi_list.append(np.rad2deg(psi))
        delta_list.append(np.rad2deg(delta))
        S_1_list.append(stokes[0])
        S_2_list.append(stokes[1])
        S_3_list.append(stokes[2])
        dolp_list.append(other_params[0])
        aolp_list.append(other_params[1])
        docp_list.append(other_params[2])

    # If this function is set to the psi configuration, output a histogram showing the outputs relative to the input.
    if config == 'psi' and show_plots:
        plt.rc('font', size=18)
        plt.figure()
        psi_hist, bins, patches = plt.hist(psi_list, bins=20)
        plt.axvline(x=np.rad2deg(psi_input), color='red', linewidth=2, linestyle='--')
        plt.title(f"Simulated $\Psi$ Error Results at $\Psi$ = {np.rad2deg(psi_input)}°, $\Delta$ = {np.rad2deg(delta_input)}°")
        plt.xlabel("$\Psi$ [°]")
        plt.ylabel("Count")
        plt.show()

    # If this is function is set to the delta configuration, output a histogram showing the outputs relative to the input.
    if config == 'delta' and show_plots:
        plt.rc('font', size=18)
        plt.figure()
        delta_hist, bins, patches = plt.hist(delta_list, bins=20)
        plt.axvline(x=np.rad2deg(delta_input), color='red', linewidth=2, linestyle='--')
        plt.title(f"Simulated $\Delta$ Error Results at $\Psi$ = {np.rad2deg(psi_input)}°, $\Delta$ = {np.rad2deg(delta_input)}°")
        plt.xlabel("$\Delta$ [°]")
        plt.ylabel("Count")
        plt.show()

    return psi_list, delta_list, S_1_list, S_2_list, S_3_list, dolp_list, aolp_list, docp_list


if __name__ == '__main__':
    # If only the psi and delta at a defined set of inputs is desired, this file can be run as MAIN using the following reconfigurable code.
    psi = cfg.psi
    delta = cfg.delta

    polarizer_err = 1
    analyzer_err = 0
    pol_mod_err = 0
    retardance_err = 0
    residual_birefringence_error = 0
    detector_noise = False
    sampling_noise = False

    n_iter = 100

    mc_sim = True

    psi_list, delta_list, s_1_list, s_2_list, s_3_list, dolp_list, aolp_list, docp_list = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_error, detector_noise, sampling_noise], [psi, delta], mc_sim=True, config='delta', show_plots=True)
