import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from monte_carlo import monte_carlo
from tqdm import tqdm
import config as cfg
import random
import snr


'''
data_vis.py

This code is a data visualization for the Monte Carlo simulation.
'''

def data_vis(retardance_err, polarizer_err, analyzer_err, pol_mod_err, residual_birefringence_err, extinction_ratio, detector_noise, sampling_noise, snr_analysis, n_iter, psi_inputs, delta_inputs, psi_locked, delta_locked, mc_sim, delta_plots=False, psi_plots=False, stokes_plots=False, other_params=False, extinction_ratio_plots=False, snr_analysis_plots=False):

    '''
    This function runs the monte_carlo() function at a range of Delta and Psi values with a defined set of errors.

    Inputs:
        - retardance_err [float]: Peak retardance error [%]
        - polarizer_err [float]: Maximum angular error for polarizer [deg]
        - analyzer_err [float]: Maximum angular error for analyzer [deg]
        - pol_mod_err [float]: Maximum angular error between polarizer and modulator [deg]
        - residual_birefringence_err [float]: Static retardation error [rad.]
        - extinction_ratio [bool]: If set to true, do a wavelength vs. error analysis with polarizer extinction ratio. Set to False if any other simulation is desired.
        - detector_noise [bool]: If set to true, add detector noise equal to 1 / SNR
        - sampling_noise [bool]: If set to true, sampling rate is limited
        - snr_analysis [bool]: If set to true, do a wavelength vs. error analysis with detector SNR. Set to False if any other simulation is desired.
        - n_iter [int]: Number of iterations for Monte Carlo simulations. Set to 1 if worst-case scenario analysis is desired.
        - psi_inputs [arr]: Range of Psi inputs to run Monte Carlo simulation
        - delta_inputs [arr]: Range of Delta inputs to run Monte Carlo simulation
        - psi_locked [float]: Value at which Psi is held constant while Delta varies
        - delta_locked [float]: Value at which Delta is held constant while Psi varies
        - mc_sim [bool]: Set to true if Monte Carlo analysis is desired, and false if worst-case scenario.
        - delta_plots [bool]: Set to true if Delta error and inputs vs outputs plots are desired.
        - psi_plots [bool]: Set to true if Psi error and inputs vs outputs plots are desired.
        - stokes_plots [bool]: Set to true if Stokes parameters plots are desired.
        - other_params [bool]: Set to true if DoLP, AoLP, and DoCP plots are desired.
        - extinction_ratio_plots [bool]: Set to true if extinction ratio plots are desired.
        - snr_analysis_plots [bool]: Set to true if SNR plots are desired.

    Outputs:
        - s_1_inputs [arr]: Stokes parameter 1 inputs
        - s_1_outputs [arr]: Stokes parameter 1 outputs
        - s_2_inputs [arr]: Stokes parameter 2 inputs
        - s_2_outputs [arr]: Stokes parameter 2 outputs
        - s_3_inputs [arr]: Stokes parameter 3 inputs
        - s_3_outputs [arr]: Stokes parameter 3 outputs
        - delta_outputs [arr]: Delta outputs 
        - psi_outputs [arr]: Psi outputs
    
    '''
    delta_error_er = []
    psi_error_er = []
    delta_error_nsr = []
    psi_error_nsr = []
    
    ### EXTINCTION RATIO LOOP ###
    if extinction_ratio:
        psi = np.deg2rad(45)
        delta = np.deg2rad(180)
        for er in tqdm(cfg.extinction):
            er = ((1 / er) + (1 / er))
            psi_output, delta_output, _, _, _, _, _, _ = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_err, detector_noise, sampling_noise, er, 0], [psi, delta], mc_sim, config='delta')
            delta_error_er.append(np.abs(np.deg2rad(delta_output) - delta))
            psi_output, delta_output, _, _, _, _, _, _ = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_err, detector_noise, sampling_noise, er, 0], [psi, delta], mc_sim, config='psi')
            psi_error_er.append(np.abs(np.deg2rad(psi_output) - psi))

    ### SIGNAL TO NOISE RATIO LOOP ###
    if snr_analysis:
        psi = np.deg2rad(45)
        delta = np.deg2rad(180)
        nsr = np.asarray([snr.wavelength, snr.SNR])
        for noise in tqdm(nsr[1, :]):
            psi_output, delta_output, _, _, _, _, _, _ = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_err, detector_noise, sampling_noise, 0, noise], [psi, delta], mc_sim, config='delta')
            delta_error_nsr.append(np.abs(np.deg2rad(delta_output) - delta))
            psi_output, delta_output, _, _, _, _, _, _ = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_err, detector_noise, sampling_noise, 0, noise], [psi, delta], mc_sim, config='psi')
            psi_error_nsr.append(np.abs(np.deg2rad(psi_output) - psi))

    # Initialize important arrays
    delta_outputs = np.zeros((np.shape(delta_inputs)[0], n_iter))
    psi_outputs = np.zeros((np.shape(psi_inputs)[0], n_iter))
    s_1_inputs = np.zeros((np.shape(psi_inputs)[0], 1))
    s_2_inputs = np.zeros((np.shape(psi_inputs)[0], 1))
    s_3_inputs = np.zeros((np.shape(psi_inputs)[0], 1))
    dolp_inputs = np.zeros((np.shape(psi_inputs)[0], 1))
    aolp_inputs = np.zeros((np.shape(psi_inputs)[0], 1))
    docp_inputs = np.zeros((np.shape(psi_inputs)[0], 1))
    s_1_outputs = np.zeros((np.shape(psi_inputs)[0], n_iter))
    s_2_outputs = np.zeros((np.shape(psi_inputs)[0], n_iter))
    s_3_outputs = np.zeros((np.shape(psi_inputs)[0], n_iter))
    dolp_outputs = np.zeros((np.shape(psi_inputs)[0], n_iter))
    aolp_outputs = np.zeros((np.shape(psi_inputs)[0], n_iter))
    docp_outputs = np.zeros((np.shape(psi_inputs)[0], n_iter))
    delta_error = []
    psi_error = []
    stokes_error = []
    dolp_error = []
    aolp_error = []
    docp_error = []   

    ### MAIN DELTA LOOP ###
    for idx in tqdm(range(len(delta_inputs))):
        # Delta varies, Psi stays locked
        delta = np.deg2rad(delta_inputs[idx])
        # Run the Monte Carlo Simulation at the delta and psi value.
        psi_list, delta_list, _, _, _, _, _, _ = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_err, detector_noise, sampling_noise, 0, 0], [psi_locked, delta], mc_sim, config='delta')
        # Append outputs to lists.
        delta_outputs[idx, :] = delta_list
        delta_error.append(np.max(abs(delta_list - np.rad2deg(delta))))

    ### MAIN PSI LOOP ###
    for idx in tqdm(range(len(psi_inputs))):
        psi = np.deg2rad(psi_inputs[idx])
        # Calculate expected Stokes parameters for troubleshooting.
        s_1_inputs[idx] = -np.cos(2*psi)
        s_2_inputs[idx] = np.sin(2*psi)*np.cos(delta_locked)
        s_3_inputs[idx] = -np.sin(2*psi)*np.sin(delta_locked)
        dolp_inputs[idx] = np.sqrt(s_1_inputs[idx]**2 + s_2_inputs[idx]**2)
        aolp_inputs[idx] = (1 / 2) * np.arctan2(s_2_inputs[idx], s_1_inputs[idx])
        if s_1_inputs[idx] > 0 and s_2_inputs[idx] <= 0:
            aolp_inputs[idx] += np.pi
        elif s_1_inputs[idx] <= 0:
            aolp_inputs[idx] += np.pi / 2
        docp_inputs[idx] = s_3_inputs[idx]
        psi_list, delta_list, s_1_list, s_2_list, s_3_list, dolp_list, aolp_list, docp_list = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_err, detector_noise, sampling_noise, 0, 0], [psi, delta_locked], mc_sim, config='psi')
        psi_outputs[idx, :] = psi_list
        s_1_outputs[idx, :] = s_1_list
        s_2_outputs[idx, :] = s_2_list
        s_3_outputs[idx, :] = s_3_list
        dolp_outputs[idx, :] = dolp_list
        aolp_outputs[idx, :] = aolp_list
        docp_outputs[idx, :] = docp_list
        stokes_error.append([np.max(abs(s_1_list - s_1_inputs[idx])), np.max(abs(s_2_list - s_2_inputs[idx])), np.max(abs(s_3_list - s_3_inputs[idx]))]) 
        dolp_error.append(np.max(abs(dolp_list - dolp_inputs[idx])))
        aolp_error.append(np.max(abs(np.rad2deg(aolp_list) - np.rad2deg(aolp_inputs[idx]))))
        docp_error.append(np.max(abs(docp_list - docp_inputs[idx])))
        psi_error.append(np.max(abs(psi_list - np.rad2deg(psi))))

    stokes_error = np.asarray(stokes_error)

    # Plot delta inputs vs. outputs and error if true
    if delta_plots:
        # Plot inputs vs outputs
        plt.rc('font', size=24)
        plt.figure()
        plt.grid()
        plt.plot(delta_inputs, delta_inputs, 'lime', linewidth=3.0)
        for delta_num, delta_spread in enumerate(delta_outputs):
            delta_input = [delta_inputs[delta_num]]*n_iter
            plt.scatter(delta_input, delta_spread, color='blue', s=36)
        plt.title(f"$\Delta$ outputs as a function of $\Delta$ inputs at $\Psi$ = {np.round(np.rad2deg(psi_locked), 1)}°")
        plt.xlabel("$\Delta$ inputs [°]")
        plt.ylabel("$\Delta$ outputs [°]")
        plt.legend(["Expected $\Delta$ outputs", "Simulated $\Delta$ outputs"])
        plt.show()
        
        # Plot delta error
        plt.figure()
        plt.grid()
        plt.plot(delta_inputs, delta_error, linewidth=3.0)
        plt.title(f"$\Delta$ error at $\Psi$ = {np.rad2deg(psi_locked)}°")
        plt.xlabel("$\Delta$ inputs [°]")
        plt.ylabel("$\Delta$ error [°]")
        plt.show()
    
    # Plot Psi inputs vs outputs and error if true
    if psi_plots:
        # Plot inputs vs outputs
        plt.rc('font', size=24)
        plt.figure()
        plt.grid()
        plt.plot(psi_inputs, psi_inputs, 'lime', linewidth=3.0)
        for psi_num, psi_spread in enumerate(psi_outputs):
            psi_input = [psi_inputs[psi_num]]*n_iter
            plt.scatter(psi_input, psi_spread, color='red', s=36)
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.title(f"$\Psi$ outputs as a function of $\Psi$ inputs at $\Delta$ = {np.round(np.rad2deg(delta_locked), 1)}°")
        plt.xlabel("$\Psi$ inputs [°]")
        plt.ylabel("$\Psi$ outputs [°]")
        plt.legend(["Expected $\Psi$ outputs", "Simulated $\Psi$ outputs"])
        plt.show()

        # Plot psi error
        plt.rc('font', size=24)
        plt.figure()
        plt.grid()
        plt.plot(psi_inputs, psi_error, linewidth=3.0)
        plt.title(f"$\Psi$ error at $\Delta$ = {np.rad2deg(delta_locked)}°")
        plt.xlabel("$\Psi$ inputs [°]")
        plt.ylabel("$\Psi$ error [°]")
        plt.show()

    # Plot Stokes inputs vs outputs and error if true
    if stokes_plots:
        plt.rc('font', size=24)
        fig, ax = plt.subplots(nrows=3, ncols=1)
        for s_1_num, s_1_spread in enumerate(s_1_outputs):
            s_1_input = [s_1_inputs[s_1_num]]*n_iter
            ax[0].scatter(s_1_input, s_1_spread, color='green')
        ax[0].set_title(f"Stokes parameter outputs as a function of Stokes inputs at $\Delta$ = {np.round(np.rad2deg(delta_locked), 1)}°")
        ax[0].set_ylabel("$S_1$ outputs")
        ax[0].set_xlabel("$S_1$ inputs")
        ax[0].plot(s_1_inputs, s_1_inputs, 'black')
        for s_2_num, s_2_spread in enumerate(s_2_outputs):
            s_2_input = [s_2_inputs[s_2_num]]*n_iter
            ax[1].scatter(s_2_input, s_2_spread, color='green')
        ax[1].set_ylabel("$S_2$ outputs")
        ax[1].set_xlabel("$S_2$ inputs")
        ax[1].plot(s_2_inputs, s_2_inputs, 'black')
        for s_3_num, s_3_spread in enumerate(s_3_outputs):
            s_3_input = [s_3_inputs[s_3_num]]*n_iter
            ax[2].scatter(s_3_input, s_3_spread, color='green')
        ax[2].set_ylabel("$S_3$ outputs")
        ax[2].plot(s_3_inputs, s_3_inputs, 'black')
        ax[2].set_xlabel("$S_3$ inputs")
        plt.show()

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(psi_inputs, stokes_error[:, 0], color='green')
        ax[0].set_title(f"Stokes parameter error as a function of $\Psi$ inputs at $\Delta$ = {np.round(np.rad2deg(delta_locked), 1)}°")
        ax[0].set_ylabel("$S_1$ error")
        ax[0].set_xlabel("$\Psi$ inputs [°]")
        ax[1].plot(psi_inputs, stokes_error[:, 1], color='green')
        ax[1].set_ylabel("$S_2$ error")
        ax[1].set_xlabel("$\Psi$ inputs [°]")
        ax[2].plot(psi_inputs, stokes_error[:, 2], color='green')
        ax[2].set_ylabel("$S_3$ error")
        ax[2].set_xlabel("$\Psi$ inputs [°]")
        plt.show()

    if other_params:
        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(psi_inputs, dolp_error, color='purple')
        ax[0].set_title(f"Polarization Parameter error as a function of $\Psi$ inputs at $\Delta$ = {np.round(np.rad2deg(delta_locked), 1)}°")
        ax[0].set_ylabel("DoLP error")
        ax[0].set_xlabel("$\Psi$ inputs [°]")
        ax[1].plot(psi_inputs, aolp_error, color='orange')
        ax[1].set_ylabel("AoLP error [°]")
        ax[1].set_xlabel("$\Psi$ inputs [°]")
        ax[2].plot(psi_inputs, docp_error, color='pink')
        ax[2].set_ylabel("DoCP error")
        ax[2].set_xlabel("$\Psi$ inputs [°]")
        plt.show()

    if extinction_ratio:
        if extinction_ratio_plots:
            # Plot psi error
            plt.rc('font', size=24)
            plt.figure()
            plt.grid()
            plt.plot(cfg.er[:, 0], np.rad2deg(psi_error_er), linewidth=3.0)
            plt.title(f"$\Psi$ error due to extinction ratio")
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("$\Psi$ error [°]")
            plt.show()

            # Plot delta error
            plt.rc('font', size=24)
            plt.figure()
            plt.grid()
            plt.plot(cfg.er[:, 0], np.rad2deg(delta_error_er), linewidth=3.0)
            plt.title(f"$\Delta$ error due to extinction ratio")
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("$\Delta$ error [°]")
            plt.show()

    if snr_analysis:
        if snr_analysis_plots:
            # Plot psi error
            plt.rc('font', size=24)
            plt.figure()
            plt.grid()
            plt.plot(nsr[0, :], psi_error_nsr, linewidth=3.0)
            plt.title(f"$\Psi$ error due to detector noise")
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("$\Psi$ error [°]")
            plt.show()

            # Plot delta error
            plt.rc('font', size=24)
            plt.figure()
            plt.grid()
            plt.plot(nsr[0, :], delta_error_nsr, linewidth=3.0)
            plt.title(f"$\Delta$ error due to detector noise")
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("$\Delta$ error [°]")
            plt.show()
        
    return s_1_inputs, s_1_outputs, s_2_inputs, s_2_outputs, s_3_inputs, s_3_outputs, delta_outputs, psi_outputs



if __name__ == '__main__':
    # If only one set of errors needed for analysis, this file can be used as MAIN by modifiying the parameters below and running this file.

    retardance_err = 0
    polarizer_err = 0
    analyzer_err = 0
    pol_mod_err = 0
    residual_birefringence_err = 0
    extinction_ratio = False
    detector_noise = False
    sampling_noise = False
    snr_analysis = False
    n_iter = 1

    psi_inputs = np.arange(1, 90, 1)
    delta_inputs = np.arange(1, 180, 2)

    psi_locked = np.deg2rad(45)
    delta_locked = np.deg2rad(160)

    mc_sim = False

    _, _, _, _, _, _, _, _ = data_vis(retardance_err, polarizer_err, analyzer_err, pol_mod_err, residual_birefringence_err, extinction_ratio, detector_noise, sampling_noise, snr_analysis, n_iter, psi_inputs, delta_inputs, psi_locked, delta_locked, mc_sim)