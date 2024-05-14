import numpy as np
import matplotlib.pyplot as plt
from monte_carlo import monte_carlo
from tqdm import tqdm


'''
data_vis.py

This code is a data visualization for the Monte Carlo simulation.
'''

def data_vis(retardance_err, polarizer_err, analyzer_err, pol_mod_err, residual_birefringence_err, detector_noise, sampling_noise, n_iter, psi_inputs, delta_inputs, psi_locked, delta_locked, mc_sim, delta_plots=True, psi_plots=True, stokes_plots=False):

    '''
    This function runs the monte_carlo() function at a range of Delta and Psi values with a defined set of errors.

    Inputs:
        - retardance_err [float]: Peak retardance error [%]
        - polarizer_err [float]: Maximum angular error for polarizer [deg]
        - analyzer_err [float]: Maximum angular error for analyzer [deg]
        - pol_mod_err [float]: Maximum angular error between polarizer and modulator [deg]
        - residual_birefringence_err [float]: Static retardation error [rad.]
        - detector_noise [bool]: If set to true, add detector noise equal to 1 / SNR
        - sampling_noise [bool]: If set to true, sampling rate is limited
        - n_iter [int]: Number of iterations for Monte Carlo simulations. Set to 1 if worst-case scenario analysis is desired.
        - psi_inputs [arr]: Range of Psi inputs to run Monte Carlo simulation
        - delta_inputs [arr]: Range of Delta inputs to run Monte Carlo simulation
        - psi_locked [float]: Value at which Psi is held constant while Delta varies
        - delta_locked [float]: Value at which Delta is held constant while Psi varies
        - mc_sim [bool]: Set to true if Monte Carlo analysis is desired, and false if worst-case scenario.
        - delta_plots [bool]: Set to true if Delta error and inputs vs outputs plots are desired.
        - psi_plots [bool]: Set to true if Psi error and inputs vs outputs plots are desired.
        - stokes_plots [bool]: Set to true if Stokes parameters plots are desired.

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
    
    # Initialize important arrays
    delta_outputs = np.zeros((np.shape(delta_inputs)[0], n_iter))
    psi_outputs = np.zeros((np.shape(psi_inputs)[0], n_iter))
    s_1_inputs = np.zeros((np.shape(delta_inputs)[0], 1))
    s_2_inputs = np.zeros((np.shape(delta_inputs)[0], 1))
    s_3_inputs = np.zeros((np.shape(delta_inputs)[0], 1))
    s_1_outputs = np.zeros((np.shape(delta_inputs)[0], n_iter))
    s_2_outputs = np.zeros((np.shape(delta_inputs)[0], n_iter))
    s_3_outputs = np.zeros((np.shape(delta_inputs)[0], n_iter))
    delta_error = []
    psi_error = []


    ### MAIN DELTA LOOP ###
    for idx in tqdm(range(len(delta_inputs))):
        # Delta varies, Psi stays locked
        delta = np.deg2rad(delta_inputs[idx])
        # Calculate expected Stokes parameters for troubleshooting.
        s_1_inputs[idx] = -np.cos(2*psi_locked)
        s_2_inputs[idx] = np.sin(2*psi_locked)*np.cos(delta)
        s_3_inputs[idx] = -np.sin(2*psi_locked)*np.sin(delta)
        # Run the Monte Carlo Simulation at the delta and psi value.
        psi_list, delta_list, s_1_list, s_2_list, s_3_list = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_err, detector_noise, sampling_noise], [psi_locked, delta], mc_sim, config='delta')
        # Append outputs to lists.
        delta_outputs[idx, :] = delta_list
        s_1_outputs[idx, :] = s_1_list
        s_2_outputs[idx, :] = s_2_list
        s_3_outputs[idx, :] = s_3_list
        delta_error.append(np.max(abs(delta_list - np.rad2deg(delta))))

    ### MAIN PSI LOOP ###
    for idx in tqdm(range(len(psi_inputs))):
        psi = np.deg2rad(psi_inputs[idx])
        psi_list, delta_list, _, _, _ = monte_carlo(n_iter, [polarizer_err, analyzer_err, pol_mod_err, retardance_err, residual_birefringence_err, detector_noise, sampling_noise], [psi, delta_locked], mc_sim, config='psi')
        psi_outputs[idx, :] = psi_list
        psi_error.append(np.max(abs(psi_list - np.rad2deg(psi))))

    # Plot delta inputs vs. outputs and error if true
    if delta_plots:
        # Plot inputs vs outputs
        plt.figure()
        plt.grid()
        plt.rc('font', size=18)
        plt.figure()
        plt.grid()
        plt.rc('font', size=18)
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
        plt.rc('font', size=18)
        plt.plot(delta_inputs, delta_error)
        plt.title(f"$\Delta$ error at $\Psi$ = {np.rad2deg(psi_locked)}°")
        plt.xlabel("$\Delta$ inputs [°]")
        plt.ylabel("$\Delta$ error [°]")
        plt.show()
    
    # Plot Psi inputs vs outputs and error if true
    if psi_plots:
        # Plot inputs vs outputs
        plt.figure()
        plt.grid()
        plt.rc('font', size=18)
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
        plt.figure()
        plt.grid()
        plt.rc('font', size=18)
        plt.plot(psi_inputs, psi_error)
        plt.title(f"$\Psi$ error at $\Delta$ = {np.rad2deg(delta_locked)}°")
        plt.xlabel("$\Psi$ inputs [°]")
        plt.ylabel("$\Psi$ error [°]")
        plt.show()

    # Plot Stokes inputs vs outputs and error if true
    if stokes_plots:
        fig, ax = plt.subplots(nrows=3, ncols=1)
        for s_1_num, s_1_spread in enumerate(s_1_outputs):
            s_1_input = [s_1_inputs[s_1_num]]*n_iter
            ax[0].scatter(s_1_input, s_1_spread, color='green')
        ax[0].set_title(f"Stokes parameter error as a function of $\Delta$ inputs at $\Psi$ = {np.round(np.rad2deg(psi_locked), 1)}°")
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

    return s_1_inputs, s_1_outputs, s_2_inputs, s_2_outputs, s_3_inputs, s_3_outputs, delta_outputs, psi_outputs


if __name__ == '__main__':
    # If only one set of errors needed for analysis, this file can be used as MAIN by modifiying the parameters below and running this file.

    retardance_err = 0
    polarizer_err = 0
    analyzer_err = 0
    pol_mod_err = 0
    residual_birefringence_err = 0
    detector_noise = False
    sampling_noise = False
    n_iter = 1

    psi_inputs = np.arange(1, 90, 1)
    delta_inputs = np.arange(1, 180, 2)

    psi_locked = np.deg2rad(45)
    delta_locked = np.deg2rad(160)

    mc_sim = False


    _, _, _, _, _, _, _, _ = data_vis(retardance_err, polarizer_err, analyzer_err, pol_mod_err, residual_birefringence_err, detector_noise, sampling_noise, n_iter, psi_inputs, delta_inputs, psi_locked, delta_locked, mc_sim)