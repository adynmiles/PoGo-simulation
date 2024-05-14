import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from scipy.special import jv

'''
polarimetry.py

Measures polarization states using Jones matrices.

'''

def polarimetry(A_angle, P_angle, M_angle, retardance_angle, psi_input, delta_input, detector_noise, sampling_noise, plot_intensity=False):
    '''
    This function produces an input signal to the detector and returns the intensity of the signal, varying in time.

    Inputs:
        - A_angle [float]: Analyzer angle [deg]
        - P_angle [float]: Polarizer angle [deg]
        - M_angle [float]: PEM angle [deg]
        - retardance_angle [float]: retardance amplitude [deg]
        - psi_input [float]: Psi input [deg]
        - delta_input [float]: Delta input [deg]
        - detector_noise [bool]: Add noise equal to 1 / SNR if true
        - sampling_noise [bool]: Limit sampling rate of detector to 60000 Hz if true.
        - plot_intensity [bool]: Display intensity plot if true. 

    Outputs:
        - I [arr]: signal intensity as a function of time.
        - expected_coeffs [arr]: each expected coefficient of the intensity equation [coeff_0, coeff_1, coeff_2]
    '''

    ### INPUTS INITIALIZATION ###
    t_meas = cfg.t_meas # Total simulation time

    # If true, limit the sampling rate.
    if sampling_noise:
        sample_rate = 60000
    else:
        sample_rate = 200000

    dt = 1 / sample_rate # Timestep
    t = np.arange(0, t_meas, dt) # Array of all timestamps

    # If true, add detector noise equal to 1 / SNR.
    if detector_noise:
        noise = np.random.uniform(-cfg.nsr, cfg.nsr, sample_rate)

    # Bessel terms calculation
    J_0 = jv(0, retardance_angle)
    J_1 = jv(1, retardance_angle)
    J_2 = jv(2, retardance_angle)
    psi = psi_input # Psi input
    delta = delta_input # Delta input
    omega = cfg.omega # PEM frequency

    I_0 = 0.5 # Averaged input intensity for unpolarized light

    # Calculation of alpha_0, alpha_1, and alpha_2 in parts.
    alpha_0_1 = np.cos(2*psi)*np.cos(2*A_angle)
    alpha_0_2 = np.cos(2*(P_angle - M_angle))*np.cos(2*M_angle)*(np.cos(2*A_angle) - np.cos(2*psi))
    alpha_0_2 = np.cos(2*(np.deg2rad(45)))*np.cos(2*M_angle)*(np.cos(2*A_angle) - np.cos(2*psi))
    alpha_0_3 = np.sin(2*A_angle)*np.cos(delta)*np.cos(2*(P_angle - M_angle))*np.sin(2*psi)*np.sin(2*M_angle)
    alpha_0_3 = np.sin(2*A_angle)*np.cos(delta)*np.cos(2*(np.deg2rad(45)))*np.sin(2*psi)*np.sin(2*M_angle)
    alpha_0 = 1 - alpha_0_1 + alpha_0_2 + alpha_0_3
    alpha_1 = np.sin(2*(P_angle - M_angle))*np.sin(2*A_angle)*np.sin(2*psi)*np.sin(delta)
    alpha_1 = np.sin(2*(np.deg2rad(45)))*np.sin(2*A_angle)*np.sin(2*psi)*np.sin(delta)
    alpha_2_1 = (np.cos(2*psi) - np.cos(2*A_angle))*np.sin(2*M_angle)
    alpha_2_2 = (np.sin(2*A_angle) * np.cos(2*M_angle) * np.sin(2*psi) * np.cos(delta))
    alpha_2 = np.sin(2*(P_angle - M_angle))*(alpha_2_1 + alpha_2_2)
    alpha_2 = np.sin(2*(np.deg2rad(45)))*(alpha_2_1 + alpha_2_2)
        
    I = []  # Initialize intensity array

    # Main loop where the signal intensity is built using the coefficients and the time steps.
    for step, t_step in enumerate(t):
        I.append(I_0 * (alpha_0 + (2*alpha_1*J_1*np.sin(omega*t_step)) + (alpha_2*(J_0 + (2*J_2))*np.cos(2*omega*t_step))))
        if detector_noise:
            I[step] += noise[step]

    # Storing the expected coefficient values for comparison with lock-in amplifier outputs.
    coeff_0 = I_0*alpha_0
    coeff_1 = I_0*2*alpha_1*J_1
    coeff_2 = I_0*alpha_2*(J_0 + (2*J_2))

    # If true, plot signal intensity for the first few periods.
    if plot_intensity:
        plt.rc('font', size=18)
        plt.figure()
        plt.plot(t[:100]*(1e6), I[:100], linewidth=3.0)
        plt.grid()
        plt.title("Signal read by detector")
        plt.xlabel("Time elapsed ($\mu s$)")
        plt.ylabel("Intensity")
        plt.show()
        

    I = np.asarray(I)

    return I, [coeff_0, coeff_1, coeff_2]