import config as cfg
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfilt, freqz
import matplotlib.pyplot as plt
from scipy.special import jv

"""
psi_delta.py

This program applies lock-in amplifier operations to the input signal, and calculates delta and psi from the lock-in amplifier outputs. The Stokes
parameters are also measured in this operation.
"""


def nsc(
    I,
    A_angle,
    configuration,
    expected_coeffs,
    delta_0,
    sampling_error,
    show_filtered_outputs=False,
    show_magnitudes=False,
    show_coeffs_err=False,
):
    """
    This function applies lock-in amplifier operations to the input signal and calculates N, S, and C parameters. Must be run separately for each of the polarimetric configurations.

    Inputs:
        - I [arr]: input intensity at each time step
        - A_angle [float]: angle of analyzer (deg)
        - configuration [int]: specifies if configuration 1 or 2 is being measured
        - expected_coeffs [arr]: error checking mechanism, inputs coefficients of signal intensity
        - delta_0 [float]: static retardation of the PEM (rad.)
        - sampling_error [bool]: if true, apply sampling error such that detector frequency response is limited.
        - show_filtered_outputs [bool]: if true, show plots of X and Y magnitudes.
        - show_magnitudes [bool]: if true, show magnitude and phase plots.
        - show_coeffs_err [bool]: if true, show the error between lock-in coefficients and original intensity coefficients.

    Outputs:
        - N [float]: algebraic term used to calculate Psi and Delta
        - S [float]: algebraic term used to calculate Psi and Delta
        - C [float]: algebraic term used to calculate Psi and Delta
        - N_actual [float]: What N should be based on the known signal
        - S_actual [float]: What S should be based on the known signal
        - C_actual [float]: What C should be based on the known signal
    """

    ### PRE-AMPLIFIER AND LOCK-IN AMPLIFIER SETTINGS ###

    # Bessel functions calculated at desired retardation amplitude
    J_0 = jv(0, np.deg2rad(cfg.F))
    J_1 = jv(1, np.deg2rad(cfg.F))
    J_2 = jv(2, np.deg2rad(cfg.F))

    duration = cfg.t_meas  # Duration of lock-in measurement
    omega = cfg.omega  # Frequency of PEM

    # If sampling error is enabled, limit sampling rate
    if sampling_error:
        sample_rate = 60000
    else:
        sample_rate = 200000
    dt = 1 / sample_rate  # Time step
    t = np.arange(0, duration, dt)  # Array with timestamps
    order = 6  # Order for Butterworth filter
    cutoff_1f = 39.8  # Cutoff frequency for 1F lock-in amplifier
    cutoff_2f = 39.8  # Cutoff frequency for 2F lock-in amplifier
    I_dc = 1  # DC Intensity

    ### APPLYING LOCK-IN AMPLIFIER OPERATIONS ###
    # Lock-in amplifier reference signals (cosine and sine, 1F and 2F)
    ref_1f_cos = np.asarray(np.cos(omega * t))
    ref_1f_sin = np.asarray(np.sin(omega * t))
    ref_2f_cos = np.asarray(np.cos(2 * omega * t))
    ref_2f_sin = np.asarray(np.sin(2 * omega * t))

    # Dual-phase demodulation scheme. Multiply input signal by reference signal and by 90 deg phase-shifted reference signal
    I_mult_1f_cos = np.multiply(I, ref_1f_cos)
    I_mult_1f_sin = np.multiply(I, ref_1f_sin)
    I_mult_2f_cos = np.multiply(I, ref_2f_cos)
    I_mult_2f_sin = np.multiply(I, ref_2f_sin)

    # Low-pass filter definitions
    sos_1f = butter(
        order, cutoff_1f, fs=sample_rate, btype="low", analog=False, output="sos"
    )
    sos_2f = butter(
        order, cutoff_2f, fs=sample_rate, btype="low", analog=False, output="sos"
    )

    # Lock-in amplifier then applies a low-pass filter to the multiplied signal
    I_lp_1f_cos = sosfilt(sos_1f, I_mult_1f_cos)
    I_lp_1f_sin = sosfilt(sos_1f, I_mult_1f_sin)
    I_lp_2f_cos = sosfilt(sos_2f, I_mult_2f_cos)
    I_lp_2f_sin = sosfilt(sos_2f, I_mult_2f_sin)

    # Calculate magnitudes and phases from this dual-phase modulated scheme.
    magnitude_1f = np.sqrt(np.square(I_lp_1f_cos) + np.square(I_lp_1f_sin))
    magnitude_2f = np.sqrt(np.square(I_lp_2f_cos) + np.square(I_lp_2f_sin))
    phase_1f = np.arctan2(
        np.array(I_lp_1f_sin, dtype=float), np.array(I_lp_1f_cos, dtype=float)
    )
    phase_2f = np.arctan2(
        np.array(I_lp_2f_sin, dtype=float), np.array(I_lp_2f_cos, dtype=float)
    )

    # Show magnitude of X and Y lock-in outputs if true.
    if show_filtered_outputs:
        plt.rc("font", size=18)
        plt.figure()
        plt.grid()
        plt.plot(t * 1e6, I_lp_1f_cos, linewidth=2.0)
        plt.title("$\omega = 42$ kHz cosine filtering output")
        plt.xlabel("Time ($\mu$s)")
        plt.ylabel("$X$ Magnitude")
        plt.figure()
        plt.grid()
        plt.rc("font", size=18)
        plt.plot(t * 1e6, I_lp_1f_sin, linewidth=2.0)
        plt.title("$\omega = 42$ kHz sine filtering output")
        plt.xlabel("Time ($\mu$s)")
        plt.ylabel("$Y$ Magnitude")
        plt.figure()
        plt.grid()
        plt.rc("font", size=18)
        plt.plot(t * 1e6, I_lp_2f_cos, linewidth=2.0)
        plt.title("$2\omega = 84$ kHz cosine filtering output")
        plt.xlabel("Time ($\mu$s)")
        plt.ylabel("$X$ Magnitude")
        plt.figure()
        plt.grid()
        plt.rc("font", size=18)
        plt.plot(t * 1e6, I_lp_2f_sin, linewidth=2.0)
        plt.title("$2\omega = 84$ kHz sine filtering output")
        plt.xlabel("Time ($\mu$s)")
        plt.ylabel("$Y$ Magnitude")
        plt.tight_layout()
        plt.show()

    # Take the phase and magnitude and convert them to the 1F and 2F frequencies we need for psi and delta.
    I_1f = 2 * magnitude_1f[-1] * np.sign(expected_coeffs[1])
    I_2f = 2 * magnitude_2f[-1] * np.sign(expected_coeffs[2])

    # Show magnitude and phase plots if true
    if show_magnitudes:
        plt.subplot(1, 2, 1)
        plt.plot(magnitude_1f)
        plt.title("Magnitude (1f)")
        plt.xlabel("Time(s)")
        plt.ylabel("Output")
        plt.subplot(1, 2, 2)
        plt.plot(magnitude_2f)
        plt.title("Magnitude (2f)")
        plt.xlabel("Time(s)")
        plt.ylabel("Output")
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(phase_1f)
        plt.title("Phase (1f)")
        plt.xlabel("Time(s)")
        plt.ylabel("Output")
        plt.subplot(1, 2, 2)
        plt.plot(phase_2f)
        plt.title("Phase (2f)")
        plt.xlabel("Time(s)")
        plt.ylabel("Output")
        plt.tight_layout()
        plt.show()

    # Show error between expected coefficients and actual outputs from lock-in amplifier if true.
    if show_coeffs_err:
        expected_err_1 = np.abs(I_1f) - np.abs(expected_coeffs[1])
        expected_err_2 = np.abs(I_2f) - np.abs(expected_coeffs[2])
        print(f"Error (1f): {expected_err_1}")
        print(f"Error (2f): {expected_err_2}")

    ### PROCESSING LOCK-IN AMPLIFIER OUTPUTS ###
    # Solve for I_x and I_y
    I_x = ((1 / (2 * J_1)) * (I_1f / I_dc)) + (
        (delta_0 / (J_0 + (2 * J_2))) * (I_2f / I_dc)
    )
    I_y = ((1 / (J_0 + (2 * J_2))) * (I_2f / I_dc)) + (
        (delta_0 / (2 * J_1)) * (I_1f / I_dc)
    )

    # Calculate actual I_x and I_y values based on expected coefficients.
    I_x_actual = ((1 / (2 * J_1)) * (expected_coeffs[1] / expected_coeffs[0])) + (
        (delta_0 / (J_0 + (2 * J_2))) * (expected_coeffs[2] / expected_coeffs[0])
    )
    I_y_actual = (
        (1 / (J_0 + (2 * J_2))) * (expected_coeffs[2] / expected_coeffs[0])
    ) + (delta_0 / (2 * J_1)) * (expected_coeffs[1] / expected_coeffs[0])

    # Calculate actual N, S, and C values based on expected coefficients.
    if configuration == 1:
        N_actual = -I_y_actual / np.abs(np.sin(2 * A_angle))
        S_actual = I_x_actual / np.abs(np.sin(2 * A_angle))
        C_actual = 0
    elif configuration == 2:
        C_actual = -I_y_actual / np.abs(np.sin(2 * A_angle))
        S_actual = 0
        N_actual = 0

    # Calculate N, S, and C values based on lock-in amplifier outputs.
    if configuration == 1:
        # Only N and S are returned in configuration 1
        N = -I_y / np.abs(np.sin(2 * A_angle))
        S = I_x / np.abs(np.sin(2 * A_angle))
        C = 0
        return N, S, C, N_actual, S_actual, C_actual
    elif configuration == 2:
        # Only C is returned in configuration 2
        C = -I_y / np.abs(np.sin(2 * A_angle))
        S = 0
        N = 0
        return N, S, C, N_actual, S_actual, C_actual


def psi_delta(
    N,
    S,
    C,
    expected_nsc,
    config,
    delta_input,
    psi_input,
    print_outputs=False,
    show_nsc_error=False,
):
    """
    This function calculates Psi, Delta, Stokes parameters, DoLP, AoLP, and DoCP given N, S, and C.

    Inputs:
        - N, S, C [float]: algebraic parameters for calculating Psi and Delta
        - expected_nsc [arr]: used for comparison of real values with expected values based on input signal coefficients [expected_n, expected_s, expected_c]
        - config [int]: set to delta for delta calculation, and psi for psi calculation, based on ellipsometer configurations.
        - delta_input [float]: store the delta input for when "psi" is the configuration option
        - psi_input [float]: store the psi input for when "delta" is the configuration option
        - print_outputs [bool]: print Psi, Delta, Stokes parameters, DoLP, AoLP, and DoCP.
        - show_nsc_error [bool]: calculate error between expected NSC and actual NSC.

    Outputs:
        - psi [float]: Psi (rad.)
        - delta [float]: Delta (rad.)
        - stokes [arr]: Stokes parameters [S_1, S_2, S_3]

    """

    # Print the error between actual NSC and lock-in amplifier N, S, and C.
    if show_nsc_error:
        print(f"Error (N): {np.abs(expected_nsc[0]) - np.abs(N)}")
        print(f"Error (S): {np.abs(expected_nsc[1]) - np.abs(S)}")
        print(f"Error (C): {np.abs(expected_nsc[2]) - np.abs(C)}")

    # Normalizing N, S, and C to improve accuracy.
    r = N**2 + S**2 + C**2
    N = N / np.sqrt(r)
    S = S / np.sqrt(r)
    C = C / np.sqrt(r)

    # Delta and Psi calculation based on N, S, and C.
    if config == "delta":
        delta = np.arctan2(S, C)
        psi = psi_input
    elif config == "psi":
        psi = (1 / 2) * np.arccos(C)
        delta = delta_input

    # Calculating Stokes parameters from S_1, S_2, and S_3. Note that S_0 is unity in the Psi and Delta formulation.
    S_1 = -np.cos(2 * psi)
    S_2 = np.sin(2 * psi) * np.cos(delta)
    S_3 = -np.sin(2 * psi) * np.sin(delta)

    # Calculating DoLP, AoLP, and DoCP.
    DOLP = np.sqrt(S_1**2 + S_2**2)
    AOLP = (1 / 2) * np.arctan2(S_2, S_1)
    if S_1 > 0 and S_2 <= 0:
        AOLP += np.pi
    elif S_1 <= 0:
        AOLP += np.pi / 2
    DOCP = S_3

    # Print Psi, Delta, and Stokes parameters if true.
    if print_outputs:
        print(
            f"Psi: {np.round(np.rad2deg(psi), 2)}, Delta: {np.round(np.rad2deg(delta), 2)}"
        )
        print(
            f"S_1: {np.round(S_1, 3)}, S_2: {np.round(S_2, 3)}, S_3: {np.round(S_3, 3)}"
        )

    return np.abs(psi), np.abs(delta), [S_1, S_2, S_3]
