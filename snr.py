import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from spectral_res_helper import spectral_resolution

"""
snr.py

Design trade for angular resolution versus signal to noise at detector.

The angular resolution refers to the solid angle that is able to be detected
by the observation arm of the ellipsometer. This will determine the position
of the collimator. However, the solid angle of this cone will determine the
amount of light collected that goes through the observation arm and reaches
the detector.

"""

### VISIBILITY CONFIGURATIONS ###
source_plots = False  # Plots source emission.
snr_plots = False  # Plots SNR vs. wavelength for a given lamp or for all lamps.
tradeoff_plots = False  # Plots angular resolution vs. SNR.
detailed_snr_plots = True  # Plots SNR vs. wavelength for one lamp with spectral band labels.
transmissivity_plots = False  # Plots transmissivity of the instrument vs. wavelength.

### ARRAY SETUP ###
lamps = [250, 1000, 100]  # Power in W for all lamps to be simulated. Write in the order [250, 1000, 100] if all are to be used.

# Collimator focal length
if tradeoff_plots:
    f_collimator = np.arange(
        100, 500, 10
    )  # Iterate through collimators for the tradeoff plot
else:
    f_collimator = [180]  # Ideal collimator length.

wave_vis = np.arange(300, 1000, 5)  # Visible wavelengths
wave_ir = np.arange(1000, 4000, 20)  # Infrared wavelengths
wavelength = np.concatenate((wave_vis, wave_ir))  # Concatenate wavelengths
VIS_cutoff = np.where(wavelength == 1100)[0][0]  # Measure the Silicon detector cutoff

### SPECTRAL RESOLUTION, DARK CURRENT, SHUNT RESISTANCE ###
spectral_res = []
dark_current = []
R_diode = []
for wave_num in range(0, len(wavelength)):
    if wave_num < VIS_cutoff:
        spectral_res.append(spectral_resolution(wavelength[wave_num])[0])
        dark_current.append(cfg.dark_current_vis)
        R_diode.append(cfg.R_diode_VIS)
    else:
        spectral_res.append(spectral_resolution(wavelength[wave_num])[0])
        dark_current.append(cfg.dark_current_IR)
        R_diode.append(cfg.R_diode_IR)
spectral_res = np.asarray(spectral_res)
dark_current = np.asarray(dark_current)
R_diode = np.asarray(R_diode)

# Linearly interpolating all transmission data to have the same wavelength intervals.
lamp_emission_1000 = np.interp(
    wavelength, cfg.lamp_emission[:, 0], cfg.lamp_emission[:, 1]
)
lamp_emission_250 = np.interp(
    wavelength, cfg.lamp_emission[:, 0], cfg.lamp_emission[:, 2]
)
lamp_emission_100 = np.interp(
    wavelength, cfg.lamp_emission[:, 0], cfg.lamp_emission[:, 3]
)
grating_eff = np.interp(wavelength, cfg.grating_eff[:, 0], cfg.grating_eff[:, 1])
mirror_eff = np.interp(wavelength, cfg.mirror_eff[:, 0], cfg.mirror_eff[:, 1])
pem_eff = np.interp(wavelength, cfg.pem_eff[:, 0], cfg.pem_eff[:, 1])
analyzer_eff = np.interp(wavelength, cfg.analyzer_eff[:, 0], cfg.analyzer_eff[:, 1])
polarizer_eff = np.interp(wavelength, cfg.polarizer_eff[:, 0], cfg.polarizer_eff[:, 1])
caf2_eff = np.interp(wavelength, cfg.caf2_eff[:, 0], cfg.caf2_eff[:, 1])
mgf2_eff = np.interp(wavelength, cfg.mgf2_eff[:, 0], cfg.mgf2_eff[:, 1])
detector_d = np.interp(
    wavelength[VIS_cutoff:], cfg.detector_d[:, 0], cfg.detector_d[:, 1]
)
fibre_eff = np.interp(wavelength, cfg.fibre_eff[:, 0], cfg.fibre_eff[:, 1])
sample_eff = cfg.sample_eff
filter_eff = np.interp(wavelength, cfg.filter_eff[:, 0], cfg.filter_eff[:, 1])

average_SNR = [[], [], []]
ang_res = [[], [], []]
legend_list = []

# Iterating through the number of lamps and the collimator distances if those options are selected
for lamp in range(0, len(lamps)):
    for f_num in range(0, len(f_collimator)):

        # Angular resolution and system f-number calculation
        # System f-number is simply the f-number of the condenser.
        # Angular resolution is the arctan of the collimator diameter divided by the focal length.
        angular_resolution = np.arctan(
            cfg.d_sample_collimator / (f_collimator[f_num])
        ) * (180 / np.pi)
        f_sys = cfg.f_condenser[lamp]

        source_emission = []

        ### IRRADIANCE FROM LIGHT SOURCE ###
        for wave_num, wave in enumerate(wavelength):
            # If the wavelength is less than 2400 nm, the irradiance values from Newport can be used.
            if wave < 2400:
                if lamps[lamp] == 1000:
                    source_emission.append(
                        lamp_emission_1000[wave_num] * 1e-3 * 1e9
                    )  # Conversion from mW/nm to W/m
                elif lamps[lamp] == 250:
                    source_emission.append(
                        lamp_emission_250[wave_num] * 1e-3 * 1e9
                    )  # Conversion from mW/nm to W/m
                elif lamps[lamp] == 100:
                    source_emission.append(lamp_emission_100[wave_num] * 1e-3 * 1e9)
            # If the wavelength is greater than 2400 nm, Planck's law is used instead.
            else:
                thermal_emission = (
                    4
                    * np.pi
                    * np.pi # Integrating over the solid angle
                    * (
                        (1e9) # Conversion to nm
                        * (cfg.bulb_width[lamp] * 1e-3 / (0.5)) ** 2 # Correct for distance from filament
                        * ((1e-3) * cfg.housing_conversion[lamp]) # Multiply by housing conversion factor
                        * cfg.housing_magnification[lamp]   # Back-reflector factor
                    )
                    * (8 * np.pi * cfg.h * cfg.c) # Planck's law
                    / ((wave * (1e-9)) ** 5)
                    * (1 / (np.exp((cfg.h * cfg.c) / (wave * (1e-9) * cfg.k_B * cfg.T_lamp[lamp]))-1))
                )
                source_emission.append(thermal_emission)

        # Quantum efficiency calculation
        quant_eff_IR = (
            (detector_d * 1e-2)
            * np.sqrt(cfg.IR_area)
            * np.sqrt(
                (cfg.c / (wavelength[VIS_cutoff] * 1e-9))
                - (cfg.c / (wavelength[-1] * 1e-9))
            )
            * (cfg.h * cfg.c * (cfg.dark_current_IR * cfg.e))
            / (cfg.e * (wavelength[VIS_cutoff:] * (1e-9)))
        )
        quant_eff_IR = (
            (quant_eff_IR - quant_eff_IR.min())
            / (quant_eff_IR.max() - quant_eff_IR.min())
            * 0.45
        ) + 0.01  # Scaling QE to fit the boundaries of the visible detector.
        quant_eff_vis = np.interp(
            wavelength[:VIS_cutoff], cfg.detector_ps[:, 0], cfg.detector_ps[:, 1]
        )
        quant_eff = np.concatenate((quant_eff_vis, quant_eff_IR))

        ### SIGNAL CALCULATION ###
        signal_const = ((cfg.epsilon[lamp]) * cfg.t_int) / (
            f_sys**2 * cfg.h * cfg.c
        )  # Part of signal calculation that does not depend on wavelength.
        reflection_eff = (
            (np.pi * (cfg.d_sample_collimator / 2) ** 2) / (f_collimator[f_num]) ** 2
        ) / (
            2 * np.pi
        )  # Light losses due to reflection off the sample.
        transmittance = (
            grating_eff
            * mirror_eff**5
            * polarizer_eff**2
            * pem_eff
            * caf2_eff**4
            * mgf2_eff**8
            * quant_eff
            * fibre_eff
            * sample_eff
            * reflection_eff
            * filter_eff
        )  # Total transmittance of the system

        # Transmittance plotting calculations
        if transmissivity_plots:
            trans_plot = transmittance * (
                spectral_res / 4200
            )  # Transmittance for plotting purposes which includes transfer from white to monochromatic light

            # Calculating the transmittance of the instrument at 3000 nm.
            lambda_3000 = np.where(wavelength == 3000)[0][0]
            t_1 = (
                cfg.epsilon[lamp]
                * caf2_eff[lambda_3000] ** 2
                * mgf2_eff[lambda_3000] ** 4
                * 1.6
            )  # After focuser and condenser
            t_2 = (
                t_1
                * grating_eff[lambda_3000]
                * mirror_eff[lambda_3000] ** 3
                * filter_eff[lambda_3000]
                * (20 / 4200)
            )  # After monochromator exit
            t_3 = (
                t_2
                * mirror_eff[lambda_3000]
                * fibre_eff[lambda_3000]
                * mirror_eff[lambda_3000]
            )  # After illumination collimator
            t_4 = t_3 * polarizer_eff[lambda_3000]  # After polarizer
            t_5 = t_4 * pem_eff[lambda_3000]  # After PEM
            t_6 = (
                t_5
                * sample_eff
                * reflection_eff
                * caf2_eff[lambda_3000]
                * mgf2_eff[lambda_3000] ** 2
            )  # After observation collimator
            t_7 = t_6 * polarizer_eff[lambda_3000]  # After analyzer
            t_8 = (
                t_7
                * caf2_eff[lambda_3000]
                * mgf2_eff[lambda_3000] ** 2
                * quant_eff[lambda_3000]
            )  # At detector

        signal_int = (
            source_emission
            * transmittance
            * (wavelength * 1e-9)
            * (spectral_res * 1e-9)
        )  # Part of signal calculation that does depend on wavelength.
        signal = signal_const * signal_int  # Total signal

        ### NOISE CALCULATION ###
        shot_noise = np.sqrt(signal)
        dark_noise = np.sqrt(dark_current * cfg.t_int)
        johnson_noise = (1 / cfg.e) * np.sqrt(
            (2 * cfg.k_B * cfg.T_detector * cfg.t_int) / (R_diode * 1e6)
        )

        noise = np.sqrt((shot_noise**2) + (dark_noise**2) + (johnson_noise**2))

        ### SNR CALCULATION ###
        SNR = signal / noise
        SNR_dB = 10 * np.log10(SNR, where=(SNR != 0))

        # Printing results
        print(f"Average SNR: {np.mean(SNR)}")
        print(f"Average SNR (dB): {np.mean(SNR_dB)}")

        average_SNR[lamp].append(np.mean(SNR_dB))

        print(f"Angular Resolution: {angular_resolution} deg")

        ang_res[lamp].append(angular_resolution)

    # Plot source power in W/nm if selected
    if source_plots:
        fig = plt.figure
        plt.rc("font", size=18)
        plt.grid()
        plt.plot(wavelength, np.asarray(source_emission) * (1e-9), linewidth=4.0)
        plt.title("Source Emission (250 W Lamp)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Source Power (W/nm)")
        # plt.legend(['100 W Lamp', '250 W Lamp', '1000 W Lamp'])
        plt.show()

    # Plot transmittance over the spectral range if selected
    if transmissivity_plots:
        fig = plt.figure
        plt.rc("font", size=18)
        plt.grid()
        plt.plot(wavelength, 1.6 * trans_plot * cfg.epsilon[lamp], linewidth=4.0)
        plt.title("Transmittance of Optical System")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmittance")
        plt.show()

    # Plot SNR of all lamps on the same plot if selected
    if snr_plots:
        # Plotting for signal to noise ratio
        plt.rc("font", size=18)
        fig = plt.figure
        plt.grid()
        plt.plot(wavelength, SNR_dB, linewidth=4.0)
        plt.title("Lamp SNR Comparison")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Signal to Noise Ratio (dB)")
        if lamps[lamp] == 250:
            legend_list.append("250 W Lamp")
        elif lamps[lamp] == 100:
            legend_list.append("100 W Lamp")
        elif lamps[lamp] == 1000:
            legend_list.append("1000 W Lamp")
        plt.legend(legend_list)

    # Plot SNR of a single lamp with labelled spectral bands if selected.
    if detailed_snr_plots:
        cn_lower_bound = np.where(wavelength == 350)[0][0]
        cn_upper_bound = np.where(wavelength == 620)[0][0]
        oli_lower_bound = np.where(wavelength == 1040)[0][0]
        oli_upper_bound = np.where(wavelength == 1060)[0][0]
        pyr_lower_bound = np.where(wavelength == 375)[0][0]
        pyr_upper_bound = np.where(wavelength == 750)[0][0]
        h2o_lower_bound = np.where(wavelength == 1900)[0][0]
        h2o_upper_bound = np.where(wavelength == 2960)[0][0]
        oh_lower_bound = np.where(wavelength == 2700)[0][0]
        oh_upper_bound = np.where(wavelength == 3000)[0][0]
        org_band = np.where(wavelength == 3400)[0][0]
        snr_cn = np.mean(SNR_dB[cn_lower_bound:cn_upper_bound])
        snr_pyr = np.mean(SNR_dB[pyr_lower_bound:pyr_upper_bound])
        snr_oli = np.mean(SNR_dB[oli_lower_bound:oli_upper_bound])
        snr_h2o = np.mean(SNR_dB[h2o_lower_bound:h2o_upper_bound])
        snr_oh = np.mean(SNR_dB[oh_lower_bound:oh_upper_bound])
        snr_org = np.mean(SNR_dB[org_band])
        fig = plt.figure
        plt.rc("font", size=18)
        plt.grid()
        plt.plot(wavelength, SNR_dB, linewidth=4.0, color="black")
        plt.title("Signal to Noise Ratio of Ellipsometer")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Signal to Noise Ratio (dB)")
        plt.axvline(
            x=350, color="purple", linewidth=2, linestyle="--", label="_nolegend_"
        )
        plt.axvline(x=620, color="purple", linewidth=2, linestyle="--")
        plt.axvline(
            x=375, color="blue", linewidth=2, linestyle="--", label="_nolegend_"
        )
        plt.axvline(x=750, color="blue", linewidth=2, linestyle="--")
        plt.axvline(x=1050, color="green", linewidth=2, linestyle="--")
        plt.axvline(
            x=1450, color="gold", linewidth=2, linestyle="--", label="_nolegend_"
        )
        plt.axvline(
            x=1900, color="gold", linewidth=2, linestyle="--", label="_nolegend_"
        )
        plt.axvline(x=2950, color="gold", linewidth=2, linestyle="--")
        plt.axvline(
            x=2700, color="darkorange", linewidth=2, linestyle="--", label="_nolegend_"
        )
        plt.axvline(x=3000, color="darkorange", linewidth=2, linestyle="--")
        plt.axvline(x=3400, color="red", linewidth=2, linestyle="--")
        plt.legend(
            [
                f"SNR, Average: {np.round(np.mean(SNR_dB), 1)} dB",
                f"$CN$ and $C_2$, Average: {np.round(snr_cn, 1)} dB",
                f"Pyroxenes, Average: {np.round(snr_pyr, 1)} dB",
                f"Olivines, Average: {np.round(snr_oli, 1)} dB",
                f"Water Ice, Average: {np.round(snr_h2o, 1)} dB",
                f"Hydroxides, Average: {np.round(snr_oh, 1)} dB",
                f"Organics, Average: {np.round(snr_org, 1)} dB",
            ],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        plt.xlim([250, 4150])
        # plt.tight_layout() # Uncomment this if you want to see the legend
        plt.show()

plt.show()

# Plot angular resolution vs average signal to noise ratio if selected
if tradeoff_plots:
    fig, ax = plt.subplots(1, 1)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.plot(ang_res[0], average_SNR[0], linewidth=4.0)
    ax.set_title("Average SNR vs Angular Resolution", fontsize=22)
    ax.set_xlabel("Angular Resolution (°)", fontsize=18)
    ax.set_ylabel("Average SNR (dB)", fontsize=18)
    ax.grid()
    plt.show()

print("SNR Analysis Complete")
