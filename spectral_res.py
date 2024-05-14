import numpy as np
import config as cfg
import matplotlib.pyplot as plt

'''
spectral_res.py

Script for calculating the spectral resolution of the ellipsometer system over the entire spectral range,
and returns information about maximum and minimum spectral range for each grating.

'''


# Define wavelengths and spectral resolution requirements.
wavelength = np.arange(300, 4500, 1)
VIS_cutoff = np.where(wavelength == 1100)[0][0]
spec_res_VIS = [5]*VIS_cutoff
spec_res_IR = [20]*(len(wavelength) - VIS_cutoff)
spec_res_requirement = np.concatenate((spec_res_VIS, spec_res_IR))

# Convert focal length and slit width to metres
focal_length = cfg.focal_length * 1e-3
slit_width = cfg.slit_width * 1e-3

# Get line density for each grating, convert to lines/m
N_lines = []
for wave in wavelength:
    if wave in range(250, 900):
        N_lines.append(1200*1000)
    elif wave in range(900, 2300):
        N_lines.append(600*1000)
    elif wave in range(2300, 3000):
        N_lines.append(200*1000)
    elif wave in range(3000, 6000):
        N_lines.append(150*1000)

# Convert wavelength to metres
wavelength_m = wavelength * 1e-9

# Breaking spectral resolution equation into sections for clearer calculations
a = np.power(np.divide(1, (np.multiply(cfg.diffraction_order * focal_length, N_lines))), 2)
b = np.power(np.divide(wavelength_m, (2 * focal_length * np.cos(np.deg2rad(cfg.aoi_grating)))), 2)
term_1 = np.multiply(np.cos(np.deg2rad(cfg.aoi_grating)), np.sqrt(a - b))
term_2 = np.tan(np.multiply(np.deg2rad(cfg.aoi_grating), wavelength_m / focal_length))
spec_res = np.multiply(slit_width, term_1 + term_2)
spec_res = spec_res * 1e9

# Get the index for each of the grating boundaries
lambda_900 = np.where(wavelength == 900)[0][0]
lambda_2300 = np.where(wavelength == 2300)[0][0]
lambda_3000 = np.where(wavelength == 3000)[0][0]

# Get the minimum and maximum spectral resolution for each grating
grating_1 = [np.min(spec_res[:lambda_900]), np.max(spec_res[:lambda_900])]
grating_2 = [np.min(spec_res[lambda_900:lambda_2300]), np.max(spec_res[lambda_900:lambda_2300])]
grating_3 = [np.min(spec_res[lambda_2300:lambda_3000]), np.max(spec_res[lambda_2300:lambda_3000])]
grating_4 = [np.min(spec_res[lambda_3000:-1]), np.max(spec_res[lambda_3000:-1])]

# Print minimum and maximum spectral resolution
print(f"Spectral Resolution: [Minimum, Maximum]")
print(f"Grating 1: {grating_1}")
print(f"Grating 2: {grating_2}")
print(f"Grating 3: {grating_3}")
print(f"Grating 4: {grating_4}")

# Plot the spectral resolution compared to the required spectral resolution
fig = plt.figure
plt.rc('font', size=18)
plt.plot(wavelength * 1e9, spec_res, linewidth=2.0, color='b')
plt.plot(wavelength * 1e9, spec_res_requirement, linewidth=2.0, color='r')
plt.title("Spectral Resolution of Monochromator", fontsize=22)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral Resolution (nm)")
plt.legend(["Spectral Resolution", "Requirement"])
plt.grid()
plt.show()