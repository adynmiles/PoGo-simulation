import numpy as np
import config as cfg
import matplotlib.pyplot as plt

'''
spectral_res_helper.py

Calculate the spectral resolution at a single inputted wavelength and return it.

'''

def spectral_resolution(wavelength):
    # Convert focal length and slit width to metres
    focal_length = cfg.focal_length * 1e-3
    slit_width = cfg.slit_width * 1e-3

    # Get line density for each grating and convert to lines/mm
    N_lines = []
    if wavelength in range(250, 900):
        N_lines.append(1200*1000)
    elif wavelength in range(900, 2300):
        N_lines.append(600*1000)
    elif wavelength in range(2300, 3000):
        N_lines.append(200*1000)
    elif wavelength in range(3000, 6000):
        N_lines.append(150*1000)

    # Convert wavelength to m
    wavelength = wavelength * 1e-9

    # Split equation into parts and calculate spectral resolution
    a = np.power(np.divide(1, (np.multiply(cfg.diffraction_order * focal_length, N_lines))), 2)
    b = np.power(np.divide(wavelength, (2 * focal_length * np.cos(np.deg2rad(cfg.aoi_grating)))), 2)
    term_1 = np.multiply(np.cos(np.deg2rad(cfg.aoi_grating)), np.sqrt(a - b))
    term_2 = np.tan(np.multiply(np.deg2rad(cfg.aoi_grating), wavelength / focal_length))
    spec_res = np.multiply(slit_width, term_1 + term_2)
    spec_res = spec_res * 1e9

    return spec_res