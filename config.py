import numpy as np
import pandas as pd

"""
config.py

This file contains the known parameters of the ellipsometer system, to be used by all components
of the end-to-end simulator.
"""

### CONSTANTS ###
h = 6.63e-34  # Planck's constant
c = 3.00e8  # Speed of light
k_B = 1.381e-23  # Boltzmann constant
e = 1.602e-19  # Elementary charge

### LIGHT SOURCE (NEWPORT) ###
""" 
This section contains specifications for the three lamp options involved in the final tradeoff, 
which are the 100 W, 250 W, and 1000 W options.
"""

power = [
    250,
    1000,
    100,
]  # QTH lamp power (W).
f_condenser = [0.85, 1, 2]  # Condenser f-number
f_focuser = [3.9, 3.9, 3.9]  # Focuser f-number
T_lamp = [3400, 3200, 3300]  # Bulb temperature
housing_magnification = [1.6, 1.6, 1.0]  # Back reflector factor
housing_conversion = [0.13, 0.13, 0.06]  # Conversion factor
bulb_height = [7, 16, 4.2]  # Lamp bulb height
bulb_width = [3.5, 6, 2.3]  # Lamp bulb width

# Load in lamp emission values from Newport.
lamp_emission = np.array(pd.read_excel("efficiencies.xlsx", sheet_name="Lamp"))

# Apply irradiance conversion factors to lamp emission
lamp_emission[:, 1] = (
    housing_conversion[0] * housing_magnification[0] * (lamp_emission[:, 1])
)
lamp_emission[:, 2] = (
    housing_conversion[1] * housing_magnification[1] * (lamp_emission[:, 2])
)
lamp_emission[:, 3] = (
    housing_conversion[2] * housing_magnification[2] * (lamp_emission[:, 3])
)


### MONOCHROMATOR ###

# Load grating, mirror, and filter wheel efficiencies.
grating_eff = np.array(pd.read_excel("efficiencies.xlsx", sheet_name="Monochromator"))
mirror_eff = np.array(pd.read_excel("efficiencies.xlsx", sheet_name="Mirror"))
filter_eff = np.array(pd.read_excel("efficiencies.xlsx", sheet_name="Filter Wheel"))


slit_width = 0.694  # Slit width (mm)
slit_height = 5.5  # Slit height (mm)
magnification = np.divide(f_focuser, f_condenser)  # Condenser and focuser magnification
epsilon = np.divide(
    (slit_width * slit_height),
    (magnification * bulb_height * magnification * bulb_width),
)  # Vignetting at the monochromator slit entrance
vis_resolution = 5  # Spectral resolution limit in the visible regime (nm)
ir_resolution = 20  # Spectral resolution limit in the infrared regime (nm)

# Spectral resolution parameters
diffraction_order = 1  # Diffraction order used for the grating
aoi_grating = 14.74  # Angle of incidence for grating reflection
focal_length = 260  # Focal length of monochromator


### DETECTOR ###
VIS_length = 2.4  # Detector length for the Silicon detector (mm)
VIS_area = (VIS_length * 1e-3) ** 2  # Detector area for the visible detector (m^2)
IR_length = 0.7  # The detection surface for the Infrared detector (mm)
IR_area = (IR_length * 1e-3) ** 2  # Detector area for the infrared detector (m^2)
# IR_length_cm = IR_length / 10
detector_spacing = (
    1.5  # Vertical spacing between detectors of the two-color detector (mm)
)
t_int = 1  # Integration time (s)

# Load in photosensitivity and detectivity from Hamamatsu
detector_ps = np.array(
    pd.read_excel("efficiencies.xlsx", sheet_name="Detector Photosensitivity")
)
detector_d = np.array(
    pd.read_excel("efficiencies.xlsx", sheet_name="Detector Detectivity")
)

# Calculate quantum efficiency from photosensitivity
quant_eff_vis = detector_ps[:, 1] * (h * c) / (detector_ps[:, 0] * (1e-9) * e)

### NOISE PARAMETERS ###
well_depth = 19000  # Full well depth (e-)
dynamic_range = 14  # Dynamic range (bits)
dark_current_vis = 100  # Maximum dark current for the visible detector (nA)
dark_current_vis = dark_current_vis * 1e-9 / e  # Convert (nA) to (e-)
dark_current_IR = 83  # Maximum dark current for the infrared detector (nA)
dark_current_IR = dark_current_IR * 1e-9 / e  # Convert (nA) to (e-)
R_diode_VIS = 300  # Shunt resistance for the visible detector (MegaOhms)
R_diode_IR = 300  # Shunt resistance for the IR detector (kiloOhms)
T_detector = 298  # Detector absolute temperature (K)

# Parameter for the polarimetric accuracy simulation
nsr = 8.48e-5  # Noise to signal ratio (Average)

### PHOTOELASTIC MODULATOR ###

# Load transmission for PEM optical head
pem_eff = np.array(pd.read_excel("efficiencies.xlsx", sheet_name="PEM"))

F = 138  # Retardance amplitude (degrees)
J_1 = 0.52  # Bessel coefficient J_1
J_2 = 0.43  # Bessel coefficient J_2
omega = 42000  # Frequency of photoelastic modulator (Hz)
psi = np.deg2rad(20)  # Psi system input (degrees)
delta = np.deg2rad(160)  # Delta system input (degrees)
delta_0 = 1e-5  # Static retardation of PEM (rad.)
t_meas = 0.3  # Measurement time (s)

### WIRE-GRID POLARIZER ###
polarizer_diameter = 25.4  # Diameter of wire-grid polarizer (mm)

# Load the efficiency of the polarizer, multiply by 0.5 due to the polarizer reflecting
# light oriented perpendicularly to the transmission axis.
polar_eff = np.array(pd.read_excel("efficiencies.xlsx", sheet_name="Polarizer"))
polarizer_eff = np.column_stack((polar_eff[:, 0], 0.5 * polar_eff[:, 1] / 100))
analyzer_eff = np.column_stack(
    (
        polar_eff[:, 0],
        (0.5 * polar_eff[:, 1] / 100),
    )
)

# Load the extinction ratio
extinction_ratio = np.array(
    pd.read_excel("polarization_performance.xlsx", sheet_name="Thorlabs UB Polarizer")
)

# Parameters for polarimetric accuracy
mount_accuracy = np.deg2rad(0.14)  # Accuracy of rotation mount (deg)
delta_accuracy = 0.01

### LENSES ###
# lens_transmittance = 0.9  # Transmittance of CaF2 lens (average) (%)
d_sample_collimator = 25.4  # Diameter of collimator after the sample.
d_fibre_collimator = 25.4  # Diameter of illumination collimator
d_detector_focuser = 25.4  # Diameter of focuser before the two-colour detector.

# Load transmission of CaF2 and MgF2 lenses
caf2_eff = np.array(pd.read_excel("efficiencies.xlsx", sheet_name="CaF2")) / 100
mgf2_eff = np.array(pd.read_excel("efficiencies.xlsx", sheet_name="MgF2")) / 100
caf2_eff[:, 0] = caf2_eff[:, 0] * 100
mgf2_eff[:, 0] = mgf2_eff[:, 0] * 100

### SAMPLE ###
sample_eff = 0.1  # Minimum reflectivity requirement for sample.

### OPTICAL FIBRES ###
fibre_eff = np.array(
    pd.read_excel("efficiencies.xlsx", sheet_name="Optical Fibre")
)  # Load optical fibre transmission
