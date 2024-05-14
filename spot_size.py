import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
spot_size.py

Calculate the size of the intersection between a cylinder and a plane at an angle from the cylinder.

"""

### PLOT OPTIONS ###
axis_plots = True # Show how spot diameter increases with goniometer angle (Illumination)
area_plots = True # Show how spot area increases with goniometer angle (Illumination)
spot_size_plots = True  # Show spots at 10 degree increments (Illumination/Observation)
obs_plots = True # Show observation spots at 10 degree increments inscribed within illumination spot

r_ill = 12.7                                        # Radius of illumination spot size at nominal position (nadir) (mm)
r_obs = 2.2                                         # Radius of observation spot size at nominal position (nadir) (mm). Calculated using 12.7 * cos(80 deg)
if axis_plots or area_plots or spot_size_plots:
    arms = [r_ill]                                   
elif obs_plots:
    arms = [r_ill, r_obs]                           # Add r_obs to also look at observation arm behaviour
parameter_range = np.arange(0, 360, 1)              # Parameter for intersection equation (t)

# Setup goniometer arm positions
if axis_plots or area_plots:
    arm_angles = np.arange(0, 81, 1)                # 1 degree increment for diameter and area plots    

if spot_size_plots or obs_plots:
    arm_angles = np.arange(0, 81, 10)               # 10 degree increment for spot size plots
    fig, ax = plt.subplots()
    ax.axis([-50, 50, -50, 50])
    ax.set_xlabel('x (mm)', fontsize=18)
    ax.set_ylabel('y (mm)', fontsize=18)
    ax.set_title(f"Illumination spot at various goniometer angles", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)

# Setting up arrays for population, plot parameters.
major_axis = np.zeros((2, len(arm_angles)))
minor_axis = np.zeros((2, len(arm_angles)))
areas = np.zeros((2, len(arm_angles)))
legend_names = []
intersection = np.zeros((len(arms), len(arm_angles), 3, len(parameter_range)))
colors = ['blue', 'red', 'black', 'magenta', 'green', 'cyan', 'yellow', 'purple', 'darkgreen']

for angle, theta in enumerate(arm_angles):
    # Sweep out intersection coordinates as a function of the t parameter.
    for t in parameter_range:
        for arm, r in enumerate(arms):
            intersection[arm, angle, 0, t] = r*np.cos(np.deg2rad(t))
            intersection[arm, angle, 1, t] = r*np.sin(np.deg2rad(t))
            intersection[arm, angle, 2, t] = -r*np.tan(np.deg2rad(theta))*np.cos(np.deg2rad(t))

    # Calculate the major axis and minor axis from these intersection ellipses, as well as the area
    for arm, r in enumerate(arms):
        # Calculating distance in 3D space, sqrt(x_distance^2 + y_distance^2 + z_distance^2)
        major = np.sqrt((intersection[arm, angle, 0, 0] - intersection[arm, angle, 0, 180])**2 + (intersection[arm, angle, 1, 0] - intersection[arm, angle, 1, 180])**2 + (intersection[arm, angle, 2, 0] - intersection[arm, angle, 2, 180])**2)
        minor = np.sqrt((intersection[arm, angle, 0, 90] - intersection[arm, angle, 0, 270])**2 + (intersection[arm, angle, 1, 90] - intersection[arm, angle, 1, 270])**2 + (intersection[arm, angle, 2, 90] - intersection[arm, angle, 2, 270])**2)
        area = np.pi * (0.5*major) * (0.5*minor)
        major_axis[arm, angle] = major
        minor_axis[arm, angle] = minor
        areas[arm, angle] = area

        if spot_size_plots == True:
            # Plot the illumination ellipse
            spot = Ellipse((0, 0), major, minor, angle=0, fill=False, edgecolor=colors[angle])
            ax.add_patch(spot)

            # Set up legend depending on if illumination or observation ellipses are being measured.
            if arm == 0:
                arm_name = 'Illumination'
            elif arm == 1:
                arm_name = 'Observation'
            legend_names.append(f"Arm: {arm_name}, Angle: {theta}°")

if spot_size_plots == True:
    # Format spot size plot
    ax.set_xlim([-80, 80])
    ax.set_ylim([-80, 80])
    ax.set_aspect('equal', adjustable='box')
    ax.legend(legend_names, fontsize=12, bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid()
    plt.show()

# Shows largest observation spot and smallest illumination spot shaded in
if obs_plots == True:
    obs_spot = Ellipse((0, 0), major_axis[1, -1], minor_axis[1, -1], angle=0, facecolor='deepskyblue', edgecolor='navy', alpha=0.5)
    ill_spot = Ellipse((0, 0), major_axis[0, 0], minor_axis[0, 0], angle=0, facecolor='gold', edgecolor='darkgoldenrod', alpha=0.5)
    ax.add_patch(ill_spot)
    ax.add_patch(obs_spot)
    legend_names = [f'Arm: Illumination, Angle: {arm_angles[0]}°', f'Arm: Observation, Angle: {arm_angles[-1]}°']
    ax.set_aspect('equal', adjustable='box')
    ax.legend(legend_names, fontsize=16)
    ax.grid()
    plt.show()

# Shows diameter as a function of goniometer arm angle
if axis_plots == True:
    fig, ax = plt.subplots(1, 1)
    plt.rc('font', size=18)
    ax.plot(arm_angles, major_axis[0, :], linewidth=3.0)
    ax.plot(arm_angles, major_axis[1, :], linewidth=3.0)
    ax.set_title("Spot diameter as a function of goniometer arm angle", fontsize=22)
    ax.set_xlabel("Theta (°)", fontsize=18)
    ax.set_ylabel("Axis diameter (mm)", fontsize=18)
    ax.legend(["Illumination", "Observation"], fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid()
    plt.show()

# Shows area as a function of goniometer arm angle
if area_plots == True:
    fig, ax = plt.subplots(1, 1)
    ax.plot(arm_angles, areas[0, :], linewidth=3.0)
    ax.plot(arm_angles, areas[1, :], linewidth=3.0)
    ax.set_title("Spot area as a function of goniometer arm angle", fontsize=22)
    ax.set_xlabel("Theta (°)", fontsize=18)
    ax.set_ylabel("Area ($\mathrm{mm}^2$)", fontsize=18)
    ax.legend(["Illumination", "Observation"], fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid()
    plt.show()

print("Spot size analysis complete.")
