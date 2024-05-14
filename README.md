# PoGosimulation
 End-to-end simulation for spectropolarimetric goniometer (PoGo).

This set of scripts covers the signal-to-noise ratio, spectral resolution,
spot size, and polarimetric accuracy simulations for the instrument. A brief
overview of each file is provided here, please see documentation included 
with each script for information about inputs, outputs, and how to operate
the scripts.

***GENERAL***
- `config.py`: Configuration file, used by all other files. Inputs can be
               reconfigured if design properties change.

***SIGNAL-TO-NOISE RATIO***
- `snr.py`: This file calculates the SNR of the instrument across its spectral 
            range under different lamp configurations and collimator distances.

***SPECTRAL RESOLUTION***
- `spectral_res.py`: This file calculates the spectral resolution across the 
                     entire spectral range, and returns the minimum and maximum
                     for each grating as well as a plot to compare performance
                     to requirements.

- `spectral_res_helper.py`: This file is meant for use with the `snr.py` file. 
                            It calculates the spectral resolution at a single
                            wavelength. This file needs to be called from the
                            `snr.py` file, it cannot be called as the main.

***SPOT SIZE ANALYSIS***
- `spot_size.py`: Calculates intersection between illumination/observation beams
                  and the sample, and draws these to show how spot size increases
                  with goniometer angle.

***POLARIMETRIC ACCURACY***
- `polarimetry.py`: Sets up the signal intensity to input into the detector. This
                    file needs to be called from `monte_carlo.py`, `data_vis.py`, 
                    or `stackup.py`, it cannot be called as the main.

- `psi_delta.py`: Applies lock-in amplifier operations to input signal, calculates
                  Delta, Psi, Stokes parameters, DoLP, AoLP, and DoCP. This file 
                  needs to be called from `monte_carlo.py`, `data_vis.py`, or 
                  `stackup.py`, it cannot be called as the main.

- `monte_carlo.py`: Runs a simulation that applies lock-in amplifier operations to
                    both polarimetric configurations to solve for Delta and Psi. 
                    Used for a single error configuration and a single input Delta
                    and Psi combination. Can be set to multiple iterations if a 
                    Monte Carlo simulation is desired, otherwise it provides a
                    worst-case scenario.

- `data_vis.py`: Runs the `monte_carlo.py` script over a range of Delta and Psi
                 inputs, and displays a visualization of the Delta/Psi inputs and 
                 outputs as well as the error between the inputs and outputs. Can be
                 configured to run a Monte Carlo simulation or a worst-case scenario
                 analysis.

- `stackup.py`: Runs the `data_vis.py` script over a set of error inputs, and 
                displays a visualization of how these errors are stacked. Can be 
                configured to run a Monte Carlo simulation or a worst-case scenario 
                analysis, but only displays the range of Monte Carlo data, not each
                individual data point.