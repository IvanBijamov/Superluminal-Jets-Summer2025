To run this code on your computer, ensure that you have Python installed on your computer.

-- PyMC_PyTensor_noise_v_main_varsig_aniso.py
This is the current main project file for the Markov-chain Monte Carlo code.
It is an anisotropic version of the model, that is configured to accept differing
 sigma values for each input jet velocity.

-- PyMC_PyTensor_noise_v_main_varsig.py
This is the main file for the isotropic version of the model. It is configured
 to accept differing sigma values for each input jet velocity.

-- simulationImport.py
This code includes a function that imports the observed velocity and uncertainty
 for both the simulated data set (filetype = "Schindler") and the MOJAVE
 data set (filetype = "Mojave").

-- aniso_simulated_data_gen_v.py (previously known as csv_file_imp_v.py)
This code generates our simulated data set with a configurable B0 and B_vec parameter

-- Mojave_html_to_csv.py
This code takes the MOJAVE dataset from the website linked, and converts it into a csv file that can be
 read by simulationImport.py, when configured with filetype = "Mojave".