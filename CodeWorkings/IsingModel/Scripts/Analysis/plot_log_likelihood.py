# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:14:44 2021
@author: Mark
"""

# Import necessary libraries
import matplotlib.pyplot as plt
import sys
import numpy as np
import os.path

# Get the path to the Data generation folder
Data_generation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data generation'))

# Add the Data generation folder to sys.path
sys.path.append(Data_generation_path)
from get_loglik_data import generate_loglik_data

# Get the path to the Analysis folder
Analysis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Analysis'))

# Add the Analysis folder to sys.path
sys.path.append(Analysis_path)
from file_restructure import load_numpy_array

def generate_pastel_colors(num_colors, alpha=0.5):
    base_colors = [(0.8, 0.6, 0.6), (0.6, 0.8, 0.8), (0.6, 0.8, 0.6), (0.8, 0.6, 0.8), (0.8, 0.8, 0.6)]
    return [color + (alpha,) for color in base_colors[:num_colors]]

def plot_shaded_regions(alpha_list, step_size_list):
    pastel_colors = generate_pastel_colors(len(alpha_list))
    
    for i, (alpha, step_size) in enumerate(zip(alpha_list, step_size_list)):
        x_value = i * step_size
        plt.axvline(x=x_value, color='black', linestyle='dotted')

        # Get the pastel color by cycling through the pastel_colors using modulo
        color = pastel_colors[i % len(pastel_colors)]

        plt.fill_betweenx(
            np.linspace(1, -5, 100), x_value, (i + 1) * step_size,
            color=color, label=f'$\\alpha = {alpha}$'
        )


    # If you want to include a last dotted line without shading, uncomment the next line
    # plt.axvline(x=(len(alpha_list) - 1) * step_size, color='black', linestyle='dotted')

def generate_loglik_plot(topdir, data_params, data_date, folder_date, data_name):
    """
    Generate a plot of log likelihood vs. epoch number.

    Parameters:
    - topdir: Top-level directory path
    - data_params: Parameters describing the data
    - data_date: Date of the data
    - folder_date: Date of the folder
    - data_name: Name of the data

    Returns:
    - None
    """

    # File paths for raw data and log likelihood
    raw_data_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_name+".npy"
    loglik_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+"loglik.npy"

    # If the log likelihood file hasn't been generated yet
    if not os.path.isfile(loglik_filepath):
        print("Generating the log likelihood data")
        generate_loglik_data(folder_date, data_params, data_date, data_name)
    else:
        print("Using older data")

    # Load log likelihood data
    loglik_list = load_numpy_array(loglik_filepath)

    # Load run parameters
    run_parameters, _ = load_numpy_array(raw_data_filepath)
    step_size_list = run_parameters[0]

    # Create a list of step numbers based on epoch and step size
    step_number_list = [i for epoch_number in range(len(step_size_list)) for i in range(sum(step_size_list[:epoch_number]), sum(step_size_list[:epoch_number+1]))]

    # File paths for saving plots
    plots_folder_path = topdir+"Plots/"+folder_date+" "+data_params
    plot_file_path = plots_folder_path+"/"+"Loglik plot"

    # If a folder for the plots doesn't exist yet, create one
    os.makedirs(plots_folder_path, exist_ok=True)

    # Set plot file path and save plot as PDF
    plot_file_path = plots_folder_path+"/"+"Loglik plot"+".pdf"

    # Plotting log likelihood vs. epoch number
    plt.figure()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.scatter(step_number_list, loglik_list, s=2, marker='o')
    plt.title("Log likelihood vs epoch number", fontsize=30)
    plt.xlabel("Epoch number", fontsize=30)
    plt.ylabel(r'$\mathcal{L}\left(\theta |S \right)$', fontsize=30)

    # Plot shaded regions
    plot_shaded_regions(run_parameters[3], run_parameters[0])

    #plt.xlim(0, 12000)  # Set x-axis limits
    #plt.ylim(-4.5, -1)   # Set y-axis limits
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(plot_file_path)
    plt.show()
    return

# Example usage:
# topdir = ""  # "../../"
# data_params = "m 6 n 6b 200 [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200][1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001]"
# data_date = "2023-07-23"
# folder_date = "2023-07-23"
# data_name = "raw_data"
