# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:06:49 2021
@author: Mark Thomas
"""

# External imports
import matplotlib.pyplot as plt
import sys
import os.path

# Get the path to the Data generation folder
Data_generation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data generation'))

# Add the Data generation folder to sys.path
sys.path.append(Data_generation_path)
from get_loglik_data import generate_loglik_data
from file_restructure import save_numpy_array, load_numpy_array, make_directory

# Get the path to the Implementation folder
implementation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Implementation'))

# Add the Implementation folder to sys.path
sys.path.append(implementation_path)

from all_vectors import all_vectors_ising
from model_and_data_averages import prob_v_given_theta, partition_function
from Learned_distribution import data_distribution


def generate_config_plot(topdir, data_params, data_date, folder_date, data_name, repeats):
    """
    Generate and save a plot of the learned distribution.

    Parameters:
    - topdir: Top-level directory path
    - data_params: Parameters describing the data
    - data_date: Date of the data
    - folder_date: Date of the folder
    - data_name: Name of the data
    - repeats: Repeat factor for plotting

    Returns:
    - None
    """

    # File paths for raw data and distribution
    raw_data_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_name+".npy"
    dist_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_date+" "+"dist.npy"

    # Load run parameters and distribution data
    run_parameters, distribution_data = load_numpy_array(raw_data_filepath)
    
    # Extract relevant information
    step_size_list = run_parameters[0]
    theta_history_list = distribution_data[1]
    m_visible = run_parameters[1]
    n_hidden = run_parameters[2]
    alpha_list = run_parameters[3]
    
    # Generate all possible visible vectors for Ising model
    allv = all_vectors_ising(m_visible)

    # If the distribution file hasn't been generated yet
    if not os.path.isfile(dist_filepath):
        allh = all_vectors_ising(n_hidden)
        
        current_step = 0
        relevant_theta_list = []
        pvgt_list = []
        
        # Loop through epochs and generate distribution data
        for epoch_number in range(len(step_size_list)):
            current_step += step_size_list[epoch_number]
            current_theta = theta_history_list[current_step-1]
            relevant_theta_list.append(current_theta)
            
            Z = partition_function(allv, allh, current_theta)
            pvgt = []
            
            # Loop through visible vectors and compute probabilities
            for v in allv:
                pvgt.append(prob_v_given_theta(allv, allh, v, current_theta, Z))
            
            pvgt_list.append(pvgt)
    
        # Save the distribution data
        dist_name = "dist"
        file_to_save = data_date + " " + dist_name
        folder_path = "Data/"+folder_date+" "+data_params+"/"
        dist_name = file_to_save
        data_to_save = [pvgt_list, relevant_theta_list]
        data_to_save = pvgt_list
        save_numpy_array(dist_name, data_to_save, folder_path)
    
    else:
        print("Using older data")
    
    # Load distribution data
    pvgt_list = load_numpy_array(dist_filepath)
    config_number_list = [i for i in range(len(allv))]
        
    # File paths for saving plots
    plots_folder_path = topdir+"Plots/"+folder_date+" "+data_params
    plot_file_path = plots_folder_path+"/"+"Learned distribution"+".pdf"
    
    # If a folder for the plots doesn't exist yet, create one
    os.makedirs(plots_folder_path, exist_ok=True)  

    # Generate the correct distribution for comparison
    correct_distribution = data_distribution(m_visible, 1)

    # Plotting the learned distribution
    plt.figure()
    plt.title("Learned distribution", fontsize=30)
    plt.xlabel("Config number", fontsize=30)
    plt.ylabel(r'$p\left(v|\theta \right)$', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    # Plot each distribution with different alpha values
    for i in range(len(pvgt_list)):
        if i % repeats == 0:
            plt.plot(config_number_list, pvgt_list[i], label=r'$\alpha$ = '+str(alpha_list[i]))
        else:
            plt.plot(config_number_list, pvgt_list[i])
    
    # Plot the true distribution as a dotted line
    plt.plot(config_number_list, correct_distribution[1], label=r'True distribution', linestyle='dotted')
    plt.yscale("log")
    plt.legend(loc='lower left', fontsize=16)
    plt.savefig(plot_file_path)
    plt.show()
    return

# Example usage:
#topdir = ""#"../../" 
#repeats = 2
#data_params = "m 20 n 2b 200 [300, 300, 300, 300][0.2, 0.1, 0.05, 0.025]"
#data_date = "2023-11-29"
#folder_date = "2023-11-29"
#data_name = "raw_data"

#generate_config_plot(topdir, data_params, data_date, folder_date, data_name, repeats)
#from plot_log_likelihood import generate_loglik_plot

#generate_loglik_plot(topdir, data_params, data_date, folder_date, data_name)
