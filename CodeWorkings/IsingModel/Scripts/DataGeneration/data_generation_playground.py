# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:06:49 2021
@author: Mark Thomas
"""

"""
Created on Tue Nov 23 23:58:48 2021
@author: Mark Thomas
"""

# Import necessary libraries
import os
import sys
import matplotlib.pyplot as plt
from tabulate import tabulate

# Get the path to the DataGeneration folder
Data_generation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataGeneration'))

# Add the DataGeneration folder to sys.path
sys.path.append(Data_generation_path)
from generate_data import run
from get_loglik_data import generate_loglik_data
from file_restructure import save_numpy_array, load_numpy_array, make_directory

# Get the path to the Implementation folder
implementation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Implementation'))

# Add the Implementation folder to sys.path
sys.path.append(implementation_path)

from all_vectors import all_vectors_ising
from model_and_data_averages import prob_v_given_theta, partition_function
from Learned_distribution import data_distribution

from plot_log_likelihood import generate_loglik_plot

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
        #if i % repeats == 0:
            plt.plot(config_number_list, pvgt_list[i], label=r'$\alpha$ = '+str(alpha_list[i]))
        #else:
        #    plt.plot(config_number_list, pvgt_list[i])
    
    # Plot the true distribution as a dotted line
    plt.plot(config_number_list, correct_distribution[1], label=r'True distribution', linestyle='dotted')
    plt.yscale("log")
    plt.legend(loc='lower left', fontsize=16)
    plt.savefig(plot_file_path)
    plt.show()
    return

# Function to repeat elements in a list N times
def repeat_elements(arr, N):
    return [elem for elem in arr for _ in range(N)]

def run_and_plot(m_visible, n_hidden, initial_alpha, alpha_ratio, batch_size, N_repeats, steps_per_epoch, double_first_epoch, number_of_epochs):
    """
    Run the model and generate log likelihood plot and learned distribution plot.

    Parameters:
    - m_visible: Number of visible units
    - n_hidden: Number of hidden units
    - alpha_size_list: List of alpha values
    - batch_size: Size of each batch
    - N_repeats: Number of repeats
    - steps_per_epoch: Steps per epoch
    - double_first_epoch: Whether to repeat the first alpha value one time
    - number_of_epochs: Total number of epochs

    Returns:
    - None
    """
    alpha_size_list = [initial_alpha*alpha_ratio**(-i) for i in range(number_of_epochs)]
    if double_first_epoch:
        alpha_size_list= [alpha_size_list[0]] + alpha_size_list
    # Repeat alpha values N_repeats times
    alpha_list = repeat_elements(alpha_size_list, N_repeats)
    
    # Set step size and k steps lists
    step_size_list = [int(steps_per_epoch) for i in range(len(alpha_list))]
    k_steps_list = [1 for i in range(len(step_size_list))]



    table = [
        ["Number of Ising configurations possible:", "2 ** "+str(m_visible)+" = "+str(2**m_visible)],
        ["Number of Ising configurations seen:", number_of_epochs*N_repeats*batch_size*steps_per_epoch],
        ["Percentage of possible configurations seen: ", str(100*(number_of_epochs*N_repeats*batch_size*steps_per_epoch)/(2**m_visible))+"%"],
        ["Step Size: " + str(step_size_list), "Alpha Size: " +str(alpha_size_list)],
    ]

    # Print the formatted table
    print()
    print("Training run details")
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    print("Waiting for " + str((int(double_first_epoch)+number_of_epochs)*N_repeats) +  " epochs to run. Please wait for "+str(2*((int(double_first_epoch)+number_of_epochs)*N_repeats))+ " loading bars to fill." )
    print()

    
    # Run the model and get the output location
    output_location = run(step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size)
    
    # Extract date and parameters from the output location
    topdir = ""  # Set the top-level directory path
    repeats = 2  # Set the repeat factor for plotting
    data_params = output_location[16:-13]
    data_date = output_location[5:15]
    folder_date = output_location[5:15]
    data_name = "raw_data"

    # Generate log likelihood plot and learned distribution plot
    generate_loglik_plot(topdir, data_params, data_date, folder_date, data_name)
    generate_config_plot(topdir, data_params, data_date, folder_date, data_name, repeats)
    
    return
