# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:06:49 2021

@author: Mark Thomas
"""

"""
Created on Tue Nov 23 23:58:48 2021

@author: Mark Thomas
"""
import os
import sys


#External imports: 
import matplotlib.pyplot as plt
import sys
import os.path



# Get the path to the Implementation folder
Data_generation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataGeneration'))

# Add the Implementation folder to sys.path
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
    
    raw_data_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_name+".npy"
    dist_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_date+" "+"dist.npy"
    
    run_parameters, distribution_data = load_numpy_array(raw_data_filepath)
    
    step_size_list = run_parameters[0]
    
    theta_history_list = distribution_data[1]
    m_visible = run_parameters[1]
    n_hidden = run_parameters[2]
   
    alpha_list = run_parameters[3]
    allv = all_vectors_ising(m_visible)
    
    if not os.path.isfile(dist_filepath): #If the loglik file hasn't been generated yet
        allh = all_vectors_ising(n_hidden)
        
        current_step = 0
        relevant_theta_list = []
        pvgt_list = []
        
        for epoch_number in range(len(step_size_list)):
            current_step += step_size_list[epoch_number]
            current_theta = theta_history_list[current_step-1]
            relevant_theta_list.append(current_theta)
            
            Z = partition_function(allv, allh, current_theta)
            pvgt = []
            for v in allv:

                pvgt.append(prob_v_given_theta(allv, allh, v,current_theta, Z))
            pvgt_list.append(pvgt)
    
        dist_name = "dist"# Putting "raw_data" here overwrites old file
        file_to_save =   data_date + " " + dist_name
        folder_path = "Data/"+folder_date+" "+data_params+"/"
        
        dist_name = file_to_save#date_file(file_to_save)
        
        data_to_save = [pvgt_list, relevant_theta_list]
        data_to_save = pvgt_list
     
        save_numpy_array(dist_name, data_to_save, folder_path)
    
    else:
        print("Using older data")
    
    
    pvgt_list = load_numpy_array(dist_filepath)
    config_number_list = [i for i in range(len(allv))]
        
        
    plots_folder_path = topdir+"Plots/"+folder_date+" "+data_params
    plot_file_path = plots_folder_path+"/"+"Learned distribution"+".pdf"
    #If a folder for the plots doesn't exist yet, create one:
    os.makedirs(plots_folder_path, exist_ok=True)  
    

    correct_distribution = data_distribution(m_visible, 1)

    plt.figure()
    plt.title("Learned distribution", fontsize = 30)
    plt.xlabel("Config number", fontsize = 30)
    plt.ylabel(r'$p\left(v|\theta \right)$', fontsize = 30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    for i in range(len(pvgt_list)):
        if i%repeats ==0:
            plt.plot(config_number_list, pvgt_list[i], label = r'$\alpha$ = '+str(alpha_list[i]))
        else:
            plt.plot(config_number_list, pvgt_list[i])
    plt.plot(config_number_list, correct_distribution[1], label=r'True distribution', linestyle = 'dotted')
    plt.yscale("log")
    plt.legend(loc = 'lower left', fontsize=16)
    plt.savefig(plot_file_path)
    plt.show()
    return

    #return relevant_theta_list
    # 
    # # -*- coding: utf-8 -*-


def repeat_elements(arr, N):
    return [elem for elem in arr for _ in range(N)]

def run_and_plot(m_visible, n_hidden, alpha_size_list, batch_size, N_repeats, steps_per_epoch):
    alpha_list = repeat_elements(alpha_size_list, N_repeats)
    step_size_list = [int(steps_per_epoch) for i in range(len(alpha_list))]
    k_steps_list = [1 for i in range(len(step_size_list))]
    output_location = run(step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size)
    topdir = ""#"../../" 
    repeats = 2
    data_params = output_location[16:-13]
    data_date = output_location[5:15]
    folder_date = output_location[5:15]
    data_name = "raw_data"

    generate_loglik_plot(topdir, data_params, data_date, folder_date, data_name)
    generate_config_plot(topdir, data_params, data_date, folder_date, data_name, repeats)
    
    return


