# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 19:26:41 2021

@author: Mark Thomas
"""
#External imports:
import numpy as np
import sys
import os

#Imports from current directory:
from file_restructure import date_file, make_directory, save_numpy_array

import sys
import os

# Get the path to the Implementation folder
implementation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Implementation'))

# Add the Implementation folder to sys.path
sys.path.append(implementation_path)

# Now you can import the module
from Learned_distribution import learn_distribution

    
def generate_data_folder(run_parameters, new_folder_absolute_path, folder_name,array_name, topdir = "../../"):
    """Given the specified graph and learning parameters, generates a folder
    to store the data associated with the graph in the specified location, named
    accordingly to the graph and learning params.
    
    Returns the location of (/path to) the saved array."""
   
    #Define the new folder location, name and path:
    new_folder_name = date_file(folder_name)
    path = topdir+new_folder_absolute_path+date_file(folder_name)
    
    #Check if folder already exists, if not, make the directory:
    truth = os.path.exists(path)
    if not truth: #If the folder isn't yet made, create it
        make_directory(new_folder_name,new_folder_absolute_path, topdir)
    #Path to that folder:
    folder_path = topdir+new_folder_absolute_path+new_folder_name+"/"
    
    #Generate all the data associated with the graph for the specified learning run:
    distribution_data = learn_distribution(run_parameters)
    
    #Save data to the new folder
    saving_array = np.array([run_parameters, distribution_data], dtype = object) #Array to be saved
    
    #Save the array with the specified name and location
    saved_array_location = save_numpy_array(array_name, saving_array, folder_path)
    
    return saved_array_location




def run(step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size):
    """Performs the contrastive divergence algorithm
    for given graph and run parameters.
    
    Data is saved as "raw_data.npy" in the Data folder, with specified m,n, alpha and S labelled
    along with date of generation.
    
    Returns the location of the saved data.
    
    The run parameters are:
    step_size_list: List containg the number of steps in each learning phase, e.g. [100,100,100].
    m_visible: Number of visible nodes, e.g. 6.
    n_hidden: Number of hidden nodes. e.g. 6
    alpha_list: List containing the learning rate for each learning phase, e.g. [0.1, 0.01, 0.001]
    k_steps_list: List containing the number of k-steps (for the contrastive divergence algorithm) used in each learning phase, e.g. [2,2,2]
    batch_size: Number of training vectors randomly generated at each cdk step, e.g. 200
    
    """
    run_parameters = np.array([step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size], dtype = object)
    new_folder_absolute_path = "Data/"
    folder_name = "m " +str(m_visible)+" " +"n "+str(n_hidden) + "b "+ str(batch_size) + " "  + str(step_size_list) + str(alpha_list)
    array_name = "raw_data"
    
    topdir = ""#"../../"
    output_location = generate_data_folder(run_parameters, new_folder_absolute_path, folder_name,array_name, topdir)
    
    return output_location