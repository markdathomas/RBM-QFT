import sys
from tqdm import trange, tqdm

import os
import sys

from file_restructure import save_numpy_array

# Get the path to the Implementation folder
Implementation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Implementation'))

# Add the Implementation folder to sys.path
sys.path.append(Implementation_path)
from all_vectors import all_vectors_ising


# Get the path to the Implementation folder
Analysis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Analysis'))

# Add the Implementation folder to sys.path
sys.path.append(Analysis_path)
from Data_load import load_data
from log_likelihood import log_likelihood

def generate_loglik_data(folder_date, data_params, data_date, data_name="raw_data"):
    """
    Load the data for the specified file, find loglik data, and save.

    Parameters:
    - folder_date (str): The date of the folder containing the data.
    - data_params (str): Parameters describing the data.
    - data_date (str): The date of the specific data.
    - data_name (str, optional): The name of the data file. Default is "raw_data".
    """

    #2021-11-24 m 6 n 6b 200 [2000, 1000, 1000][0.01, 0.001, 0.0001]/2021-11-25 m 6 n 6b 200 [2000, 1000, 1000][0.01, 0.001, 0.0001].npy'
    
    #./../Data/2021-11-25 m 6 n 6b 200 [200, 100, 100, 100][0.1, 0.01, 0.001, 0.0001]/2021-11-25 m 6 n 6b 200 [200, 100, 100, 100][0.1, 0.01, 0.001, 0.0001].npy 
    data_date = None
    relevant_data = load_data(folder_date, data_date, data_params, data_name)
    
    run_parameters, distribution_data = relevant_data
    step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size = run_parameters
    init_theta, theta_history_list, batch_history, cdk_history = distribution_data

    
    allv = all_vectors_ising(m_visible)
    allh = all_vectors_ising(n_hidden)
    step_number_list = []
    loglik_list = []

    
    
    for epoch_number in trange(len(step_size_list)):
        
        for step in tqdm(range(step_size_list[epoch_number]), position=0, leave=True):
            
            i = sum(step_size_list[:epoch_number]) + step
            step_number_list.append(i)
            current_batch = batch_history[i]
            current_theta = theta_history_list[i]
        
            #current_batch = batch_history[epoch_number][step] 
            ll = log_likelihood(current_theta, current_batch, allv, allh)
            
            loglik_list.append(ll)
    
    
    loglik_name = "loglik"  # Putting "raw_data" here overwrites old file
    file_to_save = loglik_name
    folder_path = "Data/"+folder_date+" "+data_params+"/"
    
    loglik_name = file_to_save  # date_file(file_to_save)
    
    save_numpy_array(loglik_name, loglik_list, folder_path)
    
    return

# Example usage
# topdir = "../../" 
# data_params = "m 2 n 2b 200 [1000, 1000, 1000, 1000][0.1, 0.01, 0.001, 0.0001]"
# data_date = "2021-12-06"
# folder_date = "2021-12-06"
# data_name = "raw_data"
# generate_loglik_data(folder_date, data_params, data_date, data_name)
