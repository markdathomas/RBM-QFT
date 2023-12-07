# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:49:49 2021
@author: kramm
"""

import numpy as np

from model_and_data_averages import prob_v_given_theta, partition_function
from Energy_function import energy

def log_likelihood(theta, batch, allv, allh):
    """
    Given theta, returns the log likelihood.
    
    Parameters:
    - theta (tuple): Tuple containing model parameters (w, b, c).
    - batch (list): List containing vectors in the batch.
    - allv (numpy.ndarray): All visible nodes.
    - allh (numpy.ndarray): All hidden nodes.
    
    Returns:
    - float: Log likelihood value.
    """
    loglik = 0
    Z = partition_function(allv, allh, theta)
    for v in batch:
        pvgt = prob_v_given_theta(allv, allh, v, theta, Z)
        loglik += np.log(pvgt)
        
    return loglik

def free_energy(v, theta, allh):
    """
    Calculates the free energy associated with a state v.
    
    Parameters:
    - v (numpy.ndarray): Visible node vector.
    - theta (tuple): Tuple containing model parameters (w, b, c).
    - allh (numpy.ndarray): All hidden nodes.
    
    Returns:
    - float: Free energy value.
    """
    summand = 0
    for h_vec in allh:
        summand += np.exp(-energy(v, h_vec, theta))
    F = np.log(summand)
    return F

def loss(batch_history, input_vector_history, output_vector_history, steps_list):
    """
    Takes the batches used to generate data, along with the input and output
    vectors of the cdk steps, and uses this to generate the loss as a function
    of step number (equation 22) for the learning procedure.
    
    Parameters:
    - batch_history (list): List containing batches used for each step.
    - input_vector_history (list): List containing input vectors for each step.
    - output_vector_history (list): List containing output vectors for each step.
    - steps_list (list): List containing the number of steps for each batch.
    
    Returns:
    - None: The function is currently not implemented.
    """
    number_of_batches = len(steps_list)
    for b in range(number_of_batches):
        Sb = batch_history[b]
        mod_Sb = len(Sb)
        for v in Sb:
            # Not sure what to do here
            1 == 1
    
    return

def reconstruction_error(batch_history, input_vector_history, output_vector_history):
    """
    Takes the batches used to generate data, along with the input and output
    vectors of the cdk steps, and uses this to generate the reconstruction error
    as a function of step number (equation 22) for the learning procedure.
    
    Returns:
    - None: The function is currently not implemented.
    """
    return
