# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 16:47:54 2021
@author: kramm
"""

# Import necessary modules
from model_and_data_averages import prob_v_given_theta, partition_function
import numpy as np

def log_likelihood(theta, batch, allv, allh):
    """
    Given model parameters (theta), a batch of visible vectors (batch),
    and all possible visible and hidden vectors (allv, allh),
    this function calculates the log likelihood of the batch.

    Parameters:
    - theta: Model parameters
    - batch: Batch of visible vectors
    - allv: All possible visible vectors
    - allh: All possible hidden vectors

    Returns:
    - log likelihood of the batch
    """
    
    # Initialize log likelihood
    loglik = 0
    
    # Calculate partition function Z
    Z = partition_function(allv, allh, theta)
    
    # Extract the visible vectors from the batch
    batch = batch[1]
    
    # Loop over each visible vector in the batch
    for v in batch:
        # Calculate the probability of v given theta and Z
        pvgt = prob_v_given_theta(allv, allh, v, theta, Z)
        
        # Update the log likelihood
        loglik += np.log(pvgt)
        
    # Normalize the log likelihood by the number of vectors in the batch
    return loglik / len(batch)
