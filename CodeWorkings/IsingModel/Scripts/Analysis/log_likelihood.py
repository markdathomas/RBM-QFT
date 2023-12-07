# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:47:54 2021

@author: kramm
"""

from model_and_data_averages import prob_v_given_theta, partition_function
import numpy as np

def log_likelihood(theta, batch, allv, allh):
    """Given theta, returns the log likelihood"""
    loglik = 0
    Z = partition_function(allv, allh, theta)
    batch = batch[1]
    for v in batch:
        pvgt = prob_v_given_theta(allv, allh, v, theta, Z)
        loglik+=np.log(pvgt)
        
        
    return loglik/len(batch)