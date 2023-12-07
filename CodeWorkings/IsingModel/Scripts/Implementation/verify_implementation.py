# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 09:21:22 2021
@author: Mark Thomas
"""

# External imports:
import numpy as np
from tqdm import tqdm  # Progress bar library
from math import isclose
import time

# Imports from the current directory:
from all_vectors import all_vectors_ising
from model_and_data_averages import prob_v_given_theta
from model_and_data_averages import partition_function

def generate_random_theta(m_visible, n_hidden):
    """Generate a random set of parameters theta given graph structure.
    Return this theta."""
    w = np.asarray([np.zeros(m_visible) for i in range(n_hidden)])
    b = np.asarray(np.zeros(m_visible))
    c = np.asarray(np.zeros(n_hidden))
    
    # For each theta entry, generate a random number in [-1,1]:
    for i in range(n_hidden):
        c[i] = np.random.uniform(-1,1)
        for j in range(m_visible):
            w[i][j] = np.random.uniform(-1,1)
    for j in range(m_visible):
        b[j] = np.random.uniform(-1,1)
    theta = (w,b,c)
    return theta

def check_normalisation(theta, Z):
    """Given an input set of theta values, check that the partition function
    is correctly normalising this setup.
    Returns the cumulative probability of each state being occupied, and
    so adds to 1 if everything working correctly."""
    
    m_visible = len(theta[1])
    n_hidden = len(theta[2])
    all_visible = all_vectors_ising(m_visible)
    all_hidden = all_vectors_ising(n_hidden)
    cumulative_prob = 0
    for i in range(len(all_visible)):
        v = all_visible[i]
        pi = prob_v_given_theta(all_visible, all_hidden, v, theta, Z)
        cumulative_prob += pi
    return cumulative_prob

def check_normalisation_many_graphs(end_m, end_n, checks_per_config):
    """Checks the normalisation of randomly configured graphs for 
    (m, n)={(0,0), (0,1),....(end_m, end_n)} to ensure that p(v|theta)
    is working correctly. Checks the graphs of each dimensionality
    "checks_per_config" times.
    
    Prints whether there's an issue with normalisation, if not, prints
    that all requested configurations have been checked and are normalised.
    
    Returns time taken to make the verification (can be lengthy), and
    also whether all_normalised (which = True if all graphs were correctly normalised)
    
    Note, Z is currently calculated without approximation, and so checking large
    m or n values makes this a very slow function to run.
    
    (end_m,end_n) = (10,10) takes approx 6 minutes  * checks_per_config
    to verify normalisation.
    """
    
    start_time = time.time()
    
    all_normalised = True
    issue_configs = []
    for j in tqdm(range(checks_per_config), position=0, leave=True):
        for m in range(1, end_m):
            for n in range(1, end_n):
                theta = generate_random_theta(m, n)  # Check the slow ones first
                all_visible = all_vectors_ising(m)  # SLOW STEP
                all_hidden = all_vectors_ising(n)   # SLOW STEP
                Z = partition_function(all_visible, all_hidden, theta)
                checked_norm = check_normalisation(theta, Z)
                truth = isclose(checked_norm, 1, abs_tol=10**(-17))
                if not truth:
                    issue_configs.append(np.array([m, n]))
                    all_normalised = False
    if all_normalised:
        print()
        print("All configurations checked normalised to machine accuracy")
    else:
        print()
        print("(m, n) configurations with issues:")
        print(issue_configs)
        
    end_time = time.time()
    duration = end_time - start_time
    print()
    print("Time taken: ", duration)
    return duration, all_normalised

# Example usage:
# end_m_visible = 6
# end_n_hidden = 6
# checks_per_config = 10
# check_normalisation_many_graphs(end_m_visible, end_n_hidden, checks_per_config)
