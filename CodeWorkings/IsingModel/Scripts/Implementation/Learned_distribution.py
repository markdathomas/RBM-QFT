# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:52:06 2021

@author: Mark Thomas
"""
# External imports:
import numpy as np
import random
from tqdm import tqdm

# Imports from current directory:
from learning_Step import take_N_cdk_steps
from all_vectors import all_vectors_ising


def Hamiltonian(vector, T):
    """
    Calculate the Hamiltonian of a given vector with temperature T.

    Parameters:
    - vector (numpy.ndarray): Input vector.
    - T (float): Temperature.

    Returns:
    - float: Hamiltonian contribution.
    """
    contribution = 0
    for i in range(len(vector)):
        si = 2 * vector[i - 1] - 1
        si1 = 2 * vector[i] - 1
        contribution -= si * si1 / T
    return contribution


def data_distribution(m, T):
    """
    Generate a distribution of vectors and their probabilities based on the Ising model.

    Parameters:
    - m (int): Number of nodes.
    - T (float): Temperature.

    Returns:
    - Tuple: Tuple containing all vectors and their probabilities.
    """
    allv = all_vectors_ising(m)
    vector_probs = []
    for i in range(len(allv)):
        vector = allv[i]
        H = Hamiltonian(vector, T)
        vector_probs.append(np.exp(-H))
    Z = sum(vector_probs)
    vector_probs = [vector_probs[i] / Z for i in range(len(vector_probs))]
    return allv, vector_probs


def learn_distribution(run_parameters):
    """
    Learn the distribution for a given set of run parameters.

    Parameters:
    - run_parameters (numpy.ndarray): Array containing run parameters organized as follows:
        [step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size].
        - step_size_list: List containing the number of epochs per learning stage, e.g., [100, 100, 200, 200].
        - m_visible: Number of visible nodes.
        - n_hidden: Number of hidden nodes.
        - alpha_list: List containing the learning rate at each stage.
        - k_steps_list: List containing the number of k-steps taken in the contrastive divergence algorithm at each learning stage.
        - batch_size: Number of vectors used in the batch to calculate averages of the likelihood function.

    Returns:
    - Tuple: Tuple containing distribution data in the form: init_theta, theta_history_list, batch_history, cdk_history.
        - init_theta: Initial theta values.
        - theta_history_list: History of theta values during the learning process.
        - batch_history: History of batches used during the learning process.
        - cdk_history: History of contrastive divergence steps taken during the learning process.
    
    !Notes on how batch history works (theta also works similarly)!:
    1.) batch history is a list of all batches used, divided into the step sets.
    2.) batch_history[0] is the set of batches used for each step, for the first set of steps.
    3.) batch_history[0][0] is batch used on the first step of the first set of steps.
    4.) batch_history[0][0][0] is the first vector used in batch_history[0][0].
    """

    step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size = run_parameters

    # Generate random initial theta to start the learning with:
    init_b = np.asarray([random.uniform(-1, 1) for j in range(m_visible)])
    init_c = np.asarray([random.uniform(-1, 1) for i in range(n_hidden)])
    init_w = np.asarray([[random.uniform(-1, 1) for j in range(m_visible)] for i in range(n_hidden)])
    # theta = {w_ij,b_i, c_j}:
    init_theta = init_w, init_b, init_c

    # Number of curves for the distribution plot
    N_curves = len(step_size_list)

    # Generate the set of all vectors associated with the Ising model
    allv = all_vectors_ising(m_visible)
    allh = all_vectors_ising(n_hidden)

    data_dist = data_distribution(m_visible, T=1)

    total_steps = sum(step_size_list)  # Total number of steps used in learning
    # Empty list ready to record all theta, batches, and v(k) generated:
    theta_history_list = np.zeros(sum(step_size_list), dtype=object)
    batch_history = np.zeros(total_steps, dtype=object)
    cdk_history = np.zeros(total_steps, dtype=object)

    theta = init_theta  # Set the "current" theta to the initial theta before running the algorithm to advance theta

    for step in tqdm(range(N_curves), position=0, leave=True):
        # For each learning region, extract the number of steps, alpha, and k
        N_steps = step_size_list[step]
        alpha = alpha_list[step]
        k = k_steps_list[step]

        # Empty arrays for current learning region data:
        for this_step in tqdm(range(N_steps), position=0, leave=True):
            # For each step in the current learning run

            # Generate a batch and associated theta using the contrastive divergence algorithm,
            # Note 1 step taken as generalization to N needed workaround to keep theta and batch history,
            # so instead, this function is just called many times.
            theta_new_list, batch_record = take_N_cdk_steps(allv, allh, theta, alpha, 1, k, batch_size, data_dist)
            theta = theta_new_list[-1]  # Take the kth cdk theta output

            i = sum(step_size_list[:step]) + this_step  # This step number (total)
            theta_history_list[i] = theta  # Record this as theta for this step

            batch_history[i] = batch_record[-1]  # Take the kth batch

    # Group and return the distribution data
    distribution_data = init_theta, theta_history_list, batch_history, cdk_history
    return distribution_data
