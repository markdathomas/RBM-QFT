# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 21:37:16 2021

@author: Mark Thomas
"""

# External imports:
import numpy as np

# Imports from the current directory:
from Energy_function import energy


def partition_function(All_v, All_h, theta):
    """
    Finds the partition of a system given theta under the assumption
    that it is for the Ising model. This method calculates the partition function
    slowly, and does not implement the annealed importance sampling discussed
    in pages 5 and 6 of https://arxiv.org/pdf/1810.11503.pdf.

    Parameters:
    - All_v (numpy.ndarray): All visible nodes.
    - All_h (numpy.ndarray): All hidden nodes.
    - theta (tuple): Tuple containing model parameters (w, b, c).

    Returns:
    - float: Partition function.
    """
    no_v = len(All_v)
    no_h = len(All_h)
    Z = 0
    for i in range(no_v):
        for j in range(no_h):
            Z += np.exp(-energy(All_v[i], All_h[j], theta))
    return Z


def prob_v_given_theta(all_visible, all_hidden, v, theta, Z):
    """
    For the given theta, finds p(v|theta) as written in equation 5.
    Notice this is a function of the values of the spins, and so
    is the output will be different for two equivalent vectors written in
    the different bases.

    Parameters:
    - all_visible (numpy.ndarray): All visible nodes.
    - all_hidden (numpy.ndarray): All hidden nodes.
    - v (numpy.ndarray): Visible node vector.
    - theta (tuple): Tuple containing model parameters (w, b, c).
    - Z (float): Partition function.

    Returns:
    - float: Probability p(v|theta).
    """
    m_visible = len(theta[1])
    n_hidden = len(theta[2])

    w, b, c = theta

    prod_1 = 1  # Product multiplies, and so start with 1.

    for j in range(m_visible):
        prod_1 *= np.exp(b[j] * v[j])  # First product term in equation 5.

    prod_2 = 1
    for i in range(n_hidden):
        hsum = c[i]
        for j in range(m_visible):
            hsum += w[i][j] * v[j]
        prod_2 *= 1 + np.exp(hsum)

    p = prod_1 * prod_2 / Z

    return p


def data_average(S, function, theta, arg):
    """
    Calculate the average of the function F over the batch S of vectors v
    as appears in equation 14.
    Here, S is the specified batch
    function is the function F to be averaged over
    arg is the relevant arguments for the function requested.

    Parameters:
    - S (list): List containing vectors in the batch.
    - function (function): Function to be averaged.
    - theta (tuple): Tuple containing model parameters (w, b, c).
    - arg: Relevant argument for the function.

    Returns:
    - float: Average of the function over the provided vector batch.
    """
    average = 0
    for v in S:
        average += function(v, theta, arg)
    if len(S) != 0:
        average = average / len(S)
    return average


def model_average(All_v, All_h, theta, function, arg, Z):
    """
    Similar to data_average but works out the model average
    of the function f, as in equation 15.
    Returns the average over the full theoretical model, and so requires the
    partition function.

    Parameters:
    - All_v (numpy.ndarray): All visible nodes.
    - All_h (numpy.ndarray): All hidden nodes.
    - theta (tuple): Tuple containing model parameters (w, b, c).
    - function (function): Function to be averaged.
    - arg: Relevant argument for the function.
    - Z (float): Partition function.

    Returns:
    - float: Model average of the function.
    """

    average = 0  # Average is over a sum, so start with 0 and add.
    for v in All_v:
        pv = prob_v_given_theta(All_v, All_h, v, theta, Z)
        average += pv * function(v, theta, arg)  # Notice pv here, not in data_average
    return average
