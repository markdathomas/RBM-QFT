# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 18:17:47 2021

@author: Mark Thomas
"""

# External imports:
import numpy as np

def energy(v_vec, h_vec, theta):
    """
    Calculate the energy E(v, h; theta) as written in 
    equation 2 of https://arxiv.org/pdf/1810.11503.pdf
    associated with a configuration given the weights w, b, c stored
    in theta as theta = (w_{ij}, b_{j}, c_{i}).
    
    NOTE: v_vec and h_vec should be numpy arrays containing 1s and -1s, not 0s,
    as that is the wrong spin basis; the energy assumes we're in the (-1, +1)
    basis.
    
    Parameters:
    - v_vec (numpy.ndarray): Visible vector.
    - h_vec (numpy.ndarray): Hidden vector.
    - theta (tuple): Model parameters (w_matrix, b_vec, c_vec).
    
    Returns:
    - float: Energy E.
    
    Example:
    v = np.array([1, 0, 1])
    h = np.array([1, 0])

    w = np.array([[2, 3, 2], [1, 2, 3]])
    b = np.array([4, 1, 5])
    c = np.array([5, -1])

    theta = (w, b, c)
    
    e = energy(v, h, theta)
    """
    w_matrix, b_vec, c_vec = theta
    empty = np.array([])

    n = len(h_vec)  # Number of hidden nodes
    m = len(v_vec)  # Number of visible nodes

    # Ensure that the data is the correct type and shape to calculate E with:
    assert type(w_matrix) == type(empty)  # Ensure that w is a numpy array
    assert np.shape(w_matrix) == (n, m)  # Ensure the right amount of weights
    assert len(b_vec) == m
    assert len(c_vec) == n

    E = 0  # Energy in equation 2 given by a sum, so start at zero
    
    # Add up everything in equation 2 (ignoring minus signs for now)
    for j in range(m):
        E += b_vec[j] * v_vec[j]
    for i in range(n):
        E += c_vec[i] * h_vec[i]
        for j in range(m):
            E += h_vec[i] * w_matrix[i][j] * v_vec[j]

    # Incorporate the minus signs into eqn 2:
    E = -E
    return E
