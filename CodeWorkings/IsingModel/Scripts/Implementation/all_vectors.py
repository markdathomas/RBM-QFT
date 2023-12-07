# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:06:04 2021

@author: Mark Thomas
"""
import numpy as np


def dec_to_bin(x):
    """Take in the argument x, an integer in base 10
    and output the correpsoning binary number."""
    return int(bin(x)[2:])



def bin_to_rep(x, number_of_nodes):
    """Take the decimal number label of the configuration
    number, and return the corresponding binary number with 
    zeros out the front to represent nodes that are 'down' """
    a = str(x)
    c = str()
    b = number_of_nodes - len(a)
    for i in range(b):
        c+=str(0)
    c+=a
    return  c


def all_vectors_ising(number_of_nodes):
    """Given a specified number of nodes (either hidden or visible), 
    returns every possible configuration of those nodes (as allowed
    in the Ising model) as a numpy array.
    
    NOTE: Outputs are vectors with possible entries -1 or 1, not 0 and 1.
    """
    
    vector_collection = [] #Empty set of all vectors to be filled
    
    for i in range(2**number_of_nodes): #Number of possible configs is 2^N
        bin_i = dec_to_bin(i) #Binary representation of the ith configuration
        rep_i = bin_to_rep(bin_i, number_of_nodes) #Rep with zeros
        vi = np.zeros(number_of_nodes) #Vector representing the current state in (0,1) basis, empty
        
        for j in range(number_of_nodes): #For each vector component:
            
            vector_entry = int(str(rep_i)[j]) #Find associated binary entry (0 or 1)
            vi[j] = vector_entry #Convert from (0,1)->(-1,1) basis.
            
        vector_collection.append(vi) #Add this vector to the collection.
        
    vector_collection = np.asarray(vector_collection)
    return vector_collection