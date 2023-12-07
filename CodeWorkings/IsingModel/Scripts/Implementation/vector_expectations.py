# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:39:39 2021

@author: Mark Thomas
"""

#Imports from current directory:
from model_and_data_averages import data_average, model_average
from contrastive_divergence import prob_hi_is_1



# Expectations over data
def data_exp_wij(v, theta, arg):
    """(Function to be averaged over), is argument of E_{data} as 
    it appears in equation 13, remembering to replace 
    p(hi=1) with 2p(hi=1)-1 as the spin is measured in the (-1,1)
    basis, not in the (0,1) basis as derived in the paper.
    
    Returns the assocaited term for the argument of the E_{data} sum."""
    i,j = arg
    prob_term = prob_hi_is_1(i,v,theta) #Modified prob term
    if type(v)==int:
        return 0
    else:
        vj = v[j]
        output = prob_term*vj
        return output


def data_exp_bj(v, theta, arg):
    """Function to be averaged over, as in equation 17. Notice
    this time the calculation DOES NOT depend on the choice of 
    basis for spin, and so no edit is made to this term."""
    j = arg
    if type(v)==int:
        return 0
    else:
        vj_term = v[j]
        return vj_term

def data_exp_ci(v, theta, arg):
    """(Function to be averaged over), is argument of E_{data} as 
    it appears in equation 19, remembering to replace 
    p(hi=1) with 2p(hi=1)-1 as the spin is measured in the (-1,1)
    basis, not in the (0,1) basis as derived in the paper.
    """
    
    i = arg
    p_hi_term = prob_hi_is_1(i, v, theta)
    
    return p_hi_term


def model_exp_wij(v,theta, arg):
    """Function to be averaged over), is argument of E_{data} as 
    it appears in equation 13, remembering to replace 
    p(hi=1) with 2p(hi=1)-1 as the spin is measured in the (-1,1)
    basis, not in the (0,1) basis as derived in the paper.
    """
    
    i,j=arg
    prob_term = prob_hi_is_1(i, v, theta) #Converted into new basis.
    vj_term = v[j]
    output = prob_term*vj_term
    
    return output

def model_exp_bj(v, theta, arg):
    """Function to be averaged over, as in equation 17. Notice
    this time the calculation DOES NOT depend on the choice of 
    basis for spin, and so no edit is made to this term.
    """
    
    j = arg
    vj_term=v[j]
    
    return vj_term

def model_exp_ci(v, theta, arg):
    """(Function to be averaged over), is argument of E_{model} as 
    it appears in equation 19, remembering to replace 
    p(hi=1) with 2p(hi=1)-1 as the spin is measured 
    in the (-1,1) basis, not in the (0,1) basis as derived in the paper.
    """
    
    i = arg
    prob_term = prob_hi_is_1(i, v, theta)
    new_basis_prob_term =prob_term 
    
    return new_basis_prob_term

def e_data_wij(S,i, j, theta,Z):
    """Expected value for the data term in equation 13.
    
    Be careful that data_exp_wij is in the righht basis for how things
    should be calculated.
    """
    
    arg = i,j #Function needs both an i and j argument now.
    function = data_exp_wij #Modified into (-1,1) basis.
    average = data_average(S, function, theta, arg)
    
    return average

def e_data_bj(S, j, theta):
    """Expected value for the data term in equation 17
    
    !Uniquely for these parameters, the calculation is independent
    of the choice of basis for the Isnig-model spins, and so
    no change is made to the form of data_exp_bj
    """
    
    arg = j
    function = data_exp_bj
    average = data_average(S, function,theta, arg)
    
    return average

def e_data_ci(S, i, theta):
    """Expected value for the data term in equation 19"""
    arg = i
    function = data_exp_ci
    average = data_average(S,function,theta, arg)
    
    return average


def e_model_wij(i, j, All_v, All_h, theta,Z):
    """Expected value for the model term in equation 13
    
    Remember model_exp_wij is returning the result for when in the (-1,1)
    basis, so doesn't look like eqn13, just the equivalent.
    """
    
    function = model_exp_wij
    arg = i,j
    average = model_average(All_v, All_h, theta,function, arg,Z)
    
    return average
    
def e_model_bj(j, All_v, All_h, theta,Z):
    """Expected value for the model term in equation 17
    
    Remember model_exp_bj is independent of the choice of spin basis,
    and so UNLIKE the other two, no change is made to this step.
    """
    
    function = model_exp_bj
    arg = j
    average = model_average(All_v, All_h, theta, function, arg,Z)
    
    return average

def e_model_ci(i, All_v, All_h, theta,Z):
    """Expected value for the model term in equation 19
    
    Be careful to note that equation 18 only follows from equation 10 when
    working in the spin (0,1) basis, and not the (-1,1) basis!!!
    """
    
    function = model_exp_ci #Argument of E_{model} being averaged over
    arg = i
    average = model_average(All_v, All_h, theta, function, arg, Z)
    
    return average