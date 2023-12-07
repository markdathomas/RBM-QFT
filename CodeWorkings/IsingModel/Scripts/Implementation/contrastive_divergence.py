# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 19:34:26 2021

@author: Mark Thomas
"""

import numpy as np
import random


def sigmoid(x):
    """Returns the sigmoid function given an argument x"""
    return np.exp(-np.logaddexp(0, -x)) #Writing this way ensures floating point accuracy


def prob_hi_is_1(i, v_vec,theta):
    """Find the probability that the hidden nodes ith entry is switched on,
    given v_vec, the nth visible training vector in the CD algorithm.
    
    Calculates the probability using equation 7 of 
    https://arxiv.org/pdf/1810.11503.pdf
    
    Returns this probability.
    """
    w_matrix,b_vec,c_vec = theta
    if type(v_vec)==int:
        print("Error in specifying vector") #Should be an array
        return 0
    
    else: 
        m = len(v_vec) # Number of visible nodes
        assert i <= m #Requested index not in list!
        
        marg_number = 0 #Sum that appears in equation 7 
        for j in range(m):
            marg_number+= w_matrix[i][j]*(v_vec[j]) #Add to the sum
        marg_number += c_vec[i] #Add to the sum
        prob = sigmoid(marg_number) #Prob is sigmoid of the sum, factor of 2 for (-1,+1) basis
        return prob


def prob_vj_is_1(j, h_vec,theta):
    """Find the probability that the visible nodes jth entry is switched on,
    given h_vec, the nth visible training vector in the CD algorithm.
    
    Calculates the probability using equation 8 of 
    https://arxiv.org/pdf/1810.11503.pdf
    
    Returns this probability.
    """
    w_matrix,b_vec,c_vec = theta
    n = len(h_vec) # Number of hidden nodes
    m = len(b_vec)
    assert j <= m #Requested index not in list!
    marg_number = 0 #Sum that appears in equation 8
    for i in range(n):
        marg_number+= w_matrix[i][j]*h_vec[i]#Add to the sum
    marg_number += b_vec[j] #Add to the sum
    prob = sigmoid(marg_number) #Prob is sigmoid of the sum, factor of 2 for (-1,+1) basis.
    return prob




def cd_hi_step(i,vn,theta):
    """For this step of the cd, determine hi value given training vector vn"""
    u = random.uniform(0.0, 1.0)
    
    p_hi = prob_hi_is_1(i, vn, theta)
    if p_hi>u: #Switch the node on +1
        return 1
    else: #Turn it off -1 (or 0 in the (0,1) basis)
        return 0


def cd_h_step(number_of_hidden_nodes, vn, theta):
    """Given a visible training vector vn, calculates hn
    using the contrastive divergence algorithm", and returns hn"""
    hn = np.zeros(number_of_hidden_nodes)
    
    for i in range(number_of_hidden_nodes): #For each vector component
        hn[i] = cd_hi_step(i, vn, theta) #Generate the component using CD
    return hn


def cd_vj_step(j, hn, theta):
    """For this step of the cd, determine hi value given hn"""
    u = random.uniform(0.0, 1.0)
    p_vj = prob_vj_is_1(j, hn, theta)
    if p_vj>u: #Switch the node on +1
        return 1 
    else: #Turn it off -1 (or 0 in the (0,1) basis)
        return 0


def cd_v_step(number_of_visible_nodes, hn, theta):
    """Given a hidden training vector hn, calculates vn
    using the contrastive divergence algorithm", and returns vn"""
    vn = np.zeros(number_of_visible_nodes)
    
    for j in range(number_of_visible_nodes):#For each vector component
        vn[j] = cd_vj_step(j, hn, theta) #Generate the component using CD
    return vn


def cdk(k, theta, data_dist):
    """Given a specified theta, generates the vectors vk and hk for the 
    contrastive divergence algorithm with k (specified) steps.
    
    Returns vk and hk as vk,hk"""
    
    m_visible = len(theta[1])
    n_hidden = len(theta[2])
    #Generate a random vector of length m_visible with binary entries:


    #new_training_v_01 = np.asarray(np.random.randint(2, size=m_visible))



    
    # Perform weighted random selection
    selected_node = random.choices(data_dist[0], data_dist[1], k=1)[0].flatten().astype(int)
    #print(selected_node)
    selected_node = np.asarray(selected_node)
    

    new_training_v_01 = selected_node

    #print(new_training_v_01)


    new_training_v_11 = new_training_v_01
    #Let the "current" (and first) visible training vector be selected at random
    vn = new_training_v_11

    if k==0:
        print("No cdk steps taken!") #Shouldn't be calling a cdk step with no steps!
        return 0,0
    else:
        for n in range(k):
            hn = cd_h_step(n_hidden, vn, theta)
            vn = cd_v_step(m_visible, hn, theta)
        vk,hk = vn,hn
        return vk, new_training_v_01







def batch_generation(k, theta, batch_size, data_dist):
    """Generate a batch of vectors to be used to increment theta, 
    return this batch as a list of these vectors.

    Input requires k (the number of cd steps), the current theta value, and 
    the desired batch size used for taking averages.
    
    
    To test batch_generarion(), try something like:
        
        v = np.array([1,0,1])
        h = np.array([1,0])
    
        w = np.array([[0.1,0.1,0.2],[0.1,0.2,-0.3]])
        b = np.array([0.1,0.2,0.5])
        c = np.array([0.5,-0.1])
    
        theta = (w,b,c)
    
    
        training_v = v
        k= 5
        n_hidden = 2
        n_visible = 3
    
        vk,hk = cdk(k,n_hidden,training_v, theta)
        a = batch_generation(k, n_hidden, theta, 1000, vk)
        print(a)
    """
    S_data = []
    S_k = []
    for i in range(batch_size):
        vk,v_data = cdk(k, theta, data_dist)
        v_dash = vk #Only the visible vector is stored and used for the batch.
        S_k.append(v_dash)
        S_data.append(v_data)
    
    return [S_k, S_data]