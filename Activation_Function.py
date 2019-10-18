# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:01:50 2019

@author: Arthur
"""
import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1 / (1+np.exp(-Z))
    
    assert(A.shape == Z.shape)
    
    cache = Z
    return A, cache

def tanh(Z):
    """
    Implements the tanh activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of tanh(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    assert(A.shape == Z.shape)
    
    cache = Z
    return A, cache

def relu(Z):
    """
    Implements the relu activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of relu(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = np.maximum(0, Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def tanh_backward(dA,cache):
    """
    Implement the backward propagation for a single TANH unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    
    s = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    dZ = dA * (1 - np.square(s))
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
