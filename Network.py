# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:01:50 2019

@author: Arthur
"""
import numpy as np
import matplotlib.pyplot as plt
from Activation_Function import sigmoid, sigmoid_backward, tanh, tanh_backward, relu, relu_backward
import math
import json

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    parameters = {}
    L = len(layer_dims)            # number of layers in the network,include input layer

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for cat / 0 for non-cat), of shape (10, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches : ]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A_prev", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A_prev)+b
    
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "tanh":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, linear_activation_cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation = "relu")
        caches.append(linear_activation_cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, linear_activation_cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], activation = "sigmoid")
    caches.append(linear_activation_cache)
    
    assert(AL.shape == (10, X.shape[1]))
    
    return AL, caches

def compute_cost(X, Y, parameters, lambd):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (10, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    L = len(parameters) // 2 # number of layers in the neural network

    AL, _ = L_model_forward(X, parameters)
    # Compute loss from AL and Y.
    cost = np.sum(np.nan_to_num(-Y * np.log(AL) - (1-Y) * np.log(1 - AL))) / m
    # cost = 0.5 * np.sum(np.square(AL - Y)) / m

    for l in range(1, L+1):
        cost += 0.5 * (lambd / m) * np.sum(np.square(parameters["W"+str(l)]))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache, lambd):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m + (lambd / m) * W
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, lambd, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, lambd):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = np.nan_to_num(-np.divide(Y, AL) + np.divide(1-Y, 1-AL)) # derivative of cost with respect to AL
    # dAL = AL - Y # derivative of cost with respect to AL

    assert(dAL.shape == AL.shape)
    
    # The Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, caches[L-1]". 
    #Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    linear_activation_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, linear_activation_cache, lambd, activation = "sigmoid")
    
    for l in reversed(range(1, L)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l)], caches[l-1]".
        # Outputs: "grads["dA" + str(l-1)] , grads["dW" + str(l)] , grads["db" + str(l)]
        linear_activation_cache = caches[l-1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l)], linear_activation_cache, lambd, activation = "relu")
        grads["dW"+str(l)] = dW_temp
        grads["db"+str(l)] = db_temp
        grads["dA"+str(l-1)] = dA_prev_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(1, L+1):
        parameters["W"+str(l)] -= learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)] -= learning_rate*grads["db"+str(l)]
        
    return parameters

def predict(X, Y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    L = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1, m))
    
    # Forward propagation
    probas, _ = L_model_forward(X, parameters)
    #print(probas[:,0:10])
    
    # convert probas to 0/1 predictions
    p = np.argmax(probas, axis=0)
    y = np.argmax(Y, axis=0)
    
    print("Accuracy: "  + str(np.sum(p == y) / m))
    
    return p, y

def print_mislabeled_images(X, p, y):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """

    mislabeled_indices = np.asarray(np.where(p != y))
    num_images = len(mislabeled_indices[0])
    for i in range(10):
        index = mislabeled_indices[0][i]
        
        plt.figure()
        plt.imshow(X[:,index].reshape(28,28), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + str(p[index]) + " \n Expected: " + str(y[index]))
        plt.show()

def save(filename, layer_dims, parameters):
    """
    Save the neural network to the file ``filename``.
    """

    data = {"layer_dims": layer_dims}
    for key in parameters:
        data[key] = parameters[key].tolist()
    fo = open(filename, 'w')
    json.dump(data, fo)
    fo.close()

def load(filename):
    """
    Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """

    parameters = {}
    fo = open(filename, "r")
    data = json.load(fo)
    fo.close()
    layer_dims = data["layer_dims"]
    data.pop("layer_dims")
    for key in data:
        parameters[key] = np.asarray(data[key])

    return layer_dims, parameters
