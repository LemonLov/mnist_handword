# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:01:50 2019

@author: Arthur
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import load_data_wrapper
from Network import (initialize_parameters_deep, random_mini_batches,
    L_model_forward, compute_cost, L_model_backward, update_parameters,
    predict, print_mislabeled_images, save, load)

plt.rcParams['figure.figsize'] = (8.0, 6.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x_orig, train_y_orig, test_x_orig, test_y_orig = load_data_wrapper()

m_train = train_x_orig.shape[0]
num_px = int(np.sqrt(train_x_orig.shape[1]))
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ")")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y_orig shape: " + str(train_y_orig.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y_orig shape: " + str(test_y_orig.shape))

# Example of a picture
index = 1
plt.figure()
plt.imshow(train_x_orig[index].reshape(num_px, num_px))
plt.show()

print("After Reshape...")
# Reshape the training and test examples
train_x = train_x_orig.T
test_x = test_x_orig.T
# The "-1" makes reshape flatten the remaining dimensions
train_y = train_y_orig.reshape(train_y_orig.shape[0], -1).T
test_y = test_y_orig.reshape(test_y_orig.shape[0], -1).T

print ("train_x's shape: " + str(train_x.shape))
print ("train_y's shape: " + str(train_y.shape))
print ("test_x's shape: " + str(test_x.shape))
print ("test_y's shape: " + str(test_y.shape))

training_data = (train_x, train_y)
evaluation_data = (test_x, test_y)

### CONSTANTS ###
# [784, 30, 10]
layer_dims = [784, 30, 30, 10] #  4-layer model

def L_layer_model(training_data, evaluation_data, layers_dims, learning_rate, 
                  minibatch_size, lambd, epoch,
                  monitor_training_cost = False, monitor_evaluation_cost = False):

    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (10, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    minibatch_size -- mini_batch_size -- size of the mini-batches, integer
    lambd -- regularization coefficient of the cost function
    learning_rate -- learning rate of the gradient descent update rule
    epoch -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    (train_x, train_y) = training_data[0], training_data[1]
    (test_x, test_y) = evaluation_data[0], evaluation_data[1]
    train_cost = []      # keep track of train_cost
    evaluation_cost = []
    m = train_x.shape[1] # numbers of training set
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layer_dims)
    # Load the network's parameters
    #layer_dims, parameters = load("model/myNetwork.py")
    
    # Loop (gradient descent)
    for i in range(epoch):
        minibatches = random_mini_batches(train_x, train_y, minibatch_size)

        for minibatch in minibatches:
            # Select a minibatch.
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(minibatch_X, parameters)
            
            # Backward propagation.
            grads = L_model_backward(AL, minibatch_Y, caches, lambd)
            
            # Update parameters.
            parameters = update_parameters(parameters, grads, (0.95**i)*learning_rate)
            
        # Compute cost.
        if monitor_training_cost:
            cost = compute_cost(train_x, train_y, parameters, lambd)
            print ("Cost after iteration %i: %f" % (i, cost))
            train_cost.append(cost)
        if monitor_evaluation_cost:
            cost = compute_cost(test_x, test_y, parameters, lambd)
            evaluation_cost.append(cost)
            #print ("Cost after iteration %i: %f" % (i, cost))
    
    # plot the cost
    plt.figure()
    p1, = plt.plot(np.squeeze(train_cost))
    p2, = plt.plot(np.squeeze(evaluation_cost))
    plt.legend([p1, p2], ["train", "evaluation"], loc = 'best')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

# Train the training set
# lr = 0.1, ms = 32, lm = 0.01 epoch = 30
parameters = L_layer_model(training_data, evaluation_data, layer_dims, learning_rate = 0.1, 
                            minibatch_size = 32, lambd = 0.015, epoch = 30,
                            monitor_training_cost = True, monitor_evaluation_cost = True)
# Save the network's parameters
save("model/myNetwork.py", layer_dims, parameters)

# Predict on training set and test set
print("For training set:")
pred_train, true_train = predict(train_x, train_y, parameters)
print(pred_train[0:10])
print(true_train[0:10])
print("For test set:")
pred_test, true_test = predict(test_x, test_y, parameters)
print(pred_test[0:10])
print(true_test[0:10])

print_mislabeled_images(test_x, pred_test, true_test)
