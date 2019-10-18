# load mnist_loader.py
"""
mnist_loader

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
    f.close()
    
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs =  np.array(tr_d[0])
    training_results = np.array([vectorized_result(y) for y in tr_d[1]])
    validation_inputs =  np.array(va_d[0])
    validation_results = np.array([vectorized_result(y) for y in va_d[1]])
    test_inputs = np.array(te_d[0])
    test_results = np.array([vectorized_result(y) for y in te_d[1]])
    
    return training_inputs, training_results, test_inputs, test_results

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
