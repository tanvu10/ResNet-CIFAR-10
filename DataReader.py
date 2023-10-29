import os
import pickle
import numpy as np
from ImageUtils import parse_record
""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # load training batch
    path = data_dir + 'data_batch_'
    for i in range(1, 6):
        batch_data = unpickle(path + f'{i}')
        if i == 1:
            x_train = batch_data[b'data']
            y_train = batch_data[b'labels']
        else:
            x_train = np.vstack((x_train, batch_data[b'data']))
            y_train += batch_data[b'labels']

    # Load test batch
    path = data_dir + 'test_batch'
    test_data = unpickle(path)
    x_test = test_data[b'data']
    y_test = test_data[b'labels']

    # Convert to correct data type format:
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    
    ### YOUR CODE HERE
    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

if __name__ == '__main__':
    # debugging
    x_train, y_train, x_test, y_test = load_data("/Users/vuh/Downloads/cifar-10-batches-py/")
    x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train, split_index=45000)
    test = parse_record(x_train_new[1,:], True)
    print()