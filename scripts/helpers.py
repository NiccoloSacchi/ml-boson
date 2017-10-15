# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np

#if sub_sample = true, we only load the first 100 lines of each set
#for now only the first 100 lines version works
def load_boson_data(sub_sample=True):
    path_train_dataset = "train_100.csv"
    path_test_dataset = "train_100.csv"
    
    ids_tr = np.genfromtxt(
        path_train_dataset, delimiter=",", skip_header=1,usecols=0)
        
    predictions_tr = np.genfromtxt(
        path_train_dataset,delimiter=",",skip_header=1,usecols=1,dtype=None,
        converters={1: lambda x: 1 if b's' in x else -1})
    
    data_tr = np.genfromtxt(
        path_train_dataset, delimiter=",", skip_header=1,usecols=range(2,32))
    
    ids_te = np.genfromtxt(
        path_test_dataset, delimiter=",", skip_header=1,usecols=0)
        
    data_te = np.genfromtxt(
        path_test_dataset, delimiter=",", skip_header=1,usecols=range(2,32))
    
    return ids_tr,predictions_tr,data_tr, ids_te, data_te

def load_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454
    return height, weight, gender

def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]


def load_data_from_ex02(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender

def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
