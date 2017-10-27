# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
from implementations import *
from proj1_helpers import *


def test_model_and_export(data_u,ids_u,w,degree):
    #u as "unknown"
    tx_u = build_poly(data_u, degree) 
    predictions = predict_labels(w,tx_u)

    export_data = np.vstack((ids_u, predictions)).T.astype(int)
    header = ["Id","Prediction"]
    export_data = np.vstack((header,export_data))
    
    np.savetxt('../dataset/export.csv', export_data, fmt='%s', delimiter=',')

def equalize_predictions(predictions,data):
    count_s = 0
    count_b = 0
    
    for pred in predictions:
        if pred==1:
            count_s += 1
        else:
            count_b +=1
    
    
    diff = count_s - count_b    
    original_size = data.shape[0]
    
    if diff<0: #if there are more background than signal, we remove data from background
        removed=0
        i = 0
        while removed < -diff and i< original_size+diff:
            if predictions[i] == -1:
                data=np.delete(data,i,0)
                predictions=np.delete(predictions,i,0)
                removed += 1
            i+= 1
    else: #else we remove signal data
        removed=0
        i = 0
        while removed < diff and i< original_size-diff:
            if predictions[i] == 1:
                data=np.delete(data,i,0)
                predictions=np.delete(predictions,i,0)
                removed+= 1
            i+= 1
        
    return predictions,data

# num_sets: number of sets in which the dataset will be splitted to run the cross validation
# degree_list: list of degree to be tried
# lambdas: list of lambdas to be tried 
def ridge_regression_tuning(x, y, degree_list, lambdas):    
    
    ratios = [] # matrix success ratio obtained with the training sets

    # define lists to store the loss of training data and test data
    for nfigure, degree in enumerate(degree_list): # one figure per degree
        tx = build_poly(x, degree) 
        
        # one row (figure) per degree
        ratios.append([])
        
        for npoint, lambda_ in enumerate(lambdas):
            # for each lambda we compute the expected ratio of success (this will be the a point in the figure)
            ratios[nfigure].append(0)
            
            # train the model, just line should change depending on the chosen training 
            _, w = ridge_regression(y, tx, lambda_)

            ratios[nfigure][npoint] = compute_loss(y, tx, w, costfunc=CostFunction.SUCCESS_RATIO)
            
    ratios = np.array(ratios)
    
    return ratios


# num_sets: number of sets in which the dataset will be splitted to run the cross validation
# degree_list: list of degree to be tried
# lambdas: list of lambdas to be tried 
def cross_validation_ridge_regression(x, y, num_K_sets, degree_list, lambdas):
    seed = 2
    
    # split indices in k sets
    k_indices = build_k_indices(y, num_K_sets, seed)
    
    ratio_tr = [] # matrix success ratio obtained with the training sets
    ratio_te = [] # matrix success ratio obtained with the test sets

    # define lists to store the loss of training data and test data
    for nfigure, degree in enumerate(degree_list): # one figure per degree
        tx = build_poly(x, degree) 
        
        # one row (figure) per degree
        ratio_tr.append([])
        ratio_te.append([])
        
        for npoint, lambda_ in enumerate(lambdas):
            # for each lambda we compute the expected ratio of success (this will be the a point in the figure)
            ratio_tr[nfigure].append(0)
            ratio_te[nfigure].append(0)
            
            for k_curr in range(num_K_sets):
                train, test = get_kth_set(y, tx, k_indices, k_curr)
                
                # train the model, just line should change depending on the chosen training 
                _, w = ridge_regression(train.y, train.tx, lambda_)
                
#                 # gradient descent
#                 initial_w = np.zeros(tx.shape[1])
#                 max_iters = 50
#                 gamma = lambda_
#                 _, w = gradient_descent(train.y, train.tx, initial_w, max_iters, gamma, batch_size=-1, print_output=False, plot_losses=False, costfunc=CostFunction.MSE)
                
                # compute how good is the model
                ratio_tr[nfigure][npoint] += compute_loss(train.y, train.tx, w, costfunc=CostFunction.SUCCESS_RATIO)
                ratio_te[nfigure][npoint] += compute_loss(test.y, test.tx, w, costfunc=CostFunction.SUCCESS_RATIO)
            
            # average the ratio obtained with the cross validation
            ratio_tr[nfigure][npoint] /= num_K_sets
            ratio_te[nfigure][npoint] /= num_K_sets
            
    ratio_tr = np.array(ratio_tr)
    ratio_te = np.array(ratio_te)
    
#     print(ratio_tr.shape) # #degree x #lambdas
#     print(ratio_te.shape)
#     print(degrees.shape)
#     print(lambdas.shape)
    
    return ratio_tr, ratio_te


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

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
