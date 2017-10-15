# -*- coding: utf-8 -*-
""" Functions used to for project 1 - Machine Learning 2017 """

import numpy as np
import matplotlib.pyplot as plt

# STANDARDIZE X
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

# BUILD TX
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function return the matrix formed by applying the polynomial basis to the input data
    augm = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        augm = np.column_stack((augm, x ** deg))
    return augm


# COMPUTE ERROR AND COST FUNCTION

# define the possible cost functions
class CostFunction(): 
    MSE = 1
    RMSE = 2
    MAE = 3
    PROB = 4 #  probabilistical cost function, better for labelling   

def compute_error(y, tx, w): 
    """ Compute the error e=y-X.T*w """
    # the error is independent from the used cost function 
    return y - tx @ w

# logistic function used to map y to [0, 1]
def logistic_func(z):
    ez = np.exp(z)
    return ez/(1+ez)

# use MSE by default
def compute_loss(y, tx, w, costfunc=CostFunction.MSE):
    """ Compute the cost L(w) from scratch and depending on the chosen cost function """
    if costfunc is CostFunction.MSE:
        return compute_loss_with_error(compute_error(y, tx, w), CostFunction.MSE)
    if costfunc is CostFunction.RMSE:
        return np.sqrt(2*compute_loss(y, tx, w, costfunc=CostFunction.MSE))
    if costfunc is CostFunction.MAE:
        return compute_loss_with_error(compute_error(y, tx, w), CostFunction.MAE)
    if costfunc is CostFunction.PROB:
        txw = tx@w
        return (np.log(1+np.exp(txw))-y*txw).sum()
    return "Error, cost function not recognized"
    
def compute_loss_with_error(e, costfunc=CostFunction.MSE):
    """ Compute the cost L(w) from the given error and depending on the chosen cost function """
    if costfunc is CostFunction.MSE:
        return e.T @ e/(2*len(e))
    if costfunc is CostFunction.RMSE:
        return np.sqrt(2*compute_loss_with_error(e, costfunc=CostFunction.MSE))
    if costfunc is CostFunction.MAE:
        return np.mean(np.abs(e))
    return "Error, cost function not recognized"

# TECHNIQUES TO OPTIMIZE THE COST FUNCTION
""" Grid search """
# todo: delete? or maybe can be used to test the other works
def grid_search(y, tx, costfunc=CostFunction.MSE, num_intervals=10, w0_interval=[-150, 150], w1_interval=[-150, 150]):
    """Algorithm for grid search. Returns the weights relative to the lowest loss found. """
    
    # TODO:
    # 1. plot all the losses
    
    w0 = np.linspace(w0_interval[0], w0_interval[1], num_intervals)
    w1 = np.linspace(w1_interval[0], w1_interval[1], num_intervals)

    loss_min = float('inf') # here we store the minimum loss
    w_best = [] # here we store the weight relative to the lowest loss found
    for w_0 in w0:
        for w_1 in w1:
            curr_loss = compute_loss(y, tx, np.array([w_0, w_1]), costfunc)
            if curr_loss < loss_min:
                loss_min = curr_loss
                w_best = [w_0, w_1]
    # return the loss relative to the best solution found
    return loss_min, np.array(w_best)

""" Gradient computation """
def compute_gradient(y, tx, w, costfunc=CostFunction.MSE):
    """ Compute the gradient (derivative of L(w) dimensions) from scratch. 
    N.B. To be used only with a differentiable cost function, e.g. with MSE, not with MAE. """
    if costfunc is CostFunction.MSE:
        return compute_gradient_with_e(tx, compute_error(y, tx, w))
    if costfunc is CostFunction.PROB:
        logistic_func_ = np.vectorize(logistic_func) # so to apply the function element-wise to tx@w
        return (tx.T@(logistic_func_(tx@w).reshape(-1, 1)-y))[:, 0]
    return "Error, cost function not recognized"

def compute_gradient_with_e(tx, e):
    """ Compute the gradient (derivative of L(w) dimensions) from X and error. 
    N.B. To be used only with a differentiable cost function, e.g. with MSE, not with MAE. """
    return - tx.T @ e / len(e)

""" TODO delete
"" Gradient descent. ""
def gradient_descent(y, tx, initial_w, max_iters, gamma, print_output=True, plot_losses=True):
    "" w(t+1) = w(t)-gamma*gradient(L(w)). Can not be used with the non-differentiable MAE.  ""
    
    # TODO 
    # 1. implement a decreasing gamma
    
    if plot_losses:
        plt.grid()
        plt.yscale('log')
        plt.ylabel('log(mse loss)')
        plt.xlabel('iteration n')
        plt.scatter(-1,  compute_loss(y, tx, initial_w, costfunc=CostFunction.MSE), color='red')
        
    # Define parameters to store w and loss
    w = initial_w
    loss_min = float('inf') # here we store the minimum loss
    w_best = [] # here we store the weight relative to the lowest loss found
    for n_iter in range(max_iters):
        # Compute gradient and loss at the current step. 
        # The latter will be used just to check if the algorithm is converging

        # compute next w
        g = compute_gradient(y, tx, w)        
        w = w - gamma*g

        # compute the loss L(w)
        curr_loss = compute_loss(y, tx, w, costfunc=CostFunction.MSE)  

        if curr_loss < loss_min:
            loss_min = curr_loss
            w_best = w

        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=curr_loss, w0=w[0], w1=w[1]))

        if plot_losses:
            plt.scatter(n_iter, curr_loss, color='red') # check the losses are strictly decreasing
    return loss_min, np.array(w_best)
"""


""" Gradient descent and Stochastic gradient descent """
def gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size=-1, print_output=True, plot_losses=True, costfunc=CostFunction.MSE):
    """ w(t+1) = w(t)-gamma*gradient(L(w)) where L(w) can be computed on a subset of variables depending on batch_size.
    If a different batch_size is not passed then L(w) is computed using all the points.
    Can not be used with the non-differentiable MAE.  """
    
    # TODO 
    # 1. implement a decreasing gamma
    
    # if the batch_size has not been set then do gradient_descent
    if batch_size < 0: 
        batch_size = tx.shape[0]
        plt.scatter(-1,  compute_loss(y, tx, initial_w, costfunc), color='red')

    if plot_losses:
        plt.grid()
        plt.title('Gradient descent: the loss should decrease')
        plt.yscale('log')
        plt.ylabel('log(loss)')
        plt.xlabel('iteration n')
        
    # Define parameters to store w and loss
    w = initial_w
    # store the minimum loss
    loss_min = float('inf') 
    # store the weight relative to the lowest loss found
    w_best = [] 
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # Compute gradient and loss at the current step. 
            # The latter will be used just to check if the algorithm is converging
            
            # compute next w
            g = compute_gradient(y_batch, tx_batch, w, costfunc) # relative to only the batch
            w = w - gamma*g  

            # compute the loss L(w)
            curr_loss = compute_loss(y, tx, w, costfunc) # relative to all the points
            
            if curr_loss < loss_min:
                loss_min = curr_loss
                w_best = w
            
            if print_output:
                print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                      bi=n_iter, ti=max_iters - 1, l=curr_loss, w0=w[0], w1=w[1]))

            if plot_losses:
                plt.scatter(n_iter, curr_loss, color='red') # check the losses are strictly decreasing
    
    return loss_min, np.array(w_best)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    # if the batch is actually as big as the data set then do nothing
    if batch_size == tx.shape[0]:
        yield y, tx
    else:
        # otherwise build a minibatch
        data_size = len(y)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y[shuffle_indices]
            shuffled_tx = tx[shuffle_indices]
        else:
            shuffled_y = y
            shuffled_tx = tx
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

""" Least squares """
def least_squares(y, tx):
    """ Compute the least squares solution."""
#     w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    # returns the optimal weights
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    return compute_loss(y, tx, w, costfunc=CostFunction.MSE), w

""" Ridge regression """
def ridge_regression(y, tx, lambda_):
    """ Implement the ridge regression """
    A = tx.T @ tx + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1]) # DxD
    b = tx.T @ y # Dx1
    w = np.linalg.solve(A, b)
    return compute_loss(y, tx, w, costfunc=CostFunction.MSE), w
