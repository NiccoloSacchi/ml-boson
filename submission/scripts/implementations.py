# -*- coding: utf-8 -*-
""" Machine learning functions """

import numpy as np
import matplotlib.pyplot as plt
from .proj1_helpers import predict_labels
from types import SimpleNamespace 


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=1, plot_losses=False, print_output=False, ouptut_step=1, costfunc=CostFunction.MSE)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    num_batches = tx.shape[0]
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=num_batches, plot_losses=False, print_output=False, ouptut_step=1, costfunc=CostFunction.MSE)


def least_squares(y, tx):
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    return compute_loss(y, tx, w, costfunc=CostFunction.MSE), w


def ridge_regression(y, tx, lambda_):
    A = tx.T @ tx + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1]) # DxD
    b = tx.T @ y # Dx1
    try:
        w = np.linalg.solve(A, b)
        return compute_loss(y, tx, w, costfunc=CostFunction.MSE), w
    except Exception as e:
        print("When solving the system in ridge regression: " + str(e))
        return -1, np.zeros(tx.shape[1])

    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=1, plot_losses=False, print_output=True, ouptut_step=50, costfunc=CostFunction.LIKELIHOOD)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=lambda_, num_batches=1, plot_losses=False, print_output=False, ouptut_step=100, costfunc=CostFunction.LIKELIHOOD)
    
    
#### BUILD TX
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function return the matrix formed by applying the polynomial basis to the input data
    augm = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        augm = np.column_stack((augm, x ** deg))
    return augm

#### COMPUTE ERROR AND COST FUNCTION

# define the possible cost functions
class CostFunction(): 
    MSE = 1
    RMSE = 2
    MAE = 3
    LIKELIHOOD = 4 
    SUCCESS_RATIO = 5

def logistic_func(z):
    """ Logistic function used to map y to [0, 1]. f(z) = 1/(1+e^-z) """
    return 1 / (1 + np.exp(-z))

# use MSE by default
def compute_loss(y, tx, w, lambda_=0, costfunc=CostFunction.MSE):
    """ Compute the cost L(w) from scratch and depending on the chosen cost function. """
    
    if costfunc is CostFunction.MSE:
        return np.squeeze(compute_loss_with_error(compute_error(y, tx, w), CostFunction.MSE))
    
    if costfunc is CostFunction.RMSE:
        return np.sqrt(2*compute_loss(y, tx, w, costfunc=CostFunction.MSE))
    
    if costfunc is CostFunction.MAE:
        return compute_loss_with_error(compute_error(y, tx, w), CostFunction.MAE)
    
    if costfunc is CostFunction.LIKELIHOOD:
        # if lambda_ != 0 then we copute the penalized one
        prob = logistic_func(tx @ w)
        log_likelihood = np.squeeze((y.T @ np.log(prob) + (1 - y).T @ np.log(1 - prob)))
        return -log_likelihood + lambda_ * np.squeeze(w.T @ w)
    
    if costfunc is CostFunction.SUCCESS_RATIO:
        # Given the weigths and a test set it compute the prediction for every input
        # and returns the ratio of correct predictions.
        # This cost function is discrete and therefore not differentiable.
        # Can be used only to evaluate the final model or for a grid search. 
        pred = predict_labels(w, tx)
        y_ = y.copy()
        y_[y_==0] = -1

        num_correct = np.sum(pred==y_)
        return num_correct/len(y)
    
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

def compute_error(y, tx, w): 
    """ Compute the error e=y-X.T*w """
    # the error is independent from the used cost function 
    return y - tx @ w

####

#### TECHNIQUES TO OPTIMIZE THE COST FUNCTION

## GRADIENT DESCENT
def compute_gradient(y, tx, w, costfunc=CostFunction.MSE):
    """ Compute the gradient (derivative of L(w) dimensions) from scratch. 
    N.B. To be used only with a differentiable cost function, e.g. with MSE, not with MAE. """
    
    if costfunc is CostFunction.MSE:
        return compute_gradient_with_e(tx, compute_error(y, tx, w))
    
    if costfunc is CostFunction.LIKELIHOOD:
        #print(tx.shape, w.shape, y.shape) log_func = np.vectorize(logistic_func)
        return tx.T @ (logistic_func(tx @ w)- y)
    
    return "Error, cost function not recognized"

def compute_gradient_with_e(tx, e):
    """ Compute the gradient (derivative of L(w) dimensions) from X and error. 
    N.B. To be used only with a differentiable cost function, e.g. with MSE, not with MAE. """
    return - tx.T @ e / len(e)

def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=1, plot_losses=True, print_output=True, ouptut_step=100, costfunc=CostFunction.MSE):
    """ w(t+1) = w(t)-gamma*gradient(L(w)) where L(w) is the chosen cost function in {CostFunction.MSE, CostFunction.LIKELIHOOD}. Each iteration can be done on a subset of variables depending on the given num_batches: if batch_size=N then the the dataset will be randomly splitted in N parts that will be used for the next N iterations, the
    process reiterates until max_iters is reached. lambda_ is the parameter for the penalized logistic regression """
    
    # make sure w is of the correct shape
    initial_w = initial_w.reshape((-1, 1))
    y = y.reshape((-1, 1))
    
    # if costfunc = LIKELIHOOD the y should be made only of 1s and 0s.
    if costfunc == CostFunction.LIKELIHOOD:
        y[y==-1] = 0
    
    # if the batch_size has not been set then do gradient_descent
    if num_batches < 0: 
        print("num_batches must be positive")
        return
                
    if plot_losses:
        n_cols = 2
        n_rows = 1
        width =  n_cols*4
        heigth = n_rows*4
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(width, heigth)
    
        axs[0].scatter(-1,  compute_loss(y, tx, initial_w, lambda_, costfunc), color='red', s=10)
        axs[1].scatter(-1,  compute_loss(y, tx, initial_w, costfunc=CostFunction.SUCCESS_RATIO), color='blue', s=10)
        for ax in axs:
            ax.grid()
            ax.set_xlabel('iteration n')
        axs[0].set_title('Loss')
        axs[0].set_ylabel('loss')
        
        axs[1].set_title('Prediction ratio')        
        axs[1].set_ylabel('ratio')
        axs[1].set_ylim([0.5, 1])
        

    w = initial_w
    batch_size = int(y.shape[0]/num_batches)
    n_iter = 0
    while n_iter < max_iters:
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=num_batches):
            if n_iter >= max_iters:
                break;
            # Compute gradient and loss at the current step. 
            # The latter will be used just to check if the algorithm is converging
            
            # compute next w
            g = compute_gradient(y_batch, tx_batch, w, costfunc=costfunc) # relative to only the batch
            w = w - gamma*g  

            if n_iter % ouptut_step == 0 or n_iter==max_iters-1:
                curr_loss = compute_loss(y, tx, w, lambda_, costfunc=costfunc)

                if print_output:
                    print("Gradient Descent({bi}/{ti}): loss={l}".format(
                        bi=n_iter, ti=max_iters - 1, l=curr_loss))

                if plot_losses:
                    succ_ratio = compute_loss(y, tx, w, costfunc=CostFunction.SUCCESS_RATIO)
                    axs[0].scatter(n_iter, curr_loss, color='red', s=10)
                    axs[1].scatter(n_iter, succ_ratio, color='blue', s=10)
            
            n_iter += 1
                    
    if plot_losses:
        plt.tight_layout()
        plt.savefig("gradient descent")
        plt.show()
        
    return compute_loss(y, tx, w, lambda_, costfunc=costfunc), np.array(w)

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


####

# TOOLS TEST THE MODEL    
    
def build_k_indices_(y, num_sets):
    """ Build k groups of indices for k-fold."""
    indices = np.array(range(y.shape[0])) # number of data points
    set_size = int(y.shape[0] / num_sets) # size of the sets
    k_indices = [indices[k * set_size: (k + 1) * set_size] for k in range(num_sets)]
    return np.array(k_indices)

def build_k_indices(y, num_sets, seed):
    """ Build k groups of indices for k-fold."""
    N = y.shape[0] # number of data points
    set_size = int(N / num_sets) # size of the sets
    np.random.seed(seed)
    indices = np.random.permutation(N)
    k_indices = [indices[k * set_size: (k + 1) * set_size] for k in range(num_sets)]
    return np.array(k_indices)


# Given the hyper parameters of the function (degree, lambdas, ...) and a training set
# returns the ratio of correct predictions of the built model
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """ return the loss of ridge regression. """
    test = SimpleNamespace()
    test.y = y[k_indices[k]]
    test.x = x[k_indices[k]]
    test.tx = build_poly(test.x, degree)
    
    train = SimpleNamespace()
    train_indices = np.concatenate((k_indices[:k], k_indices[k+1:])).reshape(1, -1)[0]
    train.y = y[train_indices]
    train.x = x[train_indices]
    train.tx = build_poly(train.x, degree)
    
    _, w = ridge_regression(train.y, train.tx, lambda_)
    
    train.loss = compute_loss(train.y, train.tx, w, CostFunction.RMSE)
    test.loss = compute_loss(test.y, test.tx, w, CostFunction.RMSE)
    
    return train.loss, test.loss

# given the sets of indices and k returns the kth set (the test set)
# and collect the other k-1 sets in one set (the training set)
def get_kth_set(y, tx, k_indices, kth):
    test = SimpleNamespace()
    test.y = y[k_indices[kth]]
    test.tx = tx[k_indices[kth]]
    
    train = SimpleNamespace()
    train_indices = np.concatenate((k_indices[:kth], k_indices[kth+1:])).reshape(1, -1)[0]
    train.y = y[train_indices]
    train.tx = tx[train_indices]
    
    return train, test
