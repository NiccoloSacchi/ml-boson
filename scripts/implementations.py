# -*- coding: utf-8 -*-
""" Functions used to for project 1 - Machine Learning 2017 """

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import predict_labels
from types import SimpleNamespace 
import codecs, json 


def column_labels():
    """ Return the column names. Use column_labels().index("DER_sum_pt") to retrieve the corresponding index """
    return np.array([
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep", 
        "DER_mass_vis","DER_pt_h", 
        "DER_deltaeta_jet_jet", 
        "DER_mass_jet_jet", 
        "DER_prodeta_jet_jet", 
        "DER_deltar_tau_lep", 
        "DER_pt_tot",
        "DER_sum_pt", 
        "DER_pt_ratio_lep_tau", 
        "DER_met_phi_centrality", 
        "DER_lep_eta_centrality", 
        "PRI_tau_pt",
        "PRI_tau_eta", 
        "PRI_tau_phi",
        "PRI_lep_pt", 
        "PRI_lep_eta",
        "PRI_lep_phi", 
        "PRI_met", 
        "PRI_met_phi",
        "PRI_met_sumet", 
        "PRI_jet_num", 
        "PRI_jet_leading_pt", 
        "PRI_jet_leading_eta", 
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt", 
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi", 
        "PRI_jet_all_pt"])

def column_labels_map():
    """ Return the column names. Use column_labels().index("DER_sum_pt") to retrieve the corresponding index """
    return {feature: index for index, feature in enumerate(column_labels())}
    

#### DATA ANALYSIS FUNCTIONS

def plot_features(x, col_labels = column_labels(), title="occurrencies"):
    """ Plot the features after dropping the -999 values. Can be used to find outliers. """
        
    if (x.shape[1] != len(col_labels)):
        print("You should pass the correct column labels")
        col_labels = range(x.shape[1])
    
    n_cols = 3
    n_rows = int(np.ceil(len(col_labels)/3))
    fig, ax = plt.subplots(n_rows, n_cols)
    
    width =  n_cols*5
    heigth = n_rows*4
    fig.set_size_inches(width, heigth)
    
    x_axis = range(x.shape[0])
    
    for feature in range(x.shape[1]):
        ax_curr = ax[int(feature/n_cols)][feature%n_cols] 
        
        feature_vals = x[:, feature]
        
        ind999 = feature_vals==-999
        ax_curr.scatter(np.where(ind999)[0], feature_vals[ind999]*0, color="blue", s=4)
        ax_curr.scatter(np.where(~ind999)[0], feature_vals[~ind999], color="red", s=4)        
        
        ax_curr.grid()
        ax_curr.legend(["-999 values", "raw values"])
        ax_curr.set_title(str(feature) + ': ' + str(col_labels[feature]))
        ax_curr.set_ylabel("values")
        ax_curr.set_xlabel("rows")

    plt.tight_layout()
    plt.savefig(title)
    plt.show()
    
    
def plot_distributions(x, y, col_labels = column_labels(), title="distributions", normed=False):
    if (x.shape[1] != len(col_labels)):
        print("You should pass the correct column labels")
        col_labels = range(x.shape[1])
    
    n_cols = 2
    n_rows = len(col_labels)
    fig, ax = plt.subplots(n_rows, n_cols)
    
    width =  n_cols*5
    heigth = n_rows*4
    fig.set_size_inches(width, heigth)
    
    # get indices relative to y = 1
    indices1 = np.where(y == 1)[0]
    # get indices relative to y = -1
    indices0 = np.where(y == -1)[0]
        
    for feature in range(x.shape[1]):
        ax_row = ax[feature] 
               
        # plot the feature relative to y = 1
        feature1 = x[indices1, feature]
        # plot histogram
        n, bins, patches = \
            ax_row[0].hist(feature1, histtype='step', bins=int(len(feature1)/1000), color="blue", normed=normed)
        # plot distribution
        y = plt.mlab.normpdf(bins, np.mean(feature1), np.std(feature1))  
        ax_row[1].plot(bins, y, 'b--', linewidth=1)
        
        feature0 = x[indices0, feature]
        # plot histogram
        n, bins, patches = \
            ax_row[0].hist(feature0, histtype='step', bins=int(len(feature0)/1000), color="red", normed=normed)
        # plot distribution
        y = plt.mlab.normpdf(bins, np.mean(feature0), np.std(feature0))
        ax_row[1].plot(bins, y, 'r--', linewidth=1)
        
        ylabels = ["observed frequencies", "distribution"]
        if normed:
            ylabels[0] = "normed " + ylabels[0] 
        for i in range(n_cols):
            ax_row[i].grid()
            ax_row[i].legend(["y = 1", "y = -1"])
            ax_row[i].set_title(str(feature) + ": " + col_labels[feature])
            ax_row[i].set_ylabel(ylabels[i])
            ax_row[i].set_xlabel("feature values")

    plt.tight_layout()
    plt.savefig(title)
    plt.show()

####    
    
#### DATA CLEANING FUNCTIONS 

def clean_input_data(dataset_all, output_all=np.array([]), corr=1, dimension_expansion=False, bool_col=False):
    """ Cleans the data:
    1. Standardizes.
    2. Split using the jet number
    if output_all is != None that it splits it with the input
    """
    
    # 1. standardize
    dataset_all = standardize_final(dataset_all)
    
    # 2. split the dataset
    datasets, outputs = split_input_data(dataset_all, output_all, corr=corr)
            
    # 3. fill the nans with 0 and append a boolean column
    # fill nan in the first column with 0s
    for i in range(len(datasets)):
        nan_rows = np.isnan(datasets[i][:,0])
        datasets[i][nan_rows, 0] = 0
        # also append a boolean column to indicate the position of those -999 since 
        # this information may still be usefull for determining the y
        if bool_col:
            datasets[i] = np.column_stack((datasets[i], nan_rows))
       
    if dimension_expansion:
        # 4. dimension expansion
        file_path = "percentiles.json"
        obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
        percentiles = json.loads(obj_text)

        for jet, curr_dataset in enumerate(datasets):
            for col in range(curr_dataset.shape[1]-1): # scan columns (not the boolean one)
                c = curr_dataset[:, col].copy() # column to be splitted
                col_perc = percentiles[str(jet)][str(col)]
                for perc in [col_perc["25"], col_perc["50"], col_perc["100"]]:
                    vals = (c<=perc)
                    #if not np.all(c1==c1[0]):
                    curr_dataset = np.column_stack((curr_dataset, vals*c))
                    c = c*(~vals)
                # delete the splitted col
                datasets[jet] = np.delete(curr_dataset, col, axis=1)
            
    return datasets, outputs

def split_input_data(dataset_all, output_all=np.array([]), corr=1):
    """ This function will separate the input dataset into 4 datasets depending
    on the value in the categorical column PRI_jet_num (column 22):
    (1) one dataset for jet num = 0, 
    (2) one dataset for jet num = 1, 
    (3) one dataset for jet num = 2 
    (4) one dataset for jet num = 3.
    We drop, for each obtained dataset, the columns with only -999 values and substitute
    with np.nan the -999 values in the first column. Moreover, you can pass a corr != 1 to
    drop decide the minimum correlation that you want between your columns. corr must be 
    in the set {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1}.
    """
    
    def get_with_jet(dataset, output_all, jet_num): # given jet and dataset return the rows with th egiven jet number
        rows = dataset[:, 22]==jet_num
        if output_all.size != 0:
            return dataset[rows, :], output_all[rows]
        else:
            return dataset[rows, :], np.array([])
    
    num_jets = 4
    datasets = [None]*num_jets
    outputs = [None]*num_jets
    for jet in range(num_jets):
        curr_dataset, curr_output = get_with_jet(dataset_all, output_all, jet)
        # drop columns depending on the jet, drop always the PRI_jet_num (column 22)
        if jet == 0:
            to_drop = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29] # 22 and 29 contains only 0s, the others only -999
        elif jet == 1:
            to_drop = [4, 5, 6, 12, 22, 26, 27, 28]
        else:
            to_drop = [22]
        
        to_drop = to_drop + correlated(corr)
        print("Jet", jet, "columns dropped:", to_drop)
        curr_dataset = np.delete(curr_dataset, to_drop, axis=1)
        
        datasets[jet] = curr_dataset
        outputs[jet] = curr_output

    return datasets, outputs

# helper function that, given a minimum correlation, return the list of columns that can be dropped 
def correlated(corr = 0.8):
    """ Drop the correlated columns. This step may not improve the success ratio of the model
    but it will simplify it. corr must be in {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1}"""
    # corr_map maps the minimum correlation to the list of columns that can be droppped
    return {
        0.7: [3, 4, 6, 7, 15, 16, 18, 21, 22, 25, 28],
        0.75: [3, 4, 6, 16, 21, 22, 25, 28],
        0.8: [4, 6, 9, 18, 21, 22, 25],
        0.85: [3, 6, 18, 21, 22, 28],
        0.9: [3, 6, 9, 21, 28],
        0.95: [5, 21, 28],
        1: []
    }[corr]

def standardize_final(data):
    data[data==-999] = np.nan
    data = data - whole_data_means()
    data = data / whole_data_std_devs()
    return data

def whole_data_means():
    return [  1.21867697e+02,   4.92532558e+01,   8.11405610e+01,
          5.78582923e+01,   2.40500975e+00,   3.72181050e+02,
         -8.29392140e-01,   2.37387138e+00,   1.89724461e+01,
          1.58596159e+02,   1.43877554e+00,  -1.27303822e-01,
          5.85210606e-01,   3.86981522e+01,  -1.16628927e-02,
         -1.31600006e-02,   4.66924138e+01,  -1.90822693e-02,
          4.94674822e-02,   4.16545265e+01,  -8.63571168e-03,
          2.09908730e+02,   0,   8.49042850e+01, #previously 1.63345672e+00
         -1.24824003e-03,  -1.88594668e-02,   5.78102860e+01,
         -6.67041978e-03,  -1.04712859e-02,   1.22028164e+02]

def whole_data_std_devs():
    return [  5.69424463e+01,   3.53784047e+01,   4.05826830e+01,
          6.34127052e+01,   1.74241673e+00,   3.98234556e+02,
          3.58509472e+00,   7.80874852e-01,   2.19188873e+01,
          1.16089739e+02,   8.45108795e-01,   1.19435749e+00,
          3.58011984e-01,   2.24290027e+01,   1.21351057e+00,
          1.81621078e+00,   2.21423233e+01,   1.26434229e+00,
          1.81522730e+00,   3.24960932e+01,   1.81285268e+00,
          1.26816608e+02,   1,                6.06494678e+01,#previously 7.27633897e-01
          1.77958396e+00,   1.81550712e+00,   3.24553977e+01,
          2.03190029e+00,   1.81616637e+00,   1.00796644e+02]

# -nan_values: a list of invalid values. the whole list is for all of the columns.
# returns x where all the entries that where in the list nan_values are now np.nan
def fill_with_nan_list(x, nan_values=[]):
    """ Deals with the nan values: identify and substitute them.
    Run this function before the 'standardize' one. """
    x = x.astype(float) 
    nrows, ncols = x.shape
    
    for col_id in range(ncols): # scan columns
        for row_id in range(nrows): # put nan in this column where needed
            if x[row_id][col_id] in nan_values:
                x[row_id][col_id] = np.nan
    return x 

# substitutions: array of substitutions. len(substitutions) = #columns in x
# the nan values of column i are substituted with substitutions[i]
def sustitute_nans(x, substitutions=[]):
    nrows, ncols = x.shape
    mask = np.isnan(x)
    for col_id in range(ncols): # scan columns
        x[:, col_id][mask[:, col_id]] = substitutions[col_id]
    return x

def move_outliers(x):
    print("Managing the outliers")
    # DER_mass_MMC
    x[x[:, 0] > 700] = 700
    
    #DER_mass_transverse_met_lep
    x[x[:, 1] > 300] = 300
    
    #DER_mas_vis
    x[x[:, 2] > 500] = 500
    
    #DER_pt_h
    x[x[:, 3] > 600] = 600
    
    #DER_mass_jet_jet
    x[x[:, 5] > 2800] = 2800
    
    # DER_pt_tot
    x[x[:, 8] > 400] = 400
    
    # DER_sum_pt
    x[x[:, 9] > 1000] = 1000
    
    # DER_pt_ratio_lep_tau
    x[x[:, 10] > 10] = 10
    
    # PRI_tau_pt
    x[x[:, 13] > 300] = 300
    
    # PRI_lep_pt
    x[x[:, 16] > 250] = 250
    
    # PRI_met
    x[x[:, 19] > 450] = 450
    
    # PRI_met_sumet
    x[x[:, 21] > 1100] = 1100
    
    # PRI_jet_leading_pt
    x[x[:, 23] > 500] = 500
    
    # PRI_jet_subleading_pt
    x[x[:, 26] > 250] = 250
    
    # PRI_jet_all_pt
    x[x[:, 29] > 800] = 800
    
    return x

def standardize(x):
    """Standardize the original data set."""
    # compute mean and std ignoring nan values
    mean_x = np.nanmean(x, axis=0)
    x = x - mean_x
    std_x = np.nanstd(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x   

####
 
    
#### BUILD TX

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function return the matrix formed by applying the polynomial basis to the input data
    augm = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        augm = np.column_stack((augm, x ** deg))
    return augm

####

#### COMPUTE ERROR AND COST FUNCTION

# define the possible cost functions
class CostFunction(): 
    MSE = 1
    RMSE = 2
    MAE = 3
    LIKELIHOOD = 4 
    SUCCESS_RATIO = 5

def compute_error(y, tx, w): 
    """ Compute the error e=y-X.T*w """
    # the error is independent from the used cost function 
    return y - tx @ w

def logistic_func(z):
    """ Logistic function used to map y to [0, 1]. f(z) = 1/(1+e^-z) """
    return 1 / (1 + np.exp(-z))

# use MSE by default
def compute_loss(y, tx, w, lambda_=0, costfunc=CostFunction.MSE):
    """ Compute the cost L(w) from scratch and depending on the chosen cost function. """
    
    if costfunc is CostFunction.MSE:
        return compute_loss_with_error(compute_error(y, tx, w), CostFunction.MSE)
    
    if costfunc is CostFunction.RMSE:
        return np.sqrt(2*compute_loss(y, tx, w, costfunc=CostFunction.MSE))
    
    if costfunc is CostFunction.MAE:
        return compute_loss_with_error(compute_error(y, tx, w), CostFunction.MAE)
    
    if costfunc is CostFunction.LIKELIHOOD:
        # if lambda_ != 0 then we copute the penalized one
        prob = logistic_func(tx @ w)
        log_likelihood = np.squeeze((y.T @ np.log(prob) + (1 - y).T @ np.log(1 - prob)))
        return -log_likelihood + lambda_ * np.squeeze(w.T.dot(w))
    
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

####

#### TECHNIQUES TO OPTIMIZE THE COST FUNCTION

## GRADIENT DESCENT
def compute_gradient(y, tx, w, costfunc=CostFunction.MSE):
    """ Compute the gradient (derivative of L(w) dimensions) from scratch. 
    N.B. To be used only with a differentiable cost function, e.g. with MSE, not with MAE. """
    
    if costfunc is CostFunction.MSE:
        return compute_gradient_with_e(tx, compute_error(y, tx, w))
    
    if costfunc is CostFunction.LIKELIHOOD:
        return tx.T @ (logistic_func(tx @ w)- y)
    
    return "Error, cost function not recognized"

def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=1, plot_losses=True, print_output=True, ouptut_step=100, costfunc=CostFunction.MSE):
    """ w(t+1) = w(t)-gamma*gradient(L(w)) where L(w) is the chosen cost function in {CostFunction.MSE, CostFunction.LIKELIHOOD}. Each iteration can be done on a subset of variables depending on the given num_batches: if batch_size=N then the the dataset will be randomly splitted in N parts that will be used for the next N iterations, the
    process reiterates until max_iters is reached. lambda_ is the parameter for the penalized logistic regression """
    
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
        #axs[0].set_yscale('log')
        axs[0].set_ylabel('loss')
        
        axs[1].set_title('Prediction ratio')        
        axs[1].set_ylabel('ratio')
        

    w = initial_w
    loss_min = float('inf') 
    w_best = [] 
    batch_size = int(y.shape[0]/num_batches)
    n_iter = 0
    while n_iter < max_iters:
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=num_batches):
            # Compute gradient and loss at the current step. 
            # The latter will be used just to check if the algorithm is converging
            
            # compute next w
            g = compute_gradient(y_batch, tx_batch, w, costfunc=costfunc) # relative to only the batch
            w = w - gamma*g  

            # compute the loss L(w)
            curr_loss = compute_loss(y_batch, tx_batch, w, costfunc=costfunc) # relative to all the points
            if curr_loss < loss_min:
                loss_min = curr_loss
                w_best = w
            
            if n_iter % ouptut_step == 0:
                curr_loss = compute_loss(y, tx, w_best, lambda_, costfunc=costfunc)
                succ_ratio = compute_loss(y, tx, w_best, costfunc=CostFunction.SUCCESS_RATIO)
                if print_output:
                    print("Gradient Descent({bi}/{ti}): loss={l}, prediction ratio={succ_ratio}".format(
                        bi=n_iter, ti=max_iters - 1, l=curr_loss, succ_ratio=succ_ratio))

                if plot_losses:
                    axs[0].scatter(n_iter, curr_loss, color='red', s=10)
                    axs[1].scatter(n_iter, succ_ratio, color='blue', s=10)
            
            n_iter += 1
        
    if n_iter-1 % ouptut_step != 0:
        curr_loss = compute_loss(y, tx, w_best, lambda_, costfunc=costfunc)
        succ_ratio = compute_loss(y, tx, w_best, costfunc=CostFunction.SUCCESS_RATIO)
        if print_output:
            print("Gradient Descent({bi}/{ti}): loss={l}, prediction ratio={succ_ratio}".format(
                bi=n_iter-1, ti=max_iters - 1, l=curr_loss, succ_ratio=succ_ratio))

        if plot_losses:
            axs[0].scatter(n_iter, curr_loss, color='red', s=10)
            axs[1].scatter(n_iter, succ_ratio, color='blue', s=10)
                    
    if plot_losses:
        plt.tight_layout()
        plt.savefig("gradient descent")
        plt.show()
    
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

##

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
    try:
        w = np.linalg.solve(A, b)
        return compute_loss(y, tx, w, costfunc=CostFunction.MSE), w
    except Exception as e:
        print("When solving the system in ridge regression: " + str(e))
        return -1, np.zeros(tx.shape[1])

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


# PLOTS
# ratio_tr: matrix of training ratios of correct predictions (one row per degree, one column per lambda value)
# ratio_te: matrix of test ratios of correct predictions     (one row per degree, one column per lambda value)
# degree_list: list of degrees tried, will plot one figure per degree
# x_axis: values for the x axis
# log_axis_x: if lambdas was generated with np.logspace you may want to represent the x axis with a logarithmic scale
def cross_validation_visualization(ratio_tr, ratio_te, degree_list, x_axis, x_label="x label", log_axis_x=False):
    # ratio_tr is a matrix #figure x #points (it has one list per figure with the y-coord of the points)
    plt.cla()
    plt.clf()

    if log_axis_x == True:
       #curr_ax.semilogx() does not plot anything with this one...
        x_axis = np.log(x_axis)
            
    nfigures = len(degree_list)
    
    n_cols = 3
    n_rows = int(np.ceil(nfigures/n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)
    
    width =  n_cols*4
    heigth = n_rows*4
    fig.set_size_inches(width, heigth)

    for row in range(n_rows):
        for col in range(n_cols):
            cur_figure = row*n_cols + col
            if cur_figure < nfigures:
                if n_rows > 1:
                    curr_ax = ax[row][col]
                else:
                    curr_ax = ax[col]

                curr_ax.scatter(x_axis, ratio_tr[cur_figure], color='red', s=12)
                curr_ax.scatter(x_axis, ratio_te[cur_figure], color='blue', s=12)
                
                curr_ax.grid()
                curr_ax.legend(['training', 'test'])
                curr_ax.set_ylim([0, 1])
                curr_ax.set_title("Degree: " + str(degree_list[cur_figure]))
                curr_ax.set_ylabel("Ratio of correct predictions")
                curr_ax.set_xlabel(x_label)
                #for tick in ax[row][col].get_xticklabels():
                #    tick.set_rotation(45)

    plt.tight_layout()
    # plt.savefig("visualize_polynomial_regression")
    plt.show()
    

# ratios: list of matrix of ratios of correct predictions 
    # one matrix per model (if you have just one model you pass an array with only that matrix)
    # one row per degree
    # one column per hyperparameter (e.g. lambda) value tried
    # every cell represent the ratios of success corresponding to the model with a certain degree and a certain
    # value of an hyperparameter  
# degree_list: list of degrees tried, will plot one figure per degree
# x_axis: values for the x axis
# log_axis_x: if lambdas was generated with np.logspace you may want to represent the x axis with a logarithmic scale
def ratios_visualization(ratios, degree_list, x_axis, x_label="x label", log_axis_x=False, save_figure_with_name=""):
    # ratio_tr is a matrix #figure x #points (it has one list per figure with the y-coord of the points)
    plt.cla()
    plt.clf()

    if len(ratios)>10:
        print("Pass less that 10 models")
        return
    
    if log_axis_x == True:
       #curr_ax.semilogx() does not plot anything with this one...
        x_axis = np.log(x_axis)
            
    nmodels = len(ratios)
    # one color per model 
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:nmodels] 
              
    nfigures = len(degree_list)
    
    n_cols = 3
    n_rows = int(np.ceil(nfigures/n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)
    
    width =  n_cols*4
    heigth = n_rows*4
    fig.set_size_inches(width, heigth)

    for row in range(n_rows):
        for col in range(n_cols):
            cur_figure = row*n_cols + col
            if cur_figure < nfigures:
                if n_rows > 1:
                    curr_ax = ax[row][col]
                else:
                    curr_ax = ax[col]

                for model in range(nmodels): # represent all the models in the same figure
                    curr_ax.scatter(x_axis, ratios[model][cur_figure], color=colors[model], s=12)
                
                curr_ax.grid()
                curr_ax.legend(["model "+str(model) for model in range(nmodels)])
                curr_ax.set_ylim([0, 1])
                curr_ax.set_title("Degree: " + str(degree_list[cur_figure]))
                curr_ax.set_ylabel("Ratio of correct predictions")
                curr_ax.set_xlabel(x_label)
                #for tick in ax[row][col].get_xticklabels():
                #    tick.set_rotation(45)
                


    plt.tight_layout()
    if save_figure_with_name != "":
        plt.savefig(save_figure_with_name)
    plt.show()
