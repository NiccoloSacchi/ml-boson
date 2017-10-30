# -*- coding: utf-8 -*-

from proj1_helpers import load_csv_data, create_csv_submission, predict_labels
import numpy as np
from cleaner import clean_input_data, concatenate_log
from implementations import build_poly, logistic_regression, compute_loss, CostFunction

# (CHECK THE PATHs ARE CORRECT)
train_data_path = "../../dataset/train.csv"
test_data_path = "../../dataset/test.csv"
print("Please, check the paths to the train file and to the test file are correct:")
print(train_data_path)
print(test_data_path)

# 1. load the train data 
print("Loading train data...")
y_loaded, x_loaded, ids_te = load_csv_data(train_data_path, sub_sample=False)
y_loaded = y_loaded.reshape((-1, 1))
print("Train data loaded")

# 2. clean the train data and concatenate to each dataset its log
print("Cleaning the data...")
xs, ys = clean_input_data(x_loaded.copy(), y_loaded.copy(), corr=1, dimension_expansion=5, bool_col=True)
for jet in range(4): # set -1 to 0 
    ys[jet][ys[jet]== -1] = 0 
    
xs, mean_log, std_log = concatenate_log(xs.copy())
print("Train data cleaned")

# 3. Build the polynomials (one for each one of the 4 datasets)
degree = 2
txs = [None]*4
for jet in range(4):
    txs[jet] = build_poly(xs[jet], degree)
print("The train polynomials have been built.")
    
# 4. Set the array of gammas for the logistic regression
gamma_constants = [1e-5, 1e-6] # one for the degree 1 and one for the degree 2
gammas = [None]*4
for jet in range(4):
    ncolumns = xs[jet].shape[1]
    gammas[jet] = np.concatenate([[gamma_constants[0]]] + [ncolumns*[g] for g in gamma_constants[:degree]])\
        .reshape((-1,1))

# 5. run the logistic regression on the four datasets
def logistic_regression_on_jet(jet):
    y = ys[jet]
    tx = txs[jet]
    
    initial_w = weigths[jet] #np.zeros((tx.shape[1], 1)) #
    max_iters = 2000
    print("-Logistic regression on dataset", str(jet) + "/3")
    _, w = logistic_regression(y, tx, initial_w, max_iters, gammas[jet])
    #print("Success ratio obtained: ", str(compute_loss(ys[jet], tx, weigths[jet], costfunc=CostFunction.SUCCESS_RATIO)))
    return w
   
print("Starting to run the logistic regression:")
weigths = [np.zeros((tx.shape[1])) for tx in txs]*4
for jet in range(4):
    weigths[jet] = logistic_regression_on_jet(jet)

# 6. Load the test data 
print("Loading test data...")
y_te_loaded, x_te_loaded, ids_te_loaded = load_csv_data(test_data_path, sub_sample=False)
print("Test data loaded")

# 6. Clean and append the log (in the same exact way we did with the train set)
print("Cleaning the data...")
x_te, ids_te = clean_input_data(x_te_loaded.copy(), ids_te_loaded.copy(), corr=1, dimension_expansion=5, bool_col=True)

x_te, _, _ = concatenate_log(x_te.copy(), mean_log=mean_log, std_log=std_log)
print("Test data cleaned.")

# 7. Build the polynomials
tx_te = []
for jet in range(4):
    tx_te.append(build_poly(x_te[jet], degree))
print("The test polynomials have been built.")

# 8. Predict and concatenate the predicitions
y_te_pred = []
for jet in range(4):
    y_te_pred.append(predict_labels(weigths[jet], tx_te[jet]))
    
for jet in range(4):
    ids_te[jet] = ids_te[jet].reshape((-1, 1))
y_pred = np.row_stack([y_te_pred[0], y_te_pred[1], y_te_pred[2], y_te_pred[3]])
ids = np.row_stack([ids_te[0], ids_te[1], ids_te[2], ids_te[3]])

print("I predicted ", str((y_pred==-1).sum()), "-1s and ", str((y_pred==1).sum()), "1s")

# 9. Store the predictions 
sub_file_name = "predictions"
create_csv_submission(ids, y_pred, sub_file_name)
print("Prediction stored in file '" + sub_file_name+"'")
