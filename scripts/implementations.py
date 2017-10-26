# -*- coding: utf-8 -*-
""" Functions used to for project 1 - Machine Learning 2017 """

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import predict_labels
from types import SimpleNamespace 
import codecs, json 

# workflow:
# 1. load data
# 2. fill_nan (possibly without passing a subs_func)
# 3. drop_nan_rows/column (only if you didn't passed a subs_func)
# 4. standardize
# 5. for a chosen model and a set of hyperparameters: build_poly, train model, test model

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
    

# DATA ANALYSIS FUNCTIONS
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
    
    
def plot_distributions(x, y, col_labels = column_labels(), title="distributions"):
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
#         feature1 = feature1[feature1 != -999]
#         feature1 = feature1[feature1 != 0]
        # plot histogram
        n, bins, patches = \
            ax_row[0].hist(feature1, histtype='step', bins=int(len(feature1)/1000), color="blue", normed=False)
        # plot distribution
        y = plt.mlab.normpdf(bins, np.mean(feature1), np.std(feature1))  
        ax_row[1].plot(bins, y, 'b--', linewidth=1)
        
        feature0 = x[indices0, feature]
#         feature0 = feature0[feature0 != -999]
#         feature0 = feature0[feature0 != 0]
        # plot histogram
        n, bins, patches = \
            ax_row[0].hist(feature0, histtype='step', bins=int(len(feature0)/1000), color="red", normed=False)
        # plot distribution
        y = plt.mlab.normpdf(bins, np.mean(feature0), np.std(feature0))
        ax_row[1].plot(bins, y, 'r--', linewidth=1)
        
        ylabels = ["normed frequency", "distribution"]
        for i in range(n_cols):
            ax_row[i].grid()
            ax_row[i].legend(["y = 1", "y = -1"])
            ax_row[i].set_title(str(feature) + ": " + col_labels[feature])
            ax_row[i].set_ylabel(ylabels[i])
            ax_row[i].set_xlabel("feature values")

    plt.tight_layout()
    plt.savefig(title)
    plt.show()
    
    
# DATA CLEANING FUNCTIONS 

# -nan_values: a map from column indices to the respective nan value
# e.g. nan_values = {5: -999, 20: 0, ...} (only for column which contain
# an invalid value)
# -subs_func: a function, e.g. np.nanmean, np.nanmedian, np.nanstd, that will
# be applied column-wise and whose result will be placed in the nan values
# of that column. If this function is not passed than it will keep np.nan values.
def fill_with_nan_map(x, nan_values={}, subs_func=None):
    """ Deals with the nan values: identify and substitute them.
    Run this function before the 'standardize' one. """
    x = x.astype(float) 
    for i in nan_values.keys():
        col = x[:, i]
         # first set to nan those values
        col[col==nan_values[i]] = np.nan 
        # then substitute the nan values
        if subs_func != None:
            col[np.isnan(col)] = subs_func(col)
    return x  

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
    
# drop all the rows containing np.nan
def drop_nan_rows(x, y):
    keep_indices = ~np.isnan(x).any(axis=1)
    return x[keep_indices], y[keep_indices]

# drop all the rows containing np.nan
def drop_nan_columns(x):
    return x[:, ~np.isnan(x).any(axis=0)]

def drop_same_distribution_columns(x):
    #columns : N,R,U,W,AB,AE -> 11,15,18,20,25,28
    tobe_deleted = [11,15,18,20,25,28]
    return np.delete(x, tobe_deleted, axis=1)

def corr_columns_indices(x):
    return list([4,5,6,12,26,27,28])

def nan_ratio_columns_indices(x,ratio):
    return np.where(np.sum(x==-999, axis=0)/x.shape[0] < 0.7)[0]

def same_distribution_columns_indices(x):
    return [11,15,18,20,25,28]


def compute_deletion_list(x):
    return np.array([4,5,6,11,12,15,18,20,25,26,27,28])               


def clean_data(x,data_u):
    deletion_list = compute_deletion_list(x)
    print(deletion_list.shape)
    return np.delete(x,deletion_list,axis=1),np.delete(data_u,deletion_list,axis=1)


#### definitive cleaning

def clean_input_data(dataset_all):
    
    dataset_all, _, _ = standardize(dataset_all)
    
    datasets = split_input_data(dataset_all)
    
    datasets = drop_correlated(datasets, corr = 0.8)
    
    # fill nan in the first column with 0s
    return datasets
# Steps in data cleaning:

# 1.
def split_input_data(dataset_all):
    """ This function will separate the input dataset into 4 datasets depending
    on the value in the categorical column PRI_jet_num (column 22):
    (1) one dataset for jet num = 0, 
    (2) one dataset for jet num = 1, 
    (3) one dataset for jet num = 2 
    (4) one dataset for jet num = 3.
    We also drop, for each obtained dataset, the columns with only -999 values and substitute
    with np.nan the -999 values in the first column.
    """
    
    def get_with_jet(dataset, jet_num): # given jet and dataset return the rows with th egiven jet number
        return dataset[dataset[:, 22]==jet_num, :]

    
    num_jets = 4
    datasets = [None]*num_jets
    for jet in range(num_jets):
        curr_dataset = get_with_jet(dataset_all, jet)
        # drop columns depending on the jet (drop always the PRI_jet_num column)
        # TODO drop the correlated columns (recompute correlation within the different datasets)
        if jet == 0:
            to_drop = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29] # 22 and 29 contains only 0s, the others only -999
        elif jet == 1:
            to_drop = [4, 5, 6, 12, 22, 26, 27, 28]
        else:
            to_drop = [22]
            
        curr_dataset = np.delete(curr_dataset, to_drop, axis=1)
        datasets[jet] = curr_dataset

    return datasets

# 2. (not necessary)
def drop_correlated(datasets, corr = 0.8):
    """ Drop the correlated columns. This step may not improve the success ratio of the model
    but it will simplify it. corr must be in {0.7, 0.8, 0.9}"""


####

    
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

def clean_x3(x):
    x_loaded = x.copy()
    
    ncols = x_loaded.shape[1]
    
    # 1. compute the booleans columns
    bool_cols = np.zeros((x_loaded.shape[0], 1))

    # boolean columns from the only CATEGORICAL one
    categorical = x_loaded[:, column_labels_map()["PRI_jet_num"]] # take the values    
    for cat_value in [0, 1, 2, 3]:
        #insert before the last one
        bool_cols = np.column_stack((bool_cols, categorical==cat_value))

    # boolean column from the ones with a lot of -999
    #for col in range(ncols):
    for col in [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]: # hardcoded
        vals999 = x_loaded[:, col] == -999 
        if np.sum(vals999) > 0: 
            print(col)
            bool_cols = np.column_stack((bool_cols, vals999)) # != -999 => 0. == -999 => 1
     
    # boolean column from the ones with a lot of 0
    for col in [12, 29]:
        vals0 = x_loaded[:, col] == 0 
        if np.sum(vals0) > 0:  
            bool_cols = np.column_stack((bool_cols, vals0))# != -999 => 0. == -999 => 1

    
    bool_cols = bool_cols[:, 1:] # drop the initial empty column 
    bool_cols = np.unique(bool_cols, axis=1) # drop the repeated columns 
    
    print(bool_cols.shape[1], "boolean columns have been created.") 

    # 1.5. Manage outliers
    # x_loaded = move_outliers(x_loaded)
    
    # 2. drop: 
        # the categorical column
        # the 3 columns with equal distribution 
        # the 2 columns which are highly correlated with another column
    
    lab_map= column_labels_map()
    to_be_removed = [lab_map["PRI_jet_num"], lab_map["PRI_tau_phi"], lab_map["PRI_lep_phi"], lab_map["PRI_met_phi"], 9, 23]
    to_be_removed.sort()
    
    x_loaded = np.delete(x_loaded, to_be_removed, axis=1) 
    print(len(to_be_removed), "columns have been removed: ", to_be_removed)        
    
    # 3. set -999s and 0s to np.nan 
    x_loaded = fill_with_nan_list(x_loaded, nan_values=[0, -999])
    
    # 4. split input dimensions
    # load the perceintiles
    file_path = "percentiles.json"
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    
    scan_perc = b_new["scan_perc"]
    col_perc = b_new["col_perc"]
        
    col_perc = np.delete(col_perc, to_be_removed, axis=1) # delete the same columns
    if col_perc.shape[1] != x_loaded.shape[1]:
        print("Something is wrong")

    init_cols = range(x_loaded.shape[1])
    for col in init_cols:
        c = x_loaded[:, col].copy() # column to be splitted

        for p in [scan_perc.index(25), scan_perc.index(50), scan_perc.index(100)]:
            perc = col_perc[p, col] # take the value of the percentile
            vals = (c<=perc)
            #if not np.all(c1==c1[0]):
            x_loaded = np.column_stack((x_loaded, vals*c))
            c = c*(~vals)

    x_loaded = np.delete(x_loaded, init_cols, axis=1)
    x_loaded = fill_with_nan_list(x_loaded, nan_values=[0, -999])
    
    """
    for times in range(3):
        for col in range(x_loaded.shape[1]):
            c = x_loaded[:, col].copy()
            med = np.nanmedian(c)
            #print(col, med, np.sum(c>med), np.sum(c<=med))

            c1 = c*(c>med)
            if not np.all(c1==c1[0]): # if it is a constant then don't add it
                x_loaded[:, col] = c1

            c1 = c*(c<=med)
            if not np.all(c1==c1[0]):
                x_loaded = np.insert(x_loaded, -1, c1, axis = 1)"""
    
    
    
    #before = x_loaded.shape[1]
    #x_loaded = np.unique(x_loaded, axis=1)
    #after = x_loaded.shape[1]
    #print("Dropped", before-after, "equal columns")
    
    return x_loaded, bool_cols

def clean_x2(x_loaded, double=False):
    x_all = x_loaded.copy()
    
    x_all = move_outliers(x_all)
    
    # fill -999 and 0 with the np.nan
    x_all = fill_with_nan_list(x_all, nan_values=[0, -999])

    # standardize
    x_all, mean_x, std_x = standardize(x_all)

    # substitute the nan values with something
    x_all = sustitute_nans(x_all, substitutions=np.zeros(x_all.shape[1]))
    
    # reintegrate the useful information that was removed to standardize the dataset (add new columns to store these 
    # "strange" numbers)
    threshold = 1000

    ncols = x_loaded.shape[1]
    new_index = ncols
    
    # reintegrate -999s
    for col in range(ncols):
        vals999 = x_loaded[:, col] == -999 
        sum_ = np.sum(vals999)
        if sum_ > threshold:
            print("-999:", sum_)
            # add a (double) column for each 
            
            if double == True:
                x_all = np.insert(x_all, new_index, vals999, axis = 1)
                new_index += 1
                x_all = np.insert(x_all, new_index, vals999-1, axis = 1)
                new_index += 1
            else:
                x_all = np.insert(x_all, new_index, (vals999)*2-1, axis = 1) # != -999 => -1. == -999 => 1
                new_index += 1

    # reintregrate 0s
    for col in range(ncols):
        vals0 = x_loaded[:, col] == 0
        sum_ = np.sum(vals0)
        if sum_ > threshold:
            print("0s:", sum_)
            # add a column for each 
            if double == True:
                x_all = np.insert(x_all, new_index, vals0, axis = 1)
                new_index += 1
                x_all = np.insert(x_all, new_index, vals0-1, axis = 1)
                new_index += 1
            else:
                x_all = np.insert(x_all, new_index, (vals0)*2-1, axis = 1) # != 0 => -1. == 0 => 1
                new_index += 1

    print("Added", new_index-ncols, "columns")
    
    to_be_removed = np.where(np.isin(column_labels(), ["PRI_tau_phi", "PRI_lep_phi", "PRI_met_phi"]))[0].tolist() + [9, 23]
    print("Dropped:", to_be_removed)
    x_all = np.delete(x_all, to_be_removed, axis=1) 
    
    return x_all
    
def clean_x(x_, corr, subs_func=None, bool_col=False):
    """ 
    bool_col = {True, False} if true creates a boolean column 
    subs_func = {None, np.nanmean, np.nanmedian, np.nanstd}
    
    1. Drops the 7 invalid columns.
    2. Drops the columns with |correlation| > 'corr'.
    3. Treats the -999 depending on subs_func. 
    4. Standardises. 
    """
    
    col999 = np.sum(x_==-999, axis = 1)
    
    ncol = x_.shape[1]
    
    # drop "invalid" and correlated columns
    x_, keptColumns = drop_corr_columns(x_,corr)
    
    # drop also the columns with same distribution 
    to_be_removed = np.where(np.isin(keptColumns, ["PRI_tau_phi", "PRI_lep_phi", "PRI_met_phi"])) # PRI_jet_num

    keptColumns = np.delete(keptColumns, to_be_removed)
    x_ = np.delete(x_, to_be_removed, axis=1)
    print(ncol - x_.shape[1], "columns have been dropped")
    
    # add  bolean / sum columns
    if bool_col == True:
        count = 0
        
        x_ = np.concatenate((x_, [[0]]*x_.shape[0]), axis=1)
        x_[:, -1] = col999 #np.sum(x_==-999, axis = 1)
        count += 1
        
        
        if "PRI_jet_all_pt" in keptColumns:
            count += 1
            x_ = np.concatenate((x_, [[0]]*x_.shape[0]), axis=1)
            # set column to 1 (true) if PRI_jet_all_pt == 0 
            x_[:, -1] = x_[:, -2] == 0
        
        """
        # creates a bool column, whose value is 1 if any of the other values in the row is 999, 0 otherwise
        for col in range(x_.shape[1]):
            x_col = x_[:, col]
            
            nan_vals = (x_col == -999)
            
            if np.any(nan_vals): # if there are -999 in this column, then add a boolean column to store this info 
                count += 1
                # add the column 
                x_ = np.concatenate((x_, [[0]]*x_.shape[0]), axis=1)

                # set column to 1 (true) if the row contains at least a -999
                x_[:, -1] = nan_vals

                keptColumns = np.append(keptColumns, "bool_"+keptColumns[col])"""
                
        print("Added", count , "bool columns")
        
    if subs_func != None:
        # fill -999 and 0 with the np.nan
        x_ = fill_with_nan_list(x_, nan_values=[-999])

        # standardize
        x_, mean_x, std_x = standardize(x_)

        # substitute the nan values with something
        x_ = sustitute_nans(x_, substitutions=subs_func(x_, axis=0)) 
    else:
        # standardize
        x_, mean_x, std_x = standardize(x_)

    return x_, keptColumns


def drop_columns_with_70_nan_ratio(x):
    """ drops the columns with more than 70% of entries with -999 """
    return x[:, columns_with_70_nan_ratio(x)]

def columns_with_70_nan_ratio(x):
    """ Some columns have more than 70% of entries with -999.
    This function returns the columns that can be kept """
    tot = x.shape[0]
    indices = np.where(np.sum(x==-999, axis=0)/tot < 0.7)[0]
    return indices

def drop_corr_columns(x, min_corr, important=True):
    """ 1. drop the columns with lot of nan values (hardocded).
    2. Drop the columns which have a correlation higher (in absolute value) than the passed one. 
    Returns both the new dataset and the list labels of the kept columns. """
    
    # load the correlation matrix that was computed on the whole dataset
    file_path = "corr_matrix.json"
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    corr_matrix_loaded = np.array(b_new)
    
    kept_columns = column_labels()
    # hardcoded indices of the columns with less than 70% of nan values
    toKeep = [0,  1,  2,  3,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29]    
    
    # update x, the labels of the columns and the correlation matrix
    x = x[:, toKeep]
    kept_columns = kept_columns[toKeep]
    corr_matrix_loaded = corr_matrix_loaded[toKeep, :][:, toKeep] 

    ncols = x.shape[1]
    corr_matrix_bool = np.abs(corr_matrix_loaded) > min_corr  
    
    # i is surely correlated to itself, drop that (useless) information
    for i in range(ncols):
        corr_matrix_bool[i][i] = False


    # compute the mapping of correlations
    corr = {} 
    for i in range(ncols):
        c = np.where(corr_matrix_bool[i])[0].tolist()
        if len(c) > 0: # if it is not correlated to any other column then ignore it
            corr[i] = c
            
    # print(correlated_to)
    # keep the jet all column
    tobe_deleted = []
    if (important) & ("PRI_jet_all_pt" in kept_columns): # it is the last column 
        imp_col = len(kept_columns)-1
        for key in corr: 
            corr[key] = list(set(corr[key]) - set([imp_col]))
        corr[imp_col] = []
        
    # fetch all the columns that can be deleted and put them in tobe_deleted
    for _ in range(len(corr)): 
        longer_key = -1 
        longer_length = 0

        # look for the longer list
        for key in corr:
            curr_length = len(corr[key])
            if curr_length > longer_length:
                longer_length = curr_length
                longer_key = key

        if longer_length == 0: 
            break

        # print(longer_key, "\t", corr[longer_key])

        tobe_deleted.append(corr[longer_key])
        # delete all the columns that are correlated to column longer_key
        # i.e. all the column whose index is in  corr[longer_key]
        for corr_colum in corr[longer_key]:
            corr[corr_colum] = []
            
        # since those columns have been dropped they must be removed from all the other lists
        for key in corr: 
            if key != longer_key:
                #print(key, corr[key], "-", corr[longer_key], "=", list(set(corr[key]) - set(corr[longer_key])))
                corr[key] = list(set(corr[key]) - set(corr[longer_key]))
        corr[longer_key] = []

    tobe_deleted = [val for sublist in tobe_deleted for val in sublist] # flatten the list

    return np.delete(x, tobe_deleted, axis=1), np.delete(kept_columns, tobe_deleted)

# STANDARDIZE X
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
    PROB = 4 #  probabilistical cost function, better for categorical labelling
    SUCCESS_RATIO = 5

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
    
    if costfunc is CostFunction.SUCCESS_RATIO:
        # Given the weigths and a test set it compute the prediction for every input
        # and returns the ratio of correct predictions.
        # This cost function is not continue and therefore not differentialbe, therefore
        # can be used only to evaluate the final model or for a grid search. 
        # However we have 30 dimension, therefore the grid search has a high complexity:
        # O(N^30) where N is the amount of trials per dimension.
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
def gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size=-1, print_output_with_weights=[], plot_losses=True, costfunc=CostFunction.MSE):
    """ w(t+1) = w(t)-gamma*gradient(L(w)) where L(w) can be computed on a subset of variables depending on batch_size.
    If a different batch_size is not passed then L(w) is computed using all the points.
    Can not be used with the non-differentiable MAE.  """
    
    # TODO 
    # 1. implement a decreasing gamma
    
    # if costfunc = PROB the y should be made only of 1s and 0s (just for the training phase).
    if costfunc == CostFunction.PROB:
        y[y==-1] = 0
    
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
            
            if len(print_output_with_weights) > 0:
                print("Gradient Descent({bi}/{ti}): loss={l}, weights={weights}".format(
                      bi=n_iter, ti=max_iters - 1, l=curr_loss, weights=[w[i] for i in print_output_with_weights]))

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
    try:
        w = np.linalg.solve(A, b)
        return compute_loss(y, tx, w, costfunc=CostFunction.MSE), w
    except Exception as e:
        print("When solving the system in ridge regression: " + str(e))
        return -1, np.zeros(tx.shape[1])

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
