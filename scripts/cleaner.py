# -*- coding: utf-8 -*-
""" Functions used to clean the data. """

import numpy as np
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

def clean_input_data(dataset_all, output_all=np.array([]), corr=1, dimension_expansion=False, bool_col=False):
    """ Cleans the data:
    1. Sets the -999 values to nan.
    2. Standardizes.
    3. Splits using the jet number and drops useless columns.
    4. Substitutes with 0 all the nans in the first column of each obtained dataset and if bool_col 
    = True also adds a boolean column to indicate the position of those nans.
    5. Drops other columns depending on the given correlation "corr", e.g. if corr = 0.7 drops the
    maximum amount of columns such that there are no two columns whose |correlation| > 0.7.
    6. If dimension_expansion = true also split every column into more columns (check the report for 
    an explanation).
    """
    
    # 1
    dataset_all[dataset_all==-999] = np.nan
    
    # 2.
    dataset_all = standardize_final(dataset_all)
    
    # 3.
    datasets, outputs = split_input_data(dataset_all, output_all)
            
    # 4. 
    for jet, dataset in datasets.items():
        nan_rows = np.isnan(dataset[:,0])
        datasets[jet][nan_rows, 0] = 0
        # also append a boolean column to indicate the position of those -999 since 
        # this information may still be usefull for determining the y
        if bool_col:
            datasets[jet] = np.column_stack((datasets[jet], nan_rows))
    
    # 5. 
    
    # 6.
    if dimension_expansion:
        datasets = expand_dimensions(datasets, bool_col) # pass also bool_col to know if the last one is a boolean column
        
    return datasets, outputs

def split_input_data(dataset_all, output_all=np.array([])):
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
    datasets = {}
    outputs = {}
    for jet in range(num_jets):
        curr_dataset, curr_output = get_with_jet(dataset_all, output_all, jet)
        # drop columns depending on the jet, drop always the PRI_jet_num (column 22)
        if jet == 0:
            to_drop = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29] # 22 and 29 contains only 0s, the others only -999
        elif jet == 1:
            to_drop = [4, 5, 6, 12, 22, 26, 27, 28]
        else:
            to_drop = [22]
        
        to_drop = to_drop
        print("Jet", jet, "columns dropped:", to_drop)
        curr_dataset = np.delete(curr_dataset, to_drop, axis=1)
        
        datasets[jet] = curr_dataset
        outputs[jet] = curr_output

    return datasets, outputs

def expand_dimensions(datasets, bool_col):
    file_path = "metadata/percentiles.json"
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    percentiles = json.loads(obj_text)

    for jet, curr_dataset in datasets.items():
        columns_to_be_splitted = range(curr_dataset.shape[1]-bool_col) # scan columns except the boolean one (if it was added)
        for col in columns_to_be_splitted: 
            c = curr_dataset[:, col].copy() # column to be splitted
            col_perc = percentiles[str(jet)][str(col)]
            #print([p for p in col_perc if (int(p)%50 == 0 and int(p) != 0)])
            for perc in [col_perc[p] for p in col_perc if (int(p)%50 == 0 and int(p) != 0)]:
                vals = (c<=perc)
                curr_dataset = np.column_stack((curr_dataset, vals*c))
                c = c*(~vals)
        datasets[jet] = np.delete(curr_dataset, columns_to_be_splitted, axis=1)
    return datasets

# helper function that, given a minimum correlation, return the list of columns that can be dropped 
def correlated(jet, corr):
    """ Drop the correlated columns. This step may not improve the success ratio of the model
    but it will simplify it. corr must be in {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1}."""
    # mappings: jet number (dataset) -> min correlation -> columns (features) that can be dropped
    return {
        0: {
            0.7: [5, 9, 12, 15],
            0.75: [5, 9, 12],
            0.8: [5],
            0.85: [5],
            0.9: [5],
            0.95: [5]
        },
        1: {
            0.7: [6, 12, 17, 18, 21],
            0.75: [3, 17, 18, 21],
            0.8: [6, 18, 21],
            0.85: [6, 18, 21],
            0.9: [3, 6, 21],
            0.95: [21]
        },
        2: {
            0.7: [5, 6, 9, 16, 19, 21, 22, 28],
            0.75: [3, 5, 6, 21, 22, 28],
            0.8: [5, 6, 21, 22, 28],
            0.85: [6, 21, 22, 28],
            0.9: [22, 28],
            0.95: [28]
        },
        3: {
            0.7: [5, 6, 16, 19, 21, 22, 25, 28],
            0.75: [5, 6, 16, 21, 22, 25, 28],
            0.8: [9, 21, 22, 25],
            0.85: [21, 22, 28],
            0.9: [21, 28],
            0.95: [28]
        }
    }[jet][corr]

def standardize_final(data):
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
          2.09908730e+02,   0,                8.49042850e+01, #previously 1.63345672e+00
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