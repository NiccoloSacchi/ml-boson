# -*- coding: utf-8 -*-
""" Functions used to plot the data. """

import numpy as np
import matplotlib.pyplot as plt
from cleaner import column_labels

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