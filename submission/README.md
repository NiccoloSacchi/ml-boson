 _____________________________________________
|*********************************************|
|********MACHINE LEARNING COURSE - EPFL*******|
|******PROJECT 1 - HIGGS BOSON CHALLENGE******|
|******************Autumn 2017****************|
|*********************************************|
|---------------------------------------------|
|*****Authors :*******************************|
|********Antonio Barbera**********************|  |********Valentin Nigolian********************|
|********Niccolò Sacchi***********************|
|*********************************************|
 '''''''''''''''''''''''''''''''''''''''''''''


This README is organized as follows :

1 MOTIVATION
2 THE DATA
3 SYSTEM OVERVIEW \n
  3.1 DATA PREPARATION \n
    3.1.1 OUTLIERS REMOVAL
    3.1.2 DISTRIBUTION ANALYSIS
    3.1.3 DATASET SPLIT
    3.1.4 STANDARDIZATION
    3.1.5 PERCENTILES SPLIT
  3.2 TRAINING MODEL
    3.2.1 ALGORITHM USED
    3.2.1 HYPERPARAMETERS
4 RUNNING THE SYSTEM


1 MOTIVATION
------------
This project aims at labelling at large dataset as either indicating the presence of a Higgs boson (signal) or as noise (background) by using Machine Learning techniques. 

2 THE DATA
----------
The data was produced by a physical simulator imitating the results of an experiment conducted at CERN in Geneva. It contains 250'000 items, all of which have 30 features. Each items represents the result of two particles crashing into each other in CERN's Large Hadron Collider and each features represents one particular of said crash.

3 SYSTEM OVERVIEW
-----------------
To perform our task, we have developped a Machine Learning system consisting of two main steps : data preparation and model training

3.1 DATA PREPARATION
--------------------
We applied 5 steps in the data preparation :

3.1.1 OUTLIERS REMOVAL : Outliers on one or more of the features represented a very small fraction of the overall data and were removed.

3.1.2 DISTRIBUTION ANALYSIS : Each feature for which the data was distributed evenly between the signal and background labels was dropped.

3.1.3 DATASET SPLIT : The full dataset was split into four groups depending on the feature called "PRI_jet_num". This feature is categorical and it appeared that there were

3.1.4 STANDARDIZATION : We used the mean and standard deviation from the whole set (training and testing data) to standardize each set.

3.1.5 PERCENTILES SPLIT : We expand evenly the features into subfeatures to allow finer polynomial interpolation with smaller degree.


3.2 TRAINING MODEL
------------------

3.2.1 ALGORITHM USED : We use the logistic regression algorithm using gradient descent

3.2.1 HYPERPARAMETERS : We used a polynomial of degree 2 that, with other techinques to enrich our model, allowed us to score 83.2% of correct predictions. For more details please refer to the report. 


4 RUNNING THE SYSTEM
--------------------

The system is fully functionnal and automated. You can run it by simply run the python script called "run.py". Alternatively, the Jupyter notebook called "run.ipynb" to get get the intermediary results and visualizations.















