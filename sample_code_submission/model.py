#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 14:05:20 2018

@author: rachid
"""

from sys import argv, path
import numpy as np
import pickle

path.append ("../scoring_program")    # Contains libraries you will need
path.append ("../ingestion_program")  # Contains libraries you will need
#from prepro import Preprocessor
# IG: I commented that out: no module prepro

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from prepro import Preprocessor
from sklearn.pipeline import Pipeline

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
    return np.sqrt(1.*score*(1-score)/sample_num)

#############################################################################################################################
####################################           EXEMPLE          #############################################################
#############################################################################################################################

# IG: this class should be called model
#class MyPredictor(BaseEstimator):
class model():
    
    def __init__(self):
        #self.clf = RandomForestClassifier()
        #self.clf = NearestNeighbors(n_neighbors=10)
        #self.clf = DecisionTreeClassifier()
        self.clf = Pipeline([
                ('preprocessing', Preprocessor()),
                ('predictor', Predictor())
                ])

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)
        # IG: return proba better for AUC score
        #return self.clf.predict_proba(X)[:,1]

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:,1] # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self

class Predictor(BaseEstimator):
    
    def __init__(self):
        #self.clf = RandomForestClassifier()
        #self.clf = DecisionTreeClassifier()
        #self.clf = GaussianNB()
        #self.clf = LinearRegression()
        #self.clf = MLPClassifier()
        self.clf = SVC()

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)
        # IG: return proba better for AUC score
        #return self.clf.predict_proba(X)[:,1]

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:,1] # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self


if __name__=="__main__":

    # Find the files containing corresponding data
    # To find these files successfully:
    # you should execute this "model.py" script in the folder "sample_code_submission"
    # and the folder "public_data" should be in the SAME folder as the starting kit
    path_to_training_data = "../../public_data/Opioids_train.data"
    path_to_training_label = "../../public_data/Opioids_train.solution"
    path_to_testing_data = "../../public_data/Opioids_test.data"
    path_to_validation_data = "../../public_data/Opioids_valid.data"

    # Find the program computing AUC metric
    path_to_metric = "../scoring_program/my_metric.py"
    
    import imp
    auc_metric = imp.load_source('metric', path_to_metric).auc_metric_

    # use numpy to load data
    X_train = np.loadtxt(path_to_training_data)
    y_train = np.loadtxt(path_to_training_label)
    X_test = np.loadtxt(path_to_testing_data)
    X_valid = np.loadtxt(path_to_validation_data)


    # TRAINING ERROR
    # generate an instance of our model (clf for classifier)
    clf = model()
    # train the model
    clf.fit(X_train, y_train)
    # to compute training error, first make predictions on training set
    y_hat_train = clf.predict(X_train)
    # then compare our prediction with true labels using the metric
    training_error = auc_metric(y_train, y_hat_train)


    # CROSS-VALIDATION ERROR
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from numpy import zeros, mean
    # 3-fold cross-validation
    n = 5
    kf = StratifiedKFold(n_splits=n)
    kf.get_n_splits(X_train)
    i=0
    scores = zeros(n)
    X = np.ones(15000)
    for train_index, test_index in kf.split(X_train, X):
        Xtr, Xva = X_train[train_index], X_train[test_index]
        Ytr, Yva = y_train[train_index], y_train[test_index]
        M = model()
        M.fit(Xtr, Ytr)
        Yhat = M.predict(Xva)
        scores[i] = auc_metric(Yva, Yhat)
        print ('Fold', i+1, 'example metric = ', scores[i])
        i=i+1
    cross_validation_error = mean(scores)

    # Print results
    print("\nThe scores are: ")
    print("Training: ", training_error)
    print ('Cross-Validation: ', cross_validation_error)

    print("""
To compute these errors (scores) for other models, uncomment and comment the right lines in the "Baseline models" section of the class "model".
To obtain a validation score, you should make a code submission with this model.py script on CodaLab.""")

    """  
    # IG: I moved this under main
    
    from sklearn.metrics import accuracy_score      
    # Interesting point: the M2 prepared challenges using sometimes AutoML challenge metrics
    # not scikit-learn metrics. For example:
    from libscores import bac_metric
    from libscores import auc_metric        
    from data_manager import DataManager 
    from data_converter import convert_to_num 
    
    # END IG

        # We can use this to run this file as a script and test the Classifier
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../../" # A remplacer par public_data
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
        
    basename = 'Opioids'
    D = DataManager(basename, input_dir)
    
    myclassifier = model()
    Ytrue_tr = D.data['Y_train']
    myclassifier.fit(D.data['X_train'], Ytrue_tr)
    
    Ypred_tr = myclassifier.predict(D.data['X_train'])
    Ypred_va = myclassifier.predict(D.data['X_valid'])
    Ypred_te = myclassifier.predict(D.data['X_test'])
    
    Yprob_tr = myclassifier.predict_proba(D.data['X_train'])
    Yprob_va = myclassifier.predict_proba(D.data['X_valid'])
    Yprob_te = myclassifier.predict_proba(D.data['X_test'])
    
    Yp = Ypred_tr>0
    acc = accuracy_score(Ytrue_tr, Yp)
    Ypred_tr.shape
    # Then two AutoML challenge metrics, working on the other representation
    auc = auc_metric(Ytrue_tr, Yprob_tr, task='binary.classification')
    bac = bac_metric(Ytrue_tr, Yprob_tr, task='binary.classification')

    print "%s\t%5.2f\t%5.2f\t%5.2f\t(%5.2f)" % (myclassifier, auc, bac, acc, ebar(acc, Ytrue_tr.shape[0]))
    print "The error bar is valid for Acc only"
        # Note: we do not know Ytrue_va and Ytrue_te
        # See modelTest for a better evaluation using cross-validation
        
    # Another useful tool is the confusion matrix
    from sklearn.metrics import confusion_matrix
    
    print "Confusion matrix for %s" % myclassifier
    print confusion_matrix(Ytrue_tr, Ypred_tr)
    """

































