#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from sys import argv, path
path.append ("../ingestion_program") 

from data_manager import DataManager  
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator


class Preprocessor(BaseEstimator):
    def __init__(self):
        self.transformer = 

    def fit(self, X, y=None):

        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):

        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):

        return self.transformer.transform(X)
    
if __name__=="__main__":
        input_dir = "../public_data"
        output_dir = "../sample_result_submission" 

    
    basename = 'Opioids'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D
    
    Prepro = Preprocessor()
 
    # Preprocess sur les donner et on les remet sur D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
    # Affiche si ça marche
    print("*** Transformed data ***")
    print D