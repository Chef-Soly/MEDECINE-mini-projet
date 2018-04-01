# -*- coding: utf-8 -*-
"""
Le preprocessing
"""


from sklearn.base import BaseEstimator

#Pour les fonctions de preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import GenericUnivariateSelect

class Preprocessor(BaseEstimator):

    def __init__(self):
        '''
        Initialisation de la fonction de preprocessing
        '''
        
        self.transformer = SelectFpr(alpha=0.9)
        #self.transformer = GenericUnivariateSelect(mode ='fwe', param = 1e-03)
        
               
        
    def fit(self, X, y=None):
        '''Centrage des donnees'''

        return self.transformer.fit(X, y)


    def fit_transform(self, X, y=None):
        '''Appel de fit et transform pour le set d'entrainement'''

        return self.transformer.fit_transform(X,y)


    def transform(self, X, y=None):
        '''Application de la transformation'''

        return self.transformer.transform(X)
    
    print("Data Original")
    print Data     

    '''Appel de la classe Preprocessor'''
    Prepro = Preprocessor() #J'appelle le preprocessing
 
    Data.data['X_train'] = Prepro.fit_transform(Data.data['X_train'], Data.data['Y_train'])
    Data.data['X_valid'] = Prepro.transform(Data.data['X_valid'])
    Data.data['X_test'] = Prepro.transform(Data.data['X_test'])
    
    print("Data aprespro")
    print Data