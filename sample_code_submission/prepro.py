#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:43:13 2018

@author: rachid.abbara
"""

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

class Preprocessor(BaseEstimator):
    def __init__(self):
        '''
        This example does not have comments: you must add some.
        Add also some defensive programming code, like the (calculated) 
        dimensions of the transformed X matrix.
        '''
        self.transformer = PCA(n_components=10)
        print("PREPROCESSOR=" + self.transformer.__str__())

    def fit(self, X, y=None):
        print("PREPRO FIT")
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        print("PREPRO FIT_TRANSFORM")
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        print("PREPRO TRANSFORM")
        return self.transformer.transform(X)