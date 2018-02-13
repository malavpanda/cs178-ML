# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:16:31 2017

@author: Malav
"""

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt


#(a)
X = np.genfromtxt("X_train.txt",delimiter=None) # load the data
Xt = X[1:250,:];
Xv = X[251:500,:];
Y = np.genfromtxt("Y_train.txt",delimiter=None) # load the data
Yt = Y[1:250,np.newaxis];
Yv = Y[251:500,];

Xte = np.genfromtxt("X_test.txt",delimiter=None) # load the data

learner = ml.logistic2.logisticClassify2(Xt,Yt)