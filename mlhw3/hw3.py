# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:11:01 2017

@author: Malav
"""

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import logisticClassify2 as lo

iris = np.genfromtxt("data/iris.txt",delimiter=None);
X, Y = iris[:,0:2], iris[:,-1];
X,Y = ml.shuffleData(X,Y);
X,_ = ml.transforms.rescale(X);


XA, YA = X[Y<2,:], Y[Y<2]; #get class 0 vs 1
XB, YB = X[Y>0,:], Y[Y>0]; #get class 1 vs 2

#(a)
"""
plt.plot(X[Y==0,0], X[Y==0,1], 'bo');
plt.plot(X[Y==1,0], X[Y==1,1], 'ro');
plt.title("Class 0 vs Class 1")

plt.figure();

plt.plot(X[Y==1,0], X[Y==1,1], 'bo');
plt.plot(X[Y==2,0], X[Y==2,1], 'ro');
plt.title("Class 1 vs Class 2")
"""
#The data set for class 0 vs class 1 is separable

#(b)
learner = lo.logisticClassify2(); # create "blank" learner
learner.classes = np.unique(YA) # define class labels using YA or YB


wts = np.array([0.5,1,-0.25]); # TODO: fill in values
wts = wts[:,np.newaxis];
learner.theta = wts; # set the learner's parameters

#learner.plotBoundary(XA,YA);

#(c)
YAhat = [None] * len(XA);
YAhat = learner.predict(XA);


i = 0; err = 0;
for i in range(len(YAhat)):
    err += 1 if (YAhat[i] != YA[i]) else 0 ;

fracterr = err/(len(YAhat));      

#Repeat for XB,YB           

#(d)
plt.figure();
ml.plotClassify2D(learner,XA,YA);
#The figures match

#(e)
wts = np.random.rand(3);
wts = wts[:,np.newaxis];
learner.theta = wts; # set the learner's parameters
learner.train(XA,YA);
plt.figure();
ml.plotClassify2D(learner,XA,YA);