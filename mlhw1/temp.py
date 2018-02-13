# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the data

Y = iris[:,-1]
X = iris[:,0:-3]


# Note: indexing with ":" indicates all values (in this case, all rows);
# indexing with a value ("0", "1", "-1", etc.) extracts only that one value (here, columns);
# indexing rows/columns with a range ("1:-1") extracts any row/column in that range.
import mltools as ml
# We'll use some data manipulation routines in the provided class code

X,Y = ml.shuffleData(X,Y); # shuffle data randomly
# (This is a good idea in case your data are ordered in some pathological way,
# as the Iris data are)

Xtr,Xte,Ytr,Yte = ml.splitData(X,Y, 0.75); # split data into 75/25 train/test

"""
K = 50 #for nearest neighbor prediction

knn = ml.knn.knnClassify() # create the object and train it
knn.train(Xtr, Ytr, K) # where K is an integer, e.g. 1 for nearest neighbor prediction
YteHat = knn.predict(Xte) # get estimates of y for each data point in Xte
ml.plotClassify2D( knn, Xtr, Ytr ); # make 2D classification plot with data (Xtr,Ytr)
"""
errTrain = [0] * 7;

K=[1,2,5,10,50,100,200];
for i,k in enumerate(K):
    learner = ml.knn.knnClassify(Xtr, Ytr, k) # TODO: complete code to train model
    Yhat = learner.predict(Xtr) # TODO: complete code to predict results on training data
    err = 0
    for j in range(0,len(Yhat)):
        err += 1 if (Yhat[j] != Ytr[j]) else 0 
    fracterr = err/(len(Yhat))                 
    errTrain[i] = fracterr 

plt.semilogx(K, errTrain, color = 'r') #TODO: " " to average and plot results on semi-log scale

K=[1,2,5,10,50,100,200];
for i,k in enumerate(K):
    learner = ml.knn.knnClassify(Xtr, Ytr, k) # TODO: complete code to train model
    Yhat = learner.predict(Xte) # TODO: complete code to predict results on training data
    err = 0
    for j in range(0,len(Yhat)):
        err += 1 if (Yhat[j] != Yte[j]) else 0 
    fracterr = err/(len(Yhat))                 
    errTrain[i] = fracterr 

plt.semilogx(K, errTrain, color = 'g') #TODO: " " to average and plot results on semi-log scale
"""
colors = ['b','g','r']
for c in np.unique(Y):
 plt.plot( X[Y==c,0], X[Y==c,3], 'o', color=colors[int(c)] )
 """