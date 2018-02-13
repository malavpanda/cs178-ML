# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the text file
Y = iris[:,-1] # target value is the last column
X = iris[:,0:-1] # features are the other columns

print  (X.shape[1])     #4 features
print  (X.shape[0])     #148 Data points

"""
X1 = X[:,0] # extract first feature
Bins = np.linspace(4,8,17) # use explicit bin locations
plt.hist( X1, bins=Bins ) # generate the plot 

X2 = X[:,1] # extract first feature
Bins = np.linspace(2,4,17) # use explicit bin locations
plt.hist( X2, bins=Bins ) # generate the plot 

X3 = X[:,2] # extract first feature
Bins = np.linspace(1,7,17) # use explicit bin locations
plt.hist( X3, bins=Bins ) # generate the plot 

X4 = X[:,3] # extract first feature
Bins = np.linspace(0,3,17) # use explicit bin locations
plt.hist( X4, bins=Bins ) # generate the plot 

"""
"""
print (np.mean(X, axis=0)) # compute mean of each feature

print (np.var(X, axis=0)) #compute variance of each feature

print (np.std(X, axis=0)) #compute standard deviation of each feature 

"""

print (X[0])
M = np.mean(X, axis=0)
S = np.std(X, axis=0)
print (M) # compute mean of each feature
print (S)
print ((X[0] - np.mean(X, axis=0))/S)


for i in range(0,148):  # normalizing data
    X[i] = (X[i] - M)/S

print (X)

# (f)For each pair of features (1,2), (1,3), and (1,4), plot a scatterplot
#of the feature values, colored according to their target value (class)
colors = ['b','g','r']
for c in np.unique(Y):
 plt.plot( X[Y==c,0], X[Y==c,3], 'o', color=colors[int(c)] )