# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:40:44 2017

@author: Malav
"""

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from scipy import linalg

#(a)
X = np.genfromtxt("data/iris.txt",delimiter=None) # load the data
X = X[:,0:2];

plt.plot(X, 'o');

#(b)
K = 5;
bestSum = np.inf;
for i in range(0,10):
    zz,cc,ssumd = ml.cluster.kmeans(X, K);
    if(ssumd < bestSum):
        z = zz;
        c = cc;
        sumd = ssumd;

ml.plotClassify2D(None,X,z);
plt.plot(c[:,0], c[:,1], 's', mfc='none', markeredgecolor='k', mew=2, ms=8)
plt.legend(loc="lower right")
plt.title("k = 5");

K = 20;
bestSum = np.inf;
for i in range(0,10):
    zz,cc,ssumd = ml.cluster.kmeans(X, K);
    if(ssumd < bestSum):
        z = zz;
        c = cc;
        sumd = ssumd;

plt.figure()
ml.plotClassify2D(None,X,z);
plt.plot(c[:,0], c[:,1], 's', mfc='none', markeredgecolor='k', mew=2, ms=8)
plt.legend(loc="lower right")
plt.title("k = 20");

#(c)
single linkage
K = 5;
z,join = ml.cluster.agglomerative(X,K, method = 'min');
plt.figure();
ml.plotClassify2D(None,X,z);
plt.title("single linkage, K = 5")

K = 20;
z,join = ml.cluster.agglomerative(X,K, method = 'min');
plt.figure();
ml.plotClassify2D(None,X,z);
plt.title("single linkage, K = 20")

#complete linkage
K = 5;
z,join = ml.cluster.agglomerative(X,K, method = 'max');
plt.figure();
ml.plotClassify2D(None,X,z);
plt.title("complete linkage, K = 5")

K = 20;
z,join = ml.cluster.agglomerative(X,K, method = 'max');
plt.figure();
ml.plotClassify2D(None,X,z);
plt.title("complete linkage, K = 20")

#problem 2
X = np.genfromtxt("data/faces.txt", delimiter=None) # load face dataset
plt.figure()
# pick a data point i for display
img = np.reshape(X[0,:],(24,24)) # convert vectorized data point to 24x24 image patch
plt.imshow( img.T , cmap="gray") # display image patch; you may have to squint

#(a)
mu = np.mean( X, axis=0, keepdims=True ) # find mean over data points
X0 = X - mu # zero-center the data

#(b)
U,S,Vh = linalg.svd(X0,False); #X0 = U * diag(S) * Vh
W = U.dot(np.diag(S));

#(c)
means = [None] * 10;
k_arr = np.arange(1,11);
for i,K in enumerate(k_arr):
    X_0 = W[:,:K].dot(Vh[:K,:]);
    means[i] = np.mean((X0 - X_0)**2);

plt.figure();
plt.plot(k_arr,means);

#(d)

for i in range(3):
    alpha = 2*np.median(np.abs(W[:,i]));
    pDir = mu + alpha*(Vh[i,:]);
    pDir = np.reshape(pDir[0,:],(24,24));
    plt.figure();
    plt.imshow(pDir.T, cmap = "gray");
    plt.title("mean + alpha*Vh[" + str(i) + ",:]");
          
    pDir = mu - alpha*(Vh[i,:]);
    pDir = np.reshape(pDir[0,:],(24,24));
    plt.figure();
    plt.imshow(pDir.T, cmap = "gray");
    plt.title("mean - alpha*Vh[" + str(i) + ",:]");
          
#(e)
#face 1
aimg = np.reshape(X[1,:],(24,24));
plt.figure();
plt.imshow(aimg.T, cmap = "gray");
plt.title("Face 1");
k = [5,10,50,100];
for i,K in enumerate(k):
    X_0 = mu + W[1,:K].dot(Vh[:K,:]); 
    img = np.reshape(X_0[:],(24,24));
    plt.figure();
    plt.imshow(img.T, cmap = "gray");
    plt.title("K = " + str(K));

aimg = np.reshape(X[2,:],(24,24));
plt.figure();
plt.imshow(aimg.T, cmap = "gray");
plt.title("Face 2");
k = [5,10,50,100];
for i,K in enumerate(k):
    X_0 = mu + W[2,:K].dot(Vh[:K,:]); 
    img = np.reshape(X_0[:],(24,24));
    plt.figure();
    plt.imshow(img.T, cmap = "gray");
    plt.title("K = " + str(K));

#(f)
idx = np.random.randint(1, 4915, size = 20);# pick some data at random or otherwise; a list / vector of integer indices

coord,params = ml.transforms.rescale( W[:,0:2] ) # normalize scale of "W" locations
plt.figure(); plt.hold(True); # you may need this for pyplot
for i in idx:
# compute where to place image (scaled W values) & size
    loc = (coord[i,0],coord[i,0]+0.5, coord[i,1],coord[i,1]+0.5)
    img = np.reshape( X[i,:], (24,24) ) # reshape to square
    plt.imshow( img.T , cmap="gray", extent=loc ) # draw each image
    plt.axis( (-2,2,-2,2) ) # set axis to reasonable visual scale