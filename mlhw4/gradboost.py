## -*- coding: utf-8 -*-
#"""
#Created on Tue Mar 14 19:25:08 2017
#
#@author: Malav
#"""
import numpy as np
np.random.seed(0)
import mltools as ml
#import matplotlib.pyplot as plt   # use matplotlib for plotting with inline plots
from sklearn.ensemble import BaggingClassifier

#%matplotlib inline

X = np.genfromtxt("X_train.txt",delimiter=' ')
Y = np.genfromtxt("Y_train.txt",delimiter=' ')
Xt,Xv,Yt,Yv = ml.splitData(X,Y,0.90)

Xe = np.genfromtxt('X_test.txt',delimiter=' ')



def auc(soft,Y):
    """Manual AUC function for applying to soft prediction vectors"""
    indices = np.argsort(soft)         # sort data by score value
    Y = Y[indices]
    sorted_soft = soft[indices]
    
    # compute rank (averaged for ties) of sorted data
    dif = np.hstack( ([True],np.diff(sorted_soft)!=0,[True]) )
    r1  = np.argwhere(dif).flatten()
    r2  = r1[0:-1] + 0.5*(r1[1:]-r1[0:-1]) + 0.5
    rnk = r2[np.cumsum(dif[:-1])-1]
    
    # number of true negatives and positives
    n0,n1 = sum(Y == 0), sum(Y == 1)
    
    # compute AUC using Mann-Whitney U statistic
    result = (np.sum(rnk[Y == 1]) - n1 * (n1 + 1.0) / 2.0) / n1 / n0
    return result

np.random.seed(0)
nUse= 2000
mu = Yt.mean()
dY = Yt - mu
step = 0.5
Pt2 = np.zeros((Xt.shape[0],))+mu
Pv2 = np.zeros((Xv.shape[0],))+mu
Pe2 = np.zeros((Xe.shape[0],))+mu

np.random.seed(0)
for l in range(nUse):             # this is a lot faster than the bagging loop:
    # Better: set dY = gradient of loss at soft predictions Pt
    # Note: treeRegress expects 2D target matrix
    tree = ml.dtree.treeRegress(Xt,dY[:,np.newaxis], maxDepth=3)  # train and save learner
    Pt2 += step*tree.predict(Xt)[:,0]        # predict on training data
    Pv2 += step*tree.predict(Xv)[:,0]        #    and validation data
    Pe2 += step*tree.predict(Xe)[:,0]        #    and test data
    dY  -= step*tree.predict(Xt)[:,0]        # update residual for next learner
    print (" {} trees: MSE ~ {};  AUC - {};".format(l+1, ((Yv-Pv2)**2).mean(), auc(Pv2,Yv) ))

#toKaggle('Pe2.csv',Pe2)
print("2: GradBoost, {} trees: MSE ~ {}; AUC - {};".format(nUse, ((Yv-Pv2)**2).mean(), auc(Pv2,Yv)))

X = np.genfromtxt("X_train.txt",delimiter=None) # load the data
Xt = X[0:100000,];
Y = np.genfromtxt("Y_train.txt",delimiter=None) # load the data
Yt = Y[0:100000,]; 
Xte = np.genfromtxt("X_test.txt",delimiter=None) # load the data

clf = BaggingClassifier(warm_start = True, n_estimators = 290, max_samples = 100000, max_features = 6);
clf = clf.fit(Xt,Yt);
Ypred = clf.predict_proba(Xte);

Xte = np.genfromtxt("X_test.txt",delimiter=None) # load the data
preds = np.zeros( (Xte.shape[0], 2) )
preds[:,0] = Pe2; # Apply each classifier
pred = Ypred;
pred = pred[:,1];
preds[:,1] = pred;

means = np.mean(preds,axis=1)


# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('Yhat_dtree.txt',
np.vstack( (np.arange(len(means)) , means) ).T,
'%d, %.2f',header='ID,Prob1',comments='',delimiter=',');