# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:28:21 2017

@author: Malav
"""

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt


#(a)
X = np.genfromtxt("X_train.txt",delimiter=None) # load the data
Xt = X[1:90000,:];
Xv = X[90001:100000,:];
Y = np.genfromtxt("Y_train.txt",delimiter=None) # load the data
Yt = Y[1:90000,np.newaxis];
Yv = Y[90001:100000,];

Xte = np.genfromtxt("X_test.txt",delimiter=None) # load the data


#(b)
#learner = ml.dtree.treeClassify(Xt,Yt);
##error on training
#YtHat = learner.predict(Xt);
#length = len(YtHat);
#err = 0
#for j in range(0,length):
#    err += 1 if (YtHat[j] != Yt[j]) else 0 
#tr_err = err/(length);

#error on validation                 YtHat = learner.predict(Xt);
#YvHat = learner.predict(Xv);
#length = len(YvHat);
#err = 0
#for j in range(0,length):
#    err += 1 if (YvHat[j] != Yv[j]) else 0 
#val_err = err/(length);
#
##(c)
#on training data
#errTrain = np.empty([16,1]);
#maxDepth = np.linspace(0, 15, num=16);
#maxDepth = maxDepth[:, np.newaxis];
#for i,k in enumerate(maxDepth):
#    learner = ml.dtree.treeClassify(Xt,Yt, maxDepth=k);
#    Yhat = learner.predict(Xt) 
#    err = 0
#    for j in range(0,len(Yhat)):
#        err += 1 if (Yhat[j] != Yt[j]) else 0 
#    fracterr = err/(len(Yhat))                 
#    errTrain[i] = fracterr
#
#plt.semilogx(maxDepth, errTrain, color = 'r')
#
#for i,k in enumerate(maxDepth):
#    learner = ml.dtree.treeClassify(Xt,Yt, maxDepth=k);
#    Yhat = learner.predict(Xv) 
#    err = 0
#    for j in range(0,len(Yhat)):
#        err += 1 if (Yhat[j] != Yv[j]) else 0 
#    fracterr = err/(len(Yhat))                 
#    errTrain[i] = fracterr
#
#plt.semilogx(maxDepth, errTrain, color = 'g')
#around maxdepth 5 is optimal

#(d)
#errTrain = np.empty([6,1]);
#minLeaf = np.linspace(2,12, num = 6, dtype = "int16");
#minLeaf = minLeaf[:, np.newaxis];
#
#for i,k in enumerate(minLeaf):
#    learner = ml.dtree.treeClassify(Xt,Yt, maxDepth=50, minLeaf = 2**k);
#    Yhat = learner.predict(Xt) 
#    err = 0
#    for j in range(0,len(Yhat)):
#        err += 1 if (Yhat[j] != Yt[j]) else 0 
#    fracterr = err/(len(Yhat))                 
#    errTrain[i] = fracterr
#
#plt.semilogx(minLeaf, errTrain, color = 'r')
#
#for i,k in enumerate(minLeaf):
#    learner = ml.dtree.treeClassify(Xt,Yt, maxDepth=50, minLeaf = 2**k);
#    Yhat = learner.predict(Xv) 
#    err = 0
#    for j in range(0,len(Yhat)):
#        err += 1 if (Yhat[j] != Yv[j]) else 0 
#    fracterr = err/(len(Yhat))                 
#    errTrain[i] = fracterr
#
#plt.semilogx(minLeaf, errTrain, color = 'g');
#plt.axis([2, 12, 0.05, 0.4])
#minleaf = 2^8 is optimal

#(f)
#learner = ml.dtree.treeClassify(Xt,Yt, maxDepth = 15 ); 
#fpr,tpr,_ = learner.roc(Xv,Yv);
#roc_auc = learner.auc(Xv,Yv);
#
#plt.figure()
#lw = 2
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC')
#plt.legend(loc="lower right")
#plt.show()


#(g)
#learner = ml.dtree.treeClassify(Xt,Yt,  maxDepth = 7); 
#Ypred = learner.predictSoft( Xte )
## Now output a file with two columns, a row ID and a confidence in class 1:
#np.savetxt('Yhat_dtree.txt',
#np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T,
#'%d, %.2f',header='ID,Prob1',comments='',delimiter=',');

#prb = np.genfromtxt("Yhat_dtree.txt",delimiter=",") # load the data
#av = prb[1:100001,1];
#
#average_prob = np.mean(av);

#2(a)
m,n = Xt.shape
nUse = 10000;
nBag = 200; #the number of classifiers
classifiers = [ None ] * nBag # Allocate space for learners

for i in range(0,nBag):
    ind = np.floor( m * np.random.rand(nUse) ).astype(int) # Bootstrap sample a data set:
    Xi, Yi = Xt[ind,:] , Yt[ind] # select the data at those indices
    classifiers[i] = ml.dtree.treeClassify(Xi, Yi, nFeatures = 5) # Train a model on data Xi, Yi

#boost
#mTest = Xv.shape[0];
#errors = [None] * nBag;
#predict = np.zeros( (mTest, nBag) ) # Allocate space for predictions from each model
#for i in range(0,nBag):
#    predict = classifiers[i].predict(Xv); # Apply each classifier
#    err = 0
#    for j in range(0,mTest):
#        err += 1 if (predict[j] != Yv[j]) else 0 
#    errors[i] = err/(mTest);
#
#errors = np.asarray(errors);
#mean = np.mean(errors);
#count = 0;
#for i in range(0,nBag):
#    if(errors[i]<mean):
#       count = count + 1;
#       
#error_i = [None] * count;
#ii = 0;
#for i in range(0,nBag):
#    if(errors[i]<mean):
#       error_i[ii] = i;
#       ii += 1; 
#       
#predict = np.zeros( (mTest, count) ) # Allocate space for predictions from each model
#for i in range(0,count):
#    predict[:,i] = classifiers[error_i[i]].predict(Xv); # Apply each classifier
#predict = np.mean(predict, axis=1) > 0.5;
#err = 0
#for j in range(0,mTest):
#    err += 1 if (predict[j] != Yv[j]) else 0 
#count_err = err/(mTest);
#    
#print(count_err);
      
#test on data Xtest
mTest = Xv.shape[0];
nBag = [1,5,10,200];
nBag_err = [None] * 4;
mini = 1;
for ii in range(0,4):
    predict = np.zeros( (mTest, nBag[ii]) ) # Allocate space for predictions from each model
    for i in range(0,nBag[ii]):
        predict[:,i] = classifiers[i].predict(Xv); # Apply each classifier
    predict = np.mean(predict, axis=1) > 0.5;
    err = 0
    for j in range(0,mTest):
        err += 1 if (predict[j] != Yv[j]) else 0 
    nBag_err[ii] = err/(mTest);
    if(nBag_err[ii]<mini):
        mini = nBag_err[ii]; 
        minindex = ii;
    
print(mini);
print(nBag[ii]);
#
#
#mTest = Xt.shape[0];
#nBag = [1,5,10,25];
#nBag_err = [None] * 4;
#for ii in range(0,4):
#    predict = np.zeros( (mTest, nBag[ii]) ) # Allocate space for predictions from each model
#    for i in range(0,nBag[ii]):
#        predict[:,i] = classifiers[i].predict(Xt); # Apply each classifier
#    predict = np.mean(predict, axis=1);
#    for k in range(0,mTest):
#        if(predict[k] >= 0.5):
#            predict[k] = 1;
#        else:
#            predict[k] = 0;
#    err = 0
#    for j in range(0,mTest):
#        err += 1 if (predict[j] != Yt[j]) else 0 
#    nBag_err[ii] = err/(mTest);
#    print(nBag_err[ii]);

#preds = np.zeros( (Xte.shape[0], nBag) )
#for i in range(0,nBag):
#    pred = classifiers[i].predictSoft(Xte); # Apply each classifier
#    pred = pred[:,1];
#    preds[:,i] = pred ;
#
#means = np.mean(preds,axis=1)
#
#
## Now output a file with two columns, a row ID and a confidence in class 1:
#np.savetxt('Yhat_dtree.txt',
#np.vstack( (np.arange(len(means)) , means) ).T,
#'%d, %.2f',header='ID,Prob1',comments='',delimiter=',');

#auc for valudation
#auc = 0;
#for i in range(0,nBag):
#    auc += classifiers[i].auc(Xv,Yv);
#
#avg_auc = auc / nBag;