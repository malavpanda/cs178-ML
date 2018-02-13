# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:28:21 2017

@author: Malav
"""

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import VotingClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.pipeline import Pipeline
#from sklearn.pipeline import make_pipeline
#from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
#from sklearn import tree


#(a)
X = np.genfromtxt("X_train.txt",delimiter=None) # load the data
Xt = X[1:100000,];
Xv = X[90001:100000,:];
Y = np.genfromtxt("Y_train.txt",delimiter=None) # load the data
Yt = Y[1:100000,]; 
Yv = Y[90001:100000,];

Xte = np.genfromtxt("X_test.txt",delimiter=None) # load the data


#2(a)
#m,n = Xt.shape
#nUse = 250;
#nBag = 25; #the number of classifiers
#classifiers = [ None ] * nBag # Allocate space for learners
#
#for i in range(0,nBag):
#    ind = np.floor( m * np.random.rand(nUse) ).astype(int) # Bootstrap sample a data set:
#    Xi, Yi = Xt[ind,:] , Yt[ind] # select the data at those indices
#    classifiers[i] = ml.dtree.treeClassify(Xi,Yi, nFeatures = 5) # Train a model on data Xi, Yi
#    
#
#mTest = Xv.shape[0];
#nBag = [1,5,10,25];
#nBag_err = [None] * 4;
#mini = 1;
#for ii in range(0,4):
#    predict = np.zeros( (mTest, nBag[ii]) ) # Allocate space for predictions from each model
#    for i in range(0,nBag[ii]):
#        predict[:,i] = classifiers[i].predict(Xv); # Apply each classifier
#    predict = np.mean(predict, axis=1) > 0.5;
#    err = 0
#    for j in range(0,mTest):
#        err += 1 if (predict[j] != Yv[j]) else 0 
#    nBag_err[ii] = err/(mTest);
#    if(nBag_err[ii]<mini):
#        mini = nBag_err[ii]; 
#        minindex = ii;
#    
#print(mini);
#print(nBag[ii]);

#neural net
#classifier = multilayer_perceptron.MLPClassifier(solver='lbfgs', alpha=1e-5,
#                                    hidden_layer_sizes=(10, 2), random_state=1)

#poly = PolynomialFeatures(3)
#xtt = poly.fit_transform(Xt)
#
#
#clfs = [None] * 2;
#clf1 = RandomForestClassifier(warm_start = True, random_state =1 ,n_estimators = 290, max_features = 6);
#clf = BaggingClassifier(base_estimator = clf1, max_samples = 100000);
#clf = BaggingClassifier(warm_start = True, random_state =1, n_estimators = 280, max_samples = 100000, max_features = 6);
#clf = KNeighborsClassifier(n_neighbors = 20)
#clf = GaussianNB();
#eclf1 = VotingClassifier(estimators=[
#         ('lr', clf1), ('rf', clf)], voting='hard')
#clf = GradientBoostingClassifier(n_estimators = 50000, max_depth = 3 ); #5000 = .790979 #6000,Ye1pred = .79867,#7000,Ye2pred = .80578,#8000,Ye3pred = .812
#9000,Ye4pred = 0.817, #10000,Ye5pred = 0.8228, #12000,Ye6pred = 0.8323,#13000,Ye7pred = 0.8388, AuC = 0.775, #20000,Ye9pred = 0.8601 #25000,Ye10pred = 0.87, auc = .777
#50,000,Ye11pred = 0.911
#clf =  MLPClassifier(hidden_layer_sizes=(1000, 20, 20), random_state=1)
#clf = RandomForestClassifier(warm_start = True, random_state =1 ,n_estimators = 290, max_features = 6);
#clfs[1] = linear_model.SGDClassifier(loss='perceptron')
#clf = KNeighborsClassifier(n_neighbors = 18, leaf_size = 62, p = 1);
#clf = GradientBoostingClassifier(n_estimators = 4000, max_depth = 3); #Yepred
#clf1 = DecisionTreeClassifier(max_depth=15)
#eclf = VotingClassifier(estimators=[('dt', clf),('bc', clf1)], voting='soft', weights=[1,1])
#scores = cross_val_score(eclf, Xt, Yt, cv=2, scoring='accuracy')
#print(scores);
#clf = clf.fit(xtt,Yt);
#xtt = preprocessing.scale(Xt);
#clf = clf.fit(xtt,Yt);
#scaler = StandardScaler();
#scaler.fit(Xt)  
#Xt = scaler.transform(Xt) 
#loo = LeaveOneOut()
#loo.get_n_splits(X)
#for train_index, test_index in loo.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index);
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = Y[train_index], Y[test_index]
#    clf = clf.fit(X_train, y_train);
#    score = clf.score(Xv,Yv);
#    print("AUC: " + str(score));
#clf = clf.fit(X,Y);
#clf = clf.fit(Xt, Yt);
#eclf = eclf.fit(Xt, Yt);
#clfs[0] = clfs[0].fit(Xt, Yt);
#clfs[1] = clfs[1].fit(xtt,Yt);
#xvv = poly.fit_transform(Xv)
#score = clf.score(xvv,Yv);
#xvv = preprocessing.scale(Xv);
#score = clf.score(xvv,Yv);
#score = eclf1.score(Xv,Yv);
#Xv = scaler.transform(Xv) 
#score = clf.score(Xv,Yv);
#score = eclf.score(Xv,Yv);
#print(clf.score(Xt,Yt));
#print(score);

#Xte = scaler.transform(Xte);
#Ykpred = clf.predict_proba(Xte);
#Ye11pred = clf.predict_proba(Xte);
#Ypred = clf.predict_proba( Xte );   # make "soft" predictions from your learner  (Mx2 numpy array)
preds = np.zeros( (Xte.shape[0], 3) )
preds[:,0] = Ye1pred[:,1]; # Apply each classifier

pred = Ypred;
pred = pred[:,1];
preds[:,1] = pred;
preds[:,2] = pred;
#
means = np.mean(preds,axis=1)
#
#
# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('Yhat11.txt',
np.vstack( (np.arange(len(means)) , means) ).T,
'%d, %.2f',header='ID,Prob1',comments='',delimiter=',');

#mTest = Xv.shape[0];
#predict = np.zeros( (mTest, 2) ) # Allocate space for predictions from each model
#errs = [None] * 2;
#for i in range(0,2):
#    if(i==0):
#        predict[:,i] = clfs[i].predict(Xv); # Apply each classifier
#    else:
#        predict[:,i] = clfs[i].predict(xvv);
#predict = np.mean(predict, axis=1) >= 0.5;
#err = 0
#for j in range(0,mTest):
#    err += 1 if (predict[j] != Yv[j]) else 0 
#err = err/(mTest);
#print(1 - err);


#rfc = MLPClassifier(alpha = .0001);
#st =[(15,2),(20,2),(25,2),(30,2),(40,2),(45,2)];
#param_grid = { 
#      'hidden_layer_sizes': st
##     'n_neighbors': [5,30]
##    'n_estimators': [290, 310],
##    'max_features': [5,6]
#  #  'max_samples': [10000,50000]
#}
#
#CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#scaler = StandardScaler();
#scaler.fit(Xt)  
#Xt = scaler.transform(Xt) 
#CV_rfc.fit(Xt, Yt)
##CV_rfc.fit(Xt, Yt)
#print (CV_rfc.best_params_)
#print (CV_rfc.best_score_)

#mTest = Xv.shape[0];
#err = 0;
#for j in range(0,mTest):
#        err += 1 if (predict1[j] != Yv[j]) else 0 
#nn_err = err/(mTest);
#print(nn_err);


#preds = np.zeros( (Xte.shape[0], nBag) )
#for i in range(0,nBag):
#    pred = classifiers[i].predict_proba(Xte); # Apply each classifier
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