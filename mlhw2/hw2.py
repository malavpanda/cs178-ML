# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:39:13 2017

@author: Malav
"""

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

#(a)
data = np.genfromtxt("data/curve80.txt",delimiter=None) # load the text file
X = data[:,0]
X = X[:, np.newaxis]    # code expects shape (M,N) so make sure it's 2-dimensional
Y = data[:,1]
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75)    # split data set 75/25

#(b)
lr = ml.linear.linearRegress(Xtr, Ytr); #create and train model
xs = np.linspace(0,10,200);      # densely sample possible x-values
xs = xs[:,np.newaxis];  # force "xs" to be an Mx1 matrix (expected by our code)
ys = lr.predict(xs);    # make predictions at xs

YtrHat = lr.predict(Xtr); # make predictions on the training data
#plt.rcParams['figure.figsize'] = (3.5, 4.0);
#plt.figure(1);
#plt.axis([0,10,-3,7]);

#plt.plot(xs, ys, 'b-', Xtr, Ytr, 'ro', Xte, Yte, 'go', linewidth = 2); # Plotting the training data along with its prediction

theta = lr.theta;   
theta = theta[0];
theta = theta[:,np.newaxis];
theta = theta.T;
print(theta);   #Print the linear regression coefficients (lr.theta)



#calculate and report the mean squared error in your predictions on both the training and test data.
Ytr = Ytr[:,np.newaxis];
Yte = Yte[:,np.newaxis];
m = len(Xtr)
YtrHat = lr.predict(Xte);
dotpr = Xte.dot(theta[0][1]) + theta[0][0] ;
eTr = Yte - dotpr;
JTr = eTr.T.dot( eTr ) / len(Xte);  # mean squared error on the training data
print(JTr);
"""
YteHat = lr.predict(Xte);
eTe = Yte - YteHat;
JTe = eTe.T.dot( eTe ) / m; 

#(c)
Xtr2 = np.zeros( (Xtr.shape[0],2) ) # create Mx2 array to store features

Xtr2[:,0] = Xtr[:,0] # place original "x" feature as X1

Xtr2[:,1] = Xtr[:,0]**2 # place "x^2" feature as X2

# Now, Xtr2 has two features about each data point: "x" and "x^2"
# Create polynomial features up to "degree"; don't create constant feature

# (the linear regression learner will add the constant feature automatically)
degree = 3;
XtrP = ml.transforms.fpoly(Xtr, degree, bias=False);

# Rescale the data matrix so that the features have similar ranges / variance

XtrP,params = ml.transforms.rescale(XtrP);
Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X, degree,False), params)[0]
# "params" returns the transformation parameters (shift & scale)

# Then we can train the model on the scaled feature matrix:

lr = ml.linear.linearRegress( Phi(Xtr), Ytr ); # create and train model

# Now, apply the same polynomial expansion & scaling transformation to Xtest:

#XteP,_ = ml.transforms.rescale( ml.transforms.fpoly(Xte,degree,False), params);

YhatTrain = lr.predict(Phi(xs));
#plt.rcParams['figure.figsize'] = (4.0, 3.0)
plt.figure(2)

plt.plot( xs, YhatTrain, '-');
ax = plt.axis();


plt.figure(2)
plt.rcParams['figure.figsize'] = (24.0, 4.0)
fig,sub = plt.subplots(1,6)


degree = [1, 3, 5, 7, 10, 18];
k = 0;
JaTr = [0]*degree[len(degree)- 1];
JaTe = [0]*degree[len(degree)- 1];

for k,i in enumerate(degree):
    XtrP = ml.transforms.fpoly(Xtr, i, bias=False);
    XtrP,params = ml.transforms.rescale(XtrP);
    Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X, i,False), params)[0];
    lr = ml.linear.linearRegress( Phi(Xtr), Ytr ); # create and train model
    
    YhatTrain = lr.predict(Phi(xs)); # predict on training data
    sub[k].set_title("Degree = " + str(i));
    sub[k].plot(xs,YhatTrain, 'b-', Xtr, Ytr, 'ro', Xte, Yte, 'go', linewidth = 2);
    sub[k].axis([0,10,-3,7]);

i = 1;    
for i in range(18):
    XtrP = ml.transforms.fpoly(Xtr, i, bias=False);
    XtrP,params = ml.transforms.rescale(XtrP);
    Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X, i,False), params)[0];
    lr = ml.linear.linearRegress( Phi(Xtr), Ytr ); # create and train model
    
    YtrHat = lr.predict(Phi(Xtr));
    eTr = Ytr - YtrHat;
    JTr = eTr.T.dot( eTr ) / m;
    JaTr[i] = JTr[0][0];
    
    YteHat = lr.predict(Phi(Xte));
    eTe = Yte - YteHat;
    JTe = eTe.T.dot( eTe ) / m; 
    JaTe[i] = JTe[0][0];

plt.rcParams['figure.figsize'] = (4.0, 3.0)
plt.figure();
plt.semilogy(JaTr, color = 'r' );
plt.semilogy(JaTe, color = 'g' );

#Problem 2

nFolds = 5;
J = [0] * 5;

Jmean = [0] * 18;

plt.figure();
i = 1;
for i in range(18):
    
    for iFold in range(nFolds):
        Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold);
        XtrP = ml.transforms.fpoly(Xti, i, bias=False);
        XtrP,params = ml.transforms.rescale(XtrP);
        Phi = lambda X: ml.transforms.rescale( ml.transforms.fpoly(X, i,False), params)[0];
        lr = ml.linear.linearRegress( XtrP, Yti ); # create and train model
        
        YviHat = lr.predict(Phi(Xvi));
        eVi = Yvi - YviHat;
        JVi = eVi.T.dot( eVi ) / len(Xvi);
        J[iFold] = JVi[0][0];

    Jmean[i] = np.mean(J)


plt.semilogy(Jmean);
plt.semilogy(JaTr, color = 'r' );
plt.semilogy(JaTe, color = 'g' );

for iFold in range(nFolds):
    Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold);
    learner = ml.linear.linearRegress(Xti,Yti);
    J[iFold] = learner.mse(Xvi,Yvi);

print(np.mean(J));
"""