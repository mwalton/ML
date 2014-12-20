# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:41:42 2014

@author: michaelwalton
"""

#this is set up as a multiclass problem, could maybe modify to a multilabel
#problem and then treat the p(label) as a prediciton / reproduction of the
#odorant concentration, this would be really cool if that representation works

import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy as sp
import evolvedmachines as em

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

dataFolder = "data/Otrain_4Otest/" #folders: Otrain_4Otest, OBGtrain_4OBGtest, Otrain_4OBGtest
target_names = ['odorant 0', 'odorant 1', 'odorant 2', 'odorant 3']

addNoise = True
doRsa = False
standardize = True
parameterEstimation = 'none' #options: none, exhaustive, random, fixed_range
rand_iter = 10 #number of samples in the parameter space to sample in random estimation mode

###############################################################################
#load data
reader = csv.reader(open(dataFolder + "train_c.csv","rb"), delimiter=",")
x = list(reader)
train_c = np.array(x).astype('float')

reader = csv.reader(open(dataFolder + "train_a.csv","rb"), delimiter=",")
x = list(reader)
train_a = np.array(x).astype('float')

reader = csv.reader(open(dataFolder + "test_c.csv","rb"), delimiter=",")
x = list(reader)
test_c = np.array(x).astype('float')

reader = csv.reader(open(dataFolder + "test_a.csv","rb"), delimiter=",")
x = list(reader)
test_a = np.array(x).astype('float')

###############################################################################
# Convert the concentration labels to classes
train_target = np.argmax(train_c, axis=1)
test_target = np.argmax(test_c, axis=1)

###############################################################################
# Add noise to training and test sets
if (addNoise):
    random_state = np.random.RandomState(0)
    
    n_samples, n_features = train_a.shape
    train_a = np.c_[train_a, random_state.randn(n_samples, n_features)]
    
    n_samples, n_features = test_a.shape
    test_a = np.c_[test_a, random_state.randn(n_samples, n_features)]

###############################################################################
# Data Pre-processing
if (doRsa):
    rsa = em.RSA(latencyScale=100, sigmoidRate=False, normalizeSpikes=True,
                 maxLatency=1000, maxSpikes=20)
                 
    train_a = rsa.countNspikes(train_a)
    test_a = rsa.countNspikes(test_a)

if (standardize):
    scaler = StandardScaler()
    train_a = scaler.fit_transform(train_a)
    test_a = scaler.transform(test_a)
    
###############################################################################
# Build multiple classifiers based on some parameter space (random, grid, etc.)
# assess each using stratified k-folds of test data, select the best estimator

if (parameterEstimation == 'none'):
    clf = svm.SVC()
    clf.fit(train_a, train_target)
else:
    #configure stratified k-fold cross validation               
    cv = StratifiedKFold(y=train_target, n_folds=3, shuffle=True)
    
    #set the parameter grid
    if (parameterEstimation == 'exhaustive'):
        #do exhaustive gridsearch within some range
        kernel_range = ['rbf', 'linear']
        gamma_range = np.arange(start=1e-3, stop=1e-1, step=1e-3)
        C_range = np.arange(1,1000)
        param_grid = dict(kernel=kernel_range, gamma=gamma_range, C=C_range)
        grid = GridSearchCV(svm.SVC(C=1), param_grid=param_grid, cv=cv)
                        
    elif(parameterEstimation == 'random'):
        #randomly sample values within some distribution of params
        kernel_range = ['rbf', 'linear']
        class_weight_range = ['auto', None]
        gamma_range_range =  sp.stats.expon(scale=.1)
        C_range = sp.stats.expon(scale=100)
        param_dist = dict(kernel=kernel_range, gamma=gamma_range, C=C_range,
                          class_weight=class_weight_range)
        
        grid = RandomizedSearchCV(svm.SVC(), param_distributions=param_dist,
                                  cv=cv, n_iter=rand_iter)
        
    elif(parameterEstimation == 'fixed_range'):
        #exhaustive search on an explictly defined dictionary of params
        param_grid = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-5],
                        'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'],
                        'C': [1, 10, 100, 1000]}]
        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)

    #fit and validate until we find the best estimator within the range
    grid.fit(train_a, train_target)
    
    #show the resulting estimators
    for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

    print("Best Estimator: %s" % grid.best_estimator_)
    clf = grid.best_estimator_
    
###############################################################################
#RUN PREDICTION
    
pred = clf.predict(test_a)

###############################################################################
#PLOT DATA

#plot imported data and target
pl.figure(1)
plt.plot(train_c)
plt.title('Training (Odorant Concentration)')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.ylabel('Concentration')
plt.xlabel('Time')
plt.show()

pl.figure(2)
plt.plot(test_c)
plt.title('Testing (Odorant Concentration)')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.ylabel('Concentration')
plt.xlabel('Time')
plt.show()

pl.figure(3, figsize=(6,6))
plt.imshow(np.transpose(train_a))
#plt.colorbar()
plt.title('Training (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

pl.figure(6)
plt.imshow(np.transpose(test_a))
#plt.colorbar()
plt.title('Testing (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

#show confusion matrix
cm = confusion_matrix(test_target, pred)
pl.figure(7)
plt.matshow(cm)
plt.colorbar()
plt.title('SVC')
plt.ylabel('Target label')
plt.xlabel('Predicted label')
plt.show()

print("\n")
print(classification_report(test_target, pred, target_names=target_names))
print("Accuracy Score: %s\n" % accuracy_score(test_target, pred))
print("Classifier Settings: %s" % clf)
#print("AP", average_precision_score(test_target, pred))