# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:41:42 2014

@author: michaelwalton
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn import svm

#load data
reader = csv.reader(open("data/4Atrain_4ABGtest/train_c.csv","rb"), delimiter=",")
x = list(reader)
train_c = np.array(x).astype('float')

reader = csv.reader(open("data/4Atrain_4ABGtest/train_a.csv","rb"), delimiter=",")
x = list(reader)
train_a = np.array(x).astype('float')

reader = csv.reader(open("data/4Atrain_4ABGtest/test_c.csv","rb"), delimiter=",")
x = list(reader)
test_c = np.array(x).astype('float')

reader = csv.reader(open("data/4Atrain_4ABGtest/test_a.csv","rb"), delimiter=",")
x = list(reader)
test_a = np.array(x).astype('float')

#get max(concentration) foreach t (target is max odorant index)
train_target = np.zeros([train_c.shape[0]], dtype=float)

for i in range(train_c.shape[0]):
    maxC = 0
    for j in range(train_c.shape[1]):
        if (train_c[i][j] > maxC):
            maxC = train_c[i][j]
            train_target[i] = j
            
test_target = np.zeros([test_c.shape[0]], dtype=float)

for i in range(test_c.shape[0]):
    maxC = 0
    for j in range(test_c.shape[1]):
        if (test_c[i][j] > maxC):
            maxC = test_c[i][j]
            test_target[i] = j

#plot imported data and target
"""
figure(1)
plt.plot(train_c)

figure(2)
plt.imshow(np.transpose(train_a))

figure(3)
plt.plot(train_target)

figure(4)
plt.plot(test_c)

figure(5)
plt.imshow(np.transpose(test_a))

figure(6)
plt.plot(test_target)
"""
# train a set of classifiers with a range of parameters gamma, C
# to try and optimize classification
C_range2d = [1, 1e2, 1e4]
gamma_range2d = [1e-1, 1, 1e1]
classifiers = []

for C in C_range2d:
    for gamma in gamma_range2d:
        clf = svm.SVC(C=C, gamma=gamma)
        clf.fit(train_a, train_target)
        classifiers.append((C, gamma, clf))

# test classification
for (k, (C, gamma, clf)) in enumerate(classifiers):
    correctPredictions = 0
    for i in range(test_c.shape[0]):
        if(clf.predict(test_a[i]) == test_target[i]):
            correctPredictions += 1.0
    
    print(correctPredictions / test_target.shape[0])
"""
##############################################################################
# visualization
#
# draw visualization of parameter effects
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma 10^%d, C 10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function 
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.jet)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y_2d, cmap=plt.cm.jet)
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

# plot the scores of the grid
# grid_scores_ contains parameter settings and scores
score_dict = grid.grid_scores_

# We extract just the scores
scores = [s[1] for s in score_dict]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))

# draw heatmap of accuracy as a function of gamma and C
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)

plt.show()
"""


