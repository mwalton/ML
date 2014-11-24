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
reader = csv.reader(open("data/Otrain_4OBGtest/train_c.csv","rb"), delimiter=",")
x = list(reader)
train_c = np.array(x).astype('float')

reader = csv.reader(open("data/Otrain_4OBGtest/train_a.csv","rb"), delimiter=",")
x = list(reader)
train_a = np.array(x).astype('float')

reader = csv.reader(open("data/Otrain_4OBGtest/test_c.csv","rb"), delimiter=",")
x = list(reader)
test_c = np.array(x).astype('float')

reader = csv.reader(open("data/Otrain_4OBGtest/test_a.csv","rb"), delimiter=",")
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

# train svm classifier
clf = svm.SVC()
clf.fit(train_a, train_target)

# test classification

correctPredictions = 0
for i in range(test_c.shape[0]):
    if(clf.predict(test_a[i]) == test_target[i]):
        correctPredictions += 1.0

print(correctPredictions / test_target.shape[0])
