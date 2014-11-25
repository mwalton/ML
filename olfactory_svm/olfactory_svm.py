# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:41:42 2014

@author: michaelwalton
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

#load data
reader = csv.reader(open("data/OBGtrain_4OBGtest/train_c.csv","rb"), delimiter=",")
x = list(reader)
train_c = np.array(x).astype('float')

reader = csv.reader(open("data/OBGtrain_4OBGtest/train_a.csv","rb"), delimiter=",")
x = list(reader)
train_a = np.array(x).astype('float')

reader = csv.reader(open("data/OBGtrain_4OBGtest/test_c.csv","rb"), delimiter=",")
x = list(reader)
test_c = np.array(x).astype('float')

reader = csv.reader(open("data/OBGtrain_4OBGtest/test_a.csv","rb"), delimiter=",")
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

# train svm classifier
clf = svm.SVC()
clf.fit(train_a, train_target)

# run the prediction
pred = clf.predict(test_a)

# test classification
correctPredictions = 0
for i in range(pred.shape[0]):
    if(pred[i] == test_target[i]):
        correctPredictions += 1.0

print(correctPredictions / test_target.shape[0])

####################################################
#PLOT DATA

#plot imported data and target
figure(1)
plt.plot(train_c)
plt.title('Training (Odorant Concentration)')
plt.ylabel('Concentration')
plt.xlabel('Time')
plt.show()

figure(2)
plt.imshow(np.transpose(train_a))
plt.title('Training (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

figure(3)
plt.plot(test_c)
plt.title('Testing (Odorant Concentration)')
plt.ylabel('Concentration')
plt.xlabel('Time')
plt.show()

figure(5)
plt.imshow(np.transpose(test_a))
plt.title('Testing (Sensor Pattern)')
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()

#show confusion matrix
cm = confusion_matrix(test_target, pred)
figure(6)
plt.matshow(cm)
plt.colorbar()
plt.ylabel('Target label')
plt.xlabel('Predicted label')
plt.show()