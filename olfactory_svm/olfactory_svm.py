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

reader = csv.reader(open("data/concentration.csv","rb"), delimiter=",")
x = list(reader)
concentration = np.array(x).astype('float')

reader = csv.reader(open("data/sensorActivation.csv","rb"), delimiter=",")
x = list(reader)
activation = np.array(x).astype('float')

#plot data
activation = np.transpose(activation)

figure(1)
plt.plot(concentration)

figure(2)
plt.imshow(activation)

#get max(concentration) foreach t (target is max odorant index)

# make a new support vector machine classifier
clf = svm.SVC()
#clf.fit(activation, concentration)