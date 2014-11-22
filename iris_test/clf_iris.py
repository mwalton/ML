# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:17:06 2014

@author: michaelwalton
"""

from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

print(y[0])

# serialize with pickle
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0]))

#serialize with joblib
from sklearn.externals import joblib
joblib.dump(clf, 'clfIris.pkl')
clf3 = joblib.load('clfIris.pkl')
print(clf3.predict(X[0]))