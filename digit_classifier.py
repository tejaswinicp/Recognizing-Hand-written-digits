#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:32:14 2019

@author: tejaswinicp
"""

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np

digits = load_digits()
print(digits.data.shape)
pl.matshow(digits.images[1])
X = digits.data.reshape(-1, 64)
print(X.shape)
y = digits.target


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)
print("X_train shape: %s" % repr(X_train.shape))
print("y_train shape: %s" % repr(y_train.shape))
print("X_test shape: %s" % repr(X_test.shape))
print("y_test shape: %s" % repr(y_test.shape))

svm = LinearSVC()
svm.fit(X_train, y_train)
svm.predict(X_train)
print(svm.score(X_train, y_train))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))

scores = cross_val_score(rf, X_train, y_train, cv = 5)
print("scores: %s  mean: %f  std: %f" % (str(scores), np.mean(scores), np.std(scores)))




