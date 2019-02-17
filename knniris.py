y#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:17:39 2019

@author: sai
"""

from sklearn.datasets import load_iris
import numpy as np
dataset = load_iris()
X = dataset.data[:,[2,3]]
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors = 2, p =2, metric = 'minkowski')
Knn = Knn.fit(X_train,y_train)
y_pred = Knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)

miss = (y_pred != y_test).sum()

import matplotlib.pyplot as plt
plt.scatter(y_pred, y_test)
plt.show()

