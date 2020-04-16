# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 06:22:10 2019

@author: Adonis-Note
"""

# XGBOOST ALGORITHM

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import xgboost as xgb

# importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values #(1:2) upper bound não é considerado. Nos modelos de machine learning, especialmente o regression, a variável independente precisa ser tradada como uma matriz, por esse motivo foi incluído 1:2 ao invés de apenas 1. Assim ele não resulta em um vetor, mas em uma matriz.
y = dataset.iloc[:, 13].values #python conta apartir do zero

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#aqui nós criamos uma dummy variable para evitar o dummy variable trap. Escolhemos a variável país porque ela possui mais categorias do que o gênero Female ou Male
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #retiro uma das dummy variables

# Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Xgboost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,
                      y_pred)

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10)
accuracies.mean()
accuracies.std() #variância (diferença) entre a média das accuracies e cada accaracy