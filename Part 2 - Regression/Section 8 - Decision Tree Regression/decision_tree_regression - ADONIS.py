# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 06:44:00 2019

@author: Adonis-Note
"""

# Decision Tree Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values #(1:2) upper bound não é considerado. Nos modelos de machine learning, especialmente o regression, a variável independente precisa ser tradada como uma matriz, por esse motivo foi incluído 1:2 ao invés de apenas 1. Assim ele não resulta em um vetor, mas em uma matriz.
y = dataset.iloc[:, 2].values #python conta apartir do zero


#Splitting the dataset into the Training set and Test set
""" não temos dados suficientes para fazer split dos dados. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


#Feature Scaling - necessário para normalizar os valores das variáveis, assim uma variável não dominará a outra, a diferença entre os valores não serão grandes (eucledian distance)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #no test set não precisamos fit somente transform"""


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0) #random state for getting the same results
regressor.fit(X, y)

# Predicting a new result with Decision Tree Regression
y_pred = regressor.predict([[6.5]])


# Visualising the Decision Tree Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualising the regression results (for higher resolution and smother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
