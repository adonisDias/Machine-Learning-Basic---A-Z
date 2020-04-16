# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 06:50:30 2019

@author: Adonis-Note
"""

# SVR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values #(1:2) upper bound não é considerado. Nos modelos de machine learning, especialmente o regression, a variável independente precisa ser tradada como uma matriz, por esse motivo foi incluído 1:2 ao invés de apenas 1. Assim ele não resulta em um vetor, mas em uma matriz.
y = dataset.iloc[:, 2].values #python conta apartir do zero

y = np.reshape(y,(-1,1))  #foi necessário para transformar o array para o processo de feature scaling

#Splitting the dataset into the Training set and Test set
""" não temos dados suficientes para fazer split dos dados. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


#Feature Scaling - necessário para normalizar os valores das variáveis, assim uma variável não dominará a outra, a diferença entre os valores não serão grandes (eucledian distance)
#As bibliotecas dos modelos de linear regression, (multiple, polynomial) já implementam feature scaling, porém a biblioteca SVM não, por esse motivo temos que implementá-la.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')    #rbf = gaussian (mais comum)
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Gerando o gráfico com os valores defaults, o salário do CEO é considerado um outlier por se distanciar muito do salário dos demais.

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualising the SVR results (for higher resolution and smother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
