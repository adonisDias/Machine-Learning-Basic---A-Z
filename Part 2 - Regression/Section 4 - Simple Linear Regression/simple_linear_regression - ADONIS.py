# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 06:55:13 2019

@author: Adonis-Note
"""

#Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values #python conta apartir do zero


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#Feature Scaling - necessário para normalizar os valores das variáveis, assim uma variável não dominará a outra, a diferença entre os valores não serão grandes (eucledian distance)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #no test set não precisamos fit somente transform"""

#Fitting SImple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) #fazendo essa chamada, o modelo é treinado e já está pronto para ser usado nas predições

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red') #exibe os dados em pontos dispersos
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #mostra a linha da regressão para os dados treinados, para depois compararmos com a linha dos dados previstos
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red') #exibe os dados em pontos dispersos
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #mostra a linha da regressão para os dados treinados, para depois compararmos com a linha dos dados previstos
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()